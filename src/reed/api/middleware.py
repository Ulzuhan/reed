"""ASGI guards that must run before FastAPI reads or parses a request body."""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from reed.api.deps import api_key_bytes_match

if TYPE_CHECKING:
    from reed.services import Services

MEBIBYTE = 1024 * 1024

# How long an upload waits for a spool slot before it is refused. Long enough to
# absorb a browser dropping several files at once, short enough that a client
# learns the server is saturated instead of holding a connection open on hope.
UPLOAD_SLOT_WAIT_SECONDS = 5.0


class RequestBodyTooLarge(Exception):
    """Raised internally once a chunked request crosses its route limit."""


@dataclass(slots=True)
class _ReceiveState:
    received: int = 0
    too_large: bool = False


class RequestGuardMiddleware:
    """Authenticate, throttle and size-limit expensive requests at ingress.

    FastAPI resolves ``UploadFile`` before calling a route function. A limit in
    the route therefore runs only after Starlette has already spooled the whole
    multipart upload. This pure ASGI middleware wraps ``receive`` itself, so it
    also stops chunked requests that omit ``Content-Length``.
    """

    def __init__(self, app: ASGIApp, *, services: Services) -> None:
        self.app = app
        self.services = services

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        method = str(scope.get("method", "GET")).upper()
        path = str(scope.get("path", ""))
        headers = _headers(scope)

        if (
            path.startswith("/v1/")
            and self.services.settings.api_key
            and not api_key_bytes_match(self.services.settings, headers.get(b"x-api-key"))
        ):
            await _json_error(401, "Missing or invalid X-API-Key header", send)
            return

        rate = self._rate_limit(method, path)
        if rate is not None:
            rate_scope, limit = rate
            client = _client_host(scope)
            if not self.services.rate_limiter.allow(rate_scope, client, limit):
                await _json_error(
                    429,
                    f"Rate limit exceeded for {rate_scope}",
                    send,
                    headers={"Retry-After": "60"},
                )
                return

        body_limit = self._body_limit(method, path)
        if body_limit is None:
            await self.app(scope, receive, send)
            return

        content_length = headers.get(b"content-length")
        if content_length is not None:
            try:
                declared = int(content_length)
            except ValueError:
                await _json_error(400, "Invalid Content-Length header", send)
                return
            if declared < 0:
                await _json_error(400, "Invalid Content-Length header", send)
                return
            if declared > body_limit:
                await _json_error(413, self._limit_message(path), send)
                return

        if _is_json_query(method, path) or _is_document_upload(method, path):
            self.services.start_vectorstore_bootstrap()
            if not self.services.vectorstore_ready:
                await _json_error(
                    503,
                    "The vector store is not ready; retry shortly",
                    send,
                    headers={"Retry-After": "2"},
                )
                return

        if not _is_document_upload(method, path):
            await self._serve_within_body_limit(scope, receive, send, body_limit, path)
            return

        # Two copies of an upload live on disk at once: the multipart parser
        # spools the body inside the call below, before any route code runs,
        # and the route then stages its own copy for hashing. Under the shipped
        # Compose file both land on a small /tmp tmpfs, and nothing bounded how
        # many could be in flight — the queue-depth check in the route happens
        # after the first copy already exists, and the rate limit is a rate,
        # not a concurrency. This is the only place early enough to bound it.
        if not await self._reserve_spool_slot(send):
            return
        try:
            await self._serve_within_body_limit(scope, receive, send, body_limit, path)
        finally:
            self.services.upload_access.release()

    async def _serve_within_body_limit(
        self, scope: Scope, receive: Receive, send: Send, body_limit: int, path: str
    ) -> None:
        state = _ReceiveState()

        async def limited_receive() -> Message:
            message = await receive()
            if message["type"] == "http.request":
                state.received += len(message.get("body", b""))
                if state.received > body_limit:
                    state.too_large = True
                    raise RequestBodyTooLarge
            return message

        async def guarded_send(message: Message) -> None:
            # FastAPI can translate receive-time errors into a generic 400. If
            # our byte counter fired, suppress that inner response and emit the
            # precise 413 once the inner app has unwound.
            if not state.too_large:
                await send(message)

        with contextlib.suppress(RequestBodyTooLarge):
            await self.app(scope, limited_receive, guarded_send)

        if state.too_large:
            await _json_error(413, self._limit_message(path), send)

    async def _reserve_spool_slot(self, send: Send) -> bool:
        """Hold one of ``REED_MAX_CONCURRENT_UPLOADS`` spool slots, or refuse.

        A short wait rather than an immediate refusal: a caller queued here is
        not yet using any disk, and a brief queue is a better answer than a 503
        for the burst that a browser multi-file drop produces. Past that it
        refuses the way a full ingestion queue does, because the alternative is
        ``ENOSPC`` — which fails whichever upload happens to be writing, not the
        one that overflowed.
        """
        try:
            await asyncio.wait_for(
                self.services.upload_access.acquire(), timeout=UPLOAD_SLOT_WAIT_SECONDS
            )
        except TimeoutError:
            self.services.metrics.increment("upload_rejections_total")
            await _json_error(
                503,
                "Too many uploads are in flight; retry shortly",
                send,
                headers={"Retry-After": "2"},
            )
            return False
        return True

    def _rate_limit(self, method: str, path: str) -> tuple[str, int] | None:
        settings = self.services.settings
        if method == "POST" and path == "/v1/ask":
            return "ask", settings.ask_rate_limit_per_minute
        # Its own bucket: retrieval without generation is cheap enough that
        # sharing the ask budget would throttle it for no reason.
        if method == "POST" and path == "/v1/search":
            return "search", settings.search_rate_limit_per_minute
        if _is_document_upload(method, path):
            return "upload", settings.upload_rate_limit_per_minute
        return None

    def _body_limit(self, method: str, path: str) -> int | None:
        settings = self.services.settings
        if _is_json_query(method, path):
            return settings.max_json_body_kb * 1024
        if _is_document_upload(method, path):
            return settings.max_upload_mb * MEBIBYTE + settings.max_multipart_overhead_kb * 1024
        return None

    def _limit_message(self, path: str) -> str:
        settings = self.services.settings
        if path.startswith("/v1/documents"):
            return f"Upload exceeds REED_MAX_UPLOAD_MB ({settings.max_upload_mb} MB)"
        return f"Request body exceeds REED_MAX_JSON_BODY_KB ({settings.max_json_body_kb} KB)"


def _is_json_query(method: str, path: str) -> bool:
    """Both JSON query routes: generation, and retrieval on its own.

    They share the body cap and the readiness gate — neither can be served
    without the vector store — but not the rate-limit bucket.
    """
    return method == "POST" and path in {"/v1/ask", "/v1/search"}


def _is_document_upload(method: str, path: str) -> bool:
    """Both routes that carry a document body: upload, and replace-by-PUT.

    They share every ingress guard — rate limit, body cap and the readiness
    gate — because FastAPI spools the whole multipart body while resolving
    ``UploadFile``, before any route-level check can run.
    """
    if method == "POST" and path == "/v1/documents":
        return True
    return method == "PUT" and path.startswith("/v1/documents/")


def _headers(scope: Scope) -> dict[bytes, bytes]:
    return {name.lower(): value for name, value in scope.get("headers", [])}


def _client_host(scope: Scope) -> str:
    client: tuple[str, int] | None = scope.get("client")
    return client[0] if client else "unknown"


async def _json_error(
    status_code: int,
    detail: str,
    send: Send,
    *,
    headers: dict[str, str] | None = None,
) -> None:
    response = JSONResponse({"detail": detail}, status_code=status_code, headers=headers)
    await response({"type": "http", "asgi": {"version": "3.0"}}, _never_receive, send)


async def _never_receive() -> Message:
    return {"type": "http.disconnect"}
