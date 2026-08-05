"""ASGI guards that must run before FastAPI reads or parses a request body."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from reed.api.deps import api_key_bytes_match

if TYPE_CHECKING:
    from reed.services import Services

MEBIBYTE = 1024 * 1024


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

        if (method == "POST" and path == "/v1/ask") or _is_document_upload(method, path):
            self.services.start_vectorstore_bootstrap()
            if not self.services.vectorstore_ready:
                await _json_error(
                    503,
                    "The vector store is not ready; retry shortly",
                    send,
                    headers={"Retry-After": "2"},
                )
                return

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

    def _rate_limit(self, method: str, path: str) -> tuple[str, int] | None:
        settings = self.services.settings
        if method == "POST" and path == "/v1/ask":
            return "ask", settings.ask_rate_limit_per_minute
        if _is_document_upload(method, path):
            return "upload", settings.upload_rate_limit_per_minute
        return None

    def _body_limit(self, method: str, path: str) -> int | None:
        settings = self.services.settings
        if method == "POST" and path == "/v1/ask":
            return settings.max_json_body_kb * 1024
        if _is_document_upload(method, path):
            return settings.max_upload_mb * MEBIBYTE + settings.max_multipart_overhead_kb * 1024
        return None

    def _limit_message(self, path: str) -> str:
        settings = self.services.settings
        if path.startswith("/v1/documents"):
            return f"Upload exceeds REED_MAX_UPLOAD_MB ({settings.max_upload_mb} MB)"
        return f"Request body exceeds REED_MAX_JSON_BODY_KB ({settings.max_json_body_kb} KB)"


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
