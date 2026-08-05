from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from starlette.types import Message, Receive, Scope, Send

from reed.api.middleware import RequestBodyTooLarge, RequestGuardMiddleware
from reed.config import Settings
from reed.services import Services


@pytest.mark.asyncio
async def test_chunked_bodies_are_stopped_without_content_length(tmp_path: Path) -> None:
    settings = Settings(
        profile="fake",
        data_dir=tmp_path,
        max_json_body_kb=16,
        _env_file=None,
    )
    services = Services(settings)
    _ = services.vectorstore
    sent: list[Message] = []

    async def inner(scope: Scope, receive: Receive, send: Send) -> None:
        try:
            while message := await receive():
                if message["type"] != "http.request" or not message.get("more_body", False):
                    break
        except RequestBodyTooLarge:
            # FastAPI similarly turns body parsing errors into a response. The
            # outer middleware must suppress it and preserve the precise 413.
            await send({"type": "http.response.start", "status": 400, "headers": []})
            await send({"type": "http.response.body", "body": b"bad request"})

    messages: AsyncIterator[Message] = _request_chunks([b"a" * 10_000, b"b" * 10_000])

    async def receive() -> Message:
        return await anext(messages)

    async def send(message: Message) -> None:
        sent.append(message)

    scope: Scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/ask",
        "raw_path": b"/v1/ask",
        "query_string": b"",
        "root_path": "",
        "headers": [(b"transfer-encoding", b"chunked")],
        "client": ("127.0.0.1", 1234),
        "server": ("testserver", 80),
        "state": {},
    }

    try:
        await RequestGuardMiddleware(inner, services=services)(scope, receive, send)
    finally:
        services.close()

    assert sent[0]["status"] == 413
    assert json.loads(sent[1]["body"])["detail"].startswith("Request body exceeds")


@pytest.mark.asyncio
async def test_replacement_put_shares_the_upload_body_limit(tmp_path: Path) -> None:
    # FastAPI spools the whole multipart body while resolving UploadFile, so
    # the PUT replacement route needs the same ingress cap as POST uploads.
    settings = Settings(
        profile="fake",
        data_dir=tmp_path,
        max_upload_mb=1,
        _env_file=None,
    )
    services = Services(settings)
    _ = services.vectorstore
    sent: list[Message] = []

    async def inner(scope: Scope, receive: Receive, send: Send) -> None:
        try:
            while message := await receive():
                if message["type"] != "http.request" or not message.get("more_body", False):
                    break
        except RequestBodyTooLarge:
            await send({"type": "http.response.start", "status": 400, "headers": []})
            await send({"type": "http.response.body", "body": b"bad request"})

    oversized = settings.max_upload_mb * 1024 * 1024 + settings.max_multipart_overhead_kb * 1024
    messages: AsyncIterator[Message] = _request_chunks([b"a" * oversized, b"b"])

    async def receive() -> Message:
        return await anext(messages)

    async def send(message: Message) -> None:
        sent.append(message)

    scope = _scope("PUT", "/v1/documents/l-0123456789abcdef")

    try:
        await RequestGuardMiddleware(inner, services=services)(scope, receive, send)
    finally:
        services.close()

    assert sent[0]["status"] == 413
    assert json.loads(sent[1]["body"])["detail"].startswith("Upload exceeds")


@pytest.mark.asyncio
async def test_replacement_put_shares_the_upload_rate_limit(tmp_path: Path) -> None:
    settings = Settings(
        profile="fake",
        data_dir=tmp_path,
        upload_rate_limit_per_minute=1,
        _env_file=None,
    )
    services = Services(settings)
    _ = services.vectorstore
    statuses: list[int] = []

    async def inner(scope: Scope, receive: Receive, send: Send) -> None:
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"{}"})

    async def receive() -> Message:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: Message) -> None:
        if message["type"] == "http.response.start":
            statuses.append(message["status"])

    middleware = RequestGuardMiddleware(inner, services=services)
    try:
        await middleware(_scope("PUT", "/v1/documents/l-0123456789abcdef"), receive, send)
        await middleware(_scope("PUT", "/v1/documents/l-0123456789abcdef"), receive, send)
    finally:
        services.close()

    assert statuses == [200, 429]


def _scope(method: str, path: str) -> Scope:
    return {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": [(b"transfer-encoding", b"chunked")],
        "client": ("127.0.0.1", 1234),
        "server": ("testserver", 80),
        "state": {},
    }


async def _request_chunks(chunks: list[bytes]) -> AsyncIterator[Message]:
    for index, chunk in enumerate(chunks):
        yield {
            "type": "http.request",
            "body": chunk,
            "more_body": index < len(chunks) - 1,
        }
