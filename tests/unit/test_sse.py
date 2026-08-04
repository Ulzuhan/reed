from __future__ import annotations

import json

from reed.api.sse import PING, SSE_HEADERS, sse_event


def test_frames_are_named_and_terminated() -> None:
    frame = sse_event("token", {"t": "hello"})

    assert frame.startswith("event: token\n")
    assert frame.endswith("\n\n")


def test_payload_is_single_line_json() -> None:
    frame = sse_event("done", {"answer": "line one\nline two", "latency_ms": 12})

    data = frame.split("data: ", 1)[1].rstrip("\n")
    assert "\n" not in data
    assert json.loads(data)["answer"] == "line one\nline two"


def test_non_ascii_survives_unescaped() -> None:
    frame = sse_event("token", {"t": "más allá"})

    assert "más allá" in frame


def test_heartbeat_is_a_comment_frame() -> None:
    assert PING.startswith(":")
    assert PING.endswith("\n\n")


def test_headers_disable_proxy_buffering() -> None:
    assert SSE_HEADERS["X-Accel-Buffering"] == "no"
    assert SSE_HEADERS["Cache-Control"] == "no-cache"
