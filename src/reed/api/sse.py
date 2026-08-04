"""Server-Sent Events framing.

Hand-rolled rather than pulled from a library: the format is four lines of
code, and owning it means the heartbeat and the buffering headers are visible
right where the stream is built.
"""

from __future__ import annotations

import json
from typing import Any

# Reverse proxies happily buffer a streaming response into nothing. These
# headers, plus the periodic comment below, keep tokens flowing through them.
SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}

PING = ": ping\n\n"


def sse_event(name: str, data: dict[str, Any]) -> str:
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f"event: {name}\ndata: {payload}\n\n"
