"""What a caller sees while it is queued behind `max_concurrent_asks`.

The stream commits to a 200 and its `meta` event before it needs a slot, so
everything after that has to keep the connection alive on its own. It used to
emit nothing at all until a slot freed.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest

from reed.api.routes import ask as ask_route
from reed.api.routes.ask import _sse_body
from reed.api.schemas import AskRequest
from reed.api.sse import PING
from reed.config import Settings
from reed.services import Services

# Long enough that a heartbeat is a deliberate act rather than a scheduling
# accident, short enough that the test does not sit around waiting for one.
FAST_PING = 0.02


@pytest.fixture
def services(tmp_path: Path) -> Services:
    return Services(
        Settings(
            profile="fake",
            data_dir=tmp_path / "data",
            max_concurrent_asks=1,
            _env_file=None,
        )
    )


# The regression these tests guard is "nothing arrives, ever". Waiting for it
# without a deadline would hang CI instead of failing it, which is the trap
# #44 caught elsewhere in this suite.
FRAME_DEADLINE = 5.0


async def _next(stream: AsyncGenerator[str, None]) -> str:
    try:
        return await asyncio.wait_for(anext(stream), timeout=FRAME_DEADLINE)
    except TimeoutError:
        pytest.fail(f"the stream produced no frame within {FRAME_DEADLINE}s")


async def _take(stream: AsyncGenerator[str, None], count: int) -> list[str]:
    return [await _next(stream) for _ in range(count)]


async def test_a_queued_stream_heartbeats_instead_of_going_silent(
    services: Services, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ask_route, "PING_INTERVAL_SECONDS", FAST_PING)
    # The only slot is taken, as it would be by another in-flight ask.
    await services.ask_access.acquire()

    stream = _sse_body(services, AskRequest(question="what is the threshold?"))
    try:
        frames = await _take(stream, 3)
    finally:
        await stream.aclose()

    assert frames[0].startswith("event: meta\n")
    # Before the fix these two never arrived: the generator blocked on the
    # semaphore with the response already committed and nothing else to send.
    assert frames[1:] == [PING, PING]


async def test_a_queued_stream_proceeds_once_a_slot_frees(
    services: Services, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ask_route, "PING_INTERVAL_SECONDS", FAST_PING)
    await services.ask_access.acquire()

    stream = _sse_body(services, AskRequest(question="what is the threshold?"))
    try:
        assert (await _next(stream)).startswith("event: meta\n")
        assert await _next(stream) == PING

        services.ask_access.release()

        # The corpus is empty, so retrieval abstains and the answer is the
        # deterministic no-context one — enough to prove the stream resumed.
        names: list[str] = []
        while len(names) < 6:
            frame = await _next(stream)
            if frame == PING:
                continue
            names.append(frame.split("\n", 1)[0])
            if names[-1] == "event: done":
                break
        assert "event: sources" in names
        assert names[-1] == "event: done"
    finally:
        await stream.aclose()


async def test_a_stream_abandoned_while_queued_does_not_leak_its_slot(
    services: Services, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A client that hangs up mid-queue must not consume the slot it never got."""
    monkeypatch.setattr(ask_route, "PING_INTERVAL_SECONDS", FAST_PING)
    await services.ask_access.acquire()

    stream = _sse_body(services, AskRequest(question="abandoned"))
    await _take(stream, 2)
    await stream.aclose()

    services.ask_access.release()

    # One permit, and it is available: the abandoned stream took none with it.
    await asyncio.wait_for(services.ask_access.acquire(), timeout=1)
    assert services.ask_access.locked()


async def test_a_completed_stream_hands_its_slot_back(services: Services) -> None:
    stream = _sse_body(services, AskRequest(question="anything?"))

    async def drain() -> None:
        async for _frame in stream:
            pass

    try:
        await asyncio.wait_for(drain(), timeout=FRAME_DEADLINE)
    finally:
        await stream.aclose()

    await asyncio.wait_for(services.ask_access.acquire(), timeout=1)
