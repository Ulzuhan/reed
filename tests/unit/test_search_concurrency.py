"""`/v1/search` retrieves under a bound of its own.

Retrieval runs in a worker thread. Without a limiter it draws on anyio's
default pool — one process-wide budget shared with upload spooling and
`/v1/ask`'s own retrieval — so nothing but the per-client rate limit stands
between a burst of searches and the threads everything else needs.
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import anyio.to_thread
import pytest

from reed.api.routes.search import search
from reed.api.schemas import SearchRequest
from reed.config import Settings
from reed.services import Services


class _ConcurrencyProbe:
    """Counts how many retrievals are ever in flight at the same moment."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.live = 0
        self.peak = 0

    def retrieve(self, *args: object, **kwargs: object) -> list[object]:
        with self._lock:
            self.live += 1
            self.peak = max(self.peak, self.live)
        # Long enough that every caller admitted together overlaps here.
        threading.Event().wait(0.05)
        with self._lock:
            self.live -= 1
        return []


@pytest.mark.asyncio
async def test_searches_never_exceed_their_own_thread_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = Settings(
        profile="fake",
        data_dir=tmp_path / "data",
        collection="test_chunks",
        max_concurrent_searches=2,
        _env_file=None,
    )
    services = Services(settings)
    probe = _ConcurrencyProbe()
    monkeypatch.setattr("reed.api.routes.search.retrieve", probe.retrieve)

    try:
        responses = await asyncio.gather(
            *(search(services, SearchRequest(query=f"query {n}")) for n in range(12))
        )
    finally:
        services.close()

    # Every caller is served — the budget queues them, it does not reject them.
    assert len(responses) == 12
    assert all(response.sources == [] for response in responses)
    assert probe.peak <= 2, f"{probe.peak} retrievals ran at once against a budget of 2"


@pytest.mark.asyncio
async def test_search_leaves_the_shared_thread_pool_alone(tmp_path: Path) -> None:
    """The budget is separate, not carved out of anyio's default limiter.

    Uploads and `/v1/ask` keep every token they had; a saturated search route
    cannot take one from them.
    """
    settings = Settings(
        profile="fake",
        data_dir=tmp_path / "data",
        collection="test_chunks",
        max_concurrent_searches=3,
        _env_file=None,
    )
    services = Services(settings)
    try:
        assert services.search_access is not anyio.to_thread.current_default_thread_limiter()
        assert services.search_access.total_tokens == 3
    finally:
        services.close()
