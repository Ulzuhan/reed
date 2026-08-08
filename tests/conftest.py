from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from reed.api.app import create_app
from reed.config import Settings


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Fake-profile settings pointing at a throwaway data directory."""
    return Settings(
        profile="fake",
        data_dir=tmp_path / "data",
        collection="test_chunks",
        _env_file=None,
    )


def wait_until_ready(
    client: TestClient, timeout: float = 10.0, headers: dict[str, str] | None = None
) -> None:
    """Poll /ready until the vector store is up.

    Uploads answer 503 with Retry-After until the bootstrap finishes — correct
    service behaviour, but a test asserting 202 right after startup races the
    bootstrap on a slow machine unless it waits like a real client would.

    ``headers`` is for a client that carries an API key. `/ready` itself never
    needs one — the guard middleware only authenticates `/v1/*` — but a test
    that models a keyed client should poll the way that client would.
    """
    deadline = time.monotonic() + timeout
    while client.get("/ready", headers=headers).status_code != 200:
        if time.monotonic() > deadline:
            pytest.fail("vector store never became ready")
        time.sleep(0.05)


@pytest.fixture
def app(settings: Settings) -> FastAPI:
    return create_app(settings)


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        wait_until_ready(test_client)
        yield test_client
