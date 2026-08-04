from __future__ import annotations

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


@pytest.fixture
def app(settings: Settings) -> FastAPI:
    return create_app(settings)


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client
