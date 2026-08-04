"""What happens at startup when the vector store cannot serve queries."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from reed.api.app import create_app
from reed.config import Settings
from reed.providers import FAKE_EMBEDDING_DIM
from reed.rag.vectorstore import CollectionMismatchError
from reed.services import build_services

pytestmark = pytest.mark.integration


def ingest_something(settings: Settings, tmp_path: Path) -> None:
    from reed.ingest.pipeline import ingest_path

    path = tmp_path / "note.md"
    path.write_text("# Note\n\nSomething to embed.", encoding="utf-8")

    services = build_services(settings)
    try:
        ingest_path(services, path)
    finally:
        services.close()


def test_a_collection_from_another_model_fails_the_boot(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ingest_something(settings, tmp_path)

    # Same data directory, differently sized embeddings — exactly what
    # switching REED_PROFILE against an existing ./data does.
    from langchain_core.embeddings import DeterministicFakeEmbedding

    monkeypatch.setattr(
        "reed.providers.build_embeddings",
        lambda *_, **__: DeterministicFakeEmbedding(size=FAKE_EMBEDDING_DIM * 2),
    )

    with pytest.raises(CollectionMismatchError, match="delete"), TestClient(create_app(settings)):
        pass


def test_an_unreachable_provider_degrades_instead_of_failing(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    def explode(*_: object, **__: object) -> None:
        raise ConnectionError("Ollama unreachable at http://localhost:11434")

    monkeypatch.setattr("reed.providers.build_embeddings", explode)

    with TestClient(create_app(settings)) as client:
        body = client.get("/health").json()

    # The model may well come back without a restart, so Reed keeps serving —
    # but it must not claim the store is fine.
    assert body["status"] == "degraded"
    assert "Ollama unreachable" in body["vector_store"]


def test_a_healthy_startup_reports_ok(settings: Settings) -> None:
    with TestClient(create_app(settings)) as client:
        body = client.get("/health").json()

    assert body["status"] == "ok"
    assert body["vector_store"] == "ok"
    assert FAKE_EMBEDDING_DIM > 0
