"""What a restart does to state the previous process left behind."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from reed.api.app import create_app
from reed.config import Settings
from reed.services import build_services

pytestmark = pytest.mark.integration


def stranded_document(settings: Settings, tmp_path: Path) -> str:
    """A row left claiming `processing`, as a crash mid-ingestion would."""
    path = tmp_path / "big.md"
    path.write_text("# Big\n\nContent.", encoding="utf-8")

    services = build_services(settings)
    try:
        from reed.ingest.pipeline import register_upload

        record, _ = register_upload(services, source=path, filename="big.md")
        services.registry.mark_processing(record.id)
        return record.id
    finally:
        services.close()


def test_a_restart_frees_a_document_stranded_mid_ingestion(
    settings: Settings, tmp_path: Path
) -> None:
    document_id = stranded_document(settings, tmp_path)

    with TestClient(create_app(settings)) as client:
        detail = client.get(f"/v1/documents/{document_id}").json()
        # Nothing is in flight after a restart, so the row must not keep
        # claiming it is — it could be neither deleted nor re-uploaded.
        assert detail["status"] == "error"
        assert "restart" in detail["error"]

        assert client.delete(f"/v1/documents/{document_id}").status_code == 204


def test_a_stranded_document_can_be_reuploaded_after_a_restart(
    settings: Settings, tmp_path: Path
) -> None:
    stranded_document(settings, tmp_path)
    path = tmp_path / "big.md"

    with TestClient(create_app(settings)) as client, path.open("rb") as handle:
        response = client.post("/v1/documents", files={"file": ("big.md", handle, "text/markdown")})

    assert response.status_code == 202


def test_health_recovers_once_the_provider_comes_back(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    from langchain_core.embeddings import DeterministicFakeEmbedding

    from reed.providers import FAKE_EMBEDDING_DIM

    calls = {"n": 0}

    def flaky(*_: object, **__: object) -> DeterministicFakeEmbedding:
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("Ollama unreachable at http://localhost:11434")
        return DeterministicFakeEmbedding(size=FAKE_EMBEDDING_DIM)

    monkeypatch.setattr("reed.providers.build_embeddings", flaky)

    with TestClient(create_app(settings)) as client:
        assert client.get("/health").json()["status"] == "degraded"

        # The provider is back; a successful ingestion must clear the flag
        # rather than leaving a healthy instance reporting degraded forever.
        client.post("/v1/documents", files={"file": ("a.md", b"# A\n\nText.", "text/markdown")})

        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["vector_store"] == "ok"
