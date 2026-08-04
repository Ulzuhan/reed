"""What a restart does to state the previous process left behind."""

from __future__ import annotations

import time
from pathlib import Path
from threading import Event

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


def test_a_durably_queued_document_resumes_after_restart(
    settings: Settings, tmp_path: Path
) -> None:
    path = tmp_path / "queued.md"
    path.write_text("# Queued\n\nResume this original.", encoding="utf-8")
    first = build_services(settings)
    try:
        from reed.ingest.pipeline import register_upload

        record, _ = register_upload(first, source=path, filename=path.name)
        assert record.status == "queued"
    finally:
        first.close()

    with TestClient(create_app(settings)) as client:
        deadline = time.monotonic() + 3
        status = client.get(f"/v1/documents/{record.id}").json()["status"]
        while status not in {"ready", "error"} and time.monotonic() < deadline:
            time.sleep(0.01)
            status = client.get(f"/v1/documents/{record.id}").json()["status"]

        assert status == "ready"


def test_bounded_ingestion_queue_rejects_excess_work(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    entered = Event()
    release = Event()
    original = __import__("reed.ingest.pipeline", fromlist=["parse_file_isolated"])
    real_parse = original.parse_file_isolated

    def blocked_parse(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        entered.set()
        release.wait(timeout=3)
        return real_parse(*args, **kwargs)

    monkeypatch.setattr("reed.ingest.pipeline.parse_file_isolated", blocked_parse)
    limited = settings.model_copy(
        update={"max_concurrent_ingestions": 1, "max_queued_ingestions": 1}
    )
    try:
        with TestClient(create_app(limited)) as client:
            first = client.post(
                "/v1/documents",
                files={"file": ("one.md", b"# One\n\nFirst.", "text/markdown")},
            )
            assert first.status_code == 202
            assert entered.wait(timeout=1)
            second = client.post(
                "/v1/documents",
                files={"file": ("two.md", b"# Two\n\nSecond.", "text/markdown")},
            )
            assert second.status_code == 202
            rejected = client.post(
                "/v1/documents",
                files={"file": ("three.md", b"# Three\n\nThird.", "text/markdown")},
            )
            assert rejected.status_code == 503
            assert rejected.headers["retry-after"] == "2"
    finally:
        release.set()


def test_health_notices_a_remote_qdrant_that_dies_after_startup(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A restarted or partitioned Qdrant container is invisible to a health
    # check that only remembers what startup found.
    with TestClient(create_app(settings)) as client:
        assert client.get("/ready").json()["status"] == "ok"

        services = client.app.state.services  # type: ignore[attr-defined]
        monkeypatch.setattr(services.settings, "qdrant_url", "http://qdrant:6333")
        monkeypatch.setattr(
            type(services.qdrant),
            "get_collections",
            lambda _: (_ for _ in ()).throw(ConnectionError("connection refused")),
        )

        body = client.get("/ready").json()

    assert body["status"] == "degraded"
    assert body["vector_store"] == "unavailable"


def test_readiness_returns_503_when_a_remote_store_dies(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    with TestClient(create_app(settings)) as client:
        services = client.app.state.services  # type: ignore[attr-defined]
        monkeypatch.setattr(services.settings, "qdrant_url", "http://qdrant:6333")
        monkeypatch.setattr(
            type(services.qdrant),
            "get_collections",
            lambda _: (_ for _ in ()).throw(ConnectionError("secret endpoint")),
        )

        response = client.get("/ready")

    assert response.status_code == 503
    assert "secret endpoint" not in response.text


def test_health_does_not_probe_an_embedded_store(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Embedded operations are serialised, so probing here would park /health
    # behind whatever upload happens to be embedding.
    with TestClient(create_app(settings)) as client:
        services = client.app.state.services  # type: ignore[attr-defined]
        monkeypatch.setattr(
            type(services.qdrant),
            "get_collections",
            lambda _: pytest.fail("/health must not touch the embedded store"),
        )

        assert client.get("/health").json()["status"] == "ok"


def test_health_recovers_once_the_provider_comes_back(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    from langchain_core.embeddings import DeterministicFakeEmbedding

    from reed.providers import FAKE_EMBEDDING_DIM

    calls = {"n": 0}

    def flaky(*_: object, **__: object) -> DeterministicFakeEmbedding:
        calls["n"] += 1
        if calls["n"] <= 2:
            raise ConnectionError("Ollama unreachable at http://localhost:11434")
        return DeterministicFakeEmbedding(size=FAKE_EMBEDDING_DIM)

    monkeypatch.setattr("reed.providers.build_embeddings", flaky)

    retrying = settings.model_copy(
        update={
            "readiness_retry_initial_seconds": 0.05,
            "readiness_retry_max_seconds": 0.05,
        }
    )
    with TestClient(create_app(retrying)) as client:
        assert client.get("/ready").json()["status"] == "degraded"

        # Readiness triggers at most one background retry after each cooldown.
        # The provider recovers on the third attempt without restarting Reed.
        deadline = time.monotonic() + 2
        body = client.get("/ready").json()
        while body["status"] != "ok" and time.monotonic() < deadline:
            time.sleep(0.06)
            body = client.get("/ready").json()

        assert body["status"] == "ok"
        assert body["vector_store"] == "ok"
        assert calls["n"] == 3


def test_readiness_probes_never_duplicate_a_slow_bootstrap(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    entered = Event()
    release = Event()
    calls = {"n": 0}

    def slow_failure(*_: object, **__: object) -> None:
        calls["n"] += 1
        entered.set()
        release.wait(timeout=2)
        raise ConnectionError("provider unavailable")

    monkeypatch.setattr("reed.providers.build_embeddings", slow_failure)
    immediate = settings.model_copy(update={"startup_grace_seconds": 0})

    try:
        with TestClient(create_app(immediate)) as client:
            assert entered.wait(timeout=1)
            responses = [client.get("/ready") for _ in range(20)]
            assert all(response.status_code == 503 for response in responses)
            assert calls["n"] == 1
            services = client.app.state.services  # type: ignore[attr-defined]
            release.set()
            assert services.wait_for_vectorstore_bootstrap(timeout=1)
    finally:
        release.set()
