from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from reed import __version__
from reed.config import Settings
from reed.services import Services


def test_health_reports_the_active_profile(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["version"] == __version__
    assert body["profile"] == "fake"
    assert body["chat_model"] == "fake-chat"
    assert body["vector_store"] == "not_checked"


def test_openapi_schema_is_generated(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    assert schema["info"]["title"] == "Reed"
    assert "/health" in schema["paths"]
    assert "/ready" in schema["paths"]


def test_ui_responses_have_a_content_security_policy(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert "default-src 'self'" in response.headers["content-security-policy"]
    assert response.headers["x-content-type-options"] == "nosniff"


def test_sensitive_gets_and_metrics_are_not_cacheable(client: TestClient) -> None:
    assert client.get("/v1/documents").headers["cache-control"] == "no-store"
    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    assert metrics.headers["cache-control"] == "no-store"
    assert "reed_ingestion_queue_depth" in metrics.text


def test_local_chat_readiness_is_live_and_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = Settings(
        profile="local",
        data_dir=tmp_path,
        ollama_chat_model="qwen3.5:4b",
        readiness_chat_ttl_seconds=30,
        _env_file=None,
    )
    services = Services(settings)
    monkeypatch.setattr(
        "reed.model_identity.ollama_tags",
        lambda _settings: {"models": [{"name": "qwen3.5:4b"}]},
    )
    try:
        assert services.probe_chat_readiness() == "ok"
        monkeypatch.setattr(
            "reed.model_identity.ollama_tags",
            lambda _settings: (_ for _ in ()).throw(ConnectionError("offline")),
        )
        assert services.probe_chat_readiness() == "ok"
        services.settings.readiness_chat_ttl_seconds = 0
        assert services.probe_chat_readiness() == "unavailable"
    finally:
        services.close()
