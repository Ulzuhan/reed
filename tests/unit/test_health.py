from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from reed import __version__
from reed.api.app import create_app
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
    assert schema["info"]["version"] == __version__


def test_a_keyed_deployment_publishes_its_version_nowhere(tmp_path: Path) -> None:
    """The schema has to agree with `/health` about naming the deployment.

    `/openapi.json` needs no key, so a version published there is public
    however carefully the health endpoints redact theirs. The assertion is the
    property — the string appears nowhere in the document — rather than the
    one field, because the point is that nothing leaks it.
    """
    keyed = Settings(
        profile="fake",
        data_dir=tmp_path / "data",
        collection="test_chunks",
        api_key="s3cret",
        _env_file=None,
    )
    # No lifespan: the schema is served before startup, and this test has no
    # business booting a vector store to read it.
    schema = TestClient(create_app(keyed)).get("/openapi.json").json()

    assert schema["info"]["title"] == "Reed"
    assert schema["info"]["version"] == "unspecified"
    assert __version__ not in json.dumps(schema)


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
        lambda _settings, *, timeout=None: {"models": [{"name": "qwen3.5:4b"}]},  # noqa: ARG005
    )
    try:
        assert services.probe_chat_readiness() == "ok"
        monkeypatch.setattr(
            "reed.model_identity.ollama_tags",
            lambda _settings, *, timeout=None: (  # noqa: ARG005
                _ for _ in ()
            ).throw(ConnectionError("offline")),
        )
        assert services.probe_chat_readiness() == "ok"
        services.settings.readiness_chat_ttl_seconds = 0
        assert services.probe_chat_readiness() == "unavailable"
    finally:
        services.close()
