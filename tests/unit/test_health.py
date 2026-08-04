from __future__ import annotations

from fastapi.testclient import TestClient

from reed import __version__


def test_health_reports_the_active_profile(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["version"] == __version__
    assert body["profile"] == "fake"
    assert body["chat_model"] == "fake-chat"


def test_openapi_schema_is_generated(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    assert schema["info"]["title"] == "Reed"
    assert "/health" in schema["paths"]
