"""API-key auth as it behaves through a real request, not just as a function."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from reed.api.app import create_app
from reed.config import Settings

pytestmark = pytest.mark.integration

API_KEY = "correct-horse-battery-staple"
SENT = {"X-API-Key": API_KEY}


@pytest.fixture
def guarded(settings: Settings) -> TestClient:
    return TestClient(create_app(settings.model_copy(update={"api_key": API_KEY})))


def test_the_right_key_is_accepted(guarded: TestClient) -> None:
    response = guarded.get("/v1/documents", headers=SENT)
    assert response.status_code == 200


def test_a_wrong_key_is_rejected_not_crashed(guarded: TestClient) -> None:
    wrong = [(b"X-API-Key", "wrông".encode())]
    assert guarded.get("/v1/documents", headers=wrong).status_code == 401
    assert guarded.get("/v1/documents", headers={"X-API-Key": "nope"}).status_code == 401
    assert guarded.get("/v1/documents").status_code == 401


def test_health_is_never_behind_the_key(guarded: TestClient) -> None:
    response = guarded.get("/health")
    assert response.status_code == 200
    assert response.json()["profile"] is None
    assert response.json()["documents"] is None


def test_asking_is_guarded_too(guarded: TestClient) -> None:
    body = {"question": "anything?", "stream": False}
    assert guarded.post("/v1/ask", json=body).status_code == 401
    assert guarded.post("/v1/ask", json=body, headers=SENT).status_code == 200
