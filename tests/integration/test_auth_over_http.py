"""API-key auth as it behaves through a real request, not just as a function.

Headers reach the endpoint latin-1 decoded, so a key that compares fine in
isolation can still reject every request over the wire.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from reed.api.app import create_app
from reed.config import Settings

pytestmark = pytest.mark.integration

ACCENTED_KEY = "contraseña"

# HTTP headers carry bytes; a client sends the key utf-8 encoded, and Starlette
# hands the endpoint the latin-1 decoding of exactly those bytes.
SENT = {"X-API-Key": ACCENTED_KEY.encode("utf-8")}


@pytest.fixture
def guarded(settings: Settings) -> TestClient:
    return TestClient(create_app(settings.model_copy(update={"api_key": ACCENTED_KEY})))


def test_the_right_accented_key_is_accepted(guarded: TestClient) -> None:
    response = guarded.get("/v1/documents", headers=SENT)
    assert response.status_code == 200


def test_a_wrong_key_is_rejected_not_crashed(guarded: TestClient) -> None:
    wrong = {"X-API-Key": "wrông".encode()}
    assert guarded.get("/v1/documents", headers=wrong).status_code == 401
    assert guarded.get("/v1/documents", headers={"X-API-Key": "nope"}).status_code == 401
    assert guarded.get("/v1/documents").status_code == 401


def test_a_latin1_client_is_accepted_too(guarded: TestClient) -> None:
    # `requests` encodes str headers as latin-1 rather than utf-8; the same
    # configured key must work from both kinds of client.
    latin1 = {"X-API-Key": ACCENTED_KEY.encode("latin-1")}
    assert guarded.get("/v1/documents", headers=latin1).status_code == 200


def test_health_is_never_behind_the_key(guarded: TestClient) -> None:
    assert guarded.get("/health").status_code == 200


def test_asking_is_guarded_too(guarded: TestClient) -> None:
    body = {"question": "anything?", "stream": False}
    assert guarded.post("/v1/ask", json=body).status_code == 401
    assert guarded.post("/v1/ask", json=body, headers=SENT).status_code == 200
