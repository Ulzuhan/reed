"""The /v1/search contract: evidence without generation."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from reed.api.app import create_app
from reed.config import Settings
from tests.conftest import wait_until_ready

from .test_ask import upload

pytestmark = pytest.mark.integration


def test_search_returns_ranked_evidence_without_an_answer(
    client: TestClient, tmp_path: Path
) -> None:
    document_id = upload(client, tmp_path)

    response = client.post("/v1/search", json={"query": "What is the expense threshold?"})

    assert response.status_code == 200
    body = response.json()
    assert "answer" not in body
    assert "citation_status" not in body
    assert body["latency_ms"] >= 0
    assert [source["n"] for source in body["sources"]] == list(range(1, len(body["sources"]) + 1))
    first = body["sources"][0]
    assert first["doc_id"] == document_id
    assert first["filename"] == "expenses.md"
    assert "75 euros" in first["excerpt"]
    assert first["location"]


def test_search_reports_the_threshold_verdict_instead_of_abstaining(
    client: TestClient, tmp_path: Path
) -> None:
    """The caller decides. An empty corpus is the only reason to return nothing."""
    response = client.post("/v1/search", json={"query": "What is the expense threshold?"})
    assert response.status_code == 200
    assert response.json()["sources"] == []
    assert response.json()["sufficient_evidence"] is False

    upload(client, tmp_path)
    body = client.post("/v1/search", json={"query": "What is the expense threshold?"}).json()

    assert body["sources"]
    # The fake profile leaves the threshold at zero, so any hit is sufficient;
    # what matters is that the verdict is reported against the same threshold.
    assert body["min_evidence_score"] == 0.0
    assert body["sufficient_evidence"] is True


def test_search_reports_a_retrieval_failure_as_a_bad_gateway(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def dead(*args: object, **kwargs: object) -> list[object]:
        raise RuntimeError("the vector store went away mid-query")

    monkeypatch.setattr("reed.api.routes.search.retrieve", dead)

    response = client.post("/v1/search", json={"query": "anything"})

    assert response.status_code == 502
    # The cause stays in the logs; the caller gets a stable, uninformative reason.
    assert response.json()["detail"] == "Retrieval is temporarily unavailable"
    assert "reed_search_errors_total 1" in client.get("/metrics").text


def test_search_honours_top_k_and_rejects_a_blank_query(client: TestClient, tmp_path: Path) -> None:
    upload(client, tmp_path, name="one.md")
    upload(
        client,
        tmp_path,
        name="two.md",
        content="# Travel\n\nTravel expenses above 200 euros need a written quote.\n",
    )

    capped = client.post("/v1/search", json={"query": "expenses", "top_k": 1})
    assert capped.status_code == 200
    assert len(capped.json()["sources"]) == 1

    assert client.post("/v1/search", json={"query": "   "}).status_code == 422
    assert client.post("/v1/search", json={"query": "x", "top_k": 0}).status_code == 422
    assert client.post("/v1/search", json={}).status_code == 422


def test_search_has_its_own_rate_limit_bucket(tmp_path: Path) -> None:
    settings = Settings(
        profile="fake",
        data_dir=tmp_path / "data",
        collection="test_chunks",
        search_rate_limit_per_minute=1,
        ask_rate_limit_per_minute=10,
        _env_file=None,
    )
    with TestClient(create_app(settings)) as client:
        wait_until_ready(client)
        assert client.post("/v1/search", json={"query": "first"}).status_code == 200
        throttled = client.post("/v1/search", json={"query": "second"})
        assert throttled.status_code == 429
        assert throttled.headers["retry-after"] == "60"
        # Generation keeps its own budget: the buckets do not drain each other.
        answered = client.post("/v1/ask", json={"question": "still fine", "stream": False})
        assert answered.status_code == 200


def test_search_requires_the_api_key_when_one_is_configured(tmp_path: Path) -> None:
    settings = Settings(
        profile="fake",
        data_dir=tmp_path / "data",
        collection="test_chunks",
        api_key="s3cret",
        _env_file=None,
    )
    with TestClient(create_app(settings)) as client:
        assert client.post("/v1/search", json={"query": "hello"}).status_code == 401
        keyed = {"X-API-Key": "s3cret"}
        while client.get("/ready", headers=keyed).status_code != 200:
            pass
        assert client.post("/v1/search", json={"query": "hello"}, headers=keyed).status_code == 200
