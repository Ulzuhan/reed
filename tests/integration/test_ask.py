"""The /v1/ask contract, end to end over embedded Qdrant and fake models."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient

from reed.api.app import create_app
from reed.config import Settings

pytestmark = pytest.mark.integration

HANDBOOK = """# Expenses

Expenses above 75 euros require pre-approval from your manager.
"""


def upload(
    client: TestClient, tmp_path: Path, name: str = "expenses.md", content: str = HANDBOOK
) -> str:
    path = tmp_path / name
    # Uploads deduplicate on content, so a caller wanting a second document has
    # to give it different text, not just a different filename.
    path.write_text(content, encoding="utf-8")
    with path.open("rb") as handle:
        response = client.post("/v1/documents", files={"file": (name, handle, "text/markdown")})
    document_id = cast(str, response.json()["document_id"])
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if client.get(f"/v1/documents/{document_id}").json()["status"] == "ready":
            return document_id
        time.sleep(0.01)
    raise AssertionError(f"document {document_id} did not become ready")


def read_events(raw: str) -> list[tuple[str, dict[str, Any]]]:
    events: list[tuple[str, dict[str, Any]]] = []
    for frame in raw.split("\n\n"):
        name = ""
        data = ""
        for line in frame.split("\n"):
            if line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                data = line[5:].strip()
        if name and data:
            events.append((name, json.loads(data)))
    return events


def test_stream_reports_sources_before_the_first_token(client: TestClient, tmp_path: Path) -> None:
    upload(client, tmp_path)

    with client.stream(
        "POST", "/v1/ask", json={"question": "What is the expense threshold?"}
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        assert response.headers["x-accel-buffering"] == "no"
        events = read_events("".join(response.iter_text()))

    names = [name for name, _ in events]
    assert names[0] == "meta"
    assert names.index("sources") < names.index("token")
    assert names[-1] == "done"


def test_stream_carries_citable_sources(client: TestClient, tmp_path: Path) -> None:
    document_id = upload(client, tmp_path)

    with client.stream("POST", "/v1/ask", json={"question": "expense threshold?"}) as response:
        events = read_events("".join(response.iter_text()))

    sources = next(data["sources"] for name, data in events if name == "sources")
    assert sources
    first = sources[0]
    assert first["n"] == 1
    assert first["doc_id"] == document_id
    assert first["filename"] == "expenses.md"
    assert first["snippet"]
    assert "75 euros" in first["excerpt"]


def test_done_event_matches_the_streamed_tokens(client: TestClient, tmp_path: Path) -> None:
    upload(client, tmp_path)

    with client.stream("POST", "/v1/ask", json={"question": "expense threshold?"}) as response:
        events = read_events("".join(response.iter_text()))

    streamed = "".join(data["t"] for name, data in events if name == "token")
    done = next(data for name, data in events if name == "done")
    assert done["answer"] == streamed
    assert "[1]" in streamed  # the fake model honours the citation contract
    assert done["latency_ms"] >= 0
    assert done["citation_status"] == "valid"


def test_streaming_failures_are_exposed_in_metrics(client: TestClient, tmp_path: Path) -> None:
    upload(client, tmp_path)

    class DeadModel:
        async def astream(self, *_: object, **__: object):  # type: ignore[no-untyped-def]
            if "__yield__" in __:
                yield ""
            raise RuntimeError("provider unavailable")

    services = client.app.state.services  # type: ignore[attr-defined]
    services._chat = DeadModel()
    with client.stream("POST", "/v1/ask", json={"question": "expense threshold?"}) as response:
        events = read_events("".join(response.iter_text()))

    assert any(name == "error" for name, _data in events)
    assert "reed_ask_errors_total 1" in client.get("/metrics").text


def test_non_streaming_mode_returns_one_json_body(client: TestClient, tmp_path: Path) -> None:
    upload(client, tmp_path)

    response = client.post("/v1/ask", json={"question": "expense threshold?", "stream": False})

    assert response.status_code == 200
    body = response.json()
    assert "[1]" in body["answer"]
    assert body["sources"][0]["filename"] == "expenses.md"
    assert body["latency_ms"] >= 0
    assert body["citation_status"] == "valid"


def test_asking_with_no_documents_says_so_instead_of_guessing(client: TestClient) -> None:
    response = client.post("/v1/ask", json={"question": "anything?", "stream": False})

    assert response.status_code == 200
    assert response.json()["sources"] == []
    assert "Upload" in response.json()["answer"]


def test_top_k_limits_the_sources(client: TestClient, tmp_path: Path) -> None:
    for index in range(3):
        path = tmp_path / f"doc{index}.md"
        path.write_text(f"# Doc {index}\n\nExpense rule number {index}.", encoding="utf-8")
        with path.open("rb") as handle:
            document_id = client.post(
                "/v1/documents", files={"file": (path.name, handle, "text/markdown")}
            ).json()["document_id"]
        deadline = time.monotonic() + 3
        while client.get(f"/v1/documents/{document_id}").json()["status"] != "ready":
            assert time.monotonic() < deadline
            time.sleep(0.01)

    response = client.post(
        "/v1/ask", json={"question": "expense rule", "stream": False, "top_k": 2}
    )

    assert len(response.json()["sources"]) == 2


def test_history_is_accepted(client: TestClient, tmp_path: Path) -> None:
    upload(client, tmp_path)

    response = client.post(
        "/v1/ask",
        json={
            "question": "and for contractors?",
            "stream": False,
            "history": [
                {"role": "user", "content": "What is the expense threshold?"},
                {"role": "assistant", "content": "75 euros [1]."},
            ],
        },
    )

    assert response.status_code == 200


def test_an_empty_question_is_rejected(client: TestClient) -> None:
    assert client.post("/v1/ask", json={"question": ""}).status_code == 422
    assert client.post("/v1/ask", json={"question": "   "}).status_code == 422


def test_history_has_hard_size_limits(client: TestClient) -> None:
    too_many = [{"role": "user", "content": "x"}] * 7
    assert client.post("/v1/ask", json={"question": "q", "history": too_many}).status_code == 422

    too_long = [{"role": "user", "content": "x" * 8_001}]
    assert client.post("/v1/ask", json={"question": "q", "history": too_long}).status_code == 422


def test_expensive_asks_are_rate_limited(settings: Settings) -> None:
    limited = settings.model_copy(update={"ask_rate_limit_per_minute": 1})
    with TestClient(create_app(limited)) as limited_client:
        first = limited_client.post("/v1/ask", json={"question": "one", "stream": False})
        second = limited_client.post("/v1/ask", json={"question": "two", "stream": False})

    assert first.status_code == 200
    assert second.status_code == 429
    assert second.headers["retry-after"] == "60"
