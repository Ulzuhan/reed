"""The /v1/ask contract, end to end over embedded Qdrant and fake models."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.integration

HANDBOOK = """# Expenses

Expenses above 75 euros require pre-approval from your manager.
"""


def upload(client: TestClient, tmp_path: Path, name: str = "expenses.md") -> str:
    path = tmp_path / name
    path.write_text(HANDBOOK, encoding="utf-8")
    with path.open("rb") as handle:
        response = client.post("/v1/documents", files={"file": (name, handle, "text/markdown")})
    return response.json()["document_id"]


def read_events(raw: str) -> list[tuple[str, dict[str, object]]]:
    events = []
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


def test_done_event_matches_the_streamed_tokens(client: TestClient, tmp_path: Path) -> None:
    upload(client, tmp_path)

    with client.stream("POST", "/v1/ask", json={"question": "expense threshold?"}) as response:
        events = read_events("".join(response.iter_text()))

    streamed = "".join(data["t"] for name, data in events if name == "token")
    done = next(data for name, data in events if name == "done")
    assert done["answer"] == streamed
    assert "[1]" in streamed  # the fake model honours the citation contract
    assert done["latency_ms"] >= 0


def test_non_streaming_mode_returns_one_json_body(client: TestClient, tmp_path: Path) -> None:
    upload(client, tmp_path)

    response = client.post("/v1/ask", json={"question": "expense threshold?", "stream": False})

    assert response.status_code == 200
    body = response.json()
    assert "[1]" in body["answer"]
    assert body["sources"][0]["filename"] == "expenses.md"
    assert body["latency_ms"] >= 0


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
            client.post("/v1/documents", files={"file": (path.name, handle, "text/markdown")})

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
