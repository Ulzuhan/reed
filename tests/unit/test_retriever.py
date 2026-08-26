from __future__ import annotations

import contextlib
import threading
from types import SimpleNamespace
from typing import Any

import pytest

from reed.config import Settings
from reed.rag.retriever import SNIPPET_CHARS, RetrievedChunk, diversify, retrieve


class FakeQdrant:
    """Records what retrieval actually asked Qdrant for."""

    def __init__(self, points: list[object]) -> None:
        self.points = points
        self.requested_limit: int | None = None
        self.used: str | None = None
        self.prefetched: list[Any] | None = None

    def query_points(self, *, limit: int, **kwargs: Any) -> Any:
        self.requested_limit = limit
        self.used = kwargs.get("using")
        self.prefetched = kwargs.get("prefetch")
        return SimpleNamespace(points=self.points)


def fake_services(client: FakeQdrant, embed_query: Any = None, **settings: Any) -> Any:
    return SimpleNamespace(
        settings=Settings(_env_file=None, **settings),
        vectorstore=object(),
        qdrant=client,
        active_collection_name="test_chunks",
        embeddings=SimpleNamespace(embed_query=embed_query or (lambda _query: [0.1, 0.2, 0.3])),
        sparse_embeddings=SimpleNamespace(
            embed_query=lambda _query: SimpleNamespace(indices=[7], values=[1.0])
        ),
        vector_access=contextlib.nullcontext(),
        flush_pending_vector_cleanup=lambda: None,
        registry=SimpleNamespace(ready_ids=lambda ids: set(ids)),
        metrics=SimpleNamespace(
            increment=lambda *_args, **_kwargs: None,
            observe=lambda *_args, **_kwargs: None,
        ),
    )


def point(text: str, doc_id: str, score: float = 0.9) -> object:
    return SimpleNamespace(
        score=score,
        payload={
            "page_content": text,
            "metadata": {"doc_id": doc_id, "filename": f"{doc_id}.md", "chunk_index": 0},
        },
    )


def chunk(**kwargs: object) -> RetrievedChunk:
    defaults: dict[str, object] = {
        "text": "Expenses above 75 euros require pre-approval.",
        "score": 0.9,
        "doc_id": "d-abc",
        "filename": "expenses.md",
        "page": None,
        "section": None,
    }
    return RetrievedChunk(**{**defaults, **kwargs})  # type: ignore[arg-type]


def test_location_prefers_a_page_number() -> None:
    assert chunk(page=3, section="Pre-approval").location == "expenses.md, p. 3"


def test_location_falls_back_to_the_section() -> None:
    assert chunk(section="Pre-approval").location == "expenses.md — Pre-approval"


def test_location_is_just_the_filename_when_nothing_else_is_known() -> None:
    assert chunk().location == "expenses.md"


def test_snippets_drop_markdown_syntax() -> None:
    snippet = chunk(text="## Pre-approval\n\nAnything over **75 euros** needs `approval`.").snippet

    assert snippet == "Pre-approval Anything over 75 euros needs approval."


def test_snippets_keep_list_content_but_not_bullets() -> None:
    assert chunk(text="- first\n- second").snippet == "first second"


def test_long_snippets_are_truncated_with_an_ellipsis() -> None:
    snippet = chunk(text="word " * 200).snippet

    assert len(snippet) <= SNIPPET_CHARS + 1
    assert snippet.endswith("…")


def test_short_snippets_are_left_alone() -> None:
    assert not chunk(text="short enough").snippet.endswith("…")


def test_diversity_alone_still_widens_the_candidate_pool() -> None:
    # Reranking is off, so only diversity justifies fetching beyond top_k. It
    # has nothing to choose between if retrieval asks for exactly k.
    client = FakeQdrant([point("a", "a")])
    services = fake_services(
        client, top_k=4, fetch_k=20, rerank_enabled=False, diversity_enabled=True
    )

    retrieve(services, "question")

    assert client.requested_limit == 20


def test_no_candidate_pool_is_fetched_when_nothing_reorders_it() -> None:
    client = FakeQdrant([point("a", "a")])
    services = fake_services(
        client, top_k=4, fetch_k=20, rerank_enabled=False, diversity_enabled=False
    )

    retrieve(services, "question")

    assert client.requested_limit == 4


def test_the_calibrated_threshold_does_not_leak_into_another_score_domain() -> None:
    # The 0.833 RRF threshold would abstain on every dense cosine score.
    client = FakeQdrant([point("a", "a", score=0.42)])
    services = fake_services(
        client,
        profile="local",
        ollama_embed_model="embeddinggemma",
        retrieval_mode="hybrid",
        diversity_enabled=False,
    )

    assert retrieve(services, "question", mode="dense") != []
    assert retrieve(services, "question", mode="hybrid") == []


def test_only_the_vectors_the_mode_queries_with_are_embedded() -> None:
    """A sparse search must not pay for a dense round-trip it never uses."""
    dense_calls: list[str] = []

    def embed_query(query: str) -> list[float]:
        dense_calls.append(query)
        return [0.1]

    client = FakeQdrant([point("a", "a")])
    services = fake_services(client, embed_query=embed_query, fetch_k=4)

    retrieve(services, "question", mode="sparse")
    assert dense_calls == []
    assert client.used == "sparse"

    retrieve(services, "question", mode="dense")
    assert dense_calls == ["question"]
    assert client.used == "dense"

    retrieve(services, "question", mode="hybrid")
    assert dense_calls == ["question", "question"]
    assert client.prefetched is not None
    assert [branch.using for branch in client.prefetched] == ["dense", "sparse"]


def test_an_unknown_retrieval_mode_is_refused() -> None:
    services = fake_services(FakeQdrant([]))

    with pytest.raises(ValueError, match="unsupported retrieval mode: nonsense"):
        retrieve(services, "question", mode="nonsense")


def test_the_query_is_embedded_before_the_vector_lock_is_taken() -> None:
    """The point of the exercise: embedding is a provider round-trip.

    With embedded Qdrant `vector_access` is a process-global lock that ingestion
    also needs, so holding it across the embedding call serialises every search
    against every write. Checked from another thread because the lock is an
    RLock — the thread that holds it can always re-acquire it.
    """
    lock = threading.RLock()
    free_while_embedding: list[bool] = []

    def embed_query(_query: str) -> list[float]:
        free_while_embedding.append(_lock_is_free_elsewhere(lock))
        return [0.1]

    client = FakeQdrant([point("a", "a")])
    services = fake_services(client, embed_query=embed_query)
    services.vector_access = lock

    retrieve(services, "question")

    assert free_while_embedding == [True]


def _lock_is_free_elsewhere(lock: threading.RLock) -> bool:
    """Whether a *different* thread could take the lock right now."""
    acquired: list[bool] = []

    def probe() -> None:
        if lock.acquire(blocking=False):
            acquired.append(True)
            lock.release()

    prober = threading.Thread(target=probe)
    prober.start()
    prober.join()
    return bool(acquired)


def test_diversity_limits_one_document_and_prefers_distinct_text() -> None:
    candidates = [
        chunk(doc_id="a", filename="a.md", text="same repeated policy", score=1.0),
        chunk(doc_id="a", filename="a.md", text="same repeated policy", score=0.99),
        chunk(doc_id="b", filename="b.md", text="different supporting evidence", score=0.8),
    ]

    selected = diversify(candidates, k=3, lambda_mult=0.7, max_per_document=1)

    assert [item.filename for item in selected] == ["a.md", "b.md"]
