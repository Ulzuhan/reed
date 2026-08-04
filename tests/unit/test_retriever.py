from __future__ import annotations

import contextlib
from types import SimpleNamespace
from typing import Any

from reed.config import Settings
from reed.rag.retriever import SNIPPET_CHARS, RetrievedChunk, diversify, retrieve


class FakeStore:
    """Records the candidate count retrieval actually asked Qdrant for."""

    def __init__(self, hits: list[tuple[object, float]]) -> None:
        self.hits = hits
        self.requested_k: int | None = None

    def similarity_search_with_score(
        self, _query: str, k: int, **_kwargs: Any
    ) -> list[tuple[object, float]]:
        self.requested_k = k
        return self.hits


def fake_services(store: FakeStore, **settings: Any) -> Any:
    return SimpleNamespace(
        settings=Settings(_env_file=None, **settings),
        retrieval_store=lambda _mode: store,
        vector_access=contextlib.nullcontext(),
        flush_pending_vector_cleanup=lambda: None,
        registry=SimpleNamespace(ready_ids=lambda ids: set(ids)),
        metrics=SimpleNamespace(
            increment=lambda *_args, **_kwargs: None,
            observe=lambda *_args, **_kwargs: None,
        ),
    )


def document(text: str, doc_id: str) -> object:
    return SimpleNamespace(
        page_content=text,
        metadata={"doc_id": doc_id, "filename": f"{doc_id}.md", "chunk_index": 0},
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
    store = FakeStore([(document("a", "a"), 0.9)])
    services = fake_services(
        store, top_k=4, fetch_k=20, rerank_enabled=False, diversity_enabled=True
    )

    retrieve(services, "question")

    assert store.requested_k == 20


def test_no_candidate_pool_is_fetched_when_nothing_reorders_it() -> None:
    store = FakeStore([(document("a", "a"), 0.9)])
    services = fake_services(
        store, top_k=4, fetch_k=20, rerank_enabled=False, diversity_enabled=False
    )

    retrieve(services, "question")

    assert store.requested_k == 4


def test_the_calibrated_threshold_does_not_leak_into_another_score_domain() -> None:
    # The 0.833 RRF threshold would abstain on every dense cosine score.
    store = FakeStore([(document("a", "a"), 0.42)])
    services = fake_services(
        store,
        profile="local",
        ollama_embed_model="embeddinggemma",
        retrieval_mode="hybrid",
        diversity_enabled=False,
    )

    assert retrieve(services, "question", mode="dense") != []
    assert retrieve(services, "question", mode="hybrid") == []


def test_diversity_limits_one_document_and_prefers_distinct_text() -> None:
    candidates = [
        chunk(doc_id="a", filename="a.md", text="same repeated policy", score=1.0),
        chunk(doc_id="a", filename="a.md", text="same repeated policy", score=0.99),
        chunk(doc_id="b", filename="b.md", text="different supporting evidence", score=0.8),
    ]

    selected = diversify(candidates, k=3, lambda_mult=0.7, max_per_document=1)

    assert [item.filename for item in selected] == ["a.md", "b.md"]
