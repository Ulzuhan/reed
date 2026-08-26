"""Reed issues the search itself; it must stay the search langchain issued.

The evidence threshold is a number calibrated in the fused RRF score domain
(`EMBEDDINGGEMMA_HYBRID_MIN_SCORE`). A hand-built query that ranked or scored
even slightly differently would not fail any other test — it would quietly move
the abstention boundary. These tests pin the two implementations together.

The corpus is deliberately larger than the query limit and contains uncommitted
points. With a handful of chunks and a generous limit every branch returns
everything, and the assertions hold no matter what the prefetch limit or the
committed filter say — which makes the comparison vacuous. Sized this way, both
actually bind.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator

import pytest
from qdrant_client import models

from reed.config import Settings
from reed.rag.retriever import _embed_query, _query_points
from reed.rag.vectorstore import (
    COMMITTED_PAYLOAD_KEY,
    CONTENT_PAYLOAD_KEY,
    DENSE_VECTOR_NAME,
    METADATA_PAYLOAD_KEY,
    SPARSE_VECTOR_NAME,
    build_vectorstore,
)
from reed.services import Services, build_services

pytestmark = pytest.mark.integration

# Overlapping vocabulary, so dense and sparse rank the same corpus differently
# and RRF has genuine disagreement to fuse.
TOPICS = [
    "expenses above {n} euros require pre-approval from your manager",
    "travel bookings over {n} euros need a written quote first",
    "remote days are agreed with your manager each quarter, up to {n}",
    "report a suspected breach within {n} hours of noticing it",
    "invoices are paid {n} days after approval by finance",
    "equipment budgets renew every {n} months for each engineer",
]
CORPUS_SIZE = 48
UNCOMMITTED = 6
LIMIT = 5

# BM25 scores nothing for a query that shares no term with the corpus, so this
# one is an equivalence case (both sides must agree on returning nothing) but
# not evidence that the corpus is reachable.
UNRELATED_QUERY = "zzyzx qwertyuiop flibbertigibbet"
QUERIES = [
    "expense threshold",
    "How much can I spend before asking for approval?",
    "written quote for travel",
    "breach",
    UNRELATED_QUERY,
]


def _committed_filter() -> models.Filter:
    return models.Filter(
        must=[
            models.FieldCondition(
                key=COMMITTED_PAYLOAD_KEY,
                match=models.MatchValue(value=True),
            )
        ]
    )


def _langchain_hits(
    services: Services, query: str, mode: str, limit: int
) -> list[tuple[str, float]]:
    """What `similarity_search_with_score` returns — the previous implementation."""
    store = build_vectorstore(
        services.qdrant,
        services.settings,
        services.embeddings,
        services.sparse_embeddings,
        collection_name=services.active_collection_name,
        retrieval_mode=mode,
    )
    hits = store.similarity_search_with_score(query, k=limit, filter=_committed_filter())
    return [(document.page_content, score) for document, score in hits]


def _reed_hits(services: Services, query: str, mode: str, limit: int) -> list[tuple[str, float]]:
    dense, sparse = _embed_query(services, query, mode)
    points = _query_points(
        services.qdrant,
        collection_name=services.active_collection_name,
        query_mode=mode,
        dense=dense,
        sparse=sparse,
        limit=limit,
        query_filter=_committed_filter(),
    )
    return [(str((point.payload or {})[CONTENT_PAYLOAD_KEY]), point.score) for point in points]


@pytest.fixture
def seeded(settings: Settings) -> Iterator[Services]:
    """Points written straight into the collection, in the pipeline's own shape.

    Not through `ingest_path`: a corpus this size would spawn one isolated
    parser process per document, and none of what that exercises is under test
    here.
    """
    services = build_services(settings)
    _ = services.vectorstore
    texts = [
        TOPICS[index % len(TOPICS)].format(n=index) + f" (clause {index})"
        for index in range(CORPUS_SIZE)
    ]
    dense_vectors = services.embeddings.embed_documents(texts)
    sparse_vectors = services.sparse_embeddings.embed_documents(texts)
    services.qdrant.upsert(
        collection_name=services.active_collection_name,
        points=[
            models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"equivalence:{index}")),
                vector={
                    DENSE_VECTOR_NAME: dense,
                    SPARSE_VECTOR_NAME: models.SparseVector(
                        indices=sparse.indices, values=sparse.values
                    ),
                },
                payload={
                    CONTENT_PAYLOAD_KEY: text,
                    METADATA_PAYLOAD_KEY: {
                        "doc_id": f"d-{index:032x}",
                        "filename": f"clause-{index}.md",
                        "chunk_index": 0,
                        # The tail is staged, not published. Retrieval must not
                        # see it, and the filter is what keeps it out.
                        "committed": index < CORPUS_SIZE - UNCOMMITTED,
                    },
                },
            )
            for index, (text, dense, sparse) in enumerate(
                zip(texts, dense_vectors, sparse_vectors, strict=True)
            )
        ],
        wait=True,
    )
    yield services
    services.close()


@pytest.mark.parametrize("mode", ["dense", "sparse", "hybrid"])
def test_the_hand_built_query_matches_langchain_exactly(seeded: Services, mode: str) -> None:
    for query in QUERIES:
        expected = _langchain_hits(seeded, query, mode, limit=LIMIT)
        actual = _reed_hits(seeded, query, mode, limit=LIMIT)

        if query != UNRELATED_QUERY:
            # Guards against a comparison of two empty lists, which would hold
            # however wrong the query was.
            assert len(expected) == LIMIT, f"{mode} did not fill the limit for {query!r}"
        assert [text for text, _ in actual] == [text for text, _ in expected], (
            f"{mode} ranking diverged for {query!r}"
        )
        for (_, got), (_, want) in zip(actual, expected, strict=True):
            # Identical, not merely close: the RRF threshold is calibrated to
            # one exact float, which config.py keeps to the ULP.
            assert got == want, f"{mode} score diverged for {query!r}"


def test_hybrid_rerank_queries_the_same_fused_ranking_as_hybrid(seeded: Services) -> None:
    """`hybrid_rerank` reorders hybrid's candidates; it must start from them."""
    for query in QUERIES:
        assert _reed_hits(seeded, query, "hybrid", limit=LIMIT) == _langchain_hits(
            seeded, query, "hybrid_rerank", limit=LIMIT
        )


def test_staged_points_are_never_returned(seeded: Services) -> None:
    """Guards the filter the equivalence assertions depend on."""
    staged = {
        TOPICS[index % len(TOPICS)].format(n=index) + f" (clause {index})"
        for index in range(CORPUS_SIZE - UNCOMMITTED, CORPUS_SIZE)
    }

    for mode in ("dense", "sparse", "hybrid"):
        returned = {text for text, _ in _reed_hits(seeded, "clause", mode, limit=CORPUS_SIZE)}
        assert returned
        assert not (returned & staged), f"{mode} returned staged points"
