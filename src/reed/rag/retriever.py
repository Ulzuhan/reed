"""Retrieval: hybrid search, optionally reranked.

Dense vectors catch paraphrases ("how much can I spend before asking?"), sparse
BM25 catches exact terminology ("pre-approval threshold"). Qdrant runs both and
fuses the rankings with RRF server-side, so neither phrasing style loses.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from qdrant_client import models

from reed.log import get_logger
from reed.rag.vectorstore import COMMITTED_PAYLOAD_KEY

if TYPE_CHECKING:
    from reed.services import Services

logger = get_logger(__name__)

SNIPPET_CHARS = 220

# Heading markers, emphasis and list bullets — noise in a one-line preview.
_MARKDOWN_SYNTAX = re.compile(r"^\s{0,3}#{1,6}\s+|\*{1,3}|`+|^\s*[-*+]\s+", re.MULTILINE)


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    text: str
    score: float
    doc_id: str
    filename: str
    page: int | None
    section: str | None
    chunk_index: int | None = None

    @property
    def snippet(self) -> str:
        """A preview for humans — the model still receives the raw text."""
        flat = " ".join(_MARKDOWN_SYNTAX.sub("", self.text).split())
        return flat if len(flat) <= SNIPPET_CHARS else f"{flat[:SNIPPET_CHARS].rstrip()}…"

    @property
    def location(self) -> str:
        """Human-readable citation label, e.g. ``handbook.pdf, p. 3``."""
        if self.page is not None:
            return f"{self.filename}, p. {self.page}"
        if self.section:
            return f"{self.filename} — {self.section}"
        return self.filename


def retrieve(
    services: Services,
    query: str,
    top_k: int | None = None,
    *,
    mode: str | None = None,
    apply_threshold: bool = True,
) -> list[RetrievedChunk]:
    started = time.perf_counter()
    settings = services.settings
    k = top_k or settings.top_k
    selected_mode = mode or settings.retrieval_mode
    use_reranker = settings.rerank_enabled or selected_mode == "hybrid_rerank"
    needs_candidates = use_reranker or settings.diversity_enabled
    fetch_k = max(settings.fetch_k, k) if needs_candidates else k

    services.flush_pending_vector_cleanup()
    store = services.retrieval_store(selected_mode)
    with services.vector_access:
        hits = store.similarity_search_with_score(
            query,
            k=fetch_k,
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key=COMMITTED_PAYLOAD_KEY,
                        match=models.MatchValue(value=True),
                    )
                ]
            ),
        )
    chunks = [_to_chunk(document, score) for document, score in hits]
    # Qdrant publication and SQLite status cannot share a transaction. There is
    # therefore a tiny interval after a point batch is published but before its
    # registry row becomes ready (and the inverse during deletion). Requiring
    # both stores to agree closes that interval and also hides orphaned points.
    ready_ids = services.registry.ready_ids([chunk.doc_id for chunk in chunks])
    chunks = [chunk for chunk in chunks if chunk.doc_id in ready_ids]

    if use_reranker and chunks:
        chunks = rerank(services, query, chunks)

    if settings.diversity_enabled and chunks:
        chunks = diversify(
            chunks,
            k=k,
            lambda_mult=settings.diversity_lambda,
            max_per_document=settings.max_chunks_per_document,
        )
    else:
        chunks = _limit_per_document(chunks, settings.max_chunks_per_document)

    threshold = settings.resolved_min_evidence_score
    if apply_threshold and threshold and (not chunks or chunks[0].score < threshold):
        logger.info(
            "retrieval abstained: top score %.4f is below %.4f",
            chunks[0].score if chunks else 0.0,
            threshold,
        )
        services.metrics.increment("abstentions_total")
        services.metrics.observe("retrieval_latency_seconds", time.perf_counter() - started)
        return []

    selected = chunks[:k]
    services.metrics.observe("retrieval_latency_seconds", time.perf_counter() - started)
    return selected


def rerank(services: Services, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Reorder candidates with a cross-encoder.

    Hybrid search scores a query and a chunk separately; a cross-encoder reads
    them together, which is slower but much better at spotting a chunk that
    merely shares vocabulary with the question.
    """
    scores = services.reranker.rerank(query, [chunk.text for chunk in chunks])
    rescored = [
        RetrievedChunk(
            text=chunk.text,
            score=float(score),
            doc_id=chunk.doc_id,
            filename=chunk.filename,
            page=chunk.page,
            section=chunk.section,
            chunk_index=chunk.chunk_index,
        )
        for chunk, score in zip(chunks, scores, strict=True)
    ]
    rescored.sort(key=lambda c: c.score, reverse=True)
    return rescored


def diversify(
    chunks: list[RetrievedChunk],
    *,
    k: int,
    lambda_mult: float,
    max_per_document: int,
) -> list[RetrievedChunk]:
    """Greedy MMR over ranked relevance and lexical chunk similarity."""
    if not chunks:
        return []
    high = max(chunk.score for chunk in chunks)
    low = min(chunk.score for chunk in chunks)
    span = high - low
    relevance = {id(chunk): (chunk.score - low) / span if span else 1.0 for chunk in chunks}
    tokens = {id(chunk): _tokens(chunk.text) for chunk in chunks}
    remaining = list(chunks)
    selected: list[RetrievedChunk] = []
    per_document: dict[str, int] = {}
    while remaining and len(selected) < k:
        eligible = [
            chunk for chunk in remaining if per_document.get(chunk.doc_id, 0) < max_per_document
        ]
        if not eligible:
            break

        def mmr(chunk: RetrievedChunk) -> float:
            redundancy = max(
                (_jaccard(tokens[id(chunk)], tokens[id(chosen)]) for chosen in selected),
                default=0.0,
            )
            return lambda_mult * relevance[id(chunk)] - (1 - lambda_mult) * redundancy

        winner = max(eligible, key=lambda chunk: (mmr(chunk), chunk.score))
        selected.append(winner)
        per_document[winner.doc_id] = per_document.get(winner.doc_id, 0) + 1
        remaining.remove(winner)
    return selected


def _limit_per_document(
    chunks: list[RetrievedChunk], max_per_document: int
) -> list[RetrievedChunk]:
    selected: list[RetrievedChunk] = []
    counts: dict[str, int] = {}
    for chunk in chunks:
        if counts.get(chunk.doc_id, 0) >= max_per_document:
            continue
        selected.append(chunk)
        counts[chunk.doc_id] = counts.get(chunk.doc_id, 0) + 1
    return selected


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[^\W_]+", text.casefold(), flags=re.UNICODE))


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _to_chunk(document: object, score: float) -> RetrievedChunk:
    metadata: dict[str, object] = getattr(document, "metadata", {}) or {}
    page = metadata.get("page")
    chunk_index = metadata.get("chunk_index")
    return RetrievedChunk(
        text=str(getattr(document, "page_content", "")),
        score=float(score),
        doc_id=str(metadata.get("doc_id", "")),
        filename=str(metadata.get("filename", "unknown")),
        page=int(page) if isinstance(page, int) and not isinstance(page, bool) else None,
        section=str(metadata["section"]) if metadata.get("section") else None,
        chunk_index=(
            int(chunk_index)
            if isinstance(chunk_index, int) and not isinstance(chunk_index, bool)
            else None
        ),
    )
