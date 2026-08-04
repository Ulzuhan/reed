"""Retrieval: hybrid search, optionally reranked.

Dense vectors catch paraphrases ("how much can I spend before asking?"), sparse
BM25 catches exact terminology ("pre-approval threshold"). Qdrant runs both and
fuses the rankings with RRF server-side, so neither phrasing style loses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from reed.log import get_logger

if TYPE_CHECKING:
    from reed.services import Services

logger = get_logger(__name__)

SNIPPET_CHARS = 220


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    text: str
    score: float
    doc_id: str
    filename: str
    page: int | None
    section: str | None

    @property
    def snippet(self) -> str:
        flat = " ".join(self.text.split())
        return flat if len(flat) <= SNIPPET_CHARS else f"{flat[:SNIPPET_CHARS].rstrip()}…"

    @property
    def location(self) -> str:
        """Human-readable citation label, e.g. ``handbook.pdf, p. 3``."""
        if self.page is not None:
            return f"{self.filename}, p. {self.page}"
        if self.section:
            return f"{self.filename} — {self.section}"
        return self.filename


def retrieve(services: Services, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
    settings = services.settings
    k = top_k or settings.top_k
    fetch_k = max(settings.fetch_k, k) if settings.rerank_enabled else k

    hits = services.vectorstore.similarity_search_with_score(query, k=fetch_k)
    chunks = [_to_chunk(document, score) for document, score in hits]

    if settings.rerank_enabled and chunks:
        chunks = rerank(services, query, chunks)

    return chunks[:k]


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
        )
        for chunk, score in zip(chunks, scores, strict=True)
    ]
    rescored.sort(key=lambda c: c.score, reverse=True)
    return rescored


def _to_chunk(document: object, score: float) -> RetrievedChunk:
    metadata: dict[str, object] = getattr(document, "metadata", {}) or {}
    page = metadata.get("page")
    return RetrievedChunk(
        text=str(getattr(document, "page_content", "")),
        score=float(score),
        doc_id=str(metadata.get("doc_id", "")),
        filename=str(metadata.get("filename", "unknown")),
        page=int(page) if isinstance(page, int) else None,
        section=str(metadata["section"]) if metadata.get("section") else None,
    )
