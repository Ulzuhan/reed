"""Retrieval metrics — plain arithmetic, no model involved.

These run anywhere: offline, in CI, on a laptop with no API key. If retrieval
is broken, no amount of prompt work will save the answer, so this is the first
thing to measure and the cheapest thing to keep measuring.
"""

from __future__ import annotations

from dataclasses import dataclass

from reed.evals.dataset import GoldenQuestion
from reed.rag.retriever import RetrievedChunk


@dataclass(frozen=True, slots=True)
class RetrievalOutcome:
    question_id: str
    question_type: str
    expected_docs: list[str]
    retrieved_docs: list[str]

    @property
    def hit_at_1(self) -> bool:
        return bool(self.retrieved_docs[:1]) and self.retrieved_docs[0] in self.expected_docs

    @property
    def hit_at_k(self) -> bool:
        return any(doc in self.expected_docs for doc in self.retrieved_docs)

    @property
    def reciprocal_rank(self) -> float:
        """1/rank of the first expected document, 0 if none was retrieved."""
        for rank, doc in enumerate(self.retrieved_docs, start=1):
            if doc in self.expected_docs:
                return 1.0 / rank
        return 0.0

    @property
    def full_coverage(self) -> bool:
        """Every expected document present — what multi-hop questions need."""
        return all(doc in self.retrieved_docs for doc in self.expected_docs)


@dataclass(frozen=True, slots=True)
class RetrievalMetrics:
    questions: int
    hit_at_1: float
    hit_at_k: float
    mrr: float
    full_coverage: float


def score(question: GoldenQuestion, chunks: list[RetrievedChunk]) -> RetrievalOutcome:
    """Rank retrieved documents by their best-placed chunk, keeping order."""
    seen: list[str] = []
    for chunk in chunks:
        if chunk.filename not in seen:
            seen.append(chunk.filename)
    return RetrievalOutcome(
        question_id=question.id,
        question_type=question.type,
        expected_docs=list(question.expected_docs),
        retrieved_docs=seen,
    )


def aggregate(outcomes: list[RetrievalOutcome]) -> RetrievalMetrics:
    """Average over answerable questions only.

    Negative questions have no expected document, so scoring their retrieval
    would just dilute the numbers. Whether the model correctly refuses them is
    measured by the judge instead.
    """
    scored = [o for o in outcomes if o.expected_docs]
    if not scored:
        return RetrievalMetrics(questions=0, hit_at_1=0.0, hit_at_k=0.0, mrr=0.0, full_coverage=0.0)

    total = len(scored)
    return RetrievalMetrics(
        questions=total,
        hit_at_1=sum(o.hit_at_1 for o in scored) / total,
        hit_at_k=sum(o.hit_at_k for o in scored) / total,
        mrr=sum(o.reciprocal_rank for o in scored) / total,
        full_coverage=sum(o.full_coverage for o in scored) / total,
    )
