"""Evidence-level retrieval metrics with deterministic bootstrap intervals."""

from __future__ import annotations

import math
import random
import re
from collections.abc import Iterable
from dataclasses import dataclass, field, replace

from reed.evals.dataset import EvidenceLabel, GoldenQuestion
from reed.rag.retriever import RetrievedChunk

_SPACE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class RetrievalOutcome:
    question_id: str
    question_type: str
    expected_docs: list[str]
    retrieved_docs: list[str]
    expected_evidence_ids: list[str] = field(default_factory=list)
    covered_evidence_ids: list[str] = field(default_factory=list)
    gains: list[int] = field(default_factory=list)
    abstained: bool = False
    top_score: float | None = None

    @property
    def hit_at_1(self) -> bool:
        if self.gains:
            return self.gains[0] > 0
        return bool(self.retrieved_docs[:1]) and self.retrieved_docs[0] in self.expected_docs

    @property
    def hit_at_k(self) -> bool:
        if self.expected_evidence_ids:
            return bool(self.covered_evidence_ids)
        return any(doc in self.expected_docs for doc in self.retrieved_docs)

    @property
    def reciprocal_rank(self) -> float:
        """1/rank of the first evidence-bearing chunk, 0 if none was retrieved."""
        if self.gains:
            for rank, gain in enumerate(self.gains, start=1):
                if gain:
                    return 1.0 / rank
            return 0.0
        for rank, doc in enumerate(self.retrieved_docs, start=1):
            if doc in self.expected_docs:
                return 1.0 / rank
        return 0.0

    @property
    def recall_at_k(self) -> float:
        if self.expected_evidence_ids:
            return len(set(self.covered_evidence_ids)) / len(set(self.expected_evidence_ids))
        if not self.expected_docs:
            return 0.0
        return len(set(self.expected_docs).intersection(self.retrieved_docs)) / len(
            set(self.expected_docs)
        )

    @property
    def ndcg_at_k(self) -> float:
        labels = len(set(self.expected_evidence_ids or self.expected_docs))
        if labels == 0:
            return 0.0
        gains = self.gains or [int(doc in self.expected_docs) for doc in self.retrieved_docs]
        dcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, start=1))
        ideal_count = min(labels, len(gains))
        ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
        return min(dcg / ideal, 1.0) if ideal else 0.0

    @property
    def full_coverage(self) -> bool:
        if self.expected_evidence_ids:
            return set(self.expected_evidence_ids) <= set(self.covered_evidence_ids)
        return all(doc in self.retrieved_docs for doc in self.expected_docs)


@dataclass(frozen=True, slots=True)
class RetrievalMetrics:
    questions: int
    hit_at_1: float
    hit_at_k: float
    mrr: float
    full_coverage: float
    recall_at_k: float = 0.0
    ndcg_at_k: float = 0.0
    negative_questions: int = 0
    negative_abstention: float = 0.0
    abstention_accuracy: float = 0.0
    confidence_intervals: dict[str, tuple[float, float]] = field(default_factory=dict)
    recommended_min_score: float | None = None
    calibration_balanced_accuracy: float | None = None


def score(
    question: GoldenQuestion, chunks: list[RetrievedChunk], *, abstained: bool = False
) -> RetrievalOutcome:
    """Score exact evidence coverage while retaining document-level compatibility."""
    seen_docs: list[str] = []
    covered: list[str] = []
    gains: list[int] = []
    labels = question.evidence or _document_labels(question.expected_docs)
    for chunk in chunks:
        if chunk.filename not in seen_docs:
            seen_docs.append(chunk.filename)
        newly_covered = [
            label.id for label in labels if label.id not in covered and _matches(label, chunk)
        ]
        covered.extend(newly_covered)
        # Binary gain prevents a long chunk containing multiple labels from
        # overpowering the ideal DCG while recall still records every label.
        gains.append(int(bool(newly_covered)))
    return RetrievalOutcome(
        question_id=question.id,
        question_type=question.type,
        expected_docs=list(question.expected_docs),
        retrieved_docs=seen_docs,
        expected_evidence_ids=[label.id for label in labels],
        covered_evidence_ids=covered,
        gains=gains,
        abstained=abstained,
        top_score=chunks[0].score if chunks else None,
    )


def matched_evidence(question: GoldenQuestion, chunk: RetrievedChunk) -> list[str]:
    labels = question.evidence or _document_labels(question.expected_docs)
    return [label.id for label in labels if _matches(label, chunk)]


def aggregate(
    outcomes: list[RetrievalOutcome], *, bootstrap_samples: int = 1_000
) -> RetrievalMetrics:
    metrics = _aggregate_without_ci(outcomes)
    threshold, accuracy = _calibrate_threshold(outcomes)
    metrics = replace(
        metrics,
        recommended_min_score=threshold,
        calibration_balanced_accuracy=accuracy,
    )
    if not outcomes or bootstrap_samples <= 0:
        return metrics
    rng = random.Random(0)
    samples: dict[str, list[float]] = {
        "recall_at_k": [],
        "ndcg_at_k": [],
        "mrr": [],
        "full_coverage": [],
        "abstention_accuracy": [],
    }
    for _ in range(bootstrap_samples):
        draw = [outcomes[rng.randrange(len(outcomes))] for _ in outcomes]
        estimate = _aggregate_without_ci(draw)
        for name in samples:
            samples[name].append(float(getattr(estimate, name)))
    intervals = {name: _percentile_interval(values) for name, values in samples.items()}
    return replace(metrics, confidence_intervals=intervals)


def _aggregate_without_ci(outcomes: list[RetrievalOutcome]) -> RetrievalMetrics:
    scored = [outcome for outcome in outcomes if outcome.expected_docs]
    negatives = [outcome for outcome in outcomes if not outcome.expected_docs]
    total = len(scored)
    all_count = len(outcomes)
    return RetrievalMetrics(
        questions=total,
        hit_at_1=_mean(float(outcome.hit_at_1) for outcome in scored),
        hit_at_k=_mean(float(outcome.hit_at_k) for outcome in scored),
        mrr=_mean(outcome.reciprocal_rank for outcome in scored),
        full_coverage=_mean(float(outcome.full_coverage) for outcome in scored),
        recall_at_k=_mean(outcome.recall_at_k for outcome in scored),
        ndcg_at_k=_mean(outcome.ndcg_at_k for outcome in scored),
        negative_questions=len(negatives),
        negative_abstention=_mean(float(outcome.abstained) for outcome in negatives),
        abstention_accuracy=(
            sum(not outcome.abstained for outcome in scored)
            + sum(outcome.abstained for outcome in negatives)
        )
        / all_count
        if all_count
        else 0.0,
    )


def _matches(label: EvidenceLabel, chunk: RetrievedChunk) -> bool:
    if label.document != chunk.filename:
        return False
    if not label.text:
        return True
    return _normalise(label.text) in _normalise(chunk.text)


def _document_labels(documents: list[str]) -> list[EvidenceLabel]:
    return [EvidenceLabel(id=f"doc:{name}", document=name, text="") for name in documents]


def _normalise(value: str) -> str:
    without_markdown = re.sub(r"[`*_#]", "", value)
    return _SPACE.sub(" ", without_markdown).strip().casefold()


def _mean(values: Iterable[float]) -> float:
    numbers = list(values)
    return sum(numbers) / len(numbers) if numbers else 0.0


def _percentile_interval(values: list[float]) -> tuple[float, float]:
    ordered = sorted(values)
    low = ordered[int(0.025 * (len(ordered) - 1))]
    high = ordered[int(0.975 * (len(ordered) - 1))]
    return low, high


def _calibrate_threshold(outcomes: list[RetrievalOutcome]) -> tuple[float | None, float | None]:
    answerable = [outcome for outcome in outcomes if outcome.expected_docs]
    negatives = [outcome for outcome in outcomes if not outcome.expected_docs]
    scores = sorted({outcome.top_score for outcome in outcomes if outcome.top_score is not None})
    if not answerable or not negatives or not scores:
        return None, None
    candidates = [0.0, *scores, min(1.0, scores[-1] + 1e-9)]
    best_threshold = 0.0
    best_accuracy = -1.0
    for threshold in candidates:
        true_positive = _mean(
            float(outcome.top_score is not None and outcome.top_score >= threshold)
            for outcome in answerable
        )
        true_negative = _mean(
            float(outcome.top_score is None or outcome.top_score < threshold)
            for outcome in negatives
        )
        balanced = (true_positive + true_negative) / 2
        if balanced > best_accuracy:
            best_threshold = threshold
            best_accuracy = balanced
    return best_threshold, best_accuracy
