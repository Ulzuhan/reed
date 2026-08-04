"""Turning a run into something you can paste into a README."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from reed.evals.judge import JudgeScores
from reed.evals.retrieval import RetrievalMetrics, RetrievalOutcome

WORST_CASES_SHOWN = 3


@dataclass
class QuestionResult:
    id: str
    type: str
    question: str
    answer: str
    expected_docs: list[str]
    retrieved_docs: list[str]
    reciprocal_rank: float
    latency_ms: int
    judge: JudgeScores = field(default_factory=JudgeScores)
    error: str | None = None


@dataclass
class Report:
    label: str
    profile: str
    chat_model: str
    embed_model: str
    judge_model: str | None
    top_k: int
    rerank: bool
    retrieval: RetrievalMetrics
    results: list[QuestionResult]
    skipped_reason: str | None = None

    @property
    def generation(self) -> dict[str, float | None]:
        """Judged metrics, averaged over the questions that produced a score."""
        return {
            name: _mean(getattr(result.judge, name) for result in self.results)
            for name in (
                "faithfulness",
                "answer_relevancy",
                "correctness",
                "context_precision",
                "context_recall",
                "correct_refusal",
            )
        }

    @property
    def median_latency_ms(self) -> int:
        latencies = sorted(result.latency_ms for result in self.results)
        return latencies[len(latencies) // 2] if latencies else 0

    @property
    def failures(self) -> list[QuestionResult]:
        """Questions the provider could not answer at all."""
        return [result for result in self.results if result.error]

    def to_markdown(self) -> str:
        lines = [
            f"# Evaluation — {self.label}",
            "",
            f"- Profile: `{self.profile}` · chat `{self.chat_model}` "
            f"· embeddings `{self.embed_model}`",
            f"- Retrieval: hybrid (dense + BM25), top_k={self.top_k}, "
            f"rerank={'on' if self.rerank else 'off'}",
            f"- Judge: {f'`{self.judge_model}`' if self.judge_model else 'not run'}",
            f"- Questions: {len(self.results)} · median latency {self.median_latency_ms} ms",
            "",
            "## Retrieval",
            "",
            "| hit@1 | hit@k | MRR | all expected docs |",
            "|---|---|---|---|",
            f"| {_pct(self.retrieval.hit_at_1)} | {_pct(self.retrieval.hit_at_k)} "
            f"| {self.retrieval.mrr:.3f} | {_pct(self.retrieval.full_coverage)} |",
            "",
        ]

        if self.skipped_reason:
            lines += ["## Generation", "", f"_Skipped: {self.skipped_reason}_", ""]
        else:
            generation = self.generation
            lines += [
                "## Generation",
                "",
                "| faithfulness | answer relevancy | correctness | context precision "
                "| context recall | correct refusals |",
                "|---|---|---|---|---|---|",
                "| "
                + " | ".join(
                    _pct(generation[name])
                    for name in (
                        "faithfulness",
                        "answer_relevancy",
                        "correctness",
                        "context_precision",
                        "context_recall",
                        "correct_refusal",
                    )
                )
                + " |",
                "",
            ]

        if self.failures:
            lines += [
                "## Unanswered",
                "",
                f"**{len(self.failures)} of {len(self.results)} questions failed outright** — "
                "these are provider errors, not quality scores.",
                "",
            ]
            lines += [f"- `{result.id}`: {result.error}" for result in self.failures]
            lines += [""]

        worst = self._worst_retrievals()
        if worst:
            lines += ["## Weakest retrievals", ""]
            for result in worst:
                lines += [
                    f"**{result.id}** ({result.type}) — {result.question}",
                    f"- expected: {', '.join(result.expected_docs) or '—'}",
                    f"- retrieved: {', '.join(result.retrieved_docs) or '—'}",
                    "",
                ]

        return "\n".join(lines)

    def summary_row(self) -> str:
        """One README table row, for comparing configurations side by side."""
        generation = self.generation
        cells = [
            self.label,
            _pct(self.retrieval.hit_at_1),
            _pct(self.retrieval.hit_at_k),
            f"{self.retrieval.mrr:.3f}",
            _pct(generation["faithfulness"]),
            _pct(generation["correctness"]),
            _pct(generation["context_precision"]),
            _pct(generation["context_recall"]),
        ]
        return "| " + " | ".join(cells) + " |"

    def to_json(self) -> str:
        return json.dumps(
            {
                "label": self.label,
                "profile": self.profile,
                "chat_model": self.chat_model,
                "embed_model": self.embed_model,
                "judge_model": self.judge_model,
                "top_k": self.top_k,
                "rerank": self.rerank,
                "retrieval": asdict(self.retrieval),
                "generation": self.generation,
                "median_latency_ms": self.median_latency_ms,
                "results": [asdict(result) for result in self.results],
            },
            indent=2,
            ensure_ascii=False,
        )

    def write(self, directory: Path) -> tuple[Path, Path]:
        directory.mkdir(parents=True, exist_ok=True)
        markdown = directory / "report.md"
        raw = directory / "report.json"
        markdown.write_text(self.to_markdown() + "\n", encoding="utf-8")
        raw.write_text(self.to_json() + "\n", encoding="utf-8")
        return markdown, raw

    def _worst_retrievals(self) -> list[QuestionResult]:
        misses = [
            result
            for result in self.results
            if result.expected_docs and result.reciprocal_rank < 1.0
        ]
        misses.sort(key=lambda result: result.reciprocal_rank)
        return misses[:WORST_CASES_SHOWN]


SUMMARY_HEADER = (
    "| Configuration | hit@1 | hit@k | MRR | Faithfulness | Correctness "
    "| Ctx precision | Ctx recall |\n|---|---|---|---|---|---|---|---|"
)


def build_report(
    *,
    label: str,
    profile: str,
    chat_model: str,
    embed_model: str,
    judge_model: str | None,
    top_k: int,
    rerank: bool,
    outcomes: list[RetrievalOutcome],
    results: list[QuestionResult],
    skipped_reason: str | None,
) -> Report:
    from reed.evals.retrieval import aggregate

    return Report(
        label=label,
        profile=profile,
        chat_model=chat_model,
        embed_model=embed_model,
        judge_model=judge_model,
        top_k=top_k,
        rerank=rerank,
        retrieval=aggregate(outcomes),
        results=results,
        skipped_reason=skipped_reason,
    )


def _mean(values: Iterable[float | None]) -> float | None:
    numbers = [value for value in values if value is not None]
    return sum(numbers) / len(numbers) if numbers else None


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.0f}%"
