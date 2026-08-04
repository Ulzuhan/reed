"""Packaged evaluation corpus with an optional checkout-local override."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

QuestionType = Literal[
    "factual",
    "multi_hop",
    "negative",
    "cross_language",
    "conversation",
    "adversarial",
]


@dataclass(frozen=True, slots=True)
class EvidenceLabel:
    id: str
    document: str
    text: str


def _default_eval_dir() -> Path:
    """Prefer checkout data, falling back to files bundled inside the wheel.

    This keeps local corpus editing convenient without making ``reed eval``
    depend on the caller's working directory after installation.
    """
    from_cwd = Path.cwd() / "eval"
    if (from_cwd / "golden.jsonl").exists():
        return from_cwd
    return Path(__file__).resolve().parent / "data"


EVAL_DIR = _default_eval_dir()
CORPUS_DIR = EVAL_DIR / "corpus"
GOLDEN_PATH = EVAL_DIR / "golden.jsonl"
RESULTS_DIR = EVAL_DIR / "results" if EVAL_DIR.name == "eval" else Path.cwd() / "reed-eval-results"


@dataclass(frozen=True, slots=True)
class GoldenQuestion:
    id: str
    type: QuestionType
    question: str
    reference_answer: str
    expected_docs: list[str]
    evidence: list[EvidenceLabel] = field(default_factory=list)
    language: str = "en"
    tags: list[str] = field(default_factory=list)

    @property
    def is_negative(self) -> bool:
        """True when the right behaviour is to refuse rather than answer."""
        return self.type == "negative" or not self.expected_docs


def load_golden(path: Path | None = None) -> list[GoldenQuestion]:
    selected = path or GOLDEN_PATH
    lines = selected.read_text(encoding="utf-8").splitlines()
    evidence_path = selected.with_name("evidence.json")
    evidence_by_question: dict[str, list[dict[str, str]]] = {}
    if evidence_path.is_file():
        loaded = json.loads(evidence_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            evidence_by_question = loaded
    return [_parse(line, evidence_by_question) for line in lines if line.strip()]


def _parse(line: str, evidence_by_question: dict[str, list[dict[str, str]]]) -> GoldenQuestion:
    row = json.loads(line)
    evidence_rows = row.get("evidence", evidence_by_question.get(str(row["id"]), []))
    return GoldenQuestion(
        id=row["id"],
        type=row["type"],
        question=row["question"],
        reference_answer=row["reference_answer"],
        expected_docs=list(row.get("expected_docs", [])),
        evidence=[
            EvidenceLabel(
                id=str(item["id"]),
                document=str(item["document"]),
                text=str(item["text"]),
            )
            for item in evidence_rows
        ],
        language=str(row.get("language", "en")),
        tags=[str(tag) for tag in row.get("tags", [])],
    )


def corpus_files(directory: Path | None = None) -> list[Path]:
    source = directory or CORPUS_DIR
    return sorted(p for p in source.glob("*.md") if p.name.lower() != "readme.md")
