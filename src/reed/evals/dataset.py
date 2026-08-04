"""The golden question set.

Ground truth is recorded at document level rather than chunk level. Chunk
boundaries move whenever the chunk size changes; "this question is answered by
the expenses policy" stays true.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

QuestionType = Literal["factual", "multi_hop", "negative"]

EVAL_DIR = Path(__file__).resolve().parents[3] / "eval"
CORPUS_DIR = EVAL_DIR / "corpus"
GOLDEN_PATH = EVAL_DIR / "golden.jsonl"
RESULTS_DIR = EVAL_DIR / "results"


@dataclass(frozen=True, slots=True)
class GoldenQuestion:
    id: str
    type: QuestionType
    question: str
    reference_answer: str
    expected_docs: list[str]

    @property
    def is_negative(self) -> bool:
        """True when the right behaviour is to refuse rather than answer."""
        return self.type == "negative" or not self.expected_docs


def load_golden(path: Path | None = None) -> list[GoldenQuestion]:
    lines = (path or GOLDEN_PATH).read_text(encoding="utf-8").splitlines()
    return [_parse(line) for line in lines if line.strip()]


def _parse(line: str) -> GoldenQuestion:
    row = json.loads(line)
    return GoldenQuestion(
        id=row["id"],
        type=row["type"],
        question=row["question"],
        reference_answer=row["reference_answer"],
        expected_docs=list(row.get("expected_docs", [])),
    )


def corpus_files(directory: Path | None = None) -> list[Path]:
    source = directory or CORPUS_DIR
    return sorted(p for p in source.glob("*.md") if p.name.lower() != "readme.md")
