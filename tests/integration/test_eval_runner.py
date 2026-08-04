"""The evaluation runner, end to end over a miniature corpus.

Uses its own two-document corpus rather than the real one: this asserts that
the machinery works, not that any particular model scores well.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reed.config import Settings
from reed.evals.runner import run_evaluation

pytestmark = pytest.mark.integration

EXPENSES = """# Expenses

## Pre-approval

Expenses above 75 euros require pre-approval from your manager.
"""

LEAVE = """# Time off

## Notice

Holiday requests must be submitted at least 14 days in advance.
"""

GOLDEN = [
    {
        "id": "q-001",
        "type": "factual",
        "question": "How much can I spend before I need my manager to sign off?",
        "reference_answer": "Anything above 75 euros needs pre-approval.",
        "expected_docs": ["expenses.md"],
    },
    {
        "id": "q-002",
        "type": "factual",
        "question": "How far ahead do I book holiday?",
        "reference_answer": "At least 14 days in advance.",
        "expected_docs": ["leave.md"],
    },
    {
        "id": "q-003",
        "type": "negative",
        "question": "What is the company pension provider?",
        "reference_answer": "The handbook does not cover pensions.",
        "expected_docs": [],
    },
]


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    directory = tmp_path / "corpus"
    directory.mkdir()
    (directory / "expenses.md").write_text(EXPENSES, encoding="utf-8")
    (directory / "leave.md").write_text(LEAVE, encoding="utf-8")
    (directory / "README.md").write_text("not part of the corpus", encoding="utf-8")
    return directory


@pytest.fixture
def golden(tmp_path: Path) -> Path:
    path = tmp_path / "golden.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in GOLDEN) + "\n", encoding="utf-8")
    return path


def run(settings: Settings, corpus: Path, golden: Path, results: Path, **kwargs: object):  # type: ignore[no-untyped-def]
    return run_evaluation(
        retrieval_only=True,
        corpus_dir=corpus,
        golden_path=golden,
        results_dir=results,
        settings=settings,
        **kwargs,  # type: ignore[arg-type]
    )


def test_retrieval_only_run_scores_every_question(
    settings: Settings, corpus: Path, golden: Path, tmp_path: Path
) -> None:
    report = run(settings, corpus, golden, tmp_path / "results")

    assert len(report.results) == 3
    # Negatives carry no expected document, so they sit outside the averages.
    assert report.retrieval.questions == 2
    assert report.skipped_reason == "--retrieval-only"
    assert report.judge_model is None


def test_a_readme_file_is_not_treated_as_a_corpus_document(
    settings: Settings, corpus: Path, golden: Path, tmp_path: Path
) -> None:
    report = run(settings, corpus, golden, tmp_path / "results")

    retrieved = {doc for result in report.results for doc in result.retrieved_docs}
    assert "README.md" not in retrieved


def test_run_leaves_no_data_behind(
    settings: Settings, corpus: Path, golden: Path, tmp_path: Path
) -> None:
    run(settings, corpus, golden, tmp_path / "results")

    # The evaluation gets a temporary Qdrant; the configured data directory
    # must be untouched.
    assert not settings.qdrant_path.exists()


def test_report_is_written_in_both_formats(
    settings: Settings, corpus: Path, golden: Path, tmp_path: Path
) -> None:
    report = run(settings, corpus, golden, tmp_path / "results")
    markdown, raw = report.write(tmp_path / "results")

    assert "## Retrieval" in markdown.read_text(encoding="utf-8")
    assert json.loads(raw.read_text(encoding="utf-8"))["top_k"] == settings.top_k


def test_label_defaults_to_the_configuration(
    settings: Settings, corpus: Path, golden: Path, tmp_path: Path
) -> None:
    report = run(settings, corpus, golden, tmp_path / "results")
    assert report.label == f"fake, k={settings.top_k}"

    named = run(settings, corpus, golden, tmp_path / "results", label="custom run")
    assert named.label == "custom run"


def test_top_k_override_reaches_retrieval(
    settings: Settings, corpus: Path, golden: Path, tmp_path: Path
) -> None:
    report = run(settings, corpus, golden, tmp_path / "results", top_k=1)

    assert report.top_k == 1
    assert all(len(result.retrieved_docs) <= 1 for result in report.results)


def test_an_empty_corpus_fails_loudly(settings: Settings, golden: Path, tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()

    with pytest.raises(FileNotFoundError, match="No corpus documents"):
        run(settings, empty, golden, tmp_path / "results")


def test_a_dead_chat_provider_is_reported_not_scored(
    settings: Settings,
    corpus: Path,
    golden: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DeadModel:
        async def astream(self, *_: object, **__: object):  # type: ignore[no-untyped-def]
            raise RuntimeError("model 'nope' not found")
            yield  # pragma: no cover — makes this an async generator

    monkeypatch.setattr("reed.providers.build_chat_model", lambda *_, **__: DeadModel())

    report = run(settings, corpus, golden, tmp_path / "results")

    # Otherwise a broken provider reads as "the model answers badly".
    assert len(report.failures) == len(report.results)
    assert "not found" in report.failures[0].error
    assert "failed outright" in report.to_markdown()
