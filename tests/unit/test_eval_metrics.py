from __future__ import annotations

import json
from pathlib import Path

import pytest

from reed.evals.dataset import GoldenQuestion, load_golden
from reed.evals.judge import JudgeScores, _normalise
from reed.evals.report import QuestionResult, build_report
from reed.evals.retrieval import RetrievalOutcome, aggregate, score
from reed.rag.retriever import RetrievedChunk


def question(**kwargs: object) -> GoldenQuestion:
    defaults: dict[str, object] = {
        "id": "q-001",
        "type": "factual",
        "question": "What is the expense threshold?",
        "reference_answer": "75 euros.",
        "expected_docs": ["04-expenses.md"],
    }
    return GoldenQuestion(**{**defaults, **kwargs})  # type: ignore[arg-type]


def chunk(filename: str, score_value: float = 0.9) -> RetrievedChunk:
    return RetrievedChunk(
        text="text",
        score=score_value,
        doc_id=f"d-{filename}",
        filename=filename,
        page=None,
        section=None,
    )


def outcome(expected: list[str], retrieved: list[str]) -> RetrievalOutcome:
    return RetrievalOutcome(
        question_id="q-001",
        question_type="factual",
        expected_docs=expected,
        retrieved_docs=retrieved,
    )


# -- scoring one question ------------------------------------------------


def test_documents_keep_the_rank_of_their_best_chunk() -> None:
    result = score(
        question(),
        [chunk("b.md"), chunk("a.md"), chunk("b.md"), chunk("c.md")],
    )

    assert result.retrieved_docs == ["b.md", "a.md", "c.md"]


def test_hit_at_1_needs_the_top_document() -> None:
    assert outcome(["a.md"], ["a.md", "b.md"]).hit_at_1 is True
    assert outcome(["a.md"], ["b.md", "a.md"]).hit_at_1 is False


def test_hit_at_k_accepts_any_position() -> None:
    assert outcome(["a.md"], ["b.md", "c.md", "a.md"]).hit_at_k is True
    assert outcome(["a.md"], ["b.md", "c.md"]).hit_at_k is False


def test_reciprocal_rank_follows_the_position() -> None:
    assert outcome(["a.md"], ["a.md"]).reciprocal_rank == 1.0
    assert outcome(["a.md"], ["x.md", "a.md"]).reciprocal_rank == 0.5
    assert outcome(["a.md"], ["x.md", "y.md", "a.md"]).reciprocal_rank == pytest.approx(1 / 3)
    assert outcome(["a.md"], ["x.md"]).reciprocal_rank == 0.0


def test_multi_hop_needs_every_expected_document() -> None:
    assert outcome(["a.md", "b.md"], ["a.md", "b.md"]).full_coverage is True
    # Half the evidence still counts as a hit, but not as full coverage.
    partial = outcome(["a.md", "b.md"], ["a.md", "z.md"])
    assert partial.hit_at_k is True
    assert partial.full_coverage is False


def test_empty_retrieval_scores_zero_without_crashing() -> None:
    empty = outcome(["a.md"], [])
    assert empty.hit_at_1 is False
    assert empty.reciprocal_rank == 0.0


# -- aggregation ---------------------------------------------------------


def test_negatives_are_excluded_from_retrieval_averages() -> None:
    metrics = aggregate(
        [
            outcome(["a.md"], ["a.md"]),
            outcome([], ["whatever.md"]),  # a negative question
        ]
    )

    assert metrics.questions == 1
    assert metrics.hit_at_1 == 1.0


def test_aggregate_of_nothing_is_zero_not_an_error() -> None:
    metrics = aggregate([outcome([], [])])
    assert metrics.questions == 0
    assert metrics.mrr == 0.0


def test_averages_are_computed_across_questions() -> None:
    metrics = aggregate(
        [
            outcome(["a.md"], ["a.md"]),
            outcome(["b.md"], ["x.md", "b.md"]),
        ]
    )

    assert metrics.hit_at_1 == 0.5
    assert metrics.hit_at_k == 1.0
    assert metrics.mrr == pytest.approx(0.75)


# -- judge helpers -------------------------------------------------------


def test_ratings_normalise_to_zero_one() -> None:
    assert _normalise(1) == 0.0
    assert _normalise(3) == 0.5
    assert _normalise(5) == 1.0
    # A model that ignores the rubric still produces a usable number.
    assert _normalise(9) == 1.0
    assert _normalise(0) == 0.0


# -- report --------------------------------------------------------------


def make_report(results: list[QuestionResult], skipped: str | None = None) -> object:
    return build_report(
        label="test",
        profile="fake",
        chat_model="fake-chat",
        embed_model="fake-embeddings",
        judge_model=None if skipped else "openai:gpt-5-mini",
        top_k=4,
        rerank=False,
        outcomes=[outcome(r.expected_docs, r.retrieved_docs) for r in results],
        results=results,
        skipped_reason=skipped,
    )


def result(**kwargs: object) -> QuestionResult:
    defaults: dict[str, object] = {
        "id": "q-001",
        "type": "factual",
        "question": "q?",
        "answer": "a [1].",
        "expected_docs": ["a.md"],
        "retrieved_docs": ["a.md"],
        "reciprocal_rank": 1.0,
        "latency_ms": 100,
        "judge": JudgeScores(faithfulness=1.0, correctness=0.75),
    }
    return QuestionResult(**{**defaults, **kwargs})  # type: ignore[arg-type]


def test_generation_averages_skip_missing_scores() -> None:
    report = make_report(
        [
            result(judge=JudgeScores(faithfulness=1.0)),
            result(judge=JudgeScores(faithfulness=0.5)),
            result(judge=JudgeScores()),  # judge failed on this one
        ]
    )

    assert report.generation["faithfulness"] == pytest.approx(0.75)  # type: ignore[attr-defined]
    assert report.generation["correctness"] is None  # type: ignore[attr-defined]


def test_markdown_says_why_generation_was_skipped() -> None:
    report = make_report([result()], skipped="--retrieval-only")
    markdown = report.to_markdown()  # type: ignore[attr-defined]

    assert "Skipped: --retrieval-only" in markdown
    assert "## Retrieval" in markdown


def test_markdown_lists_the_weakest_retrievals() -> None:
    report = make_report(
        [
            result(id="q-good"),
            result(id="q-bad", retrieved_docs=["z.md"], reciprocal_rank=0.0),
        ]
    )
    markdown = report.to_markdown()  # type: ignore[attr-defined]

    assert "q-bad" in markdown
    assert "q-good" not in markdown.split("Weakest retrievals")[1]


def test_report_json_round_trips(tmp_path: Path) -> None:
    report = make_report([result()])
    markdown, raw = report.write(tmp_path)  # type: ignore[attr-defined]

    assert markdown.exists()
    parsed = json.loads(raw.read_text(encoding="utf-8"))
    assert parsed["retrieval"]["hit_at_1"] == 1.0
    assert parsed["results"][0]["id"] == "q-001"


# -- dataset -------------------------------------------------------------


def test_golden_rows_parse(tmp_path: Path) -> None:
    path = tmp_path / "golden.jsonl"
    path.write_text(
        '{"id":"q-001","type":"factual","question":"q","reference_answer":"r",'
        '"expected_docs":["a.md"]}\n'
        "\n"
        '{"id":"q-002","type":"negative","question":"q","reference_answer":"r",'
        '"expected_docs":[]}\n',
        encoding="utf-8",
    )

    rows = load_golden(path)

    assert [row.id for row in rows] == ["q-001", "q-002"]
    assert rows[0].is_negative is False
    assert rows[1].is_negative is True
