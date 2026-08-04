"""Guards on the shipped evaluation dataset.

A golden set that quietly references a renamed document, or whose negative
questions are actually answerable, produces metrics that look fine and mean
nothing. These checks run in CI so that cannot happen unnoticed.
"""

from __future__ import annotations

import re
from collections import Counter

import pytest

from reed.evals.dataset import CORPUS_DIR, GoldenQuestion, corpus_files, load_golden

EXPECTED_QUESTIONS = 41


@pytest.fixture(scope="module")
def golden() -> list[GoldenQuestion]:
    return load_golden()


@pytest.fixture(scope="module")
def corpus_names() -> set[str]:
    return {path.name for path in corpus_files()}


def test_the_corpus_is_present(corpus_names: set[str]) -> None:
    assert len(corpus_names) == 10
    assert all(name.endswith(".md") for name in corpus_names)


def test_question_ids_are_unique(golden: list[GoldenQuestion]) -> None:
    duplicates = [id_ for id_, count in Counter(q.id for q in golden).items() if count > 1]
    assert not duplicates, f"duplicate ids: {duplicates}"


def test_the_set_is_the_expected_size(golden: list[GoldenQuestion]) -> None:
    assert len(golden) == EXPECTED_QUESTIONS


def test_every_expected_document_exists(
    golden: list[GoldenQuestion], corpus_names: set[str]
) -> None:
    referenced = {doc for question in golden for doc in question.expected_docs}
    missing = referenced - corpus_names
    assert not missing, f"golden.jsonl references files not in {CORPUS_DIR}: {sorted(missing)}"


def test_every_document_is_exercised(golden: list[GoldenQuestion], corpus_names: set[str]) -> None:
    referenced = {doc for question in golden for doc in question.expected_docs}
    unused = corpus_names - referenced
    assert not unused, f"no question expects these documents: {sorted(unused)}"


def test_every_evidence_label_is_an_exact_corpus_excerpt(
    golden: list[GoldenQuestion], corpus_names: set[str]
) -> None:
    def normalise(value: str) -> str:
        return " ".join(re.sub(r"[`*_#]", "", value).casefold().split())

    documents = {path.name: normalise(path.read_text(encoding="utf-8")) for path in corpus_files()}
    for question in golden:
        for evidence in question.evidence:
            assert evidence.document in corpus_names, f"{question.id}: unknown evidence document"
            assert normalise(evidence.text) in documents[evidence.document], (
                f"{question.id}: evidence {evidence.id} is not an exact excerpt"
            )


def test_negative_questions_expect_nothing(golden: list[GoldenQuestion]) -> None:
    for question in golden:
        if question.type == "negative":
            assert question.expected_docs == [], f"{question.id} is negative but expects documents"


def test_answerable_questions_name_their_evidence(golden: list[GoldenQuestion]) -> None:
    for question in golden:
        if question.type != "negative":
            assert question.expected_docs, f"{question.id} has no expected document"
            assert question.evidence, f"{question.id} has no exact evidence labels"


def test_multi_hop_questions_need_more_than_one_document(golden: list[GoldenQuestion]) -> None:
    for question in golden:
        if question.type == "multi_hop":
            assert len(question.evidence) >= 2, f"{question.id} is not actually multi-hop"


def test_every_question_type_is_represented(golden: list[GoldenQuestion]) -> None:
    counts = Counter(question.type for question in golden)
    assert counts["factual"] >= 15
    assert counts["multi_hop"] >= 4
    assert counts["negative"] >= 3
    assert counts["cross_language"] >= 3
    assert counts["adversarial"] >= 1


def test_spanish_and_adversarial_cases_are_tagged(golden: list[GoldenQuestion]) -> None:
    assert sum(question.language == "es" for question in golden) >= 5
    assert any("prompt-injection" in question.tags for question in golden)


def test_questions_and_answers_are_substantial(golden: list[GoldenQuestion]) -> None:
    for question in golden:
        assert len(question.question) > 15, f"{question.id}: question too short to be realistic"
        assert len(question.reference_answer) > 10, f"{question.id}: reference answer too thin"
