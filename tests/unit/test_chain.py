from __future__ import annotations

from reed.rag.chain import Turn, audit_citations, build_retrieval_query


def test_unrelated_new_question_does_not_reuse_old_history() -> None:
    history = [Turn(role="user", content="What is the expense limit?")]
    question = "Who owns incident response?"

    assert build_retrieval_query(question, history) == question


def test_short_follow_up_includes_the_previous_user_question() -> None:
    history = [
        Turn(role="user", content="What is the expense limit?"),
        Turn(role="assistant", content="75 euros [1]."),
    ]

    query = build_retrieval_query("And for contractors?", history)

    assert "expense limit" in query
    assert "contractors" in query


def test_citation_audit_rejects_unknown_source_numbers() -> None:
    status, warnings = audit_citations("The threshold is 75 euros [3].", source_count=2)

    assert status == "invalid"
    assert "3" in warnings[0]


def test_citation_audit_reports_missing_markers() -> None:
    status, _ = audit_citations("The threshold is 75 euros.", source_count=1)
    assert status == "missing"
