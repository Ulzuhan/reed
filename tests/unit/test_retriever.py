from __future__ import annotations

from reed.rag.retriever import SNIPPET_CHARS, RetrievedChunk, diversify


def chunk(**kwargs: object) -> RetrievedChunk:
    defaults: dict[str, object] = {
        "text": "Expenses above 75 euros require pre-approval.",
        "score": 0.9,
        "doc_id": "d-abc",
        "filename": "expenses.md",
        "page": None,
        "section": None,
    }
    return RetrievedChunk(**{**defaults, **kwargs})  # type: ignore[arg-type]


def test_location_prefers_a_page_number() -> None:
    assert chunk(page=3, section="Pre-approval").location == "expenses.md, p. 3"


def test_location_falls_back_to_the_section() -> None:
    assert chunk(section="Pre-approval").location == "expenses.md — Pre-approval"


def test_location_is_just_the_filename_when_nothing_else_is_known() -> None:
    assert chunk().location == "expenses.md"


def test_snippets_drop_markdown_syntax() -> None:
    snippet = chunk(text="## Pre-approval\n\nAnything over **75 euros** needs `approval`.").snippet

    assert snippet == "Pre-approval Anything over 75 euros needs approval."


def test_snippets_keep_list_content_but_not_bullets() -> None:
    assert chunk(text="- first\n- second").snippet == "first second"


def test_long_snippets_are_truncated_with_an_ellipsis() -> None:
    snippet = chunk(text="word " * 200).snippet

    assert len(snippet) <= SNIPPET_CHARS + 1
    assert snippet.endswith("…")


def test_short_snippets_are_left_alone() -> None:
    assert not chunk(text="short enough").snippet.endswith("…")


def test_diversity_limits_one_document_and_prefers_distinct_text() -> None:
    candidates = [
        chunk(doc_id="a", filename="a.md", text="same repeated policy", score=1.0),
        chunk(doc_id="a", filename="a.md", text="same repeated policy", score=0.99),
        chunk(doc_id="b", filename="b.md", text="different supporting evidence", score=0.8),
    ]

    selected = diversify(candidates, k=3, lambda_mult=0.7, max_per_document=1)

    assert [item.filename for item in selected] == ["a.md", "b.md"]
