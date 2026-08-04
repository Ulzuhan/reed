from __future__ import annotations

from reed.rag.prompts import build_context_block, build_system_prompt
from reed.rag.retriever import RetrievedChunk


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


def test_context_blocks_are_numbered_from_one() -> None:
    block = build_context_block([chunk(text="first"), chunk(text="second")])

    assert block.index("[1]") < block.index("[2]")
    assert "first" in block
    assert "second" in block


def test_pdf_chunks_cite_a_page() -> None:
    assert "expenses.md, p. 3" in build_context_block([chunk(page=3)])


def test_markdown_chunks_cite_their_section() -> None:
    assert "expenses.md — Pre-approval" in build_context_block([chunk(section="Pre-approval")])


def test_page_wins_over_section_when_both_exist() -> None:
    assert "p. 2" in build_context_block([chunk(page=2, section="Pre-approval")])


def test_system_prompt_states_the_citation_contract() -> None:
    prompt = build_system_prompt([chunk()])

    assert "ONLY the excerpts" in prompt
    assert "square brackets" in prompt
    assert "75 euros" in prompt


def test_snippets_are_flattened_and_capped() -> None:
    long_chunk = chunk(text="word " * 200)

    assert len(long_chunk.snippet) <= 221
    assert "\n" not in long_chunk.snippet
    assert long_chunk.snippet.endswith("…")
