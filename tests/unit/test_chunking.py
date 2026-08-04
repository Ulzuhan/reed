from __future__ import annotations

from reed.ingest.chunking import split_sections
from reed.ingest.parsers import RawSection

MARKDOWN = """# Employee Handbook

## Remote work

Employees may work remotely up to four days a week.

## Expenses

Expenses above 75 euros require pre-approval from your manager.
"""


def test_chunks_are_numbered_consecutively() -> None:
    sections = [RawSection(text="word " * 600, page=None)]

    chunks = split_sections(sections, source_type="txt", chunk_size=200, chunk_overlap=20)

    assert len(chunks) > 1
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_chunks_stay_within_their_page() -> None:
    sections = [
        RawSection(text="page one " * 80, page=1),
        RawSection(text="page two " * 80, page=2),
    ]

    chunks = split_sections(sections, source_type="pdf", chunk_size=200, chunk_overlap=20)

    assert {c.page for c in chunks} == {1, 2}
    for chunk in chunks:
        expected = "page one" if chunk.page == 1 else "page two"
        assert expected in chunk.text


def test_markdown_chunks_carry_their_nearest_heading() -> None:
    chunks = split_sections(
        [RawSection(text=MARKDOWN, page=None)],
        source_type="md",
        chunk_size=120,
        chunk_overlap=0,
    )

    sections_seen = {c.section for c in chunks}
    assert "Expenses" in sections_seen
    expenses_chunk = next(c for c in chunks if "75 euros" in c.text)
    assert expenses_chunk.section == "Expenses"


def test_headings_come_from_the_chunk_not_from_a_lookup() -> None:
    """A regression guard on section labels.

    The Markdown splitter rewrites separators, so chunks cannot reliably be
    located back in the source. Doing so mislabelled chunks with a heading from
    further down the document.
    """
    document = "\n\n".join(
        f"## Section {n}\n\nBody text for section {n}. " + ("filler " * 40) for n in range(1, 6)
    )

    chunks = split_sections(
        [RawSection(text=document, page=None)],
        source_type="md",
        chunk_size=400,
        chunk_overlap=60,
    )

    for chunk in chunks:
        # Whatever heading a chunk claims, its body must belong to that section.
        assert chunk.section is not None
        number = chunk.section.split()[-1]
        assert f"section {number}" in chunk.text.lower()


def test_plain_text_has_no_section_labels() -> None:
    chunks = split_sections(
        [RawSection(text="# not a heading here\nplain text", page=None)],
        source_type="txt",
        chunk_size=500,
        chunk_overlap=0,
    )

    assert all(c.section is None for c in chunks)


def test_overlap_keeps_content_contiguous() -> None:
    text = " ".join(f"w{i}" for i in range(300))

    chunks = split_sections(
        [RawSection(text=text, page=None)],
        source_type="txt",
        chunk_size=200,
        chunk_overlap=50,
    )

    # Every word survives somewhere, so retrieval can never miss a sentence
    # that happened to fall on a chunk boundary.
    joined = " ".join(c.text for c in chunks)
    assert "w0" in joined
    assert "w299" in joined
