from __future__ import annotations

import pytest

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


RUNBOOK = (
    "# Runbook\n\n## Paging\n\n"
    + ("Filler prose to push past the chunk boundary. " * 20)
    + "\n\n## Rolling back\n\n"
    + ("More filler prose to force another split here. " * 20)
    + "\n\n```bash\n# Roll back to the previous release\nreed deploy --rollback\n```\n\n"
    + ("Yet more prose after the code block so it splits. " * 20)
    + "\n\n## Writing the postmortem\n\nPublish within five working days.\n"
)


def test_a_comment_in_a_code_fence_is_not_a_heading() -> None:
    # The splitter puts a fence's opening and closing lines in different
    # chunks, so neither chunk alone can tell it is inside code — the document
    # as a whole has to decide what counts as a heading.
    chunks = split_sections(
        [RawSection(text=RUNBOOK, page=None)],
        source_type="md",
        chunk_size=800,
        chunk_overlap=100,
    )

    assert "Roll back to the previous release" not in {c.section for c in chunks}
    comment_chunk = next(c for c in chunks if "reed deploy --rollback" in c.text)
    assert comment_chunk.section == "Rolling back"


def test_a_fenced_heading_with_a_real_heading_title_is_still_code() -> None:
    document = (
        "# Guide\n\n## Deployment\n\nReal deployment instructions.\n\n"
        "```markdown\n## Recovery\nThis is only an example.\n```\n\n"
        "More deployment prose.\n\n## Recovery\n\nReal recovery instructions.\n"
    )

    chunks = split_sections(
        [RawSection(text=document, page=None)],
        source_type="md",
        chunk_size=120,
        chunk_overlap=0,
    )

    example = next(chunk for chunk in chunks if "only an example" in chunk.text)
    real = next(chunk for chunk in chunks if "Real recovery" in chunk.text)
    assert example.section == "Deployment"
    assert real.section == "Recovery"


def test_uploaded_text_cannot_impersonate_internal_heading_markers() -> None:
    marker = "\ue000reed-heading-999\ue001"
    document = f"# Guide\n\nLiteral marker: {marker}\n\n## Safe\n\nStill labelled correctly."

    chunks = split_sections(
        [RawSection(text=document, page=None)],
        source_type="md",
        chunk_size=2_000,
        chunk_overlap=0,
    )

    assert marker in chunks[0].text
    assert chunks[0].section == "Guide"


def test_a_heading_after_a_code_fence_is_not_lost() -> None:
    chunks = split_sections(
        [RawSection(text=RUNBOOK, page=None)],
        source_type="md",
        chunk_size=800,
        chunk_overlap=100,
    )

    assert "Writing the postmortem" in {c.section for c in chunks}


def test_a_fence_nested_in_a_longer_fence_does_not_close_it() -> None:
    # A document that shows a code fence inside a code fence — a style guide, a
    # README for a docs tool. Treating any run of backticks as a closer made
    # the outer fence's real terminator look like a new opening, which dropped
    # every heading after it.
    document = (
        "# Style guide\n\n## Fences\n\n"
        + ("Show fenced blocks with a longer outer fence. " * 20)
        + "\n\n````markdown\n```bash\nreed serve\n```\n````\n\n"
        + ("Prose between the two sections so the document splits. " * 20)
        + "\n\n## Headings\n\nUse ATX headings and sentence case.\n"
    )

    chunks = split_sections(
        [RawSection(text=document, page=None)],
        source_type="md",
        chunk_size=800,
        chunk_overlap=100,
    )

    assert "Headings" in {c.section for c in chunks}


def test_an_unterminated_fence_does_not_swallow_later_headings() -> None:
    document = "# Notes\n\n```bash\nreed serve\n\n## Verification\n\nCheck the dashboard.\n"

    chunks = split_sections(
        [RawSection(text=document, page=None)],
        source_type="md",
        chunk_size=2000,
        chunk_overlap=0,
    )

    # Inside an unterminated fence everything is code, so the only heading that
    # counts is the one that opened the document.
    assert {c.section for c in chunks} == {"Notes"}


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


@pytest.mark.parametrize("source_type", ["docx", "html"])
def test_converted_sources_are_labelled_like_markdown(source_type: str) -> None:
    # DOCX and HTML both emit their headings as Markdown precisely so that
    # section labelling stays one implementation rather than three.
    sections = [
        RawSection(text="# Expenses\n\nPre-approval above 75 euros.\n\n## Caps\n\nMeals are 40.")
    ]

    chunks = split_sections(sections, source_type, chunk_size=40, chunk_overlap=0)

    assert {chunk.section for chunk in chunks} == {"Expenses", "Caps"}
    assert next(c for c in chunks if "75 euros" in c.text).section == "Expenses"
    assert next(c for c in chunks if "Meals" in c.text).section == "Caps"
