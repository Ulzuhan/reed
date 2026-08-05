from __future__ import annotations

from pathlib import Path

import pytest

from reed.ingest.parser_worker import ParseJob, _run_job, parse_file_isolated
from reed.ingest.parsers import (
    DocumentLimitError,
    EmptyDocumentError,
    UnsupportedFileError,
    parse_file,
    source_type,
)
from tests.docx_fixture import write_docx, write_docx_with_revisions
from tests.pdf_fixture import write_pdf


def test_source_type_maps_extensions() -> None:
    assert source_type(Path("a.pdf")) == "pdf"
    assert source_type(Path("a.MD")) == "md"
    assert source_type(Path("a.markdown")) == "md"
    assert source_type(Path("a.txt")) == "txt"
    assert source_type(Path("a.DOCX")) == "docx"


def test_unsupported_extension_names_the_alternatives() -> None:
    with pytest.raises(UnsupportedFileError, match=r"\.pdf"):
        source_type(Path("sheet.xlsx"))


def test_markdown_is_one_section_without_a_page(tmp_path: Path) -> None:
    path = tmp_path / "policy.md"
    path.write_text("# Remote work\n\nEmployees may work remotely.", encoding="utf-8")

    kind, sections = parse_file(path)

    assert kind == "md"
    assert len(sections) == 1
    assert sections[0].page is None
    assert "remotely" in sections[0].text


def test_empty_file_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "blank.txt"
    path.write_text("   \n", encoding="utf-8")

    with pytest.raises(EmptyDocumentError):
        parse_file(path)


def test_extracted_text_has_a_hard_limit(tmp_path: Path) -> None:
    path = tmp_path / "large.txt"
    path.write_text("x" * 101, encoding="utf-8")

    with pytest.raises(ValueError, match="extracted-text limit"):
        parse_file(path, max_chars=100)


def test_pdf_pages_become_numbered_sections(tmp_path: Path) -> None:
    pdf = write_pdf(
        tmp_path / "handbook.pdf",
        [
            "Expenses above 75 euros require pre-approval from your manager.",
            "P1 incidents page the on-call engineer within 15 minutes.",
        ],
    )

    kind, sections = parse_file(pdf)

    assert kind == "pdf"
    assert [s.page for s in sections] == [1, 2]
    assert "Expenses above 75 euros" in sections[0].text
    assert "on-call" in sections[1].text


def test_pdf_page_count_has_a_hard_limit(tmp_path: Path) -> None:
    pdf = write_pdf(tmp_path / "long.pdf", ["page one has enough text"] * 2)

    with pytest.raises(ValueError, match="configured maximum"):
        parse_file(pdf, max_pages=1)


def test_scanned_pdf_without_text_is_rejected(tmp_path: Path) -> None:
    pdf = write_pdf(tmp_path / "scan.pdf", ["x"])

    with pytest.raises(EmptyDocumentError, match="OCR"):
        parse_file(pdf)


def test_isolated_parser_returns_sections_without_sharing_the_process(tmp_path: Path) -> None:
    path = tmp_path / "policy.md"
    path.write_text("# Policy\n\nExpense approval is required.", encoding="utf-8")

    kind, sections = parse_file_isolated(
        path,
        max_pages=10,
        max_chars=1_000,
        timeout_seconds=5,
        memory_mb=1_024,
        cpu_seconds=2,
    )

    assert kind == "md"
    assert sections[0].text.endswith("required.")


def test_isolated_parser_is_terminated_at_its_deadline(tmp_path: Path) -> None:
    path = tmp_path / "slow.txt"
    path.write_text("content", encoding="utf-8")
    job = ParseJob(
        path=path,
        max_pages=10,
        max_chars=1_000,
        memory_mb=1_024,
        cpu_seconds=2,
        delay_seconds=0.2,
    )

    with pytest.raises(DocumentLimitError, match="timeout"):
        _run_job(job, timeout_seconds=0.01)


def test_docx_headings_paragraphs_and_tables_survive_in_order(tmp_path: Path) -> None:
    from docx import Document

    path = tmp_path / "handbook.docx"
    document = Document()
    document.add_heading("Expenses", level=1)
    document.add_paragraph("Anything above 75 euros needs pre-approval.")
    document.add_heading("Caps", level=2)
    table = document.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "Meals"
    # A pipe inside a cell must not reshape the row it lands in.
    table.cell(0, 1).text = "40 | per day"
    document.save(str(path))

    kind, sections = parse_file(path, max_chars=100_000)

    assert kind == "docx"
    assert sections[0].page is None
    assert sections[0].text == (
        "# Expenses\n\n"
        "Anything above 75 euros needs pre-approval.\n\n"
        "## Caps\n\n"
        r"| Meals | 40 \| per day |"
    )


def test_docx_extraction_applies_tracked_changes(tmp_path: Path) -> None:
    # python-docx's own paragraph.text drops insertions and deletions alike,
    # which is a view of the document that nobody reviewing it would recognise.
    path = write_docx_with_revisions(
        tmp_path / "review.docx", inserted="ADDED CLAUSE", deleted="REMOVED CLAUSE"
    )

    _, sections = parse_file(path, max_chars=100_000)

    assert "ADDED CLAUSE" in sections[0].text
    assert "REMOVED CLAUSE" not in sections[0].text


def test_docx_respects_the_extracted_character_limit(tmp_path: Path) -> None:
    path = write_docx(tmp_path / "long.docx", [("Normal", "word " * 200)] * 5)

    with pytest.raises(DocumentLimitError, match="extracted-text limit"):
        parse_file(path, max_chars=500)


def test_docx_refuses_an_archive_that_declares_an_implausible_expansion(
    tmp_path: Path,
) -> None:
    path = write_docx(tmp_path / "bomb.docx", [("Normal", "x" * 5_000)])

    with pytest.raises(DocumentLimitError, match="expands to"):
        parse_file(path, max_chars=10)


def test_docx_rejects_files_that_only_look_like_one(tmp_path: Path) -> None:
    renamed = tmp_path / "legacy.docx"
    renamed.write_bytes(b"\xd0\xcf\x11\xe0 old binary Word document")

    with pytest.raises(UnsupportedFileError, match=r"\.doc files must be converted"):
        parse_file(renamed)


def test_an_empty_docx_is_reported_as_empty(tmp_path: Path) -> None:
    path = write_docx(tmp_path / "blank.docx", [("Normal", "   ")])

    with pytest.raises(EmptyDocumentError, match="no text"):
        parse_file(path, max_chars=100_000)
