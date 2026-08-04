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
from tests.pdf_fixture import write_pdf


def test_source_type_maps_extensions() -> None:
    assert source_type(Path("a.pdf")) == "pdf"
    assert source_type(Path("a.MD")) == "md"
    assert source_type(Path("a.markdown")) == "md"
    assert source_type(Path("a.txt")) == "txt"


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
