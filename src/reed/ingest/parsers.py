"""File parsing into sections.

A *section* is the smallest unit that still carries a citable location — a page
for PDFs, the whole file for text formats. Chunking happens later, inside a
section, so a chunk never straddles two pages and citations stay precise.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SUPPORTED_SUFFIXES = frozenset({".pdf", ".md", ".markdown", ".txt", ".text"})

# Below this, a PDF is almost certainly scanned images rather than text.
MIN_EXTRACTED_CHARS = 50


class UnsupportedFileError(ValueError):
    """Raised for a file type Reed cannot read."""


class EmptyDocumentError(ValueError):
    """Raised when a file yields no usable text."""


@dataclass(frozen=True, slots=True)
class RawSection:
    text: str
    page: int | None = None


def source_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return "pdf"
    if suffix in {".md", ".markdown"}:
        return "md"
    if suffix in {".txt", ".text"}:
        return "txt"
    raise UnsupportedFileError(
        f"Unsupported file type '{suffix or path.name}'. "
        f"Supported: {', '.join(sorted(SUPPORTED_SUFFIXES))}"
    )


def parse_pdf(path: Path) -> list[RawSection]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    sections = [
        RawSection(text=text.strip(), page=number)
        for number, page in enumerate(reader.pages, start=1)
        if (text := page.extract_text() or "").strip()
    ]
    if sum(len(s.text) for s in sections) < MIN_EXTRACTED_CHARS:
        raise EmptyDocumentError(
            "No extractable text found. Scanned PDFs need OCR before ingestion."
        )
    return sections


def parse_text(path: Path) -> list[RawSection]:
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        raise EmptyDocumentError("File is empty")
    return [RawSection(text=text, page=None)]


def parse_file(path: Path) -> tuple[str, list[RawSection]]:
    """Parse ``path`` into ``(source_type, sections)``."""
    kind = source_type(path)
    return kind, (parse_pdf(path) if kind == "pdf" else parse_text(path))
