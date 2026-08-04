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


class DocumentLimitError(ValueError):
    """Raised when extracted document content exceeds a configured limit."""


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


def parse_pdf(
    path: Path,
    *,
    max_pages: int | None = None,
    max_chars: int | None = None,
) -> list[RawSection]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    if max_pages is not None and len(reader.pages) > max_pages:
        raise DocumentLimitError(
            f"PDF has {len(reader.pages)} pages; the configured maximum is {max_pages}"
        )

    sections: list[RawSection] = []
    extracted_chars = 0
    for number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        extracted_chars += len(text)
        if max_chars is not None and extracted_chars > max_chars:
            raise DocumentLimitError(
                f"Document exceeds the configured extracted-text limit ({max_chars} characters)"
            )
        sections.append(RawSection(text=text, page=number))
    if sum(len(s.text) for s in sections) < MIN_EXTRACTED_CHARS:
        raise EmptyDocumentError(
            "No extractable text found. Scanned PDFs need OCR before ingestion."
        )
    return sections


def parse_text(path: Path, *, max_chars: int | None = None) -> list[RawSection]:
    if max_chars is None:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
    else:
        blocks: list[str] = []
        extracted_chars = 0
        with path.open(encoding="utf-8", errors="replace") as handle:
            while block := handle.read(min(1 << 20, max_chars + 1 - extracted_chars)):
                blocks.append(block)
                extracted_chars += len(block)
                if extracted_chars > max_chars:
                    raise DocumentLimitError(
                        "Document exceeds the configured extracted-text limit "
                        f"({max_chars} characters)"
                    )
        text = "".join(blocks).strip()
    if not text:
        raise EmptyDocumentError("File is empty")
    if max_chars is not None and len(text) > max_chars:
        raise DocumentLimitError(
            f"Document exceeds the configured extracted-text limit ({max_chars} characters)"
        )
    return [RawSection(text=text, page=None)]


def parse_file(
    path: Path,
    *,
    max_pages: int | None = None,
    max_chars: int | None = None,
) -> tuple[str, list[RawSection]]:
    """Parse ``path`` into ``(source_type, sections)``."""
    kind = source_type(path)
    return kind, (
        parse_pdf(path, max_pages=max_pages, max_chars=max_chars)
        if kind == "pdf"
        else parse_text(path, max_chars=max_chars)
    )
