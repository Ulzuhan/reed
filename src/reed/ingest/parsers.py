"""File parsing into sections.

A *section* is the smallest unit that still carries a citable location — a page
for PDFs, the whole file for text formats. Chunking happens later, inside a
section, so a chunk never straddles two pages and citations stay precise.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_SUFFIXES = frozenset(
    {".pdf", ".md", ".markdown", ".txt", ".text", ".docx", ".html", ".htm"}
)

# Below this, a PDF is almost certainly scanned images rather than text.
MIN_EXTRACTED_CHARS = 50

# Uncompressed OOXML is mostly markup, so it dwarfs the text it carries — but
# not by a thousandfold. Refusing on the size the archive declares turns a zip
# bomb into a limit error instead of an out-of-memory kill, before anything is
# decompressed. A lying central directory still meets the parser's memory cap.
DOCX_EXPANSION_BUDGET = 20

# Same idea for HTML, whose markup also dwarfs its text, measured on the source
# before it is read into memory.
HTML_MARKUP_BUDGET = 20

# Elements carrying no readable content, or content a reader never sees.
_HTML_DISCARDED = "script, style, template, noscript, svg, iframe, object, embed, canvas"

# Hidden by an inline style. CSS in a stylesheet or a <style> block is *not*
# applied — see parse_html's docstring, which says so rather than implying a
# guarantee this cannot keep.
_HIDDEN_STYLE = re.compile(
    r"display\s*:\s*none"
    r"|visibility\s*:\s*hidden"
    r"|opacity\s*:\s*0(?:\.0+)?(?![\d.])"
    r"|font-size\s*:\s*0(?:\.0+)?(?:px|pt|em|rem|%)?(?![\d.])",
    re.IGNORECASE,
)


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
    if suffix == ".docx":
        return "docx"
    if suffix in {".html", ".htm"}:
        return "html"
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


def _accepted_text(element: object) -> str:
    """Text of a paragraph with tracked changes applied.

    ``paragraph.text`` reads only the runs directly under ``w:p``, so it drops
    insertions, which live under ``w:ins``, while deletions are already out
    because they use ``w:delText``. The result matches neither the accepted nor
    the rejected document — text nobody reviewing the file would recognise.
    Reading every ``w:t`` is the accepted view.
    """
    nodes = element.xpath(".//w:t")  # type: ignore[attr-defined]
    return "".join(node.text or "" for node in nodes)


def _cell_text(cell: object) -> str:
    # python-docx exposes no public handle on a paragraph's XML element.
    return " ".join(
        _accepted_text(paragraph._p)
        for paragraph in cell.paragraphs  # type: ignore[attr-defined]
    )


def _table_row(cells: list[str]) -> str:
    """One table row, with cells that cannot reshape the row they sit in."""
    escaped = [" ".join(cell.replace("|", r"\|").split()) for cell in cells]
    return f"| {' | '.join(escaped)} |"


def _declared_uncompressed_size(path: Path) -> int:
    import zipfile

    try:
        with zipfile.ZipFile(path) as archive:
            return sum(info.file_size for info in archive.infolist())
    except zipfile.BadZipFile as exc:
        raise UnsupportedFileError(
            f"'{path.name}' is not a readable .docx. Word 97-2003 .doc files must be "
            "converted first."
        ) from exc


def parse_docx(path: Path, *, max_chars: int | None = None) -> list[RawSection]:
    """Extract headings, paragraphs and tables in document order.

    Headings become Markdown so that chunking can label sections exactly as it
    does for Markdown files; tables become one line per row. Comments and
    footnotes live in other parts of the package and are not ingested: a
    document's suggestions are not what it says.
    """
    from docx import Document
    from docx.opc.exceptions import PackageNotFoundError
    from docx.oxml.table import CT_Tbl
    from docx.oxml.text.paragraph import CT_P
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    declared = _declared_uncompressed_size(path)
    if max_chars is not None and declared > max_chars * DOCX_EXPANSION_BUDGET:
        raise DocumentLimitError(
            f"Document expands to {declared} bytes, beyond what the configured "
            f"extracted-text limit ({max_chars} characters) can plausibly need"
        )

    try:
        document = Document(str(path))
    except (PackageNotFoundError, ValueError) as exc:
        raise UnsupportedFileError(f"'{path.name}' is not a readable .docx") from exc

    lines: list[str] = []
    extracted_chars = 0

    def keep(line: str) -> None:
        nonlocal extracted_chars
        extracted_chars += len(line)
        if max_chars is not None and extracted_chars > max_chars:
            raise DocumentLimitError(
                f"Document exceeds the configured extracted-text limit ({max_chars} characters)"
            )
        lines.append(line)

    body = document.element.body
    for child in body.iterchildren():
        if isinstance(child, CT_P):
            paragraph = Paragraph(child, document)
            text = _accepted_text(child).strip()
            if not text:
                continue
            level = _heading_level(paragraph.style.name if paragraph.style else None)
            keep(f"{'#' * level} {text}" if level else text)
        elif isinstance(child, CT_Tbl):
            for row in Table(child, document).rows:
                cells = [_cell_text(cell) for cell in row.cells]
                if any(cell.strip() for cell in cells):
                    keep(_table_row(cells))

    text = "\n\n".join(lines).strip()
    if not text:
        raise EmptyDocumentError("Document contains no text")
    return [RawSection(text=text, page=None)]


def _heading_level(style_name: str | None) -> int:
    """Markdown level for a Word heading style, or 0 for body text."""
    if not style_name:
        return 0
    if style_name == "Title":
        return 1
    if style_name.startswith("Heading "):
        suffix = style_name.removeprefix("Heading ").strip()
        if suffix.isdigit():
            return min(int(suffix), 6)
    return 0


def parse_html(path: Path, *, max_chars: int | None = None) -> list[RawSection]:
    """Extract readable text, dropping what a reader would never see.

    Scripts, styles and embedded objects go first, then anything hidden by an
    inline style, a ``hidden`` attribute or ``aria-hidden``. That matters more
    here than in other formats: text a human cannot see in the browser is fully
    visible to the embedder and to the prompt, which makes an HTML upload a
    place to smuggle instructions past whoever approved the document.

    The check is inline styles only. A rule in a stylesheet or a ``<style>``
    block that hides an element is **not** detected, because Reed parses HTML
    and does not apply CSS. Parsing is entirely offline: no stylesheet, image
    or frame is ever fetched.

    Headings become Markdown so section labelling matches every other format.
    """
    from selectolax.lexbor import LexborHTMLParser

    if max_chars is not None:
        declared = path.stat().st_size
        if declared > max_chars * HTML_MARKUP_BUDGET:
            raise DocumentLimitError(
                f"Document is {declared} bytes, beyond what the configured extracted-text "
                f"limit ({max_chars} characters) can plausibly need"
            )

    tree = LexborHTMLParser(path.read_text(encoding="utf-8", errors="replace"))
    body = tree.body
    if body is None:
        raise EmptyDocumentError("Document has no body")

    for node in tree.css(_HTML_DISCARDED):
        node.decompose()
    for node in body.css("[hidden], [aria-hidden]"):
        if node.attributes.get("aria-hidden", "true") == "true":
            node.decompose()
    for node in body.css("[style]"):
        if _HIDDEN_STYLE.search(node.attributes.get("style") or ""):
            node.decompose()

    for level in range(1, 7):
        for node in body.css(f"h{level}"):
            title = " ".join((node.text(separator=" ", strip=True) or "").split())
            node.replace_with(f"\n\n{'#' * level} {title}\n\n" if title else "\n\n")
    for table in body.css("table"):
        rows = [
            _table_row([cell.text(separator=" ", strip=True) for cell in row.css("th, td")])
            for row in table.css("tr")
        ]
        table.replace_with("\n\n" + "\n\n".join(rows) + "\n\n" if rows else "\n\n")

    text = _collapse_blank_lines(body.text(separator="\n", strip=False))
    if not text:
        raise EmptyDocumentError("Document contains no readable text")
    if max_chars is not None and len(text) > max_chars:
        raise DocumentLimitError(
            f"Document exceeds the configured extracted-text limit ({max_chars} characters)"
        )
    return [RawSection(text=text, page=None)]


def _collapse_blank_lines(text: str) -> str:
    lines = [" ".join(line.split()) for line in text.splitlines()]
    collapsed: list[str] = []
    for line in lines:
        if line or (collapsed and collapsed[-1]):
            collapsed.append(line)
    return "\n".join(collapsed).strip()


def parse_file(
    path: Path,
    *,
    max_pages: int | None = None,
    max_chars: int | None = None,
) -> tuple[str, list[RawSection]]:
    """Parse ``path`` into ``(source_type, sections)``."""
    kind = source_type(path)
    if kind == "pdf":
        return kind, parse_pdf(path, max_pages=max_pages, max_chars=max_chars)
    if kind == "docx":
        return kind, parse_docx(path, max_chars=max_chars)
    if kind == "html":
        return kind, parse_html(path, max_chars=max_chars)
    return kind, parse_text(path, max_chars=max_chars)
