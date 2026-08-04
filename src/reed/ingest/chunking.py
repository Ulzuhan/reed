"""Splitting sections into chunks.

Sizes are measured in characters rather than tokens: Reed has to behave the
same whether the embedding model behind it is OpenAI's or a local Gemma, and
those disagree on tokenization. The cost is slightly uneven token counts per
chunk; the benefit is one chunking behaviour across every provider.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from reed.ingest.parsers import RawSection

_HEADING = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$", re.MULTILINE)
_FENCE = re.compile(r"^\s{0,3}(`{3,}|~{3,})(.*)$")


@dataclass(frozen=True, slots=True)
class Chunk:
    text: str
    page: int | None
    section: str | None
    index: int


def _splitter(
    source_type: str, chunk_size: int, chunk_overlap: int
) -> RecursiveCharacterTextSplitter:
    if source_type == "md":
        return RecursiveCharacterTextSplitter.from_language(
            Language.MARKDOWN,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def _real_headings(text: str) -> list[str]:
    """Every heading in a whole document, ignoring fenced code blocks.

    A shell comment inside a fence is not a heading, and mislabelling one as a
    section poisons every chunk that follows it. This has to run on the whole
    document: the splitter routinely puts a fence's opening and closing lines
    in different chunks, so neither chunk alone can tell it is inside code.
    """
    headings: list[str] = []
    fence: str | None = None

    for line in text.splitlines():
        marker = _FENCE.match(line)
        if marker is not None:
            if fence is None:
                fence = marker.group(1)[0]  # ` or ~
            elif marker.group(1).startswith(fence) and not marker.group(2).strip():
                fence = None
            continue
        if fence is None and (heading := _HEADING.match(line)):
            headings.append(heading.group(2).strip())

    return headings


def _headings_in(chunk: str, expected: list[str]) -> list[str]:
    """Headings found in a chunk, filtered to the ones the document really has."""
    found = [match.group(2).strip() for match in _HEADING.finditer(chunk)]
    return [title for title in found if title in expected]


def split_sections(
    sections: list[RawSection],
    source_type: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    """Split into chunks, labelling each with the heading it lives under.

    Headings are read out of the chunk text and carried forward, but only the
    ones the whole document agrees are headings. Locating each chunk back in
    the source would be the obvious alternative, and it is what this used to
    do — but the splitter rewrites separators, so the lookup fails on exactly
    the documents that need it and mislabels silently.
    """
    splitter = _splitter(source_type, chunk_size, chunk_overlap)
    chunks: list[Chunk] = []

    for section in sections:
        expected = _real_headings(section.text) if source_type == "md" else []
        current: str | None = None

        for piece in splitter.split_text(section.text):
            headings = _headings_in(piece, expected) if expected else []
            # A chunk that opens a section belongs to it, not to the previous
            # one; a chunk spanning several hands the last one to its successor.
            label = headings[0] if headings else current
            if headings:
                current = headings[-1]

            chunks.append(
                Chunk(
                    text=piece,
                    page=section.page,
                    section=label,
                    index=len(chunks),
                )
            )

    return chunks
