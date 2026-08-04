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


def _headings(text: str) -> list[tuple[int, str]]:
    return [(m.start(), m.group(2).strip()) for m in _HEADING.finditer(text)]


def _section_label(headings: list[tuple[int, str]], start: int, end: int) -> str | None:
    """Which heading a chunk belongs under.

    A chunk that opens a new section is labelled with that section, not with
    the one it happens to follow.
    """
    inside = next((title for offset, title in headings if start <= offset < end), None)
    if inside is not None:
        return inside
    preceding = [title for offset, title in headings if offset < start]
    return preceding[-1] if preceding else None


def split_sections(
    sections: list[RawSection],
    source_type: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    splitter = _splitter(source_type, chunk_size, chunk_overlap)
    chunks: list[Chunk] = []

    for section in sections:
        headings = _headings(section.text) if source_type == "md" else []
        search_from = 0

        for piece in splitter.split_text(section.text):
            start = section.text.find(piece, search_from)
            if start < 0:
                # The splitter normalises whitespace in some cases; fall back to
                # the previous position rather than mislabelling the chunk.
                start = search_from
            else:
                search_from = start + 1

            chunks.append(
                Chunk(
                    text=piece,
                    page=section.page,
                    section=_section_label(headings, start, start + len(piece)),
                    index=len(chunks),
                )
            )

    return chunks
