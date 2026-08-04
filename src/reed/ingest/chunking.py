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


def _headings(text: str) -> list[str]:
    return [match.group(2).strip() for match in _HEADING.finditer(text)]


def split_sections(
    sections: list[RawSection],
    source_type: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    """Split into chunks, labelling each with the heading it lives under.

    The heading is read out of the chunk text itself, carried forward to the
    chunks that follow it. Locating each chunk back in the source would be the
    obvious alternative, but the splitter rewrites separators, so the lookup
    fails on exactly the documents that need it and mislabels silently.
    """
    splitter = _splitter(source_type, chunk_size, chunk_overlap)
    chunks: list[Chunk] = []

    for section in sections:
        current: str | None = None

        for piece in splitter.split_text(section.text):
            headings = _headings(piece) if source_type == "md" else []
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
