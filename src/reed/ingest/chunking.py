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


def _real_heading_spans(text: str) -> list[tuple[int, str]]:
    """Insertion offsets and titles for headings outside fenced code.

    This has to scan the whole document: the splitter routinely puts a fence's
    opening and closing lines in different chunks, so neither chunk alone can
    tell whether a shell comment is really a heading.
    """
    headings: list[tuple[int, str]] = []
    fence: str | None = None
    offset = 0

    for raw_line in text.splitlines(keepends=True):
        line = raw_line.rstrip("\r\n")
        marker = _FENCE.match(line)
        if marker is not None:
            delimiter = marker.group(1)
            if fence is None:
                fence = delimiter
            elif _closes(delimiter, fence) and not marker.group(2).strip():
                fence = None
        elif fence is None and (heading := _HEADING.match(line)):
            # Put the marker at the end of the heading line. Prefixing the line
            # would hide Markdown's heading separator from the splitter.
            headings.append((offset + len(line), heading.group(2).strip()))
        offset += len(raw_line)

    return headings


def _closes(delimiter: str, opening: str) -> bool:
    """Whether a fence line closes the block ``opening`` started.

    Length matters: a three-backtick fence nested inside a four-backtick one
    does not close it. Treating any run of backticks as a closer made the outer
    fence's real terminator look like a new opening, which swallowed every
    heading after it — the exact shape of document that shows code fences
    inside code fences, like this project's own README.
    """
    return delimiter[0] == opening[0] and len(delimiter) >= len(opening)


def _mark_real_headings(text: str, spans: list[tuple[int, str]]) -> tuple[str, re.Pattern[str]]:
    # Uploaded text can contain arbitrary private-use characters. Grow the
    # prefix until it is absent so user content can never impersonate one of
    # our internal markers or produce an out-of-range heading index.
    prefix = "\ue000reed-heading-"
    while prefix in text:
        prefix = f"\ue000{prefix}"
    marker_pattern = re.compile(rf"{re.escape(prefix)}(\d+)\ue001")
    marked = text
    for index, (offset, _) in reversed(list(enumerate(spans))):
        marker = f"{prefix}{index}\ue001"
        marked = f"{marked[:offset]}{marker}{marked[offset:]}"
    return marked, marker_pattern


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
        spans = _real_heading_spans(section.text) if source_type == "md" else []
        headings_by_index = [title for _, title in spans]
        if spans:
            text_to_split, heading_marker = _mark_real_headings(section.text, spans)
        else:
            text_to_split, heading_marker = section.text, None
        current: str | None = None

        for marked_piece in splitter.split_text(text_to_split):
            heading_indices = (
                [int(match.group(1)) for match in heading_marker.finditer(marked_piece)]
                if heading_marker is not None
                else []
            )
            headings = [headings_by_index[index] for index in heading_indices]
            piece = (
                heading_marker.sub("", marked_piece) if heading_marker is not None else marked_piece
            )
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
