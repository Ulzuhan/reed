"""Retrieve, then generate — the two steps, as a stream of typed events.

One generator feeds all three consumers (the SSE endpoint, the plain JSON
endpoint and the evaluation runner), so what the evaluation measures is exactly
what users get.
"""

from __future__ import annotations

import re
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import anyio.to_thread
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from reed.log import get_logger
from reed.rag.prompts import (
    build_query_envelope,
    build_system_prompt,
    insufficient_evidence_answer,
    no_context_answer,
)
from reed.rag.retriever import RetrievedChunk, retrieve

if TYPE_CHECKING:
    from reed.services import Services

logger = get_logger(__name__)

# How many prior turns to replay. Enough for follow-ups ("and for contractors?")
# without letting an old topic drown out the current question.
MAX_HISTORY_MESSAGES = 6
CitationStatus = Literal["valid", "missing", "invalid", "not_applicable"]
_CITATION = re.compile(r"\[(\d+)\]")
# Spanish follow-ups routinely open with inverted punctuation ("¿y si…?"),
# which must not defeat the ^ anchor.
_FOLLOW_UP = re.compile(
    r"^[¿¡]?\s*(?:and\b|also\b|but\b|what about\b|how about\b|y\b|también\b|pero\b|"
    r"qué hay de\b|y si\b|et\b|mais\b|und\b|aber\b)",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class SourcesEvent:
    chunks: list[RetrievedChunk]


@dataclass(frozen=True, slots=True)
class TokenEvent:
    text: str


@dataclass(frozen=True, slots=True)
class DoneEvent:
    answer: str
    latency_ms: int
    context_chars: int
    citation_status: CitationStatus
    citation_warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class ErrorEvent:
    message: str


StreamEvent = SourcesEvent | TokenEvent | DoneEvent | ErrorEvent


@dataclass
class Turn:
    role: str
    content: str


@dataclass
class Answer:
    """The collected result of a stream, for non-streaming callers."""

    text: str = ""
    sources: list[RetrievedChunk] = field(default_factory=list)
    latency_ms: int = 0
    error: str | None = None
    citation_status: CitationStatus = "not_applicable"
    citation_warnings: list[str] = field(default_factory=list)


def build_messages(
    question: str, chunks: list[RetrievedChunk], history: list[Turn]
) -> list[BaseMessage]:
    messages: list[BaseMessage] = [SystemMessage(content=build_system_prompt())]
    for turn in history[-MAX_HISTORY_MESSAGES:]:
        messages.append(
            HumanMessage(content=turn.content)
            if turn.role == "user"
            else AIMessage(content=turn.content)
        )
    messages.append(HumanMessage(content=build_query_envelope(question, chunks)))
    return messages


async def answer_stream(
    services: Services,
    question: str,
    history: list[Turn] | None = None,
    top_k: int | None = None,
) -> AsyncIterator[StreamEvent]:
    started = time.perf_counter()
    history = history or []

    try:
        # Retrieval is synchronous (Qdrant client and optional local reranker), so it
        # goes to a worker thread rather than stalling the event loop.
        retrieval_query = build_retrieval_query(question, history)
        chunks = await anyio.to_thread.run_sync(lambda: retrieve(services, retrieval_query, top_k))
        chunks = _within_context_budget(chunks, services.settings.max_context_chars)
    except Exception:
        # Anything from the vector store or the reranker: report it as a stream
        # event rather than as a dead connection.
        logger.exception("retrieval failed")
        yield ErrorEvent(message="Retrieval is temporarily unavailable")
        return

    yield SourcesEvent(chunks=chunks)

    if not chunks:
        corpus_size = await anyio.to_thread.run_sync(services.registry.count)
        empty_answer = (
            insufficient_evidence_answer(question) if corpus_size else no_context_answer(question)
        )
        yield TokenEvent(text=empty_answer)
        yield DoneEvent(
            answer=empty_answer,
            latency_ms=_elapsed_ms(started),
            context_chars=0,
            citation_status="not_applicable",
        )
        return

    messages = build_messages(question, chunks, history)
    collected: list[str] = []

    try:
        async for chunk in services.chat.astream(messages):
            text = _text_of(chunk)
            if text:
                collected.append(text)
                yield TokenEvent(text=text)
    except Exception:
        # A provider timing out mid-stream still has to reach the user.
        logger.exception("generation failed")
        yield ErrorEvent(message="Answer generation is temporarily unavailable")
        return

    answer_text = "".join(collected)
    citation_status, citation_warnings = audit_citations(answer_text, chunks=chunks)
    yield DoneEvent(
        answer=answer_text,
        latency_ms=_elapsed_ms(started),
        context_chars=sum(len(chunk.text) for chunk in chunks),
        citation_status=citation_status,
        citation_warnings=citation_warnings,
    )


async def answer(
    services: Services,
    question: str,
    history: list[Turn] | None = None,
    top_k: int | None = None,
) -> Answer:
    """Consume the stream and return the whole answer at once."""
    result = Answer()
    async for event in answer_stream(services, question, history, top_k):
        if isinstance(event, SourcesEvent):
            result.sources = event.chunks
        elif isinstance(event, DoneEvent):
            result.text = event.answer
            result.latency_ms = event.latency_ms
            result.citation_status = event.citation_status
            result.citation_warnings = event.citation_warnings
        elif isinstance(event, ErrorEvent):
            result.error = event.message
    return result


def _text_of(chunk: BaseMessage) -> str:
    content = chunk.content
    if isinstance(content, str):
        return content
    # Some providers stream content as a list of typed blocks.
    return "".join(part.get("text", "") for part in content if isinstance(part, dict))


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def build_retrieval_query(question: str, history: list[Turn]) -> str:
    """Resolve short follow-ups with the most recent user question."""
    if len(question) > 300 or _FOLLOW_UP.search(question.strip()) is None:
        return question
    previous = next((turn.content for turn in reversed(history) if turn.role == "user"), "")
    if not previous:
        return question
    return f"Previous user question: {previous[-2_000:]}\nCurrent question: {question}"


def _within_context_budget(chunks: list[RetrievedChunk], max_chars: int) -> list[RetrievedChunk]:
    selected: list[RetrievedChunk] = []
    used = 0
    for chunk in chunks:
        if used + len(chunk.text) > max_chars:
            continue
        selected.append(chunk)
        used += len(chunk.text)
    return selected


def audit_citations(
    answer: str,
    source_count: int | None = None,
    *,
    chunks: list[RetrievedChunk] | None = None,
) -> tuple[CitationStatus, list[str]]:
    """Validate citation shape plus high-confidence grounding signals.

    This deliberately does not pretend to prove semantic entailment. It does
    catch three objective failures locally: unknown sources, uncited sentences,
    and cited numbers or verbatim quotes absent from the referenced excerpts.
    Deeper faithfulness remains the job of the optional evaluation judge.
    """
    if chunks is not None:
        source_count = len(chunks)
    if source_count is None:
        raise TypeError("source_count or chunks is required")
    if source_count == 0:
        return "not_applicable", []
    references = [int(match.group(1)) for match in _CITATION.finditer(answer)]
    invalid = sorted({reference for reference in references if not 1 <= reference <= source_count})
    if invalid:
        return "invalid", [f"Answer references unavailable source(s): {invalid}"]
    if not references:
        return "missing", ["The model returned no source citations"]

    claims = _answer_claims(answer)
    uncited = [claim for claim in claims if _CITATION.search(claim) is None]
    if uncited:
        return "missing", [f"Answer contains {len(uncited)} uncited sentence(s)"]

    if chunks is not None:
        unsupported: list[str] = []
        for claim in claims:
            cited = {int(match.group(1)) for match in _CITATION.finditer(claim)}
            source_text = "\n".join(chunks[index - 1].text for index in cited).casefold()
            plain_claim = _CITATION.sub("", claim)
            numbers = set(re.findall(r"(?<!\w)\d[\d.,/%:-]*(?!\w)", plain_claim))
            # Straight and curly quotes are matched as distinct pairs, so a
            # closing quote cannot pair with the next sentence's opening one
            # and manufacture a "quote" no source contains.
            quotes = {
                (straight or curly).strip().casefold()
                for straight, curly in re.findall(r'"([^"]{4,})"|“([^”]{4,})”', plain_claim)
            }
            missing_numbers = sorted(
                number for number in numbers if number.casefold() not in source_text
            )
            missing_quotes = sorted(quote for quote in quotes if quote not in source_text)
            if missing_numbers or missing_quotes:
                unsupported.append(claim[:120])
        if unsupported:
            return "invalid", [
                f"Answer contains {len(unsupported)} cited sentence(s) with "
                "numbers or quotes absent from their sources"
            ]
    return "valid", []


def _answer_claims(answer: str) -> list[str]:
    """Split prose and list items into non-empty sentences for citation checks."""
    return [
        claim.strip(" \t-*#")
        for claim in re.split(r"(?<=[.!?])(?:\s+|$)|\n+", answer)
        if re.search(r"[\w\d]", claim, flags=re.UNICODE)
    ]
