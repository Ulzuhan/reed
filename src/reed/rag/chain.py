"""Retrieve, then generate — the two steps, as a stream of typed events.

One generator feeds all three consumers (the SSE endpoint, the plain JSON
endpoint and the evaluation runner), so what the evaluation measures is exactly
what users get.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import anyio.to_thread
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from reed.log import get_logger
from reed.rag.prompts import NO_CONTEXT_ANSWER, build_system_prompt
from reed.rag.retriever import RetrievedChunk, retrieve

if TYPE_CHECKING:
    from reed.services import Services

logger = get_logger(__name__)

# How many prior turns to replay. Enough for follow-ups ("and for contractors?")
# without letting an old topic drown out the current question.
MAX_HISTORY_MESSAGES = 6


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


def build_messages(
    question: str, chunks: list[RetrievedChunk], history: list[Turn]
) -> list[BaseMessage]:
    messages: list[BaseMessage] = [SystemMessage(content=build_system_prompt(chunks))]
    for turn in history[-MAX_HISTORY_MESSAGES:]:
        messages.append(
            HumanMessage(content=turn.content)
            if turn.role == "user"
            else AIMessage(content=turn.content)
        )
    messages.append(HumanMessage(content=question))
    return messages


async def answer_stream(
    services: Services,
    question: str,
    history: list[Turn] | None = None,
    top_k: int | None = None,
) -> AsyncIterator[StreamEvent]:
    started = time.perf_counter()

    try:
        # Retrieval is synchronous (Qdrant client, local ONNX reranker), so it
        # goes to a worker thread rather than stalling the event loop.
        chunks = await anyio.to_thread.run_sync(lambda: retrieve(services, question, top_k))
    except Exception as exc:
        # Anything from the vector store or the reranker: report it as a stream
        # event rather than as a dead connection.
        logger.exception("retrieval failed")
        yield ErrorEvent(message=f"Retrieval failed: {type(exc).__name__}: {exc}")
        return

    yield SourcesEvent(chunks=chunks)

    if not chunks:
        yield TokenEvent(text=NO_CONTEXT_ANSWER)
        yield DoneEvent(
            answer=NO_CONTEXT_ANSWER,
            latency_ms=_elapsed_ms(started),
            context_chars=0,
        )
        return

    messages = build_messages(question, chunks, history or [])
    collected: list[str] = []

    try:
        async for chunk in services.chat.astream(messages):
            text = _text_of(chunk)
            if text:
                collected.append(text)
                yield TokenEvent(text=text)
    except Exception as exc:
        # A provider timing out mid-stream still has to reach the user.
        logger.exception("generation failed")
        yield ErrorEvent(message=f"Generation failed: {type(exc).__name__}: {exc}")
        return

    yield DoneEvent(
        answer="".join(collected),
        latency_ms=_elapsed_ms(started),
        context_chars=sum(len(chunk.text) for chunk in chunks),
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
