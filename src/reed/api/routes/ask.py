"""Ask a question: streamed over SSE, or in one JSON response."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from reed.api.deps import ServicesDep, require_api_key
from reed.api.schemas import AskRequest, AskResponse, Source
from reed.api.sse import PING, SSE_HEADERS, sse_event
from reed.log import get_logger
from reed.rag.chain import (
    DoneEvent,
    SourcesEvent,
    StreamEvent,
    TokenEvent,
    Turn,
    answer,
    answer_stream,
)
from reed.rag.retriever import RetrievedChunk
from reed.services import Services

logger = get_logger(__name__)

router = APIRouter(prefix="/v1", tags=["ask"], dependencies=[Depends(require_api_key)])

# Long enough to stay quiet on a healthy connection, short enough to beat the
# idle timeout of a typical reverse proxy.
PING_INTERVAL_SECONDS = 15.0

_STREAM_END = object()


def _to_sources(chunks: list[RetrievedChunk]) -> list[Source]:
    """Number the chunks — these are the ``[n]`` markers the model must use."""
    return [
        Source(
            n=number,
            doc_id=chunk.doc_id,
            filename=chunk.filename,
            page=chunk.page,
            section=chunk.section,
            location=chunk.location,
            score=round(chunk.score, 4),
            snippet=chunk.snippet,
        )
        for number, chunk in enumerate(chunks, start=1)
    ]


def _render(event: StreamEvent) -> str:
    if isinstance(event, SourcesEvent):
        sources = [source.model_dump() for source in _to_sources(event.chunks)]
        return sse_event("sources", {"sources": sources})
    if isinstance(event, TokenEvent):
        return sse_event("token", {"t": event.text})
    if isinstance(event, DoneEvent):
        return sse_event(
            "done",
            {
                "answer": event.answer,
                "latency_ms": event.latency_ms,
                "context_chars": event.context_chars,
            },
        )
    return sse_event("error", {"message": event.message})


async def _sse_body(services: Services, request: AskRequest) -> AsyncIterator[str]:
    request_id = f"r-{uuid.uuid4().hex[:12]}"
    yield sse_event(
        "meta",
        {
            "request_id": request_id,
            "profile": services.settings.profile,
            "model": services.settings.chat_model_name,
        },
    )

    queue: asyncio.Queue[object] = asyncio.Queue()
    history = [Turn(role=m.role, content=m.content) for m in request.history]

    async def produce() -> None:
        try:
            async for event in answer_stream(services, request.question, history, request.top_k):
                await queue.put(event)
        finally:
            await queue.put(_STREAM_END)

    producer = asyncio.create_task(produce())
    try:
        while True:
            try:
                # A timeout here only cancels the queue read, never the producer,
                # so a slow first token turns into a heartbeat rather than a gap.
                item = await asyncio.wait_for(queue.get(), timeout=PING_INTERVAL_SECONDS)
            except TimeoutError:
                yield PING
                continue

            if item is _STREAM_END:
                break
            yield _render(item)  # type: ignore[arg-type]
    finally:
        producer.cancel()


@router.post(
    "/ask",
    # Two shapes behind one route: an SSE stream, or a single JSON body when
    # `stream` is false. OpenAPI documents the JSON one.
    response_model=None,
    responses={200: {"model": AskResponse, "description": "Answer with its sources"}},
)
async def ask(services: ServicesDep, request: AskRequest) -> StreamingResponse | AskResponse:
    if request.stream:
        return StreamingResponse(
            _sse_body(services, request),
            media_type="text/event-stream",
            headers=SSE_HEADERS,
        )

    history = [Turn(role=m.role, content=m.content) for m in request.history]
    result = await answer(services, request.question, history, request.top_k)
    if result.error:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=result.error,
        )

    return AskResponse(
        answer=result.text,
        sources=_to_sources(result.sources),
        latency_ms=result.latency_ms,
    )
