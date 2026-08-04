"""Ask a question: streamed over SSE, or in one JSON response."""

from __future__ import annotations

import asyncio
import contextlib
import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from reed.api.deps import ServicesDep, enforce_rate_limit, require_api_key
from reed.api.schemas import AskRequest, AskResponse, Source
from reed.api.sse import PING, SSE_HEADERS, sse_event
from reed.log import get_logger
from reed.rag.chain import (
    DoneEvent,
    ErrorEvent,
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
            excerpt=chunk.text,
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
                "citation_status": event.citation_status,
                "citation_warnings": event.citation_warnings,
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

    queue: asyncio.Queue[object] = asyncio.Queue(maxsize=services.settings.sse_queue_size)
    history = [Turn(role=m.role, content=m.content) for m in request.history]

    async def produce() -> None:
        cancelled = False
        try:
            async with asyncio.timeout(services.settings.provider_timeout_seconds * 2):
                async for event in answer_stream(
                    services, request.question, history, request.top_k
                ):
                    await queue.put(event)
        except asyncio.CancelledError:
            cancelled = True
            raise
        except TimeoutError:
            logger.warning("ask request %s timed out", request_id)
            await queue.put(ErrorEvent(message="The answer timed out"))
        except Exception:
            # answer_stream reports its own failures as events; anything landing
            # here would otherwise end the stream silently, with no error and no
            # log, because the task's exception is never retrieved.
            logger.exception("ask stream producer failed")
            await queue.put(ErrorEvent(message="The answer could not be completed"))
        finally:
            if cancelled:
                # The consumer has gone away and may have left a full queue.
                # Never block task cancellation waiting for a reader that no
                # longer exists.
                with contextlib.suppress(asyncio.QueueFull):
                    queue.put_nowait(_STREAM_END)
            else:
                await queue.put(_STREAM_END)

    async with services.ask_access:
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
            with contextlib.suppress(asyncio.CancelledError):
                await producer


@router.post(
    "/ask",
    # Two shapes behind one route: an SSE stream, or a single JSON body when
    # `stream` is false. OpenAPI documents the JSON one.
    response_model=None,
    responses={200: {"model": AskResponse, "description": "Answer with its sources"}},
)
async def ask(
    services: ServicesDep,
    http_request: Request,
    request: AskRequest,
) -> StreamingResponse | AskResponse:
    enforce_rate_limit(
        http_request,
        services,
        scope="ask",
        limit=services.settings.ask_rate_limit_per_minute,
    )
    if request.stream:
        return StreamingResponse(
            _sse_body(services, request),
            media_type="text/event-stream",
            headers=SSE_HEADERS,
        )

    history = [Turn(role=m.role, content=m.content) for m in request.history]
    try:
        async with (
            services.ask_access,
            asyncio.timeout(services.settings.provider_timeout_seconds * 2),
        ):
            result = await answer(services, request.question, history, request.top_k)
    except TimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="The answer timed out",
        ) from exc
    if result.error:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=result.error,
        )

    return AskResponse(
        answer=result.text,
        sources=_to_sources(result.sources),
        latency_ms=result.latency_ms,
        citation_status=result.citation_status,
        citation_warnings=result.citation_warnings,
    )
