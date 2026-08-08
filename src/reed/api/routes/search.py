"""Retrieval without generation: the evidence, and nothing built on top of it."""

from __future__ import annotations

import asyncio
import time

import anyio.to_thread
from fastapi import APIRouter, Depends, HTTPException, status

from reed.api.deps import ServicesDep, require_api_key
from reed.api.schemas import SearchRequest, SearchResponse, to_sources
from reed.log import get_logger
from reed.rag.retriever import retrieve

logger = get_logger(__name__)

router = APIRouter(prefix="/v1", tags=["search"], dependencies=[Depends(require_api_key)])


@router.post("/search", response_model=SearchResponse)
async def search(services: ServicesDep, request: SearchRequest) -> SearchResponse:
    """Return the ranked evidence `/v1/ask` would have answered from.

    The evidence threshold is reported, not applied: a caller whose own model
    does the generation needs the scores to decide with, and abstaining on its
    behalf would hide the very rows it uses to decide. `/v1/ask` still abstains.
    """
    services.metrics.increment("searches_total")
    started = time.perf_counter()
    try:
        # Retrieval is synchronous (Qdrant client and optional local reranker),
        # so it goes to a worker thread rather than stalling the event loop.
        async with asyncio.timeout(services.settings.provider_timeout_seconds):
            chunks = await anyio.to_thread.run_sync(
                lambda: retrieve(services, request.query, request.top_k, apply_threshold=False)
            )
    except TimeoutError as exc:
        services.metrics.increment("search_errors_total")
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Retrieval timed out",
        ) from exc
    except Exception as exc:
        services.metrics.increment("search_errors_total")
        logger.exception("retrieval failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Retrieval is temporarily unavailable",
        ) from exc

    threshold = services.settings.resolved_min_evidence_score
    return SearchResponse(
        sources=to_sources(chunks),
        latency_ms=int((time.perf_counter() - started) * 1000),
        sufficient_evidence=bool(chunks) and (not threshold or chunks[0].score >= threshold),
        min_evidence_score=threshold,
    )
