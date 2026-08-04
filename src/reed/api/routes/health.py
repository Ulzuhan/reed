"""Liveness and configuration probe."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Response, status

from reed import __version__
from reed.api.deps import ServicesDep
from reed.api.schemas import HealthResponse
from reed.log import get_logger
from reed.services import Services

router = APIRouter(tags=["health"])
logger = get_logger(__name__)


def _probe_vector_store(services: Services) -> str:
    """Ask the vector store whether it is alive — when that is free to do.

    A Qdrant server can die long after startup (a restarted container, a
    partition), and only a live probe notices. Embedded Qdrant is skipped: every
    operation on it is serialised, so probing here would park /ready behind
    whatever large upload happens to be embedding, and there is no separate
    process to lose anyway.
    """
    if not services.settings.qdrant_url:
        return "ok"

    try:
        services.qdrant.get_collections()
    except Exception as exc:  # noqa: BLE001 — health must report, never raise
        logger.warning("Qdrant readiness probe failed: %s", exc)
        return "unavailable"
    return "ok"


@router.get("/health", response_model=HealthResponse)
def health(services: ServicesDep) -> HealthResponse:
    # Liveness deliberately touches no database, model or network dependency.
    # Orchestrators need a fast answer even while readiness is degraded.
    return _response(
        services,
        health_status="ok",
        vector_store="not_checked",
        documents=None,
    )


@router.get(
    "/ready",
    response_model=HealthResponse,
    responses={503: {"model": HealthResponse, "description": "Dependencies are unavailable"}},
)
def ready(services: ServicesDep, response: Response) -> HealthResponse:
    result = _health_response(services, retry_startup=True)
    if result.status != "ok":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return result


def _health_response(services: Services, *, retry_startup: bool) -> HealthResponse:
    documents: int | None = None
    try:
        documents = services.registry.count()
        if retry_startup and services.startup_error:
            try:
                _ = services.vectorstore
                services.flush_pending_vector_cleanup()
            except Exception as exc:  # noqa: BLE001 — readiness reports below
                logger.warning("vector store recovery probe failed: %s", exc)
        vector_store = "unavailable" if services.startup_error else _probe_vector_store(services)
    except Exception as exc:  # noqa: BLE001 — health must report, never raise
        logger.warning("health probe failed: %s", exc)
        vector_store = "unavailable"

    health_status: Literal["ok", "degraded"] = "ok" if vector_store == "ok" else "degraded"

    return _response(
        services,
        health_status=health_status,
        vector_store=vector_store,
        documents=documents,
    )


def _response(
    services: Services,
    *,
    health_status: Literal["ok", "degraded"],
    vector_store: str,
    documents: int | None,
) -> HealthResponse:
    settings = services.settings
    disclose_details = not bool(settings.api_key)
    return HealthResponse(
        status=health_status,
        version=__version__ if disclose_details else None,
        profile=settings.profile if disclose_details else None,
        chat_model=settings.chat_model_name if disclose_details else None,
        embed_model=settings.embed_model_name if disclose_details else None,
        vector_store=vector_store,
        documents=documents if disclose_details else None,
    )
