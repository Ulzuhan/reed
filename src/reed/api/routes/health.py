"""Liveness and configuration probe."""

from __future__ import annotations

from fastapi import APIRouter

from reed import __version__
from reed.api.deps import ServicesDep
from reed.api.schemas import HealthResponse
from reed.services import Services

router = APIRouter(tags=["health"])


def _probe_vector_store(services: Services) -> str:
    """Ask the vector store whether it is alive — when that is free to do.

    A Qdrant server can die long after startup (a restarted container, a
    partition), and only a live probe notices. Embedded Qdrant is skipped: every
    operation on it is serialised, so probing here would park /health behind
    whatever large upload happens to be embedding, and there is no separate
    process to lose anyway.
    """
    if not services.settings.qdrant_url:
        return "ok"

    try:
        services.qdrant.get_collections()
    except Exception as exc:  # noqa: BLE001 — health must report, never raise
        return f"{type(exc).__name__}: {exc}"
    return "ok"


@router.get("/health", response_model=HealthResponse)
def health(services: ServicesDep) -> HealthResponse:
    settings = services.settings

    documents: int | None = None
    try:
        documents = services.registry.count()
        vector_store = services.startup_error or _probe_vector_store(services)
    except Exception as exc:  # noqa: BLE001 — health must report, never raise
        vector_store = f"{type(exc).__name__}: {exc}"

    status = "ok" if vector_store == "ok" else "degraded"

    return HealthResponse(
        status=status,
        version=__version__,
        profile=settings.profile,
        chat_model=settings.chat_model_name,
        embed_model=settings.embed_model_name,
        vector_store=vector_store,
        documents=documents,
    )
