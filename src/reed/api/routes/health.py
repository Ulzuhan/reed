"""Liveness and configuration probe."""

from __future__ import annotations

from fastapi import APIRouter

from reed import __version__
from reed.api.deps import ServicesDep
from reed.api.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(services: ServicesDep) -> HealthResponse:
    settings = services.settings

    documents: int | None = None
    try:
        # Deliberately does not touch the vector store: with embedded Qdrant
        # every operation is serialised, so probing it here would park /health
        # behind whatever large upload happens to be embedding. Startup records
        # whether the store opened, and ingestion updates it from then on.
        documents = services.registry.count()
        vector_store = services.startup_error or "ok"
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
