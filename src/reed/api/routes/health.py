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
    return HealthResponse(
        status="ok",
        version=__version__,
        profile=settings.profile,
        chat_model=settings.chat_model_name,
        embed_model=settings.embed_model_name,
        vector_store="ok",
    )
