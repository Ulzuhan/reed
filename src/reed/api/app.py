"""FastAPI application factory."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from reed import __version__
from reed.api.routes import documents, health
from reed.config import Settings
from reed.log import get_logger
from reed.services import Services, build_services

logger = get_logger(__name__)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

DESCRIPTION = """
Ask questions about your own documents and get answers with citations.

Upload PDF, Markdown or plain text files, then query them through
`/v1/ask` — with hybrid retrieval (dense + BM25) and token streaming over
Server-Sent Events. Runs on the OpenAI API or fully offline via Ollama.
"""


def create_app(settings: Settings | None = None) -> FastAPI:
    services = build_services(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.services = services
        logger.info(
            "Reed %s starting — profile=%s chat=%s embeddings=%s",
            __version__,
            services.settings.profile,
            services.settings.chat_model_name,
            services.settings.embed_model_name,
        )
        try:
            yield
        finally:
            services.close()
            logger.info("Reed stopped")

    app = FastAPI(
        title="Reed",
        summary="Reed has read your documents.",
        description=DESCRIPTION,
        version=__version__,
        lifespan=lifespan,
    )
    # Available before startup too, so tests and tooling can reach it directly.
    app.state.services = services

    origins = services.settings.cors_origin_list
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    app.include_router(health.router)
    app.include_router(documents.router)
    _mount_ui(app)
    return app


def _mount_ui(app: FastAPI) -> None:
    if not STATIC_DIR.is_dir():
        return

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")


def get_app_services(app: FastAPI) -> Services:
    return app.state.services  # type: ignore[no-any-return]
