"""FastAPI application factory."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path

import anyio.to_thread
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from reed import __version__
from reed.api.routes import ask, documents, health
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
        if services.settings.host not in {"127.0.0.1", "localhost", "::1"} and not (
            services.settings.api_key
        ):
            logger.warning(
                "server is configured for a non-loopback host without REED_API_KEY; "
                "restrict the network binding or configure authentication"
            )
        await anyio.to_thread.run_sync(_open_vector_store, services)
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
            allow_credentials="*" not in origins,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.middleware("http")
    async def security_headers(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        if request.url.path == "/" or request.url.path.startswith("/static/"):
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; script-src 'self'; style-src 'self'; "
                "img-src 'self' data:; connect-src 'self'; object-src 'none'; "
                "base-uri 'none'; frame-ancestors 'none'; form-action 'self'"
            )
        return response

    app.include_router(health.router)
    app.include_router(documents.router)
    app.include_router(ask.router)
    _mount_ui(app)
    return app


def _open_vector_store(services: Services) -> None:
    """Touch the collection at startup so misconfiguration surfaces here.

    A collection built with a different embedding model can never work, so that
    fails the boot outright. A provider that is merely unreachable does not:
    the server keeps serving, `/ready` reports it, and the flag clears as soon
    as the store opens — the model may well come back without a restart.
    """
    from reed.rag.vectorstore import CollectionMismatchError

    interrupted_records = services.registry.list_by_status(
        {"pending", "processing", "error", "deleting"}
    )
    for record in interrupted_records:
        services.schedule_vector_cleanup(record.id)
    interrupted = services.registry.fail_interrupted(
        "interrupted by a restart — re-upload the file"
    )
    if interrupted:
        logger.warning("%d document(s) were left mid-ingestion by a restart", interrupted)
    interrupted_deletions = services.registry.fail_interrupted_deletions(
        "deletion was interrupted by a restart — retry deletion"
    )
    if interrupted_deletions:
        logger.warning(
            "%d document deletion(s) were interrupted by a restart",
            interrupted_deletions,
        )

    try:
        _ = services.vectorstore
        services.flush_pending_vector_cleanup()
    except CollectionMismatchError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("vector store not ready at startup: %s", exc)
        services.startup_error = f"{type(exc).__name__}: {exc}"


def _mount_ui(app: FastAPI) -> None:
    if not STATIC_DIR.is_dir():
        return

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")


def get_app_services(app: FastAPI) -> Services:
    return app.state.services  # type: ignore[no-any-return]
