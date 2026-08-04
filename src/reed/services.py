"""Wiring: one container holding the objects the API, CLI and evaluator share.

Everything expensive is built lazily, so starting the server never blocks on a
network call and the `fake` profile stays instant.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections import defaultdict, deque
from contextlib import nullcontext
from typing import TYPE_CHECKING

from reed.config import Settings, get_settings
from reed.ingest.registry import DocumentRegistry
from reed.log import get_logger, setup_logging

if TYPE_CHECKING:
    from fastembed.rerank.cross_encoder import TextCrossEncoder
    from langchain_core.embeddings import Embeddings
    from langchain_core.language_models import BaseChatModel
    from langchain_qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient

logger = get_logger(__name__)


class SlidingWindowRateLimiter:
    """Small in-process limiter for the costly public endpoints.

    Deployments with multiple workers should still enforce a shared limit at
    their reverse proxy. This limiter makes the safe single-process default
    resistant to accidental or trivial request floods.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events: dict[tuple[str, str], deque[float]] = defaultdict(deque)
        self._checks = 0

    def allow(self, scope: str, client: str, limit: int, *, now: float | None = None) -> bool:
        if limit == 0:
            return True
        current = time.monotonic() if now is None else now
        cutoff = current - 60.0
        key = (scope, client)
        with self._lock:
            self._checks += 1
            if self._checks % 1_000 == 0:
                self._discard_idle(cutoff)
            events = self._events[key]
            while events and events[0] <= cutoff:
                events.popleft()
            if len(events) >= limit:
                return False
            events.append(current)
            return True

    def _discard_idle(self, cutoff: float) -> None:
        for key, events in list(self._events.items()):
            while events and events[0] <= cutoff:
                events.popleft()
            if not events:
                del self._events[key]


class Services:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        # Background ingestion runs on worker threads while requests are served,
        # so two callers can reach an unbuilt dependency at the same moment.
        # Separate locks keep a slow model load from blocking /health, which
        # only needs the registry.
        self._lock = threading.Lock()
        self._store_lock = threading.Lock()
        self._cleanup_lock = threading.Lock()
        self._cleanup_flush_lock = threading.Lock()
        # Embedded Qdrant has no internal locking — concurrent writes corrupt
        # the store. Every vector operation goes through this; with a Qdrant
        # server, its own concurrency control applies instead.
        #
        # Lock order is _store_lock > vector_access > _lock. In practice that
        # means: resolve `services.vectorstore` BEFORE entering vector_access,
        # never inside it, or a build in flight on another thread deadlocks
        # against you.
        self.vector_access = threading.RLock() if not settings.qdrant_url else nullcontext()
        self._chat: BaseChatModel | None = None
        self._embeddings: Embeddings | None = None
        self._registry: DocumentRegistry | None = None
        self._qdrant: QdrantClient | None = None
        self._vectorstore: QdrantVectorStore | None = None
        self._reranker: TextCrossEncoder | None = None
        self._pending_vector_cleanup: dict[str, int] = {}
        self.ingestion_access = threading.BoundedSemaphore(settings.max_concurrent_ingestions)
        self.ask_access = asyncio.Semaphore(settings.max_concurrent_asks)
        self.rate_limiter = SlidingWindowRateLimiter()
        # Set when the vector store could not be opened at startup, so /ready
        # can say so instead of reporting ok for a store that serves nothing.
        self.startup_error: str | None = None

    @property
    def chat(self) -> BaseChatModel:
        with self._lock:
            if self._chat is None:
                from reed.providers import build_chat_model

                self._chat = build_chat_model(self.settings)
                logger.info("chat model ready: %s", self.settings.chat_model_name)
            return self._chat

    @property
    def embeddings(self) -> Embeddings:
        with self._lock:
            if self._embeddings is None:
                from reed.providers import build_embeddings

                self._embeddings = build_embeddings(self.settings)
                logger.info("embedding model ready: %s", self.settings.embed_model_name)
            return self._embeddings

    @property
    def registry(self) -> DocumentRegistry:
        with self._lock:
            if self._registry is None:
                self._registry = DocumentRegistry(self.settings.registry_path)
            return self._registry

    @property
    def qdrant(self) -> QdrantClient:
        with self._lock:
            if self._qdrant is None:
                from reed.rag.vectorstore import get_qdrant_client

                self._qdrant = get_qdrant_client(self.settings)
            return self._qdrant

    @property
    def vectorstore(self) -> QdrantVectorStore:
        """The hybrid store, with its collection created and validated.

        Built under its own lock: probing the embedding dimension is a live
        model call, and a cold Ollama can take tens of seconds. Holding the
        general lock for that would stall `/health` too.
        """
        with self._store_lock:
            if self._vectorstore is None:
                from reed.providers import embedding_dimension
                from reed.rag.vectorstore import (
                    build_sparse_embeddings,
                    build_vectorstore,
                    ensure_collection,
                )

                client = self.qdrant
                embeddings = self.embeddings
                dimension = embedding_dimension(embeddings)
                sparse = build_sparse_embeddings(self.settings)
                with self.vector_access:
                    ensure_collection(client, self.settings, dimension)
                    # Reads the collection config, which the local backend
                    # serves from the same structures a write mutates.
                    self._vectorstore = build_vectorstore(client, self.settings, embeddings, sparse)
                # Whatever kept the store from opening at startup is over.
                self.startup_error = None
            return self._vectorstore

    @property
    def reranker(self) -> TextCrossEncoder:
        """Cross-encoder for reranking, downloaded on first use only."""
        with self._lock:
            if self._reranker is None:
                from fastembed.rerank.cross_encoder import TextCrossEncoder

                logger.info("loading reranker %s", self.settings.rerank_model)
                self._reranker = TextCrossEncoder(model_name=self.settings.rerank_model)
            return self._reranker

    def close(self) -> None:
        # Lock order as documented above, so shutdown cannot close the client
        # out from under a thread that is mid-upsert.
        with self._store_lock, self.vector_access, self._lock:
            if self._registry is not None:
                self._registry.close()
                self._registry = None
            if self._qdrant is not None:
                # Releases the file lock held by the embedded backend.
                self._qdrant.close()
                self._qdrant = None
            self._vectorstore = None

    def schedule_vector_cleanup(self, doc_id: str) -> None:
        with self._cleanup_lock:
            self._pending_vector_cleanup[doc_id] = self._pending_vector_cleanup.get(doc_id, 0) + 1

    def flush_pending_vector_cleanup(self) -> None:
        """Delete points left by failed or interrupted ingestion attempts.

        Cleanup happens before retrieval and after startup. If Qdrant is still
        unavailable the ids remain queued and the caller fails instead of
        querying a store known to contain uncommitted points.
        """
        with self._cleanup_flush_lock:
            with self._cleanup_lock:
                pending = dict(self._pending_vector_cleanup)
            if not pending:
                return

            from reed.rag.vectorstore import delete_document_points

            client = self.qdrant
            with self.vector_access:
                if client.collection_exists(self.settings.collection):
                    for doc_id in pending:
                        delete_document_points(client, self.settings, doc_id)
            with self._cleanup_lock:
                for doc_id, generation in pending.items():
                    if self._pending_vector_cleanup.get(doc_id) == generation:
                        del self._pending_vector_cleanup[doc_id]


def build_services(settings: Settings | None = None) -> Services:
    settings = settings or get_settings()
    setup_logging(settings.log_level)
    settings.validate_ready()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    return Services(settings)
