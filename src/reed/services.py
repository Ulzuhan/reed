"""Wiring: one container holding the objects the API, CLI and evaluator share.

Everything expensive is built lazily, so starting the server never blocks on a
network call and the `fake` profile stays instant.
"""

from __future__ import annotations

import asyncio
import queue
import threading
import time
from collections import defaultdict, deque
from contextlib import nullcontext
from typing import TYPE_CHECKING

from reed.config import Settings, get_settings
from reed.ingest.registry import DocumentRegistry
from reed.log import get_logger, setup_logging
from reed.observability import RuntimeMetrics

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings
    from langchain_core.language_models import BaseChatModel
    from langchain_qdrant import QdrantVectorStore
    from langchain_qdrant.sparse_embeddings import SparseEmbeddings
    from qdrant_client import QdrantClient

    from reed.rerankers import Reranker

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
        self._reranker_lock = threading.Lock()
        self._cleanup_lock = threading.Lock()
        self._cleanup_flush_lock = threading.Lock()
        self._bootstrap_lock = threading.Lock()
        self._bootstrap_finished = threading.Event()
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
        self._retrieval_stores: dict[str, QdrantVectorStore] = {}
        self._sparse_embeddings: SparseEmbeddings | None = None
        self._active_collection: str | None = None
        self._reranker: Reranker | None = None
        self._pending_vector_cleanup: dict[str, int] = {}
        self._bootstrap_thread: threading.Thread | None = None
        self._bootstrap_failures = 0
        self._bootstrap_retry_at = 0.0
        self._bootstrap_fatal_error: Exception | None = None
        self._chat_probe_lock = threading.Lock()
        self._chat_probe_at = 0.0
        self._chat_probe_result = "unavailable"
        self._closing = False
        self._ingestion_queue: queue.Queue[str] = queue.Queue(
            maxsize=settings.max_queued_ingestions
        )
        self._ingestion_claims_lock = threading.Lock()
        self._ingestion_claims: set[str] = set()
        self._ingestion_stop = threading.Event()
        self._ingestion_workers: list[threading.Thread] = []
        self.ingestion_access = threading.BoundedSemaphore(settings.max_concurrent_ingestions)
        self.ask_access = asyncio.Semaphore(settings.max_concurrent_asks)
        self.rate_limiter = SlidingWindowRateLimiter()
        self.metrics = RuntimeMetrics()
        # Private diagnostic for readiness; public responses expose only a
        # stable unavailable state while the detailed cause stays in logs.
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
                from reed.model_identity import resolve_embedding_identity
                from reed.providers import embedding_dimension
                from reed.rag.vectorstore import (
                    build_sparse_embeddings,
                    build_vectorstore,
                    collection_fingerprint,
                    ensure_collection,
                )

                client = self.qdrant
                embeddings = self.embeddings
                dimension = embedding_dimension(embeddings)
                identity = resolve_embedding_identity(self.settings)
                fingerprint = collection_fingerprint(self.settings, dimension, identity)
                sparse = build_sparse_embeddings(self.settings)
                self._sparse_embeddings = sparse
                with self.vector_access:
                    active = self.registry.active_generation(self.settings.collection)
                    if active is None:
                        # No generation registered yet: either a fresh install
                        # or an upgrade. A collection that already exists must
                        # match the fingerprint; ensure_collection refuses it
                        # otherwise and names the reindex as the remedy.
                        collection_name = self.settings.collection
                        ensure_collection(
                            client,
                            self.settings,
                            dimension,
                            collection_name=collection_name,
                            fingerprint=fingerprint,
                        )
                        active = self.registry.create_generation(
                            logical_collection=self.settings.collection,
                            physical_collection=collection_name,
                            fingerprint=fingerprint,
                            status="active",
                        )
                    else:
                        collection_name = active.physical_collection
                        ensure_collection(
                            client,
                            self.settings,
                            dimension,
                            collection_name=collection_name,
                            fingerprint=fingerprint,
                        )
                    # Reads the collection config, which the local backend
                    # serves from the same structures a write mutates.
                    self._active_collection = collection_name
                    self._vectorstore = build_vectorstore(
                        client,
                        self.settings,
                        embeddings,
                        sparse,
                        collection_name=collection_name,
                        retrieval_mode="hybrid",
                    )
            return self._vectorstore

    def retrieval_store(self, mode: str | None = None) -> QdrantVectorStore:
        """Return a query view over the same dense+sparse physical index."""
        selected = mode or self.settings.retrieval_mode
        query_mode = "hybrid" if selected == "hybrid_rerank" else selected
        base = self.vectorstore
        if query_mode == "hybrid":
            return base
        with self._store_lock:
            cached = self._retrieval_stores.get(query_mode)
            if cached is not None:
                return cached
            from reed.rag.vectorstore import build_vectorstore

            assert self._sparse_embeddings is not None
            cached = build_vectorstore(
                self.qdrant,
                self.settings,
                self.embeddings,
                self._sparse_embeddings,
                collection_name=self.active_collection_name,
                retrieval_mode=query_mode,
            )
            self._retrieval_stores[query_mode] = cached
            return cached

    @property
    def sparse_embeddings(self) -> SparseEmbeddings:
        if self._sparse_embeddings is None:
            _ = self.vectorstore
        assert self._sparse_embeddings is not None
        return self._sparse_embeddings

    def probe_chat_readiness(self) -> str:
        """Cached liveness check for the configured chat provider."""
        if self.settings.profile in {"fake", "openai"}:
            # OpenAI has no cheap model-specific health endpoint; request
            # failures remain visible on /ask. Ollama can be checked locally.
            return "ok"
        now = time.monotonic()
        with self._chat_probe_lock:
            if now - self._chat_probe_at < self.settings.readiness_chat_ttl_seconds:
                return self._chat_probe_result
            try:
                from reed.model_identity import ollama_tags

                payload = ollama_tags(self.settings)
                rows = payload.get("models")
                configured = self.settings.ollama_chat_model
                wanted = {
                    configured,
                    f"{configured}:latest" if ":" not in configured else configured,
                }
                names = (
                    {str(row.get("name", "")) for row in rows if isinstance(row, dict)}
                    if isinstance(rows, list)
                    else set()
                )
                result = "ok" if wanted.intersection(names) else "unavailable"
            except Exception as exc:  # noqa: BLE001 — readiness reports rather than raises
                logger.warning("chat readiness probe failed: %s", exc)
                result = "unavailable"
            self._chat_probe_at = now
            self._chat_probe_result = result
            return result

    @property
    def active_collection_name(self) -> str:
        """Physical collection currently selected by the registry."""
        if self._active_collection is not None:
            return self._active_collection
        active = self.registry.active_generation(self.settings.collection)
        return active.physical_collection if active is not None else self.settings.collection

    def select_generation(self, collection_name: str) -> None:
        """Invalidate cached store objects after an atomic registry switch."""
        with self._store_lock:
            self._vectorstore = None
            self._retrieval_stores.clear()
            self._active_collection = collection_name
        with self._bootstrap_lock:
            self._bootstrap_fatal_error = None
            self.startup_error = None

    @property
    def vectorstore_ready(self) -> bool:
        return self._vectorstore is not None and self.startup_error is None

    @property
    def bootstrap_fatal_error(self) -> Exception | None:
        with self._bootstrap_lock:
            return self._bootstrap_fatal_error

    def start_vectorstore_bootstrap(self) -> bool:
        """Start one non-blocking vector-store initialisation attempt.

        Calls made while an attempt is active or its retry cooldown has not
        elapsed are no-ops. This keeps a frequent readiness probe from filling
        the server threadpool when a provider is black-holed.
        """
        with self._bootstrap_lock:
            if self._closing or self._bootstrap_fatal_error is not None:
                return False
            if self.vectorstore_ready:
                self._bootstrap_finished.set()
                return False
            if self._bootstrap_thread is not None and self._bootstrap_thread.is_alive():
                return False
            if time.monotonic() < self._bootstrap_retry_at:
                return False

            self.startup_error = "initializing"
            self._bootstrap_finished.clear()
            thread = threading.Thread(
                target=self._run_vectorstore_bootstrap,
                name="reed-vector-bootstrap",
                daemon=True,
            )
            self._bootstrap_thread = thread
            thread.start()
            return True

    def wait_for_vectorstore_bootstrap(self, timeout: float) -> bool:
        return self._bootstrap_finished.wait(timeout)

    def note_vectorstore_failure(self, exc: Exception) -> None:
        """Invalidate a remote store that failed a live readiness probe."""
        with self._store_lock:
            self._vectorstore = None
            self._retrieval_stores.clear()
            self._sparse_embeddings = None
        with self._bootstrap_lock:
            self.startup_error = f"{type(exc).__name__}: {exc}"
            self._bootstrap_retry_at = (
                time.monotonic() + self.settings.readiness_retry_initial_seconds
            )

    def _run_vectorstore_bootstrap(self) -> None:
        from reed.rag.vectorstore import CollectionMismatchError

        try:
            _ = self.vectorstore
            self.flush_pending_vector_cleanup()
        except CollectionMismatchError as exc:
            logger.error("vector collection is incompatible: %s", exc)
            with self._bootstrap_lock:
                self._bootstrap_fatal_error = exc
                self.startup_error = f"{type(exc).__name__}: {exc}"
        except Exception as exc:  # noqa: BLE001 — retried with bounded backoff
            with self._bootstrap_lock:
                self._bootstrap_failures += 1
                delay = min(
                    self.settings.readiness_retry_initial_seconds
                    * (2 ** (self._bootstrap_failures - 1)),
                    self.settings.readiness_retry_max_seconds,
                )
                self._bootstrap_retry_at = time.monotonic() + delay
                self.startup_error = f"{type(exc).__name__}: {exc}"
            logger.warning("vector store bootstrap failed; retrying in %.1fs: %s", delay, exc)
        else:
            with self._bootstrap_lock:
                self._bootstrap_failures = 0
                self._bootstrap_retry_at = 0.0
                self.startup_error = None
        finally:
            with self._bootstrap_lock:
                self._bootstrap_thread = None
                closing = self._closing
            self._bootstrap_finished.set()
            if closing:
                self._close_dependencies()

    @property
    def reranker(self) -> Reranker:
        """Cross-encoder for reranking, downloaded on first use only."""
        with self._reranker_lock:
            if self._reranker is None:
                from reed.rerankers import build_reranker

                logger.info("loading reranker %s", self.settings.rerank_model)
                self._reranker = build_reranker(self.settings)
            return self._reranker

    def close(self) -> None:
        self.stop_ingestion_queue()
        with self._bootstrap_lock:
            self._closing = True
            bootstrap = self._bootstrap_thread
        if bootstrap is not None and bootstrap.is_alive():
            bootstrap.join(timeout=1)
            if bootstrap.is_alive():
                logger.warning("leaving an in-flight provider bootstrap to exit with the process")
                return
        self._close_dependencies()

    @property
    def ingestion_queue_full(self) -> bool:
        return self._ingestion_queue.full()

    @property
    def ingestion_queue_size(self) -> int:
        return self._ingestion_queue.qsize()

    def start_ingestion_queue(self) -> None:
        if self._ingestion_workers:
            return
        self._ingestion_stop.clear()
        for index in range(self.settings.max_concurrent_ingestions):
            worker = threading.Thread(
                target=self._run_ingestion_worker,
                name=f"reed-ingestion-{index + 1}",
                daemon=True,
            )
            self._ingestion_workers.append(worker)
            worker.start()
        self.refill_ingestion_queue()

    def enqueue_ingestion(self, doc_id: str) -> bool:
        with self._ingestion_claims_lock:
            if doc_id in self._ingestion_claims:
                return True
            try:
                self._ingestion_queue.put_nowait(doc_id)
            except queue.Full:
                return False
            self._ingestion_claims.add(doc_id)
        return True

    def refill_ingestion_queue(self) -> None:
        """Load durable queued work after restart or after a worker frees space."""
        queued_ids = {record.id for record in self.registry.list_by_status({"queued"})}
        with self._ingestion_claims_lock:
            claimed = set(self._ingestion_claims)
        for doc_id in sorted(queued_ids - claimed):
            if not self.enqueue_ingestion(doc_id):
                break

    def stop_ingestion_queue(self) -> None:
        self._ingestion_stop.set()
        for worker in self._ingestion_workers:
            worker.join(timeout=1)
        if any(worker.is_alive() for worker in self._ingestion_workers):
            logger.warning("leaving in-flight ingestion workers to exit with the process")
        self._ingestion_workers.clear()

    def _run_ingestion_worker(self) -> None:
        from reed.ingest.pipeline import process_document

        while not self._ingestion_stop.is_set():
            try:
                doc_id = self._ingestion_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                process_document(self, doc_id)
            finally:
                self._ingestion_queue.task_done()
                with self._ingestion_claims_lock:
                    self._ingestion_claims.discard(doc_id)
                self.refill_ingestion_queue()

    def _close_dependencies(self) -> None:
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
            self._active_collection = None
            self._retrieval_stores.clear()
            self._sparse_embeddings = None

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
                collection_name = self.active_collection_name
                if client.collection_exists(collection_name):
                    for doc_id in pending:
                        delete_document_points(
                            client,
                            self.settings,
                            doc_id,
                            collection_name=collection_name,
                        )
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
