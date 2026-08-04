"""Wiring: one container holding the objects the API, CLI and evaluator share.

Everything expensive is built lazily, so starting the server never blocks on a
network call and the `fake` profile stays instant.
"""

from __future__ import annotations

import threading
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


class Services:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        # Background ingestion runs on worker threads while requests are served,
        # so two callers can reach an unbuilt dependency at the same moment. The
        # collection would then be created twice.
        self._lock = threading.Lock()
        self._chat: BaseChatModel | None = None
        self._embeddings: Embeddings | None = None
        self._registry: DocumentRegistry | None = None
        self._qdrant: QdrantClient | None = None
        self._vectorstore: QdrantVectorStore | None = None
        self._reranker: TextCrossEncoder | None = None

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
            return self._build_embeddings()

    @property
    def registry(self) -> DocumentRegistry:
        with self._lock:
            if self._registry is None:
                self._registry = DocumentRegistry(self.settings.registry_path)
            return self._registry

    @property
    def qdrant(self) -> QdrantClient:
        with self._lock:
            return self._build_qdrant()

    @property
    def vectorstore(self) -> QdrantVectorStore:
        """The hybrid store, with its collection created and validated."""
        with self._lock:
            if self._vectorstore is None:
                from reed.providers import embedding_dimension
                from reed.rag.vectorstore import (
                    build_sparse_embeddings,
                    build_vectorstore,
                    ensure_collection,
                )

                client = self._build_qdrant()
                embeddings = self._build_embeddings()
                ensure_collection(client, self.settings, embedding_dimension(embeddings))
                self._vectorstore = build_vectorstore(
                    client,
                    self.settings,
                    embeddings,
                    build_sparse_embeddings(self.settings),
                )
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

    # Unlocked builders, for callers that already hold the lock.

    def _build_embeddings(self) -> Embeddings:
        if self._embeddings is None:
            from reed.providers import build_embeddings

            self._embeddings = build_embeddings(self.settings)
            logger.info("embedding model ready: %s", self.settings.embed_model_name)
        return self._embeddings

    def _build_qdrant(self) -> QdrantClient:
        if self._qdrant is None:
            from reed.rag.vectorstore import get_qdrant_client

            self._qdrant = get_qdrant_client(self.settings)
        return self._qdrant

    def close(self) -> None:
        with self._lock:
            if self._registry is not None:
                self._registry.close()
                self._registry = None
            if self._qdrant is not None:
                # Releases the file lock held by the embedded backend.
                self._qdrant.close()
                self._qdrant = None
            self._vectorstore = None


def build_services(settings: Settings | None = None) -> Services:
    settings = settings or get_settings()
    setup_logging(settings.log_level)
    settings.validate_ready()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    return Services(settings)
