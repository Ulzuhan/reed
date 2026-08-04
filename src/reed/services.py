"""Wiring: one container holding the objects the API, CLI and evaluator share.

Everything expensive is built lazily, so starting the server never blocks on a
network call and the `fake` profile stays instant.
"""

from __future__ import annotations

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
        self._chat: BaseChatModel | None = None
        self._embeddings: Embeddings | None = None
        self._registry: DocumentRegistry | None = None
        self._qdrant: QdrantClient | None = None
        self._vectorstore: QdrantVectorStore | None = None
        self._reranker: TextCrossEncoder | None = None

    @property
    def chat(self) -> BaseChatModel:
        if self._chat is None:
            from reed.providers import build_chat_model

            self._chat = build_chat_model(self.settings)
            logger.info("chat model ready: %s", self.settings.chat_model_name)
        return self._chat

    @property
    def embeddings(self) -> Embeddings:
        if self._embeddings is None:
            from reed.providers import build_embeddings

            self._embeddings = build_embeddings(self.settings)
            logger.info("embedding model ready: %s", self.settings.embed_model_name)
        return self._embeddings

    @property
    def registry(self) -> DocumentRegistry:
        if self._registry is None:
            self._registry = DocumentRegistry(self.settings.registry_path)
        return self._registry

    @property
    def qdrant(self) -> QdrantClient:
        if self._qdrant is None:
            from reed.rag.vectorstore import get_qdrant_client

            self._qdrant = get_qdrant_client(self.settings)
        return self._qdrant

    @property
    def vectorstore(self) -> QdrantVectorStore:
        """The hybrid store, with its collection created and validated."""
        if self._vectorstore is None:
            from reed.providers import embedding_dimension
            from reed.rag.vectorstore import (
                build_sparse_embeddings,
                build_vectorstore,
                ensure_collection,
            )

            ensure_collection(self.qdrant, self.settings, embedding_dimension(self.embeddings))
            self._vectorstore = build_vectorstore(
                self.qdrant,
                self.settings,
                self.embeddings,
                build_sparse_embeddings(self.settings),
            )
        return self._vectorstore

    @property
    def reranker(self) -> TextCrossEncoder:
        """Cross-encoder for reranking, downloaded on first use only."""
        if self._reranker is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            logger.info("loading reranker %s", self.settings.rerank_model)
            self._reranker = TextCrossEncoder(model_name=self.settings.rerank_model)
        return self._reranker

    def close(self) -> None:
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
