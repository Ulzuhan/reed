"""Qdrant wiring — embedded folder or remote server, same code either way.

The collection carries two named vectors: a dense one from the embedding model
and a sparse BM25 one. Qdrant fuses them server-side with Reciprocal Rank
Fusion, which is what makes a question phrased in the user's words find a
chunk that uses the document's exact terminology.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from qdrant_client import QdrantClient, models

from reed.config import Settings
from reed.log import get_logger

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings
    from langchain_qdrant import QdrantVectorStore
    from langchain_qdrant.sparse_embeddings import SparseEmbeddings

logger = get_logger(__name__)

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
DOC_ID_PAYLOAD_KEY = "metadata.doc_id"
COMMITTED_PAYLOAD_KEY = "metadata.committed"
COLLECTION_SCHEMA_VERSION = 3


class CollectionMismatchError(RuntimeError):
    """The existing collection was built with a different embedding model."""


def get_qdrant_client(settings: Settings) -> QdrantClient:
    if settings.qdrant_url:
        logger.info("connecting to Qdrant server at %s", settings.qdrant_url)
        return QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
            timeout=math.ceil(settings.provider_timeout_seconds),
        )

    path = settings.qdrant_path
    path.mkdir(parents=True, exist_ok=True)
    logger.info("using embedded Qdrant at %s", path)
    return QdrantClient(
        path=str(path),
        # Reed serialises every embedded operation with `vector_access`, but
        # those operations legitimately run on different worker threads.
        # Making that contract explicit also skips Qdrant's legacy SQLite
        # threading probe, whose temporary connection is not explicitly closed
        # and raises ResourceWarning on Python 3.13.
        force_disable_check_same_thread=True,
    )


def ensure_collection(client: QdrantClient, settings: Settings, dimension: int) -> None:
    """Create the collection, or verify the existing one still fits."""
    if client.collection_exists(settings.collection):
        _assert_dimension_matches(client, settings, dimension)
        _ensure_payload_indexes(client, settings)
        return

    client.create_collection(
        collection_name=settings.collection,
        vectors_config={
            DENSE_VECTOR_NAME: models.VectorParams(
                size=dimension,
                distance=models.Distance.COSINE,
            )
        },
        sparse_vectors_config={
            # IDF is computed by Qdrant itself; BM25 scoring needs it.
            SPARSE_VECTOR_NAME: models.SparseVectorParams(modifier=models.Modifier.IDF)
        },
        metadata={"reed": _collection_fingerprint(settings, dimension)},
    )
    _ensure_payload_indexes(client, settings)
    logger.info("created collection '%s' (dim=%d)", settings.collection, dimension)


def _assert_dimension_matches(client: QdrantClient, settings: Settings, dimension: int) -> None:
    collection = client.get_collection(settings.collection)
    vectors = collection.config.params.vectors
    existing = vectors.get(DENSE_VECTOR_NAME) if isinstance(vectors, dict) else vectors
    if existing is None:
        raise _mismatch(settings, f"does not contain the required '{DENSE_VECTOR_NAME}' vector")
    if existing.size != dimension:
        raise _mismatch(
            settings,
            "stores "
            f"{existing.size}-dimensional vectors but the current model produces {dimension}",
        )
    if existing.distance != models.Distance.COSINE:
        raise _mismatch(settings, "uses a different dense-vector distance metric")

    sparse = collection.config.params.sparse_vectors
    sparse_config = sparse.get(SPARSE_VECTOR_NAME) if isinstance(sparse, dict) else None
    if sparse_config is None or sparse_config.modifier != models.Modifier.IDF:
        raise _mismatch(settings, "does not contain the expected IDF-modified sparse vector")

    metadata = collection.config.metadata or {}
    expected = _collection_fingerprint(settings, dimension)
    if metadata.get("reed") != expected:
        raise _mismatch(
            settings,
            "was built with a different embedding model, prefix, sparse model, or chunking setup",
        )


def _collection_fingerprint(settings: Settings, dimension: int) -> dict[str, object]:
    return {
        "schema": COLLECTION_SCHEMA_VERSION,
        "profile": settings.profile,
        "dense_model": settings.embed_model_name,
        "dense_dimension": dimension,
        "query_prefix": settings.resolved_query_prefix,
        "document_prefix": settings.resolved_doc_prefix,
        "sparse_model": settings.sparse_model,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
    }


def _ensure_payload_indexes(client: QdrantClient, settings: Settings) -> None:
    if not settings.qdrant_url:
        return
    # Remote Qdrant benefits from indexes for the filters every read/delete
    # uses. The embedded backend scans directly and warns that indexes do
    # nothing there.
    client.create_payload_index(
        collection_name=settings.collection,
        field_name=DOC_ID_PAYLOAD_KEY,
        field_schema=models.PayloadSchemaType.KEYWORD,
    )
    client.create_payload_index(
        collection_name=settings.collection,
        field_name=COMMITTED_PAYLOAD_KEY,
        field_schema=models.PayloadSchemaType.BOOL,
    )


def _mismatch(settings: Settings, reason: str) -> CollectionMismatchError:
    reset = (
        f"delete {settings.qdrant_path}"
        if not settings.qdrant_url
        else f"delete the remote collection '{settings.collection}'"
    )
    return CollectionMismatchError(
        f"Collection '{settings.collection}' {reason}. Use a fresh collection: "
        f"{reset}, or set REED_COLLECTION to a new name, then ingest again."
    )


def build_sparse_embeddings(settings: Settings) -> SparseEmbeddings:
    from langchain_qdrant import FastEmbedSparse

    return FastEmbedSparse(model_name=settings.sparse_model)


def build_vectorstore(
    client: QdrantClient,
    settings: Settings,
    embeddings: Embeddings,
    sparse_embeddings: SparseEmbeddings,
) -> QdrantVectorStore:
    from langchain_qdrant import QdrantVectorStore, RetrievalMode

    return QdrantVectorStore(
        client=client,
        collection_name=settings.collection,
        embedding=embeddings,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name=DENSE_VECTOR_NAME,
        sparse_vector_name=SPARSE_VECTOR_NAME,
    )


def delete_document_points(client: QdrantClient, settings: Settings, doc_id: str) -> None:
    client.delete(
        collection_name=settings.collection,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key=DOC_ID_PAYLOAD_KEY,
                        match=models.MatchValue(value=doc_id),
                    )
                ]
            )
        ),
    )
