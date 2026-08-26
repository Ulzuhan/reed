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
from reed.ingest.extraction import EXTRACTION_VERSION
from reed.log import get_logger
from reed.model_identity import ModelIdentity

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings
    from langchain_qdrant import QdrantVectorStore
    from langchain_qdrant.sparse_embeddings import SparseEmbeddings

logger = get_logger(__name__)

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
# langchain-qdrant's default payload layout. Ingestion writes it by hand and
# retrieval reads it by hand, so both ends name it from here rather than
# restating the strings and drifting apart.
CONTENT_PAYLOAD_KEY = "page_content"
METADATA_PAYLOAD_KEY = "metadata"
DOC_ID_PAYLOAD_KEY = f"{METADATA_PAYLOAD_KEY}.doc_id"
COMMITTED_PAYLOAD_KEY = f"{METADATA_PAYLOAD_KEY}.committed"
COLLECTION_SCHEMA_VERSION = 5


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


def delete_collection_safely(client: QdrantClient, collection_name: str) -> None:
    """Delete after closing QdrantLocal's per-collection SQLite persistence.

    qdrant-client 1.18 removes a local collection from its mapping without
    calling ``LocalCollection.close()``, leaking the connection until garbage
    collection. Remote clients have no local collection mapping.
    """
    backend = getattr(client, "_client", None)
    collections = getattr(backend, "collections", None)
    if isinstance(collections, dict):
        collection = collections.get(collection_name)
        close = getattr(collection, "close", None)
        if callable(close):
            close()
    client.delete_collection(collection_name)


def ensure_collection(
    client: QdrantClient,
    settings: Settings,
    dimension: int,
    *,
    collection_name: str | None = None,
    fingerprint: dict[str, object] | None = None,
) -> None:
    """Create the collection, or verify the existing one still fits."""
    name = collection_name or settings.collection
    expected = fingerprint or collection_fingerprint(
        settings,
        dimension,
        ModelIdentity(
            digest=settings.embed_model_digest,
            revision=settings.embed_model_revision,
            quantization=settings.embed_model_quantization,
        ),
    )
    if client.collection_exists(name):
        _assert_dimension_matches(
            client,
            dimension,
            collection_name=name,
            fingerprint=expected,
        )
        _ensure_payload_indexes(client, settings, name)
        return

    client.create_collection(
        collection_name=name,
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
        metadata={"reed": expected},
    )
    _ensure_payload_indexes(client, settings, name)
    logger.info("created collection '%s' (dim=%d)", name, dimension)


def _assert_dimension_matches(
    client: QdrantClient,
    dimension: int,
    *,
    collection_name: str,
    fingerprint: dict[str, object],
) -> None:
    collection = client.get_collection(collection_name)
    vectors = collection.config.params.vectors
    existing = vectors.get(DENSE_VECTOR_NAME) if isinstance(vectors, dict) else vectors
    if existing is None:
        raise _mismatch(
            collection_name,
            f"does not contain the required '{DENSE_VECTOR_NAME}' vector",
        )
    if existing.size != dimension:
        raise _mismatch(
            collection_name,
            "stores "
            f"{existing.size}-dimensional vectors but the current model produces {dimension}",
        )
    if existing.distance != models.Distance.COSINE:
        raise _mismatch(collection_name, "uses a different dense-vector distance metric")

    sparse = collection.config.params.sparse_vectors
    sparse_config = sparse.get(SPARSE_VECTOR_NAME) if isinstance(sparse, dict) else None
    if sparse_config is None or sparse_config.modifier != models.Modifier.IDF:
        raise _mismatch(
            collection_name,
            "does not contain the expected IDF-modified sparse vector",
        )

    metadata = collection.config.metadata or {}
    stored = metadata.get("reed")
    if stored != fingerprint:
        # Adopting a collection whose fingerprint differs was possible until
        # v0.4 and was unsound: Reed 0.2 embedded the raw chunk text and Reed
        # 0.3 embeds a title/section header, so adoption mixed two kinds of
        # vector in one collection with nothing to detect it afterwards.
        raise _mismatch(
            collection_name,
            "was built with a different embedding model, prefix, sparse model, chunking setup, "
            "or extraction pipeline",
        )


def collection_fingerprint(
    settings: Settings, dimension: int, identity: ModelIdentity
) -> dict[str, object]:
    return {
        "schema": COLLECTION_SCHEMA_VERSION,
        "profile": settings.profile,
        "dense_model": settings.embed_model_name,
        "dense_digest": identity.digest,
        "dense_revision": identity.revision,
        "dense_quantization": identity.quantization,
        "dense_dimension": dimension,
        "query_prefix": settings.resolved_query_prefix,
        "document_prefix": settings.resolved_doc_prefix,
        "sparse_model": settings.sparse_model,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        # What the vectors were made from, not just which model made them.
        "extraction": EXTRACTION_VERSION,
    }


def _ensure_payload_indexes(client: QdrantClient, settings: Settings, collection_name: str) -> None:
    if not settings.qdrant_url:
        return
    # Remote Qdrant benefits from indexes for the filters every read/delete
    # uses. The embedded backend scans directly and warns that indexes do
    # nothing there.
    client.create_payload_index(
        collection_name=collection_name,
        field_name=DOC_ID_PAYLOAD_KEY,
        field_schema=models.PayloadSchemaType.KEYWORD,
    )
    client.create_payload_index(
        collection_name=collection_name,
        field_name=COMMITTED_PAYLOAD_KEY,
        field_schema=models.PayloadSchemaType.BOOL,
    )


def _mismatch(collection_name: str, reason: str) -> CollectionMismatchError:
    return CollectionMismatchError(
        f"Collection '{collection_name}' {reason}. Run 'reed index reindex' to build and "
        "atomically activate a compatible generation; do not delete the current index, which "
        "will remain available."
    )


def build_sparse_embeddings(settings: Settings) -> SparseEmbeddings:
    from langchain_qdrant import FastEmbedSparse

    return FastEmbedSparse(model_name=settings.sparse_model)


def build_vectorstore(
    client: QdrantClient,
    settings: Settings,
    embeddings: Embeddings,
    sparse_embeddings: SparseEmbeddings,
    *,
    collection_name: str | None = None,
    retrieval_mode: str = "hybrid",
) -> QdrantVectorStore:
    from langchain_qdrant import QdrantVectorStore, RetrievalMode

    modes = {
        "dense": RetrievalMode.DENSE,
        "sparse": RetrievalMode.SPARSE,
        "hybrid": RetrievalMode.HYBRID,
        "hybrid_rerank": RetrievalMode.HYBRID,
    }
    try:
        selected_mode = modes[retrieval_mode]
    except KeyError as exc:
        raise ValueError(f"unsupported retrieval mode: {retrieval_mode}") from exc

    return QdrantVectorStore(
        client=client,
        collection_name=collection_name or settings.collection,
        embedding=embeddings,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=selected_mode,
        vector_name=DENSE_VECTOR_NAME,
        sparse_vector_name=SPARSE_VECTOR_NAME,
    )


def delete_document_points(
    client: QdrantClient,
    settings: Settings,
    doc_id: str,
    *,
    collection_name: str | None = None,
) -> None:
    client.delete(
        collection_name=collection_name or settings.collection,
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
