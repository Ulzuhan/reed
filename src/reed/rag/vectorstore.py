"""Qdrant wiring — embedded folder or remote server, same code either way.

The collection carries two named vectors: a dense one from the embedding model
and a sparse BM25 one. Qdrant fuses them server-side with Reciprocal Rank
Fusion, which is what makes a question phrased in the user's words find a
chunk that uses the document's exact terminology.
"""

from __future__ import annotations

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


class CollectionMismatchError(RuntimeError):
    """The existing collection was built with a different embedding model."""


def get_qdrant_client(settings: Settings) -> QdrantClient:
    if settings.qdrant_url:
        logger.info("connecting to Qdrant server at %s", settings.qdrant_url)
        return QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
        )

    path = settings.qdrant_path
    path.mkdir(parents=True, exist_ok=True)
    logger.info("using embedded Qdrant at %s", path)
    return QdrantClient(path=str(path))


def ensure_collection(client: QdrantClient, settings: Settings, dimension: int) -> None:
    """Create the collection, or verify the existing one still fits."""
    if client.collection_exists(settings.collection):
        _assert_dimension_matches(client, settings, dimension)
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
    )
    if settings.qdrant_url:
        # Deleting a document filters on this key. The embedded backend scans
        # payloads directly and warns that indexes do nothing there.
        client.create_payload_index(
            collection_name=settings.collection,
            field_name=DOC_ID_PAYLOAD_KEY,
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
    logger.info("created collection '%s' (dim=%d)", settings.collection, dimension)


def _assert_dimension_matches(client: QdrantClient, settings: Settings, dimension: int) -> None:
    vectors = client.get_collection(settings.collection).config.params.vectors
    existing = vectors.get(DENSE_VECTOR_NAME) if isinstance(vectors, dict) else vectors
    if existing is None or existing.size == dimension:
        return
    raise CollectionMismatchError(
        f"Collection '{settings.collection}' stores {existing.size}-dimensional vectors "
        f"but the current embedding model produces {dimension}. Switching providers needs "
        f"a fresh collection: delete {settings.qdrant_path} (or set REED_COLLECTION to a "
        f"new name) and ingest again."
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
