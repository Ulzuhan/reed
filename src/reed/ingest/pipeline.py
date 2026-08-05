"""The ingestion pipeline: file in, retrievable chunks out.

Idempotent by construction. A document's id and its chunk point ids both come
from the file's SHA-256, so ingesting the same content twice overwrites the
same points instead of duplicating them — including after a crash halfway
through a previous run.
"""

from __future__ import annotations

import hashlib
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from qdrant_client import models

from reed.ingest.chunking import Chunk, split_sections
from reed.ingest.parser_worker import parse_file_isolated
from reed.ingest.parsers import (
    DocumentLimitError,
    EmptyDocumentError,
    UnsupportedFileError,
)
from reed.ingest.registry import DocumentRecord, new_logical_id
from reed.log import get_logger

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings
    from langchain_qdrant import QdrantVectorStore
    from langchain_qdrant.sparse_embeddings import SparseEmbeddings

    from reed.services import Services

logger = get_logger(__name__)

# Stable namespace so point ids are reproducible across machines and runs.
POINT_NAMESPACE = uuid.UUID("6ee0b0a2-9e3e-5a6f-9c5a-7d1f2c3b4a50")

UPSERT_BATCH_SIZE = 32


@dataclass(frozen=True, slots=True)
class IngestResult:
    document_id: str
    status: str
    chunks: int = 0
    duplicate: bool = False
    error: str | None = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def document_id_for(sha256: str) -> str:
    # 128 bits keeps ids compact while making birthday collisions negligible.
    # Existing records keep their legacy ids because deduplication uses the full
    # hash and returns the row already stored in the registry.
    return f"d-{sha256[:32]}"


def point_id_for(sha256: str, chunk_index: int) -> str:
    return str(uuid.uuid5(POINT_NAMESPACE, f"{sha256}:{chunk_index}"))


def register_upload(
    services: Services,
    *,
    source: Path,
    filename: str,
    name: str | None = None,
    copy: bool = True,
) -> tuple[DocumentRecord, bool]:
    """Record a file as durably queued for ingestion.

    Returns the record and whether it duplicates one Reed already has.
    """
    sha256 = sha256_file(source)
    doc_id = document_id_for(sha256)

    existing = services.registry.find_by_sha256(sha256)
    if existing is not None and existing.status == "error":
        # An earlier process may have died with staged points still present.
        # Finish that generation's cleanup before claiming a new retry, so a
        # delayed cleanup can never delete the retry's freshly published data.
        services.schedule_vector_cleanup(existing.id)
        services.flush_pending_vector_cleanup()
    stored_path = source
    if copy:
        services.settings.uploads_dir.mkdir(parents=True, exist_ok=True)
        stored_path = services.settings.uploads_dir / f"{doc_id}__{Path(filename).name}"

    record, duplicate = services.registry.claim_upload(
        doc_id=doc_id,
        filename=filename,
        sha256=sha256,
        size_bytes=source.stat().st_size,
        stored_path=str(stored_path),
        logical_id=new_logical_id(),
        name=name or filename,
    )
    if duplicate:
        return record, True

    claimed_path = Path(record.stored_path) if record.stored_path else source
    try:
        if copy and source.resolve() != claimed_path.resolve():
            shutil.copyfile(source, claimed_path)
    except Exception:
        claimed_path.unlink(missing_ok=True)
        services.registry.mark_error(record.id, "Could not store the uploaded file")
        raise

    # A retry under a differently named upload must not strand the old copy.
    if (
        existing is not None
        and existing.status == "error"
        and existing.stored_path
        and Path(existing.stored_path) != claimed_path
    ):
        Path(existing.stored_path).unlink(missing_ok=True)

    refreshed = services.registry.get(record.id)
    assert refreshed is not None
    return refreshed, False


def register_replacement(
    services: Services,
    *,
    logical_id: str,
    source: Path,
    filename: str,
) -> DocumentRecord:
    """Queue new content as the next version of an existing lineage."""
    sha256 = sha256_file(source)
    record = services.registry.claim_version(
        doc_id=document_id_for(sha256),
        filename=filename,
        sha256=sha256,
        size_bytes=source.stat().st_size,
        stored_path=str(services.settings.uploads_dir / f"{document_id_for(sha256)}__{filename}"),
        logical_id=logical_id,
    )
    services.settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    claimed = Path(record.stored_path or "")
    try:
        shutil.copyfile(source, claimed)
    except Exception:
        claimed.unlink(missing_ok=True)
        services.registry.delete(record.id, expected_status="queued")
        raise
    return record


def retire_superseded_versions(services: Services, record: DocumentRecord) -> None:
    """Drop the chunks of versions this one replaces, once it is serving.

    Superseded versions are not indexed at all rather than filtered at query
    time: the registry join in retrieval already hides them, so keeping their
    points would cost storage and reindex time to no end.
    """
    retired = services.registry.supersede_others(record.logical_id, keep_id=record.id)
    for previous in retired:
        services.schedule_vector_cleanup(previous.id)
        logger.info(
            "superseded %s v%d in favour of v%d", previous.name, previous.version, record.version
        )
    if retired:
        services.flush_pending_vector_cleanup()


def process_document(services: Services, doc_id: str) -> IngestResult:
    """Parse, chunk, embed and upsert a document already in the registry."""
    with services.ingestion_access:
        try:
            result = _process_document(services, doc_id)
            services.metrics.increment(f'ingestions_total{{status="{result.status}"}}')
            return result
        except Exception:
            logger.exception("ingestion task crashed for %s", doc_id)
            services.schedule_vector_cleanup(doc_id)
            try:
                services.registry.mark_error(
                    doc_id,
                    "Ingestion failed; inspect the server logs for details",
                )
            except Exception:
                logger.exception("could not record ingestion failure for %s", doc_id)
            services.metrics.increment('ingestions_total{status="error"}')
            return IngestResult(
                document_id=doc_id,
                status="error",
                error="Ingestion failed; inspect the server logs for details",
            )


def _process_document(services: Services, doc_id: str) -> IngestResult:
    record = services.registry.get(doc_id)
    if record is None:
        return IngestResult(document_id=doc_id, status="error", error="unknown document")
    if record.stored_path is None:
        services.registry.mark_error(doc_id, "no stored file")
        return IngestResult(document_id=doc_id, status="error", error="no stored file")

    if not services.registry.mark_processing(doc_id):
        current = services.registry.get(doc_id)
        status = current.status if current else "missing"
        return IngestResult(
            document_id=doc_id,
            status="error",
            error=f"document is not pending (current status: {status})",
        )
    record = services.registry.get(doc_id)
    assert record is not None
    try:
        chunks, pages = index_record_into(services, record, track_status=True)
    except Exception as exc:  # noqa: BLE001 — the message is surfaced to the user
        internal = f"{type(exc).__name__}: {exc}"
        message = _public_ingestion_error(exc)
        logger.warning("ingestion failed for %s: %s", record.filename, internal)
        services.schedule_vector_cleanup(doc_id)
        try:
            services.flush_pending_vector_cleanup()
        except Exception as cleanup_exc:  # noqa: BLE001 — retried before the next search
            logger.warning("deferred vector cleanup for %s: %s", doc_id, cleanup_exc)
        services.registry.mark_error(doc_id, message)
        return IngestResult(document_id=doc_id, status="error", error=message)

    if not services.registry.mark_ready(doc_id, chunks=chunks, pages=pages):
        services.schedule_vector_cleanup(doc_id)
        services.registry.mark_error(doc_id, "Ingestion could not be committed")
        return IngestResult(
            document_id=doc_id,
            status="error",
            error="Ingestion could not be committed",
        )
    # Only now, with the new version serving, does the old one stop.
    retire_superseded_versions(services, record)
    logger.info("ingested %s: %d chunks", record.filename, chunks)
    return IngestResult(document_id=doc_id, status="ready", chunks=chunks)


def index_record_into(
    services: Services,
    record: DocumentRecord,
    *,
    store: QdrantVectorStore | None = None,
    collection_name: str | None = None,
    embeddings: Embeddings | None = None,
    sparse_embeddings: SparseEmbeddings | None = None,
    track_status: bool = False,
) -> tuple[int, int | None]:
    """Parse and index one stored original into a selected generation."""
    settings = services.settings
    path = Path(record.stored_path or "")

    kind, sections = parse_file_isolated(
        path,
        max_pages=settings.max_pdf_pages,
        max_chars=settings.max_document_chars,
        timeout_seconds=settings.parser_timeout_seconds,
        memory_mb=settings.parser_memory_mb,
        cpu_seconds=settings.parser_cpu_seconds,
    )
    chunks = split_sections(
        sections,
        source_type=kind,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    if not chunks:
        raise EmptyDocumentError("Document produced no chunks")
    if track_status and not services.registry.mark_stage(
        record.id, current="parsing", next="embedding"
    ):
        raise RuntimeError("document left the parsing stage unexpectedly")

    display_texts = [chunk.text for chunk in chunks]
    metadatas = [_metadata_for(record, chunk, kind) for chunk in chunks]
    ids: list[int | str] = [point_id_for(record.sha256, chunk.index) for chunk in chunks]
    qdrant_ids: list[int | str | uuid.UUID] = list(ids)

    # Ensure the collection exists. Candidate reindexes pass an already-built
    # store because the active generation may intentionally be incompatible.
    _ = store or services.vectorstore
    selected_collection = collection_name or services.active_collection_name
    dense_model = embeddings or services.embeddings
    sparse_model = sparse_embeddings or services.sparse_embeddings
    embedding_texts = [
        _embedding_text(record, chunk, display_text)
        for chunk, display_text in zip(chunks, display_texts, strict=True)
    ]

    # Model inference is deliberately outside vector_access. Embedded Qdrant
    # needs short serialised mutations, not a global lock around minutes of CPU
    # or provider latency.
    dense_vectors = dense_model.embed_documents(embedding_texts)
    sparse_vectors = sparse_model.embed_documents(embedding_texts)
    if track_status and not services.registry.mark_stage(
        record.id, current="embedding", next="indexing"
    ):
        raise RuntimeError("document left the embedding stage unexpectedly")
    points = [
        models.PointStruct(
            id=point_id,
            vector={
                "dense": dense,
                "sparse": models.SparseVector(indices=sparse.indices, values=sparse.values),
            },
            payload={"page_content": display, "metadata": metadata},
        )
        for point_id, dense, sparse, display, metadata in zip(
            qdrant_ids,
            dense_vectors,
            sparse_vectors,
            display_texts,
            metadatas,
            strict=True,
        )
    ]
    with services.vector_access:
        # A retry may follow a partial write under a different chunking setup.
        # Removing every old point first prevents obsolete trailing chunks.
        from reed.rag.vectorstore import delete_document_points

        if services.qdrant.collection_exists(selected_collection):
            delete_document_points(
                services.qdrant,
                settings,
                record.id,
                collection_name=selected_collection,
            )
        for offset in range(0, len(points), UPSERT_BATCH_SIZE):
            services.qdrant.upsert(
                collection_name=selected_collection,
                points=points[offset : offset + UPSERT_BATCH_SIZE],
                wait=True,
            )
        # A remote Qdrant can be queried while a multi-batch upload is still in
        # progress. Publish every point in one payload update only after all
        # batches succeeded; retrieval filters out the staging points.
        services.qdrant.set_payload(
            collection_name=selected_collection,
            payload={"committed": True},
            points=qdrant_ids,
            key="metadata",
            wait=True,
        )

    pages = max((c.page for c in chunks if c.page is not None), default=None)
    return len(chunks), pages


def _embedding_text(record: DocumentRecord, chunk: Chunk, display_text: str) -> str:
    """Add provenance for retrieval without polluting prompts or citations."""
    title = record.filename
    section = chunk.section or "(document root)"
    return f"Title: {title}\nSection: {section}\n\n{display_text}"


def _metadata_for(record: DocumentRecord, chunk: Chunk, source_type: str) -> dict[str, object]:
    return {
        "doc_id": record.id,
        "filename": record.filename,
        "source_type": source_type,
        "page": chunk.page,
        "chunk_index": chunk.index,
        "section": chunk.section,
        "doc_sha256": record.sha256,
        "ingested_at": record.created_at,
        "committed": False,
    }


def ingest_path(services: Services, path: Path) -> IngestResult:
    """Register and immediately process a local file (used by the CLI)."""
    record, duplicate = register_upload(services, source=path, filename=path.name)
    if duplicate:
        return IngestResult(
            document_id=record.id,
            status=record.status,
            chunks=record.chunks,
            duplicate=True,
        )
    return process_document(services, record.id)


class DocumentBusyError(RuntimeError):
    """The document is being ingested and cannot be removed yet."""


def delete_lineage(services: Services, logical_id: str) -> int:
    """Remove every version of one document. Returns how many were removed."""
    versions = services.registry.lineage(logical_id)
    if not versions:
        return 0
    removed = 0
    # Oldest first, so an interruption leaves the current version last standing
    # rather than a lineage whose only survivors are retired.
    for record in sorted(versions, key=lambda item: item.version):
        if delete_document(services, record.id):
            removed += 1
    return removed


def delete_document(services: Services, doc_id: str) -> bool:
    """Remove a document's chunks, registry row and stored file.

    Refuses while ingestion is in flight: deleting mid-run would let the
    background task upsert its chunks afterwards, leaving vectors that are
    retrievable and citable for a document the API says does not exist.
    """
    from reed.rag.vectorstore import delete_document_points

    record, busy = services.registry.begin_delete(doc_id)
    if record is None:
        return False
    if busy:
        raise DocumentBusyError(
            f"'{record.filename}' is busy — try again once the current operation finishes"
        )

    try:
        with services.vector_access:
            # The collection only exists once something has been ingested; a
            # document that failed to parse may never have created it.
            collection_name = services.active_collection_name
            if services.qdrant.collection_exists(collection_name):
                delete_document_points(
                    services.qdrant,
                    services.settings,
                    doc_id,
                    collection_name=collection_name,
                )

        if record.stored_path:
            Path(record.stored_path).unlink(missing_ok=True)

        return services.registry.delete(doc_id, expected_status="deleting")
    except Exception:
        services.registry.mark_error(doc_id, "Deletion was interrupted; retry deletion")
        raise


def _public_ingestion_error(exc: Exception) -> str:
    if isinstance(exc, (DocumentLimitError, EmptyDocumentError, UnsupportedFileError)):
        return str(exc)
    return "Ingestion failed; inspect the server logs for details"
