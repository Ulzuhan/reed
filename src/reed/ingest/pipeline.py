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

from reed.ingest.chunking import Chunk, split_sections
from reed.ingest.parsers import parse_file
from reed.ingest.registry import DocumentRecord
from reed.log import get_logger

if TYPE_CHECKING:
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
    return f"d-{sha256[:12]}"


def point_id_for(sha256: str, chunk_index: int) -> str:
    return str(uuid.uuid5(POINT_NAMESPACE, f"{sha256}:{chunk_index}"))


def register_upload(
    services: Services,
    *,
    source: Path,
    filename: str,
    copy: bool = True,
) -> tuple[DocumentRecord, bool]:
    """Record a file as pending ingestion.

    Returns the record and whether it duplicates one Reed already has.
    """
    sha256 = sha256_file(source)
    doc_id = document_id_for(sha256)

    existing = services.registry.find_by_sha256(sha256)
    # Only a previous failure earns a retry. Re-registering a document that is
    # mid-ingestion would start a second run and truncate the stored file the
    # first one is still reading.
    if existing is not None and existing.status in {"ready", "pending", "processing"}:
        return existing, True

    stored_path = source
    if copy:
        services.settings.uploads_dir.mkdir(parents=True, exist_ok=True)
        stored_path = services.settings.uploads_dir / f"{doc_id}__{Path(filename).name}"
        if source.resolve() != stored_path.resolve():
            shutil.copyfile(source, stored_path)

    record = services.registry.add(
        doc_id=doc_id,
        filename=filename,
        sha256=sha256,
        size_bytes=source.stat().st_size,
        stored_path=str(stored_path),
    )
    return record, False


def process_document(services: Services, doc_id: str) -> IngestResult:
    """Parse, chunk, embed and upsert a document already in the registry."""
    record = services.registry.get(doc_id)
    if record is None:
        return IngestResult(document_id=doc_id, status="error", error="unknown document")
    if record.stored_path is None:
        services.registry.mark_error(doc_id, "no stored file")
        return IngestResult(document_id=doc_id, status="error", error="no stored file")

    services.registry.mark_processing(doc_id)
    try:
        chunks, pages = _embed_and_store(services, record)
    except Exception as exc:  # noqa: BLE001 — the message is surfaced to the user
        message = f"{type(exc).__name__}: {exc}"
        logger.warning("ingestion failed for %s: %s", record.filename, message)
        services.registry.mark_error(doc_id, message)
        return IngestResult(document_id=doc_id, status="error", error=message)

    services.registry.mark_ready(doc_id, chunks=chunks, pages=pages)
    logger.info("ingested %s: %d chunks", record.filename, chunks)
    return IngestResult(document_id=doc_id, status="ready", chunks=chunks)


def _embed_and_store(services: Services, record: DocumentRecord) -> tuple[int, int | None]:
    settings = services.settings
    path = Path(record.stored_path or "")

    kind, sections = parse_file(path)
    chunks = split_sections(
        sections,
        source_type=kind,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    if not chunks:
        raise ValueError("document produced no chunks")

    texts = [chunk.text for chunk in chunks]
    metadatas = [_metadata_for(record, chunk, kind) for chunk in chunks]
    ids = [point_id_for(record.sha256, chunk.index) for chunk in chunks]

    store = services.vectorstore
    with services.vector_access:
        store.add_texts(
            texts=texts,
            metadatas=metadatas,
            ids=ids,
            batch_size=UPSERT_BATCH_SIZE,
        )

    pages = max((c.page for c in chunks if c.page is not None), default=None)
    return len(chunks), pages


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


def delete_document(services: Services, doc_id: str) -> bool:
    """Remove a document's chunks, registry row and stored file.

    Refuses while ingestion is in flight: deleting mid-run would let the
    background task upsert its chunks afterwards, leaving vectors that are
    retrievable and citable for a document the API says does not exist.
    """
    from reed.rag.vectorstore import delete_document_points

    record = services.registry.get(doc_id)
    if record is None:
        return False
    if record.status in {"pending", "processing"}:
        raise DocumentBusyError(
            f"'{record.filename}' is still being ingested — try again once it is ready"
        )

    with services.vector_access:
        # The collection only exists once something has been ingested; a
        # document that failed to parse may never have created it.
        if services.qdrant.collection_exists(services.settings.collection):
            delete_document_points(services.qdrant, services.settings, doc_id)

    if record.stored_path:
        Path(record.stored_path).unlink(missing_ok=True)

    return services.registry.delete(doc_id)
