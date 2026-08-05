"""Upload, list and delete documents."""

from __future__ import annotations

import contextlib
import functools
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path

import anyio.to_thread
from fastapi import (
    APIRouter,
    Depends,
    Form,
    HTTPException,
    Query,
    Response,
    UploadFile,
    status,
)

from reed.api.deps import ServicesDep, require_api_key
from reed.api.schemas import DocumentInfo, DocumentList, UploadAccepted
from reed.ingest.parsers import UnsupportedFileError, source_type
from reed.ingest.pipeline import (
    DocumentBusyError,
    delete_document,
    delete_lineage,
    register_replacement,
    register_upload,
)
from reed.ingest.registry import DocumentRecord, NameConflictError
from reed.log import get_logger

logger = get_logger(__name__)

router = APIRouter(
    prefix="/v1/documents", tags=["documents"], dependencies=[Depends(require_api_key)]
)

READ_CHUNK = 1 << 20

# Reed assigns both ids, so the prefix reliably tells a lineage from a version.
LINEAGE_PREFIX = "l-"


def _to_info(record: DocumentRecord) -> DocumentInfo:
    return DocumentInfo(
        id=record.id,
        logical_id=record.logical_id,
        name=record.name,
        version=record.version,
        filename=record.filename,
        status=record.status,
        chunks=record.chunks,
        pages=record.pages,
        size_bytes=record.size_bytes,
        created_at=record.created_at,
        error=record.error,
    )


@contextlib.asynccontextmanager
async def _staged_upload(
    services: ServicesDep, file: UploadFile
) -> AsyncIterator[tuple[Path, str]]:
    """Spool an upload to a bounded temporary file, always cleaned up."""
    filename = _safe_filename(file.filename or "upload")
    try:
        source_type(Path(filename))
    except UnsupportedFileError as exc:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=str(exc),
        ) from exc

    limit_mb = services.settings.max_upload_mb
    max_bytes = limit_mb * 1024 * 1024
    staged_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=Path(filename).suffix, delete=False) as staged:
            staged_path = Path(staged.name)
            written = 0
            while block := await file.read(READ_CHUNK):
                written += len(block)
                if written > max_bytes:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"File exceeds REED_MAX_UPLOAD_MB ({limit_mb} MB)",
                    )
                # Disk writes block; a worker thread keeps concurrent SSE
                # streams flowing while a large upload spools.
                await anyio.to_thread.run_sync(staged.write, block)
        if written == 0:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="Uploaded file is empty",
            )
        assert staged_path is not None
        yield staged_path, filename
    finally:
        if staged_path is not None:
            staged_path.unlink(missing_ok=True)
        with contextlib.suppress(Exception):
            await file.close()


@router.post("", response_model=UploadAccepted, status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    services: ServicesDep,
    file: UploadFile,
    name: str | None = Form(default=None),
) -> UploadAccepted:
    if services.ingestion_queue_full:
        services.metrics.increment("upload_rejections_total")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The ingestion queue is full; retry after current work completes",
            headers={"Retry-After": "2"},
        )
    async with _staged_upload(services, file) as (staged_path, filename):
        lineage_name = _safe_filename(name) if name else filename
        try:
            # Hashing and copying up to REED_MAX_UPLOAD_MB is blocking work,
            # so it runs off the event loop. The registry claims the display
            # name in the same transaction that claims the content hash; a
            # separate check here would race a concurrent upload into two
            # lineages sharing one name.
            record, duplicate = await anyio.to_thread.run_sync(
                functools.partial(
                    register_upload,
                    services,
                    source=staged_path,
                    filename=filename,
                    name=lineage_name,
                )
            )
        except NameConflictError as exc:
            # Not superseded silently: replacing is a PUT, and the caller is
            # told which lineage to send it to. A genuinely different document
            # that happens to share a filename can pass its own `name`.
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "message": (
                        f"A document named '{lineage_name}' already exists. Replace it with "
                        "PUT /v1/documents/{logical_id}, or upload this one under another name."
                    ),
                    "logical_id": exc.existing.logical_id,
                    "document_id": exc.existing.id,
                },
            ) from exc
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("upload could not be registered")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="The upload could not be registered",
            ) from exc

    if duplicate:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "message": "This file has already been ingested",
                "document_id": record.id,
            },
        )

    if not services.enqueue_ingestion(record.id):
        services.metrics.increment("upload_rejections_total")
        if record.stored_path:
            Path(record.stored_path).unlink(missing_ok=True)
        services.registry.delete(record.id, expected_status="queued")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The ingestion queue filled concurrently; retry shortly",
            headers={"Retry-After": "2"},
        )
    services.metrics.increment("uploads_total")
    return UploadAccepted(
        document_id=record.id,
        logical_id=record.logical_id,
        version=record.version,
        filename=record.filename,
        status=record.status,
    )


@router.put("/{logical_id}", response_model=UploadAccepted, status_code=status.HTTP_202_ACCEPTED)
async def replace_document(
    services: ServicesDep,
    logical_id: str,
    file: UploadFile,
) -> UploadAccepted:
    """Publish new content for an existing document.

    The version currently serving stays ready and indexed until the new one is,
    so a replacement never leaves the document unfindable.
    """
    if services.ingestion_queue_full:
        services.metrics.increment("upload_rejections_total")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The ingestion queue is full; retry after current work completes",
            headers={"Retry-After": "2"},
        )
    if not services.registry.lineage(logical_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown document")

    async with _staged_upload(services, file) as (staged_path, filename):
        try:
            # Same blocking hash-and-copy as the upload route.
            record = await anyio.to_thread.run_sync(
                functools.partial(
                    register_replacement,
                    services,
                    logical_id=logical_id,
                    source=staged_path,
                    filename=filename,
                )
            )
        except FileExistsError as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Unknown document"
            ) from exc
        except Exception as exc:
            logger.exception("replacement could not be registered")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="The replacement could not be registered",
            ) from exc

    if not services.enqueue_ingestion(record.id):
        services.metrics.increment("upload_rejections_total")
        if record.stored_path:
            Path(record.stored_path).unlink(missing_ok=True)
        services.registry.delete(record.id, expected_status="queued")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The ingestion queue filled concurrently; retry shortly",
            headers={"Retry-After": "2"},
        )
    services.metrics.increment("uploads_total")
    return UploadAccepted(
        document_id=record.id,
        logical_id=record.logical_id,
        version=record.version,
        filename=record.filename,
        status=record.status,
    )


@router.get("", response_model=DocumentList)
def list_documents(
    services: ServicesDep,
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> DocumentList:
    records = services.registry.list(limit=limit, offset=offset)
    return DocumentList(
        documents=[_to_info(record) for record in records],
        total=services.registry.count(),
        limit=limit,
        offset=offset,
    )


@router.get("/{logical_id}/versions", response_model=DocumentList)
def list_versions(services: ServicesDep, logical_id: str) -> DocumentList:
    """Every version of one document, newest first, superseded ones included."""
    versions = services.registry.lineage(logical_id)
    if not versions:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown document")
    return DocumentList(
        documents=[_to_info(record) for record in versions],
        total=len(versions),
        limit=len(versions),
        offset=0,
    )


@router.delete("/{logical_id}/versions/{version}", status_code=status.HTTP_204_NO_CONTENT)
def remove_version(services: ServicesDep, logical_id: str, version: int) -> Response:
    """Forget one superseded version, content and all.

    The serving version is refused: removing it would silently promote nothing,
    and dropping a document is what deleting the lineage is for. This exists so
    content that should never have been uploaded can actually be purged.
    """
    match = [
        record for record in services.registry.lineage(logical_id) if record.version == version
    ]
    if not match:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown version")
    if match[0].status != "superseded":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This version is the one in use; delete the document instead",
        )
    _delete_or_fail(services, match[0].id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/{document_id}", response_model=DocumentInfo)
def get_document(services: ServicesDep, document_id: str) -> DocumentInfo:
    record = services.registry.get(document_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown document")
    return _to_info(record)


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def remove_document(services: ServicesDep, document_id: str) -> Response:
    """Delete a document.

    Given a lineage id the whole history goes, which is what "remove this
    document" means; deleting only the current version would make an older one
    reappear in search results. Given a version's own id, only that version
    goes. The two are told apart by the prefix Reed assigns them.
    """
    if document_id.startswith(LINEAGE_PREFIX):
        removed = _lineage_or_fail(services, document_id)
    else:
        removed = 1 if _delete_or_fail(services, document_id) else 0

    if not removed:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown document")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


def _delete_or_fail(services: ServicesDep, document_id: str) -> bool:
    try:
        return delete_document(services, document_id)
    except DocumentBusyError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except Exception as exc:
        # The id comes from the URL. Keeping it out of the log prevents forged
        # entries through encoded control characters (CWE-117).
        logger.exception("document deletion failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Document deletion could not be completed; retry the operation",
        ) from exc


def _lineage_or_fail(services: ServicesDep, logical_id: str) -> int:
    try:
        return delete_lineage(services, logical_id)
    except DocumentBusyError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("document lineage deletion failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Document deletion could not be completed; retry the operation",
        ) from exc


def _safe_filename(filename: str) -> str:
    basename = Path(filename).name
    printable = "".join(character for character in basename if character.isprintable())
    if not printable:
        return "upload"
    if len(printable) <= 255:
        return printable
    suffix = Path(printable).suffix[:20]
    stem_limit = 255 - len(suffix)
    return f"{Path(printable).stem[:stem_limit]}{suffix}"
