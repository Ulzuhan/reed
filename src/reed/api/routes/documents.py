"""Upload, list and delete documents."""

from __future__ import annotations

import contextlib
import tempfile
from pathlib import Path

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
    status,
)

from reed.api.deps import ServicesDep, enforce_rate_limit, require_api_key
from reed.api.schemas import DocumentInfo, DocumentList, UploadAccepted
from reed.ingest.parsers import UnsupportedFileError, source_type
from reed.ingest.pipeline import (
    DocumentBusyError,
    delete_document,
    process_document,
    register_upload,
)
from reed.ingest.registry import DocumentRecord
from reed.log import get_logger

logger = get_logger(__name__)

router = APIRouter(
    prefix="/v1/documents", tags=["documents"], dependencies=[Depends(require_api_key)]
)

READ_CHUNK = 1 << 20


def _to_info(record: DocumentRecord) -> DocumentInfo:
    return DocumentInfo(
        id=record.id,
        filename=record.filename,
        status=record.status,
        chunks=record.chunks,
        pages=record.pages,
        size_bytes=record.size_bytes,
        created_at=record.created_at,
        error=record.error,
    )


@router.post("", response_model=UploadAccepted, status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    services: ServicesDep,
    background: BackgroundTasks,
    http_request: Request,
    file: UploadFile,
) -> UploadAccepted:
    enforce_rate_limit(
        http_request,
        services,
        scope="upload",
        limit=services.settings.upload_rate_limit_per_minute,
    )
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
                staged.write(block)

        if written == 0:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="Uploaded file is empty",
            )
        assert staged_path is not None
        record, duplicate = register_upload(services, source=staged_path, filename=filename)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("upload could not be staged or registered")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The upload could not be registered",
        ) from exc
    finally:
        if staged_path is not None:
            staged_path.unlink(missing_ok=True)
        with contextlib.suppress(Exception):
            await file.close()

    if duplicate:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "message": "This file has already been ingested",
                "document_id": record.id,
            },
        )

    background.add_task(process_document, services, record.id)
    return UploadAccepted(document_id=record.id, filename=record.filename, status=record.status)


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


@router.get("/{document_id}", response_model=DocumentInfo)
def get_document(services: ServicesDep, document_id: str) -> DocumentInfo:
    record = services.registry.get(document_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown document")
    return _to_info(record)


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def remove_document(services: ServicesDep, document_id: str) -> Response:
    try:
        removed = delete_document(services, document_id)
    except DocumentBusyError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("document deletion failed for %s", document_id)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Document deletion could not be completed; retry the operation",
        ) from exc

    if not removed:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown document")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


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
