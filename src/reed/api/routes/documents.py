"""Upload, list and delete documents."""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Response, UploadFile, status

from reed.api.deps import ServicesDep, require_api_key
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
    file: UploadFile,
) -> UploadAccepted:
    filename = Path(file.filename or "upload").name
    try:
        source_type(Path(filename))
    except UnsupportedFileError as exc:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=str(exc),
        ) from exc

    limit_mb = services.settings.max_upload_mb
    max_bytes = limit_mb * 1024 * 1024
    with tempfile.NamedTemporaryFile(suffix=Path(filename).suffix, delete=False) as staged:
        written = 0
        while block := await file.read(READ_CHUNK):
            written += len(block)
            if written > max_bytes:
                staged.close()
                Path(staged.name).unlink(missing_ok=True)
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"File exceeds REED_MAX_UPLOAD_MB ({limit_mb} MB)",
                )
            staged.write(block)
        staged_path = Path(staged.name)

    try:
        record, duplicate = register_upload(services, source=staged_path, filename=filename)
    finally:
        staged_path.unlink(missing_ok=True)

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
def list_documents(services: ServicesDep) -> DocumentList:
    return DocumentList(documents=[_to_info(r) for r in services.registry.list()])


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

    if not removed:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown document")
    return Response(status_code=status.HTTP_204_NO_CONTENT)
