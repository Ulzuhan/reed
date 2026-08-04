"""Request and response models for the HTTP API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

DocumentStatus = Literal["pending", "processing", "ready", "error"]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    profile: str
    chat_model: str
    embed_model: str
    vector_store: str = Field(description="'ok' or an error description")
    documents: int | None = Field(default=None, description="Documents in the registry")


class DocumentInfo(BaseModel):
    id: str
    filename: str
    status: DocumentStatus
    chunks: int = 0
    pages: int | None = None
    size_bytes: int = 0
    created_at: str
    error: str | None = None


class DocumentList(BaseModel):
    documents: list[DocumentInfo]


class UploadAccepted(BaseModel):
    document_id: str
    filename: str
    status: DocumentStatus


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    history: list[ChatMessage] = Field(
        default_factory=list,
        description="Prior turns; only the most recent ones are used.",
    )
    top_k: int | None = Field(default=None, ge=1, le=50)
    stream: bool = True


class Source(BaseModel):
    n: int = Field(description="Citation number, matching the [n] markers in the answer")
    doc_id: str
    filename: str
    page: int | None = None
    score: float
    snippet: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    latency_ms: int
