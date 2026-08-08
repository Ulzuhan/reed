"""Request and response models for the HTTP API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field, field_validator

if TYPE_CHECKING:
    from reed.rag.retriever import RetrievedChunk

DocumentStatus = Literal[
    "pending",
    "processing",
    "queued",
    "parsing",
    "embedding",
    "indexing",
    "ready",
    "error",
    "deleting",
    "superseded",
]
CitationStatus = Literal["valid", "missing", "invalid", "not_applicable"]

MAX_HISTORY_MESSAGES = 6
MAX_HISTORY_CONTENT_CHARS = 8_000


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str | None = None
    profile: str | None = None
    chat_model: str | None = None
    embed_model: str | None = None
    vector_store: str = Field(description="'ok', 'not_checked', or 'unavailable'")
    chat_provider: str = Field(default="not_checked", description="'ok' or 'unavailable'")
    documents: int | None = Field(default=None, description="Documents in the registry")


class DocumentInfo(BaseModel):
    id: str
    logical_id: str = Field(default="", description="Identity that survives a replacement")
    name: str = Field(default="", description="Display name of the lineage")
    version: int = 1
    filename: str
    status: DocumentStatus
    chunks: int = 0
    pages: int | None = None
    size_bytes: int = 0
    created_at: str
    error: str | None = None


class DocumentList(BaseModel):
    documents: list[DocumentInfo]
    total: int = 0
    limit: int = 100
    offset: int = 0


class UploadAccepted(BaseModel):
    document_id: str
    logical_id: str = ""
    version: int = 1
    filename: str
    status: DocumentStatus


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=MAX_HISTORY_CONTENT_CHARS)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    history: list[ChatMessage] = Field(
        default_factory=list,
        max_length=MAX_HISTORY_MESSAGES,
        description="Prior turns; only the most recent ones are used.",
    )
    top_k: int | None = Field(default=None, ge=1, le=50)
    stream: bool = True

    @field_validator("question")
    @classmethod
    def _question_is_not_whitespace(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("question must contain non-whitespace characters")
        return stripped


class Source(BaseModel):
    n: int = Field(description="Citation number, matching the [n] markers in the answer")
    doc_id: str
    filename: str
    page: int | None = None
    section: str | None = Field(default=None, description="Nearest heading, for Markdown sources")
    location: str = Field(description="How this passage was labelled in the prompt")
    score: float
    snippet: str
    excerpt: str = Field(description="The complete retrieved chunk cited by the answer")


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    latency_ms: int
    citation_status: CitationStatus
    citation_warnings: list[str] = Field(default_factory=list)


class SearchRequest(BaseModel):
    """Retrieval without generation: the same query rules as :class:`AskRequest`."""

    query: str = Field(min_length=1, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=50)

    @field_validator("query")
    @classmethod
    def _query_is_not_whitespace(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("query must contain non-whitespace characters")
        return stripped


class SearchResponse(BaseModel):
    sources: list[Source]
    latency_ms: int
    sufficient_evidence: bool = Field(
        description=(
            "Whether the top score clears the configured evidence threshold. Results are "
            "returned either way; /v1/ask is what abstains."
        )
    )
    min_evidence_score: float = Field(
        description="The threshold this verdict was taken against, in the queried score domain"
    )


def to_sources(chunks: list[RetrievedChunk]) -> list[Source]:
    """Number the chunks — these are the ``[n]`` markers the model must use."""
    return [
        Source(
            n=number,
            doc_id=chunk.doc_id,
            filename=chunk.filename,
            page=chunk.page,
            section=chunk.section,
            location=chunk.location,
            score=round(chunk.score, 4),
            snippet=chunk.snippet,
            excerpt=chunk.text,
        )
        for number, chunk in enumerate(chunks, start=1)
    ]
