"""Application settings.

Every knob is a ``REED_*`` environment variable (or an entry in ``.env``). The
``profile`` picks sensible defaults for a whole provider stack; individual
settings can still be overridden one by one.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Profile = Literal["openai", "local", "fake"]
LogLevel = Literal["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"]
RetrievalMode = Literal["dense", "sparse", "hybrid", "hybrid_rerank"]

# EmbeddingGemma is trained with task prefixes and loses accuracy without them.
# https://ai.google.dev/gemma/docs/embeddinggemma/model_card
EMBEDDINGGEMMA_QUERY_PREFIX = "task: search result | query: "
# Document titles/sections are injected per chunk by the ingestion pipeline;
# keeping a static "title: none" prefix would throw that signal away.
EMBEDDINGGEMMA_DOC_PREFIX = ""
QWEN3_EMBED_QUERY_PREFIX = (
    "Instruct: Given a user query, retrieve relevant passages that answer the query\nQuery:"
)
EMBEDDINGGEMMA_HYBRID_MIN_SCORE = 0.8333333333333333


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="REED_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        validate_assignment=True,
    )

    profile: Profile = "openai"

    # --- OpenAI ---------------------------------------------------------
    # The conventional OPENAI_API_KEY works, and so does the prefixed spelling —
    # an explicit alias list suppresses env_prefix, so both must be named here.
    openai_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("OPENAI_API_KEY", "REED_OPENAI_API_KEY", "openai_api_key"),
        repr=False,
    )
    openai_chat_model: str = "gpt-5-mini"
    openai_embed_model: str = "text-embedding-3-small"

    # --- Ollama ---------------------------------------------------------
    ollama_base_url: str = "http://localhost:11434"
    ollama_chat_model: str = "qwen3.5:4b"
    ollama_embed_model: str = "embeddinggemma"

    # Immutable identity for hosted/custom providers. Ollama resolves these
    # fields from its local API unless explicit overrides are supplied.
    embed_model_digest: str = ""
    embed_model_revision: str = ""
    embed_model_quantization: str = ""

    # Task prefixes for embeddings. ``None`` means "use the profile default",
    # an empty string means "explicitly disabled".
    embed_query_prefix: str | None = None
    embed_doc_prefix: str | None = None

    # --- Vector store ---------------------------------------------------
    # Empty url => embedded Qdrant under ``data_dir/qdrant``, no server needed.
    qdrant_url: str = ""
    qdrant_api_key: str = Field(default="", repr=False)
    collection: str = "reed_chunks"

    # --- Retrieval ------------------------------------------------------
    top_k: int = Field(default=4, ge=1, le=50)
    fetch_k: int = Field(default=20, ge=1, le=200)
    retrieval_mode: RetrievalMode = "hybrid"
    rerank_enabled: bool = False
    rerank_backend: Literal["fastembed", "sentence_transformers"] = "sentence_transformers"
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    sparse_model: str = "Qdrant/bm25"
    max_context_chars: int = Field(default=24_000, ge=1_000, le=200_000)
    max_chunks_per_document: int = Field(default=2, ge=1, le=20)
    diversity_enabled: bool = True
    diversity_lambda: float = Field(default=0.7, ge=0.0, le=1.0)
    # ``None`` selects the calibrated preset when its exact score domain is in
    # use. A numeric value is always an explicit operator override.
    min_evidence_score: float | None = Field(default=None, ge=0.0, le=1.0)

    # --- Ingestion ------------------------------------------------------
    chunk_size: int = Field(default=1000, ge=100, le=8000)
    chunk_overlap: int = Field(default=150, ge=0, le=2000)
    max_upload_mb: int = Field(default=25, ge=1, le=500)
    max_multipart_overhead_kb: int = Field(default=256, ge=16, le=4_096)
    max_document_chars: int = Field(default=5_000_000, ge=1_000, le=50_000_000)
    max_pdf_pages: int = Field(default=1_000, ge=1, le=10_000)
    parser_timeout_seconds: float = Field(default=30.0, ge=1.0, le=300.0)
    parser_cpu_seconds: int = Field(default=20, ge=1, le=300)
    parser_memory_mb: int = Field(default=1_024, ge=128, le=4_096)

    # --- Server ---------------------------------------------------------
    data_dir: Path = Path("./data")
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    log_level: LogLevel = "INFO"
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    api_key: str = Field(default="", repr=False)
    cors_origins: str = ""
    provider_timeout_seconds: float = Field(default=120.0, ge=1.0, le=600.0)
    startup_grace_seconds: float = Field(default=2.0, ge=0.0, le=10.0)
    readiness_retry_initial_seconds: float = Field(default=2.0, ge=0.05, le=60.0)
    readiness_retry_max_seconds: float = Field(default=60.0, ge=0.05, le=600.0)
    readiness_chat_ttl_seconds: float = Field(default=15.0, ge=0.0, le=300.0)
    # A readiness probe answers on an orchestrator's budget of seconds; it must
    # not inherit the minutes-long generation timeout.
    readiness_probe_timeout_seconds: float = Field(default=5.0, ge=0.5, le=60.0)
    max_output_tokens: int = Field(default=1_024, ge=64, le=8_192)
    max_concurrent_asks: int = Field(default=8, ge=1, le=100)
    max_concurrent_ingestions: int = Field(default=2, ge=1, le=32)
    max_queued_ingestions: int = Field(default=16, ge=1, le=1_000)
    ask_rate_limit_per_minute: int = Field(default=60, ge=0, le=10_000)
    upload_rate_limit_per_minute: int = Field(default=20, ge=0, le=10_000)
    # Retrieval without generation costs one embedding call and one query, so
    # it gets its own, looser bucket rather than competing with generation.
    search_rate_limit_per_minute: int = Field(default=120, ge=0, le=10_000)
    sse_queue_size: int = Field(default=64, ge=4, le=4_096)
    max_json_body_kb: int = Field(default=128, ge=16, le=1_024)
    # Compose-only sizing knob, kept here so every REED_* setting has one
    # documented source of truth and parity tests can prevent drift.
    tmpfs_size: str = "64m"

    # --- Evaluation -----------------------------------------------------
    eval_judge_profile: Profile = "openai"
    eval_judge_model: str = ""

    @field_validator("log_level", mode="before")
    @classmethod
    def _normalise_log_level(cls, value: object) -> object:
        return value.upper() if isinstance(value, str) else value

    @field_validator("api_key")
    @classmethod
    def _api_key_is_canonical_ascii(cls, value: str) -> str:
        """Keep one unambiguous wire representation for the shared secret.

        HTTP clients disagree on how to encode non-ASCII header strings. Accepting
        both UTF-8 and latin-1 makes distinct human strings aliases for the same
        bytes, so configured keys are deliberately restricted to portable ASCII.
        """
        if value and not value.isascii():
            raise ValueError("REED_API_KEY must contain ASCII characters only")
        return value

    @model_validator(mode="after")
    def _validate_related_fields(self) -> Settings:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be smaller than "
                f"chunk_size ({self.chunk_size})"
            )
        if self.max_context_chars < self.chunk_size:
            raise ValueError(
                f"max_context_chars ({self.max_context_chars}) must be at least "
                f"chunk_size ({self.chunk_size})"
            )
        if self.readiness_retry_max_seconds < self.readiness_retry_initial_seconds:
            raise ValueError(
                "readiness_retry_max_seconds must be at least readiness_retry_initial_seconds"
            )
        return self

    @property
    def chat_model_name(self) -> str:
        """The model that answers questions, whichever profile is active."""
        if self.profile == "openai":
            return self.openai_chat_model
        if self.profile == "local":
            return self.ollama_chat_model
        return "fake-chat"

    @property
    def embed_model_name(self) -> str:
        if self.profile == "openai":
            return self.openai_embed_model
        if self.profile == "local":
            return self.ollama_embed_model
        return "fake-embeddings"

    @property
    def resolved_query_prefix(self) -> str:
        if self.embed_query_prefix is not None:
            return self.embed_query_prefix
        if self.profile == "local" and "embeddinggemma" in self.ollama_embed_model.lower():
            return EMBEDDINGGEMMA_QUERY_PREFIX
        if self.profile == "local" and "qwen3-embedding" in self.ollama_embed_model.lower():
            return QWEN3_EMBED_QUERY_PREFIX
        return ""

    @property
    def resolved_doc_prefix(self) -> str:
        if self.embed_doc_prefix is not None:
            return self.embed_doc_prefix
        if self.profile == "local" and "embeddinggemma" in self.ollama_embed_model.lower():
            return EMBEDDINGGEMMA_DOC_PREFIX
        return ""

    @property
    def qdrant_path(self) -> Path:
        """Folder backing the embedded Qdrant instance."""
        return self.data_dir / "qdrant"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def registry_path(self) -> Path:
        return self.data_dir / "reed.db"

    @property
    def cors_origin_list(self) -> list[str]:
        return list(dict.fromkeys(o.strip() for o in self.cors_origins.split(",") if o.strip()))

    def min_evidence_score_for(self, *, mode: str, reranked: bool) -> float:
        """Return a threshold valid for the score domain actually queried.

        The mode is a parameter because a caller may query a mode other than
        the configured one; the calibration belongs to the scores in front of
        it, not to the settings object.
        """
        if self.min_evidence_score is not None:
            return self.min_evidence_score
        if (
            self.profile == "local"
            and "embeddinggemma" in self.ollama_embed_model.lower()
            and mode == "hybrid"
            and not reranked
        ):
            # RRF score calibrated on the v0.3 golden set. Dense similarity,
            # sparse and cross-encoder scores are not numerically comparable.
            # Keep the exact calibrated float: Python's ``5 / 6`` is one ULP
            # higher and would incorrectly reject candidates on the boundary.
            return EMBEDDINGGEMMA_HYBRID_MIN_SCORE
        return 0.0

    @property
    def resolved_min_evidence_score(self) -> float:
        """The threshold for the configured retrieval mode."""
        return self.min_evidence_score_for(mode=self.retrieval_mode, reranked=self.rerank_enabled)

    def validate_ready(self) -> None:
        """Fail fast at startup on configurations that cannot possibly work."""
        if self.profile == "openai" and not self.openai_api_key:
            raise RuntimeError(
                "REED_PROFILE=openai requires OPENAI_API_KEY. "
                "Set the key, or run fully offline with REED_PROFILE=local."
            )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
