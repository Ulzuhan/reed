"""Application settings.

Every knob is a ``REED_*`` environment variable (or an entry in ``.env``). The
``profile`` picks sensible defaults for a whole provider stack; individual
settings can still be overridden one by one.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Profile = Literal["openai", "local", "fake"]

# EmbeddingGemma is trained with task prefixes and loses accuracy without them.
# https://ai.google.dev/gemma/docs/embeddinggemma/model_card
EMBEDDINGGEMMA_QUERY_PREFIX = "task: search result | query: "
EMBEDDINGGEMMA_DOC_PREFIX = "title: none | text: "


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="REED_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    profile: Profile = "openai"

    # --- OpenAI ---------------------------------------------------------
    # Read from the conventional OPENAI_API_KEY, not REED_OPENAI_API_KEY.
    openai_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("OPENAI_API_KEY", "openai_api_key"),
    )
    openai_chat_model: str = "gpt-5-mini"
    openai_embed_model: str = "text-embedding-3-small"

    # --- Ollama ---------------------------------------------------------
    ollama_base_url: str = "http://localhost:11434"
    ollama_chat_model: str = "qwen3.5:4b"
    ollama_embed_model: str = "embeddinggemma"

    # Task prefixes for embeddings. ``None`` means "use the profile default",
    # an empty string means "explicitly disabled".
    embed_query_prefix: str | None = None
    embed_doc_prefix: str | None = None

    # --- Vector store ---------------------------------------------------
    # Empty url => embedded Qdrant under ``data_dir/qdrant``, no server needed.
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    collection: str = "reed_chunks"

    # --- Retrieval ------------------------------------------------------
    top_k: int = Field(default=4, ge=1, le=50)
    fetch_k: int = Field(default=20, ge=1, le=200)
    rerank_enabled: bool = False
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    sparse_model: str = "Qdrant/bm25"

    # --- Ingestion ------------------------------------------------------
    chunk_size: int = Field(default=1000, ge=100, le=8000)
    chunk_overlap: int = Field(default=150, ge=0, le=2000)
    max_upload_mb: int = Field(default=25, ge=1, le=500)

    # --- Server ---------------------------------------------------------
    data_dir: Path = Path("./data")
    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "INFO"
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    api_key: str = ""
    cors_origins: str = ""

    # --- Evaluation -----------------------------------------------------
    eval_judge_profile: Profile = "openai"
    eval_judge_model: str = ""

    @field_validator("chunk_overlap")
    @classmethod
    def _overlap_below_size(cls, value: int, info: object) -> int:
        # ``info.data`` holds the fields validated so far (chunk_size precedes us).
        data = getattr(info, "data", {})
        size = data.get("chunk_size")
        if size is not None and value >= size:
            raise ValueError(f"chunk_overlap ({value}) must be smaller than chunk_size ({size})")
        return value

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
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def effective_fetch_k(self) -> int:
        """How many candidates retrieval pulls before (optional) reranking."""
        return max(self.fetch_k, self.top_k) if self.rerank_enabled else self.top_k

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
