from __future__ import annotations

import pytest
from pydantic import ValidationError

from reed.config import (
    EMBEDDINGGEMMA_DOC_PREFIX,
    EMBEDDINGGEMMA_QUERY_PREFIX,
    Settings,
)


def make(**kwargs: object) -> Settings:
    return Settings(_env_file=None, **kwargs)  # type: ignore[call-arg,arg-type]


def test_openai_profile_requires_a_key() -> None:
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        make(profile="openai", openai_api_key="").validate_ready()


def test_local_profile_needs_no_key() -> None:
    make(profile="local").validate_ready()


def test_embeddinggemma_gets_task_prefixes_by_default() -> None:
    settings = make(profile="local", ollama_embed_model="embeddinggemma")
    assert settings.resolved_query_prefix == EMBEDDINGGEMMA_QUERY_PREFIX
    assert settings.resolved_doc_prefix == EMBEDDINGGEMMA_DOC_PREFIX


def test_prefixes_can_be_disabled_explicitly() -> None:
    settings = make(
        profile="local",
        ollama_embed_model="embeddinggemma",
        embed_query_prefix="",
        embed_doc_prefix="",
    )
    assert settings.resolved_query_prefix == ""
    assert settings.resolved_doc_prefix == ""


def test_openai_embeddings_take_no_prefixes() -> None:
    settings = make(profile="openai", openai_api_key="sk-test")
    assert settings.resolved_query_prefix == ""
    assert settings.resolved_doc_prefix == ""


def test_overlap_must_be_smaller_than_chunk_size() -> None:
    with pytest.raises(ValidationError, match="chunk_overlap"):
        make(chunk_size=500, chunk_overlap=500)


def test_fetch_k_only_applies_when_reranking() -> None:
    assert make(top_k=4, fetch_k=20, rerank_enabled=False).effective_fetch_k == 4
    assert make(top_k=4, fetch_k=20, rerank_enabled=True).effective_fetch_k == 20


def test_paths_derive_from_data_dir(tmp_path: object) -> None:
    settings = make(data_dir=tmp_path)
    assert settings.qdrant_path.name == "qdrant"
    assert settings.uploads_dir.name == "uploads"
    assert settings.registry_path.name == "reed.db"


def test_cors_origins_are_split_and_trimmed() -> None:
    settings = make(cors_origins="http://a.test, http://b.test ,")
    assert settings.cors_origin_list == ["http://a.test", "http://b.test"]


def test_model_names_follow_the_profile() -> None:
    assert make(profile="local", ollama_chat_model="qwen3.5:4b").chat_model_name == "qwen3.5:4b"
    assert make(profile="fake").chat_model_name == "fake-chat"
