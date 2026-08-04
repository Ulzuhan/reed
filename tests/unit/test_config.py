from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from reed.config import (
    EMBEDDINGGEMMA_DOC_PREFIX,
    EMBEDDINGGEMMA_QUERY_PREFIX,
    QWEN3_EMBED_QUERY_PREFIX,
    Settings,
)


def make(**kwargs: Any) -> Settings:
    return Settings(_env_file=None, **kwargs)


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


def test_qwen3_embedding_gets_its_asymmetric_query_instruction() -> None:
    settings = make(profile="local", ollama_embed_model="qwen3-embedding:0.6b")
    assert settings.resolved_query_prefix == QWEN3_EMBED_QUERY_PREFIX
    assert settings.resolved_doc_prefix == ""


def test_openai_embeddings_take_no_prefixes() -> None:
    settings = make(profile="openai", openai_api_key="sk-test")
    assert settings.resolved_query_prefix == ""
    assert settings.resolved_doc_prefix == ""


def test_overlap_must_be_smaller_than_chunk_size() -> None:
    with pytest.raises(ValidationError, match="chunk_overlap"):
        make(chunk_size=500, chunk_overlap=500)


def test_context_budget_must_fit_at_least_one_chunk() -> None:
    with pytest.raises(ValidationError, match="max_context_chars"):
        make(chunk_size=2_000, max_context_chars=1_000)


def test_api_keys_have_one_portable_wire_encoding() -> None:
    with pytest.raises(ValidationError, match="ASCII"):
        make(api_key="contraseña")


def test_assignment_cannot_bypass_validation() -> None:
    settings = make(profile="fake")
    with pytest.raises(ValidationError):
        settings.port = 70_000


def test_fetch_k_only_applies_when_reranking() -> None:
    assert make(top_k=4, fetch_k=20, rerank_enabled=False).effective_fetch_k == 4
    assert make(top_k=4, fetch_k=20, rerank_enabled=True).effective_fetch_k == 20


def test_evidence_threshold_is_calibrated_only_for_its_score_domain() -> None:
    calibrated = make(profile="local", ollama_embed_model="embeddinggemma", retrieval_mode="hybrid")
    assert calibrated.resolved_min_evidence_score == pytest.approx(5 / 6)
    assert make(profile="fake").resolved_min_evidence_score == 0
    assert (
        make(
            profile="local", ollama_embed_model="embeddinggemma", retrieval_mode="dense"
        ).resolved_min_evidence_score
        == 0
    )
    assert (
        make(
            profile="local", ollama_embed_model="embeddinggemma", rerank_enabled=True
        ).resolved_min_evidence_score
        == 0
    )
    assert (
        calibrated.model_copy(update={"min_evidence_score": 0.42}).resolved_min_evidence_score
        == 0.42
    )


def test_paths_derive_from_data_dir(tmp_path: object) -> None:
    settings = make(data_dir=tmp_path)
    assert settings.qdrant_path.name == "qdrant"
    assert settings.uploads_dir.name == "uploads"
    assert settings.registry_path.name == "reed.db"


def test_cors_origins_are_split_and_trimmed() -> None:
    settings = make(cors_origins="http://a.test, http://b.test, http://a.test ,")
    assert settings.cors_origin_list == ["http://a.test", "http://b.test"]


def test_model_names_follow_the_profile() -> None:
    assert make(profile="local", ollama_chat_model="qwen3.5:4b").chat_model_name == "qwen3.5:4b"
    assert make(profile="fake").chat_model_name == "fake-chat"
