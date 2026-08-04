from __future__ import annotations

from typing import Any

from langchain_core.embeddings import Embeddings

from reed.config import Settings
from reed.providers import (
    FAKE_EMBEDDING_DIM,
    PrefixedEmbeddings,
    build_chat_model,
    build_embeddings,
    embedding_dimension,
)


class RecordingEmbeddings(Embeddings):
    """Captures the exact strings handed to the underlying model."""

    def __init__(self) -> None:
        self.seen: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.seen.extend(texts)
        return [[0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        self.seen.append(text)
        return [0.0]


def make(**kwargs: Any) -> Settings:
    return Settings(_env_file=None, **kwargs)


def test_prefixes_are_applied_asymmetrically() -> None:
    inner = RecordingEmbeddings()
    wrapped = PrefixedEmbeddings(inner, query_prefix="Q: ", doc_prefix="D: ")

    wrapped.embed_query("how many days of leave?")
    wrapped.embed_documents(["employees get 25 days"])

    assert inner.seen == ["Q: how many days of leave?", "D: employees get 25 days"]


def test_documents_skip_the_wrapper_when_no_doc_prefix() -> None:
    inner = RecordingEmbeddings()
    wrapped = PrefixedEmbeddings(inner, query_prefix="Q: ", doc_prefix="")

    wrapped.embed_documents(["untouched"])

    assert inner.seen == ["untouched"]


def test_fake_profile_builds_offline_models() -> None:
    settings = make(profile="fake")

    embeddings = build_embeddings(settings)
    assert embedding_dimension(embeddings) == FAKE_EMBEDDING_DIM
    # Deterministic: the same text must always land on the same vector.
    assert embeddings.embed_query("stable") == embeddings.embed_query("stable")

    chat = build_chat_model(settings)
    assert "[1]" in str(chat.invoke("anything").content)


def test_local_profile_wraps_embeddinggemma_with_prefixes() -> None:
    settings = make(profile="local", ollama_embed_model="embeddinggemma")
    assert isinstance(build_embeddings(settings), PrefixedEmbeddings)


def test_openai_profile_returns_the_bare_embeddings() -> None:
    settings = make(profile="openai", openai_api_key="sk-test")
    assert not isinstance(build_embeddings(settings), PrefixedEmbeddings)


def test_reasoning_models_get_no_temperature() -> None:
    # gpt-5 rejects an explicit temperature; older models accept it.
    assert build_chat_model(make(profile="openai", openai_api_key="sk-test")).temperature is None  # type: ignore[attr-defined]
    legacy = build_chat_model(
        make(profile="openai", openai_api_key="sk-test", openai_chat_model="gpt-4.1-mini")
    )
    assert legacy.temperature == 0.1  # type: ignore[attr-defined]
