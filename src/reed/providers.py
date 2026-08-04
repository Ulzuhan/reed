"""Provider abstraction: the same RAG pipeline over OpenAI, Ollama or fakes.

Everything downstream depends on the two ``langchain_core`` interfaces
(``BaseChatModel`` and ``Embeddings``), so swapping providers is a config
change rather than a code change.
"""

from __future__ import annotations

from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings
from langchain_core.language_models import BaseChatModel, FakeListChatModel

from reed.config import Profile, Settings

FAKE_EMBEDDING_DIM = 32

# Canned answer for the `fake` profile: it carries a citation marker so the
# whole streaming + citation path can be tested without a real model.
FAKE_ANSWER = "This is a canned answer from the fake profile [1]."

# OpenAI reasoning models reject an explicit temperature.
_FIXED_TEMPERATURE_PREFIXES = ("gpt-5", "o1", "o3", "o4")


class PrefixedEmbeddings(Embeddings):
    """Wraps an embedding model to add the task prefixes it was trained with.

    EmbeddingGemma asymmetrically prefixes queries and documents; OpenAI models
    take none. Keeping this out of the call sites means the rest of the code
    never has to know which model is behind the interface.
    """

    def __init__(self, inner: Embeddings, query_prefix: str = "", doc_prefix: str = "") -> None:
        self.inner = inner
        self.query_prefix = query_prefix
        self.doc_prefix = doc_prefix

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not self.doc_prefix:
            return self.inner.embed_documents(texts)
        return self.inner.embed_documents([f"{self.doc_prefix}{t}" for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self.inner.embed_query(f"{self.query_prefix}{text}")

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        if not self.doc_prefix:
            return await self.inner.aembed_documents(texts)
        return await self.inner.aembed_documents([f"{self.doc_prefix}{t}" for t in texts])

    async def aembed_query(self, text: str) -> list[float]:
        return await self.inner.aembed_query(f"{self.query_prefix}{text}")


def build_chat_model(settings: Settings, profile: Profile | None = None) -> BaseChatModel:
    """Build the chat model for ``profile`` (defaults to the active one)."""
    profile = profile or settings.profile

    if profile == "fake":
        return FakeListChatModel(responses=[FAKE_ANSWER])

    if profile == "local":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=settings.ollama_chat_model,
            base_url=settings.ollama_base_url,
            temperature=settings.temperature,
            # Qwen3.5 emits <think> blocks otherwise, which would leak into the
            # token stream and the citations.
            reasoning=False,
        )

    from langchain_openai import ChatOpenAI

    if settings.openai_chat_model.startswith(_FIXED_TEMPERATURE_PREFIXES):
        return ChatOpenAI(
            model=settings.openai_chat_model,
            openai_api_key=settings.openai_api_key,
        )
    return ChatOpenAI(
        model=settings.openai_chat_model,
        openai_api_key=settings.openai_api_key,
        temperature=settings.temperature,
    )


def build_embeddings(settings: Settings, profile: Profile | None = None) -> Embeddings:
    """Build the embedding model for ``profile``, prefixes already applied."""
    profile = profile or settings.profile

    if profile == "fake":
        return DeterministicFakeEmbedding(size=FAKE_EMBEDDING_DIM)

    inner: Embeddings
    if profile == "local":
        from langchain_ollama import OllamaEmbeddings

        inner = OllamaEmbeddings(
            model=settings.ollama_embed_model,
            base_url=settings.ollama_base_url,
        )
    else:
        from langchain_openai import OpenAIEmbeddings

        inner = OpenAIEmbeddings(
            model=settings.openai_embed_model,
            openai_api_key=settings.openai_api_key,
        )

    query_prefix = settings.resolved_query_prefix
    doc_prefix = settings.resolved_doc_prefix
    if not query_prefix and not doc_prefix:
        return inner
    return PrefixedEmbeddings(inner, query_prefix=query_prefix, doc_prefix=doc_prefix)


def embedding_dimension(embeddings: Embeddings) -> int:
    """Probe the embedding size instead of hardcoding it.

    1536 for text-embedding-3-small, 768 for EmbeddingGemma — getting this
    wrong only surfaces as a Qdrant error deep inside ingestion.
    """
    return len(embeddings.embed_query("dimension probe"))
