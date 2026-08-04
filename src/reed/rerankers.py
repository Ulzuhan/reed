"""Optional reranker backends kept out of Reed's base installation."""

from __future__ import annotations

from typing import Protocol

from reed.config import Settings


class Reranker(Protocol):
    def rerank(self, query: str, documents: list[str]) -> list[float]: ...


class FastEmbedReranker:
    def __init__(self, model_name: str) -> None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self._model = TextCrossEncoder(model_name=model_name)

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        return [float(score) for score in self._model.rerank(query, documents)]


class SentenceTransformersReranker:
    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                "The multilingual reranker requires the optional extra: "
                "install 'reed[rerank]' before setting REED_RERANK_ENABLED=true"
            ) from exc
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        pairs = [(query, document) for document in documents]
        scores = self._model.predict(pairs)
        return [float(score) for score in scores]


def build_reranker(settings: Settings) -> Reranker:
    if settings.rerank_backend == "fastembed":
        return FastEmbedReranker(settings.rerank_model)
    return SentenceTransformersReranker(settings.rerank_model)
