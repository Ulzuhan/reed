from __future__ import annotations

import pytest

from reed.config import Settings
from reed.rerankers import build_reranker


def test_multilingual_reranker_has_an_actionable_optional_extra_error() -> None:
    settings = Settings(
        profile="fake",
        rerank_backend="sentence_transformers",
        rerank_model="BAAI/bge-reranker-v2-m3",
        _env_file=None,
    )

    with pytest.raises(RuntimeError, match=r"reed\[rerank\]"):
        build_reranker(settings)
