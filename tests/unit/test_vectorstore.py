from __future__ import annotations

import pytest

from reed.config import Settings
from reed.ingest.extraction import EXTRACTION_VERSION
from reed.model_identity import ModelIdentity
from reed.rag.vectorstore import collection_fingerprint

IDENTITY = ModelIdentity(digest="sha256:abc", revision="", quantization="Q8_0")


def fingerprint() -> dict[str, object]:
    return collection_fingerprint(Settings(profile="fake", _env_file=None), 32, IDENTITY)


def test_the_fingerprint_covers_the_extraction_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    # Models and chunking were already covered. Changing how a file becomes
    # embedding input changes the vectors just as thoroughly.
    before = fingerprint()
    assert before["extraction"] == EXTRACTION_VERSION

    monkeypatch.setattr("reed.rag.vectorstore.EXTRACTION_VERSION", EXTRACTION_VERSION + 1)

    assert fingerprint() != before
