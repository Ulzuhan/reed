from __future__ import annotations

import io
import json

import pytest

from reed.config import Settings
from reed.model_identity import ModelIdentityError, resolve_embedding_identity


class Response(io.BytesIO):
    def __enter__(self) -> Response:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def test_ollama_tag_resolves_to_digest_and_quantization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "models": [
            {
                "name": "embeddinggemma:latest",
                "digest": "85462619ee72deadbeef",
                "details": {"quantization_level": "Q8_0"},
            }
        ]
    }
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_args, **_kwargs: Response(json.dumps(payload).encode()),
    )
    settings = Settings(profile="local", ollama_embed_model="embeddinggemma", _env_file=None)

    identity = resolve_embedding_identity(settings)

    assert identity.digest == "85462619ee72deadbeef"
    assert identity.quantization == "Q8_0"


def test_explicit_digest_avoids_a_model_api_call(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *_args, **_kwargs: pytest.fail("must not call Ollama")
    )
    settings = Settings(
        profile="local",
        embed_model_digest="pinned",
        embed_model_quantization="Q4_K_M",
        _env_file=None,
    )

    assert resolve_embedding_identity(settings).digest == "pinned"


def test_missing_ollama_model_is_an_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_args, **_kwargs: Response(json.dumps({"models": []}).encode()),
    )
    settings = Settings(profile="local", _env_file=None)

    with pytest.raises(ModelIdentityError, match="REED_EMBED_MODEL_DIGEST"):
        resolve_embedding_identity(settings)
