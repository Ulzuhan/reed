"""Resolve immutable provenance for the configured embedding model."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass

from reed.config import Settings


@dataclass(frozen=True, slots=True)
class ModelIdentity:
    digest: str
    revision: str
    quantization: str


class ModelIdentityError(RuntimeError):
    """A mutable local model tag could not be resolved to immutable bytes."""


def resolve_embedding_identity(settings: Settings) -> ModelIdentity:
    """Return stable model identity without silently trusting a mutable tag."""
    if settings.profile == "fake":
        return ModelIdentity(
            digest=settings.embed_model_digest or "reed-deterministic-fake-v1",
            revision=settings.embed_model_revision,
            quantization=settings.embed_model_quantization,
        )
    if settings.profile == "openai":
        return ModelIdentity(
            digest=settings.embed_model_digest,
            revision=settings.embed_model_revision or "provider-managed",
            quantization=settings.embed_model_quantization,
        )
    if settings.embed_model_digest:
        return ModelIdentity(
            digest=settings.embed_model_digest,
            revision=settings.embed_model_revision,
            quantization=settings.embed_model_quantization,
        )

    payload = ollama_tags(settings)
    configured = settings.ollama_embed_model
    wanted = {configured}
    if ":" not in configured:
        wanted.add(f"{configured}:latest")
    models = payload.get("models")
    if not isinstance(models, list):
        raise ModelIdentityError("Ollama returned a model list without a 'models' array")
    for item in models:
        if not isinstance(item, dict):
            continue
        names = {str(item.get("name", "")), str(item.get("model", ""))}
        if not wanted.intersection(names):
            continue
        raw_details = item.get("details")
        details: dict[object, object] = raw_details if isinstance(raw_details, dict) else {}
        digest = str(item.get("digest", ""))
        if not digest:
            break
        return ModelIdentity(
            digest=digest,
            revision=settings.embed_model_revision,
            quantization=settings.embed_model_quantization
            or str(details.get("quantization_level", "")),
        )
    raise ModelIdentityError(
        f"Ollama did not report an immutable digest for '{configured}'. Pull the exact model "
        "tag or set REED_EMBED_MODEL_DIGEST explicitly."
    )


def ollama_tags(settings: Settings, *, timeout: float | None = None) -> dict[str, object]:
    """List Ollama's local models.

    ``timeout`` defaults to the generation timeout; readiness probes pass
    their own short one, because an orchestrator polls /ready on a budget of
    seconds, not the minutes a generation may take.
    """
    url = f"{settings.ollama_base_url.rstrip('/')}/api/tags"
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    if timeout is None:
        timeout = settings.provider_timeout_seconds
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            value = json.load(response)
    except (OSError, urllib.error.URLError, ValueError) as exc:
        raise ModelIdentityError(
            f"Could not resolve the Ollama digest at {url}: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise ModelIdentityError(f"Ollama returned an invalid model list from {url}")
    return value
