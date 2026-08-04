"""Safe index generation lifecycle: build, activate, roll back and clean up."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from pathlib import Path

from qdrant_client import models

from reed.ingest.pipeline import index_record_into, sha256_file
from reed.ingest.registry import IndexGeneration
from reed.model_identity import resolve_embedding_identity
from reed.providers import embedding_dimension
from reed.rag.vectorstore import (
    COMMITTED_PAYLOAD_KEY,
    build_sparse_embeddings,
    build_vectorstore,
    collection_fingerprint,
    delete_collection_safely,
    ensure_collection,
)
from reed.services import Services

_COLLECTION_SAFE = re.compile(r"[^A-Za-z0-9_-]+")


@dataclass(frozen=True, slots=True)
class ReindexResult:
    generation: IndexGeneration
    previous: IndexGeneration | None


def desired_fingerprint(services: Services) -> tuple[dict[str, object], int]:
    embeddings = services.embeddings
    dimension = embedding_dimension(embeddings)
    identity = resolve_embedding_identity(services.settings)
    return collection_fingerprint(services.settings, dimension, identity), dimension


def reindex(services: Services) -> ReindexResult:
    """Build a complete candidate without mutating the active collection."""
    settings = services.settings
    fingerprint, dimension = desired_fingerprint(services)
    generation_id = uuid.uuid4().hex
    logical = settings.collection
    prefix = _COLLECTION_SAFE.sub("_", logical).strip("_") or "reed_chunks"
    physical = f"{prefix}__g_{generation_id[:12]}"
    generation = services.registry.create_generation(
        generation_id=generation_id,
        logical_collection=logical,
        physical_collection=physical,
        fingerprint=fingerprint,
    )
    previous = services.registry.active_generation(logical)

    try:
        ensure_collection(
            services.qdrant,
            settings,
            dimension,
            collection_name=physical,
            fingerprint=fingerprint,
        )
        sparse = build_sparse_embeddings(settings)
        store = build_vectorstore(
            services.qdrant,
            settings,
            services.embeddings,
            sparse,
            collection_name=physical,
        )
        records = list(services.registry.list_by_status({"ready"}))
        chunks = 0
        for record in records:
            if not record.stored_path:
                raise RuntimeError(
                    f"{record.filename}: stored original is missing from the registry"
                )
            path = Path(record.stored_path)
            if not path.is_file():
                raise RuntimeError(f"{record.filename}: stored original no longer exists at {path}")
            if sha256_file(path) != record.sha256:
                raise RuntimeError(
                    f"{record.filename}: stored original no longer matches its SHA-256"
                )
            indexed, _ = index_record_into(
                services,
                record,
                store=store,
                collection_name=physical,
                embeddings=services.embeddings,
                sparse_embeddings=sparse,
            )
            chunks += indexed

        committed = services.qdrant.count(
            collection_name=physical,
            count_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key=COMMITTED_PAYLOAD_KEY,
                        match=models.MatchValue(value=True),
                    )
                ]
            ),
            exact=True,
        ).count
        if int(committed) != chunks:
            raise RuntimeError(
                f"candidate verification failed: expected {chunks} committed chunks, "
                f"found {committed}"
            )
        services.registry.set_generation_counts(
            generation.id,
            document_count=len(records),
            chunk_count=chunks,
        )
        active = services.registry.activate_generation(generation.id)
        services.select_generation(active.physical_collection)
        return ReindexResult(generation=active, previous=previous)
    except Exception as exc:
        services.registry.fail_generation(generation.id, f"{type(exc).__name__}: {exc}")
        if services.qdrant.collection_exists(physical):
            delete_collection_safely(services.qdrant, physical)
        raise


def rollback(services: Services) -> tuple[IndexGeneration, IndexGeneration]:
    """Reactivate the most recently active compatible generation."""
    logical = services.settings.collection
    current = services.registry.active_generation(logical)
    target = services.registry.previous_generation(logical)
    if current is None:
        raise RuntimeError("There is no active index generation")
    if target is None:
        raise RuntimeError("There is no previous index generation to roll back to")
    if not services.qdrant.collection_exists(target.physical_collection):
        raise RuntimeError(
            f"Previous collection '{target.physical_collection}' no longer exists; "
            "run a full reindex instead"
        )
    expected, _ = desired_fingerprint(services)
    if target.fingerprint != expected:
        raise RuntimeError(
            "The previous generation uses a different embedding configuration. Restore that "
            "configuration (model tag/digest, prefixes and chunking) before rolling back."
        )
    activated = services.registry.activate_generation(target.id)
    services.select_generation(activated.physical_collection)
    return activated, current


def cleanup(services: Services, *, keep: int = 2) -> list[IndexGeneration]:
    """Remove old physical generations, never the active one."""
    if keep < 1:
        raise ValueError("keep must be at least 1")
    generations = services.registry.list_generations(services.settings.collection)
    retained = 0
    removed: list[IndexGeneration] = []
    for generation in generations:
        if generation.status == "active":
            retained += 1
            continue
        if generation.status == "previous" and retained < keep:
            retained += 1
            continue
        if services.qdrant.collection_exists(generation.physical_collection):
            delete_collection_safely(services.qdrant, generation.physical_collection)
        if services.registry.delete_generation(generation.id):
            removed.append(generation)
    return removed
