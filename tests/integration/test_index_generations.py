from __future__ import annotations

from pathlib import Path

import pytest

from reed.config import Settings
from reed.indexes import reindex, rollback
from reed.ingest.pipeline import ingest_path
from reed.rag.retriever import retrieve
from reed.services import build_services

pytestmark = pytest.mark.integration


def test_changed_digest_reindexes_without_touching_the_active_collection(
    settings: Settings, tmp_path: Path
) -> None:
    path = tmp_path / "policy.md"
    path.write_text("# Policy\n\nThe launch code is ORCHID.", encoding="utf-8")
    first = build_services(settings)
    try:
        assert ingest_path(first, path).status == "ready"
        original = first.registry.active_generation(settings.collection)
        assert original is not None
    finally:
        first.close()

    changed = settings.model_copy(update={"embed_model_digest": "same-tag-new-digest"})
    second = build_services(changed)
    try:
        result = reindex(second)
        assert result.previous is not None
        assert result.previous.id == original.id
        assert result.generation.physical_collection != original.physical_collection
        assert second.qdrant.collection_exists(original.physical_collection)
        assert retrieve(second, "What is the launch code?")[0].filename == "policy.md"
    finally:
        second.close()

    restored = build_services(settings)
    try:
        activated, replaced = rollback(restored)
        assert activated.id == original.id
        assert replaced.id == result.generation.id
        assert retrieve(restored, "What is the launch code?")[0].filename == "policy.md"
    finally:
        restored.close()


def test_failed_reindex_keeps_active_generation_and_deletes_candidate(
    settings: Settings, tmp_path: Path
) -> None:
    path = tmp_path / "policy.md"
    path.write_text("# Policy\n\nKeep the current index safe.", encoding="utf-8")
    first = build_services(settings)
    try:
        assert ingest_path(first, path).status == "ready"
        active = first.registry.active_generation(settings.collection)
        assert active is not None
        stored = Path(first.registry.list_by_status({"ready"})[0].stored_path or "")
    finally:
        first.close()
    stored.unlink()

    changed = settings.model_copy(update={"embed_model_digest": "candidate"})
    second = build_services(changed)
    try:
        with pytest.raises(RuntimeError, match="no longer exists"):
            reindex(second)
        assert second.registry.active_generation(settings.collection).id == active.id  # type: ignore[union-attr]
        failed = [
            item
            for item in second.registry.list_generations(settings.collection)
            if item.status == "failed"
        ]
        assert len(failed) == 1
        assert not second.qdrant.collection_exists(failed[0].physical_collection)
    finally:
        second.close()
