from __future__ import annotations

from types import SimpleNamespace

import pytest

from reed.indexes import cleanup, rollback
from reed.ingest.registry import IndexGeneration


def generation(
    name: str, status: str, fingerprint: dict[str, object] | None = None
) -> IndexGeneration:
    return IndexGeneration(
        id=name,
        logical_collection="logical",
        physical_collection=f"physical-{name}",
        fingerprint=fingerprint or {"schema": 4},
        status=status,  # type: ignore[arg-type]
        document_count=1,
        chunk_count=2,
        error=None,
        created_at="2026-08-04T00:00:00+00:00",
        completed_at=None,
        activated_at=None,
    )


class FakeRegistry:
    def __init__(self, generations: list[IndexGeneration]) -> None:
        self.generations = generations
        self.deleted: list[str] = []

    def list_generations(self, _collection: str) -> list[IndexGeneration]:
        return self.generations

    def delete_generation(self, generation_id: str) -> bool:
        self.deleted.append(generation_id)
        return True


class FakeQdrant:
    def __init__(self, existing: set[str]) -> None:
        self.existing = existing
        self.deleted: list[str] = []

    def collection_exists(self, collection: str) -> bool:
        return collection in self.existing

    def delete_collection(self, collection: str) -> None:
        self.deleted.append(collection)


def test_cleanup_retains_active_and_requested_previous_generation() -> None:
    active = generation("active", "active")
    previous = generation("previous", "previous")
    old = generation("old", "previous")
    failed = generation("failed", "failed")
    registry = FakeRegistry([active, previous, old, failed])
    qdrant = FakeQdrant({item.physical_collection for item in registry.generations})
    services = SimpleNamespace(
        settings=SimpleNamespace(collection="logical"), registry=registry, qdrant=qdrant
    )

    removed = cleanup(services, keep=2)  # type: ignore[arg-type]

    assert [item.id for item in removed] == ["old", "failed"]
    assert registry.deleted == ["old", "failed"]
    assert qdrant.deleted == ["physical-old", "physical-failed"]


def test_cleanup_validates_keep_and_skips_an_already_missing_collection() -> None:
    failed = generation("failed", "failed")
    registry = FakeRegistry([failed])
    services = SimpleNamespace(
        settings=SimpleNamespace(collection="logical"),
        registry=registry,
        qdrant=FakeQdrant(set()),
    )

    with pytest.raises(ValueError, match="at least 1"):
        cleanup(services, keep=0)  # type: ignore[arg-type]
    assert cleanup(services, keep=1) == [failed]  # type: ignore[arg-type]
    assert services.qdrant.deleted == []


@pytest.mark.parametrize(
    ("active", "previous", "message"),
    [
        (None, generation("previous", "previous"), "no active"),
        (generation("active", "active"), None, "no previous"),
    ],
)
def test_rollback_requires_both_generations(
    active: IndexGeneration | None,
    previous: IndexGeneration | None,
    message: str,
) -> None:
    registry = SimpleNamespace(
        active_generation=lambda _logical: active,
        previous_generation=lambda _logical: previous,
    )
    services = SimpleNamespace(
        settings=SimpleNamespace(collection="logical"),
        registry=registry,
        qdrant=FakeQdrant(set()),
    )

    with pytest.raises(RuntimeError, match=message):
        rollback(services)  # type: ignore[arg-type]


def test_rollback_requires_the_collection_and_matching_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active = generation("active", "active")
    previous = generation("previous", "previous")
    registry = SimpleNamespace(
        active_generation=lambda _logical: active,
        previous_generation=lambda _logical: previous,
    )
    services = SimpleNamespace(
        settings=SimpleNamespace(collection="logical"),
        registry=registry,
        qdrant=FakeQdrant(set()),
    )

    with pytest.raises(RuntimeError, match="no longer exists"):
        rollback(services)  # type: ignore[arg-type]

    services.qdrant.existing.add(previous.physical_collection)
    monkeypatch.setattr("reed.indexes.desired_fingerprint", lambda _services: ({"schema": 5}, 32))
    with pytest.raises(RuntimeError, match="different embedding configuration"):
        rollback(services)  # type: ignore[arg-type]
