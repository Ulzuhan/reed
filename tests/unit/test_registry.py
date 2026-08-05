from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from reed.ingest.registry import DocumentRegistry, NameConflictError


@pytest.fixture
def registry(tmp_path: Path) -> Iterator[DocumentRegistry]:
    instance = DocumentRegistry(tmp_path / "reed.db")
    try:
        yield instance
    finally:
        instance.close()


def add(registry: DocumentRegistry, doc_id: str = "d-abc", sha: str = "abc123") -> None:
    registry.add(doc_id=doc_id, filename="handbook.md", sha256=sha, size_bytes=42)


def test_new_documents_start_queued(registry: DocumentRegistry) -> None:
    record = registry.add(doc_id="d-abc", filename="a.md", sha256="abc", size_bytes=10)

    assert record.status == "queued"
    assert record.chunks == 0
    assert record.created_at


def test_status_transitions(registry: DocumentRegistry) -> None:
    add(registry)

    registry.mark_processing("d-abc")
    assert registry.get("d-abc").status == "parsing"  # type: ignore[union-attr]

    registry.mark_ready("d-abc", chunks=12, pages=3)
    ready = registry.get("d-abc")
    assert ready is not None
    assert (ready.status, ready.chunks, ready.pages) == ("ready", 12, 3)


def test_errors_are_recorded_and_cleared_on_retry(registry: DocumentRegistry) -> None:
    add(registry)

    registry.mark_error("d-abc", "ValueError: no chunks")
    assert registry.get("d-abc").error == "ValueError: no chunks"  # type: ignore[union-attr]

    retried, duplicate = registry.claim_upload(
        doc_id="d-new",
        filename="handbook.md",
        sha256="abc123",
        size_bytes=42,
        logical_id="l-retry",
        name="handbook.md",
    )
    assert duplicate is False
    assert retried.id == "d-abc"
    assert retried.status == "queued"
    assert retried.error is None


def test_a_late_error_cannot_demote_a_serving_version(registry: DocumentRegistry) -> None:
    # A failed post-commit cleanup or a double-reporting worker must not turn
    # a document with committed points into a dead one.
    add(registry)
    registry.mark_processing("d-abc")
    registry.mark_ready("d-abc", chunks=12, pages=3)

    assert registry.mark_error("d-abc", "late cleanup failure") is False

    record = registry.get("d-abc")
    assert record is not None
    assert record.status == "ready"
    assert record.error is None


def test_claiming_a_taken_name_conflicts_inside_the_transaction(
    registry: DocumentRegistry,
) -> None:
    first, _ = registry.claim_upload(
        doc_id="d-abc",
        filename="handbook.md",
        sha256="abc123",
        size_bytes=42,
        logical_id="l-one",
        name="handbook.md",
    )
    registry.mark_processing("d-abc")
    registry.mark_ready("d-abc", chunks=1, pages=None)

    with pytest.raises(NameConflictError) as excinfo:
        registry.claim_upload(
            doc_id="d-new",
            filename="handbook.md",
            sha256="different456",
            size_bytes=99,
            logical_id="l-two",
            name="handbook.md",
        )

    assert excinfo.value.existing.logical_id == first.logical_id
    # The refused claim left nothing behind.
    assert registry.get("d-new") is None
    # A different display name is not a conflict.
    renamed, duplicate = registry.claim_upload(
        doc_id="d-new",
        filename="handbook.md",
        sha256="different456",
        size_bytes=99,
        logical_id="l-two",
        name="other-team-handbook.md",
    )
    assert duplicate is False
    assert renamed.status == "queued"


def test_lookup_by_hash_powers_deduplication(registry: DocumentRegistry) -> None:
    add(registry, sha="deadbeef")

    assert registry.find_by_sha256("deadbeef") is not None
    assert registry.find_by_sha256("other") is None


def test_concurrent_claims_have_exactly_one_winner(registry: DocumentRegistry) -> None:
    def claim(index: int) -> bool:
        _, duplicate = registry.claim_upload(
            doc_id=f"d-{index}",
            filename=f"copy-{index}.md",
            sha256="same-content",
            size_bytes=42,
            logical_id=f"l-{index}",
            name=f"copy-{index}.md",
        )
        return duplicate

    with ThreadPoolExecutor(max_workers=2) as pool:
        duplicates = list(pool.map(claim, range(2)))

    assert sorted(duplicates) == [False, True]
    assert registry.count() == 1


def test_delete_claim_blocks_a_retry(registry: DocumentRegistry) -> None:
    add(registry)
    registry.mark_error("d-abc", "failed")

    record, busy = registry.begin_delete("d-abc")
    assert record is not None
    assert busy is False

    existing, duplicate = registry.claim_upload(
        doc_id="d-new",
        filename="handbook.md",
        sha256="abc123",
        size_bytes=42,
        logical_id="l-new",
        name="handbook.md",
    )
    assert duplicate is True
    assert existing.status == "deleting"


def test_delete_reports_whether_anything_was_removed(registry: DocumentRegistry) -> None:
    add(registry)

    assert registry.delete("d-abc") is True
    assert registry.delete("d-abc") is False
    assert registry.count() == 0


def test_survives_reopening(tmp_path: Path) -> None:
    first = DocumentRegistry(tmp_path / "reed.db")
    add(first)
    first.close()

    second = DocumentRegistry(tmp_path / "reed.db")
    try:
        assert second.count() == 1
    finally:
        second.close()


def test_index_generation_activation_and_rollback_are_atomic(registry: DocumentRegistry) -> None:
    first = registry.create_generation(
        logical_collection="chunks",
        physical_collection="chunks__g_1",
        fingerprint={"digest": "one"},
        status="active",
    )
    second = registry.create_generation(
        logical_collection="chunks",
        physical_collection="chunks__g_2",
        fingerprint={"digest": "two"},
    )
    registry.set_generation_counts(second.id, document_count=2, chunk_count=7)

    activated = registry.activate_generation(second.id)

    assert activated.status == "active"
    assert (activated.document_count, activated.chunk_count) == (2, 7)
    assert registry.active_generation("chunks").id == second.id  # type: ignore[union-attr]
    assert registry.previous_generation("chunks").id == first.id  # type: ignore[union-attr]
    assert [item.status for item in registry.list_generations("chunks")].count("active") == 1


def test_failed_generation_cannot_replace_the_active_one(registry: DocumentRegistry) -> None:
    active = registry.create_generation(
        logical_collection="chunks",
        physical_collection="chunks__g_active",
        fingerprint={},
        status="active",
    )
    candidate = registry.create_generation(
        logical_collection="chunks",
        physical_collection="chunks__g_candidate",
        fingerprint={},
    )

    assert registry.fail_generation(candidate.id, "provider failed") is True
    assert registry.active_generation("chunks") == active
    with pytest.raises(ValueError, match="cannot be activated"):
        registry.activate_generation(candidate.id)


def test_a_building_generation_cannot_be_forgotten(registry: DocumentRegistry) -> None:
    # Its collection may be receiving points from a reindex right now.
    candidate = registry.create_generation(
        logical_collection="chunks",
        physical_collection="chunks__g_candidate",
        fingerprint={},
    )

    assert registry.delete_generation(candidate.id) is False
    assert registry.fail_generation(candidate.id, "aborted") is True
    assert registry.delete_generation(candidate.id) is True


def test_registry_rejects_a_future_schema_before_creating_tables(tmp_path: Path) -> None:
    path = tmp_path / "future.db"
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA user_version=999")
    connection.close()

    with pytest.raises(RuntimeError, match="newer"):
        DocumentRegistry(path)

    check = sqlite3.connect(path)
    try:
        tables = check.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='documents'"
        ).fetchall()
    finally:
        check.close()
    assert tables == []


def test_upgrading_gives_every_existing_document_its_own_lineage(tmp_path: Path) -> None:
    # A registry written before lineages existed: no logical_id, no name, no
    # version, and the schema version to match.
    path = tmp_path / "old.db"
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE documents (
            id TEXT PRIMARY KEY, filename TEXT NOT NULL, sha256 TEXT NOT NULL,
            status TEXT NOT NULL, chunks INTEGER NOT NULL DEFAULT 0, pages INTEGER,
            size_bytes INTEGER NOT NULL DEFAULT 0, stored_path TEXT, error TEXT,
            created_at TEXT NOT NULL
        );
        INSERT INTO documents VALUES
            ('d-one', 'handbook.md', 'aaa', 'ready', 3, NULL, 10, NULL, NULL, '2026-01-01'),
            ('d-two', 'policy.md', 'bbb', 'ready', 2, NULL, 20, NULL, NULL, '2026-01-02');
        PRAGMA user_version=2;
        """
    )
    connection.commit()
    connection.close()

    registry = DocumentRegistry(path)
    try:
        one = registry.get("d-one")
        two = registry.get("d-two")
        assert one is not None and two is not None
        assert (one.name, one.version) == ("handbook.md", 1)
        assert (two.name, two.version) == ("policy.md", 1)
        # Separate documents, so separate lineages.
        assert one.logical_id != two.logical_id
        assert one.logical_id.startswith("l-")
        assert registry.lineage(one.logical_id) == [one]
        assert registry.find_by_name("policy.md") == two
    finally:
        registry.close()


def test_each_upload_starts_its_own_lineage(registry: DocumentRegistry) -> None:
    first = registry.add(doc_id="d-1", filename="a.md", sha256="aaa", size_bytes=1)
    second = registry.add(doc_id="d-2", filename="b.md", sha256="bbb", size_bytes=1)

    assert first.logical_id != second.logical_id
    assert (first.version, second.version) == (1, 1)
    assert registry.find_by_name("a.md") == first
    assert registry.find_by_name("missing.md") is None
