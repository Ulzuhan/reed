from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from reed.ingest.registry import DocumentRegistry


@pytest.fixture
def registry(tmp_path: Path) -> Iterator[DocumentRegistry]:
    instance = DocumentRegistry(tmp_path / "reed.db")
    try:
        yield instance
    finally:
        instance.close()


def add(registry: DocumentRegistry, doc_id: str = "d-abc", sha: str = "abc123") -> None:
    registry.add(doc_id=doc_id, filename="handbook.md", sha256=sha, size_bytes=42)


def test_new_documents_start_pending(registry: DocumentRegistry) -> None:
    record = registry.add(doc_id="d-abc", filename="a.md", sha256="abc", size_bytes=10)

    assert record.status == "pending"
    assert record.chunks == 0
    assert record.created_at


def test_status_transitions(registry: DocumentRegistry) -> None:
    add(registry)

    registry.mark_processing("d-abc")
    assert registry.get("d-abc").status == "processing"  # type: ignore[union-attr]

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
    )
    assert duplicate is False
    assert retried.id == "d-abc"
    assert retried.status == "pending"
    assert retried.error is None


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
