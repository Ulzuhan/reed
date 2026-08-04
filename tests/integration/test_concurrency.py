"""Regression guards for concurrent access to the embedded vector store.

QdrantLocal has no internal locking — only a cross-process file lock. Ingestion
runs on a worker thread while requests are served, so without serialising the
vector operations, uploading a handful of files corrupts the store: documents
land in `error`, and searches fail on mismatched internal arrays.
"""

from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from reed.config import Settings
from reed.ingest.pipeline import ingest_path
from reed.rag.retriever import retrieve
from reed.services import Services, build_services

pytestmark = pytest.mark.integration

DOCUMENTS = 6


@pytest.fixture
def services(settings: Settings) -> Iterator[Services]:
    built = build_services(settings)
    yield built
    built.close()


def write_corpus(tmp_path: Path) -> list[Path]:
    paths = []
    for index in range(DOCUMENTS):
        path = tmp_path / f"policy-{index}.md"
        path.write_text(
            f"# Policy {index}\n\n## Rules\n\nRule {index} applies to everyone. "
            + ("Filler sentence for bulk. " * 30),
            encoding="utf-8",
        )
        paths.append(path)
    return paths


def test_parallel_ingestion_does_not_corrupt_the_store(services: Services, tmp_path: Path) -> None:
    paths = write_corpus(tmp_path)

    with ThreadPoolExecutor(max_workers=DOCUMENTS) as pool:
        results = list(pool.map(lambda p: ingest_path(services, p), paths))

    assert [r.status for r in results] == ["ready"] * DOCUMENTS

    stored = services.qdrant.count(services.settings.collection, exact=True).count
    assert stored == sum(r.chunks for r in results)
    assert services.registry.count() == DOCUMENTS


def test_searching_while_ingesting_keeps_working(services: Services, tmp_path: Path) -> None:
    paths = write_corpus(tmp_path)
    ingest_path(services, paths[0])

    def search(_: int) -> int:
        return len(retrieve(services, "which rule applies to everyone?"))

    with ThreadPoolExecutor(max_workers=4) as pool:
        ingesting = [pool.submit(ingest_path, services, path) for path in paths[1:]]
        searching = [pool.submit(search, n) for n in range(8)]

        assert all(future.result().status == "ready" for future in ingesting)
        # The point is that none of these raise — a corrupted store surfaces as
        # a broadcast error deep inside the local backend.
        assert all(future.result() >= 0 for future in searching)


def test_deleting_while_another_document_ingests(services: Services, tmp_path: Path) -> None:
    from reed.ingest.pipeline import delete_document

    paths = write_corpus(tmp_path)
    first = ingest_path(services, paths[0])

    with ThreadPoolExecutor(max_workers=3) as pool:
        ingesting = [pool.submit(ingest_path, services, path) for path in paths[1:3]]
        removed = pool.submit(delete_document, services, first.document_id)

        assert removed.result() is True
        assert all(future.result().status == "ready" for future in ingesting)

    remaining = services.qdrant.count(services.settings.collection, exact=True).count
    assert remaining == sum(r.result().chunks for r in ingesting)
