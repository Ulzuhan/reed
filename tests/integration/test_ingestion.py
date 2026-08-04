"""End-to-end ingestion against embedded Qdrant with real BM25 sparse vectors.

Only the dense embeddings are faked here — the named-vector collection, the
hybrid upsert path and the deletion filter are all the real thing.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from reed.config import Settings
from reed.ingest.pipeline import document_id_for, point_id_for, sha256_file
from reed.services import Services, build_services
from tests.pdf_fixture import write_pdf

pytestmark = pytest.mark.integration

HANDBOOK = """# Expenses

Expenses above 75 euros require pre-approval from your manager.
Travel bookings go through the finance portal.
"""


@pytest.fixture
def services(settings: Settings) -> Iterator[Services]:
    built = build_services(settings)
    yield built
    built.close()


def write_handbook(tmp_path: Path, name: str = "expenses.md", body: str = HANDBOOK) -> Path:
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


def count_points(services: Services) -> int:
    return services.qdrant.count(services.settings.collection, exact=True).count


def test_ingesting_a_markdown_file_stores_retrievable_chunks(
    services: Services, tmp_path: Path
) -> None:
    from reed.ingest.pipeline import ingest_path

    result = ingest_path(services, write_handbook(tmp_path))

    assert result.status == "ready"
    assert result.chunks >= 1
    assert count_points(services) == result.chunks

    record = services.registry.get(result.document_id)
    assert record is not None
    assert record.status == "ready"
    assert record.filename == "expenses.md"


def test_reingesting_identical_content_does_not_duplicate_points(
    services: Services, tmp_path: Path
) -> None:
    from reed.ingest.pipeline import ingest_path, process_document

    path = write_handbook(tmp_path)
    first = ingest_path(services, path)
    before = count_points(services)

    # Force the work to run again rather than short-circuiting on the registry:
    # deterministic point ids must overwrite, not append.
    process_document(services, first.document_id)

    assert count_points(services) == before


def test_the_same_file_under_a_new_name_is_recognised_as_a_duplicate(
    services: Services, tmp_path: Path
) -> None:
    from reed.ingest.pipeline import ingest_path

    ingest_path(services, write_handbook(tmp_path, "expenses.md"))
    again = ingest_path(services, write_handbook(tmp_path, "expenses-copy.md"))

    assert again.duplicate is True
    assert services.registry.count() == 1


def test_point_ids_are_derived_from_content(tmp_path: Path) -> None:
    path = write_handbook(tmp_path)
    digest = sha256_file(path)

    assert document_id_for(digest).startswith("d-")
    assert point_id_for(digest, 0) == point_id_for(digest, 0)
    assert point_id_for(digest, 0) != point_id_for(digest, 1)


def test_deleting_a_document_removes_only_its_chunks(services: Services, tmp_path: Path) -> None:
    from reed.ingest.pipeline import delete_document, ingest_path

    keep = ingest_path(services, write_handbook(tmp_path, "keep.md", "# Keep\n\nStays put."))
    drop = ingest_path(services, write_handbook(tmp_path, "drop.md", HANDBOOK))
    assert count_points(services) == keep.chunks + drop.chunks

    assert delete_document(services, drop.document_id) is True

    assert count_points(services) == keep.chunks
    assert services.registry.get(drop.document_id) is None
    assert services.registry.get(keep.document_id) is not None


def test_pdf_chunks_remember_their_page(services: Services, tmp_path: Path) -> None:
    from reed.ingest.pipeline import ingest_path

    pdf = write_pdf(
        tmp_path / "handbook.pdf",
        [
            "Expenses above 75 euros require pre-approval from your manager.",
            "P1 incidents page the on-call engineer within 15 minutes.",
        ],
    )

    result = ingest_path(services, pdf)

    assert result.status == "ready"
    records, _ = services.qdrant.scroll(services.settings.collection, limit=100, with_payload=True)
    pages = {r.payload["metadata"]["page"] for r in records if r.payload}
    assert pages == {1, 2}


def test_upload_endpoint_reports_progress_until_ready(client: TestClient, tmp_path: Path) -> None:
    path = write_handbook(tmp_path)

    with path.open("rb") as handle:
        response = client.post(
            "/v1/documents", files={"file": ("expenses.md", handle, "text/markdown")}
        )

    assert response.status_code == 202
    document_id = response.json()["document_id"]

    # TestClient runs background tasks before returning, so it is already done.
    detail = client.get(f"/v1/documents/{document_id}")
    assert detail.status_code == 200
    assert detail.json()["status"] == "ready"
    assert detail.json()["chunks"] >= 1

    listing = client.get("/v1/documents").json()
    assert [d["id"] for d in listing["documents"]] == [document_id]


def test_uploading_the_same_file_twice_conflicts(client: TestClient, tmp_path: Path) -> None:
    path = write_handbook(tmp_path)

    for _ in range(2):
        with path.open("rb") as handle:
            last = client.post(
                "/v1/documents", files={"file": ("expenses.md", handle, "text/markdown")}
            )

    assert last.status_code == 409
    assert last.json()["detail"]["document_id"].startswith("d-")


def test_unsupported_types_are_rejected_before_any_work(client: TestClient) -> None:
    response = client.post(
        "/v1/documents",
        files={"file": ("sheet.xlsx", b"binary", "application/vnd.ms-excel")},
    )

    assert response.status_code == 415


def test_deleting_through_the_api(client: TestClient, tmp_path: Path) -> None:
    path = write_handbook(tmp_path)
    with path.open("rb") as handle:
        document_id = client.post(
            "/v1/documents", files={"file": ("expenses.md", handle, "text/markdown")}
        ).json()["document_id"]

    assert client.delete(f"/v1/documents/{document_id}").status_code == 204
    assert client.get(f"/v1/documents/{document_id}").status_code == 404
    assert client.delete(f"/v1/documents/{document_id}").status_code == 404


def test_health_counts_documents(client: TestClient, tmp_path: Path) -> None:
    path = write_handbook(tmp_path)
    with path.open("rb") as handle:
        client.post("/v1/documents", files={"file": ("expenses.md", handle, "text/markdown")})

    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["vector_store"] == "ok"
    assert body["documents"] == 1
