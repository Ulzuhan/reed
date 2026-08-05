"""End-to-end ingestion against embedded Qdrant with real BM25 sparse vectors.

Only the dense embeddings are faked here — the named-vector collection, the
hybrid upsert path and the deletion filter are all the real thing.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient

from reed.api.app import create_app
from reed.config import Settings
from reed.ingest.pipeline import (
    document_id_for,
    point_id_for,
    process_document,
    register_replacement,
    register_upload,
    sha256_file,
)
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


def force_status(services: Services, doc_id: str, status: str, error: str) -> None:
    """Fabricate a stored state the guarded API can no longer produce.

    `mark_error` refuses to demote a ready row, so tests simulating a crashed
    process or a pre-guard database write the row directly.
    """
    registry = services.registry
    with registry._lock:
        registry._conn.execute(
            "UPDATE documents SET status=?, error=? WHERE id=?", (status, error, doc_id)
        )
        registry._conn.commit()


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
    points, _ = services.qdrant.scroll(
        services.settings.collection,
        limit=100,
        with_payload=True,
    )
    assert all(point.payload["metadata"]["committed"] is True for point in points if point.payload)


def test_a_failed_partial_upsert_leaves_no_queryable_points(
    services: Services, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from reed.ingest.pipeline import process_document, register_upload

    path = write_handbook(tmp_path, body=HANDBOOK * 20)
    record, _ = register_upload(services, source=path, filename=path.name)
    _ = services.vectorstore
    client_type = type(services.qdrant)
    original = client_type.upsert

    def fail_after_one(self: object, **kwargs: object) -> object:
        points = kwargs["points"]
        collection_name = kwargs["collection_name"]
        assert isinstance(points, list)
        assert isinstance(collection_name, str)
        original(
            self,  # type: ignore[arg-type]
            collection_name=collection_name,
            points=points[:1],
            wait=True,
        )
        raise RuntimeError("secret provider failure")

    monkeypatch.setattr(client_type, "upsert", fail_after_one)
    result = process_document(services, record.id)

    assert result.status == "error"
    assert result.error == "Ingestion failed; inspect the server logs for details"
    assert count_points(services) == 0


def test_embedding_metadata_does_not_pollute_stored_display_text(
    services: Services, tmp_path: Path
) -> None:
    from reed.ingest.pipeline import ingest_path

    path = tmp_path / "named-policy.md"
    path.write_text("# Travel\n\nThe clean policy text.", encoding="utf-8")
    assert ingest_path(services, path).status == "ready"

    points, _ = services.qdrant.scroll(
        services.active_collection_name,
        limit=10,
        with_payload=True,
    )
    page_content = str(points[0].payload["page_content"])  # type: ignore[index]
    assert "The clean policy text." in page_content
    assert "Title: named-policy.md" not in page_content
    assert points[0].payload["metadata"]["section"] == "Travel"  # type: ignore[index]


def test_dense_embedding_inference_happens_outside_qdrant_lock(
    services: Services, tmp_path: Path
) -> None:
    from langchain_core.embeddings import Embeddings

    from reed.ingest.pipeline import ingest_path

    original = services.embeddings
    _ = services.vectorstore

    class LockCheckingEmbeddings(Embeddings):
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            assert not cast(Any, services.vector_access)._is_owned()
            assert "Title: named-policy.md" in texts[0]
            assert "Section: Travel" in texts[0]
            return original.embed_documents(texts)

        def embed_query(self, text: str) -> list[float]:
            return original.embed_query(text)

    services._embeddings = LockCheckingEmbeddings()
    path = tmp_path / "named-policy.md"
    path.write_text("# Travel\n\nThe clean policy text.", encoding="utf-8")

    assert ingest_path(services, path).status == "ready"


def test_retrieval_requires_vector_and_registry_commits(services: Services, tmp_path: Path) -> None:
    from reed.ingest.pipeline import ingest_path
    from reed.rag.retriever import retrieve

    result = ingest_path(services, write_handbook(tmp_path))
    assert retrieve(services, "expense threshold")

    # Simulate the cross-store publication window or an orphan left by a
    # failed registry commit: committed vectors alone must never be visible.
    force_status(services, result.document_id, "error", "registry commit failed")

    assert retrieve(services, "expense threshold") == []


def test_reingesting_identical_content_does_not_duplicate_points(
    services: Services, tmp_path: Path
) -> None:
    from reed.ingest.pipeline import ingest_path, process_document, register_upload

    path = write_handbook(tmp_path)
    first = ingest_path(services, path)
    before = count_points(services)

    # A failed row is the supported retry path. Deterministic point ids must
    # overwrite rather than append when that retry runs.
    force_status(services, first.document_id, "error", "retry me")
    record, duplicate = register_upload(services, source=path, filename=path.name)
    assert duplicate is False
    process_document(services, record.id)

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

    deadline = time.monotonic() + 3
    detail = client.get(f"/v1/documents/{document_id}")
    while detail.json()["status"] not in {"ready", "error"}:
        assert time.monotonic() < deadline
        time.sleep(0.01)
        detail = client.get(f"/v1/documents/{document_id}")
    assert detail.status_code == 200
    assert detail.json()["status"] == "ready"
    assert detail.json()["chunks"] >= 1

    listing = client.get("/v1/documents").json()
    assert [d["id"] for d in listing["documents"]] == [document_id]

    # The identity a replacement will address, distinct from the content id.
    accepted = response.json()
    listed = listing["documents"][0]
    assert accepted["logical_id"].startswith("l-")
    assert accepted["version"] == 1
    assert listed["logical_id"] == accepted["logical_id"]
    assert (listed["name"], listed["version"]) == ("expenses.md", 1)


def test_document_listing_is_paginated(client: TestClient, tmp_path: Path) -> None:
    for index in range(3):
        path = write_handbook(tmp_path, f"{index}.md", f"# Document {index}\n\nUnique {index}.")
        with path.open("rb") as handle:
            client.post("/v1/documents", files={"file": (path.name, handle, "text/markdown")})

    first = client.get("/v1/documents?limit=2&offset=0").json()
    second = client.get("/v1/documents?limit=2&offset=2").json()

    assert first["total"] == 3
    assert len(first["documents"]) == 2
    assert len(second["documents"]) == 1


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


def test_a_zero_byte_upload_is_rejected_synchronously(client: TestClient) -> None:
    response = client.post(
        "/v1/documents",
        files={"file": ("empty.md", b"", "text/markdown")},
    )

    assert response.status_code == 422
    assert client.get("/v1/documents").json()["total"] == 0


def test_multipart_body_is_rejected_before_fastapi_spools_it(settings: Settings) -> None:
    limited = settings.model_copy(update={"max_upload_mb": 1})
    with TestClient(create_app(limited)) as client:
        response = client.post(
            "/v1/documents",
            files={"file": ("huge.md", b"x" * (2 * 1024 * 1024), "text/markdown")},
        )

        assert response.status_code == 413
        assert client.get("/v1/documents").json()["total"] == 0


def test_deleting_through_the_api(client: TestClient, tmp_path: Path) -> None:
    path = write_handbook(tmp_path)
    with path.open("rb") as handle:
        document_id = client.post(
            "/v1/documents", files={"file": ("expenses.md", handle, "text/markdown")}
        ).json()["document_id"]

    deadline = time.monotonic() + 3
    while client.get(f"/v1/documents/{document_id}").json()["status"] != "ready":
        assert time.monotonic() < deadline
        time.sleep(0.01)

    assert client.delete(f"/v1/documents/{document_id}").status_code == 204
    assert client.get(f"/v1/documents/{document_id}").status_code == 404
    assert client.delete(f"/v1/documents/{document_id}").status_code == 404


def test_deletion_failure_does_not_log_the_user_controlled_id(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def fail_deletion(_services: Services, _document_id: str) -> bool:
        raise RuntimeError("backend failure")

    monkeypatch.setattr("reed.api.routes.documents.delete_document", fail_deletion)

    response = client.delete("/v1/documents/forged%0Aadmin-entry")

    assert response.status_code == 502
    assert "forged" not in caplog.text
    assert "admin-entry" not in caplog.text


def test_a_failed_document_can_still_be_deleted(client: TestClient) -> None:
    # Nothing has been ingested, so the collection does not exist yet; deleting
    # used to blow up on the missing collection and strand the row forever.
    response = client.post("/v1/documents", files={"file": ("empty.md", b"   \n", "text/markdown")})
    document_id = response.json()["document_id"]
    deadline = time.monotonic() + 3
    while client.get(f"/v1/documents/{document_id}").json()["status"] != "error":
        assert time.monotonic() < deadline
        time.sleep(0.01)

    assert client.delete(f"/v1/documents/{document_id}").status_code == 204
    assert client.get(f"/v1/documents/{document_id}").status_code == 404


def test_a_document_being_ingested_cannot_be_deleted(services: Services, tmp_path: Path) -> None:
    from reed.ingest.pipeline import DocumentBusyError, delete_document, register_upload

    record, _ = register_upload(services, source=write_handbook(tmp_path), filename="expenses.md")
    services.registry.mark_processing(record.id)

    with pytest.raises(DocumentBusyError, match="busy"):
        delete_document(services, record.id)


def test_reuploading_a_document_mid_ingestion_is_a_duplicate(
    services: Services, tmp_path: Path
) -> None:
    from reed.ingest.pipeline import register_upload

    path = write_handbook(tmp_path)
    record, first = register_upload(services, source=path, filename="expenses.md")
    assert first is False
    services.registry.mark_processing(record.id)

    # A double-clicked upload must not start a second run over the same file.
    again, duplicate = register_upload(services, source=path, filename="expenses.md")

    assert duplicate is True
    assert again.status == "parsing"


def test_health_counts_documents(client: TestClient, tmp_path: Path) -> None:
    path = write_handbook(tmp_path)
    with path.open("rb") as handle:
        client.post("/v1/documents", files={"file": ("expenses.md", handle, "text/markdown")})

    body = client.get("/ready").json()
    assert body["status"] == "ok"
    assert body["vector_store"] == "ok"
    assert body["documents"] == 1


def _upload(client: TestClient, tmp_path: Path, name: str, body: str) -> dict[str, object]:
    path = write_handbook(tmp_path, name, body)
    with path.open("rb") as handle:
        response = client.post("/v1/documents", files={"file": (name, handle, "text/markdown")})
    assert response.status_code == 202, response.text
    return dict(response.json())


def _await_ready(client: TestClient, document_id: str) -> None:
    deadline = time.monotonic() + 5
    while client.get(f"/v1/documents/{document_id}").json()["status"] not in {"ready", "error"}:
        assert time.monotonic() < deadline
        time.sleep(0.01)
    assert client.get(f"/v1/documents/{document_id}").json()["status"] == "ready"


def test_a_replacement_supersedes_the_previous_version(client: TestClient, tmp_path: Path) -> None:
    first = _upload(client, tmp_path, "handbook.md", "# Handbook\n\nThe cap is 75 euros.")
    _await_ready(client, str(first["document_id"]))
    logical_id = str(first["logical_id"])

    revised = write_handbook(tmp_path, "handbook-v2.md", "# Handbook\n\nThe cap is 120 euros.")
    with revised.open("rb") as handle:
        response = client.put(
            f"/v1/documents/{logical_id}", files={"file": ("handbook.md", handle, "text/markdown")}
        )

    assert response.status_code == 202
    assert response.json()["version"] == 2
    assert response.json()["logical_id"] == logical_id
    _await_ready(client, response.json()["document_id"])

    # The list shows one document, at its new version.
    listing = client.get("/v1/documents").json()
    assert listing["total"] == 1
    current = listing["documents"][0]
    assert (current["version"], current["name"]) == (2, "handbook.md")

    # The history is still there, and the retired version is no longer serving.
    versions = client.get(f"/v1/documents/{logical_id}/versions").json()
    assert [(v["version"], v["status"]) for v in versions["documents"]] == [
        (2, "ready"),
        (1, "superseded"),
    ]

    answer = client.post("/v1/ask", json={"question": "What is the cap?", "stream": False}).json()
    assert "120" in " ".join(source["excerpt"] for source in answer["sources"])
    assert "75 euros" not in " ".join(source["excerpt"] for source in answer["sources"])


def test_replacement_body_is_rejected_before_fastapi_spools_it(settings: Settings) -> None:
    # The PUT route resolves UploadFile exactly like the POST route does, so
    # it shares the same ingress body cap.
    limited = settings.model_copy(update={"max_upload_mb": 1})
    with TestClient(create_app(limited)) as client:
        response = client.put(
            "/v1/documents/l-0123456789abcdef",
            files={"file": ("huge.md", b"x" * (2 * 1024 * 1024), "text/markdown")},
        )

        assert response.status_code == 413


def test_a_replacement_survives_a_failing_cleanup_flush(
    services: Services, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Once the new version has committed, a store blip while retiring the old
    # one must not demote it back to error — that would schedule deletion of
    # points that are already serving.
    first_path = write_handbook(tmp_path, "handbook.md", "# Handbook\n\nThe cap is 75 euros.")
    record, _ = register_upload(services, source=first_path, filename="handbook.md")
    assert process_document(services, record.id).status == "ready"

    revised = write_handbook(tmp_path, "handbook-v2.md", "# Handbook\n\nThe cap is 120 euros.")
    replacement = register_replacement(
        services, logical_id=record.logical_id, source=revised, filename="handbook.md"
    )

    def failing_flush() -> None:
        raise RuntimeError("the vector store blipped")

    monkeypatch.setattr(services, "flush_pending_vector_cleanup", failing_flush)
    result = process_document(services, replacement.id)

    assert result.status == "ready"
    current = services.registry.get(replacement.id)
    assert current is not None and current.status == "ready"
    superseded = services.registry.get(record.id)
    assert superseded is not None and superseded.status == "superseded"

    # The old version's cleanup stayed queued; a later flush completes it
    # without touching the serving version's points.
    monkeypatch.undo()
    services.flush_pending_vector_cleanup()
    assert count_points(services) == result.chunks


def test_a_second_upload_of_the_same_name_names_the_lineage_to_replace(
    client: TestClient, tmp_path: Path
) -> None:
    first = _upload(client, tmp_path, "handbook.md", "# Handbook\n\nOriginal.")
    _await_ready(client, str(first["document_id"]))

    other = write_handbook(tmp_path, "other.md", "# Other\n\nDifferent content entirely.")
    with other.open("rb") as handle:
        clash = client.post(
            "/v1/documents", files={"file": ("handbook.md", handle, "text/markdown")}
        )

    assert clash.status_code == 409
    assert clash.json()["detail"]["logical_id"] == first["logical_id"]

    # An unrelated document that happens to share a filename says so.
    with other.open("rb") as handle:
        renamed = client.post(
            "/v1/documents",
            files={"file": ("handbook.md", handle, "text/markdown")},
            data={"name": "other-team-handbook.md"},
        )
    assert renamed.status_code == 202
    assert renamed.json()["logical_id"] != first["logical_id"]


def test_deleting_a_lineage_removes_every_version(client: TestClient, tmp_path: Path) -> None:
    first = _upload(client, tmp_path, "handbook.md", "# Handbook\n\nVersion one.")
    _await_ready(client, str(first["document_id"]))
    logical_id = str(first["logical_id"])
    revised = write_handbook(tmp_path, "handbook-v2.md", "# Handbook\n\nVersion two.")
    with revised.open("rb") as handle:
        second = client.put(
            f"/v1/documents/{logical_id}", files={"file": ("handbook.md", handle, "text/markdown")}
        )
    _await_ready(client, second.json()["document_id"])

    # The serving version cannot be dropped on its own.
    assert client.delete(f"/v1/documents/{logical_id}/versions/2").status_code == 409
    assert client.delete(f"/v1/documents/{logical_id}/versions/1").status_code == 204

    assert client.delete(f"/v1/documents/{logical_id}").status_code == 204
    assert client.get(f"/v1/documents/{logical_id}/versions").status_code == 404
    assert client.get("/v1/documents").json()["total"] == 0
