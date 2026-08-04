"""Document registry: what has been ingested, and how it went.

Plain ``sqlite3`` in WAL mode. The vector store holds the chunks; this holds
the per-document bookkeeping the API needs (status, errors, counts) and the
content hashes that make re-ingestion idempotent.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

DocumentStatus = Literal["pending", "processing", "ready", "error", "deleting"]

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id          TEXT PRIMARY KEY,
    filename    TEXT NOT NULL,
    sha256      TEXT NOT NULL,
    status      TEXT NOT NULL,
    chunks      INTEGER NOT NULL DEFAULT 0,
    pages       INTEGER,
    size_bytes  INTEGER NOT NULL DEFAULT 0,
    stored_path TEXT,
    error       TEXT,
    created_at  TEXT NOT NULL
);
"""


@dataclass(frozen=True, slots=True)
class DocumentRecord:
    id: str
    filename: str
    sha256: str
    status: DocumentStatus
    chunks: int
    pages: int | None
    size_bytes: int
    stored_path: str | None
    error: str | None
    created_at: str


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class DocumentRegistry:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        # Ingestion runs in a worker thread while requests read on another.
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._conn.executescript(_SCHEMA)
        self._migrate()
        self._conn.commit()

    def _migrate(self) -> None:
        """Apply small, explicit migrations to databases from older releases."""
        version = int(self._conn.execute("PRAGMA user_version").fetchone()[0])
        if version < 1:
            self._conn.execute("DROP INDEX IF EXISTS idx_documents_sha256")
            self._conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_sha256_unique ON documents(sha256)"
            )
            self._conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def add(
        self,
        *,
        doc_id: str,
        filename: str,
        sha256: str,
        size_bytes: int,
        stored_path: str | None = None,
    ) -> DocumentRecord:
        record, _ = self.claim_upload(
            doc_id=doc_id,
            filename=filename,
            sha256=sha256,
            size_bytes=size_bytes,
            stored_path=stored_path,
        )
        return record

    def claim_upload(
        self,
        *,
        doc_id: str,
        filename: str,
        sha256: str,
        size_bytes: int,
        stored_path: str | None = None,
    ) -> tuple[DocumentRecord, bool]:
        """Atomically claim one content hash for ingestion.

        Existing ready or in-flight content is a duplicate. A failed document
        can be retried, but the transition from ``error`` to ``pending`` happens
        under the same SQLite write transaction as the duplicate check.
        """
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                existing = self._conn.execute(
                    "SELECT * FROM documents WHERE sha256=?", (sha256,)
                ).fetchone()
                if existing is not None and existing["status"] != "error":
                    self._conn.commit()
                    return _to_record(existing), True

                created_at = _now()
                if existing is not None:
                    claimed_id = str(existing["id"])
                    self._conn.execute(
                        "UPDATE documents SET filename=?, status='pending', chunks=0, pages=NULL, "
                        "size_bytes=?, stored_path=?, error=NULL, created_at=? WHERE id=?",
                        (filename, size_bytes, stored_path, created_at, claimed_id),
                    )
                else:
                    claimed_id = doc_id
                    self._conn.execute(
                        "INSERT INTO documents "
                        "(id, filename, sha256, status, chunks, pages, size_bytes, stored_path, "
                        " error, created_at) "
                        "VALUES (?, ?, ?, 'pending', 0, NULL, ?, ?, NULL, ?)",
                        (claimed_id, filename, sha256, size_bytes, stored_path, created_at),
                    )
                row = self._conn.execute(
                    "SELECT * FROM documents WHERE id=?", (claimed_id,)
                ).fetchone()
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
        assert row is not None
        return _to_record(row), False

    def mark_processing(self, doc_id: str) -> bool:
        return self._update(
            doc_id,
            "UPDATE documents SET status='processing', error=NULL WHERE id=? AND status='pending'",
            (),
        )

    def mark_ready(self, doc_id: str, *, chunks: int, pages: int | None) -> bool:
        return self._update(
            doc_id,
            "UPDATE documents SET status='ready', chunks=?, pages=?, error=NULL "
            "WHERE id=? AND status='processing'",
            (chunks, pages),
        )

    def mark_error(self, doc_id: str, message: str) -> bool:
        return self._update(
            doc_id,
            "UPDATE documents SET status='error', error=? WHERE id=?",
            (message,),
        )

    def fail_interrupted(self, message: str) -> int:
        """Mark rows left mid-ingestion by a crash as failed.

        Nothing is in flight when the process starts, so a row still claiming
        `pending` or `processing` is a leftover. Without this it can never be
        deleted (the delete guard refuses) nor re-uploaded (the duplicate guard
        refuses), and the UI spins on it forever.
        """
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE documents SET status='error', error=? "
                "WHERE status IN ('pending', 'processing')",
                (message,),
            )
            self._conn.commit()
        return int(cursor.rowcount)

    def fail_interrupted_deletions(self, message: str) -> int:
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE documents SET status='error', error=? WHERE status='deleting'",
                (message,),
            )
            self._conn.commit()
        return int(cursor.rowcount)

    def _update(self, doc_id: str, sql: str, params: tuple[object, ...]) -> bool:
        with self._lock:
            cursor = self._conn.execute(sql, (*params, doc_id))
            self._conn.commit()
        return cursor.rowcount > 0

    def get(self, doc_id: str) -> DocumentRecord | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
        return _to_record(row) if row else None

    def find_by_sha256(self, sha256: str) -> DocumentRecord | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM documents WHERE sha256=? ORDER BY created_at DESC LIMIT 1",
                (sha256,),
            ).fetchone()
        return _to_record(row) if row else None

    def list(self, *, limit: int | None = None, offset: int = 0) -> list[DocumentRecord]:
        sql = "SELECT * FROM documents ORDER BY created_at DESC, filename"
        params: tuple[object, ...] = ()
        if limit is not None:
            sql += " LIMIT ? OFFSET ?"
            params = (limit, offset)
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [_to_record(row) for row in rows]

    def list_by_status(self, statuses: set[DocumentStatus]) -> Sequence[DocumentRecord]:
        if not statuses:
            return []
        placeholders = ",".join("?" for _ in statuses)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM documents WHERE status IN ({placeholders})",
                tuple(sorted(statuses)),
            ).fetchall()
        return [_to_record(row) for row in rows]

    def ready_ids(self, document_ids: Sequence[str]) -> set[str]:
        """Return the requested ids whose registry transaction is committed."""
        unique_ids = sorted(set(document_ids))
        if not unique_ids:
            return set()
        placeholders = ",".join("?" for _ in unique_ids)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT id FROM documents WHERE status='ready' AND id IN ({placeholders})",
                unique_ids,
            ).fetchall()
        return {str(row["id"]) for row in rows}

    def count(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) AS n FROM documents").fetchone()
        return int(row["n"])

    def begin_delete(self, doc_id: str) -> tuple[DocumentRecord | None, bool]:
        """Atomically change a deletable row to ``deleting``.

        Returns ``(record, busy)``. The returned record captures the state before
        the transition so the caller still knows the stored file path.
        """
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                row = self._conn.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
                if row is None:
                    self._conn.commit()
                    return None, False
                record = _to_record(row)
                if record.status in {"pending", "processing", "deleting"}:
                    self._conn.commit()
                    return record, True
                cursor = self._conn.execute(
                    "UPDATE documents SET status='deleting', error=NULL "
                    "WHERE id=? AND status IN ('ready', 'error')",
                    (doc_id,),
                )
                self._conn.commit()
                return record, cursor.rowcount == 0
            except Exception:
                self._conn.rollback()
                raise

    def delete(self, doc_id: str, *, expected_status: DocumentStatus | None = None) -> bool:
        sql = "DELETE FROM documents WHERE id=?"
        params: tuple[object, ...] = (doc_id,)
        if expected_status is not None:
            sql += " AND status=?"
            params = (doc_id, expected_status)
        with self._lock:
            cursor = self._conn.execute(sql, params)
            self._conn.commit()
        return cursor.rowcount > 0


def _to_record(row: sqlite3.Row) -> DocumentRecord:
    return DocumentRecord(
        id=row["id"],
        filename=row["filename"],
        sha256=row["sha256"],
        status=row["status"],
        chunks=row["chunks"],
        pages=row["pages"],
        size_bytes=row["size_bytes"],
        stored_path=row["stored_path"],
        error=row["error"],
        created_at=row["created_at"],
    )
