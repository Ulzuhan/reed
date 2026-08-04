"""Document registry: what has been ingested, and how it went.

Plain ``sqlite3`` in WAL mode. The vector store holds the chunks; this holds
the per-document bookkeeping the API needs (status, errors, counts) and the
content hashes that make re-ingestion idempotent.
"""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

DocumentStatus = Literal["pending", "processing", "ready", "error"]

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
CREATE INDEX IF NOT EXISTS idx_documents_sha256 ON documents(sha256);
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
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

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
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO documents "
                "(id, filename, sha256, status, chunks, pages, size_bytes, stored_path, "
                " error, created_at) "
                "VALUES (?, ?, ?, 'pending', 0, NULL, ?, ?, NULL, ?)",
                (doc_id, filename, sha256, size_bytes, stored_path, _now()),
            )
            self._conn.commit()
        record = self.get(doc_id)
        assert record is not None
        return record

    def mark_processing(self, doc_id: str) -> None:
        self._update(doc_id, "UPDATE documents SET status='processing', error=NULL WHERE id=?", ())

    def mark_ready(self, doc_id: str, *, chunks: int, pages: int | None) -> None:
        self._update(
            doc_id,
            "UPDATE documents SET status='ready', chunks=?, pages=?, error=NULL WHERE id=?",
            (chunks, pages),
        )

    def mark_error(self, doc_id: str, message: str) -> None:
        self._update(doc_id, "UPDATE documents SET status='error', error=? WHERE id=?", (message,))

    def _update(self, doc_id: str, sql: str, params: tuple[object, ...]) -> None:
        with self._lock:
            self._conn.execute(sql, (*params, doc_id))
            self._conn.commit()

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

    def list(self) -> list[DocumentRecord]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM documents ORDER BY created_at DESC, filename"
            ).fetchall()
        return [_to_record(row) for row in rows]

    def count(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) AS n FROM documents").fetchone()
        return int(row["n"])

    def delete(self, doc_id: str) -> bool:
        with self._lock:
            cursor = self._conn.execute("DELETE FROM documents WHERE id=?", (doc_id,))
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
