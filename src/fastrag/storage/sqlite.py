"""SQLite document store: the system of record for documents, chunks and embeddings.

SQLite keeps the reference implementation dependency-free; the schema maps directly to
Postgres. Every write is a single transaction, so a crash can never leave a document
half-replaced.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from fastrag.domain import Chunk, DocumentRecord
from fastrag.errors import ConfigurationError

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS documents (
    tenant_id    TEXT NOT NULL,
    doc_id       TEXT NOT NULL,
    source_id    TEXT NOT NULL,
    title        TEXT NOT NULL,
    version      INTEGER NOT NULL,
    content_hash TEXT NOT NULL,
    acl          TEXT NOT NULL,   -- JSON array of ACL tokens
    metadata     TEXT NOT NULL,   -- JSON object
    chunk_count  INTEGER NOT NULL,
    updated_at   TEXT NOT NULL,
    PRIMARY KEY (tenant_id, doc_id)
);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id  TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    doc_id    TEXT NOT NULL,
    version   INTEGER NOT NULL,
    ordinal   INTEGER NOT NULL,
    text      TEXT NOT NULL,
    embedding BLOB NOT NULL,      -- float32, little-endian
    FOREIGN KEY (tenant_id, doc_id) REFERENCES documents (tenant_id, doc_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS chunks_by_doc ON chunks (tenant_id, doc_id);
CREATE TABLE IF NOT EXISTS tenant_generation (
    tenant_id  TEXT PRIMARY KEY,
    generation INTEGER NOT NULL
);
"""


class SQLiteDocumentStore:
    def __init__(self, path: str | Path = ":memory:"):
        path = str(path)
        self._on_disk = path != ":memory:"
        if self._on_disk:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).touch(mode=0o600, exist_ok=True)  # owner-only before any data lands
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        if path != ":memory:":
            self._conn.execute("PRAGMA journal_mode = WAL")
            # Deleted content is overwritten with zeros rather than left in free pages.
            self._conn.execute("PRAGMA secure_delete = ON")
        self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # --- metadata ----------------------------------------------------------

    def check_embedder(self, name: str) -> None:
        with self._lock:
            row = self._conn.execute("SELECT value FROM meta WHERE key = 'embedder'").fetchone()
            if row is None:
                self._conn.execute("INSERT INTO meta (key, value) VALUES ('embedder', ?)", (name,))
            elif row["value"] != name:
                raise ConfigurationError(
                    f"store was built with embedder {row['value']!r} but {name!r} is configured; "
                    "re-ingest into a new store to change embedders"
                )

    def ping(self) -> None:
        with self._lock:
            self._conn.execute("SELECT 1").fetchone()

    # --- reads -------------------------------------------------------------

    def get_document(self, tenant_id: str, doc_id: str) -> DocumentRecord | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM documents WHERE tenant_id = ? AND doc_id = ?", (tenant_id, doc_id)
            ).fetchone()
        return _record(row) if row else None

    def list_documents(self, tenant_id: str) -> list[DocumentRecord]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM documents WHERE tenant_id = ? ORDER BY source_id", (tenant_id,)
            ).fetchall()
        return [_record(r) for r in rows]

    def iter_chunks(self) -> Iterator[tuple[Chunk, np.ndarray]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT c.chunk_id, c.tenant_id, c.doc_id, c.version, c.ordinal, c.text,
                       c.embedding, d.acl, d.source_id, d.title
                FROM chunks c JOIN documents d USING (tenant_id, doc_id)
                ORDER BY c.tenant_id, c.doc_id, c.ordinal
                """
            ).fetchall()
        for r in rows:
            chunk = Chunk(
                id=r["chunk_id"],
                tenant_id=r["tenant_id"],
                doc_id=r["doc_id"],
                version=r["version"],
                ordinal=r["ordinal"],
                text=r["text"],
                acl=frozenset(json.loads(r["acl"])),
                source_id=r["source_id"],
                title=r["title"],
            )
            yield chunk, np.frombuffer(r["embedding"], dtype="<f4")

    def generation(self, tenant_id: str) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT generation FROM tenant_generation WHERE tenant_id = ?", (tenant_id,)
            ).fetchone()
        return int(row["generation"]) if row else 0

    # --- writes ------------------------------------------------------------

    def replace_document(
        self, record: DocumentRecord, chunks: list[Chunk], embeddings: np.ndarray
    ) -> list[str]:
        if embeddings.shape[0] != len(chunks):
            raise ValueError("one embedding per chunk is required")
        with self._lock, self._transaction():
            old_ids = self._chunk_ids(record.tenant_id, record.doc_id)
            self._conn.execute(
                "DELETE FROM chunks WHERE tenant_id = ? AND doc_id = ?",
                (record.tenant_id, record.doc_id),
            )
            self._conn.execute(
                """
                INSERT INTO documents (tenant_id, doc_id, source_id, title, version, content_hash,
                                       acl, metadata, chunk_count, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (tenant_id, doc_id) DO UPDATE SET
                    source_id = excluded.source_id, title = excluded.title,
                    version = excluded.version, content_hash = excluded.content_hash,
                    acl = excluded.acl, metadata = excluded.metadata,
                    chunk_count = excluded.chunk_count, updated_at = excluded.updated_at
                """,
                (
                    record.tenant_id,
                    record.doc_id,
                    record.source_id,
                    record.title,
                    record.version,
                    record.content_hash,
                    json.dumps(sorted(record.acl)),
                    json.dumps(record.metadata, sort_keys=True),
                    len(chunks),
                    record.updated_at or _now(),
                ),
            )
            self._conn.executemany(
                """
                INSERT INTO chunks (chunk_id, tenant_id, doc_id, version, ordinal, text, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        c.id,
                        c.tenant_id,
                        c.doc_id,
                        c.version,
                        c.ordinal,
                        c.text,
                        np.asarray(e, dtype="<f4").tobytes(),
                    )
                    for c, e in zip(chunks, embeddings, strict=True)
                ],
            )
        return old_ids

    def delete_document(self, tenant_id: str, doc_id: str) -> list[str] | None:
        with self._lock, self._transaction():
            exists = self._conn.execute(
                "SELECT 1 FROM documents WHERE tenant_id = ? AND doc_id = ?", (tenant_id, doc_id)
            ).fetchone()
            if not exists:
                return None
            ids = self._chunk_ids(tenant_id, doc_id)
            self._conn.execute(
                "DELETE FROM documents WHERE tenant_id = ? AND doc_id = ?", (tenant_id, doc_id)
            )
        if self._on_disk:
            # secure_delete zeroes freed pages, but deleted rows can linger in the WAL
            # until a checkpoint; truncate it so erased content is gone from disk.
            with self._lock:
                self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        return ids

    # --- helpers -----------------------------------------------------------

    def _chunk_ids(self, tenant_id: str, doc_id: str) -> list[str]:
        rows = self._conn.execute(
            "SELECT chunk_id FROM chunks WHERE tenant_id = ? AND doc_id = ?", (tenant_id, doc_id)
        ).fetchall()
        return [r["chunk_id"] for r in rows]

    def bump_generation(self, tenant_id: str) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO tenant_generation (tenant_id, generation) VALUES (?, 1)
                ON CONFLICT (tenant_id) DO UPDATE SET generation = generation + 1
                """,
                (tenant_id,),
            )

    def _transaction(self) -> _Transaction:
        return _Transaction(self._conn)


class _Transaction:
    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    def __enter__(self) -> None:
        self._conn.execute("BEGIN IMMEDIATE")

    def __exit__(self, exc_type, exc, tb) -> None:
        self._conn.execute("ROLLBACK" if exc_type else "COMMIT")


def _record(row: sqlite3.Row) -> DocumentRecord:
    return DocumentRecord(
        tenant_id=row["tenant_id"],
        doc_id=row["doc_id"],
        source_id=row["source_id"],
        title=row["title"],
        version=row["version"],
        content_hash=row["content_hash"],
        acl=frozenset(json.loads(row["acl"])),
        metadata=json.loads(row["metadata"]),
        chunk_count=row["chunk_count"],
        updated_at=row["updated_at"],
    )


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
