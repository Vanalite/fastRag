"""PostgreSQL + pgvector storage: the system of record *and* the search indexes.

``PostgresDocumentStore`` implements ``DocumentStore`` with the same semantics as
``SQLiteDocumentStore``. Unlike the in-memory indexes, ``PgVectorIndex`` and
``PgKeywordIndex`` query the store's own tables, so every replica sees the same data the
moment a write commits: there is nothing to rebuild at startup and nothing to drift.

Access control is a pre-filter in SQL: every search has ``tenant_id = %s AND acl &&
%s::text[]`` in its WHERE clause, so rows the caller cannot read never reach ranking or
LIMIT. See docs/adr/0007-postgres-storage.md.

Requires ``psycopg[binary]``, ``psycopg-pool`` and ``pgvector`` (the ``postgres`` extra).
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from typing import Any

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg import sql
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from fastrag.domain import Chunk, DocumentRecord
from fastrag.errors import ConcurrentModification, ConfigurationError
from fastrag.security.acl import AccessFilter
from fastrag.storage.base import UNCHECKED
from fastrag.storage.bm25 import tokenize
from fastrag.storage.indexes import SearchIndexes

log = logging.getLogger(__name__)

# One key for the whole database: migrations are rare and short, and a single key also
# serializes CREATE EXTENSION across deployments that live in different schemas.
_MIGRATION_LOCK_KEY = 0x66617374726167  # "fastrag"

# pgvector's HNSW (and IVFFlat) indexes support up to 2000 dimensions for `vector`.
_MAX_ANN_DIM = 2000
_MAX_DIM = 16000  # pgvector's limit for the `vector` type itself

# tsvector limits: lexemes are at most 2047 bytes, positions at most 16383, and only
# 256 positions are kept per lexeme.
_MAX_LEXEME_BYTES = 2046
_MAX_POSITION = 16383
_MAX_POSITIONS_PER_LEXEME = 256
_WORD = re.compile(r"\w+", re.UNICODE)

_CHUNK_COLUMNS = "chunk_id, tenant_id, doc_id, version, ordinal, text, acl, source_id, title"


# --- schema migrations --------------------------------------------------------------
# Each entry upgrades the schema by one version. Steps run in order, inside one
# transaction, under an advisory lock, so replicas starting together cannot race and a
# failed step leaves the database untouched. Never edit a released step: add a new one.


def _migration_1(dim: int) -> list[sql.Composable]:
    hnsw = (
        [
            sql.SQL(
                "CREATE INDEX chunks_embedding_hnsw ON chunks "
                "USING hnsw (embedding vector_cosine_ops)"
            )
        ]
        if dim <= _MAX_ANN_DIM
        else []
    )
    return [
        sql.SQL("CREATE EXTENSION IF NOT EXISTS vector"),
        sql.SQL(
            """
            CREATE TABLE documents (
                tenant_id    text NOT NULL,
                doc_id       text NOT NULL,
                source_id    text NOT NULL,
                title        text NOT NULL,
                version      integer NOT NULL,
                content_hash text NOT NULL,
                acl          text[] NOT NULL,                  -- who may read
                editors      text[] NOT NULL DEFAULT '{}',     -- who may write
                metadata     jsonb NOT NULL DEFAULT '{}',
                chunk_count  integer NOT NULL,
                updated_at   text NOT NULL,
                PRIMARY KEY (tenant_id, doc_id)
            )
            """
        ),
        # Dimension is a type modifier, which cannot be a bind parameter; it is an
        # integer validated by the constructor and rendered as a SQL literal.
        sql.SQL(
            """
            CREATE TABLE chunks (
                chunk_id  text PRIMARY KEY,
                tenant_id text NOT NULL,
                doc_id    text NOT NULL,
                version   integer NOT NULL,
                ordinal   integer NOT NULL,
                text      text NOT NULL,
                acl       text[] NOT NULL,     -- denormalized from the document
                title     text NOT NULL,
                source_id text NOT NULL,
                embedding vector({dim}) NOT NULL,
                tsv       tsvector NOT NULL,   -- fastrag's own terms; see PgKeywordIndex
                FOREIGN KEY (tenant_id, doc_id)
                    REFERENCES documents (tenant_id, doc_id) ON DELETE CASCADE
            )
            """
        ).format(dim=sql.Literal(dim)),
        sql.SQL("CREATE INDEX chunks_by_doc ON chunks (tenant_id, doc_id)"),
        sql.SQL("CREATE INDEX chunks_acl_gin ON chunks USING gin (acl)"),
        sql.SQL("CREATE INDEX chunks_tsv_gin ON chunks USING gin (tsv)"),
        *hnsw,
        sql.SQL(
            """
            CREATE TABLE tenant_generation (
                tenant_id  text PRIMARY KEY,
                generation bigint NOT NULL
            )
            """
        ),
    ]


_MIGRATIONS = [_migration_1]


class PostgresDocumentStore:
    SCHEMA_VERSION = len(_MIGRATIONS)

    def __init__(self, dsn: str, *, dim: int, pool_size: int = 10, migrate: bool = True):
        """Open the store. With ``migrate=False`` the schema is only *checked*: the
        application can then run with a DML-only role, and migrations run separately
        (``fastrag db migrate``) with a role that may change the schema."""
        if isinstance(dim, bool) or not isinstance(dim, int) or not 0 < dim <= _MAX_DIM:
            raise ConfigurationError(f"embedding dim must be an integer in 1..{_MAX_DIM}")
        self.dim = dim
        # Migrate on a dedicated connection first: the pool's per-connection setup
        # registers the `vector` type, which only exists once the extension is created.
        with psycopg.connect(dsn, row_factory=dict_row) as conn:
            if migrate:
                self._migrate(conn)
            else:
                self._check_schema(conn)
            row = conn.execute(
                "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
            ).fetchone()
        # hnsw.iterative_scan (keeps a filtered HNSW search going until k rows match)
        # arrived in pgvector 0.8; older versions reject the setting.
        self.supports_iterative_scan = _version_at_least(row["extversion"] if row else "0", (0, 8))
        self._pool = ConnectionPool(
            dsn,
            min_size=1,
            max_size=max(1, pool_size),
            kwargs={"row_factory": dict_row},
            configure=_configure_connection,
            check=ConnectionPool.check_connection,  # survive database restarts
            open=True,
            name="fastrag",
        )

    def close(self) -> None:
        self._pool.close()

    # --- schema migrations ----------------------------------------------------

    def _check_schema(self, conn: psycopg.Connection) -> None:
        """Verify an already-migrated schema without changing anything."""
        exists = conn.execute("SELECT to_regclass('meta') IS NOT NULL AS ok").fetchone()
        version = int(_meta(conn, "schema_version") or 0) if exists["ok"] else 0
        if version != self.SCHEMA_VERSION:
            raise ConfigurationError(
                f"database schema is v{version}, this release needs v{self.SCHEMA_VERSION}; "
                "run `fastrag db migrate` (with a role allowed to change the schema)"
            )
        stored_dim = _meta(conn, "embedding_dim")
        if stored_dim is not None and int(stored_dim) != self.dim:
            raise ConfigurationError(
                f"database stores {stored_dim}-dimensional embeddings but dim={self.dim} "
                "is configured; re-ingest into a new database to change embedders"
            )
        conn.rollback()  # read-only check; leave the connection idle

    def _migrate(self, conn: psycopg.Connection) -> None:
        with conn.transaction():
            conn.execute("SELECT pg_advisory_xact_lock(%s)", (_MIGRATION_LOCK_KEY,))
            conn.execute("CREATE TABLE IF NOT EXISTS meta (key text PRIMARY KEY, value text)")
            version = int(_meta(conn, "schema_version") or 0)
            if version > self.SCHEMA_VERSION:
                raise ConfigurationError(
                    f"database schema v{version} is newer than this release supports "
                    f"(v{self.SCHEMA_VERSION}); upgrade fastrag instead of downgrading data"
                )
            for step in range(version, self.SCHEMA_VERSION):
                for statement in _MIGRATIONS[step](self.dim):
                    conn.execute(statement)
                log.info("applied postgres schema migration", extra={"version": step + 1})
            if version != self.SCHEMA_VERSION:
                _set_meta(conn, "schema_version", str(self.SCHEMA_VERSION))
            conn.execute(
                "INSERT INTO meta (key, value) VALUES ('embedding_dim', %s) "
                "ON CONFLICT (key) DO NOTHING",
                (str(self.dim),),
            )
            stored_dim = int(_meta(conn, "embedding_dim") or self.dim)
            if stored_dim != self.dim:
                raise ConfigurationError(
                    f"database stores {stored_dim}-dimensional embeddings but dim={self.dim} "
                    "is configured; re-ingest into a new database to change embedders"
                )

    # --- metadata ----------------------------------------------------------

    def check_embedder(self, name: str) -> None:
        with self._pool.connection() as conn:
            # Insert-if-absent then read back: two replicas starting at once agree on
            # whichever name committed first.
            conn.execute(
                "INSERT INTO meta (key, value) VALUES ('embedder', %s) "
                "ON CONFLICT (key) DO NOTHING",
                (name,),
            )
            pinned = _meta(conn, "embedder")
        if pinned != name:
            raise ConfigurationError(
                f"store was built with embedder {pinned!r} but {name!r} is configured; "
                "re-ingest into a new store to change embedders"
            )

    def ping(self) -> None:
        with self._pool.connection() as conn:
            conn.execute("SELECT 1").fetchone()

    # --- reads -------------------------------------------------------------

    def get_document(self, tenant_id: str, doc_id: str) -> DocumentRecord | None:
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE tenant_id = %s AND doc_id = %s",
                (tenant_id, doc_id),
            ).fetchone()
        return _record(row) if row else None

    def list_documents(self, tenant_id: str) -> list[DocumentRecord]:
        with self._pool.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM documents WHERE tenant_id = %s ORDER BY source_id",
                (tenant_id,),
            ).fetchall()
        return [_record(r) for r in rows]

    def iter_chunks(self) -> Iterator[tuple[Chunk, np.ndarray]]:
        # A server-side cursor streams the corpus instead of loading it all at once.
        with (
            self._pool.connection() as conn,
            conn.cursor(name="fastrag_iter_chunks") as cur,
        ):
            cur.itersize = 1000
            cur.execute(
                f"SELECT {_CHUNK_COLUMNS}, embedding FROM chunks "
                "ORDER BY tenant_id, doc_id, ordinal"
            )
            for row in cur:
                yield _chunk(row), _as_array(row["embedding"])

    def generation(self, tenant_id: str) -> int:
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT generation FROM tenant_generation WHERE tenant_id = %s", (tenant_id,)
            ).fetchone()
        return int(row["generation"]) if row else 0

    def chunk_ids(self) -> set[str]:
        with self._pool.connection() as conn:
            return {r["chunk_id"] for r in conn.execute("SELECT chunk_id FROM chunks")}

    # --- writes ------------------------------------------------------------

    def replace_document(
        self,
        record: DocumentRecord,
        chunks: list[Chunk],
        embeddings: np.ndarray,
        expected_version: int | None | object = UNCHECKED,
    ) -> list[str]:
        if embeddings.shape[0] != len(chunks):
            raise ValueError("one embedding per chunk is required")
        if chunks and embeddings.shape != (len(chunks), self.dim):
            raise ValueError(f"expected embeddings of shape {(len(chunks), self.dim)}")
        acl = sorted(record.acl)
        with self._pool.connection() as conn, conn.transaction():
            self._lock_document(conn, record.tenant_id, record.doc_id)
            _check_version(conn, record.tenant_id, record.doc_id, expected_version)
            old_ids = _chunk_ids(conn, record.tenant_id, record.doc_id)
            conn.execute(
                "DELETE FROM chunks WHERE tenant_id = %s AND doc_id = %s",
                (record.tenant_id, record.doc_id),
            )
            conn.execute(
                """
                INSERT INTO documents (tenant_id, doc_id, source_id, title, version,
                                       content_hash, acl, editors, metadata, chunk_count,
                                       updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (tenant_id, doc_id) DO UPDATE SET
                    source_id = excluded.source_id, title = excluded.title,
                    version = excluded.version, content_hash = excluded.content_hash,
                    acl = excluded.acl, editors = excluded.editors,
                    metadata = excluded.metadata, chunk_count = excluded.chunk_count,
                    updated_at = excluded.updated_at
                """,
                (
                    record.tenant_id,
                    record.doc_id,
                    record.source_id,
                    record.title,
                    record.version,
                    record.content_hash,
                    acl,
                    sorted(record.editors),
                    Jsonb(record.metadata),
                    len(chunks),
                    record.updated_at or _now(),
                ),
            )
            if chunks:
                with conn.cursor() as cur:
                    cur.executemany(
                        """
                        INSERT INTO chunks (chunk_id, tenant_id, doc_id, version, ordinal, text,
                                            acl, title, source_id, embedding, tsv)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::tsvector)
                        """,
                        [
                            (
                                c.id,
                                c.tenant_id,
                                c.doc_id,
                                c.version,
                                c.ordinal,
                                c.text,
                                # Like SQLite's join on documents: the document is the
                                # authority for the ACL, title and source of its chunks.
                                acl,
                                record.title,
                                record.source_id,
                                np.asarray(e, dtype=np.float32),
                                _tsvector(f"{record.title}\n{c.text}"),
                            )
                            for c, e in zip(chunks, embeddings, strict=True)
                        ],
                    )
        return old_ids

    def delete_document(
        self, tenant_id: str, doc_id: str, expected_version: int | None | object = UNCHECKED
    ) -> list[str] | None:
        with self._pool.connection() as conn, conn.transaction():
            self._lock_document(conn, tenant_id, doc_id)
            _check_version(conn, tenant_id, doc_id, expected_version)
            exists = conn.execute(
                "SELECT 1 FROM documents WHERE tenant_id = %s AND doc_id = %s",
                (tenant_id, doc_id),
            ).fetchone()
            if not exists:
                return None
            ids = _chunk_ids(conn, tenant_id, doc_id)
            # Chunks go with it (ON DELETE CASCADE). Dead tuples stay on disk until
            # VACUUM; see the ADR for right-to-be-forgotten handling.
            conn.execute(
                "DELETE FROM documents WHERE tenant_id = %s AND doc_id = %s",
                (tenant_id, doc_id),
            )
        return ids

    def bump_generation(self, tenant_id: str) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                """
                INSERT INTO tenant_generation (tenant_id, generation) VALUES (%s, 1)
                ON CONFLICT (tenant_id)
                DO UPDATE SET generation = tenant_generation.generation + 1
                """,
                (tenant_id,),
            )

    # --- helpers -----------------------------------------------------------

    @staticmethod
    def _lock_document(conn: psycopg.Connection, tenant_id: str, doc_id: str) -> None:
        # Serializes writers of one document across replicas (the ingest service's own
        # lock is per process). Released at commit/rollback.
        conn.execute(
            "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
            (f"fastrag:doc\x1f{tenant_id}\x1f{doc_id}",),
        )

    def _connection(self):
        return self._pool.connection()


class PgVectorIndex:
    """Dense search over ``chunks.embedding``. The store persists vectors, so add/remove
    are no-ops: the index is the table."""

    def __init__(self, store: PostgresDocumentStore, *, ef_search: int = 100):
        self.store = store
        self.ef_search = ef_search

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        return None

    def remove(self, chunk_ids: Iterable[str]) -> None:
        return None

    def search(
        self, vector: np.ndarray, k: int, *, access: AccessFilter
    ) -> list[tuple[Chunk, float]]:
        if k <= 0 or not access.tokens:
            return []
        query = np.asarray(vector, dtype=np.float32).reshape(self.store.dim)
        with self.store._connection() as conn, conn.transaction():
            # HNSW visits only ef_search candidates; with a selective WHERE clause most
            # of them can be filtered out. Iterative scans (pgvector >= 0.8) keep
            # searching until k rows pass the filter. Both are transaction-local.
            conn.execute(
                "SELECT set_config('hnsw.ef_search', %s, true)",
                (str(min(1000, max(self.ef_search, k))),),
            )
            if self.store.supports_iterative_scan:
                conn.execute("SELECT set_config('hnsw.iterative_scan', 'strict_order', true)")
            rows = conn.execute(
                f"""
                SELECT {_CHUNK_COLUMNS}, 1 - (embedding <=> %(q)s) AS similarity
                FROM chunks
                WHERE tenant_id = %(tenant)s AND acl && %(tokens)s::text[]
                ORDER BY embedding <=> %(q)s
                LIMIT %(k)s
                """,
                {
                    "q": query,
                    "tenant": access.tenant_id,
                    "tokens": sorted(access.tokens),
                    "k": k,
                },
            ).fetchall()
        return [(_chunk(r), float(r["similarity"])) for r in rows]

    def chunk_ids(self) -> set[str]:
        return self.store.chunk_ids()


class PgKeywordIndex:
    """Keyword search with Postgres full-text search over fastrag's own terms.

    ``chunks.tsv`` holds ``tokenize(title + text)`` (stop words removed, plurals folded),
    so candidates match exactly the terms ``BM25Index`` would. Ranking is ``ts_rank_cd``;
    coverage and matched-term counts are computed exactly as BM25Index computes them.
    """

    def __init__(self, store: PostgresDocumentStore):
        self.store = store

    def add(self, chunks: list[Chunk]) -> None:
        return None

    def remove(self, chunk_ids: Iterable[str]) -> None:
        return None

    def search(
        self, query: str, k: int, *, access: AccessFilter
    ) -> list[tuple[Chunk, float, float, int]]:
        """Return (chunk, ts_rank_cd score, coverage, matched_terms), best first."""
        terms = sorted(set(tokenize(query)))
        searchable = [t for t in terms if _indexable(t)]
        if k <= 0 or not searchable or not access.tokens:
            return []
        tsquery = " | ".join(_quote_lexeme(t) for t in searchable)
        with self.store._connection() as conn, conn.transaction():
            # Tenant-wide statistics (like BM25Index: ACLs don't change IDF): the chunk
            # count, and the document frequency of each query term.
            stats = conn.execute(
                """
                SELECT NULL AS term, count(*) AS n FROM chunks WHERE tenant_id = %(tenant)s
                UNION ALL
                SELECT term, count(*) FROM chunks c,
                       unnest(tsvector_to_array(c.tsv)) AS term
                WHERE c.tenant_id = %(tenant)s AND c.tsv @@ %(q)s::tsquery
                  AND term = ANY(%(terms)s::text[])
                GROUP BY term
                """,
                {"tenant": access.tenant_id, "q": tsquery, "terms": searchable},
            ).fetchall()
            rows = conn.execute(
                f"""
                SELECT {_CHUNK_COLUMNS}, ts_rank_cd(tsv, %(q)s::tsquery) AS score
                FROM chunks
                WHERE tenant_id = %(tenant)s AND acl && %(tokens)s::text[]
                  AND tsv @@ %(q)s::tsquery
                ORDER BY score DESC, chunk_id
                LIMIT %(k)s
                """,
                {
                    "q": tsquery,
                    "tenant": access.tenant_id,
                    "tokens": sorted(access.tokens),
                    "k": k,
                },
            ).fetchall()
        if not rows:
            return []
        n = next(int(s["n"]) for s in stats if s["term"] is None)
        df = {s["term"]: int(s["n"]) for s in stats if s["term"] is not None}

        def idf(freq: int) -> float:
            return math.log(1.0 + (n - freq + 0.5) / (freq + 0.5))

        # Terms absent from the tenant's corpus get df=0, i.e. the maximum weight.
        weights = {t: idf(df.get(t, 0)) for t in terms}
        total_weight = sum(weights.values())
        results: list[tuple[Chunk, float, float, int]] = []
        for row in rows:
            chunk = _chunk(row)
            present = set(tokenize(f"{chunk.title}\n{chunk.text}"))
            hit = [t for t in terms if t in present]
            coverage = sum(weights[t] for t in hit) / total_weight if total_weight else 0.0
            results.append((chunk, float(row["score"]), coverage, len(hit)))
        return results

    def chunk_ids(self) -> set[str]:
        return self.store.chunk_ids()


def postgres_indexes(store: PostgresDocumentStore, *, ef_search: int = 100) -> SearchIndexes:
    """Search indexes backed by the store's tables. ``rebuild_from`` is unnecessary (and
    harmless: add/remove are no-ops); every replica queries the same rows."""
    return SearchIndexes(PgVectorIndex(store, ef_search=ef_search), PgKeywordIndex(store))


# --- helpers ---------------------------------------------------------------------------


def _configure_connection(conn: psycopg.Connection) -> None:
    register_vector(conn)
    conn.commit()  # the type lookup opened a transaction; return the connection idle


def _version_at_least(version: str, minimum: tuple[int, ...]) -> bool:
    parts = []
    for piece in version.split("."):
        digits = "".join(ch for ch in piece if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts) >= minimum


def _meta(conn: psycopg.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key = %s", (key,)).fetchone()
    return row["value"] if row else None


def _set_meta(conn: psycopg.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta (key, value) VALUES (%s, %s) "
        "ON CONFLICT (key) DO UPDATE SET value = excluded.value",
        (key, value),
    )


def _check_version(
    conn: psycopg.Connection, tenant_id: str, doc_id: str, expected: int | None | object
) -> None:
    """Compare-and-set under the document's advisory lock (see ``_lock_document``)."""
    if expected is UNCHECKED:
        return
    row = conn.execute(
        "SELECT version FROM documents WHERE tenant_id = %s AND doc_id = %s",
        (tenant_id, doc_id),
    ).fetchone()
    current = row["version"] if row else None
    if current != expected:
        raise ConcurrentModification(
            f"document {doc_id!r} is at version {current}, expected {expected}"
        )


def _chunk_ids(conn: psycopg.Connection, tenant_id: str, doc_id: str) -> list[str]:
    rows = conn.execute(
        "SELECT chunk_id FROM chunks WHERE tenant_id = %s AND doc_id = %s ORDER BY ordinal",
        (tenant_id, doc_id),
    ).fetchall()
    return [r["chunk_id"] for r in rows]


def _indexable(term: str) -> bool:
    return _WORD.fullmatch(term) is not None and len(term.encode()) <= _MAX_LEXEME_BYTES


def _quote_lexeme(term: str) -> str:
    # tokenize() yields \w+ tokens, but quote defensively: inside quotes, tsvector and
    # tsquery input take the lexeme literally (no parsing, no operators).
    return "'" + term.replace("\\", "\\\\").replace("'", "''") + "'"


def _tsvector(text: str) -> str:
    """A tsvector literal of fastrag's own terms, with positions for ts_rank_cd.

    Built here rather than with to_tsvector() so index and query use exactly the
    tokenizer BM25Index uses (stop words, plural folding), which Postgres' parser and
    dictionaries would not reproduce.
    """
    positions: dict[str, list[int]] = {}
    for pos, term in enumerate(tokenize(text), start=1):
        if not _indexable(term):
            continue
        slots = positions.setdefault(term, [])
        if len(slots) < _MAX_POSITIONS_PER_LEXEME:
            slots.append(min(pos, _MAX_POSITION))
    return " ".join(
        f"{_quote_lexeme(term)}:{','.join(map(str, sorted(set(p))))}"
        for term, p in positions.items()
    )


def _as_array(value: Any) -> np.ndarray:
    # Text-format results load as numpy arrays; binary ones (server-side cursors) as
    # pgvector.Vector objects.
    if hasattr(value, "to_numpy"):
        value = value.to_numpy()
    return np.asarray(value, dtype=np.float32)


def _chunk(row: dict[str, Any]) -> Chunk:
    return Chunk(
        id=row["chunk_id"],
        tenant_id=row["tenant_id"],
        doc_id=row["doc_id"],
        version=row["version"],
        ordinal=row["ordinal"],
        text=row["text"],
        acl=frozenset(row["acl"]),
        source_id=row["source_id"],
        title=row["title"],
    )


def _record(row: dict[str, Any]) -> DocumentRecord:
    return DocumentRecord(
        tenant_id=row["tenant_id"],
        doc_id=row["doc_id"],
        source_id=row["source_id"],
        title=row["title"],
        version=row["version"],
        content_hash=row["content_hash"],
        acl=frozenset(row["acl"]),
        editors=frozenset(row["editors"]),
        metadata=row["metadata"],
        chunk_count=row["chunk_count"],
        updated_at=row["updated_at"],
    )


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
