"""PostgreSQL + pgvector adapters against a real database (skipped without a DSN)."""

from __future__ import annotations

import os
import threading

import numpy as np
import pytest

if not os.environ.get("FASTRAG_TEST_POSTGRES_DSN"):
    pytest.skip("FASTRAG_TEST_POSTGRES_DSN is not set", allow_module_level=True)

psycopg = pytest.importorskip("psycopg")
pytest.importorskip("psycopg_pool")
pytest.importorskip("pgvector")

from fastrag.config import Settings  # noqa: E402
from fastrag.domain import (  # noqa: E402
    EVERYONE,
    Chunk,
    DocumentIn,
    DocumentRecord,
    Principal,
    Role,
    group_token,
)
from fastrag.embeddings.hashing import HashingEmbedder  # noqa: E402
from fastrag.errors import ConfigurationError, NotFound  # noqa: E402
from fastrag.generation import ExtractiveGenerator  # noqa: E402
from fastrag.security.acl import AccessFilter  # noqa: E402
from fastrag.service import RAGService  # noqa: E402
from fastrag.storage import BM25Index  # noqa: E402
from fastrag.storage.postgres import (  # noqa: E402
    PgKeywordIndex,
    PgVectorIndex,
    PostgresDocumentStore,
    postgres_indexes,
)

pytestmark = pytest.mark.integration

ACME = AccessFilter.for_principal(Principal("acme", "alice"))
DIM = 4


def chunk(i: int, tenant="acme", acl=frozenset({EVERYONE}), text=None, doc="d1") -> Chunk:
    return Chunk(
        id=f"{tenant}-{doc}-{i}",
        tenant_id=tenant,
        doc_id=doc,
        version=1,
        ordinal=i,
        text=text or f"chunk number {i}",
        acl=acl,
        source_id=f"{doc}.md",
        title="T",
    )


def record(doc="d1", version=1, tenant="acme", acl=frozenset({EVERYONE})) -> DocumentRecord:
    return DocumentRecord(
        tenant_id=tenant,
        doc_id=doc,
        source_id=f"{doc}.md",
        title="T",
        version=version,
        content_hash=f"h{version}",
        acl=acl,
        metadata={"owner": "ops", "tags": ["a", "b"]},
        chunk_count=0,
        updated_at="",
        editors=frozenset({"user:alice"}),
    )


def unit(hot: int, dim: int = DIM) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    v[hot] = 1.0
    return v


@pytest.fixture
def store(pg_dsn):
    s = PostgresDocumentStore(pg_dsn(), dim=DIM, pool_size=4)
    yield s
    s.close()


# --- (a) the DocumentStore contract, mirroring TestSQLiteStore --------------------------


class TestStoreContract:
    def test_replace_delete_and_generation(self, store):
        assert store.generation("acme") == 0
        v1 = [chunk(0), chunk(1)]
        assert store.replace_document(record(), v1, np.ones((2, DIM), np.float32)) == []
        assert store.generation("acme") == 0  # the ingest service bumps after indexing
        store.bump_generation("acme")
        assert store.generation("acme") == 1
        v2 = [chunk(5)]
        replaced = store.replace_document(record(version=2), v2, np.ones((1, DIM), np.float32))
        assert sorted(replaced) == ["acme-d1-0", "acme-d1-1"]
        assert [c.id for c, _ in store.iter_chunks()] == ["acme-d1-5"]
        doc = store.get_document("acme", "d1")
        assert doc.version == 2 and doc.chunk_count == 1
        assert doc.editors == frozenset({"user:alice"})
        assert doc.metadata == {"owner": "ops", "tags": ["a", "b"]}
        assert doc.updated_at  # filled in when the record had none
        assert [d.doc_id for d in store.list_documents("acme")] == ["d1"]
        assert store.list_documents("globex") == []
        assert store.delete_document("acme", "d1") == ["acme-d1-5"]
        assert store.delete_document("acme", "d1") is None
        assert list(store.iter_chunks()) == []
        assert store.get_document("acme", "d1") is None

    def test_iter_chunks_round_trips_embeddings(self, store):
        vectors = np.eye(2, DIM, dtype=np.float32)
        store.replace_document(record(), [chunk(0), chunk(1)], vectors)
        got = list(store.iter_chunks())
        assert [c.id for c, _ in got] == ["acme-d1-0", "acme-d1-1"]
        np.testing.assert_allclose(np.stack([v for _, v in got]), vectors)
        assert got[0][0].acl == frozenset({EVERYONE}) and got[0][0].title == "T"

    def test_failed_write_rolls_back(self, store):
        store.replace_document(record(), [chunk(0)], np.ones((1, DIM), np.float32))
        with pytest.raises(psycopg.IntegrityError):  # duplicate chunk id inside one write
            store.replace_document(
                record(version=2), [chunk(9), chunk(9)], np.ones((2, DIM), np.float32)
            )
        assert store.get_document("acme", "d1").version == 1
        assert [c.id for c, _ in store.iter_chunks()] == ["acme-d1-0"]

    def test_wrong_embedding_shape_is_rejected(self, store):
        with pytest.raises(ValueError):
            store.replace_document(record(), [chunk(0)], np.ones((1, DIM + 1), np.float32))
        with pytest.raises(ValueError):
            store.replace_document(record(), [chunk(0)], np.ones((2, DIM), np.float32))

    def test_embedder_pinning(self, pg_dsn, store):
        store.check_embedder("hashing:4")
        other = PostgresDocumentStore(pg_dsn(), dim=DIM)
        try:
            other.check_embedder("hashing:4")
            with pytest.raises(ConfigurationError):
                other.check_embedder("st:other")
        finally:
            other.close()

    def test_dimension_is_pinned(self, pg_dsn, store):
        with pytest.raises(ConfigurationError):
            PostgresDocumentStore(pg_dsn(), dim=DIM * 2)

    def test_migrations_are_idempotent_and_safe_concurrently(self, pg_dsn):
        dsn = pg_dsn()
        errors: list[BaseException] = []
        stores: list[PostgresDocumentStore] = []

        def start() -> None:
            try:
                stores.append(PostgresDocumentStore(dsn, dim=DIM, pool_size=1))
            except BaseException as exc:  # noqa: BLE001 - collected and asserted below
                errors.append(exc)

        threads = [threading.Thread(target=start) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
        stores.append(PostgresDocumentStore(dsn, dim=DIM))  # and once more, sequentially
        for s in stores:
            s.close()
        with psycopg.connect(dsn) as conn:
            version = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
        assert version[0] == str(PostgresDocumentStore.SCHEMA_VERSION)

    def test_newer_schema_is_refused(self, pg_dsn, store):
        with psycopg.connect(pg_dsn()) as conn:
            conn.execute("UPDATE meta SET value = '99' WHERE key = 'schema_version'")
        with pytest.raises(ConfigurationError):
            PostgresDocumentStore(pg_dsn(), dim=DIM)

    def test_ping(self, store):
        store.ping()


# --- (b) pre-filter correctness ---------------------------------------------------------


def _crowded_store(store) -> list[Chunk]:
    """50 other-tenant and 50 HR-only chunks that match perfectly; 3 visible weak ones."""
    perfect = "secret merger plan acquisition"
    other = [chunk(i, tenant="globex", text=perfect) for i in range(50)]
    store.replace_document(
        record(tenant="globex"), other, np.stack([unit(0)] * 50).astype(np.float32)
    )
    hr = frozenset({group_token("hr")})
    hidden = [chunk(i, acl=hr, doc="hr", text=perfect) for i in range(50)]
    store.replace_document(record(doc="hr", acl=hr), hidden, np.stack([unit(0)] * 50))
    visible = [chunk(i, doc="pub", text=f"picnic plan number {i}") for i in range(3)]
    weak = (unit(0) * 0.1 + unit(1)).astype(np.float32)
    store.replace_document(record(doc="pub"), visible, np.stack([weak] * 3))
    return visible


@pytest.mark.parametrize("force_index", [False, True], ids=["planner", "no-seqscan"])
def test_vector_search_prefilters(pg_dsn, force_index):
    # With seq scans disabled the planner must use an index (HNSW or btree); the ACL
    # filter has to hold either way.
    dsn = pg_dsn(enable_seqscan="off") if force_index else pg_dsn()
    store = PostgresDocumentStore(dsn, dim=DIM)
    try:
        visible = _crowded_store(store)
        results = PgVectorIndex(store).search(unit(0), k=3, access=ACME)
        assert {c.id for c, _ in results} == {c.id for c in visible}
        assert all(score == pytest.approx(0.1 / np.sqrt(1.01), abs=1e-5) for _, score in results)
        hr_reader = AccessFilter.for_principal(Principal("acme", "hank", frozenset({"hr"})))
        top = PgVectorIndex(store).search(unit(0), k=3, access=hr_reader)
        assert all(c.doc_id == "hr" and s == pytest.approx(1.0) for c, s in top)
        assert (
            PgVectorIndex(store).search(unit(0), k=3, access=AccessFilter("acme", frozenset()))
            == []
        )
    finally:
        store.close()


def test_keyword_search_prefilters(store):
    visible = _crowded_store(store)
    results = PgKeywordIndex(store).search("secret merger plan acquisition", 3, access=ACME)
    assert {c.id for c, *_ in results} == {c.id for c in visible}
    for _, score, coverage, matched in results:
        assert score > 0 and matched == 1 and 0 < coverage < 1


def test_keyword_search_matches_bm25_semantics(store):
    texts = [
        "Error NW-4012 means the VPN certificate expired.",
        "The VPN client connects to the corporate network.",
        "Laptops are refreshed every three years under the hardware policies.",
    ]
    chunks = [chunk(i, text=t) for i, t in enumerate(texts)]
    store.replace_document(record(), chunks, np.ones((3, DIM), np.float32))
    bm25 = BM25Index()
    bm25.add(chunks)
    pg = PgKeywordIndex(store)
    for query in ["what is error NW-4012", "laptop policy", "vpn zebra", "the and of"]:
        expected = {c.id: (cov, m) for c, _, cov, m in bm25.search(query, 5, access=ACME)}
        got = {c.id: (cov, m) for c, _, cov, m in pg.search(query, 5, access=ACME)}
        assert got.keys() == expected.keys(), query
        for cid, (cov, m) in got.items():
            assert cov == pytest.approx(expected[cid][0]) and m == expected[cid][1], query
    top = pg.search("what is error NW-4012", 2, access=ACME)
    assert top[0][0].id == "acme-d1-0" and top[0][2] == pytest.approx(1.0)


def test_keyword_query_is_not_injectable(store):
    store.replace_document(record(), [chunk(0, text="alpha beta")], np.ones((1, DIM)))
    for query in ["alpha' | 'beta", "alpha & !beta", "alpha:* <-> (beta", "'; DROP TABLE x; --"]:
        PgKeywordIndex(store).search(query, 5, access=ACME)  # must not raise
    assert [c.id for c, *_ in PgKeywordIndex(store).search("ALPHA", 5, access=ACME)] == [
        "acme-d1-0"
    ]


def test_index_helpers(store):
    store.replace_document(record(), [chunk(0), chunk(1)], np.eye(2, DIM, dtype=np.float32))
    indexes = postgres_indexes(store)
    assert indexes.rebuild_from(store, DIM) == 2  # add() is a no-op: harmless
    assert indexes.consistent_with(store)
    indexes.apply(remove_ids=["acme-d1-0"])  # remove() is a no-op too: the store owns rows
    assert indexes.vectors.chunk_ids() == {"acme-d1-0", "acme-d1-1"}


# --- (c) end to end through RAGService --------------------------------------------------

EMBED_DIM = 256
ADMIN_A = Principal("tenant-a", "root-a", roles=frozenset({Role.ADMIN}))
ADMIN_B = Principal("tenant-b", "root-b", roles=frozenset({Role.ADMIN}))
ALICE = Principal("tenant-a", "alice", frozenset({"finance"}))
BOB = Principal("tenant-a", "bob")
CAROL = Principal("tenant-b", "carol", frozenset({"finance"}))


def _service(dsn: str) -> tuple[RAGService, PostgresDocumentStore]:
    embedder = HashingEmbedder(dim=EMBED_DIM)
    store = PostgresDocumentStore(dsn, dim=EMBED_DIM, pool_size=4)
    store.check_embedder(embedder.name)
    settings = Settings.for_testing(rate_limit_per_minute=0, user_rate_limit_per_minute=0)
    service = RAGService(
        settings=settings,
        store=store,
        indexes=postgres_indexes(store),
        embedder=embedder,
        generator=ExtractiveGenerator(),
    )
    return service, store


def _sources(result) -> set[str]:
    return {s.chunk.source_id for s in result.chunks}


def test_end_to_end_isolation_delete_and_versions(pg_dsn):
    service, store = _service(pg_dsn())
    try:
        finance = frozenset({group_token("finance")})
        service.ingest(
            ADMIN_A,
            DocumentIn(
                "budget.md", "The 2027 marketing budget is 4.2 million euros.", finance, "Budget"
            ),
        )
        service.ingest(
            ADMIN_A,
            DocumentIn(
                "vpn.md", "Error NW-4012 means the VPN certificate expired.", frozenset({EVERYONE})
            ),
        )
        service.ingest(
            ADMIN_B,
            DocumentIn(
                "budget.md", "Tenant B marketing budget is 9 million dollars.", finance, "Budget"
            ),
        )

        # Dense search always returns nearest neighbours, so vpn.md may appear too; what
        # matters is that only the caller's tenant and groups are ever visible.
        alice_hits = service.search(ALICE, "marketing budget").chunks
        assert "budget.md" in {s.chunk.source_id for s in alice_hits}
        assert all(s.chunk.tenant_id == "tenant-a" for s in alice_hits)
        assert "budget.md" not in _sources(service.search(BOB, "marketing budget"))
        carol_hits = service.search(CAROL, "marketing budget").chunks
        assert {s.chunk.tenant_id for s in carol_hits} == {"tenant-b"}

        answer = service.ask(ALICE, "What is the marketing budget?")
        assert not answer.abstained and "4.2 million" in answer.text
        assert "9 million" not in answer.text
        assert service.ask(BOB, "What is the 2027 marketing budget?").abstained

        # Re-ingest creates a new version and replaces the old chunks.
        result = service.ingest(
            ADMIN_A,
            DocumentIn(
                "budget.md", "The 2027 marketing budget is 5.0 million euros.", finance, "Budget"
            ),
        )
        assert (result.status, result.version) == ("updated", 2)
        texts = [s.chunk.text for s in service.search(ALICE, "marketing budget").chunks]
        assert any("5.0 million" in t for t in texts)
        assert not any("4.2 million" in t for t in texts)
        assert "5.0 million" in service.ask(ALICE, "What is the marketing budget?").text

        # Delete removes it from every result path.
        service.delete(ADMIN_A, result.doc_id)
        assert "budget.md" not in _sources(service.search(ALICE, "marketing budget"))
        assert service.ask(ALICE, "What is the 2027 marketing budget?").abstained
        with pytest.raises(NotFound):
            service.delete(ADMIN_A, result.doc_id)
        assert "budget.md" in _sources(service.search(CAROL, "marketing budget"))
    finally:
        store.close()


# --- (d) two replicas on one database ---------------------------------------------------


def test_two_replicas_share_one_database(pg_dsn):
    dsn = pg_dsn()
    writer, writer_store = _service(dsn)
    reader, reader_store = _service(dsn)
    try:
        assert reader.search(BOB, "parental leave weeks").chunks == []
        result = writer.ingest(
            ADMIN_A,
            DocumentIn(
                "leave.md", "Parental leave is 16 weeks at full pay.", frozenset({EVERYONE})
            ),
        )
        # Visible to the other replica immediately: no index rebuild, no event bus.
        assert _sources(reader.search(BOB, "parental leave weeks")) == {"leave.md"}
        assert reader_store.generation("tenant-a") == writer_store.generation("tenant-a") == 1
        assert "16 weeks" in reader.ask(BOB, "How many weeks of parental leave?").text

        writer.delete(ADMIN_A, result.doc_id)
        assert reader.search(BOB, "parental leave weeks").chunks == []
        assert reader_store.generation("tenant-a") == 2
    finally:
        writer_store.close()
        reader_store.close()


@pytest.mark.integration
def test_check_only_startup_needs_a_migrated_schema(pg_dsn):
    """With migrate=False the app never changes the schema (it can run as a DML-only
    role); it refuses to start until `fastrag db migrate` has brought it up to date."""
    dsn = pg_dsn()
    with pytest.raises(ConfigurationError):
        PostgresDocumentStore(dsn, dim=DIM, migrate=False)
    PostgresDocumentStore(dsn, dim=DIM).close()  # what `fastrag db migrate` does
    store = PostgresDocumentStore(dsn, dim=DIM, migrate=False)
    store.close()
    with pytest.raises(ConfigurationError):  # wrong embedder dimension is still caught
        PostgresDocumentStore(dsn, dim=DIM + 1, migrate=False)


@pytest.mark.integration
def test_concurrent_writers_never_produce_duplicate_versions(pg_dsn):
    from fastrag.domain import DocumentRecord as Record
    from fastrag.errors import ConcurrentModification

    store = PostgresDocumentStore(pg_dsn(), dim=DIM)
    try:
        rec = Record("t", "d", "d.md", "", 1, "h1", frozenset({EVERYONE}), {}, 0, "")
        empty = np.zeros((0, DIM), dtype=np.float32)
        store.replace_document(rec, [], empty, expected_version=None)
        with pytest.raises(ConcurrentModification):
            store.replace_document(rec, [], empty, expected_version=None)
        v2 = Record("t", "d", "d.md", "", 2, "h2", frozenset({EVERYONE}), {}, 0, "")
        store.replace_document(v2, [], empty, expected_version=1)
        with pytest.raises(ConcurrentModification):
            store.delete_document("t", "d", expected_version=1)
    finally:
        store.close()
