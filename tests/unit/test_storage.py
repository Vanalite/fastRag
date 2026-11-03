import sqlite3

import numpy as np
import pytest

from fastrag.domain import EVERYONE, Chunk, DocumentRecord, Principal, group_token
from fastrag.errors import ConfigurationError
from fastrag.security.acl import AccessFilter
from fastrag.storage import BM25Index, NumpyVectorIndex, SearchIndexes, SQLiteDocumentStore

ACME = AccessFilter.for_principal(Principal("acme", "alice"))


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
    )


def unit(dim: int, hot: int) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    v[hot] = 1.0
    return v


def record(doc="d1", version=1, tenant="acme") -> DocumentRecord:
    return DocumentRecord(
        tenant_id=tenant,
        doc_id=doc,
        source_id=f"{doc}.md",
        title="T",
        version=version,
        content_hash=f"h{version}",
        acl=frozenset({EVERYONE}),
        metadata={},
        chunk_count=0,
        updated_at="",
    )


class TestVectorIndex:
    def test_prefilter_returns_k_results_despite_better_hidden_matches(self):
        index = NumpyVectorIndex(dim=4)
        # 50 chunks in another tenant and 50 restricted chunks match the query perfectly...
        index.add([chunk(i, tenant="globex") for i in range(50)], np.stack([unit(4, 0)] * 50))
        hidden = [chunk(i, acl=frozenset({group_token("hr")}), doc="hr") for i in range(50)]
        index.add(hidden, np.stack([unit(4, 0)] * 50))
        # ...while the caller's visible chunks match it only weakly.
        visible = [chunk(i) for i in range(3)]
        index.add(visible, np.stack([unit(4, 0) * 0.1 + unit(4, 1)] * 3))
        results = index.search(unit(4, 0), k=3, access=ACME)
        assert {c.id for c, _ in results} == {c.id for c in visible}

    def test_remove(self):
        index = NumpyVectorIndex(dim=4)
        index.add([chunk(0), chunk(1)], np.stack([unit(4, 0), unit(4, 1)]))
        index.remove(["acme-d1-0"])
        assert index.chunk_ids() == {"acme-d1-1"}

    def test_shape_mismatch(self):
        with pytest.raises(ValueError):
            NumpyVectorIndex(dim=4).add([chunk(0)], np.zeros((1, 3), dtype=np.float32))


class TestBM25:
    def test_exact_identifier_ranks_first_and_coverage(self):
        index = BM25Index()
        index.add(
            [
                chunk(0, text="Error NW-4012 means the VPN certificate expired."),
                chunk(1, text="The VPN client connects to the corporate network."),
            ]
        )
        results = index.search("what is error NW-4012", k=2, access=ACME)
        assert results[0][0].id == "acme-d1-0"
        assert results[0][2] == pytest.approx(1.0)  # error, nw, 4012 all present

    def test_acl_and_tenant_prefilter(self):
        index = BM25Index()
        index.add([chunk(0, tenant="globex", text="secret merger plan")])
        index.add([chunk(1, acl=frozenset({group_token("hr")}), text="secret salary bands")])
        assert index.search("secret", k=5, access=ACME) == []

    def test_remove_keeps_statistics_consistent(self):
        index = BM25Index()
        index.add([chunk(0, text="alpha beta"), chunk(1, text="alpha gamma")])
        index.remove(["acme-d1-0"])
        assert index.chunk_ids() == {"acme-d1-1"}
        assert [c.id for c, *_ in index.search("beta", 5, access=ACME)] == []
        assert [c.id for c, *_ in index.search("alpha", 5, access=ACME)] == ["acme-d1-1"]


class TestSQLiteStore:
    def test_replace_delete_and_generation(self):
        store = SQLiteDocumentStore()
        assert store.generation("acme") == 0
        v1 = [chunk(0), chunk(1)]
        assert store.replace_document(record(), v1, np.zeros((2, 4), np.float32)) == []
        assert store.generation("acme") == 0  # the ingest service bumps after indexing
        store.bump_generation("acme")
        assert store.generation("acme") == 1
        v2 = [chunk(5)]
        replaced = store.replace_document(record(version=2), v2, np.ones((1, 4), np.float32))
        assert sorted(replaced) == ["acme-d1-0", "acme-d1-1"]
        assert [c.id for c, _ in store.iter_chunks()] == ["acme-d1-5"]
        assert store.get_document("acme", "d1").version == 2
        assert store.delete_document("acme", "d1") == ["acme-d1-5"]
        assert store.delete_document("acme", "d1") is None
        assert list(store.iter_chunks()) == []

    def test_failed_write_rolls_back(self):
        store = SQLiteDocumentStore()
        store.replace_document(record(), [chunk(0)], np.zeros((1, 4), np.float32))
        with pytest.raises(sqlite3.IntegrityError):  # duplicate chunk id inside one write
            store.replace_document(
                record(version=2), [chunk(9), chunk(9)], np.zeros((2, 4), np.float32)
            )
        assert store.get_document("acme", "d1").version == 1
        assert [c.id for c, _ in store.iter_chunks()] == ["acme-d1-0"]

    def test_embedder_pinning(self, tmp_path):
        path = tmp_path / "db.sqlite"
        SQLiteDocumentStore(path).check_embedder("hashing:1024")
        store = SQLiteDocumentStore(path)
        store.check_embedder("hashing:1024")
        with pytest.raises(ConfigurationError):
            store.check_embedder("st:other")

    def test_indexes_rebuild_from_store(self):
        store = SQLiteDocumentStore()
        store.replace_document(record(), [chunk(0), chunk(1)], np.eye(2, 4, dtype=np.float32))
        indexes = SearchIndexes(NumpyVectorIndex(4), BM25Index())
        assert indexes.rebuild_from(store, 4) == 2
        assert indexes.consistent_with(store)
