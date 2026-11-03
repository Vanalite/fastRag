"""Deletion, versioning, idempotency, persistence."""

from __future__ import annotations

import json

import pytest
from conftest import audit_dump, doc, make_principal

from fastrag.config import Settings
from fastrag.domain import Role
from fastrag.errors import NotFound
from fastrag.service import RAGService

GONE = "CANARY-FORGET-ME-3d9e"
TEXT = f"Employee onboarding handbook entry {GONE} describes laptop provisioning steps."
Q = f"Employee onboarding handbook entry {GONE}"


def test_delete_leaves_no_trace(service, generator, admin_a):
    reader = make_principal(user="r")
    res = service.ingest(admin_a, doc("hb.txt", TEXT))
    assert service.ask(reader, Q).sources  # populate the cache
    service.search(reader, Q)

    service.delete(admin_a, res.doc_id)

    assert all(s.chunk.doc_id != res.doc_id for s in service.search(reader, Q).chunks)
    ans = service.ask(reader, Q)
    assert ans.abstained and not ans.cached and GONE not in ans.text
    assert ans.sources == []
    assert service.list_documents(admin_a) == []
    assert all(GONE not in c.text for c, _ in service.store.iter_chunks())
    assert list(service.store.iter_chunks()) == []
    assert service.indexes.vectors.chunk_ids() == set()
    assert service.indexes.keywords.chunk_ids() == set()
    assert service.indexes.consistent_with(service.store)
    assert GONE not in audit_dump(service)
    assert service.audit.verify()
    with pytest.raises(NotFound):
        service.delete(admin_a, res.doc_id)


def test_deleted_document_stays_gone_after_restart(tmp_path, generator, admin_a):
    path = str(tmp_path / "db.sqlite")
    s1 = RAGService.from_settings(Settings.for_testing(database_path=path), generator=generator)
    keep = s1.ingest(admin_a, doc("keep.txt", "Retention schedule keeps invoices for seven years."))
    gone = s1.ingest(admin_a, doc("hb.txt", TEXT))
    s1.delete(admin_a, gone.doc_id)

    s2 = RAGService.from_settings(Settings.for_testing(database_path=path), generator=generator)
    reader = make_principal(user="r")
    assert all(s.chunk.doc_id != gone.doc_id for s in s2.search(reader, Q).chunks)
    assert all(GONE not in s.chunk.text for s in s2.search(reader, Q).chunks)
    assert s2.ask(reader, Q).abstained
    assert [d.doc_id for d in s2.list_documents(admin_a)] == [keep.doc_id]
    assert all(GONE not in c.text for c, _ in s2.store.iter_chunks())
    assert s2.indexes.consistent_with(s2.store)
    # The surviving document is still served after restart.
    assert s2.search(reader, "retention schedule invoices seven years").chunks


def test_audit_contains_no_document_text(service, admin_a):
    res = service.ingest(admin_a, doc("hb.txt", TEXT, title="Handbook"))
    service.ask(admin_a, Q)
    service.search(admin_a, Q)
    service.delete(admin_a, res.doc_id)
    dump = audit_dump(service)
    assert GONE not in dump
    assert "laptop provisioning" not in dump
    assert Q not in dump  # question stored only as a hash by default


def test_reingest_new_version_replaces_old(service, admin_a):
    reader = make_principal(user="r")
    old = "Remote work stipend is CANARY-OLD-V1-aa11 per month for staff."
    new = "Remote work stipend is CANARY-NEW-V2-bb22 per month for staff, updated."
    r1 = service.ingest(admin_a, doc("stipend.txt", old))
    assert (r1.version, r1.status) == (1, "created")
    r2 = service.ingest(admin_a, doc("stipend.txt", new))
    assert (r2.version, r2.status) == (2, "updated")
    assert r2.doc_id == r1.doc_id

    chunks = service.search(reader, "remote work stipend per month staff", 10).chunks
    assert chunks
    assert all(s.chunk.version == 2 for s in chunks)
    texts = " ".join(s.chunk.text for s in chunks)
    assert "CANARY-NEW-V2-bb22" in texts and "CANARY-OLD-V1-aa11" not in texts
    stored = list(service.store.iter_chunks())
    assert len(stored) == r2.chunks
    assert all(c.version == 2 for c, _ in stored)
    assert service.indexes.consistent_with(service.store)
    docs = service.list_documents(admin_a)
    assert len(docs) == 1 and docs[0].version == 2


def test_identical_reingest_is_unchanged(service, admin_a):
    r1 = service.ingest(admin_a, doc("a.txt", TEXT))
    before = {c.id for c, _ in service.store.iter_chunks()}
    r2 = service.ingest(admin_a, doc("a.txt", TEXT))
    assert r2.status == "unchanged" and r2.version == r1.version == 1
    assert {c.id for c, _ in service.store.iter_chunks()} == before
    assert service.indexes.consistent_with(service.store)


def test_same_source_id_in_two_tenants_do_not_collide(service, admin_a, admin_b):
    ra = service.ingest(admin_a, doc("same.txt", "Alpha tenant text about gardening tools."))
    rb = service.ingest(admin_b, doc("same.txt", "Beta tenant text about gardening tools."))
    assert ra.doc_id != rb.doc_id
    assert rb.version == 1 and rb.status == "created"


def test_audit_chain_verifies_after_mixed_workload_and_detects_tampering(service, admin_a):
    from fastrag.errors import FastRagError
    from fastrag.observability import verify_chain

    reader = make_principal(user="r")
    r = service.ingest(admin_a, doc("a.txt", TEXT))
    service.search(reader, Q)
    service.ask(reader, Q)
    service.ask(reader, "zzz qqq unrelated nonsense")
    service.audit_events(admin_a)
    with pytest.raises(FastRagError):
        service.ingest(reader, doc("b.txt", "nope text"))
    service.delete(admin_a, r.doc_id)

    entries = service.audit.sink.read()
    assert len(entries) >= 6
    assert service.audit.verify() and verify_chain(entries)
    assert any(e["outcome"] == "denied" for e in entries)

    # Modify a field in a copy.
    mod = json.loads(json.dumps(entries))
    mod[2]["outcome"] = "success-forged"
    assert not verify_chain(mod)
    # Remove an entry.
    assert not verify_chain(entries[:1] + entries[2:])
    # Reorder.
    assert not verify_chain([entries[1], entries[0], *entries[2:]])
    # Original untouched and still valid.
    assert verify_chain(entries)
    assert Role.ADMIN in admin_a.roles
