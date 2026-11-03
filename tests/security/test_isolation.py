"""Tenant isolation, group/user ACLs and cache isolation."""

from __future__ import annotations

import pytest
from conftest import CANARY_A, CANARY_B, doc, make_principal

from fastrag.domain import Role, group_token, user_token
from fastrag.errors import NotFound

TEXT_A = f"The quarterly budget review policy is {CANARY_A} and covers travel spending."
TEXT_B = f"The quarterly budget review policy is {CANARY_B} and covers travel spending."


@pytest.fixture
def two_tenants(service, admin_a, admin_b):
    ra = service.ingest(admin_a, doc("budget.txt", TEXT_A))
    rb = service.ingest(admin_b, doc("budget.txt", TEXT_B))
    return ra, rb


QUERIES = [
    "quarterly budget review policy",
    CANARY_B,
    f"The quarterly budget review policy is {CANARY_B}",
]


@pytest.mark.parametrize("query", QUERIES)
def test_search_never_returns_other_tenant(service, generator, admin_a, two_tenants, query):
    _, rb = two_tenants
    res = service.search(admin_a, query, 10)
    for s in res.chunks:
        assert s.chunk.tenant_id == "tenant-a"
        assert CANARY_B not in s.chunk.text
        assert s.chunk.doc_id != rb.doc_id


@pytest.mark.parametrize("query", QUERIES)
def test_ask_never_leaks_other_tenant(service, generator, admin_a, two_tenants, query):
    _, rb = two_tenants
    ans = service.ask(admin_a, query, 10)
    assert CANARY_B not in ans.text
    assert all(CANARY_B not in s.chunk.text for s in ans.sources)
    assert all(c.doc_id != rb.doc_id for c in ans.citations)
    assert all(CANARY_B not in t for t in generator.shown_texts)
    assert rb.doc_id not in str(service.audit_events(admin_a))
    for entry in service.audit.events("tenant-a"):
        assert rb.doc_id not in str(entry)
        assert CANARY_B not in str(entry)


def test_exact_canary_query_abstains_for_other_tenant(service, generator, admin_a):
    from fastrag.domain import Role  # noqa: F401

    service.ingest(make_principal("tenant-b", "b", roles=(Role.ADMIN,)), doc("x", TEXT_B))
    ans = service.ask(admin_a, CANARY_B)
    assert ans.abstained
    assert generator.calls == []


def test_list_documents_scoped_to_tenant(service, admin_a, two_tenants):
    ra, rb = two_tenants
    ids = {d.doc_id for d in service.list_documents(admin_a)}
    assert ids == {ra.doc_id}
    assert rb.doc_id not in ids


def test_cannot_delete_other_tenants_document(service, admin_a, admin_b, two_tenants):
    _, rb = two_tenants
    with pytest.raises(NotFound):
        service.delete(admin_a, rb.doc_id)
    assert [d.doc_id for d in service.list_documents(admin_b)] == [rb.doc_id]
    assert service.search(admin_b, CANARY_B).chunks


# --- group / user ACLs -----------------------------------------------------------

SECRET = "CANARY-GROUP-FIN-55aa"
SECRET_TEXT = f"Executive compensation bands are confidential: {SECRET} applies to directors."
SECRET_PHRASE = f"Executive compensation bands are confidential: {SECRET}"


@pytest.mark.parametrize(
    ("acl", "insider", "outsider"),
    [
        (
            [group_token("finance")],
            make_principal(user="ivy", groups=("finance",)),
            make_principal(user="oscar", groups=("eng",)),
        ),
        (
            [user_token("ivy")],
            make_principal(user="ivy"),
            make_principal(user="oscar", groups=("finance",)),
        ),
    ],
    ids=["group", "user"],
)
def test_acl_hides_document_from_same_tenant_outsider(
    service, generator, admin_a, acl, insider, outsider
):
    service.ingest(admin_a, doc("comp.txt", SECRET_TEXT, acl=acl))

    # Outsider: nothing in retrieval, generator input, response, or audit.
    res = service.search(outsider, SECRET_PHRASE, 10)
    assert all(SECRET not in s.chunk.text for s in res.chunks)
    ans = service.ask(outsider, SECRET_PHRASE, 10)
    assert ans.abstained and ans.sources == [] and SECRET not in ans.text
    assert generator.calls == []
    assert service.list_documents(outsider) == []
    entries = [e for e in service.audit.sink.read() if e["user_id"] == outsider.user_id]
    assert entries
    for e in entries:
        assert e["details"].get("shown_chunk_ids", []) == []
        assert e["details"].get("chunk_ids", []) == []

    # Insider sees it.
    assert any(SECRET in s.chunk.text for s in service.search(insider, SECRET_PHRASE, 10).chunks)
    ans = service.ask(insider, SECRET_PHRASE, 10)
    assert not ans.abstained and ans.sources
    assert any(SECRET in t for t in generator.shown_texts)
    last = [e for e in service.audit.sink.read() if e["action"] == "ask"][-1]
    assert last["details"]["shown_chunk_ids"]


def test_empty_acl_is_rejected(service, admin_a):
    from fastrag.errors import InvalidInput

    with pytest.raises(InvalidInput):
        service.ingest(admin_a, doc("x", "some text here", acl=()))


# --- cache isolation -------------------------------------------------------------

Q = SECRET_PHRASE


def test_cache_not_shared_between_acl_scopes(service, generator, admin_a):
    insider = make_principal(user="ivy", groups=("finance",))
    outsider = make_principal(user="oscar", groups=("eng",))
    service.ingest(admin_a, doc("comp.txt", SECRET_TEXT, acl=[group_token("finance")]))

    first = service.ask(insider, Q)
    assert not first.cached and first.sources
    assert service.ask(insider, Q).cached  # cache works for the same scope

    ans = service.ask(outsider, Q)
    assert not ans.cached and ans.abstained and ans.sources == []
    assert SECRET not in ans.text


def test_cache_not_shared_across_tenants(service, generator, admin_a, admin_b):
    service.ingest(admin_a, doc("d", f"Shared question topic zebra {CANARY_A} details."))
    service.ingest(admin_b, doc("d", f"Shared question topic zebra {CANARY_B} details."))
    q = "shared question topic zebra"
    a = service.ask(admin_a, q)
    b = service.ask(admin_b, q)
    assert not b.cached
    assert all(CANARY_A not in s.chunk.text for s in b.sources)
    assert all(CANARY_B not in s.chunk.text for s in a.sources)
    assert any(CANARY_B in s.chunk.text for s in b.sources)


def test_acl_change_invalidates_cached_answer(service, generator, admin_a):
    insider = make_principal(user="ivy", groups=("finance",))
    service.ingest(admin_a, doc("comp.txt", SECRET_TEXT, acl=[group_token("finance")]))
    assert service.ask(insider, Q).sources
    assert service.ask(insider, Q).cached

    res = service.ingest(admin_a, doc("comp.txt", SECRET_TEXT, acl=[group_token("legal")]))
    assert res.status == "updated"

    ans = service.ask(insider, Q)
    assert not ans.cached
    assert ans.abstained and ans.sources == []
    assert SECRET not in ans.text


def test_reader_role_cannot_see_admin_only_scope_via_cache(service, admin_a):
    """Different users with identical groups must not share cache entries for user ACLs."""
    service.ingest(admin_a, doc("p", SECRET_TEXT, acl=[user_token("ivy")]))
    ivy = make_principal(user="ivy")
    other = make_principal(user="mallory")
    assert service.ask(ivy, Q).sources
    ans = service.ask(other, Q)
    assert not ans.cached and ans.sources == []
    assert Role.READER in other.roles
