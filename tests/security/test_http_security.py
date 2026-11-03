"""AuthN/AuthZ over HTTP, and role/ACL write protection."""

from __future__ import annotations

import base64
import json

import pytest
from conftest import doc, make_principal
from fastapi.testclient import TestClient

from fastrag.api import create_app
from fastrag.domain import DocumentIn, Role, group_token
from fastrag.errors import NotFound, PermissionDenied
from fastrag.ratelimit import RateLimiter
from fastrag.security.auth import ApiKeyStore, Authenticator, JwtVerifier

SECRET = "s" * 40


def _b64(obj: dict) -> str:
    return base64.urlsafe_b64encode(json.dumps(obj).encode()).rstrip(b"=").decode()


@pytest.fixture
def env(service):
    keys = ApiKeyStore(None)
    jwt_v = JwtVerifier(SECRET)
    client = TestClient(create_app(service, Authenticator(api_keys=keys, jwt_verifier=jwt_v)))
    return service, client, keys, jwt_v


def _h(key: str) -> dict[str, str]:
    return {"X-API-Key": key}


BODY = {"source_id": "a.txt", "text": "Some text about refunds.", "acl": ["tenant:*"]}


def _last(service, action):
    return [e for e in service.audit.sink.read() if e["action"] == action][-1]


def _bad_tokens(jwt_v: JwtVerifier) -> dict[str, str]:
    import jwt as pyjwt

    p = make_principal(roles=(Role.ADMIN,))
    good = jwt_v.issue(p, 3600, now=1_000_000_000 - 7200)  # long expired
    valid = jwt_v.issue(p, 3600)
    h, pl, sig = valid.split(".")
    claims = json.loads(base64.urlsafe_b64decode(pl + "=" * (-len(pl) % 4)))
    claims["tenant"] = "tenant-b"
    tampered = f"{h}.{_b64(claims)}.{sig}"
    wrong = pyjwt.encode(claims, "x" * 40, algorithm="HS256") if False else None
    other = JwtVerifier("z" * 40).issue(p, 3600)
    none = (
        f"{_b64({'alg': 'none', 'typ': 'JWT'})}.{_b64({**claims, 'exp': 9999999999, 'sub': 'x'})}."
    )
    del wrong
    return {"expired": good, "tampered": tampered, "wrong_secret": other, "alg_none": none}


@pytest.mark.parametrize("kind", ["expired", "tampered", "wrong_secret", "alg_none"])
def test_bad_jwts_rejected(env, kind):
    service, client, _, jwt_v = env
    token = _bad_tokens(jwt_v)[kind]
    r = client.get("/v1/documents", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 401
    assert _last(service, "authenticate")["outcome"] == "unauthenticated"


def test_valid_jwt_accepted(env):
    _, client, _, jwt_v = env
    token = jwt_v.issue(make_principal(roles=(Role.READER,)))
    assert client.get("/v1/me", headers={"Authorization": f"Bearer {token}"}).status_code == 200


def test_no_or_invalid_credentials_401_without_side_effects(env):
    service, client, keys, _ = env
    admin = keys.create(make_principal(roles=(Role.ADMIN,)))
    assert client.post("/v1/documents", json=BODY, headers=_h(admin)).status_code == 201
    count = len(service.store.list_documents("tenant-a"))
    before = len(service.audit.sink.read())

    attempts = [
        client.post("/v1/documents", json=BODY | {"source_id": "b"}),
        client.post("/v1/documents", json=BODY | {"source_id": "b"}, headers=_h("frk_bogus")),
        client.post("/v1/search", json={"question": "refunds"}, headers=_h("frk_bogus")),
        client.get("/v1/audit", headers={"Authorization": "Basic abc"}),
        client.delete("/v1/documents/whatever"),
    ]
    assert [r.status_code for r in attempts] == [401] * 5
    new = service.audit.sink.read()[before:]
    assert len(service.store.list_documents("tenant-a")) == count
    assert len(new) == 5
    assert all(e["action"] == "authenticate" and e["outcome"] == "unauthenticated" for e in new)
    assert all(e["tenant_id"] is None for e in new)
    assert service.audit.verify()
    # Error body does not explain why the credential failed.
    assert "invalid API key" not in attempts[1].text


def test_reader_forbidden_to_write_or_read_audit(env):
    service, client, keys, _ = env
    admin = keys.create(make_principal(roles=(Role.ADMIN,)))
    reader = keys.create(make_principal(user="rita", roles=(Role.READER,)))
    doc_id = client.post("/v1/documents", json=BODY, headers=_h(admin)).json()["doc_id"]

    r = client.post("/v1/documents", json=BODY | {"source_id": "evil"}, headers=_h(reader))
    assert r.status_code == 403
    denied = _last(service, "ingest")
    assert denied["outcome"] == "denied" and denied["user_id"] == "rita"

    assert client.delete(f"/v1/documents/{doc_id}", headers=_h(reader)).status_code == 403
    assert _last(service, "delete")["outcome"] == "denied"
    assert client.get("/v1/audit", headers=_h(reader)).status_code == 403
    assert _last(service, "read_audit")["outcome"] == "denied"

    docs = client.get("/v1/documents", headers=_h(admin)).json()
    assert [d["doc_id"] for d in docs] == [doc_id]
    # Reader can query.
    assert (
        client.post("/v1/search", json={"question": "refunds"}, headers=_h(reader)).status_code
        == 200
    )


def test_ingester_cannot_read_audit_but_admin_can(env):
    _, client, keys, _ = env
    ing = keys.create(make_principal(roles=(Role.INGESTER,)))
    admin = keys.create(make_principal(roles=(Role.ADMIN,)))
    assert client.get("/v1/audit", headers=_h(ing)).status_code == 403
    assert client.get("/v1/audit", headers=_h(admin)).status_code == 200


def test_audit_endpoint_is_tenant_scoped(env):
    _, client, keys, _ = env
    a = keys.create(make_principal("tenant-a", "a", roles=(Role.ADMIN,)))
    b = keys.create(make_principal("tenant-b", "b", roles=(Role.ADMIN,)))
    client.post("/v1/documents", json=BODY, headers=_h(a))
    events = client.get("/v1/audit", headers=_h(b)).json()["events"]
    assert all(e["tenant_id"] == "tenant-b" for e in events)


def test_rate_limit_returns_429_with_retry_after(make_service):
    svc = make_service(rate_limit_per_minute=1, rate_limit_burst=2)
    keys = ApiKeyStore(None)
    client = TestClient(create_app(svc, Authenticator(api_keys=keys)))
    a = keys.create(make_principal("tenant-a", "a"))
    b = keys.create(make_principal("tenant-b", "b"))
    q = {"question": "refunds"}
    assert client.post("/v1/search", json=q, headers=_h(a)).status_code == 200
    assert client.post("/v1/search", json=q, headers=_h(a)).status_code == 200
    r = client.post("/v1/search", json=q, headers=_h(a))
    assert r.status_code == 429
    assert int(r.headers["Retry-After"]) >= 1
    assert client.post("/v1/search", json=q, headers=_h(b)).status_code == 200


def test_http_cross_tenant_delete_is_404(env):
    _, client, keys, _ = env
    a = keys.create(make_principal("tenant-a", "a", roles=(Role.ADMIN,)))
    b = keys.create(make_principal("tenant-b", "b", roles=(Role.ADMIN,)))
    doc_id = client.post("/v1/documents", json=BODY, headers=_h(b)).json()["doc_id"]
    assert client.delete(f"/v1/documents/{doc_id}", headers=_h(a)).status_code == 404
    assert client.delete(f"/v1/documents/{doc_id}", headers=_h(b)).status_code == 204


# --- ingester vs. documents they cannot read ---------------------------------------


def test_ingester_cannot_overwrite_or_delete_unreadable_doc(service, admin_a):
    ing = make_principal(user="ingrid", roles=(Role.INGESTER,))
    res = service.ingest(
        admin_a, doc("hr.txt", "Confidential HR salary table.", acl=[group_token("hr")])
    )
    with pytest.raises(PermissionDenied):
        service.ingest(ing, doc("hr.txt", "Overwritten by ingrid.", acl=["tenant:*"]))
    with pytest.raises(NotFound):
        service.delete(ing, res.doc_id)
    # Original intact.
    recs = service.store.list_documents("tenant-a")
    assert recs[0].version == 1 and recs[0].acl == frozenset({group_token("hr")})
    assert any("salary" in c.text for c, _ in service.store.iter_chunks())
    assert service.audit.verify()


def test_reading_does_not_grant_writing(service, admin_a):
    # Ingrid can read the HR document, but she isn't one of its editors.
    ing = make_principal(user="ingrid", roles=(Role.INGESTER,), groups=("hr",))
    res = service.ingest(admin_a, doc("hr.txt", "HR table one.", acl=[group_token("hr")]))
    assert service.search(ing, "HR table").chunks
    with pytest.raises(PermissionDenied):
        service.ingest(ing, doc("hr.txt", "HR table two.", acl=[group_token("hr")]))
    with pytest.raises(NotFound):
        service.delete(ing, res.doc_id)
    assert service.store.get_document("tenant-a", res.doc_id).version == 1


def test_listed_editor_can_overwrite_and_delete(service, admin_a):
    ing = make_principal(user="ingrid", roles=(Role.INGESTER,), groups=("hr",))
    first = DocumentIn(
        "hr.txt", "HR table one.", frozenset({group_token("hr")}),
        editors=frozenset({group_token("hr")}),
    )  # fmt: skip
    res = service.ingest(admin_a, first)
    assert service.ingest(ing, doc("hr.txt", "HR table two.", acl=[group_token("hr")])).version == 2
    # Editors carry over when an update doesn't name them.
    assert service.store.get_document("tenant-a", res.doc_id).editors == {group_token("hr")}
    service.delete(ing, res.doc_id)
    assert service.store.list_documents("tenant-a") == []


def test_uploader_is_default_editor(service):
    ing = make_principal(user="ingrid", roles=(Role.INGESTER,))
    other = make_principal(user="olaf", roles=(Role.INGESTER,))
    res = service.ingest(ing, doc("notes.txt", "Team notes v1."))
    assert service.store.get_document("tenant-a", res.doc_id).editors == {"user:ingrid"}
    assert service.ingest(ing, doc("notes.txt", "Team notes v2.")).version == 2
    with pytest.raises(PermissionDenied):
        service.ingest(other, doc("notes.txt", "Team notes hijacked."))


def test_failed_logins_are_throttled_before_credentials_are_checked(make_service):
    clock = [0.0]
    limiter = RateLimiter(rate_per_minute=60, burst=3, clock=lambda: clock[0])
    svc = make_service()
    keys = ApiKeyStore()
    good = keys.create(make_principal(user="ok"))
    client = TestClient(create_app(svc, Authenticator(api_keys=keys), auth_failure_limiter=limiter))

    def me(key):
        return client.get("/v1/me", headers={"X-API-Key": key}).status_code

    assert [me("frk_wrong") for _ in range(6)] == [401, 401, 401, 429, 429, 429]
    entries = [e for e in svc.audit.sink.read() if e["action"] == "authenticate"]
    # Three failures, then one aggregated entry for the throttled burst - not one per try.
    assert [e["outcome"] for e in entries] == ["unauthenticated"] * 3 + ["throttled"]
    assert entries[-1]["details"]["attempts"] == 1
    # While throttled, even a valid key waits: the address can't keep guessing.
    assert me(good) == 429
    clock[0] += 2.0  # a token refills
    assert me(good) == 200  # and success spends nothing
    assert me(good) == 200


def test_editors_cannot_change_who_may_read_or_edit(service, admin_a):
    ing = make_principal(user="ingrid", roles=(Role.INGESTER,), groups=("hr",))
    hr_only = frozenset({group_token("hr")})
    res = service.ingest(admin_a, DocumentIn("hr.txt", "HR table one.", hr_only, editors=hr_only))
    # Content updates are fine...
    service.ingest(ing, DocumentIn("hr.txt", "HR table two.", hr_only))
    # ...but publishing to the whole tenant, or granting edit rights, needs an admin.
    with pytest.raises(PermissionDenied):
        service.ingest(ing, DocumentIn("hr.txt", "HR table two.", frozenset({"tenant:*"})))
    with pytest.raises(PermissionDenied):
        service.ingest(
            ing, DocumentIn("hr.txt", "HR v3.", hr_only, editors=frozenset({"tenant:*"}))
        )
    record = service.store.get_document("tenant-a", res.doc_id)
    assert record.acl == hr_only and record.editors == hr_only and record.version == 2
    # An admin can.
    service.ingest(admin_a, DocumentIn("hr.txt", "HR table two.", frozenset({"tenant:*"})))
    assert service.store.get_document("tenant-a", res.doc_id).acl == {"tenant:*"}
