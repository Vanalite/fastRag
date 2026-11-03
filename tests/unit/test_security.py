import time

import jwt
import pytest

from fastrag.domain import EVERYONE, Principal, Role, group_token, user_token
from fastrag.errors import AuthenticationError, ConfigurationError, PermissionDenied
from fastrag.security.acl import AccessFilter
from fastrag.security.auth import ApiKeyStore, Authenticator, JwtVerifier, hash_api_key
from fastrag.security.rbac import Permission, permissions_for, require

SECRET = "x" * 40
ALICE = Principal("acme", "alice", frozenset({"finance"}), frozenset({Role.READER}))


def test_access_filter_requires_tenant_and_token_overlap():
    access = AccessFilter.for_principal(ALICE)
    assert access.allows("acme", {EVERYONE})
    assert access.allows("acme", {group_token("finance")})
    assert access.allows("acme", {user_token("alice")})
    assert not access.allows("acme", {group_token("hr")})
    assert not access.allows("acme", set())  # deny by default
    assert not access.allows("globex", {EVERYONE})


def test_scope_key_differs_by_groups_and_tenant():
    base = AccessFilter.for_principal(ALICE).scope_key()
    other_group = Principal("acme", "alice", frozenset({"hr"}))
    other_tenant = Principal("globex", "alice", frozenset({"finance"}))
    assert AccessFilter.for_principal(other_group).scope_key() != base
    assert AccessFilter.for_principal(other_tenant).scope_key() != base


def test_rbac():
    reader, ingester = ALICE, Principal("acme", "ing", roles=frozenset({Role.INGESTER}))
    admin = Principal("acme", "root", roles=frozenset({Role.ADMIN}))
    require(reader, Permission.QUERY)
    with pytest.raises(PermissionDenied):
        require(reader, Permission.INGEST)
    require(ingester, Permission.DELETE)
    with pytest.raises(PermissionDenied):
        require(ingester, Permission.READ_AUDIT)
    assert permissions_for(admin) == frozenset(Permission)


def test_api_key_store_hashes_and_persists(tmp_path):
    path = tmp_path / "keys.json"
    key = ApiKeyStore(path).create(ALICE)
    assert key.startswith("frk_")
    raw = path.read_text()
    assert key not in raw and hash_api_key(key) in raw
    assert (path.stat().st_mode & 0o777) == 0o600
    assert ApiKeyStore(path).lookup(key) == ALICE
    assert ApiKeyStore(path).lookup(key + "x") is None


def test_jwt_roundtrip_and_rejections():
    verifier = JwtVerifier(SECRET, issuer="idp", audience="fastrag")
    assert verifier.verify(verifier.issue(ALICE)) == ALICE

    expired = verifier.issue(ALICE, ttl_seconds=10, now=time.time() - 100)
    wrong_secret = JwtVerifier("y" * 40, issuer="idp", audience="fastrag").issue(ALICE)
    no_exp = jwt.encode({"sub": "a", "tenant": "acme", "roles": ["reader"]}, SECRET)
    none_alg = jwt.encode(
        {"sub": "a", "tenant": "acme", "roles": ["admin"], "exp": time.time() + 60},
        key=None,
        algorithm="none",
    )
    bad_role = jwt.encode(
        {"sub": "a", "tenant": "acme", "roles": ["god"], "exp": time.time() + 60,
         "iss": "idp", "aud": "fastrag"},
        SECRET,
    )  # fmt: skip
    for token in (expired, wrong_secret, no_exp, none_alg, bad_role, "garbage"):
        with pytest.raises(AuthenticationError):
            verifier.verify(token)


def test_jwt_secret_must_be_long():
    with pytest.raises(ConfigurationError):
        JwtVerifier("short")


def test_authenticator_dispatch():
    keys = ApiKeyStore()
    key = keys.create(ALICE)
    verifier = JwtVerifier(SECRET)
    auth = Authenticator(api_keys=keys, jwt_verifier=verifier)
    assert auth.authenticate(None, key) == ALICE
    assert auth.authenticate(f"Bearer {verifier.issue(ALICE)}", None) == ALICE
    for header, api_key in ((None, None), ("Basic abc", None), (None, "frk_nope")):
        with pytest.raises(AuthenticationError):
            auth.authenticate(header, api_key)
    with pytest.raises(ConfigurationError):
        Authenticator()
