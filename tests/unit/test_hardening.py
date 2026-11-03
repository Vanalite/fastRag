"""Tests for production hardening: audit concurrency, key lifecycle, OIDC, production
config, schema migration, redaction of metadata, chunk structure, dedup, grounding."""

import json
import sqlite3
import threading
import time
from types import SimpleNamespace

import jwt
import numpy as np
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient
from pydantic import ValidationError

from fastrag import DocumentIn, Principal, RAGService, Role, Settings
from fastrag.api import create_app
from fastrag.domain import EVERYONE
from fastrag.errors import AuthenticationError, ConfigurationError
from fastrag.generation import ClaudeGenerator, Generation
from fastrag.ingest.chunking import chunk_text
from fastrag.observability import AuditLog, JsonlAuditSink, MemoryAuditSink, verify_chain
from fastrag.security.auth import ApiKeyStore, Authenticator, JwtVerifier, OidcVerifier
from fastrag.storage import SQLiteDocumentStore

ADMIN = Principal("acme", "root", roles=frozenset({Role.ADMIN}))
ALICE = Principal("acme", "alice", frozenset({"finance"}))


class Clock:
    def __init__(self, now: float = 1_000_000.0):
        self.now = now

    def __call__(self) -> float:
        return self.now


# --- audit log ---------------------------------------------------------------------


def test_truncated_memory_audit_still_verifies():
    log = AuditLog(MemoryAuditSink(max_entries=3))
    for i in range(10):
        log.record("ask", "success", ALICE, i=i)
    assert len(log.sink.read()) == 3
    assert log.verify()  # anchored on the oldest retained entry, not genesis


def test_jsonl_audit_handles_long_lines_and_threads(tmp_path):
    path = tmp_path / "audit.jsonl"
    log = AuditLog(JsonlAuditSink(path))
    log.record("ingest", "success", ADMIN, pad="x" * 20_000)  # longer than the read window

    def worker():
        own = AuditLog(JsonlAuditSink(path))  # separate sink objects, shared file
        for _ in range(25):
            own.record("ask", "success", ALICE)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    entries = JsonlAuditSink(path).read()
    assert len(entries) == 101
    assert verify_chain(entries)


# --- API key lifecycle ---------------------------------------------------------------


def test_api_key_expiry_and_revocation(tmp_path):
    clock = Clock()
    store = ApiKeyStore(tmp_path / "keys.json", clock=clock)
    short = store.create(ALICE, ttl_seconds=60)
    long_lived = store.create(ADMIN)
    assert store.lookup(short) == ALICE
    clock.now += 61
    assert store.lookup(short) is None  # expired
    record = next(r for r in store.records() if r.principal == ADMIN)
    store.revoke(record.key_id[:8])
    assert store.lookup(long_lived) is None


def test_running_store_sees_changes_made_by_another_process(tmp_path):
    path = tmp_path / "keys.json"
    server_view = ApiKeyStore(path)
    key = ApiKeyStore(path).create(ALICE)  # e.g. `fastrag keys create` in another process
    assert server_view.lookup(key) == ALICE
    cli_view = ApiKeyStore(path)
    cli_view.revoke(next(iter(cli_view.records())).key_id)
    assert server_view.lookup(key) is None


def test_revoke_requires_a_unique_prefix(tmp_path):
    store = ApiKeyStore(tmp_path / "keys.json")
    store.create(ALICE)
    with pytest.raises(ValueError):
        store.revoke("abc")  # too short to be safe
    with pytest.raises(ValueError):
        store.revoke("zzzzzzzz")  # no match


def test_concurrent_key_creation_loses_nothing(tmp_path):
    path = tmp_path / "keys.json"
    keys: list[str] = []

    def create():
        keys.append(ApiKeyStore(path).create(ALICE))

    threads = [threading.Thread(target=create) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    store = ApiKeyStore(path)
    assert all(store.lookup(k) == ALICE for k in keys)


# --- OIDC and token routing ------------------------------------------------------------


@pytest.fixture(scope="module")
def rsa_key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


class FakeJwks:
    def __init__(self, public_key):
        self.public_key = public_key

    def get_signing_key_from_jwt(self, token):
        return SimpleNamespace(key=self.public_key)


def _idp_token(key, **overrides):
    now = int(time.time())
    claims = {
        "sub": "alice",
        "tenant": "acme",
        "groups": ["finance"],
        "roles": ["reader", "some-other-app-role"],
        "iss": "https://idp.example.com",
        "aud": "fastrag",
        "iat": now,
        "exp": now + 300,
        **overrides,
    }
    return jwt.encode(claims, key, algorithm="RS256", headers={"kid": "k1"})


def _oidc(rsa_key):
    return OidcVerifier(
        "https://idp.example.com/.well-known/jwks.json",
        issuer="https://idp.example.com",
        audience="fastrag",
        jwk_client=FakeJwks(rsa_key.public_key()),
    )


def test_oidc_accepts_valid_tokens_and_ignores_foreign_roles(rsa_key):
    principal = _oidc(rsa_key).verify(_idp_token(rsa_key))
    assert principal == ALICE  # unknown role dropped, reader kept


@pytest.mark.parametrize(
    "overrides",
    [
        {"aud": "another-app"},
        {"iss": "https://evil.example.com"},
        {"exp": int(time.time()) - 600},
        {"roles": ["some-other-app-role"]},
        {"tenant": ""},
    ],
)
def test_oidc_rejects_bad_claims(rsa_key, overrides):
    with pytest.raises(AuthenticationError):
        _oidc(rsa_key).verify(_idp_token(rsa_key, **overrides))


def test_oidc_rejects_tokens_signed_by_another_key(rsa_key):
    other = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    with pytest.raises(AuthenticationError):
        _oidc(rsa_key).verify(_idp_token(other))


def test_oidc_requires_issuer_audience_and_https():
    with pytest.raises(ConfigurationError):
        OidcVerifier("https://idp/jwks", issuer="", audience="fastrag")
    with pytest.raises(ConfigurationError):
        OidcVerifier("http://idp/jwks", issuer="https://idp", audience="fastrag")


def test_authenticator_routes_by_algorithm(rsa_key):
    shared = JwtVerifier("s" * 40)
    auth = Authenticator(jwt_verifier=shared, oidc_verifier=_oidc(rsa_key))
    assert auth.authenticate(f"Bearer {_idp_token(rsa_key)}", None) == ALICE
    assert auth.authenticate(f"Bearer {shared.issue(ALICE)}", None) == ALICE
    # alg=none matches no verifier, and garbage is rejected before any verification.
    none_token = jwt.encode({"sub": "x", "tenant": "acme", "roles": ["admin"]}, None, "none")
    for token in (none_token, "not.a.jwt"):
        with pytest.raises(AuthenticationError):
            auth.authenticate(f"Bearer {token}", None)
    only_shared = Authenticator(jwt_verifier=shared)
    with pytest.raises(AuthenticationError):
        only_shared.authenticate(f"Bearer {_idp_token(rsa_key)}", None)


# --- production configuration --------------------------------------------------------


def test_production_rejects_unsafe_settings():
    with pytest.raises(ValidationError) as exc:
        Settings(
            _env_file=None,
            environment="production",
            database_path=":memory:",
            jwt_secret="SECRET-VALUE-" * 4,
            log_json=False,
            rate_limit_per_minute=0,
        )
    message = str(exc.value)
    for expected in ("database_path", "jwt_issuer", "log_json", "rate limits"):
        assert expected in message
    assert "SECRET-VALUE" not in message  # raw input never echoed


def test_production_accepts_a_safe_configuration():
    settings = Settings(
        _env_file=None,
        environment="production",
        jwt_secret="s" * 40,
        jwt_issuer="https://idp",
        jwt_audience="fastrag",
    )
    assert settings.environment == "production"


def test_oidc_settings_need_issuer_and_audience():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, oidc_jwks_url="https://idp/jwks")


# --- schema migration ----------------------------------------------------------------


def test_v1_database_is_migrated_in_place(tmp_path):
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE documents (
            tenant_id TEXT NOT NULL, doc_id TEXT NOT NULL, source_id TEXT NOT NULL,
            title TEXT NOT NULL, version INTEGER NOT NULL, content_hash TEXT NOT NULL,
            acl TEXT NOT NULL, metadata TEXT NOT NULL, chunk_count INTEGER NOT NULL,
            updated_at TEXT NOT NULL, PRIMARY KEY (tenant_id, doc_id));
        INSERT INTO documents VALUES
            ('acme', 'd1', 'a.md', 'A', 1, 'h', '["tenant:*"]', '{}', 0, '2026-01-01');
        """
    )
    conn.commit()
    conn.close()
    store = SQLiteDocumentStore(path)
    record = store.get_document("acme", "d1")
    assert record is not None and record.editors == frozenset()  # admin-only until set
    version = store._conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
    assert version["value"] == str(SQLiteDocumentStore.SCHEMA_VERSION)


# --- ingestion -------------------------------------------------------------------------


def _service(**overrides) -> RAGService:
    defaults = {"rate_limit_per_minute": 0, "user_rate_limit_per_minute": 0}
    return RAGService.from_settings(Settings.for_testing(**{**defaults, **overrides}))


def test_metadata_strings_are_redacted():
    service = _service()
    service.ingest(
        ADMIN,
        DocumentIn(
            "contact.md",
            "Escalation contacts for the on-call rotation.",
            frozenset({EVERYONE}),
            metadata={"owner": "jane.doe@acme.com", "tags": ["call 555-123-4567"], "n": 3},
        ),
    )
    (record,) = service.store.list_documents("acme")
    assert record.metadata == {
        "owner": "[REDACTED_EMAIL]",
        "tags": ["call [REDACTED_PHONE]"],
        "n": 3,
    }


def test_chunks_keep_paragraph_and_line_structure():
    text = "Steps:\n- unplug the laptop\n- call security\n\nSecond paragraph here."
    (chunk,) = chunk_text(text, chunk_size=500, overlap=50)
    assert chunk == text


def test_overlap_carries_whole_sentences_across_paragraph_sized_units():
    para = " ".join(f"Sentence {i} talks about topic {i}." for i in range(40))
    chunks = chunk_text(para, chunk_size=300, overlap=80)
    assert len(chunks) > 1
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        last_sentence = prev.rsplit(". ", 1)[-1]
        assert last_sentence in nxt


# --- retrieval and answers -------------------------------------------------------------


def test_duplicate_documents_do_not_crowd_out_other_evidence():
    service = _service()
    body = "Visitors park in garage B on level two, free for three hours."
    for i in range(3):
        service.ingest(ADMIN, DocumentIn(f"copy-{i}.md", body, frozenset({EVERYONE})))
    service.ingest(
        ADMIN, DocumentIn("rates.md", "Garage B charges after three hours.", frozenset({EVERYONE}))
    )
    sources = [s.chunk.source_id for s in service.search(ALICE, "garage B parking", k=4).chunks]
    assert sum(s.startswith("copy-") for s in sources) == 1
    assert "rates.md" in sources


class NoCitationGenerator:
    name = "fake"

    def generate(self, question, sources):
        return Generation(text="Probably three hours, I think.")


def test_answers_without_citations_are_flagged_ungrounded():
    service = RAGService.from_settings(
        Settings.for_testing(rate_limit_per_minute=0, user_rate_limit_per_minute=0),
        generator=NoCitationGenerator(),
    )
    service.ingest(
        ADMIN, DocumentIn("p.md", "Visitor parking is free for three hours.", frozenset({EVERYONE}))
    )
    answer = service.ask(ALICE, "How long is visitor parking free?")
    assert not answer.abstained and not answer.grounded
    assert service.audit.sink.read()[-1]["details"]["ungrounded"] is True
    assert service.metrics.ungrounded.value({"tenant": "acme"}) == 1


def test_missing_claude_credentials_is_a_configuration_error():
    def no_credentials(**kwargs):
        raise TypeError('"Could not resolve authentication method. Expected one of api_key"')

    client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=no_credentials)))
    with pytest.raises(ConfigurationError):
        ClaudeGenerator(client=client).generate("q", [])


def test_document_listing_is_paginated():
    service = _service()
    for i in range(5):
        service.ingest(
            ADMIN, DocumentIn(f"d{i}.md", f"Document number {i}.", frozenset({EVERYONE}))
        )
    keys = ApiKeyStore()
    key = keys.create(ADMIN)
    client = TestClient(create_app(service, Authenticator(api_keys=keys)))
    resp = client.get("/v1/documents?limit=2&offset=1", headers={"X-API-Key": key})
    assert resp.status_code == 200
    assert resp.headers["X-Total-Count"] == "5"
    assert [d["source_id"] for d in resp.json()] == ["d1.md", "d2.md"]
    assert json.loads(resp.text)[0]["editors"] == ["user:root"]


# --- regression tests for review findings --------------------------------------------


def test_torn_audit_tail_is_repaired_but_corruption_stops_writes(tmp_path):
    path = tmp_path / "audit.jsonl"
    log = AuditLog(JsonlAuditSink(path))
    log.record("ask", "success", ALICE)
    with path.open("a") as f:
        f.write('{"ts": "2026-11-03", "hash": "ab')  # crash mid-write: no newline
    log.record("ask", "success", ALICE)
    entries = JsonlAuditSink(path).read()
    assert len(entries) == 2 and verify_chain(entries)

    with path.open("a") as f:
        f.write("this is not json\n")  # a complete but corrupt line
    with pytest.raises(ConfigurationError):
        log.record("ask", "success", ALICE)


def _create_key_in_subprocess(path_str: str, queue) -> None:
    from pathlib import Path as _Path

    queue.put(ApiKeyStore(_Path(path_str)).create(ALICE))


def test_key_creation_across_processes_loses_nothing(tmp_path):
    import multiprocessing

    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    path = tmp_path / "keys.json"
    procs = [
        ctx.Process(target=_create_key_in_subprocess, args=(str(path), queue)) for _ in range(4)
    ]
    for p in procs:
        p.start()
    keys = [queue.get(timeout=60) for _ in procs]
    for p in procs:
        p.join(timeout=60)
    store = ApiKeyStore(path)
    assert len(store.records()) == 4
    assert all(store.lookup(k) == ALICE for k in keys)


def test_key_ttl_must_be_positive_and_revoke_is_idempotent(tmp_path):
    clock = Clock()
    store = ApiKeyStore(tmp_path / "keys.json", clock=clock)
    for bad in (0, -5):
        with pytest.raises(ValueError):
            store.create(ALICE, ttl_seconds=bad)
    store.create(ALICE)
    key_id = store.records()[0].key_id
    first = store.revoke(key_id)
    clock.now += 100
    assert store.revoke(key_id).revoked_at == first.revoked_at


def test_hand_edited_key_file_with_bad_types_fails_closed(tmp_path):
    path = tmp_path / "keys.json"
    ApiKeyStore(path).create(ALICE)
    records = json.loads(path.read_text())
    records[0]["expires_at"] = "tomorrow"
    path.write_text(json.dumps(records))
    with pytest.raises(ConfigurationError):
        ApiKeyStore(path)


@pytest.mark.parametrize(
    "claims",
    [
        {"sub": "", "tenant": "acme", "roles": ["reader"]},
        {"sub": "a", "tenant": "acme", "roles": [["reader"]]},
        {"sub": "a", "tenant": "acme", "roles": [{"r": 1}, "nope"]},
        {"sub": 42, "tenant": "acme", "roles": ["reader"]},
    ],
)
def test_malformed_claims_are_401_not_500(claims):
    verifier = JwtVerifier("s" * 40)
    now = int(time.time())
    token = jwt.encode({**claims, "iat": now, "exp": now + 60}, "s" * 40, algorithm="HS256")
    with pytest.raises(AuthenticationError):
        verifier.verify(token)


def test_long_lived_and_iat_less_tokens_are_rejected(rsa_key):
    verifier = JwtVerifier("s" * 40)
    now = int(time.time())
    base = {"sub": "a", "tenant": "acme", "roles": ["reader"]}
    week = jwt.encode({**base, "iat": now, "exp": now + 7 * 86400}, "s" * 40, "HS256")
    no_iat = jwt.encode({**base, "exp": now + 60}, "s" * 40, "HS256")
    for token in (week, no_iat):
        with pytest.raises(AuthenticationError):
            verifier.verify(token)
    with pytest.raises(AuthenticationError):
        _oidc(rsa_key).verify(_idp_token(rsa_key, exp=now + 2 * 86400))


def test_metadata_keys_are_redacted_and_depth_is_bounded():
    from fastrag.errors import InvalidInput

    service = _service()
    service.ingest(
        ADMIN,
        DocumentIn(
            "k.md",
            "Contact sheet.",
            frozenset({EVERYONE}),
            metadata={"bob@acme.com": "owner", "eve@acme.com": "backup"},
        ),
    )
    (record,) = service.store.list_documents("acme")
    assert set(record.metadata) == {"[REDACTED_EMAIL]", "[REDACTED_EMAIL]_"}

    deep: dict = {}
    node = deep
    for _ in range(20):
        node["x"] = {}
        node = node["x"]
    with pytest.raises(InvalidInput):
        service.ingest(ADMIN, DocumentIn("d.md", "Deep.", frozenset({EVERYONE}), metadata=deep))


def test_near_copies_with_different_facts_are_both_kept():
    service = _service()
    real = "Refunds are processed within 30 days of the request being approved by finance."
    fake = "Refunds are processed within 90 days of the request being approved by finance."
    service.ingest(ADMIN, DocumentIn("policy.md", real, frozenset({EVERYONE})))
    service.ingest(ADMIN, DocumentIn("copy.md", fake, frozenset({EVERYONE})))
    sources = {s.chunk.source_id for s in service.search(ALICE, "refunds processed days").chunks}
    assert sources == {"policy.md", "copy.md"}


def test_newer_schema_is_refused(tmp_path):
    path = tmp_path / "db.sqlite"
    SQLiteDocumentStore(path).close()
    conn = sqlite3.connect(path)
    conn.execute("UPDATE meta SET value = '99' WHERE key = 'schema_version'")
    conn.commit()
    conn.close()
    with pytest.raises(ConfigurationError):
        SQLiteDocumentStore(path)


def test_rate_limiter_memory_is_bounded_and_peek_is_free():
    from fastrag.errors import RateLimited
    from fastrag.ratelimit import RateLimiter

    clock = Clock(0.0)
    limiter = RateLimiter(rate_per_minute=60, burst=1, clock=clock, max_keys=100)
    for i in range(1000):
        limiter.check(f"client-{i}")
    assert len(limiter) == 100
    limiter.ensure_available("fresh")
    limiter.ensure_available("fresh")  # peeking spends nothing
    limiter.check("fresh")
    with pytest.raises(RateLimited):
        limiter.ensure_available("fresh")


def test_admin_update_keeps_a_legacy_document_admin_only(tmp_path):
    service = _service()
    service.ingest(ADMIN, DocumentIn("legacy.md", "Old text.", frozenset({EVERYONE})))
    doc_id = service.store.list_documents("acme")[0].doc_id
    service.store._conn.execute("UPDATE documents SET editors = '[]' WHERE doc_id = ?", (doc_id,))
    service.ingest(ADMIN, DocumentIn("legacy.md", "New text.", frozenset({EVERYONE})))
    assert service.store.get_document("acme", doc_id).editors == frozenset()


def test_unrelated_type_errors_from_the_sdk_are_not_masked():
    def buggy(**kwargs):
        raise TypeError("create() got an unexpected keyword argument 'fallbackz'")

    client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=buggy)))
    with pytest.raises(TypeError):
        ClaudeGenerator(client=client).generate("q", [])


def test_empty_custom_cache_and_limiters_are_not_replaced():
    from fastrag.cache import AnswerCache
    from fastrag.ratelimit import RateLimiter

    cache, limiter = AnswerCache(ttl_seconds=1), RateLimiter(0, 0)
    service = RAGService.from_settings(
        Settings.for_testing(), cache=cache, limiter=limiter, user_limiter=limiter
    )
    assert service.cache is cache and service.limiter is limiter
    assert service.user_limiter is limiter


def test_cli_bulk_ingest_is_not_rate_limited(tmp_path, monkeypatch, capsys):
    from fastrag.cli import main

    for i in range(40):
        (tmp_path / f"doc{i}.md").write_text(f"# Doc {i}\n\nBody of document number {i}.")
    monkeypatch.setenv("FASTRAG_DATABASE_PATH", str(tmp_path / "db.sqlite"))
    monkeypatch.setenv("FASTRAG_AUDIT_LOG_PATH", str(tmp_path / "audit.jsonl"))
    monkeypatch.setenv("FASTRAG_API_KEYS_FILE", str(tmp_path / "keys.json"))
    assert main(["ingest", str(tmp_path), "--tenant", "acme"]) == 0
    assert '"created": 40' in capsys.readouterr().out


def test_cli_token_uses_configured_claim_names(monkeypatch, capsys):
    from fastrag.cli import main

    monkeypatch.setenv("FASTRAG_JWT_SECRET", "s" * 40)
    monkeypatch.setenv("FASTRAG_TOKEN_TENANT_CLAIM", "org")
    assert main(["token", "--tenant", "acme", "--user", "alice"]) == 0
    token = capsys.readouterr().out.strip()
    claims = jwt.decode(token, "s" * 40, algorithms=["HS256"])
    assert claims["org"] == "acme" and "tenant" not in claims


class ReverseReranker:
    """Puts the fused ranking upside down, and tries to sneak in a chunk of its own."""

    def __init__(self):
        self.seen = 0

    def rerank(self, query, results):
        from dataclasses import replace as _replace

        self.seen = len(results)
        out = [_replace(r, rerank_score=float(i)) for i, r in enumerate(results)]
        intruder = _replace(results[0], chunk=_replace(results[0].chunk, id="intruder"))
        return [*reversed(out), intruder]


def test_reranker_reorders_a_wider_pool_and_cannot_add_chunks():
    reranker = ReverseReranker()
    service = RAGService.from_settings(
        Settings.for_testing(rate_limit_per_minute=0, user_rate_limit_per_minute=0, rerank_pool=6),
        reranker=reranker,
    )
    for i in range(8):
        service.ingest(
            ADMIN,
            DocumentIn(f"g{i}.md", f"Garage note {i}: parking rule {i}.", frozenset({EVERYONE})),
        )
    plain = RAGService.from_settings(
        Settings.for_testing(rate_limit_per_minute=0, user_rate_limit_per_minute=0)
    )
    for i in range(8):
        plain.ingest(
            ADMIN,
            DocumentIn(f"g{i}.md", f"Garage note {i}: parking rule {i}.", frozenset({EVERYONE})),
        )
    reranked = service.search(ALICE, "garage parking rule", k=2).chunks
    baseline = plain.search(ALICE, "garage parking rule", k=6).chunks
    assert reranker.seen == 6  # reranked the wider pool, not just k
    assert [r.chunk.source_id for r in reranked] == [b.chunk.source_id for b in baseline[::-1][:2]]
    assert all(r.chunk.id != "intruder" for r in reranked)
    assert reranked[0].rerank_score is not None


def test_unknown_reranker_is_a_configuration_error():
    from fastrag.retrieval import get_reranker

    assert get_reranker(None) is None
    with pytest.raises(ConfigurationError):
        get_reranker("magic")


# --- optimistic concurrency -------------------------------------------------------------


def test_stale_writes_are_rejected_by_the_store():
    from fastrag.domain import DocumentRecord
    from fastrag.errors import ConcurrentModification

    store = SQLiteDocumentStore()
    record = DocumentRecord("acme", "d1", "d1.md", "", 1, "h1", frozenset({EVERYONE}), {}, 0, "")
    store.replace_document(record, [], np.zeros((0, 4), np.float32), expected_version=None)
    with pytest.raises(ConcurrentModification):  # someone else already created it
        store.replace_document(record, [], np.zeros((0, 4), np.float32), expected_version=None)
    v2 = DocumentRecord("acme", "d1", "d1.md", "", 2, "h2", frozenset({EVERYONE}), {}, 0, "")
    with pytest.raises(ConcurrentModification):
        store.replace_document(v2, [], np.zeros((0, 4), np.float32), expected_version=5)
    store.replace_document(v2, [], np.zeros((0, 4), np.float32), expected_version=1)
    with pytest.raises(ConcurrentModification):
        store.delete_document("acme", "d1", expected_version=1)
    assert store.delete_document("acme", "d1", expected_version=2) == []


def test_ingest_retries_after_a_concurrent_write(monkeypatch):
    from fastrag.errors import ConcurrentModification

    service = _service()
    service.ingest(ADMIN, DocumentIn("p.md", "Version one.", frozenset({EVERYONE})))
    real = service.store.replace_document
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConcurrentModification("simulated race")
        return real(*args, **kwargs)

    monkeypatch.setattr(service.store, "replace_document", flaky)
    result = service.ingest(ADMIN, DocumentIn("p.md", "Version two.", frozenset({EVERYONE})))
    assert calls["n"] == 2 and result.version == 2


def test_two_processes_writing_one_document_never_lose_a_version(tmp_path):
    """Two services on one database file stand in for two processes or replicas."""
    path = str(tmp_path / "shared.db")
    services = [_service(database_path=path) for _ in range(2)]
    errors: list[Exception] = []

    def writer(service, label):
        for i in range(15):
            try:
                service.ingest(
                    ADMIN, DocumentIn("shared.md", f"{label} edit {i}.", frozenset({EVERYONE}))
                )
            except Exception as exc:  # noqa: BLE001 - collected and asserted below
                errors.append(exc)

    threads = [
        threading.Thread(target=writer, args=(s, n)) for s, n in zip(services, "AB", strict=True)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    record = SQLiteDocumentStore(path).get_document(
        "acme", services[0].store.list_documents("acme")[0].doc_id
    )
    assert record.version == 30  # every write produced its own version


def test_answer_is_not_cached_if_the_corpus_changed_while_answering():
    class IngestingGenerator:
        """Deletes the source document while 'thinking', like a concurrent admin."""

        name = "racy"

        def __init__(self):
            self.service = None

        def generate(self, question, sources):
            self.service.delete(ADMIN, sources[0].chunk.doc_id)
            return Generation(text="Three hours [1].")

    generator = IngestingGenerator()
    service = RAGService.from_settings(
        Settings.for_testing(rate_limit_per_minute=0, user_rate_limit_per_minute=0),
        generator=generator,
    )
    generator.service = service
    service.ingest(
        ADMIN, DocumentIn("p.md", "Visitor parking is free for three hours.", frozenset({EVERYONE}))
    )
    service.ask(ALICE, "How long is visitor parking free?")
    assert len(service.cache) == 0  # the answer quoted a now-deleted document


def test_reranker_failure_falls_back_to_fused_order():
    class BrokenReranker:
        def rerank(self, query, results):
            raise RuntimeError("model server down")

    service = RAGService.from_settings(
        Settings.for_testing(rate_limit_per_minute=0, user_rate_limit_per_minute=0),
        reranker=BrokenReranker(),
    )
    service.ingest(
        ADMIN, DocumentIn("p.md", "Visitor parking is free for three hours.", frozenset({EVERYONE}))
    )
    assert service.search(ALICE, "visitor parking free").chunks


def test_tracing_failures_never_block_the_audit(monkeypatch):
    import fastrag.service as service_module

    def broken(*args, **kwargs):
        raise RuntimeError("exporter exploded")

    monkeypatch.setattr(service_module, "end_span", broken)
    service = _service()
    service.list_documents(ALICE)
    assert service.audit.sink.read()[-1]["action"] == "list_documents"


def test_pgvector_version_check():
    pytest.importorskip("psycopg")
    from fastrag.storage.postgres import _version_at_least

    assert _version_at_least("0.8.0", (0, 8))
    assert _version_at_least("0.10.1", (0, 8))
    assert not _version_at_least("0.7.4", (0, 8))
    assert not _version_at_least("0.5", (0, 8))
