"""Redis-backed rate limits and answer cache, shared by replicas (tested on fakeredis)."""

import fakeredis
import pytest
from fastapi.testclient import TestClient

from fastrag import DocumentIn, Principal, RAGService, Role, Settings
from fastrag.api import create_app
from fastrag.cache import RedisAnswerCache, answer_from_json, answer_to_json
from fastrag.domain import EVERYONE
from fastrag.errors import RateLimited
from fastrag.ratelimit import RedisRateLimiter
from fastrag.security.auth import ApiKeyStore, Authenticator

ADMIN = Principal("acme", "root", roles=frozenset({Role.ADMIN}))
ALICE = Principal("acme", "alice", frozenset({"finance"}))


@pytest.fixture
def redis():
    return fakeredis.FakeRedis()


def replica(redis, **overrides) -> RAGService:
    """A service instance sharing `redis` (and nothing else) with other replicas."""
    return RAGService.from_settings(Settings.for_testing(**overrides), redis=redis)


def test_rate_limit_budget_is_shared_across_replicas(redis):
    a = replica(redis, rate_limit_per_minute=1, rate_limit_burst=2, user_rate_limit_per_minute=0)
    b = replica(redis, rate_limit_per_minute=1, rate_limit_burst=2, user_rate_limit_per_minute=0)
    a.list_documents(ALICE)
    b.list_documents(ALICE)
    with pytest.raises(RateLimited):
        a.list_documents(ALICE)  # the tenant spent its burst across both replicas
    other_tenant = Principal("globex", "g")
    b.list_documents(other_tenant)


def test_limiter_keys_are_opaque_and_expire(redis):
    limiter = RedisRateLimiter(redis, rate_per_minute=60, burst=5)
    limiter.check("acme\x00alice@example.com")
    (key,) = list(redis.scan_iter("fastrag:rl:*"))
    assert b"alice" not in key  # hashed: no user ids in Redis key names
    assert 0 < redis.ttl(key) <= 65


def test_answer_cache_roundtrip_and_tenant_invalidation(redis):
    service = replica(redis, rate_limit_per_minute=0, user_rate_limit_per_minute=0)
    service.ingest(
        ADMIN, DocumentIn("p.md", "Visitor parking is free for three hours.", frozenset({EVERYONE}))
    )
    first = service.ask(ALICE, "How long is visitor parking free?")
    assert not first.cached
    # A second replica answers from the shared cache.
    other = replica(redis, rate_limit_per_minute=0, user_rate_limit_per_minute=0)
    other.store = service.store  # same corpus generation, as with a shared database
    second = other.ask(ALICE, "How long is visitor parking free?")
    assert second.cached and second.text == first.text and second.sources == first.sources
    # Changing the corpus purges the tenant's entries from Redis, not just their keys.
    assert list(redis.scan_iter("fastrag:answer:*"))
    service.ingest(
        ADMIN, DocumentIn("p.md", "Visitor parking is free for two hours.", frozenset({EVERYONE}))
    )
    assert list(redis.scan_iter("fastrag:answer:*")) == []


def test_answers_are_stored_as_json_not_pickle(redis):
    service = replica(redis, rate_limit_per_minute=0, user_rate_limit_per_minute=0)
    service.ingest(
        ADMIN, DocumentIn("p.md", "Visitor parking is free for three hours.", frozenset({EVERYONE}))
    )
    answer = service.ask(ALICE, "How long is visitor parking free?")
    (key,) = [k for k in redis.scan_iter("fastrag:answer:*") if b":tenant:" not in k]
    raw = redis.get(key)
    assert raw.startswith(b"{")  # JSON; a pickle would start with b"\x80"
    assert answer_from_json(answer_to_json(answer)) == answer


def test_invalidate_tenant_leaves_other_tenants_alone(redis):
    cache = RedisAnswerCache(redis, ttl_seconds=60)
    from fastrag.domain import Answer

    cache.put("k1", Answer(text="a", citations=[], sources=[]), "acme")
    cache.put("k2", Answer(text="b", citations=[], sources=[]), "globex")
    cache.invalidate_tenant("acme")
    assert cache.get("k1") is None and cache.get("k2") is not None


def test_failed_login_budget_is_shared_across_replicas(redis):
    settings = {"auth_failures_per_minute": 1, "auth_failure_burst": 2, "rate_limit_per_minute": 0}
    keys = ApiKeyStore()
    keys.create(ALICE)
    clients = [
        TestClient(create_app(replica(redis, **settings), Authenticator(api_keys=keys)))
        for _ in range(2)
    ]
    codes = [
        clients[i % 2].get("/v1/me", headers={"X-API-Key": "frk_bad"}).status_code for i in range(4)
    ]
    assert codes == [401, 401, 429, 429]


def test_missing_redis_package_is_a_clear_configuration_error(monkeypatch):
    import builtins

    from fastrag.errors import ConfigurationError
    from fastrag.service import connect_redis

    real_import = builtins.__import__

    def no_redis(name, *args, **kwargs):
        if name == "redis":
            raise ImportError("no redis")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_redis)
    with pytest.raises(ConfigurationError):
        connect_redis("redis://localhost:6379/0")


class BrokenRedis:
    """Every command fails, as during a Redis outage."""

    def __getattr__(self, name):
        def fail(*args, **kwargs):
            raise ConnectionError("redis is down")

        if name == "register_script":
            return lambda script: fail
        return fail


def test_redis_outage_degrades_instead_of_failing_requests():
    from fastrag.domain import Answer

    cache = RedisAnswerCache(BrokenRedis(), ttl_seconds=60)
    assert cache.get("k") is None
    cache.put("k", Answer(text="a", citations=[], sources=[]), "acme")  # no exception
    cache.invalidate_tenant("acme")  # no exception
    RedisRateLimiter(BrokenRedis(), 60, 5).check("acme")  # fails open: allowed


def test_unreadable_cache_entries_are_misses(redis):
    cache = RedisAnswerCache(redis, ttl_seconds=60)
    redis.set("fastrag:answer:old", '{"text": "written by an older release"}')
    redis.set("fastrag:answer:junk", "not json")
    assert cache.get("old") is None and cache.get("junk") is None


def test_rerank_scores_survive_the_cache():
    from dataclasses import replace

    from fastrag.domain import Answer, Chunk, ScoredChunk

    chunk = Chunk("c", "acme", "d", 1, 0, "text", frozenset({EVERYONE}), "s.md")
    answer = Answer(
        text="t", citations=[], sources=[replace(ScoredChunk(chunk, 0.1), rerank_score=0.9)]
    )
    assert answer_from_json(answer_to_json(answer)).sources[0].rerank_score == 0.9


def test_cost_above_burst_is_rejected_up_front(redis):
    from fastrag.ratelimit import RateLimiter

    for limiter in (RateLimiter(60, 2), RedisRateLimiter(redis, 60, 2)):
        with pytest.raises(ValueError):
            limiter.check("acme", cost=3)
