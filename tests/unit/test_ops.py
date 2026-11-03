"""Cache, rate limiter, metrics, audit log and fusion."""

import json

import pytest

from fastrag.cache import AnswerCache
from fastrag.domain import Answer, Principal
from fastrag.errors import RateLimited
from fastrag.observability import AuditLog, JsonlAuditSink, MetricsRegistry, verify_chain
from fastrag.ratelimit import RateLimiter
from fastrag.retrieval import reciprocal_rank_fusion
from fastrag.security.acl import AccessFilter


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_cache_key_partitions_by_scope_generation_and_k():
    a = AccessFilter.for_principal(Principal("acme", "a", frozenset({"x"})))
    b = AccessFilter.for_principal(Principal("acme", "b"))
    base = AnswerCache.key(a, 1, "Hello  World", 4)
    assert base == AnswerCache.key(a, 1, "hello world", 4)  # normalized
    assert base != AnswerCache.key(b, 1, "hello world", 4)
    assert base != AnswerCache.key(a, 2, "hello world", 4)
    assert base != AnswerCache.key(a, 1, "hello world", 5)


def test_cache_ttl_and_lru():
    clock = Clock()
    cache = AnswerCache(ttl_seconds=10, max_entries=2, clock=clock)
    answer = Answer(text="x", citations=[], sources=[])
    cache.put("a", answer, "acme")
    cache.put("b", answer, "acme")
    cache.get("a")
    cache.put("c", answer, "acme")  # evicts b, the least recently used
    assert cache.get("b") is None and cache.get("a") is answer
    clock.now = 11
    assert cache.get("a") is None


def test_rate_limiter_is_per_key_and_refills():
    clock = Clock()
    limiter = RateLimiter(rate_per_minute=60, burst=2, clock=clock)
    limiter.check("acme")
    limiter.check("acme")
    with pytest.raises(RateLimited) as exc:
        limiter.check("acme")
    assert exc.value.retry_after == pytest.approx(1.0)
    limiter.check("globex")  # other tenants unaffected
    clock.now = 1.0
    limiter.check("acme")


def test_metrics_render_prometheus_text():
    metrics = MetricsRegistry()
    metrics.requests.inc({"action": "ask", "outcome": "success", "tenant": "acme"})
    metrics.latency.observe(0.02, {"stage": "retrieval"})
    text = metrics.render()
    assert 'fastrag_requests_total{action="ask",outcome="success",tenant="acme"} 1.0' in text
    assert 'fastrag_stage_seconds_bucket{stage="retrieval",le="0.025"} 1.0' in text
    assert 'fastrag_stage_seconds_count{stage="retrieval"} 1.0' in text


def test_audit_chain_detects_tampering(tmp_path):
    log = AuditLog(JsonlAuditSink(tmp_path / "audit.jsonl"))
    alice = Principal("acme", "alice")
    log.record("ask", "success", alice, doc_ids=["d1"])
    log.record("delete", "denied", alice)
    # A new process continues the same chain.
    log2 = AuditLog(JsonlAuditSink(tmp_path / "audit.jsonl"))
    log2.record("ask", "success", alice)
    entries = log2.sink.read()
    assert len(entries) == 3 and verify_chain(entries)
    assert [e["action"] for e in log2.events("acme")] == ["ask", "delete", "ask"]
    assert log2.events("globex") == []

    tampered = json.loads(json.dumps(entries))
    tampered[1]["outcome"] = "success"
    assert not verify_chain(tampered)
    assert not verify_chain(entries[1:])  # deleting an entry breaks the chain too


def test_rrf_rewards_agreement():
    fused = dict(reciprocal_rank_fusion([["a", "b", "c"], ["b", "d"]], k=60))
    assert max(fused, key=fused.get) == "b"
    assert fused["a"] > fused["d"]
