"""Answer cache, partitioned by access scope and corpus generation.

A cache keyed on the question alone is the classic RAG data leak: user A's answer,
built from documents only A may read, is served to user B who asks the same thing.
Here the key includes the caller's whole access scope (tenant + every ACL token) and
the tenant's corpus generation, which the store bumps on every ingest or delete - so a
deleted document can never come back out of the cache either.

Generation keys make stale entries unreachable; ``invalidate_tenant`` also *removes*
them, so content from a deleted document doesn't linger in memory or Redis until its
TTL runs out ("right to be forgotten").

Two implementations share the same interface: ``AnswerCache`` (in process) and
``RedisAnswerCache`` (shared by every replica).
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from typing import Any, Protocol

from fastrag.domain import Answer, Chunk, Citation, ScoredChunk, Usage
from fastrag.security.acl import AccessFilter

log = logging.getLogger(__name__)


def cache_key(access: AccessFilter, generation: int, question: str, k: int) -> str:
    normalized = " ".join(question.lower().split())
    material = json.dumps([access.scope_key(), generation, k, normalized])
    return hashlib.sha256(material.encode()).hexdigest()


class Cache(Protocol):
    def get(self, key: str) -> Answer | None: ...

    def put(self, key: str, answer: Answer, tenant_id: str) -> None: ...

    def invalidate_tenant(self, tenant_id: str) -> None: ...


class AnswerCache:
    def __init__(
        self,
        ttl_seconds: float = 300,
        max_entries: int = 1000,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.ttl = ttl_seconds
        self.max_entries = max_entries
        self._clock = clock
        self._entries: OrderedDict[str, tuple[float, str, Answer]] = OrderedDict()
        self._lock = threading.Lock()

    key = staticmethod(cache_key)

    @property
    def enabled(self) -> bool:
        return self.ttl > 0 and self.max_entries > 0

    def get(self, key: str) -> Answer | None:
        if not self.enabled:
            return None
        with self._lock:
            item = self._entries.get(key)
            if item is None:
                return None
            expires, _, answer = item
            if self._clock() >= expires:
                del self._entries[key]
                return None
            self._entries.move_to_end(key)
            return answer

    def put(self, key: str, answer: Answer, tenant_id: str) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._entries[key] = (self._clock() + self.ttl, tenant_id, answer)
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    def invalidate_tenant(self, tenant_id: str) -> None:
        with self._lock:
            for key in [k for k, (_, t, _) in self._entries.items() if t == tenant_id]:
                del self._entries[key]

    def __len__(self) -> int:
        return len(self._entries)


class RedisAnswerCache:
    """Answer cache in Redis, shared by every replica.

    Entries are JSON, never pickle: unpickling data from a shared store would let anyone
    who can write to Redis run code in every replica. Cached answers contain document
    text, so treat Redis like the document store: authentication, TLS, private network.
    Each tenant's keys are tracked in a set so ``invalidate_tenant`` can delete them.
    """

    def __init__(self, client: Any, ttl_seconds: float = 300, prefix: str = "fastrag:answer:"):
        self._redis = client
        self.ttl = ttl_seconds
        self._prefix = prefix
        # Delete every tracked entry and the set itself in one atomic step, so a put
        # racing the purge can't survive it untracked (holding deleted text until TTL).
        self._purge = client.register_script(_PURGE_LUA)

    key = staticmethod(cache_key)

    @property
    def enabled(self) -> bool:
        return self.ttl > 0

    def _tenant_set(self, tenant_id: str) -> str:
        return f"{self._prefix}tenant:{hashlib.sha256(tenant_id.encode()).hexdigest()}"

    # A cache is an optimization: if Redis is unreachable or an entry can't be read
    # (e.g. written by an older release), behave as a miss and log, never fail the
    # request. Correctness doesn't depend on invalidation either: keys include the
    # corpus generation, so a skipped purge only delays freeing memory.

    def get(self, key: str) -> Answer | None:
        if not self.enabled:
            return None
        try:
            raw = self._redis.get(self._prefix + key)
            return answer_from_json(raw) if raw is not None else None
        except (KeyError, TypeError, ValueError):
            log.warning("unreadable answer-cache entry; treating as a miss")
            return None
        except Exception:  # noqa: BLE001 - redis errors, by design not fatal here
            log.exception("answer cache unavailable; treating as a miss")
            return None

    def put(self, key: str, answer: Answer, tenant_id: str) -> None:
        if not self.enabled:
            return
        ttl = max(1, int(self.ttl))
        tenant_set = self._tenant_set(tenant_id)
        try:
            pipe = self._redis.pipeline(transaction=True)
            pipe.set(self._prefix + key, answer_to_json(answer), ex=ttl)
            pipe.sadd(tenant_set, key)
            pipe.expire(tenant_set, ttl)
            pipe.execute()
        except Exception:  # noqa: BLE001
            log.exception("answer cache unavailable; not caching")

    def invalidate_tenant(self, tenant_id: str) -> None:
        try:
            self._purge(keys=[self._tenant_set(tenant_id)], args=[self._prefix])
        except Exception:  # noqa: BLE001
            log.exception("could not purge cached answers for a tenant")


_PURGE_LUA = """
local members = redis.call('SMEMBERS', KEYS[1])
for _, key in ipairs(members) do
  redis.call('DEL', ARGV[1] .. key)
end
redis.call('DEL', KEYS[1])
return #members
"""


# --- JSON (de)serialization of answers --------------------------------------------


def _chunk_to_dict(chunk: Chunk) -> dict[str, Any]:
    return {
        "id": chunk.id,
        "tenant_id": chunk.tenant_id,
        "doc_id": chunk.doc_id,
        "version": chunk.version,
        "ordinal": chunk.ordinal,
        "text": chunk.text,
        "acl": sorted(chunk.acl),
        "source_id": chunk.source_id,
        "title": chunk.title,
    }


def answer_to_json(answer: Answer) -> str:
    return json.dumps(
        {
            "text": answer.text,
            "abstained": answer.abstained,
            "usage": vars(answer.usage),
            "citations": [vars(c) for c in answer.citations],
            "sources": [
                {
                    "chunk": _chunk_to_dict(s.chunk),
                    "score": s.score,
                    "dense_score": s.dense_score,
                    "keyword_score": s.keyword_score,
                    "keyword_coverage": s.keyword_coverage,
                    "keyword_matches": s.keyword_matches,
                    "rerank_score": s.rerank_score,
                }
                for s in answer.sources
            ],
        }
    )


def answer_from_json(raw: str | bytes) -> Answer:
    data = json.loads(raw)
    return Answer(
        text=data["text"],
        abstained=data["abstained"],
        usage=Usage(**data["usage"]),
        citations=[Citation(**c) for c in data["citations"]],
        sources=[
            ScoredChunk(
                chunk=Chunk(**{**s["chunk"], "acl": frozenset(s["chunk"]["acl"])}),
                score=s["score"],
                dense_score=s["dense_score"],
                keyword_score=s["keyword_score"],
                keyword_coverage=s["keyword_coverage"],
                keyword_matches=s["keyword_matches"],
                rerank_score=s.get("rerank_score"),
            )
            for s in data["sources"]
        ],
    )
