"""Answer cache, partitioned by access scope and corpus generation.

A cache keyed on the question alone is the classic RAG data leak: user A's answer,
built from documents only A may read, is served to user B who asks the same thing.
Here the key includes the caller's whole access scope (tenant + every ACL token) and
the tenant's corpus generation, which the store bumps on every ingest or delete - so a
deleted document can never come back out of the cache either.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from collections.abc import Callable

from fastrag.domain import Answer
from fastrag.security.acl import AccessFilter


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
        self._entries: OrderedDict[str, tuple[float, Answer]] = OrderedDict()
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.ttl > 0 and self.max_entries > 0

    @staticmethod
    def key(access: AccessFilter, generation: int, question: str, k: int) -> str:
        normalized = " ".join(question.lower().split())
        material = json.dumps([access.scope_key(), generation, k, normalized])
        return hashlib.sha256(material.encode()).hexdigest()

    def get(self, key: str) -> Answer | None:
        if not self.enabled:
            return None
        with self._lock:
            item = self._entries.get(key)
            if item is None:
                return None
            expires, answer = item
            if self._clock() >= expires:
                del self._entries[key]
                return None
            self._entries.move_to_end(key)
            return answer

    def put(self, key: str, answer: Answer) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._entries[key] = (self._clock() + self.ttl, answer)
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    def __len__(self) -> int:
        return len(self._entries)
