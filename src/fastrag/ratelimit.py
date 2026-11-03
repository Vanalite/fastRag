"""Per-tenant token-bucket rate limiting, so one tenant can't starve the others.

In-process state works for a single replica. With several replicas, move the buckets
to Redis (or rate-limit at the API gateway); the ``check`` interface stays the same.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

from fastrag.errors import RateLimited


class RateLimiter:
    def __init__(
        self,
        rate_per_minute: float,
        burst: int,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.rate = rate_per_minute / 60.0  # tokens per second
        self.burst = float(burst)
        self._clock = clock
        self._buckets: dict[str, tuple[float, float]] = {}  # key -> (tokens, last_refill)
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.rate > 0 and self.burst > 0

    def check(self, key: str, cost: float = 1.0) -> None:
        if not self.enabled:
            return
        now = self._clock()
        with self._lock:
            tokens, last = self._buckets.get(key, (self.burst, now))
            tokens = min(self.burst, tokens + (now - last) * self.rate)
            if tokens < cost:
                self._buckets[key] = (tokens, now)
                raise RateLimited(retry_after=(cost - tokens) / self.rate)
            self._buckets[key] = (tokens - cost, now)
