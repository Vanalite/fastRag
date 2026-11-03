"""Per-tenant token-bucket rate limiting, so one tenant can't starve the others.

``RateLimiter`` keeps buckets in process, which is right for one replica. With several
replicas, set ``FASTRAG_REDIS_URL`` and ``RedisRateLimiter`` shares the buckets; the
interface is the same.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from typing import Any, Protocol

from fastrag.errors import RateLimited

log = logging.getLogger(__name__)


def _check_cost(cost: float, burst: float) -> None:
    if cost > burst:
        # Such a request could never succeed, and its retry_after would be a lie.
        raise ValueError(f"cost {cost} exceeds the bucket size {burst}")


class RateLimiter:
    def __init__(
        self,
        rate_per_minute: float,
        burst: int,
        clock: Callable[[], float] = time.monotonic,
        max_keys: int = 100_000,
    ):
        self.rate = rate_per_minute / 60.0  # tokens per second
        self.burst = float(burst)
        self._clock = clock
        # key -> (tokens, last_refill), least recently used first. Memory is bounded:
        # evicting the oldest key forgets a bucket that has most likely refilled anyway.
        self._buckets: OrderedDict[str, tuple[float, float]] = OrderedDict()
        self._max_keys = max_keys
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.rate > 0 and self.burst > 0

    def check(self, key: str, cost: float = 1.0) -> None:
        """Take ``cost`` tokens from ``key``'s bucket, or raise ``RateLimited``."""
        if not self.enabled:
            return
        _check_cost(cost, self.burst)
        now = self._clock()
        with self._lock:
            tokens = self._refilled(key, now)
            if tokens < cost:
                self._store(key, tokens, now)
                raise RateLimited(retry_after=(cost - tokens) / self.rate)
            self._store(key, tokens - cost, now)

    def ensure_available(self, key: str, cost: float = 1.0) -> None:
        """Raise ``RateLimited`` if ``key`` couldn't pay ``cost`` now, without taking any."""
        if not self.enabled:
            return
        now = self._clock()
        with self._lock:
            tokens = self._refilled(key, now)
            if tokens < cost:
                raise RateLimited(retry_after=(cost - tokens) / self.rate)

    def _refilled(self, key: str, now: float) -> float:
        tokens, last = self._buckets.get(key, (self.burst, now))
        return min(self.burst, tokens + (now - last) * self.rate)

    def _store(self, key: str, tokens: float, now: float) -> None:
        self._buckets[key] = (tokens, now)
        self._buckets.move_to_end(key)
        while len(self._buckets) > self._max_keys:
            self._buckets.popitem(last=False)

    def __len__(self) -> int:
        return len(self._buckets)


class Limiter(Protocol):
    def check(self, key: str, cost: float = 1.0) -> None: ...

    def ensure_available(self, key: str, cost: float = 1.0) -> None: ...


# Token bucket as one atomic Redis script. It uses the Redis server's clock (TIME), so
# replicas with skewed clocks still share one consistent bucket per key.
_BUCKET_LUA = """
local rate = tonumber(ARGV[1])
local burst = tonumber(ARGV[2])
local cost = tonumber(ARGV[3])
local consume = tonumber(ARGV[4])
local t = redis.call('TIME')
local now = tonumber(t[1]) + tonumber(t[2]) / 1000000
local state = redis.call('HMGET', KEYS[1], 'tokens', 'ts')
local tokens = tonumber(state[1]) or burst
local ts = tonumber(state[2]) or now
tokens = math.min(burst, tokens + math.max(0, now - ts) * rate)
local allowed = tokens >= cost
if consume == 1 then
  if allowed then tokens = tokens - cost end
  redis.call('HSET', KEYS[1], 'tokens', tostring(tokens), 'ts', tostring(now))
  redis.call('EXPIRE', KEYS[1], math.ceil(burst / rate) + 60)
end
if allowed then return {1, '0'} end
return {0, tostring((cost - tokens) / rate)}
"""


class RedisRateLimiter:
    """The same token bucket as ``RateLimiter``, stored in Redis and shared by replicas.

    Keys expire once their bucket would be full again, so idle clients cost no memory.
    """

    def __init__(
        self, client: Any, rate_per_minute: float, burst: int, prefix: str = "fastrag:rl:"
    ):
        self.rate = rate_per_minute / 60.0
        self.burst = float(burst)
        self._prefix = prefix
        self._script = client.register_script(_BUCKET_LUA)

    @property
    def enabled(self) -> bool:
        return self.rate > 0 and self.burst > 0

    def _run(self, key: str, cost: float, consume: bool) -> None:
        if not self.enabled:
            return
        _check_cost(cost, self.burst)
        hashed = hashlib.sha256(key.encode()).hexdigest()  # bounded, opaque key names
        try:
            allowed, retry_after = self._script(
                keys=[self._prefix + hashed], args=[self.rate, self.burst, cost, int(consume)]
            )
        except Exception:  # noqa: BLE001 - redis errors
            # Fail open: an unreachable Redis must not take the API down with it. The
            # outage is logged loudly; alert on it, since limits are off meanwhile.
            log.exception("rate limiter unavailable; allowing the request")
            return
        if not int(allowed):
            raise RateLimited(retry_after=float(retry_after))

    def check(self, key: str, cost: float = 1.0) -> None:
        self._run(key, cost, consume=True)

    def ensure_available(self, key: str, cost: float = 1.0) -> None:
        self._run(key, cost, consume=False)
