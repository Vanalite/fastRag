"""Shared HTTP client with retries for model providers."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

RETRYABLE = {408, 425, 429, 500, 502, 503, 504}


class ProviderError(RuntimeError):
    pass


async def post_json(
    client: httpx.AsyncClient,
    url: str,
    payload: dict[str, Any],
    *,
    headers: dict[str, str] | None = None,
    attempts: int = 3,
    backoff_s: float = 0.5,
) -> Any:
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            resp = await client.post(url, json=payload, headers=headers)
            if resp.status_code in RETRYABLE and attempt < attempts - 1:
                last = ProviderError(f"{url} returned {resp.status_code}")
            else:
                if resp.status_code >= 400:
                    raise ProviderError(f"{url} returned {resp.status_code}: {resp.text[:300]}")
                return resp.json()
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            last = exc
            if attempt == attempts - 1:
                break
        await asyncio.sleep(backoff_s * (2**attempt))
    raise ProviderError(f"{url} failed after {attempts} attempts: {last}")


def bearer(token: str | None) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"} if token else {}
