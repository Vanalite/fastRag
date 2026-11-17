"""Reranker providers (D14). Scores are calibrated to 0..1 so thresholds mean
the same thing whichever provider is configured."""

from __future__ import annotations

import asyncio
import math
from typing import Protocol

import httpx

from fastrag.config import Settings
from fastrag.providers.http import bearer, post_json


class Reranker(Protocol):
    name: str

    async def rerank(self, query: str, texts: list[str]) -> list[float]: ...
    async def aclose(self) -> None: ...


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


class _Http:
    def __init__(self, s: Settings, default_url: str | None = None):
        self.name = f"{s.reranker_provider}:{s.reranker_model}"
        self.model = s.reranker_model
        self.base = (default_url or s.reranker_url).rstrip("/")
        self.headers = bearer(s.reranker_api_key.get_secret_value() if s.reranker_api_key else None)
        # One attempt only: the reranker is on the latency-critical path and
        # retrieval degrades to fused ranking if it fails.
        self.client = httpx.AsyncClient(timeout=s.reranker_timeout_s)

    async def aclose(self) -> None:
        await self.client.aclose()


class TeiReranker(_Http):
    async def rerank(self, query: str, texts: list[str]) -> list[float]:
        data = await post_json(
            self.client,
            f"{self.base}/rerank",
            {"query": query, "texts": texts, "raw_scores": True, "truncate": True},
            headers=self.headers,
            attempts=1,
        )
        scores = [0.0] * len(texts)
        for row in data:
            scores[row["index"]] = sigmoid(float(row["score"]))
        return scores


class CohereReranker(_Http):
    """Cohere /v2/rerank. Also fits providers that copy its API."""

    def __init__(self, s: Settings):
        url = s.reranker_url if "cohere" in s.reranker_url else "https://api.cohere.com"
        super().__init__(s, url)

    async def rerank(self, query: str, texts: list[str]) -> list[float]:
        data = await post_json(
            self.client,
            f"{self.base}/v2/rerank",
            {"model": self.model, "query": query, "documents": texts, "top_n": len(texts)},
            headers=self.headers,
            attempts=1,
        )
        scores = [0.0] * len(texts)
        for row in data["results"]:
            scores[row["index"]] = float(row["relevance_score"])
        return scores


class JinaReranker(_Http):
    def __init__(self, s: Settings):
        url = s.reranker_url if "jina" in s.reranker_url else "https://api.jina.ai"
        super().__init__(s, url)

    async def rerank(self, query: str, texts: list[str]) -> list[float]:
        data = await post_json(
            self.client,
            f"{self.base}/v1/rerank",
            {"model": self.model, "query": query, "documents": texts, "top_n": len(texts)},
            headers=self.headers,
            attempts=1,
        )
        scores = [0.0] * len(texts)
        for row in data["results"]:
            scores[row["index"]] = float(row["relevance_score"])
        return scores


class LocalReranker:
    def __init__(self, s: Settings):
        from sentence_transformers import CrossEncoder

        self.name = f"local:{s.reranker_model}"
        self._ce = CrossEncoder(s.reranker_model)

    async def rerank(self, query: str, texts: list[str]) -> list[float]:
        import torch

        # Ask for raw logits: CrossEncoder applies its own sigmoid by default,
        # and we calibrate the same way for every provider.
        raw = await asyncio.to_thread(self._ce.predict, [(query, t) for t in texts], activation_fn=torch.nn.Identity())
        return [sigmoid(float(x)) for x in raw]

    async def aclose(self) -> None:
        return None


def logit(p: float) -> float:
    p = min(max(p, 1e-9), 1 - 1e-9)
    return math.log(p / (1 - p))


# Fitted on Vietnamese real-estate questions (2026-11-07, 13 questions, see
# design log D21). Refit on the pilot's reference set with `fastrag calibrate`.
DEFAULT_CALIBRATION: dict[str, tuple[float, float]] = {
    "BAAI/bge-reranker-v2-m3": (0.63, 2.15),
}


def parse_calibration(s: Settings) -> tuple[float, float]:
    if s.reranker_calibration:
        a, b = (float(x) for x in s.reranker_calibration.split(","))
        return a, b
    return DEFAULT_CALIBRATION.get(s.reranker_model, (1.0, 0.0))


class CalibratedReranker:
    """Maps a provider's 0..1 scores so that 0.5 means "as likely relevant as
    not" for this model. Profile thresholds assume that scale."""

    def __init__(self, inner: Reranker, a: float, b: float):
        self.inner, self.a, self.b = inner, a, b
        self.name = inner.name

    async def raw(self, query: str, texts: list[str]) -> list[float]:
        return await self.inner.rerank(query, texts)

    async def rerank(self, query: str, texts: list[str]) -> list[float]:
        return [sigmoid(self.a * logit(p) + self.b) for p in await self.inner.rerank(query, texts)]

    async def aclose(self) -> None:
        await self.inner.aclose()


def build_reranker(s: Settings) -> Reranker | None:
    inner: Reranker
    match s.reranker_provider:
        case "tei":
            inner = TeiReranker(s)
        case "cohere":
            inner = CohereReranker(s)
        case "jina":
            inner = JinaReranker(s)
        case "local":
            inner = LocalReranker(s)
        case _:
            return None
    a, b = parse_calibration(s)
    return inner if (a, b) == (1.0, 0.0) else CalibratedReranker(inner, a, b)
