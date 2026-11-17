"""Embedding providers (D14). All return L2-normalized vectors."""

from __future__ import annotations

import asyncio
import hashlib
import math
from typing import Protocol

import httpx

from fastrag.config import Settings
from fastrag.providers.http import ProviderError, bearer, post_json
from fastrag.text.vietnamese import strip_diacritics


class Embedder(Protocol):
    model: str
    dim: int

    async def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    async def embed_query(self, text: str) -> list[float]: ...
    async def aclose(self) -> None: ...


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


class _HttpEmbedder:
    def __init__(self, s: Settings):
        self.model = s.embedding_model
        self.dim = s.embedding_dim
        self.batch = s.embedding_batch_size
        self.base = s.embedding_url.rstrip("/")
        self.headers = bearer(s.embedding_api_key.get_secret_value() if s.embedding_api_key else None)
        self.client = httpx.AsyncClient(timeout=s.embedding_timeout_s)

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch):
            out.extend(await self._embed(texts[i : i + self.batch]))
        for v in out:
            if len(v) != self.dim:
                raise ProviderError(
                    f"embedding has {len(v)} dimensions, database expects {self.dim} (FASTRAG_EMBEDDING_DIM)"
                )
        return [_normalize(v) for v in out]

    async def embed_query(self, text: str) -> list[float]:
        return (await self.embed_documents([text]))[0]

    async def aclose(self) -> None:
        await self.client.aclose()


class TeiEmbedder(_HttpEmbedder):
    """Hugging Face text-embeddings-inference (the default in docker compose)."""

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        data = await post_json(
            self.client,
            f"{self.base}/embed",
            {"inputs": texts, "normalize": True, "truncate": True},
            headers=self.headers,
        )
        return [list(map(float, v)) for v in data]


class OpenAIEmbedder(_HttpEmbedder):
    """Any OpenAI-compatible /v1/embeddings endpoint (LiteLLM, vLLM, Infinity, hosted)."""

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        url = self.base if self.base.endswith("/embeddings") else f"{self.base}/v1/embeddings"
        data = await post_json(self.client, url, {"model": self.model, "input": texts}, headers=self.headers)
        rows = sorted(data["data"], key=lambda r: r["index"])
        return [list(map(float, r["embedding"])) for r in rows]


class LocalEmbedder:
    """In-process sentence-transformers model. Needs the `local-models` extra."""

    def __init__(self, s: Settings):
        from sentence_transformers import SentenceTransformer

        self.model = s.embedding_model
        self._st = SentenceTransformer(s.embedding_model)
        self.dim = s.embedding_dim
        self.batch = s.embedding_batch_size

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vecs = await asyncio.to_thread(self._st.encode, texts, batch_size=self.batch, normalize_embeddings=True)
        return [list(map(float, v)) for v in vecs]

    async def embed_query(self, text: str) -> list[float]:
        return (await self.embed_documents([text]))[0]

    async def aclose(self) -> None:
        return None


class HashEmbedder:
    """Character n-gram feature hashing. No semantics: for tests and for
    trying the API without a model server. Never use it in production."""

    def __init__(self, dim: int):
        self.model = "hash"
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        t = " " + strip_diacritics(text.lower()) + " "
        for n in (3, 4):
            for i in range(len(t) - n + 1):
                h = int.from_bytes(hashlib.blake2b(t[i : i + n].encode(), digest_size=8).digest(), "big")
                vec[h % self.dim] += 1.0 if (h >> 63) else -1.0
        return _normalize(vec)

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._vec(text)

    async def aclose(self) -> None:
        return None


def build_embedder(s: Settings) -> Embedder:
    if s.embedding_provider == "tei":
        return TeiEmbedder(s)
    if s.embedding_provider == "openai":
        return OpenAIEmbedder(s)
    if s.embedding_provider == "local":
        return LocalEmbedder(s)
    return HashEmbedder(s.embedding_dim)
