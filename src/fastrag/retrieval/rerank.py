"""Rerankers: a second, more expensive relevance pass over the fused candidates.

Hybrid retrieval is tuned for recall; a cross-encoder reads the query and each candidate
*together*, which is much better at precision. Running it over the top ~20 candidates
only keeps the cost bounded. Rerankers reorder; they never add chunks, so they can't
widen what a caller sees.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Protocol

from fastrag.domain import ScoredChunk
from fastrag.errors import ConfigurationError


class Reranker(Protocol):
    def rerank(self, query: str, results: list[ScoredChunk]) -> list[ScoredChunk]:
        """Return the same chunks, best first, with ``rerank_score`` set."""
        ...


class CrossEncoderReranker:
    """sentence-transformers cross-encoder (``pip install 'fastrag[st]'``)."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ConfigurationError(
                "the cross-encoder reranker needs sentence-transformers; "
                "run: pip install 'fastrag[st]'"
            ) from exc
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, results: list[ScoredChunk]) -> list[ScoredChunk]:
        if not results:
            return results
        scores = self._model.predict([(query, r.chunk.text) for r in results])
        rescored = [replace(r, rerank_score=float(s)) for r, s in zip(results, scores, strict=True)]
        return sorted(rescored, key=lambda r: r.rerank_score, reverse=True)


def get_reranker(spec: str | None) -> Reranker | None:
    """``None``/empty: no reranker. ``cross-encoder`` or ``cross-encoder:<model>``."""
    if not spec:
        return None
    kind, _, model = spec.partition(":")
    if kind == "cross-encoder":
        return CrossEncoderReranker(model) if model else CrossEncoderReranker()
    raise ConfigurationError(f"unknown reranker {spec!r}; expected 'cross-encoder[:<model>]'")
