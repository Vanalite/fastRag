"""Embedding backends.

Use ``get_embedder(spec)`` to build one from a short string:

- ``"hashing"`` or ``"hashing:2048"``: dependency-free feature hashing (default).
- ``"st:all-MiniLM-L6-v2"``: a sentence-transformers model (``pip install 'fastrag[st]'``).
"""

from __future__ import annotations

from fastrag.embeddings.base import Embedder
from fastrag.embeddings.hashing import HashingEmbedder

__all__ = ["Embedder", "HashingEmbedder", "get_embedder"]


def get_embedder(spec: str) -> Embedder:
    kind, _, arg = spec.partition(":")
    if kind == "hashing":
        return HashingEmbedder(dim=int(arg)) if arg else HashingEmbedder()
    if kind == "st":
        from fastrag.embeddings.sentence_transformers import SentenceTransformerEmbedder

        return SentenceTransformerEmbedder(arg or "all-MiniLM-L6-v2")
    raise ValueError(f"Unknown embedder {spec!r}; expected 'hashing[:dim]' or 'st:<model>'")
