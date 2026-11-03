"""Groups the derived indexes so they always change together."""

from __future__ import annotations

import threading
from collections.abc import Iterable

import numpy as np

from fastrag.domain import Chunk
from fastrag.security.acl import AccessFilter
from fastrag.storage.base import DocumentStore, KeywordIndex, VectorIndex


class SearchIndexes:
    """Applies every change to the vector and keyword indexes under one lock.

    Readers take the same lock, so a search never sees a document half-replaced
    (old chunks gone from one index but still present in the other). A readers-writer
    lock would allow concurrent searches; a plain lock keeps the reference simple.
    """

    def __init__(self, vectors: VectorIndex, keywords: KeywordIndex):
        self.vectors = vectors
        self.keywords = keywords
        self._lock = threading.RLock()

    def apply(
        self,
        remove_ids: Iterable[str] = (),
        add_chunks: list[Chunk] | None = None,
        add_vectors: np.ndarray | None = None,
    ) -> None:
        remove_ids = list(remove_ids)
        with self._lock:
            self.vectors.remove(remove_ids)
            self.keywords.remove(remove_ids)
            if add_chunks:
                assert add_vectors is not None
                self.vectors.add(add_chunks, add_vectors)
                self.keywords.add(add_chunks)

    def rebuild_from(self, store: DocumentStore, dim: int) -> int:
        """Load every chunk from the system of record. Returns the number of chunks."""
        chunks: list[Chunk] = []
        vectors: list[np.ndarray] = []
        for chunk, vector in store.iter_chunks():
            chunks.append(chunk)
            vectors.append(vector)
        if chunks:
            self.apply(add_chunks=chunks, add_vectors=np.stack(vectors).reshape(len(chunks), dim))
        return len(chunks)

    def dense(
        self, vector: np.ndarray, k: int, *, access: AccessFilter
    ) -> list[tuple[Chunk, float]]:
        with self._lock:
            return self.vectors.search(vector, k, access=access)

    def keyword(
        self, query: str, k: int, *, access: AccessFilter
    ) -> list[tuple[Chunk, float, float, int]]:
        with self._lock:
            return self.keywords.search(query, k, access=access)

    def consistent_with(self, store: DocumentStore) -> bool:
        """Invariant check: both indexes hold exactly the chunks in the store."""
        stored = {c.id for c, _ in store.iter_chunks()}
        with self._lock:
            return self.vectors.chunk_ids() == stored == self.keywords.chunk_ids()
