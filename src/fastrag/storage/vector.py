"""In-memory dense vector index, partitioned by tenant.

Partitioning by tenant means a search only ever touches the caller's tenant's vectors;
within the partition, an ACL mask is applied before top-k selection. Search is exact
(one matrix-vector product), which is fast up to a few hundred thousand chunks per tenant.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from fastrag.domain import Chunk
from fastrag.security.acl import AccessFilter


class _Partition:
    def __init__(self, dim: int):
        self.vectors = np.zeros((0, dim), dtype=np.float32)
        self.chunks: list[Chunk] = []

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        self.chunks.extend(chunks)
        self.vectors = np.vstack([self.vectors, vectors.astype(np.float32, copy=False)])

    def remove(self, ids: set[str]) -> None:
        keep = [i for i, c in enumerate(self.chunks) if c.id not in ids]
        if len(keep) != len(self.chunks):
            self.chunks = [self.chunks[i] for i in keep]
            self.vectors = self.vectors[keep]


class NumpyVectorIndex:
    def __init__(self, dim: int):
        self.dim = dim
        self._partitions: dict[str, _Partition] = {}

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if vectors.shape != (len(chunks), self.dim):
            raise ValueError(f"expected vectors of shape {(len(chunks), self.dim)}")
        by_tenant: dict[str, list[int]] = {}
        for i, chunk in enumerate(chunks):
            by_tenant.setdefault(chunk.tenant_id, []).append(i)
        for tenant_id, rows in by_tenant.items():
            partition = self._partitions.setdefault(tenant_id, _Partition(self.dim))
            partition.add([chunks[i] for i in rows], vectors[rows])

    def remove(self, chunk_ids: Iterable[str]) -> None:
        ids = set(chunk_ids)
        if ids:
            for partition in self._partitions.values():
                partition.remove(ids)

    def search(
        self, vector: np.ndarray, k: int, *, access: AccessFilter
    ) -> list[tuple[Chunk, float]]:
        partition = self._partitions.get(access.tenant_id)
        if partition is None or k <= 0 or not partition.chunks:
            return []
        allowed = np.fromiter(
            (access.allows(c.tenant_id, c.acl) for c in partition.chunks),
            dtype=bool,
            count=len(partition.chunks),
        )
        candidates = np.flatnonzero(allowed)
        if candidates.size == 0:
            return []
        scores = partition.vectors[candidates] @ vector.astype(np.float32).reshape(self.dim)
        k = min(k, scores.size)
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top], kind="stable")]
        return [(partition.chunks[candidates[i]], float(scores[i])) for i in top]

    def chunk_ids(self) -> set[str]:
        return {c.id for p in self._partitions.values() for c in p.chunks}
