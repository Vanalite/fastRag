"""Storage interfaces. Swap implementations (Postgres, pgvector, OpenSearch...) behind these.

Note the keyword-only ``access`` argument on every search method: there is no way to
call an index without stating who is asking. Adapters must apply it as a pre-filter
(for pgvector: a WHERE clause on tenant_id and an ACL-array overlap, or Postgres
row-level security) - see docs/adr/0002-access-control.md.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Protocol

import numpy as np

from fastrag.domain import Chunk, DocumentRecord
from fastrag.security.acl import AccessFilter


class DocumentStore(Protocol):
    def check_embedder(self, name: str) -> None:
        """Record the embedder on first use; fail if a different one is configured later."""

    def get_document(self, tenant_id: str, doc_id: str) -> DocumentRecord | None: ...

    def list_documents(self, tenant_id: str) -> list[DocumentRecord]: ...

    def replace_document(
        self, record: DocumentRecord, chunks: list[Chunk], embeddings: np.ndarray
    ) -> list[str]:
        """Atomically write a new version; return the ids of the chunks it replaced."""

    def delete_document(self, tenant_id: str, doc_id: str) -> list[str] | None:
        """Remove a document and its chunks; return removed chunk ids, or None if absent."""

    def bump_generation(self, tenant_id: str) -> None:
        """Mark the tenant's corpus as changed. Call after the indexes reflect the change."""

    def iter_chunks(self) -> Iterator[tuple[Chunk, np.ndarray]]:
        """Every stored chunk with its embedding, for rebuilding indexes."""

    def generation(self, tenant_id: str) -> int:
        """A counter bumped on every change to the tenant's corpus (for cache invalidation)."""

    def ping(self) -> None: ...


class VectorIndex(Protocol):
    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None: ...

    def remove(self, chunk_ids: Iterable[str]) -> None: ...

    def search(
        self, vector: np.ndarray, k: int, *, access: AccessFilter
    ) -> list[tuple[Chunk, float]]: ...

    def chunk_ids(self) -> set[str]: ...


class KeywordIndex(Protocol):
    def add(self, chunks: list[Chunk]) -> None: ...

    def remove(self, chunk_ids: Iterable[str]) -> None: ...

    def search(
        self, query: str, k: int, *, access: AccessFilter
    ) -> list[tuple[Chunk, float, float, int]]:
        """Return (chunk, score, query_term_coverage, matched_term_count) tuples."""

    def chunk_ids(self) -> set[str]: ...
