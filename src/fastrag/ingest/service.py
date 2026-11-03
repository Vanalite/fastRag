"""Ingest and delete documents, keeping the store and every index in step."""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Callable
from datetime import UTC, datetime

from fastrag.domain import (
    Chunk,
    DocumentIn,
    DocumentRecord,
    IngestResult,
    document_id,
    is_valid_acl_token,
)
from fastrag.embeddings import Embedder
from fastrag.errors import InvalidInput, PermissionDenied
from fastrag.ingest.chunking import chunk_text
from fastrag.ingest.parsers import parse
from fastrag.ingest.redact import Redactor
from fastrag.storage.base import DocumentStore
from fastrag.storage.indexes import SearchIndexes

# Bump when parsing, redaction or chunking changes, so re-ingesting unchanged content
# still re-processes it instead of short-circuiting as "unchanged".
PIPELINE_VERSION = "2"


class _StripedLocks:
    """A fixed pool of locks keyed by hash: bounded memory, per-document serialization."""

    def __init__(self, stripes: int = 64):
        self._locks = [threading.Lock() for _ in range(stripes)]

    def for_key(self, *parts: str) -> threading.Lock:
        digest = hashlib.blake2b("\0".join(parts).encode(), digest_size=4).digest()
        return self._locks[int.from_bytes(digest) % len(self._locks)]


class IngestService:
    def __init__(
        self,
        store: DocumentStore,
        indexes: SearchIndexes,
        embedder: Embedder,
        *,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        redactor: Redactor | None = None,
        max_document_chars: int = 2_000_000,
    ):
        self.store = store
        self.indexes = indexes
        self.embedder = embedder
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.redactor = redactor
        self.max_document_chars = max_document_chars
        self._locks = _StripedLocks()

    def ingest(
        self,
        tenant_id: str,
        doc: DocumentIn,
        can_modify: Callable[[DocumentRecord], bool] = lambda _: True,
    ) -> tuple[IngestResult, dict[str, int]]:
        """Create or replace a document. Returns the result and PII redaction counts.

        ``can_modify`` is checked against the existing version, inside the per-document
        lock, before anything else is revealed about it.
        """
        if not doc.source_id.strip():
            raise InvalidInput("source_id must not be empty")
        if not doc.acl:
            raise InvalidInput("acl must list at least one principal (deny-by-default)")
        bad = sorted(t for t in doc.acl if not is_valid_acl_token(t))
        if bad:
            raise InvalidInput(f"invalid ACL tokens {bad}; use tenant:*, user:<id> or group:<id>")
        if len(doc.text) > self.max_document_chars:
            raise InvalidInput(f"document exceeds {self.max_document_chars} characters")

        parsed = parse(doc.text, doc.media_type)
        title, text = doc.title or parsed.title, parsed.text
        redactions: dict[str, int] = {}
        if self.redactor is not None:
            result = self.redactor.redact(text)
            text, redactions = result.text, result.counts
            title = self.redactor.redact(title).text
        if not text.strip():
            raise InvalidInput("document has no text content")

        doc_id = document_id(tenant_id, doc.source_id)
        # The hash covers everything that affects retrieval or access, so an ACL-only
        # change still produces a new version (and re-stamps every chunk's ACL).
        content_hash = hashlib.sha256(
            json.dumps(
                [
                    PIPELINE_VERSION,
                    self.embedder.name,
                    text,
                    title,
                    sorted(doc.acl),
                    doc.metadata,
                    self.chunk_size,
                    self.chunk_overlap,
                ],
                sort_keys=True,
                default=str,
            ).encode()
        ).hexdigest()

        with self._locks.for_key(tenant_id, doc_id):
            existing = self.store.get_document(tenant_id, doc_id)
            if existing is not None and not can_modify(existing):
                raise PermissionDenied("not allowed to modify this document")
            if existing is not None and existing.content_hash == content_hash:
                return IngestResult(doc_id, existing.version, "unchanged", existing.chunk_count), {}

            version = existing.version + 1 if existing else 1
            pieces = chunk_text(text, self.chunk_size, self.chunk_overlap)
            chunks = [
                Chunk(
                    id=hashlib.sha256(
                        f"{tenant_id}\0{doc_id}\0{version}\0{i}\0{content_hash}".encode()
                    ).hexdigest()[:32],
                    tenant_id=tenant_id,
                    doc_id=doc_id,
                    version=version,
                    ordinal=i,
                    text=piece,
                    acl=doc.acl,
                    source_id=doc.source_id,
                    title=title,
                )
                for i, piece in enumerate(pieces)
            ]
            # Prefixing the title gives each chunk's embedding document-level context.
            vectors = self.embedder.embed(
                [f"{title}\n{c.text}" if title else c.text for c in chunks]
            )
            record = DocumentRecord(
                tenant_id=tenant_id,
                doc_id=doc_id,
                source_id=doc.source_id,
                title=title,
                version=version,
                content_hash=content_hash,
                acl=doc.acl,
                metadata=dict(doc.metadata),
                chunk_count=len(chunks),
                updated_at=datetime.now(UTC).isoformat(timespec="seconds"),
            )
            # The store commit is the point of truth. If the process dies before the
            # index update, the indexes are rebuilt from the store on next start.
            replaced = self.store.replace_document(record, chunks, vectors)
            self.indexes.apply(remove_ids=replaced, add_chunks=chunks, add_vectors=vectors)
            # Bump only once the indexes serve the new version: a cache entry keyed on the
            # new generation can then never hold an answer built from the old index.
            self.store.bump_generation(tenant_id)

        status = "updated" if existing else "created"
        return IngestResult(doc_id, version, status, len(chunks)), redactions

    def delete(
        self,
        tenant_id: str,
        doc_id: str,
        can_modify: Callable[[DocumentRecord], bool] = lambda _: True,
    ) -> bool:
        """Delete a document everywhere. Returns False if absent or not modifiable."""
        with self._locks.for_key(tenant_id, doc_id):
            existing = self.store.get_document(tenant_id, doc_id)
            if existing is None or not can_modify(existing):
                return False
            removed = self.store.delete_document(tenant_id, doc_id)
            if removed is None:
                return False
            self.indexes.apply(remove_ids=removed)
            self.store.bump_generation(tenant_id)
            return True
