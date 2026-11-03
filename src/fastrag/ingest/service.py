"""Ingest and delete documents, keeping the store and every index in step."""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from fastrag.domain import (
    Chunk,
    DocumentIn,
    DocumentRecord,
    IngestResult,
    document_id,
    is_valid_acl_token,
)
from fastrag.embeddings import Embedder
from fastrag.errors import ConcurrentModification, InvalidInput, PermissionDenied
from fastrag.ingest.chunking import chunk_text
from fastrag.ingest.parsers import parse
from fastrag.ingest.redact import Redactor
from fastrag.storage.base import DocumentStore
from fastrag.storage.indexes import SearchIndexes

# Bump when parsing, redaction or chunking changes, so re-ingesting unchanged content
# still re-processes it instead of short-circuiting as "unchanged".
PIPELINE_VERSION = "3"
MAX_WRITE_ATTEMPTS = 3

log = logging.getLogger(__name__)


MAX_METADATA_DEPTH = 8


def _redact_values(value: Any, redactor: Redactor, counts: Counter[str], depth: int = 0) -> Any:
    """Redact every string inside a metadata structure, dictionary keys included."""
    if depth > MAX_METADATA_DEPTH:
        raise InvalidInput(f"metadata is nested deeper than {MAX_METADATA_DEPTH} levels")
    if isinstance(value, str):
        result = redactor.redact(value)
        counts.update(result.counts)
        return result.text
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            new_key = _redact_values(str(key), redactor, counts, depth + 1)
            while new_key in out:  # two keys can redact to the same placeholder
                new_key += "_"
            out[new_key] = _redact_values(item, redactor, counts, depth + 1)
        return out
    if isinstance(value, list):
        return [_redact_values(v, redactor, counts, depth + 1) for v in value]
    return value


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
        default_editors: frozenset[str] = frozenset(),
        can_change_sharing: Callable[[DocumentRecord], bool] = lambda _: True,
    ) -> tuple[IngestResult, dict[str, int]]:
        """Create or replace a document. Returns the result and PII redaction counts.

        ``can_modify`` is checked against the existing version, inside the per-document
        lock, before anything else is revealed about it. ``default_editors`` (normally
        the uploader) applies when a new document is submitted without editors.
        """
        if not doc.source_id.strip():
            raise InvalidInput("source_id must not be empty")
        if not doc.acl:
            raise InvalidInput("acl must list at least one principal (deny-by-default)")
        bad = sorted(t for t in doc.acl | doc.editors if not is_valid_acl_token(t))
        if bad:
            raise InvalidInput(f"invalid ACL tokens {bad}; use tenant:*, user:<id> or group:<id>")
        if len(doc.text) > self.max_document_chars:
            raise InvalidInput(f"document exceeds {self.max_document_chars} characters")

        parsed = parse(doc.text, doc.media_type)
        title, text = doc.title or parsed.title, parsed.text
        metadata = dict(doc.metadata)
        redactions: dict[str, int] = {}
        if self.redactor is not None:
            counts: Counter[str] = Counter()
            result = self.redactor.redact(text)
            text = result.text
            counts.update(result.counts)
            title_result = self.redactor.redact(title)
            title = title_result.text
            counts.update(title_result.counts)
            metadata = _redact_values(metadata, self.redactor, counts)
            redactions = dict(counts)
        if not text.strip():
            raise InvalidInput("document has no text content")

        doc_id = document_id(tenant_id, doc.source_id)

        # The per-process lock serializes writers here; the expected version makes the
        # write safe against other processes and replicas too. If the document changed
        # since we read it, start over with fresh state (permissions included).
        for attempt in range(MAX_WRITE_ATTEMPTS):
            try:
                with self._locks.for_key(tenant_id, doc_id):
                    existing = self.store.get_document(tenant_id, doc_id)
                    if existing is not None and not can_modify(existing):
                        raise PermissionDenied("not allowed to modify this document")
                    # Changing *who* may read or edit an existing document is a separate right
                    # from editing its content: otherwise any editor could publish a restricted
                    # document to the whole tenant by re-submitting it with a wider ACL.
                    if existing is not None and not can_change_sharing(existing):
                        widened_editors = bool(doc.editors) and doc.editors != existing.editors
                        if doc.acl != existing.acl or widened_editors:
                            raise PermissionDenied(
                                "only an admin can change who may read or edit this document"
                            )
                    # Explicit editors win; otherwise an update keeps the current editors (even
                    # none, which means admin-only) and a new document gets the uploader.
                    editors = doc.editors or (
                        existing.editors if existing is not None else default_editors
                    )
                    if existing is not None:
                        # Reserved keys ("_connector", ...) belong to the system and API
                        # callers can't send them: keep them so ownership isn't lost.
                        reserved = {k: v for k, v in existing.metadata.items() if k.startswith("_")}
                        metadata = {**reserved, **metadata}
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
                                sorted(editors),
                                metadata,
                                self.chunk_size,
                                self.chunk_overlap,
                            ],
                            sort_keys=True,
                            default=str,
                        ).encode()
                    ).hexdigest()
                    if existing is not None and existing.content_hash == content_hash:
                        return IngestResult(
                            doc_id, existing.version, "unchanged", existing.chunk_count
                        ), {}

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
                        metadata=metadata,
                        chunk_count=len(chunks),
                        updated_at=datetime.now(UTC).isoformat(timespec="seconds"),
                        editors=editors,
                    )
                    # The store commit is the point of truth. If the process dies before the
                    # index update, the indexes are rebuilt from the store on next start.
                    replaced = self.store.replace_document(
                        record,
                        chunks,
                        vectors,
                        expected_version=existing.version if existing is not None else None,
                    )
                    self.indexes.apply(remove_ids=replaced, add_chunks=chunks, add_vectors=vectors)
                    # Bump only once the indexes serve the new version: a cache entry keyed on the
                    # new generation can then never hold an answer built from the old index.
                    self.store.bump_generation(tenant_id)
                break
            except ConcurrentModification:
                if attempt == MAX_WRITE_ATTEMPTS - 1:
                    raise
                log.info("document changed concurrently; retrying", extra={"doc_id": doc_id})

        status = "updated" if existing else "created"
        return IngestResult(doc_id, version, status, len(chunks)), redactions

    def delete(
        self,
        tenant_id: str,
        doc_id: str,
        can_modify: Callable[[DocumentRecord], bool] = lambda _: True,
    ) -> bool:
        """Delete a document everywhere. Returns False if absent or not modifiable."""
        for attempt in range(MAX_WRITE_ATTEMPTS):
            try:
                with self._locks.for_key(tenant_id, doc_id):
                    existing = self.store.get_document(tenant_id, doc_id)
                    if existing is None or not can_modify(existing):
                        return False
                    removed = self.store.delete_document(
                        tenant_id, doc_id, expected_version=existing.version
                    )
                    if removed is None:
                        return False
                    self.indexes.apply(remove_ids=removed)
                    self.store.bump_generation(tenant_id)
                    return True
            except ConcurrentModification:
                # Re-check permissions against the new version before deleting it.
                if attempt == MAX_WRITE_ATTEMPTS - 1:
                    raise
        return False
