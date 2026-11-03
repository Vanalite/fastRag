"""Domain model shared by every layer.

Two ideas matter most here:

* A ``Principal`` is *who* is asking: tenant, user, groups and roles. Every service
  call takes one; there is no "system" path that skips it.
* A ``Chunk`` carries its own ``tenant_id`` and ``acl``, copied from its document at
  ingest time, so every index can enforce access without a lookup elsewhere.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

# ACL tokens. A document's ACL lists who may read it; a principal holds the tokens it
# matches. Access is granted when the tenant matches and the two sets intersect.
EVERYONE = "tenant:*"
ACL_TOKEN_PATTERN = re.compile(r"^(tenant:\*|user:[\w.@+-]{1,128}|group:[\w.@+-]{1,128})$")


def is_valid_acl_token(token: str) -> bool:
    return ACL_TOKEN_PATTERN.fullmatch(token) is not None


def user_token(user_id: str) -> str:
    return f"user:{user_id}"


def group_token(group: str) -> str:
    return f"group:{group}"


class Role(StrEnum):
    READER = "reader"
    INGESTER = "ingester"
    ADMIN = "admin"


@dataclass(frozen=True)
class Principal:
    tenant_id: str
    user_id: str
    groups: frozenset[str] = frozenset()
    roles: frozenset[Role] = frozenset({Role.READER})

    def acl_tokens(self) -> frozenset[str]:
        return frozenset(
            {EVERYONE, user_token(self.user_id)} | {group_token(g) for g in self.groups}
        )


@dataclass(frozen=True)
class DocumentIn:
    """A document as submitted for ingestion.

    ``source_id`` is the caller's stable identifier (a path, URL or record id). Re-ingesting
    the same ``source_id`` creates a new version that replaces the old one.
    """

    source_id: str
    text: str
    acl: frozenset[str]
    title: str = ""
    media_type: str = "text/plain"
    metadata: dict[str, Any] = field(default_factory=dict)


def document_id(tenant_id: str, source_id: str) -> str:
    """Deterministic id, scoped to the tenant so equal source ids never collide across tenants."""
    return hashlib.sha256(f"{tenant_id}\0{source_id}".encode()).hexdigest()[:24]


@dataclass(frozen=True)
class DocumentRecord:
    tenant_id: str
    doc_id: str
    source_id: str
    title: str
    version: int
    content_hash: str
    acl: frozenset[str]
    metadata: dict[str, Any]
    chunk_count: int
    updated_at: str


@dataclass(frozen=True)
class Chunk:
    id: str
    tenant_id: str
    doc_id: str
    version: int
    ordinal: int
    text: str
    acl: frozenset[str]
    source_id: str
    title: str = ""


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float  # fused (RRF) score: rank-based, only meaningful for ordering
    dense_score: float | None = None  # cosine similarity, if the dense retriever found it
    keyword_score: float | None = None  # BM25 score, if the keyword retriever found it
    keyword_coverage: float | None = None  # share of query terms present in the chunk
    keyword_matches: int | None = None  # how many distinct query terms matched


@dataclass(frozen=True)
class RetrievalResult:
    chunks: list[ScoredChunk]
    confident: bool  # False -> evidence too weak; the service abstains instead of generating


@dataclass(frozen=True)
class Citation:
    index: int
    chunk_id: str
    doc_id: str
    source_id: str
    title: str


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


@dataclass(frozen=True)
class Answer:
    text: str
    citations: list[Citation]
    sources: list[ScoredChunk]
    abstained: bool = False
    cached: bool = False
    usage: Usage = field(default_factory=Usage)


@dataclass(frozen=True)
class IngestResult:
    doc_id: str
    version: int
    status: str  # "created" | "updated" | "unchanged"
    chunks: int
