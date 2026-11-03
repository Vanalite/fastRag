"""Request and response models. The HTTP contract lives here, separate from the domain."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field, field_validator

from fastrag.domain import Answer, DocumentRecord, ScoredChunk

MAX_METADATA_BYTES = 16_384
MAX_METADATA_DEPTH = 8


def _depth(value: Any, level: int = 0) -> int:
    """Nesting depth, iteratively (a recursive walk could itself be the DoS)."""
    deepest, stack = level, [(value, level)]
    while stack:
        item, depth = stack.pop()
        deepest = max(deepest, depth)
        if deepest > MAX_METADATA_DEPTH:
            return deepest
        children = (
            item.values() if isinstance(item, dict) else item if isinstance(item, list) else ()
        )
        stack.extend((child, depth + 1) for child in children)
    return deepest


class IngestRequest(BaseModel):
    source_id: str = Field(min_length=1, max_length=1024, examples=["policies/expenses.md"])
    text: str = Field(min_length=1, max_length=2_000_000)
    acl: list[str] = Field(
        min_length=1,
        max_length=256,
        description="Who may read the document: tenant:*, user:<id> or group:<id>",
        examples=[["group:finance"]],
    )
    title: str = Field("", max_length=512)
    media_type: str = Field("text/plain", examples=["text/markdown"])
    metadata: dict[str, Any] = Field(default_factory=dict)
    editors: list[str] = Field(
        default_factory=list,
        max_length=256,
        description="Who may replace or delete it. Default: keep current editors on update, "
        "or just the uploader on create. Admins can always edit.",
    )

    @field_validator("metadata")
    @classmethod
    def _bounded_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        reserved = sorted(k for k in value if str(k).startswith("_"))
        if reserved:
            raise ValueError(f"metadata keys starting with '_' are reserved: {reserved}")
        if _depth(value) > MAX_METADATA_DEPTH:
            raise ValueError(f"metadata must be nested at most {MAX_METADATA_DEPTH} levels")
        if len(json.dumps(value, default=str)) > MAX_METADATA_BYTES:
            raise ValueError(f"metadata must serialize to at most {MAX_METADATA_BYTES} bytes")
        return value

    def acl_tokens(self) -> frozenset[str]:
        return frozenset(self.acl)


MAX_JOB_TEXT_CHARS = 20_000_000


class IngestJobRequest(BaseModel):
    documents: list[IngestRequest] = Field(min_length=1, max_length=500)

    @field_validator("documents")
    @classmethod
    def _bounded_total(cls, docs: list[IngestRequest]) -> list[IngestRequest]:
        if sum(len(d.text) for d in docs) > MAX_JOB_TEXT_CHARS:
            raise ValueError(f"a job's documents may total at most {MAX_JOB_TEXT_CHARS} chars")
        return docs


class JobItemOut(BaseModel):
    source_id: str
    status: str
    doc_id: str | None
    version: int | None
    error: str | None


class IngestJobOut(BaseModel):
    id: str
    status: str
    counts: dict[str, int]
    created_at: float
    started_at: float | None
    finished_at: float | None
    items: list[JobItemOut]

    @classmethod
    def of(cls, job: Any) -> IngestJobOut:
        return cls(
            id=job.id,
            status=job.status,
            counts=job.counts,
            created_at=job.created_at,
            started_at=job.started_at,
            finished_at=job.finished_at,
            items=[JobItemOut(**vars(i)) for i in job.items],
        )


class IngestResponse(BaseModel):
    doc_id: str
    version: int
    status: str
    chunks: int


class DocumentOut(BaseModel):
    doc_id: str
    source_id: str
    title: str
    version: int
    chunk_count: int
    acl: list[str]
    editors: list[str]
    updated_at: str

    @classmethod
    def of(cls, record: DocumentRecord) -> DocumentOut:
        return cls(
            doc_id=record.doc_id,
            source_id=record.source_id,
            title=record.title,
            version=record.version,
            chunk_count=record.chunk_count,
            acl=sorted(record.acl),
            editors=sorted(record.editors),
            updated_at=record.updated_at,
        )


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    k: int | None = Field(default=None, ge=1, le=20)


class SourceOut(BaseModel):
    chunk_id: str
    doc_id: str
    source_id: str
    title: str
    text: str
    score: float
    dense_score: float | None
    keyword_score: float | None
    rerank_score: float | None = None

    @classmethod
    def of(cls, s: ScoredChunk) -> SourceOut:
        return cls(
            chunk_id=s.chunk.id,
            doc_id=s.chunk.doc_id,
            source_id=s.chunk.source_id,
            title=s.chunk.title,
            text=s.chunk.text,
            score=s.score,
            dense_score=s.dense_score,
            keyword_score=s.keyword_score,
            rerank_score=s.rerank_score,
        )


class SearchResponse(BaseModel):
    confident: bool
    results: list[SourceOut]


class CitationOut(BaseModel):
    index: int
    doc_id: str
    source_id: str
    title: str


class AskResponse(BaseModel):
    answer: str
    abstained: bool
    grounded: bool
    cached: bool
    citations: list[CitationOut]
    sources: list[SourceOut]
    usage: dict[str, int]

    @classmethod
    def of(cls, answer: Answer) -> AskResponse:
        return cls(
            answer=answer.text,
            abstained=answer.abstained,
            grounded=answer.grounded,
            cached=answer.cached,
            citations=[
                CitationOut(index=c.index, doc_id=c.doc_id, source_id=c.source_id, title=c.title)
                for c in answer.citations
            ],
            sources=[SourceOut.of(s) for s in answer.sources],
            usage=dict(vars(answer.usage)),
        )


class PrincipalOut(BaseModel):
    tenant_id: str
    user_id: str
    groups: list[str]
    roles: list[str]
