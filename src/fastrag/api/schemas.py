"""Request and response models. The HTTP contract lives here, separate from the domain."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field, field_validator

from fastrag.domain import Answer, DocumentRecord, ScoredChunk

MAX_METADATA_BYTES = 16_384


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

    @field_validator("metadata")
    @classmethod
    def _bounded_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        if len(json.dumps(value, default=str)) > MAX_METADATA_BYTES:
            raise ValueError(f"metadata must serialize to at most {MAX_METADATA_BYTES} bytes")
        return value

    def acl_tokens(self) -> frozenset[str]:
        return frozenset(self.acl)


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
    cached: bool
    citations: list[CitationOut]
    sources: list[SourceOut]
    usage: dict[str, int]

    @classmethod
    def of(cls, answer: Answer) -> AskResponse:
        return cls(
            answer=answer.text,
            abstained=answer.abstained,
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
