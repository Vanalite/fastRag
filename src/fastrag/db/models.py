"""Database schema.

Every row that can be returned to a caller carries `tenant_id`, `collection_id`
and `groups` (access groups, D11/D12). Retrieval filters on these inside the
SQL query, never after it.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, ClassVar

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy import (
    text as sql,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from fastrag.config import get_settings

EMBEDDING_DIM = get_settings().embedding_dim

# Content with no explicit mapping is internal (D12).
DEFAULT_GROUPS = ["staff"]


class Base(DeclarativeBase):
    type_annotation_map: ClassVar[dict[Any, Any]] = {dict[str, Any]: JSONB, list[str]: ARRAY(Text)}


def _uuid() -> Mapped[uuid.UUID]:
    return mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)


def _created() -> Mapped[datetime]:
    return mapped_column(DateTime(timezone=True), server_default=func.now())


class Tenant(Base):
    __tablename__ = "tenants"
    id: Mapped[uuid.UUID] = _uuid()
    slug: Mapped[str] = mapped_column(String(64), unique=True)
    name: Mapped[str] = mapped_column(Text)
    settings: Mapped[dict[str, Any]] = mapped_column(JSONB, server_default=sql("'{}'::jsonb"))
    created_at: Mapped[datetime] = _created()


class ApiKey(Base):
    """Service tokens. A key is bound to one tenant and a ceiling of access
    groups: a key issued for the customer agent can never read staff content,
    whatever groups the request claims."""

    __tablename__ = "api_keys"
    id: Mapped[uuid.UUID] = _uuid()
    tenant_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tenants.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(Text)
    prefix: Mapped[str] = mapped_column(String(16), index=True)
    key_hash: Mapped[str] = mapped_column(String(128), unique=True)
    allowed_groups: Mapped[list[str]] = mapped_column(ARRAY(Text))
    allowed_profiles: Mapped[list[str] | None] = mapped_column(ARRAY(Text), nullable=True)
    is_admin: Mapped[bool] = mapped_column(Boolean, server_default=sql("false"))
    created_at: Mapped[datetime] = _created()
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class Collection(Base):
    __tablename__ = "collections"
    __table_args__ = (UniqueConstraint("tenant_id", "slug"),)
    id: Mapped[uuid.UUID] = _uuid()
    tenant_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tenants.id", ondelete="CASCADE"), index=True)
    slug: Mapped[str] = mapped_column(String(64))
    name: Mapped[str] = mapped_column(Text)
    type: Mapped[str] = mapped_column(String(16))  # "documents" | "listings"
    settings: Mapped[dict[str, Any]] = mapped_column(JSONB, server_default=sql("'{}'::jsonb"))
    embedding_model: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = _created()


class Source(Base):
    """Where content comes from: an upload area, a Drive folder, a sheet.

    `groups` is the default for everything in the source; `config.folder_groups`
    maps sub-folders to other groups (D12)."""

    __tablename__ = "sources"
    id: Mapped[uuid.UUID] = _uuid()
    tenant_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tenants.id", ondelete="CASCADE"), index=True)
    collection_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("collections.id", ondelete="CASCADE"), index=True)
    kind: Mapped[str] = mapped_column(String(32))
    name: Mapped[str] = mapped_column(Text)
    config: Mapped[dict[str, Any]] = mapped_column(JSONB, server_default=sql("'{}'::jsonb"))
    groups: Mapped[list[str]] = mapped_column(ARRAY(Text))
    sync_interval_minutes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    next_sync_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_synced_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_sync_status: Mapped[str | None] = mapped_column(Text, nullable=True)
    webhook_secret_hash: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = _created()


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (UniqueConstraint("source_id", "external_id"),)
    id: Mapped[uuid.UUID] = _uuid()
    tenant_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tenants.id", ondelete="CASCADE"), index=True)
    collection_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("collections.id", ondelete="CASCADE"), index=True)
    source_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("sources.id", ondelete="CASCADE"), index=True)
    external_id: Mapped[str] = mapped_column(Text)
    uri: Mapped[str | None] = mapped_column(Text, nullable=True)
    title: Mapped[str] = mapped_column(Text)
    mime_type: Mapped[str | None] = mapped_column(Text, nullable=True)
    content_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    version: Mapped[int] = mapped_column(Integer, server_default=sql("0"))
    groups: Mapped[list[str]] = mapped_column(ARRAY(Text))
    status: Mapped[str] = mapped_column(String(16), server_default=sql("'pending'"), index=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    storage_key: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_modified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    ingested_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    meta: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, server_default=sql("'{}'::jsonb"))
    report: Mapped[dict[str, Any]] = mapped_column(JSONB, server_default=sql("'{}'::jsonb"))
    created_at: Mapped[datetime] = _created()


class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (
        Index("ix_chunks_scope", "tenant_id", "collection_id"),
        Index(
            "ix_chunks_embedding",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    tenant_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tenants.id", ondelete="CASCADE"))
    collection_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("collections.id", ondelete="CASCADE"))
    document_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    ordinal: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    context: Mapped[str | None] = mapped_column(Text, nullable=True)
    heading: Mapped[str | None] = mapped_column(Text, nullable=True)
    page_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    page_end: Mapped[int | None] = mapped_column(Integer, nullable=True)
    token_count: Mapped[int] = mapped_column(Integer)
    groups: Mapped[list[str]] = mapped_column(ARRAY(Text))
    terms: Mapped[list[str]] = mapped_column(ARRAY(Text))
    length: Mapped[int] = mapped_column(Integer)
    embedding: Mapped[Any] = mapped_column(Vector(EMBEDDING_DIM), nullable=True)
    flags: Mapped[list[str]] = mapped_column(ARRAY(Text), server_default=sql("'{}'"))


class ChunkPosting(Base):
    """Inverted index for BM25 (D15): one row per (term, chunk). The primary
    key includes tf, so keyword search is an index-only scan."""

    __tablename__ = "chunk_postings"
    __table_args__ = (Index("ix_chunk_postings_chunk", "chunk_id"),)
    collection_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("collections.id", ondelete="CASCADE"), primary_key=True)
    term: Mapped[str] = mapped_column(Text, primary_key=True)
    chunk_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("chunks.id", ondelete="CASCADE"), primary_key=True)
    tf: Mapped[int] = mapped_column(SmallInteger)


class TermStat(Base):
    """Document frequency per keyword term and collection, for BM25 IDF."""

    __tablename__ = "term_stats"
    collection_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("collections.id", ondelete="CASCADE"), primary_key=True)
    term: Mapped[str] = mapped_column(Text, primary_key=True)
    df: Mapped[int] = mapped_column(Integer)


class CollectionStat(Base):
    __tablename__ = "collection_stats"
    collection_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("collections.id", ondelete="CASCADE"), primary_key=True)
    chunk_count: Mapped[int] = mapped_column(BigInteger, server_default=sql("0"))
    total_length: Mapped[int] = mapped_column(BigInteger, server_default=sql("0"))


class SheetMapping(Base):
    """Column mapping for one spreadsheet layout (D12). Unmapped or new columns
    are internal until an admin marks them public."""

    __tablename__ = "sheet_mappings"
    __table_args__ = (UniqueConstraint("collection_id", "layout_hash"),)
    id: Mapped[uuid.UUID] = _uuid()
    collection_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("collections.id", ondelete="CASCADE"))
    layout_hash: Mapped[str] = mapped_column(String(64))
    headers: Mapped[list[str]] = mapped_column(ARRAY(Text))
    columns: Mapped[dict[str, Any]] = mapped_column(JSONB)
    confirmed: Mapped[bool] = mapped_column(Boolean, server_default=sql("false"))
    created_at: Mapped[datetime] = _created()
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class ListingSnapshot(Base):
    __tablename__ = "listing_snapshots"
    id: Mapped[uuid.UUID] = _uuid()
    collection_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("collections.id", ondelete="CASCADE"), index=True)
    source_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("sources.id", ondelete="CASCADE"))
    sheet_key: Mapped[str] = mapped_column(Text)
    as_of: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    row_count: Mapped[int] = mapped_column(Integer)
    rejected: Mapped[list[Any]] = mapped_column(JSONB, server_default=sql("'[]'::jsonb"))
    report: Mapped[dict[str, Any]] = mapped_column(JSONB, server_default=sql("'{}'::jsonb"))
    created_at: Mapped[datetime] = _created()


class Listing(Base):
    """One spreadsheet row in typed columns (D6). `field_groups` holds the
    access groups of each field; a field the caller may not see is masked in
    SQL and cannot be filtered on."""

    __tablename__ = "listings"
    __table_args__ = (
        UniqueConstraint("collection_id", "sheet_key", "listing_code"),
        Index("ix_listings_scope", "tenant_id", "collection_id"),
        Index("ix_listings_groups", "groups", postgresql_using="gin"),
        Index("ix_listings_terms", "terms", postgresql_using="gin"),
        Index("ix_listings_price", "collection_id", "price_vnd"),
        Index("ix_listings_district", "collection_id", "district_code"),
    )
    id: Mapped[uuid.UUID] = _uuid()
    tenant_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tenants.id", ondelete="CASCADE"))
    collection_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("collections.id", ondelete="CASCADE"))
    source_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("sources.id", ondelete="CASCADE"))
    snapshot_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("listing_snapshots.id", ondelete="CASCADE"))
    sheet_key: Mapped[str] = mapped_column(Text)
    listing_code: Mapped[str] = mapped_column(Text)
    row_number: Mapped[int] = mapped_column(Integer)
    groups: Mapped[list[str]] = mapped_column(ARRAY(Text))
    field_groups: Mapped[dict[str, Any]] = mapped_column(JSONB)
    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    project: Mapped[str | None] = mapped_column(Text, nullable=True)
    project_slug: Mapped[str | None] = mapped_column(Text, nullable=True)
    property_type: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str | None] = mapped_column(Text, nullable=True)
    price_vnd: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    price_per_m2_vnd: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    area_m2: Mapped[float | None] = mapped_column(Numeric(12, 2), nullable=True)
    bedrooms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    bathrooms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    direction: Mapped[str | None] = mapped_column(String(4), nullable=True)
    city: Mapped[str | None] = mapped_column(Text, nullable=True)
    district_code: Mapped[str | None] = mapped_column(Text, nullable=True)
    ward: Mapped[str | None] = mapped_column(Text, nullable=True)
    address: Mapped[str | None] = mapped_column(Text, nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    attributes: Mapped[dict[str, Any]] = mapped_column(JSONB, server_default=sql("'{}'::jsonb"))
    terms: Mapped[list[str]] = mapped_column(ARRAY(Text), server_default=sql("'{}'"))
    as_of: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    flags: Mapped[list[str]] = mapped_column(ARRAY(Text), server_default=sql("'{}'"))


class Job(Base):
    __tablename__ = "jobs"
    __table_args__ = (Index("ix_jobs_queue", "status", "run_after"),)
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    tenant_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("tenants.id", ondelete="CASCADE"), nullable=True)
    kind: Mapped[str] = mapped_column(String(32))
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB)
    dedupe_key: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(16), server_default=sql("'queued'"))
    attempts: Mapped[int] = mapped_column(Integer, server_default=sql("0"))
    max_attempts: Mapped[int] = mapped_column(Integer, server_default=sql("3"))
    run_after: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    locked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    locked_by: Mapped[str | None] = mapped_column(Text, nullable=True)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    result: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = _created()
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


# At most one queued job per dedupe key (e.g. one pending sync per source).
Index(
    "uq_jobs_dedupe_active",
    Job.dedupe_key,
    unique=True,
    postgresql_where=sql("status = 'queued' AND dedupe_key IS NOT NULL"),
)


class SearchLog(Base):
    """One row per retrieval call: the source for growing the reference
    question set from hand-offs and thumbs-down feedback (Q12)."""

    __tablename__ = "search_logs"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    trace_id: Mapped[str] = mapped_column(String(64), index=True)
    tenant_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tenants.id", ondelete="CASCADE"), index=True)
    endpoint: Mapped[str] = mapped_column(String(32))
    profile: Mapped[str | None] = mapped_column(String(32), nullable=True)
    request: Mapped[dict[str, Any]] = mapped_column(JSONB)
    result_ids: Mapped[list[str]] = mapped_column(ARRAY(Text))
    signals: Mapped[dict[str, Any]] = mapped_column(JSONB, server_default=sql("'{}'::jsonb"))
    latency_ms: Mapped[int] = mapped_column(Integer)
    feedback: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
