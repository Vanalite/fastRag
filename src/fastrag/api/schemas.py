"""Versioned request/response schema shared with the agent backend (D3)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "1.0"

ProfileName = Literal["customer", "handoff_copilot", "internal_search"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SearchFilters(_Strict):
    document_ids: list[str] | None = None
    metadata: dict[str, str] | None = Field(default=None, description="Exact match on document metadata.")


class SearchRequest(_Strict):
    query: str = Field(min_length=1, max_length=2000, description="A standalone question (D13).")
    profile: ProfileName = "customer"
    collections: list[str] | None = Field(
        default=None, description="Collection slugs; default: all documents collections."
    )
    groups: list[str] | None = Field(default=None, description="Caller's access groups; capped by the API key.")
    top_k: int | None = Field(default=None, ge=1, le=50)
    filters: SearchFilters | None = None


class PassageSource(BaseModel):
    document_id: str
    collection: str
    title: str
    uri: str | None
    external_id: str
    version: int
    page_start: int | None
    page_end: int | None
    modified_at: str | None
    ingested_at: str | None


class PassageScores(BaseModel):
    rerank: float | None
    similarity: float | None
    keyword_rank: int | None
    vector_rank: int | None
    fused: float
    term_match: float = 0.0


class PassageOut(BaseModel):
    id: str
    text: str
    context: str | None
    heading: str | None
    score: float | None
    scores: PassageScores
    source: PassageSource
    flags: list[str]
    age_days: int | None
    stale: bool


class SearchResponse(BaseModel):
    schema_version: str = SCHEMA_VERSION
    trace_id: str
    profile: str
    groups: list[str]
    passages: list[PassageOut]
    signals: dict[str, Any]
    timings_ms: dict[str, float]


class ListingFilters(_Strict):
    price_min: str | float | None = Field(default=None, description='e.g. "3 tỷ", "850tr", 3500000000')
    price_max: str | float | None = None
    price_per_m2_min: str | float | None = None
    price_per_m2_max: str | float | None = None
    area_min: str | float | None = Field(default=None, description='e.g. "70m2", 70')
    area_max: str | float | None = None
    bedrooms: int | str | list[int | str] | None = None
    bedrooms_min: int | str | None = None
    bedrooms_max: int | str | None = None
    bathrooms_min: int | str | None = None
    district: str | list[str] | None = Field(default=None, description='e.g. "Q7", "Quận 7", "Thủ Đức"')
    city: str | list[str] | None = None
    ward: str | list[str] | None = None
    project: str | list[str] | None = None
    property_type: str | list[str] | None = None
    status: str | list[str] | None = None
    direction: str | list[str] | None = None
    keywords: str | None = Field(default=None, max_length=300)


class ListingQueryRequest(_Strict):
    profile: ProfileName = "customer"
    collections: list[str] | None = None
    groups: list[str] | None = None
    filters: ListingFilters = Field(default_factory=ListingFilters)
    sort: str | None = Field(
        default=None, description="price_asc, price_desc, area_asc, area_desc, price_per_m2_asc, newest"
    )
    limit: int = Field(default=10, ge=1, le=50)
    offset: int = Field(default=0, ge=0, le=10_000)


class ListingLookupRequest(_Strict):
    profile: ProfileName = "customer"
    collections: list[str] | None = None
    groups: list[str] | None = None
    ids: list[str] | None = Field(default=None, max_length=50, description="Listing ids returned earlier.")
    codes: list[str] | None = Field(default=None, max_length=50, description="Listing codes from the sheet.")


class ListingsResponse(BaseModel):
    schema_version: str = SCHEMA_VERSION
    trace_id: str
    profile: str
    groups: list[str]
    items: list[dict[str, Any]]
    total: int
    as_of: str | None
    applied_filters: dict[str, Any] = {}


class FeedbackRequest(_Strict):
    trace_id: str = Field(max_length=64)
    rating: Literal["up", "down"]
    comment: str | None = Field(default=None, max_length=2000)
    expected_document_ids: list[str] | None = None
    handed_off: bool = False
