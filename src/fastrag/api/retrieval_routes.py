"""Retrieval endpoints used by the agent backend."""

from __future__ import annotations

import time
import uuid
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException, Request
from sqlalchemy import update

from fastrag import metrics
from fastrag.api.auth import Principal, get_principal, resolve_tenant, trace_id
from fastrag.api.common import collections_for, profile_and_groups
from fastrag.api.schemas import (
    FeedbackRequest,
    ListingLookupRequest,
    ListingQueryRequest,
    ListingsResponse,
    PassageOut,
    PassageScores,
    PassageSource,
    SearchResponse,
)
from fastrag.api.schemas import SearchRequest as SearchBody
from fastrag.db.engine import get_sessionmaker, session_scope
from fastrag.db.models import SearchLog
from fastrag.retrieval import listings as L
from fastrag.retrieval.search import SearchRequest, search
from fastrag.retrieval.signals import staleness_days
from fastrag.services import Services

router = APIRouter(prefix="/v1", tags=["retrieval"])


def _services(request: Request) -> Services:
    return request.app.state.services  # type: ignore[no-any-return]


async def _log(entry: dict[str, Any]) -> None:
    try:
        async with session_scope() as s:
            s.add(SearchLog(**entry))
    except Exception:  # logging must never break retrieval
        import logging

        logging.getLogger(__name__).exception("search log write failed")


def _uuids(values: list[str] | None, what: str) -> list[uuid.UUID] | None:
    if not values:
        return None
    try:
        return [uuid.UUID(v) for v in values]
    except ValueError as exc:
        raise HTTPException(422, f"invalid {what}") from exc


@router.post("/search", response_model=SearchResponse, response_model_exclude_none=False)
async def search_documents(
    body: SearchBody,
    request: Request,
    background: BackgroundTasks,
    principal: Principal = Depends(get_principal),
    x_tenant: str | None = Header(default=None),
) -> SearchResponse:
    """Hybrid search over document collections. Returns passages with
    citations and evidence signals; the caller writes the answer."""
    t0 = time.perf_counter()
    services = _services(request)
    tid = trace_id(request)
    async with get_sessionmaker()() as session:
        tenant_id = await resolve_tenant(session, principal, x_tenant)
        cols = await collections_for(session, tenant_id, body.collections, "documents")
    profile, groups = profile_and_groups(principal, body.profile, body.groups, cols)
    slug_by_id = {c.id: c.slug for c in cols}
    req = SearchRequest(
        tenant_id=tenant_id,
        collection_ids=[c.id for c in cols],
        query=body.query,
        groups=groups,
        profile=profile,
        top_k=body.top_k,
        document_ids=_uuids(body.filters.document_ids if body.filters else None, "document id"),
        metadata=body.filters.metadata if body.filters else None,
    )
    result = await search(get_sessionmaker(), services, req)
    passages = []
    for p in result.passages:
        age = staleness_days(p.modified_at or p.ingested_at)
        passages.append(
            PassageOut(
                id=f"chunk:{p.chunk_id}",
                text=p.text,
                context=p.context,
                heading=p.heading,
                score=p.score,
                scores=PassageScores(
                    rerank=p.rerank_score,
                    similarity=round(p.similarity, 4) if p.similarity is not None else None,
                    keyword_rank=p.keyword_rank,
                    vector_rank=p.vector_rank,
                    fused=round(p.fused_score, 5),
                    term_match=p.term_match,
                ),
                source=PassageSource(
                    document_id=str(p.document_id),
                    collection=slug_by_id.get(p.collection_id, ""),
                    title=p.title,
                    uri=p.uri,
                    external_id=p.external_id,
                    version=p.version,
                    page_start=p.page_start,
                    page_end=p.page_end,
                    modified_at=p.modified_at.isoformat() if p.modified_at else None,
                    ingested_at=p.ingested_at.isoformat() if p.ingested_at else None,
                ),
                flags=p.flags,
                age_days=age,
                stale=age is not None and age > profile.staleness_days,
            )
        )
    latency = int((time.perf_counter() - t0) * 1000)
    metrics.observe_search("search", body.profile, result.signals, result.timings_ms)
    if services.settings.search_log_enabled:
        background.add_task(
            _log,
            {
                "trace_id": tid,
                "tenant_id": tenant_id,
                "endpoint": "search",
                "profile": body.profile,
                "request": body.model_dump(),
                "result_ids": [p.id for p in passages],
                "signals": result.signals,
                "latency_ms": latency,
            },
        )
    return SearchResponse(
        trace_id=tid,
        profile=body.profile,
        groups=groups,
        passages=passages,
        signals=result.signals,
        timings_ms=result.timings_ms,
    )


async def _run_listings(
    request: Request,
    principal: Principal,
    x_tenant: str | None,
    body: Any,
    q_kwargs: dict[str, Any],
    background: BackgroundTasks,
    endpoint: str,
) -> ListingsResponse:
    t0 = time.perf_counter()
    services = _services(request)
    tid = trace_id(request)
    async with get_sessionmaker()() as session:
        tenant_id = await resolve_tenant(session, principal, x_tenant)
        cols = await collections_for(session, tenant_id, body.collections, "listings")
        _, groups = profile_and_groups(principal, body.profile, body.groups, cols)
        gaz = services.gazetteer
        aliases = (cols[0].settings or {}).get("gazetteer") if cols else None
        if aliases:
            gaz = gaz.with_aliases(aliases)
        q = L.ListingQuery(tenant_id=tenant_id, collection_ids=[c.id for c in cols], groups=groups, **q_kwargs)
        if not groups or not cols:
            result: dict[str, Any] = {"items": [], "total": 0, "as_of": None, "applied_filters": {}}
        else:
            try:
                result = await L.query_listings(session, q, gaz, services.settings.segmenter)
            except L.FilterError as exc:
                raise HTTPException(
                    422, {"error": "invalid_filters", "rejected": exc.rejected, "trace_id": tid}
                ) from exc
    metrics.RETRIEVAL_RESULTS.labels(endpoint, body.profile, "answered" if result["items"] else "empty").inc()
    if services.settings.search_log_enabled:
        background.add_task(
            _log,
            {
                "trace_id": tid,
                "tenant_id": tenant_id,
                "endpoint": endpoint,
                "profile": body.profile,
                "request": body.model_dump(),
                "result_ids": [i["id"] for i in result["items"]],
                "signals": {"total": result["total"]},
                "latency_ms": int((time.perf_counter() - t0) * 1000),
            },
        )
    return ListingsResponse(trace_id=tid, profile=body.profile, groups=groups, **result)


@router.post("/listings/query", response_model=ListingsResponse)
async def query_listings(
    body: ListingQueryRequest,
    request: Request,
    background: BackgroundTasks,
    principal: Principal = Depends(get_principal),
    x_tenant: str | None = Header(default=None),
) -> ListingsResponse:
    """Typed listing search with predefined filters (D6). Values such as
    "3,5 tỷ" or "Q7" are standardized; unreadable filters are rejected (422)."""
    return await _run_listings(
        request,
        principal,
        x_tenant,
        body,
        {
            "filters": body.filters.model_dump(exclude_none=True),
            "sort": body.sort,
            "limit": body.limit,
            "offset": body.offset,
        },
        background,
        "listings_query",
    )


@router.post("/listings/lookup", response_model=ListingsResponse)
async def lookup_listings(
    body: ListingLookupRequest,
    request: Request,
    background: BackgroundTasks,
    principal: Principal = Depends(get_principal),
    x_tenant: str | None = Header(default=None),
) -> ListingsResponse:
    """Exact lookup by listing id or code, with a fresh `as_of` (D13)."""
    if not body.ids and not body.codes:
        raise HTTPException(422, "give ids or codes")
    return await _run_listings(
        request,
        principal,
        x_tenant,
        body,
        {"filters": {}, "ids": _uuids(body.ids, "listing id"), "codes": body.codes, "limit": 50},
        background,
        "listings_lookup",
    )


@router.post("/feedback", status_code=204)
async def feedback(
    body: FeedbackRequest,
    principal: Principal = Depends(get_principal),
    x_tenant: str | None = Header(default=None),
) -> None:
    """Thumbs up/down and hand-offs, keyed by trace id. Thumbs-down and
    hand-off cases become candidate reference questions (Q12)."""
    async with session_scope() as session:
        tenant_id = await resolve_tenant(session, principal, x_tenant)
        res = await session.execute(
            update(SearchLog)
            .where(SearchLog.trace_id == body.trace_id, SearchLog.tenant_id == tenant_id)
            .values(feedback=body.model_dump(exclude={"trace_id"}))
        )
        if not res.rowcount:  # type: ignore[attr-defined]
            raise HTTPException(404, "trace id not found")


@router.get("/tools")
async def tool_schemas() -> dict[str, Any]:
    """Tool definitions in JSON Schema, for agent platforms that take
    function-calling specs (OpenAI, Anthropic, LangChain, etc.)."""
    return {
        "schema_version": "1.0",
        "tools": [
            {
                "name": "search_documents",
                "description": (
                    "Search the company's documents (policies, project information, FAQs). Pass a standalone "
                    "question written in the documents' language (Vietnamese for Vietnamese content; translate "
                    "other languages first). Returns passages with citations and evidence signals; if "
                    "signals.nothing_relevant is true, do not answer from these passages."
                ),
                "endpoint": "POST /v1/search",
                "input_schema": SearchBody.model_json_schema(),
            },
            {
                "name": "query_listings",
                "description": (
                    "Find property listings with filters (price, area, district, bedrooms, project, status). "
                    "Values like '3,5 tỷ' or 'Q7' are accepted. Each result has an id and as_of; quote as_of "
                    "when stating availability or price."
                ),
                "endpoint": "POST /v1/listings/query",
                "input_schema": ListingQueryRequest.model_json_schema(),
            },
            {
                "name": "get_listings",
                "description": "Fetch listings by the ids returned earlier, to refresh price and status.",
                "endpoint": "POST /v1/listings/lookup",
                "input_schema": ListingLookupRequest.model_json_schema(),
            },
        ],
    }
