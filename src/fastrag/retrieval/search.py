"""Hybrid document retrieval (D1, D3, D11).

1. Keyword search: BM25 over Vietnamese-segmented terms, computed in SQL.
2. Vector search: pgvector HNSW, cosine.
3. Reciprocal rank fusion of the two lists.
4. Rerank the fused top candidates (falls back to fused order if the
   reranker is slow or down).
5. Evidence signals.

Tenant, collection, access groups and quarantine flags are all conditions of
the SQL queries, so excluded content is never fetched.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from fastrag.retrieval.profiles import Profile
from fastrag.retrieval.signals import PassageEvidence, conflicts, coverage, staleness_days, term_match
from fastrag.services import Services
from fastrag.text.vietnamese import ACCENTLESS_PREFIX, is_accentless, normalize_query, query_terms, query_words

logger = logging.getLogger(__name__)

BM25_K1 = 1.2
BM25_B = 0.75


@dataclass
class SearchRequest:
    tenant_id: uuid.UUID
    collection_ids: list[uuid.UUID]
    query: str
    groups: list[str]
    profile: Profile
    top_k: int | None = None
    document_ids: list[uuid.UUID] | None = None
    metadata: dict[str, str] | None = None


@dataclass
class Passage:
    chunk_id: int
    document_id: uuid.UUID
    collection_id: uuid.UUID
    text: str
    context: str | None
    heading: str | None
    page_start: int | None
    page_end: int | None
    groups: list[str]
    flags: list[str]
    title: str
    uri: str | None
    external_id: str
    version: int
    modified_at: datetime | None
    ingested_at: datetime | None
    terms: set[str]
    keyword_rank: int | None = None
    keyword_score: float | None = None
    vector_rank: int | None = None
    similarity: float | None = None
    fused_score: float = 0.0
    rerank_score: float | None = None
    term_match: float = 0.0

    @property
    def score(self) -> float | None:
        """Reranker score when available. Otherwise the stronger of vector
        similarity and the share of question words in the passage, so a
        clear keyword match survives a reranker outage."""
        if self.rerank_score is not None:
            return self.rerank_score
        return max(self.similarity or 0.0, self.term_match)


@dataclass
class SearchResult:
    passages: list[Passage]
    signals: dict[str, Any]
    timings_ms: dict[str, float] = field(default_factory=dict)


def _filters(req: SearchRequest, alias: str = "c") -> tuple[str, dict[str, Any]]:
    clauses = [
        f"{alias}.tenant_id = :tenant_id",
        f"{alias}.collection_id = ANY(:collection_ids)",
        f"{alias}.groups && CAST(:groups AS text[])",
    ]
    params: dict[str, Any] = {
        "tenant_id": req.tenant_id,
        "collection_ids": req.collection_ids,
        "groups": req.groups,
    }
    if req.profile.exclude_flags:
        clauses.append(f"NOT ({alias}.flags && CAST(:exclude_flags AS text[]))")
        params["exclude_flags"] = req.profile.exclude_flags
    if req.document_ids:
        clauses.append(f"{alias}.document_id = ANY(:document_ids)")
        params["document_ids"] = req.document_ids
    if req.metadata:
        clauses.append(f"{alias}.document_id IN (SELECT id FROM documents WHERE metadata @> CAST(:meta AS jsonb))")
        params["meta"] = json.dumps(req.metadata)
    return " AND ".join(clauses), params


async def _term_stats(session: AsyncSession, req: SearchRequest, terms: list[str]) -> tuple[dict[str, int], int, float]:
    stats = (
        await session.execute(
            text(
                "SELECT COALESCE(SUM(chunk_count), 0) AS n, COALESCE(SUM(total_length), 0) AS len "
                "FROM collection_stats WHERE collection_id = ANY(:cids)"
            ),
            {"cids": req.collection_ids},
        )
    ).one()
    n, total = int(stats.n), int(stats.len)
    if n == 0:
        return {}, 0, 1.0
    rows = (
        await session.execute(
            text(
                "SELECT term, SUM(df) AS df FROM term_stats "
                "WHERE collection_id = ANY(:cids) AND term = ANY(:terms) GROUP BY term"
            ),
            {"cids": req.collection_ids, "terms": terms},
        )
    ).all()
    return {r.term: int(r.df) for r in rows}, n, total / n


def select_terms(
    qterms: dict[str, int], dfs: dict[str, int], n: int, budget: int, accentless_weight: float
) -> tuple[list[str], list[float], int]:
    """Terms to score, rarest first, within the postings budget. Returns
    (terms, BM25 query weights, number of terms skipped)."""
    present = sorted((t for t in qterms if t in dfs), key=lambda t: dfs[t])
    chosen: list[str] = []
    used = 0
    for t in present:
        if chosen and used + dfs[t] > budget:
            continue
        chosen.append(t)
        used += dfs[t]
    weights = []
    for t in chosen:
        idf = math.log(1 + (n - dfs[t] + 0.5) / (dfs[t] + 0.5))
        w = idf * (accentless_weight if t.startswith(ACCENTLESS_PREFIX) else 1.0)
        # A word repeated in the question counts a little more, not linearly.
        weights.append(w * (1 + math.log(qterms[t])))
    return chosen, weights, len(present) - len(chosen)


async def keyword_search(
    session: AsyncSession, req: SearchRequest, qterms: dict[str, int], limit: int
) -> list[tuple[int, float]]:
    dfs, n, avgdl = await _term_stats(session, req, list(qterms))
    terms, weights, _ = select_terms(qterms, dfs, n, req.profile.keyword_postings_budget, req.profile.accentless_weight)
    if not terms:
        return []
    where, params = _filters(req)
    sql = f"""
        WITH q AS (
            SELECT * FROM unnest(CAST(:terms AS text[]), CAST(:weights AS float8[])) AS q(term, w)
        )
        SELECT p.chunk_id AS id, SUM(
            q.w * p.tf * {BM25_K1 + 1}
            / (p.tf + {BM25_K1} * (1 - {BM25_B} + {BM25_B} * c.length / :avgdl))
        ) AS score
        FROM q
        JOIN chunk_postings p ON p.collection_id = ANY(:collection_ids) AND p.term = q.term
        JOIN chunks c ON c.id = p.chunk_id
        WHERE {where}
        GROUP BY p.chunk_id
        ORDER BY score DESC
        LIMIT :limit
    """
    params |= {"terms": terms, "weights": weights, "avgdl": avgdl, "limit": limit}
    rows = (await session.execute(text(sql), params)).all()
    return [(int(r.id), float(r.score)) for r in rows]


async def vector_search(
    session: AsyncSession, req: SearchRequest, vector: list[float], limit: int
) -> list[tuple[int, float]]:
    where, params = _filters(req)
    # Filtered HNSW: keep scanning until enough rows pass the filters (pgvector 0.8).
    await session.execute(text(f"SET LOCAL hnsw.ef_search = {int(req.profile.ef_search)}"))
    await session.execute(text("SET LOCAL hnsw.iterative_scan = relaxed_order"))
    # The CTE parses the query vector once instead of once per returned row.
    sql = f"""
        WITH q AS (SELECT CAST(:qv AS vector) AS v)
        SELECT c.id, 1 - (c.embedding <=> q.v) AS sim
        FROM chunks c, q
        WHERE {where} AND c.embedding IS NOT NULL
        ORDER BY c.embedding <=> q.v
        LIMIT :limit
    """
    params |= {"qv": "[" + ",".join(f"{x:.7f}" for x in vector) + "]", "limit": limit}
    rows = (await session.execute(text(sql), params)).all()
    out = [(int(r.id), float(r.sim)) for r in rows]
    out.sort(key=lambda x: -x[1])  # relaxed_order may return rows slightly out of order
    return out


def rrf(lists: list[list[int]], k: int = 60) -> dict[int, float]:
    scores: dict[int, float] = {}
    for ranked in lists:
        for rank, cid in enumerate(ranked, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return scores


async def _load_passages(session: AsyncSession, req: SearchRequest, ids: list[int]) -> dict[int, Passage]:
    if not ids:
        return {}
    where, params = _filters(req)
    rows = (
        await session.execute(
            text(
                f"""
                SELECT c.id, c.document_id, c.collection_id, c.text, c.context, c.heading, c.page_start,
                       c.page_end, c.groups, c.flags, c.terms, d.title, d.uri, d.external_id, d.version,
                       d.source_modified_at, d.ingested_at
                FROM chunks c JOIN documents d ON d.id = c.document_id
                WHERE {where} AND c.id = ANY(:ids)
                """
            ),
            params | {"ids": ids},
        )
    ).all()
    return {
        int(r.id): Passage(
            chunk_id=int(r.id),
            document_id=r.document_id,
            collection_id=r.collection_id,
            text=r.text,
            context=r.context,
            heading=r.heading,
            page_start=r.page_start,
            page_end=r.page_end,
            groups=list(r.groups),
            flags=list(r.flags),
            title=r.title,
            uri=r.uri,
            external_id=r.external_id,
            version=r.version,
            modified_at=r.source_modified_at,
            ingested_at=r.ingested_at,
            terms=set(r.terms),
        )
        for r in rows
    }


async def search(
    sessionmaker: async_sessionmaker[AsyncSession], services: Services, req: SearchRequest
) -> SearchResult:
    t0 = time.perf_counter()
    timings: dict[str, float] = {}
    prof = req.profile
    top_k = min(req.top_k or prof.top_k, 50)
    q = normalize_query(req.query)
    aliases = ["@" + c for c in services.gazetteer.find_in_text(q)]
    qterms = query_terms(q, services.settings.segmenter, aliases)
    words = []
    for w in query_words(q, services.settings.segmenter):
        codes = services.gazetteer.find_in_text(w.replace("_", " "))
        words.append("@" + codes[0] if codes else w)

    if not req.groups or not req.collection_ids:
        return SearchResult([], _empty_signals("no_access" if not req.groups else "no_collections"), timings)

    async def run_keyword() -> list[tuple[int, float]]:
        t = time.perf_counter()
        async with sessionmaker() as s:
            out = await keyword_search(s, req, qterms, prof.keyword_candidates)
        timings["keyword_ms"] = round((time.perf_counter() - t) * 1000, 1)
        return out

    async def run_vector() -> list[tuple[int, float]]:
        t = time.perf_counter()
        vec = await services.embedder.embed_query(q)
        timings["embed_ms"] = round((time.perf_counter() - t) * 1000, 1)
        t = time.perf_counter()
        async with sessionmaker() as s, s.begin():
            out = await vector_search(s, req, vec, prof.vector_candidates)
        timings["vector_ms"] = round((time.perf_counter() - t) * 1000, 1)
        return out

    kw_res, vec_res = await asyncio.gather(run_keyword(), run_vector(), return_exceptions=True)
    degraded: list[str] = []
    if isinstance(kw_res, BaseException):
        logger.error("keyword search failed: %s", kw_res)
        degraded.append("keyword_search_failed")
        kw_res = []
    if isinstance(vec_res, BaseException):
        logger.error("vector search failed: %s", vec_res)
        degraded.append("vector_search_failed")
        vec_res = []
    if len(degraded) == 2:
        raise RuntimeError("both keyword and vector search failed")

    kw_ids = [cid for cid, _ in kw_res]
    vec_ids = [cid for cid, _ in vec_res]
    fused = rrf([kw_ids, vec_ids], prof.rrf_k)
    order = sorted(fused, key=lambda c: -fused[c])[: max(prof.rerank_candidates, top_k)]

    async with sessionmaker() as s:
        loaded = await _load_passages(s, req, order)
    kw_rank = {cid: i for i, cid in enumerate(kw_ids, start=1)}
    kw_score = dict(kw_res)
    vec_rank = {cid: i for i, cid in enumerate(vec_ids, start=1)}
    sims = dict(vec_res)
    candidates: list[Passage] = []
    for cid in order:
        p = loaded.get(cid)
        if p is None:
            continue
        p.keyword_rank, p.keyword_score = kw_rank.get(cid), kw_score.get(cid)
        p.vector_rank, p.similarity = vec_rank.get(cid), sims.get(cid)
        p.fused_score = fused[cid]
        p.term_match = round(term_match(words, p.terms), 3)
        candidates.append(p)

    reranker_state = "disabled"
    if services.reranker and candidates:
        t = time.perf_counter()
        try:
            scores = await asyncio.wait_for(
                services.reranker.rerank(q, [(p.context or "") + "\n" + p.text for p in candidates]),
                timeout=services.settings.reranker_timeout_s,
            )
            for p, sc in zip(candidates, scores, strict=True):
                p.rerank_score = round(sc, 4)
            candidates.sort(key=lambda p: -(p.rerank_score or 0.0))
            reranker_state = "ok"
        except Exception as exc:
            logger.warning("reranker unavailable, using fused ranking: %s", exc)
            reranker_state = "unavailable"
            degraded.append("reranker_unavailable")
        timings["rerank_ms"] = round((time.perf_counter() - t) * 1000, 1)

    # Rerankers give low absolute scores to questions typed without diacritics
    # even when they rank the right passage first, so for those questions a
    # clear word match also counts as evidence (D17).
    accentless = is_accentless(q)
    reranked = reranker_state == "ok"

    def keep(p: Passage) -> bool:
        if reranked:
            return (p.rerank_score or 0.0) >= prof.min_score or (accentless and p.term_match >= prof.min_similarity)
        return (p.score or 0.0) >= prof.min_similarity

    def relevant(p: Passage) -> bool:
        if reranked:
            return (p.rerank_score or 0.0) >= prof.relevance_threshold or (
                accentless and p.term_match >= prof.min_similarity + 0.1
            )
        return (p.score or 0.0) >= prof.min_similarity + 0.1

    passages = [p for p in candidates if keep(p)][:top_k]
    timings["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    signals = _signals(words, passages, prof, reranker_state, any(relevant(p) for p in passages), degraded)
    signals["accentless_query"] = accentless
    return SearchResult(passages, signals, timings)


def _empty_signals(reason: str) -> dict[str, Any]:
    return {
        "nothing_relevant": True,
        "reason": reason,
        "top_score": None,
        "score_gap": None,
        "coverage": 0.0,
        "distinct_documents": 0,
        "conflicts": [],
        "stale_passages": 0,
        "reranker": "disabled",
        "degraded": [],
    }


def _signals(
    words: list[str],
    passages: list[Passage],
    prof: Profile,
    reranker_state: str,
    any_relevant: bool,
    degraded: list[str],
) -> dict[str, Any]:
    if not passages:
        s = _empty_signals("no_passage_above_threshold")
        s["reranker"], s["degraded"] = reranker_state, degraded
        return s
    scores = sorted((p.score or 0.0 for p in passages), reverse=True)
    evidence = [
        PassageEvidence(p.chunk_id, str(p.document_id), p.text, p.terms, p.score, p.modified_at or p.ingested_at)
        for p in passages
    ]
    stale = 0
    for p in passages:
        age = staleness_days(p.modified_at or p.ingested_at)
        if age is not None and age > prof.staleness_days:
            stale += 1
    return {
        "nothing_relevant": not any_relevant,
        "reason": None if any_relevant else "top_score_below_threshold",
        "top_score": round(scores[0], 4),
        "score_gap": round(scores[0] - scores[1], 4) if len(scores) > 1 else None,
        "coverage": coverage(words, evidence),
        "distinct_documents": len({p.document_id for p in passages}),
        "conflicts": conflicts(evidence),
        "stale_passages": stale,
        "reranker": reranker_state,
        "degraded": degraded,
    }
