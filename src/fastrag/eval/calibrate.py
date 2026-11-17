"""Fit reranker score calibration (Platt scaling) on reference questions.

For each document question the reranker scores the fused candidates. A
candidate is labelled relevant when it matches the question's `relevant`
entries; every candidate of an `expect_nothing` question is labelled not
relevant. A two-parameter logistic regression on logit(raw score) gives
(a, b) for FASTRAG_RERANKER_CALIBRATION.
"""

from __future__ import annotations

import math
import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from fastrag.db.models import Collection
from fastrag.eval.runner import matches
from fastrag.providers.reranker import CalibratedReranker, logit, sigmoid
from fastrag.retrieval.profiles import resolve_profile
from fastrag.retrieval.search import SearchRequest, search
from fastrag.services import Services


def _loss(xs: list[float], ts: list[float], a: float, b: float) -> float:
    total = 0.0
    for x, t in zip(xs, ts, strict=True):
        z = a * x + b
        # log(1 + e^z) - t*z, computed stably
        total += (z if z > 0 else 0.0) + math.log1p(math.exp(-abs(z))) - t * z
    return total


def fit_platt(xs: list[float], ys: list[int], iterations: int = 100) -> tuple[float, float]:
    """Platt (1999) with smoothed targets, which keeps the fit finite when the
    reranker already separates the classes perfectly; damped Newton steps."""
    n_pos = sum(ys)
    n_neg = len(ys) - n_pos
    t_pos, t_neg = (n_pos + 1) / (n_pos + 2), 1 / (n_neg + 2)
    ts = [t_pos if y else t_neg for y in ys]
    a, b = 1.0, 0.0
    loss = _loss(xs, ts, a, b)
    for _ in range(iterations):
        ga = gb = haa = hab = hbb = 0.0
        for x, t in zip(xs, ts, strict=True):
            p = sigmoid(a * x + b)
            w = max(p * (1 - p), 1e-12)
            ga += (p - t) * x
            gb += p - t
            haa += w * x * x
            hab += w * x
            hbb += w
        haa += 1e-9
        hbb += 1e-9
        det = haa * hbb - hab * hab
        if abs(det) < 1e-18:
            break
        da = (hbb * ga - hab * gb) / det
        db = (haa * gb - hab * ga) / det
        step = 1.0
        while step > 1e-8:
            na, nb = a - step * da, b - step * db
            new_loss = _loss(xs, ts, na, nb)
            if new_loss < loss:
                break
            step /= 2
        if step <= 1e-8:
            break
        a, b, improvement = na, nb, loss - new_loss
        loss = new_loss
        if abs(improvement) < 1e-9:
            break
    return a, b


async def collect_pairs(
    sessionmaker: async_sessionmaker[AsyncSession],
    services: Services,
    tenant_id: uuid.UUID,
    refs: list[dict[str, Any]],
) -> list[tuple[float, int]]:
    if services.reranker is None:
        raise ValueError("calibration needs a reranker (FASTRAG_RERANKER_PROVIDER)")
    inner = services.reranker.inner if isinstance(services.reranker, CalibratedReranker) else services.reranker
    async with sessionmaker() as s:
        cols = list((await s.execute(select(Collection).where(Collection.tenant_id == tenant_id))).scalars())
    docs = [c for c in cols if c.type == "documents"]
    prof = resolve_profile("internal_search").model_copy(update={"min_score": 0.0, "min_similarity": 0.0})
    saved, services.reranker = services.reranker, None  # fused candidates only; we score them below
    pairs: list[tuple[float, int]] = []
    try:
        for ref in refs:
            if ref.get("type") == "listings" or not (ref.get("relevant") or ref.get("expect_nothing")):
                continue
            targets = [c for c in docs if not ref.get("collections") or c.slug in ref["collections"]]
            req = SearchRequest(tenant_id, [c.id for c in targets], ref["query"], ["public", "staff"], prof, top_k=20)
            res = await search(sessionmaker, services, req)
            if not res.passages:
                continue
            raw = await inner.rerank(ref["query"], [(p.context or "") + "\n" + p.text for p in res.passages])
            for p, score in zip(res.passages, raw, strict=True):
                row = {
                    "external_id": p.external_id,
                    "title": p.title,
                    "document_id": str(p.document_id),
                    "text": p.text,
                }
                label = 0 if ref.get("expect_nothing") else int(any(matches(row, r) for r in ref["relevant"]))
                pairs.append((logit(score), label))
    finally:
        services.reranker = saved
    return pairs


def summarize(pairs: list[tuple[float, int]], a: float, b: float) -> dict[str, Any]:
    pos = sorted(sigmoid(a * x + b) for x, y in pairs if y)
    neg = sorted(sigmoid(a * x + b) for x, y in pairs if not y)

    def q(v: list[float], f: float) -> float | None:
        return round(v[min(len(v) - 1, int(f * (len(v) - 1)))], 3) if v else None

    loss = -sum(
        y * math.log(max(sigmoid(a * x + b), 1e-12)) + (1 - y) * math.log(max(1 - sigmoid(a * x + b), 1e-12))
        for x, y in pairs
    ) / max(len(pairs), 1)
    return {
        "a": round(a, 4),
        "b": round(b, 4),
        "pairs": len(pairs),
        "relevant": len(pos),
        "log_loss": round(loss, 4),
        "relevant_scores_p10_p50": [q(pos, 0.1), q(pos, 0.5)],
        "other_scores_p50_p90": [q(neg, 0.5), q(neg, 0.9)],
    }
