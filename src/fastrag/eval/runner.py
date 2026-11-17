"""Quality checks (D4, Q12 proposal).

Reference questions live in a JSONL file, one object per line:

  Document questions:
    {"id": "q1", "query": "Phí quản lý chung cư bao nhiêu?", "profile": "customer",
     "collections": ["docs"], "relevant": [{"document": "<external_id or title>", "contains": "optional text"}]}
    {"id": "q2", "query": "...", "expect_nothing": true}

  Listing filter checks (exact standardization):
    {"id": "l1", "type": "listings", "filters": {"price_max": "3,5 tỷ", "district": "Q7"},
     "expect_applied": {"price_max": 3500000000, "district": ["hcm-q7"]},
     "expect_ids": ["optional listing codes that must be returned"]}

Three release checks:
  1. Access safety: no customer-profile result may carry a group other than
     public. Blocks a release.
  2. Listing accuracy: 100% of listing checks pass. Blocks a release.
  3. Document retrieval: Recall@k and MRR, compared with a baseline file.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from fastrag.db.models import Collection
from fastrag.retrieval import listings as L
from fastrag.retrieval.profiles import effective_groups, resolve_profile
from fastrag.retrieval.search import SearchRequest, search
from fastrag.services import Services
from fastrag.text.vietnamese import strip_diacritics


@dataclass
class EvalReport:
    questions: int = 0
    recall_at_k: float = 0.0
    mrr: float = 0.0
    nothing_accuracy: float | None = None
    access_violations: list[dict[str, Any]] = field(default_factory=list)
    listing_checks: int = 0
    listing_failures: list[dict[str, Any]] = field(default_factory=list)
    per_question: list[dict[str, Any]] = field(default_factory=list)
    baseline: dict[str, Any] | None = None
    k: int = 10

    @property
    def passed(self) -> bool:
        if self.access_violations or self.listing_failures:
            return False
        if self.baseline and self.questions:
            return self.recall_at_k >= self.baseline["recall_at_k"] and self.mrr >= self.baseline["mrr"]
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "k": self.k,
            "documents": {
                "questions": self.questions,
                f"recall@{self.k}": round(self.recall_at_k, 4),
                "mrr": round(self.mrr, 4),
                "nothing_relevant_accuracy": self.nothing_accuracy,
            },
            "access_safety": {"violations": self.access_violations, "passed": not self.access_violations},
            "listings": {
                "checks": self.listing_checks,
                "failures": self.listing_failures,
                "passed": not self.listing_failures,
            },
            "baseline": self.baseline,
            "per_question": self.per_question,
        }


def load_references(path: str | Path) -> list[dict[str, Any]]:
    out = []
    for i, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{i}: {exc}") from exc
        obj.setdefault("id", f"line{i}")
        out.append(obj)
    return out


def _fold(s: str) -> str:
    return " ".join(strip_diacritics(s).lower().split())


def matches(result: dict[str, Any], rel: dict[str, Any]) -> bool:
    doc = rel.get("document")
    if doc and doc not in (result.get("external_id"), result.get("title"), result.get("document_id")):
        return False
    contains = rel.get("contains")
    return not (contains and _fold(contains) not in _fold(result.get("text", "")))


def score_ranked(results: list[dict[str, Any]], relevant: list[dict[str, Any]], k: int) -> tuple[float, float]:
    """Recall@k over relevant items and reciprocal rank of the first hit."""
    if not relevant:
        return 0.0, 0.0
    found = 0
    for rel in relevant:
        if any(matches(r, rel) for r in results[:k]):
            found += 1
    rr = 0.0
    for rank, r in enumerate(results[:k], start=1):
        if any(matches(r, rel) for rel in relevant):
            rr = 1.0 / rank
            break
    return found / len(relevant), rr


def baseline_metrics(path: str | Path, refs: list[dict[str, Any]], k: int) -> dict[str, Any]:
    """Score another system's results (e.g. RAGFlow, D10) with the same metrics.
    File: JSONL of {"id": "q1", "results": [{"document": "...", "text": "..."}]}."""
    by_id = {r["id"]: r for r in load_references(path)}
    recall, rr, n = 0.0, 0.0, 0
    for ref in refs:
        if ref.get("type") == "listings" or ref.get("expect_nothing") or not ref.get("relevant"):
            continue
        res = [
            {"external_id": x.get("document"), "title": x.get("document"), "text": x.get("text", "")}
            for x in by_id.get(ref["id"], {}).get("results", [])
        ]
        rc, r = score_ranked(res, ref["relevant"], k)
        recall, rr, n = recall + rc, rr + r, n + 1
    return {
        "file": str(path),
        "questions": n,
        "recall_at_k": recall / n if n else 0.0,
        "mrr": rr / n if n else 0.0,
    }


async def _access_check(session: AsyncSession, chunk_ids: list[int]) -> list[int]:
    if not chunk_ids:
        return []
    rows = await session.execute(
        # Content tagged public (alone or with other groups) is customer-visible by design.
        text("SELECT id FROM chunks WHERE id = ANY(:ids) AND NOT ('public' = ANY(groups))"),
        {"ids": chunk_ids},
    )
    return [int(r.id) for r in rows]


async def run_eval(
    sessionmaker: async_sessionmaker[AsyncSession],
    services: Services,
    tenant_id: uuid.UUID,
    refs: list[dict[str, Any]],
    k: int = 10,
    baseline: str | None = None,
    staff_groups: tuple[str, ...] = ("public", "staff"),
    document_id_map: dict[str, str] | None = None,
) -> EvalReport:
    """`document_id_map` translates result document ids before matching, so a
    trial copy of a collection is scored against references written for the
    original (D25)."""
    report = EvalReport(k=k)
    id_map = document_id_map or {}
    async with sessionmaker() as s:
        cols = list((await s.execute(select(Collection).where(Collection.tenant_id == tenant_id))).scalars())
    by_slug = {c.slug: c for c in cols}
    recall_sum = rr_sum = 0.0
    nothing_total = nothing_ok = 0

    for ref in refs:
        if ref.get("type") == "listings":
            await _listing_check(sessionmaker, services, tenant_id, cols, ref, report)
            continue
        wanted = ref.get("collections")
        targets = (
            [by_slug[x] for x in wanted if x in by_slug]
            if wanted
            else [c for c in cols if c.type == "documents" and "trial_of" not in (c.settings or {})]
        )
        prof_name = ref.get("profile", "customer")
        prof = resolve_profile(prof_name, targets[0].settings if targets else None)
        groups = effective_groups(ref.get("groups"), list(staff_groups), prof)
        req = SearchRequest(tenant_id, [c.id for c in targets], ref["query"], groups, prof, top_k=k)
        result = await search(sessionmaker, services, req)
        rows: list[dict[str, Any]] = [
            {
                "external_id": p.external_id,
                "title": p.title,
                "document_id": id_map.get(str(p.document_id), str(p.document_id)),
                "text": p.text,
                "chunk_id": p.chunk_id,
            }
            for p in result.passages
        ]
        # Release check 1 runs every question again in the customer scope.
        cust = resolve_profile("customer", targets[0].settings if targets else None)
        creq = SearchRequest(
            tenant_id,
            [c.id for c in targets],
            ref["query"],
            effective_groups(None, list(staff_groups), cust),
            cust,
            top_k=k,
        )
        if prof_name == "customer":
            customer_ids = [int(r["chunk_id"]) for r in rows]
        else:
            customer_ids = [p.chunk_id for p in (await search(sessionmaker, services, creq)).passages]
        async with sessionmaker() as s:
            bad = await _access_check(s, customer_ids)
        if bad:
            report.access_violations.append({"id": ref["id"], "chunk_ids": bad})

        entry: dict[str, Any] = {
            "id": ref["id"],
            "query": ref["query"],
            "profile": prof_name,
            "nothing_relevant": result.signals.get("nothing_relevant"),
            "top": [f"{r['title']}#{r['chunk_id']}" for r in rows[:3]],
        }
        if ref.get("expect_nothing"):
            nothing_total += 1
            ok = bool(result.signals.get("nothing_relevant"))
            nothing_ok += ok
            entry["expect_nothing_ok"] = ok
        elif ref.get("relevant"):
            rc, rr = score_ranked(rows, ref["relevant"], k)
            recall_sum, rr_sum = recall_sum + rc, rr_sum + rr
            report.questions += 1
            entry.update({"recall": rc, "rr": rr})
        report.per_question.append(entry)

    if report.questions:
        report.recall_at_k = recall_sum / report.questions
        report.mrr = rr_sum / report.questions
    if nothing_total:
        report.nothing_accuracy = round(nothing_ok / nothing_total, 4)
    if baseline:
        report.baseline = baseline_metrics(baseline, refs, k)
    return report


async def _listing_check(
    sessionmaker: async_sessionmaker[AsyncSession],
    services: Services,
    tenant_id: uuid.UUID,
    cols: list[Collection],
    ref: dict[str, Any],
    report: EvalReport,
) -> None:
    report.listing_checks += 1
    targets = [c for c in cols if c.type == "listings" and (not ref.get("collections") or c.slug in ref["collections"])]
    q = L.ListingQuery(
        tenant_id,
        [c.id for c in targets],
        list(ref.get("groups") or ["public", "staff"]),
        dict(ref.get("filters") or {}),
        limit=50,
    )
    failure: dict[str, Any] = {"id": ref["id"]}
    try:
        async with sessionmaker() as s:
            res = await L.query_listings(s, q, services.gazetteer, services.settings.segmenter)
    except L.FilterError as exc:
        if not ref.get("expect_rejected"):
            failure["error"] = exc.rejected
            report.listing_failures.append(failure)
        return
    if ref.get("expect_rejected"):
        failure["error"] = "expected the filters to be rejected"
        report.listing_failures.append(failure)
        return
    for key, want in (ref.get("expect_applied") or {}).items():
        got = res["applied_filters"].get(key)
        if got != want:
            failure.setdefault("applied", {})[key] = {"want": want, "got": got}
    want_codes = set(ref.get("expect_ids") or [])
    got_codes = {i.get("listing_code") for i in res["items"]}
    if want_codes - got_codes:
        failure["missing_ids"] = sorted(want_codes - got_codes)
    if len(failure) > 1:
        report.listing_failures.append(failure)
