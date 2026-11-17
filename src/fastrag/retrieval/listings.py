"""query_listings: predefined filters over typed listing rows (D6, D13).

Filter values are standardized deterministically ("3,5 tỷ", "Q7"). A filter
that cannot be read is rejected, never guessed. A field the caller may not
see is masked in SQL and cannot be filtered or sorted on.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from fastrag.ingest.listings import FIELDS, normalize_property_type, normalize_status
from fastrag.text import values as V
from fastrag.text.vietnamese import query_terms

OUTPUT_FIELDS = [f for f in FIELDS if f != "listing_code"] + ["listing_code"]
SORTS = {
    "price_asc": ("price_vnd", "ASC"),
    "price_desc": ("price_vnd", "DESC"),
    "area_asc": ("area_m2", "ASC"),
    "area_desc": ("area_m2", "DESC"),
    "price_per_m2_asc": ("price_per_m2_vnd", "ASC"),
    "newest": ("as_of", "DESC"),
}
MAX_LIMIT = 50


class FilterError(ValueError):
    def __init__(self, rejected: list[dict[str, Any]]):
        super().__init__("; ".join(f"{r['filter']}: {r['reason']}" for r in rejected))
        self.rejected = rejected


@dataclass
class ListingQuery:
    tenant_id: uuid.UUID
    collection_ids: list[uuid.UUID]
    groups: list[str]
    filters: dict[str, Any]
    sort: str | None = None
    limit: int = 10
    offset: int = 0
    ids: list[uuid.UUID] | None = None
    codes: list[str] | None = None


def _as_list(v: Any) -> list[Any]:
    return v if isinstance(v, list) else [v]


def _visible(fieldname: str) -> str:
    return f"(l.field_groups -> '{fieldname}') ?| CAST(:groups AS text[])"


def build_where(q: ListingQuery, gaz: V.Gazetteer, segmenter: str) -> tuple[list[str], dict[str, Any]]:
    clauses = [
        "l.tenant_id = :tenant_id",
        "l.collection_id = ANY(:collection_ids)",
        "l.groups && CAST(:groups AS text[])",
    ]
    params: dict[str, Any] = {
        "tenant_id": q.tenant_id,
        "collection_ids": q.collection_ids,
        "groups": q.groups,
    }
    rejected: list[dict[str, Any]] = []
    n = 0

    def param(value: Any) -> str:
        nonlocal n
        n += 1
        params[f"p{n}"] = value
        return f":p{n}"

    def rng(name: str, col: str, parse: Any, op: str) -> None:
        raw = q.filters.get(name)
        if raw is None:
            return
        try:
            val = parse(raw)
        except ValueError as exc:
            rejected.append({"filter": name, "value": raw, "reason": str(exc)})
            return
        clauses.append(f"({_visible(col)} AND l.{col} {op} {param(val)})")

    def price(raw: Any) -> int:
        p = V.parse_price(raw)
        if p.per_m2:
            raise ValueError("this is a price per m²; use price_per_m2_min / price_per_m2_max")
        return p.vnd

    def price_m2(raw: Any) -> int:
        return V.parse_price(raw).vnd

    rng("price_min", "price_vnd", price, ">=")
    rng("price_max", "price_vnd", price, "<=")
    rng("price_per_m2_min", "price_per_m2_vnd", price_m2, ">=")
    rng("price_per_m2_max", "price_per_m2_vnd", price_m2, "<=")
    rng("area_min", "area_m2", V.parse_area, ">=")
    rng("area_max", "area_m2", V.parse_area, "<=")
    rng("bedrooms_min", "bedrooms", V.parse_bedrooms, ">=")
    rng("bedrooms_max", "bedrooms", V.parse_bedrooms, "<=")
    rng("bathrooms_min", "bathrooms", V.parse_bedrooms, ">=")

    def any_of(name: str, col: str, parse: Any) -> None:
        raw = q.filters.get(name)
        if raw is None or raw == []:
            return
        vals: list[Any] = []
        for item in _as_list(raw):
            try:
                parsed = parse(item)
                vals.extend(parsed if isinstance(parsed, list) else [parsed])
            except ValueError as exc:
                rejected.append({"filter": name, "value": item, "reason": str(exc)})
        if vals:
            clauses.append(f"({_visible(col)} AND l.{col} = ANY({param(sorted(set(vals)))}))")

    def district(v: Any) -> list[str]:
        return gaz.descendants(gaz.district(v).code)

    def bedrooms_exact(v: Any) -> int:
        return V.parse_bedrooms(v)

    any_of("district", "district_code", district)
    any_of("city", "city", gaz.city)
    any_of("ward", "ward", V.normalize_ward)
    any_of("project", "project_slug", V.slug)
    any_of("property_type", "property_type", normalize_property_type)
    any_of("status", "status", normalize_status)
    any_of("direction", "direction", V.parse_direction)
    any_of("bedrooms", "bedrooms", bedrooms_exact)

    kw = q.filters.get("keywords")
    if kw:
        terms = [t for t in query_terms(str(kw), segmenter) if not t.startswith("~")] or list(
            query_terms(str(kw), segmenter)
        )
        scoped = [f"{g}|{t}" for g in q.groups for t in terms]
        params["kw"] = scoped
        clauses.append("l.terms && CAST(:kw AS text[])")

    if q.ids:
        clauses.append(f"l.id = ANY({param(q.ids)})")
    if q.codes:
        clauses.append(f"({_visible('listing_code')} AND l.listing_code = ANY({param(q.codes)}))")

    known = {
        "price_min",
        "price_max",
        "price_per_m2_min",
        "price_per_m2_max",
        "area_min",
        "area_max",
        "bedrooms_min",
        "bedrooms_max",
        "bathrooms_min",
        "district",
        "city",
        "ward",
        "project",
        "property_type",
        "status",
        "direction",
        "bedrooms",
        "keywords",
    }
    for name in q.filters:
        if name not in known:
            rejected.append({"filter": name, "value": q.filters[name], "reason": "unknown filter"})
    if q.sort and q.sort not in SORTS and q.sort != "relevance":
        rejected.append({"filter": "sort", "value": q.sort, "reason": f"use one of {sorted(SORTS)}"})
    if rejected:
        raise FilterError(rejected)
    return clauses, params


async def query_listings(session: AsyncSession, q: ListingQuery, gaz: V.Gazetteer, segmenter: str) -> dict[str, Any]:
    clauses, params = build_where(q, gaz, segmenter)
    cols = ", ".join(f"CASE WHEN {_visible(f)} THEN l.{f} END AS {f}" for f in OUTPUT_FIELDS)
    attrs = (
        "(SELECT COALESCE(jsonb_object_agg(a.key, a.value), '{}'::jsonb) FROM jsonb_each(l.attributes) a "
        "WHERE (l.field_groups -> ('attr:' || a.key)) ?| CAST(:groups AS text[])) AS attributes"
    )
    order = "l.as_of DESC, l.row_number ASC"
    if q.sort in SORTS:
        col, direction = SORTS[q.sort]
        expr = f"l.{col}" if col == "as_of" else f"CASE WHEN {_visible(col)} THEN l.{col} END"
        order = f"{expr} {direction} NULLS LAST, l.row_number ASC"
    elif "kw" in params:
        order = "cardinality(ARRAY(SELECT unnest(l.terms) INTERSECT SELECT unnest(CAST(:kw AS text[])))) DESC, " + order
    params["limit"] = max(1, min(q.limit, MAX_LIMIT))
    params["offset"] = max(0, q.offset)
    sql = f"""
        SELECT l.id, l.collection_id, l.sheet_key, l.as_of, {cols}, {attrs}, COUNT(*) OVER() AS total
        FROM listings l
        WHERE {" AND ".join(clauses)}
        ORDER BY {order}
        LIMIT :limit OFFSET :offset
    """
    rows = (await session.execute(text(sql), params)).mappings().all()
    items = []
    for r in rows:
        item: dict[str, Any] = {
            "id": str(r["id"]),
            "collection_id": str(r["collection_id"]),
            "as_of": r["as_of"].isoformat(),
        }
        for f in OUTPUT_FIELDS:
            v = r[f]
            if v is not None:
                item[f] = float(v) if f == "area_m2" else v
        if r["attributes"]:
            item["attributes"] = r["attributes"]
        items.append(item)
    as_of: datetime | None = min((r["as_of"] for r in rows), default=None)
    return {
        "items": items,
        "total": int(rows[0]["total"]) if rows else 0,
        "as_of": as_of.isoformat() if as_of else None,
        "applied_filters": _describe(q, gaz),
    }


def _describe(q: ListingQuery, gaz: V.Gazetteer) -> dict[str, Any]:
    """The standardized values that were applied, so the agent can confirm
    with the customer ("3,5 tỷ" -> 3500000000)."""
    out: dict[str, Any] = {}
    for k, v in q.filters.items():
        if v is None:
            continue
        try:
            if k in ("price_min", "price_max", "price_per_m2_min", "price_per_m2_max"):
                out[k] = V.parse_price(v).vnd
            elif k in ("area_min", "area_max"):
                out[k] = V.parse_area(v)
            elif k == "district":
                out[k] = [gaz.district(x).code for x in _as_list(v)]
            else:
                out[k] = v
        except ValueError:
            out[k] = v
    return out
