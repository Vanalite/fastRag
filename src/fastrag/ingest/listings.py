"""Spreadsheet listings into typed rows (D6, D8, D12).

Each sync replaces one sheet's rows as a complete snapshot stamped with
`as_of`. Columns are mapped per sheet layout; new or unmapped columns are
internal until an admin marks them public.
"""

from __future__ import annotations

import hashlib
import logging
import re
import statistics
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from fastrag.db.models import DEFAULT_GROUPS, Collection, Listing, ListingSnapshot, SheetMapping, Source
from fastrag.ingest import hygiene
from fastrag.log import log_event
from fastrag.services import Services
from fastrag.text import values as V
from fastrag.text.vietnamese import keyword_terms, normalize_text

logger = logging.getLogger(__name__)

# Typed fields a column can map to. Anything else becomes "attr:<slug>".
FIELDS = (
    "listing_code",
    "title",
    "project",
    "property_type",
    "status",
    "price_vnd",
    "price_per_m2_vnd",
    "area_m2",
    "bedrooms",
    "bathrooms",
    "direction",
    "city",
    "district_code",
    "ward",
    "address",
    "description",
)
TEXT_FIELDS = ("title", "project", "address", "description")

_SYNONYMS: dict[str, list[str]] = {
    "listing_code": [
        "ma can",
        "ma",
        "ma so",
        "ma sp",
        "ma san pham",
        "code",
        "id",
        "ma bds",
        "ma tin",
        "so can",
        "can so",
        "ma can ho",
        "unit",
        "unit code",
        "listing id",
    ],
    "title": ["tieu de", "ten", "ten san pham", "title", "name", "ten can"],
    "project": ["du an", "ten du an", "project", "khu", "toa", "block"],
    "property_type": ["loai", "loai hinh", "loai bds", "loai san pham", "type", "property type", "loai can"],
    "status": ["tinh trang", "trang thai", "status", "tinh trang ban"],
    "price_vnd": [
        "gia",
        "gia ban",
        "gia niem yet",
        "tong gia",
        "price",
        "gia ban niem yet",
        "gia tong",
        "gia chao",
        "gia ban ra",
    ],
    "price_per_m2_vnd": ["don gia", "gia m2", "gia tren m2", "price per m2", "gia moi m2", "don gia m2"],
    "area_m2": [
        "dien tich",
        "dt",
        "area",
        "dien tich su dung",
        "dt su dung",
        "dt thong thuy",
        "dien tich thong thuy",
        "dien tich dat",
        "dt dat",
    ],
    "bedrooms": ["phong ngu", "pn", "so phong ngu", "so pn", "bedrooms", "br", "so phong"],
    "bathrooms": ["wc", "toilet", "phong tam", "so wc", "bathrooms", "ve sinh", "so phong tam"],
    "direction": ["huong", "huong nha", "huong ban cong", "huong cua", "direction", "huong cua chinh"],
    "city": ["thanh pho", "tinh", "tinh thanh pho", "tinh thanh", "city", "tp"],
    "district_code": ["quan", "huyen", "quan huyen", "district"],
    "ward": ["phuong", "xa", "phuong xa", "ward"],
    "address": ["dia chi", "vi tri", "address", "location", "duong", "so nha"],
    "description": ["mo ta", "mo ta chi tiet", "description", "noi dung", "dac diem", "tien ich"],
}
_SYN_LOOKUP = {syn: f for f, syns in _SYNONYMS.items() for syn in syns}

_STATUS = {
    "available": ["con hang", "dang ban", "con", "available", "chua ban", "mo ban", "trong"],
    "sold": ["da ban", "sold", "het hang", "het"],
    "reserved": ["da coc", "giu cho", "booking", "dat coc", "reserved", "lock", "khoa can"],
}
_TYPES = {
    "apartment": ["can ho", "chung cu", "apartment", "condo", "ch"],
    "townhouse": ["nha pho", "townhouse", "nha lien ke", "lien ke", "nha"],
    "villa": ["biet thu", "villa"],
    "land": ["dat nen", "dat", "land", "lo dat"],
    "shophouse": ["shophouse", "nha pho thuong mai"],
    "officetel": ["officetel", "van phong"],
    "penthouse": ["penthouse"],
    "duplex": ["duplex"],
}


def _fold(s: Any) -> str:
    return V._fold(s)


def _lookup(table: dict[str, list[str]], value: Any) -> str:
    folded = _fold(value)
    for canon, names in table.items():
        if folded in names:
            return canon
    return V.slug(value)


def normalize_status(value: Any) -> str:
    return _lookup(_STATUS, value)


def normalize_property_type(value: Any) -> str:
    return _lookup(_TYPES, value)


def _header_key(header: Any) -> str:
    h = _fold(header)
    return re.sub(r"\b(vnd|ty|trieu|tr|m2|ve|dong|d|\d+)\b", " ", h).split("(")[0].strip()


def _unit_hint(header: Any) -> str | None:
    h = " " + _fold(header) + " "
    if " ty " in h or " ti " in h:
        return "ty"
    if " trieu " in h or " tr " in h:
        return "trieu"
    return None


def suggest_column(header: Any) -> str:
    key = re.sub(r"\s+", " ", _header_key(header))
    if key in _SYN_LOOKUP:
        return _SYN_LOOKUP[key]
    # "Giá bán (tỷ)", "Diện tích (m²)": try the longest synonym contained in the header.
    for syn in sorted(_SYN_LOOKUP, key=len, reverse=True):
        if len(syn) >= 4 and re.search(rf"\b{re.escape(syn)}\b", key):
            return _SYN_LOOKUP[syn]
    return "attr:" + (V.slug(header) or "column")


def find_header_row(rows: list[list[Any]]) -> int:
    best, best_score = 0, -1
    for i, row in enumerate(rows[:15]):
        cells = [c for c in row if c is not None]
        strings = [c for c in cells if isinstance(c, str) and not re.fullmatch(r"[\d.,\s]+", c)]
        known = sum(not suggest_column(c).startswith("attr:") for c in strings)
        score = known * 3 + len(strings)
        if len(strings) >= 2 and score > best_score:
            best, best_score = i, score
    return best


def layout_hash(headers: list[str]) -> str:
    return hashlib.sha256("\x1f".join(_fold(h) for h in headers).encode()).hexdigest()


def _infer_price_unit(rows: list[list[Any]], col: int) -> str | None:
    nums = [r[col] for r in rows if col < len(r) and isinstance(r[col], int | float)]
    if len(nums) < 3:
        return None
    med = statistics.median(nums)
    if 0.3 <= med < 500:
        return "ty"
    if 500 <= med < 100_000:
        return "trieu"
    return None


def suggest_mapping(headers: list[str], rows: list[list[Any]]) -> dict[str, Any]:
    columns: dict[str, Any] = {}
    used: set[str] = set()
    for i, h in enumerate(headers):
        if not h:
            continue
        target = suggest_column(h)
        if target in used:
            target = "attr:" + V.slug(h)
        used.add(target)
        entry: dict[str, Any] = {"field": target, "groups": list(DEFAULT_GROUPS)}
        if target in ("price_vnd", "price_per_m2_vnd"):
            unit = _unit_hint(h) or _infer_price_unit(rows, i)
            if unit:
                entry["unit"] = unit
        columns[h] = entry
    return columns


@dataclass
class RowResult:
    values: dict[str, Any] = field(default_factory=dict)
    attributes: dict[str, Any] = field(default_factory=dict)
    field_groups: dict[str, list[str]] = field(default_factory=dict)
    errors: list[dict[str, Any]] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)


_UNIT_MULT = {"ty": 1_000_000_000, "trieu": 1_000_000}


def _parse_cell(fieldname: str, value: Any, entry: dict[str, Any], gaz: V.Gazetteer) -> Any:
    if fieldname in ("price_vnd", "price_per_m2_vnd"):
        unit = entry.get("unit")
        if isinstance(value, int | float) and unit in _UNIT_MULT:
            return round(value * _UNIT_MULT[unit])
        if isinstance(value, str) and unit in _UNIT_MULT and re.fullmatch(r"\s*\d+(?:[.,]\d+)?\s*", value):
            return round(V.parse_number(value) * _UNIT_MULT[unit])
        return V.parse_price(value).vnd
    if fieldname == "area_m2":
        return V.parse_area(value)
    if fieldname in ("bedrooms", "bathrooms"):
        return V.parse_bedrooms(value)
    if fieldname == "direction":
        return V.parse_direction(value)
    if fieldname == "district_code":
        return gaz.district(value).code
    if fieldname == "city":
        return gaz.city(value)
    if fieldname == "ward":
        return V.normalize_ward(value)
    if fieldname == "status":
        return normalize_status(value)
    if fieldname == "property_type":
        return normalize_property_type(value)
    text = normalize_text(str(value))[0]
    return text


def parse_row(
    headers: list[str], row: list[Any], columns: dict[str, Any], row_groups: list[str], gaz: V.Gazetteer
) -> RowResult:
    out = RowResult()
    for i, h in enumerate(headers):
        if not h or i >= len(row) or row[i] is None:
            continue
        entry = columns.get(h) or {"field": "attr:" + V.slug(h), "groups": list(DEFAULT_GROUPS)}
        target = entry["field"]
        if target == "ignore":
            continue
        groups = list(entry.get("groups") or DEFAULT_GROUPS)
        raw = row[i]
        # Safety net: sensitive values in a public column stay internal (D12).
        if "public" in groups:
            flags = hygiene.scan(str(raw), groups)
            if hygiene.FLAG_SENSITIVE_PUBLIC in flags:
                groups = [g for g in groups if g != "public"] or list(DEFAULT_GROUPS)
                out.flags.append(f"{hygiene.FLAG_SENSITIVE_PUBLIC}:{target}")
        if target.startswith("attr:"):
            key = target[5:]
            out.attributes[key] = normalize_text(str(raw))[0] if isinstance(raw, str) else raw
            out.field_groups[target] = groups
            continue
        try:
            out.values[target] = _parse_cell(target, raw, entry, gaz)
            out.field_groups[target] = groups
        except ValueError as exc:
            # Keep the original text as an attribute so nothing is lost.
            out.errors.append({"column": h, "field": target, "value": str(raw)[:200], "error": str(exc)})
            out.attributes[V.slug(h)] = str(raw)
            out.field_groups["attr:" + V.slug(h)] = groups
    # Derive the district from the address when there is no district column.
    if "district_code" not in out.values and out.values.get("address"):
        found = gaz.find_in_text(out.values["address"])
        if found:
            out.values["district_code"] = found[0]
            out.field_groups["district_code"] = out.field_groups.get("address", list(DEFAULT_GROUPS))
    if "district_code" in out.values and "city" not in out.values:
        d = gaz.districts.get(out.values["district_code"])
        if d:
            out.values["city"] = d.city
            out.field_groups["city"] = out.field_groups["district_code"]
    if out.values.get("project"):
        out.values["project_slug"] = V.slug(out.values["project"])
        out.field_groups["project_slug"] = out.field_groups["project"]
    return out


_LISTING_NS = uuid.UUID("5d4f3a9e-7b1c-4c55-9a0e-6f1b2c3d4e5f")


def listing_uuid(collection_id: uuid.UUID, sheet_key: str, code: str) -> uuid.UUID:
    """Stable across syncs, so an agent can look up "that unit" again later (D13)."""
    return uuid.uuid5(_LISTING_NS, f"{collection_id}:{sheet_key}:{code}")


def _listing_terms(res: RowResult, segmenter: str) -> list[str]:
    """Keyword terms prefixed by access group ("public|căn_hộ"), so a keyword
    filter can only match text the caller is allowed to see."""
    terms: set[str] = set()
    texts: list[tuple[str, list[str]]] = [
        (str(res.values[f]), res.field_groups.get(f, DEFAULT_GROUPS)) for f in TEXT_FIELDS if res.values.get(f)
    ]
    for k, v in res.attributes.items():
        if isinstance(v, str):
            texts.append((v, res.field_groups.get("attr:" + k, DEFAULT_GROUPS)))
    for text, groups in texts:
        for t in keyword_terms(text, segmenter).freqs:
            for g in groups:
                terms.add(f"{g}|{t}")
    return sorted(terms)


async def get_or_create_mapping(
    session: AsyncSession, collection_id: uuid.UUID, headers: list[str], rows: list[list[Any]]
) -> SheetMapping:
    lh = layout_hash(headers)
    mapping = (
        await session.execute(
            select(SheetMapping).where(SheetMapping.collection_id == collection_id, SheetMapping.layout_hash == lh)
        )
    ).scalar_one_or_none()
    if mapping is None:
        mapping = SheetMapping(
            id=uuid.uuid4(),
            collection_id=collection_id,
            layout_hash=lh,
            headers=headers,
            columns=suggest_mapping(headers, rows),
            confirmed=False,
        )
        session.add(mapping)
        await session.flush()
    return mapping


async def import_sheet(
    session: AsyncSession,
    services: Services,
    source: Source,
    sheet_key: str,
    rows: list[list[Any]],
    row_groups: list[str] | None = None,
    as_of: datetime | None = None,
    storage_key: str | None = None,
) -> dict[str, Any]:
    collection = await session.get(Collection, source.collection_id)
    assert collection is not None
    if collection.type != "listings":
        raise ValueError("spreadsheets with listings need a collection of type 'listings'")
    as_of = as_of or datetime.now(UTC)
    groups = row_groups or list(source.groups)
    gaz = services.gazetteer
    aliases = (collection.settings or {}).get("gazetteer")
    if aliases:
        gaz = gaz.with_aliases(aliases)

    hdr_idx = find_header_row(rows)
    headers = [normalize_text(str(h))[0] if h is not None else "" for h in rows[hdr_idx]] if rows else []
    body = rows[hdr_idx + 1 :]
    mapping = await get_or_create_mapping(session, collection.id, headers, body)

    snapshot = ListingSnapshot(
        id=uuid.uuid4(),
        collection_id=collection.id,
        source_id=source.id,
        sheet_key=sheet_key,
        as_of=as_of,
        row_count=0,
    )
    session.add(snapshot)
    await session.flush()

    listings: list[Listing] = []
    rejected: list[dict[str, Any]] = []
    cell_errors = 0
    seen_codes: dict[str, int] = {}
    flagged = 0
    for n, row in enumerate(body, start=hdr_idx + 2):
        if not any(c is not None for c in row):
            continue
        res = parse_row(headers, row, mapping.columns, groups, gaz)
        if not res.values and not res.attributes:
            continue
        if res.errors:
            cell_errors += len(res.errors)
            rejected.append({"row": n, "errors": res.errors})
        code = str(res.values.pop("listing_code", "") or "").strip()
        if not code:
            basis = "|".join(str(res.values.get(k, "")) for k in ("project", "address", "title", "area_m2"))
            code = "row-" + hashlib.sha1(basis.encode(), usedforsecurity=False).hexdigest()[:10]
        if code in seen_codes:
            seen_codes[code] += 1
            res.flags.append(f"duplicate_code:{code}")
            code = f"{code}~{seen_codes[code]}"
        else:
            seen_codes[code] = 1
        res.field_groups.setdefault("listing_code", res.field_groups.get("listing_code", list(DEFAULT_GROUPS)))
        if res.flags:
            flagged += 1
        listings.append(
            Listing(
                id=listing_uuid(collection.id, sheet_key, code),
                tenant_id=source.tenant_id,
                collection_id=collection.id,
                source_id=source.id,
                snapshot_id=snapshot.id,
                sheet_key=sheet_key,
                listing_code=code,
                row_number=n,
                groups=groups,
                field_groups=res.field_groups,
                attributes=res.attributes,
                terms=_listing_terms(res, services.settings.segmenter),
                as_of=as_of,
                flags=res.flags,
                **{k: v for k, v in res.values.items() if hasattr(Listing, k)},
            )
        )

    # Complete snapshot: the previous rows of this sheet are replaced.
    await session.execute(delete(Listing).where(Listing.collection_id == collection.id, Listing.sheet_key == sheet_key))
    session.add_all(listings)
    snapshot.row_count = len(listings)
    snapshot.rejected = rejected[:500]
    unmapped_public = [h for h, e in mapping.columns.items() if "public" in (e.get("groups") or [])]
    snapshot.report = {
        "header_row": hdr_idx + 1,
        "mapping_id": str(mapping.id),
        "mapping_confirmed": mapping.confirmed,
        "public_columns": unmapped_public,
        "cell_errors": cell_errors,
        "flagged_rows": flagged,
        "storage_key": storage_key,
        "row_groups": groups,
    }
    await session.flush()
    # Keep the last 10 snapshot records per sheet for the audit trail.
    keep = (
        select(ListingSnapshot.id)
        .where(ListingSnapshot.collection_id == collection.id, ListingSnapshot.sheet_key == sheet_key)
        .order_by(ListingSnapshot.created_at.desc(), ListingSnapshot.id)
        .limit(10)
        .scalar_subquery()
    )
    await session.execute(
        delete(ListingSnapshot).where(
            ListingSnapshot.collection_id == collection.id,
            ListingSnapshot.sheet_key == sheet_key,
            ListingSnapshot.id.not_in(keep),
            ListingSnapshot.id != snapshot.id,
        )
    )
    log_event(logger, "sheet imported", sheet=sheet_key, rows=len(listings), cell_errors=cell_errors)
    return {
        "snapshot_id": str(snapshot.id),
        "rows": len(listings),
        "cell_errors": cell_errors,
        "flagged_rows": flagged,
        "mapping_id": str(mapping.id),
        "mapping_confirmed": mapping.confirmed,
        "as_of": as_of.isoformat(),
    }
