"""Evidence signals (D3). fastRag states facts about the evidence; the agent
decides what to do with them."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from fastrag.text.values import parse_price
from fastrag.text.vietnamese import word_in_terms

_MONEY = re.compile(r"\d+(?:[.,]\d+)*\s*(?:tỷ|tỉ|triệu|tr|đồng|vnđ|vnd|đ)(?:\s*/\s*m2|\s*/\s*m²)?", re.IGNORECASE)
_PERCENT = re.compile(r"\d+(?:[.,]\d+)?\s*%")


@dataclass
class PassageEvidence:
    chunk_id: int
    document_id: str
    text: str
    terms: set[str]
    score: float | None
    modified_at: datetime | None


def staleness_days(modified_at: datetime | None, now: datetime | None = None) -> int | None:
    if modified_at is None:
        return None
    now = now or datetime.now(UTC)
    return max(0, (now - modified_at).days)


def term_match(words: list[str], terms: set[str]) -> float:
    """Share of the question's content words present in a passage."""
    if not words:
        return 0.0
    return sum(word_in_terms(w, terms) for w in words) / len(words)


def coverage(words: list[str], passages: list[PassageEvidence]) -> float:
    """Share of the question's words found in at least one returned passage."""
    seen: set[str] = set()
    for p in passages:
        seen |= p.terms
    return round(term_match(words, seen), 3)


def _money_values(text: str) -> set[tuple[int, bool]]:
    out: set[tuple[int, bool]] = set()
    for m in _MONEY.findall(text):
        try:
            p = parse_price(m)
        except ValueError:
            continue
        out.add((p.vnd, p.per_m2))
    return out


def conflicts(passages: list[PassageEvidence], min_overlap: float = 0.5) -> list[dict[str, Any]]:
    """Possible conflicts: two passages from different documents about the
    same topic (high term overlap) that state different prices or
    percentages. A heuristic for the agent to check, not a verdict."""
    found: list[dict[str, Any]] = []
    for i, a in enumerate(passages):
        for b in passages[i + 1 :]:
            if a.document_id == b.document_id or not a.terms or not b.terms:
                continue
            overlap = len(a.terms & b.terms) / min(len(a.terms), len(b.terms))
            if overlap < min_overlap:
                continue
            ma, mb = _money_values(a.text), _money_values(b.text)
            pa, pb = set(_PERCENT.findall(a.text)), set(_PERCENT.findall(b.text))
            kind = None
            if ma and mb and not (ma & mb):
                kind = "price"
            elif pa and pb and not ({x.replace(" ", "") for x in pa} & {x.replace(" ", "") for x in pb}):
                kind = "percentage"
            if kind:
                newer = None
                if a.modified_at and b.modified_at and a.modified_at != b.modified_at:
                    newer = a.chunk_id if a.modified_at > b.modified_at else b.chunk_id
                found.append({"kind": kind, "passages": [a.chunk_id, b.chunk_id], "newer_passage": newer})
    return found[:10]
