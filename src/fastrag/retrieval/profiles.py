"""Retrieval profiles are configuration, not code paths (D11).

A collection can override any value under settings.profiles.<name>.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from fastrag.ingest.hygiene import QUARANTINE_FLAGS


class Profile(BaseModel):
    name: str
    # Hard ceiling on visible groups. None means "whatever the API key allows".
    max_groups: list[str] | None = None
    keyword_candidates: int = 50
    vector_candidates: int = 50
    rerank_candidates: int = 30
    top_k: int = 5
    # Calibrated reranker score below which a passage is not returned.
    min_score: float = 0.2
    # Top score below this means "nothing relevant found".
    relevance_threshold: float = 0.35
    # Without a reranker: keep a passage when its cosine similarity or the share
    # of question words it contains reaches this value.
    min_similarity: float = 0.5
    exclude_flags: list[str] = []
    staleness_days: int = 365
    rrf_k: int = 60
    # Keyword search reads at most this many postings: the rarest question
    # terms come first, and the most common ones (lowest IDF) are skipped
    # once the budget is used. Bounds latency when every term is common.
    keyword_postings_budget: int = 60_000
    accentless_weight: float = 0.5
    ef_search: int = 80


BUILTIN: dict[str, Profile] = {
    "customer": Profile(
        name="customer",
        max_groups=["public"],
        keyword_candidates=40,
        vector_candidates=40,
        rerank_candidates=20,
        top_k=5,
        min_score=0.3,
        relevance_threshold=0.45,
        exclude_flags=list(QUARANTINE_FLAGS),
        staleness_days=180,
        keyword_postings_budget=40_000,
    ),
    "handoff_copilot": Profile(
        name="handoff_copilot",
        keyword_candidates=50,
        vector_candidates=50,
        rerank_candidates=30,
        top_k=8,
        min_score=0.2,
        exclude_flags=["prompt_injection"],
    ),
    "internal_search": Profile(
        name="internal_search",
        keyword_candidates=100,
        vector_candidates=100,
        rerank_candidates=50,
        top_k=15,
        min_score=0.1,
        relevance_threshold=0.3,
        ef_search=150,
        keyword_postings_budget=120_000,
    ),
}


def resolve_profile(name: str, collection_settings: dict[str, Any] | None = None) -> Profile:
    if name not in BUILTIN:
        raise KeyError(f"unknown profile {name!r}; use one of {sorted(BUILTIN)}")
    base = BUILTIN[name]
    override = ((collection_settings or {}).get("profiles") or {}).get(name)
    if not override:
        return base
    merged = base.model_dump() | override
    # A collection may tighten a profile's group ceiling but not lift it.
    if base.max_groups is not None:
        merged["max_groups"] = [g for g in (override.get("max_groups") or base.max_groups) if g in base.max_groups]
    return Profile.model_validate(merged)


def effective_groups(requested: list[str] | None, key_groups: list[str], profile: Profile) -> list[str]:
    """Groups the caller can see: request ∩ API key ceiling ∩ profile ceiling."""
    groups = set(key_groups)
    if requested is not None:
        groups &= set(requested)
    if profile.max_groups is not None:
        groups &= set(profile.max_groups)
    return sorted(groups)
