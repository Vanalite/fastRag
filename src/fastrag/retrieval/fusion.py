"""Reciprocal rank fusion (Cormack et al., 2009).

Dense cosine scores and BM25 scores live on different scales, so adding them is
meaningless. RRF combines *ranks* instead: score(d) = sum over lists of 1 / (k + rank).
It needs no tuning beyond ``k`` and is robust when one retriever returns nothing.
"""

from __future__ import annotations

from collections.abc import Sequence


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]], k: int = 60
) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
