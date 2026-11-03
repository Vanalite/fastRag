"""Ranking metrics for retrieval evaluation, with binary relevance.

Every function takes the *retrieved* list in rank order and the set of *relevant* ids.
Retrieval returns chunks, but relevance is judged per document, so retrieved ids are
de-duplicated first, keeping the first (best) occurrence of each. Two chunks of the same
relevant document therefore count once, and a duplicate never pushes a later document
out of the top k.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


def dedupe(retrieved: Iterable[str]) -> list[str]:
    """Drop repeated ids, keeping the first occurrence of each, in order."""
    return list(dict.fromkeys(retrieved))


def _relevant_set(relevant: Iterable[str]) -> set[str]:
    relevant = set(relevant)
    if not relevant:
        raise ValueError("relevance metrics are undefined when there are no relevant ids")
    return relevant


def _check_k(k: int) -> None:
    if k < 1:
        raise ValueError("k must be at least 1")


def recall_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Share of the relevant ids that appear in the top ``k`` (de-duplicated) results.

    ``recall_at_k(["a", "x", "b"], {"a", "b", "c"}, k=2) == 1 / 3``.
    Raises ``ValueError`` if ``relevant`` is empty or ``k < 1``.
    """
    _check_k(k)
    relevant = _relevant_set(relevant)
    hits = sum(1 for doc in dedupe(retrieved)[:k] if doc in relevant)
    return hits / len(relevant)


def reciprocal_rank(retrieved: Sequence[str], relevant: Iterable[str]) -> float:
    """``1 / rank`` of the first relevant id (1-based, after de-duplication), else ``0``.

    Averaged over queries this is MRR. ``reciprocal_rank(["x", "a"], {"a"}) == 0.5``.
    Raises ``ValueError`` if ``relevant`` is empty.
    """
    relevant = _relevant_set(relevant)
    for rank, doc in enumerate(dedupe(retrieved), start=1):
        if doc in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Normalized discounted cumulative gain at ``k`` with binary gains.

    ``DCG = sum(1 / log2(rank + 1))`` over relevant ids in the top ``k``; it is divided
    by the DCG of an ideal ranking (all relevant ids first, at most ``k`` of them), so
    the result is in ``[0, 1]`` and rewards putting relevant documents higher.
    Raises ``ValueError`` if ``relevant`` is empty or ``k < 1``.
    """
    _check_k(k)
    relevant = _relevant_set(relevant)
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, doc in enumerate(dedupe(retrieved)[:k], start=1)
        if doc in relevant
    )
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
    return dcg / ideal
