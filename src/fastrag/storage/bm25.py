"""In-memory BM25 keyword index, partitioned by tenant.

Dense embeddings miss exact identifiers (error codes, product SKUs, policy numbers);
BM25 catches them. Term statistics are per tenant, so one tenant's corpus never
shifts another tenant's scores.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable

from fastrag.domain import Chunk
from fastrag.security.acl import AccessFilter

_TOKEN = re.compile(r"\w+", re.UNICODE)
STOPWORDS = frozenset(
    (  # noqa: SIM905 - a word list reads better as one string
        "a about all am an and any are as at be been being but by can could did do does for "
        "from get gets got had has have he her him his how i if in into is it its just many me "
        "mean much must my need of on or our she should so than that the their them then there "
        "they this to us was we were what when where which who why will with would you your"
    ).split()
)


def _normalize(term: str) -> str:
    """Very light stemming: fold simple plurals ("laptops" -> "laptop").

    Applied identically at index and query time. A real deployment would use a proper
    stemmer or lemmatizer (Snowball, or the search engine's analyzer) per language.
    """
    if len(term) > 4 and term.endswith("ies"):
        return term[:-3] + "y"
    if len(term) > 3 and term.endswith("s") and not term.endswith(("ss", "us", "is")):
        return term[:-1]
    return term


def tokenize(text: str) -> list[str]:
    return [_normalize(t) for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


def _indexed_terms(chunk: Chunk) -> list[str]:
    # The title is indexed too, matching the dense side (which embeds title + text).
    return tokenize(f"{chunk.title}\n{chunk.text}")


class _Partition:
    def __init__(self) -> None:
        self.postings: dict[str, dict[str, int]] = {}
        self.lengths: dict[str, int] = {}
        self.chunks: dict[str, Chunk] = {}
        self.total_length = 0

    def add(self, chunk: Chunk) -> None:
        if chunk.id in self.chunks:
            return
        counts = Counter(_indexed_terms(chunk))
        for term, tf in counts.items():
            self.postings.setdefault(term, {})[chunk.id] = tf
        length = sum(counts.values())
        self.lengths[chunk.id] = length
        self.total_length += length
        self.chunks[chunk.id] = chunk

    def remove(self, chunk_id: str) -> None:
        chunk = self.chunks.pop(chunk_id, None)
        if chunk is None:
            return
        for term in set(_indexed_terms(chunk)):
            postings = self.postings.get(term)
            if postings is not None:
                postings.pop(chunk_id, None)
                if not postings:
                    del self.postings[term]
        self.total_length -= self.lengths.pop(chunk_id)


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._partitions: dict[str, _Partition] = {}
        self._tenant_of: dict[str, str] = {}

    def add(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            self._partitions.setdefault(chunk.tenant_id, _Partition()).add(chunk)
            self._tenant_of[chunk.id] = chunk.tenant_id

    def remove(self, chunk_ids: Iterable[str]) -> None:
        for chunk_id in chunk_ids:
            tenant_id = self._tenant_of.pop(chunk_id, None)
            if tenant_id is not None:
                self._partitions[tenant_id].remove(chunk_id)

    def search(
        self, query: str, k: int, *, access: AccessFilter
    ) -> list[tuple[Chunk, float, float, int]]:
        """Return (chunk, bm25_score, coverage, matched_terms), best first.

        ``coverage`` is the IDF-weighted share of the query's distinct terms found in the
        chunk: matching the rare, informative words counts for much more than matching
        common ones. Terms that appear nowhere in the corpus get the maximum weight, so a
        question about something the corpus doesn't cover scores low.
        """
        partition = self._partitions.get(access.tenant_id)
        terms = sorted(set(tokenize(query)))
        if partition is None or not terms or k <= 0 or not partition.chunks:
            return []

        n = len(partition.chunks)
        avg_len = partition.total_length / n if n else 1.0

        def idf(df: int) -> float:
            return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

        weights = {t: idf(len(partition.postings.get(t, ()))) for t in terms}
        total_weight = sum(weights.values())
        scores: dict[str, float] = {}
        matched: dict[str, int] = {}
        matched_weight: dict[str, float] = {}
        for term in terms:
            postings = partition.postings.get(term)
            if not postings:
                continue
            term_idf = weights[term]
            for chunk_id, tf in postings.items():
                chunk = partition.chunks[chunk_id]
                if not access.allows(chunk.tenant_id, chunk.acl):  # pre-filter
                    continue
                norm = tf + self.k1 * (1 - self.b + self.b * partition.lengths[chunk_id] / avg_len)
                scores[chunk_id] = scores.get(chunk_id, 0.0) + term_idf * tf * (self.k1 + 1) / norm
                matched[chunk_id] = matched.get(chunk_id, 0) + 1
                matched_weight[chunk_id] = matched_weight.get(chunk_id, 0.0) + term_idf

        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        return [
            (partition.chunks[cid], score, matched_weight[cid] / total_weight, matched[cid])
            for cid, score in ranked
        ]

    def chunk_ids(self) -> set[str]:
        return set(self._tenant_of)
