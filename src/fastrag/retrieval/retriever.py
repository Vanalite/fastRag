"""The retriever: the only component that queries indexes on a caller's behalf."""

from __future__ import annotations

import logging

from fastrag.domain import Chunk, RetrievalResult, ScoredChunk
from fastrag.embeddings import Embedder
from fastrag.retrieval.fusion import reciprocal_rank_fusion
from fastrag.retrieval.rerank import Reranker
from fastrag.security.acl import AccessFilter
from fastrag.storage.indexes import SearchIndexes

log = logging.getLogger(__name__)

DEFAULT_MIN_DENSE_SCORE = 0.35


class Retriever:
    def __init__(
        self,
        embedder: Embedder,
        indexes: SearchIndexes,
        *,
        candidate_pool: int = 20,
        rrf_k: int = 60,
        min_dense_score: float | None = None,
        min_keyword_coverage: float = 0.3,
        min_keyword_matches: int = 2,
        reranker: Reranker | None = None,
        rerank_pool: int = 20,
    ):
        self.embedder = embedder
        self.indexes = indexes
        self.candidate_pool = candidate_pool
        self.rrf_k = rrf_k
        # Cosine scales differ wildly between embedders, so by default the threshold
        # comes from the embedder itself (calibrated on the golden eval set).
        self.min_dense_score = (
            min_dense_score
            if min_dense_score is not None
            else getattr(embedder, "default_min_score", DEFAULT_MIN_DENSE_SCORE)
        )
        self.min_keyword_coverage = min_keyword_coverage
        self.min_keyword_matches = min_keyword_matches
        self.reranker = reranker
        self.rerank_pool = rerank_pool

    def retrieve(self, query: str, access: AccessFilter, k: int) -> RetrievalResult:
        pool = max(k, self.candidate_pool)
        dense = self.indexes.dense(self.embedder.embed([query])[0], pool, access=access)
        keyword = self.indexes.keyword(query, pool, access=access)

        chunks: dict[str, Chunk] = {}
        dense_scores: dict[str, float] = {}
        keyword_scores: dict[str, tuple[float, float, int]] = {}
        for chunk, score in dense:
            chunks[chunk.id] = chunk
            dense_scores[chunk.id] = score
        for chunk, score, coverage, matched in keyword:
            chunks[chunk.id] = chunk
            keyword_scores[chunk.id] = (score, coverage, matched)

        fused = reciprocal_rank_fusion(
            [[c.id for c, _ in dense], [c.id for c, *_ in keyword]], k=self.rrf_k
        )
        # With a reranker, collect a wider pool for it to reorder, then cut to k.
        limit = max(k, self.rerank_pool) if self.reranker is not None else k
        results: list[ScoredChunk] = []
        seen_texts: set[str] = set()
        for chunk_id, score in fused:
            if len(results) == limit:
                break
            chunk = chunks[chunk_id]
            # Defence in depth: indexes already pre-filter, but a buggy adapter must not
            # be able to leak data. Drop and log loudly rather than trust it.
            if not access.allows(chunk.tenant_id, chunk.acl):
                log.critical(
                    "index returned a chunk outside the caller's access scope: %s", chunk_id
                )
                continue
            # The same passage uploaded twice would only crowd out other evidence, so
            # keep the best-ranked copy. Only *identical* text counts (case and spacing
            # aside): near copies may differ in exactly the fact that matters ("30
            # days" vs "90 days"), and silently hiding one would let anyone with write
            # access suppress an authoritative passage with a doctored copy.
            normalized = " ".join(chunk.text.lower().split())
            if normalized in seen_texts:
                log.debug("dropped duplicate chunk %s (%s)", chunk_id, chunk.source_id)
                continue
            seen_texts.add(normalized)
            kw = keyword_scores.get(chunk_id)
            results.append(
                ScoredChunk(
                    chunk=chunk,
                    score=score,
                    dense_score=dense_scores.get(chunk_id),
                    keyword_score=kw[0] if kw else None,
                    keyword_coverage=kw[1] if kw else None,
                    keyword_matches=kw[2] if kw else None,
                )
            )
        if self.reranker is not None:
            try:
                reranked = self.reranker.rerank(query, results)
            except Exception:  # noqa: BLE001 - a quality step must not fail the request
                log.exception("reranker failed; using the fused order")
                reranked = results
            allowed = {r.chunk.id for r in results}
            # A reranker may only reorder: anything it invents is dropped.
            results = [r for r in reranked if r.chunk.id in allowed][:k]
        return RetrievalResult(chunks=results, confident=any(map(self.supports, results)))

    def supports(self, result: ScoredChunk) -> bool:
        """Is this chunk real evidence for the query, rather than a nearest-but-unrelated hit?

        RRF scores are rank-based and say nothing about relevance, so this looks at the
        raw signals: a close semantic match, or enough of the query's informative terms
        present (IDF-weighted coverage). Unless every query term matched, at least two
        must, so one lucky word can't pass as evidence.
        """
        if (result.dense_score or 0.0) >= self.min_dense_score:
            return True
        coverage, matches = result.keyword_coverage or 0.0, result.keyword_matches or 0
        return coverage >= self.min_keyword_coverage and (
            matches >= self.min_keyword_matches or coverage >= 1.0 - 1e-9
        )
