# ADR 0003: Hybrid retrieval with reciprocal rank fusion and a raw-signal confidence gate

- **Status:** Accepted
- **Date:** 2026-11-03

## Context

Dense embeddings match meaning but miss exact identifiers: error codes, SKUs, policy numbers. Keyword search (BM25) finds identifiers but misses paraphrases. Enterprise questions use both. The system also needs to decide when the evidence is too weak to answer at all, because the answer then has to be an abstention, not a guess.

## Decision

**Two retrievers over the same candidate pool.** `Retriever.retrieve` (`src/fastrag/retrieval/retriever.py:34-80`) asks each index for `max(k, candidate_pool)` results, both pre-filtered by the caller's `AccessFilter`:

- Dense: cosine similarity from the vector index. Embeddings are L2-normalized (`embeddings/base.py`), so the dot product is the cosine.
- Keyword: BM25 with `k1=1.5` and `b=0.75` (`storage/bm25.py:64`). Terms are lower-cased, stop words removed and simple plurals folded (`laptops` to `laptop`); titles are indexed along with the text. The index also returns **keyword coverage**: the IDF-weighted share of distinct query terms present in the chunk, so matching rare, informative words counts for more than matching common ones, and terms absent from the corpus count against a match.

**Fusion by rank.** `reciprocal_rank_fusion` (`src/fastrag/retrieval/fusion.py:13-17`) scores each chunk as the sum over lists of `1 / (rrf_k + rank)`, with `rrf_k = 60` by default. The result is ordered, then truncated to `k`.

**Confidence from raw signals, not from the fused score.** A chunk counts as evidence when it passes either signal:

- dense: `dense_score >= min_dense_score`. The threshold defaults to the embedder's own calibrated value: 0.25 for hashing, 0.35 for sentence-transformers.
- keyword: `keyword_coverage >= min_keyword_coverage` (default 0.3), and at least two matched query terms when the query has two or more.

Chunks that pass neither signal are pruned before generation. Confidence is recomputed after suspicious chunks are dropped. When no chunk is confident, the service returns the fixed abstention text without calling the LLM.

## Why the gate does not use the RRF score

- An RRF score is `1 / (60 + rank)`. A rank-1 hit scores about 0.0164 whether it matched strongly or weakly. The score carries only the position in each list, not how good the match was. Thresholding it would make the gate depend on how many candidates happened to exist.
- BM25 raw scores grow with corpus size and document length, so no fixed threshold works across tenants or across time.
- **Keyword coverage** is a fraction of query terms, so it is scale-free. "Most of the query's words appear in this chunk" means the same thing in every corpus.
- **Dense cosine** has a fixed scale, but its distribution depends on the embedder. Each embedder therefore declares its own default (`HashingEmbedder.default_min_score` 0.25, sentence-transformers 0.35), and the setting can override it. Thresholds are checked against the golden set (`eval/golden/`) through `fastrag eval`.

## Consequences

Good:

- The gate has two independent signals. A query can pass on exact-term matches with no semantic match, or on a strong semantic match with no shared words.
- The fused order is stable and needs no per-corpus tuning beyond `rrf_k`.

Costs:

- **Check the gate's input set after the change lands.** Before the pending change, the gate read only `fused[:k]`, so a strong candidate ranked below `k` was ignored. Pruning now happens on the candidate set before generation. Confirm whether the gate itself still sees only the top `k`, and if so, decide whether that is intended.
- Thresholds are tied to an embedder. Changing the embedder changes how often the system abstains, so run `fastrag eval` after any change.
- With the hashing embedder, dense scores are weak for paraphrases, so the keyword signal does most of the work by default.

## Alternatives considered

- **Weighted sum of normalized scores.** Min-max normalization per query is unstable on small candidate pools, and the weights need tuning per corpus. Rejected for the reference.
- **Cross-encoder reranking.** Better precision, but a model call per candidate. Not implemented; it is the natural next stage after fusion.
- **Maximal marginal relevance (MMR).** Reduces near-duplicate chunks in the context. Not implemented. Duplicates are less harmful here because the citation check still applies.
- **Dense only.** Simpler, but fails on identifiers, which are common in enterprise corpora.
- **Keyword only.** Fails on paraphrases.
- **A single hybrid index (for example, a sparse-dense vector in one store).** Equally valid. It moves the fusion into the database, and the gate would still need the raw signals.
