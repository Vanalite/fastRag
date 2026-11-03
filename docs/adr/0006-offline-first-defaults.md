# ADR 0006: Offline-first defaults, pluggable adapters, and evaluation as a regression gate

- **Status:** Accepted
- **Date:** 2026-11-03

## Context

A reference implementation has to run on a laptop with no accounts, no API keys and no network, so a reader can run it, read it and test it. It also has to show where the production components plug in. Two risks follow. A reader may take the offline numbers as evidence of production quality. Or the offline defaults may hide the places where real adapters need care.

## Decision

**Defaults need nothing external.** From `src/fastrag/config.py` and the package code:

| Concern | Default | Real option |
|---|---|---|
| Document store | SQLite file, `.fastrag/fastrag.db` (`config.py:16`) | Postgres (not implemented) |
| Embedder | `hashing` (`config.py:17`, `embeddings/hashing.py`): feature hashing over words and word pairs, deterministic, no model | `st:<model>` (`embeddings/sentence_transformers.py`), optional extra |
| Vector and keyword indexes | In-process NumPy and BM25 (`storage/vector.py`, `storage/bm25.py`) | pgvector, OpenSearch (not implemented) |
| Generator | `extractive` (`config.py:33`, `generation/extractive.py`): quotes the top passages with citations. No LLM call. | `claude` (`generation/claude.py`), optional extra |
| Cache, rate limiter | In-process (`cache.py`, `ratelimit.py`) | Redis (not implemented) |
| Audit | JSONL file (`config.py:46`) | WORM storage or a SIEM through a new `AuditSink` |
| API keys | JSON file with hashes (`config.py:40`) | Identity provider with OIDC or JWKS (not implemented) |

With these defaults, `fastrag ingest`, `fastrag ask` and `fastrag serve` work offline. `pip install 'fastrag[claude]'` or `[st]` adds the optional backends.

**Adapters sit behind small protocols.** `Embedder` (`embeddings/base.py`), `DocumentStore`, `VectorIndex`, `KeywordIndex` (`storage/base.py`), `Generator` (`generation/base.py`) and `AuditSink` (`observability/audit.py`) are `typing.Protocol` classes. `RAGService` takes them as constructor arguments (`service.py:65-77`), so a test or an adapter can swap one without touching the rest. The `storage/base.py` docstring is the contract for a new store.

**Evaluation is a regression gate, not a quality claim.** `fastrag eval` (`src/fastrag/eval/__init__.py`) ingests the golden dataset in `eval/golden/` into a fresh in-memory service. It asks each question as its own principal and reports:

- recall@k, MRR and nDCG@k over answerable questions, document-level and binary;
- abstention accuracy, for out-of-scope questions and for questions about documents the asker may not read;
- ACL leaks, which must always be zero. The dataset manifest (`eval/golden/corpus.json`) gives each file's ACL.

Exit codes are 0 for pass, 1 for a gate failure and 2 for an error. The thresholds are set a little below the scores measured with the default hashing embedder. The module's own docstring says they are not a quality claim, and that they should be raised as retrieval improves and never lowered to get CI green. The coordinator's most recent measurement with the hashing embedder was recall@4 1.0, MRR 0.98 and nDCG@4 0.985 on the ten-document, 32-question set.

## Consequences

Good:

- A new reader can run the whole pipeline in minutes, and the tests run without network access.
- Each production component has a named seam, so the reference shows where to put a real one.
- The regression gate catches a change that breaks retrieval, the ACL filter or abstention, including a change that breaks isolation.

Costs and risks:

- **The offline numbers are not production numbers.** A ten-document corpus with hashing embeddings is easy. The scores say that nothing got worse. They say nothing about how good retrieval would be on another corpus. Report them as "no regression against the golden set", not as accuracy.
- **Thresholds can drift.** A threshold that is never raised stops telling you anything once the system improves. Re-measure after each retrieval change.
- **Defaults can be mistaken for recommendations.** The extractive generator and the hashing embedder are there so the code runs offline. Neither is a recommended production choice. `docs/security.md` lists the changes a production deployment needs.
- **Adapter contracts are only as strong as their tests.** The protocols do not enforce the ACL rule. The mandatory `access` argument and the retriever's post-check do, and the isolation tests exercise them (see ADR 0002).

## Alternatives considered

- **Require an LLM and a hosted embedding service from the start.** Matches production more closely, but makes the reference unrunnable without accounts and costs money per test run. Rejected for defaults. The optional backends cover the production path.
- **Ship with Docker Compose for Postgres, Redis and OpenSearch.** Realistic, but it turns a reference into an infrastructure project and hides the logic in configuration. Possible as a later addition, once the protocols settle.
- **Use a quality benchmark (for example, a public retrieval dataset) as the gate.** Gives comparable numbers, but a public dataset does not share your documents or ACLs. The golden set tests the properties that matter here: isolation and abstention, as well as ranking.
- **Keep the evaluation out of the CLI.** Would make it easy to skip. Exposing it as `fastrag eval`, with an exit code, lets CI run it on every change.
