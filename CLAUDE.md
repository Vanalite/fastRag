# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

v0.1 is implemented: the HTTP API, the ingestion worker, the document and listings paths, connectors, the quality-check runner, Docker Compose and CI. It has not yet been run against the pilot's real documents and questions (D4), and Q12 is still awaiting an answer. Keep the **Commands** and **Architecture** sections below in step with the code.

## Project goal

fastRag is meant to be a production-ready Retrieval-Augmented Generation (RAG) service for small and medium-sized businesses. It should be easy to install, run, and attach to an AI agent platform. Two main use cases drive the design:

1. **Customer-service agents** that answer customer questions from a company's own domain knowledge (docs, FAQs, policies, product data).
2. **Data/automation pipelines** where retrieval is one step in a larger multi-step workflow.

Retrieval is **hybrid-first**: keyword search (BM25) and vector search, followed by a reranker. A LightRAG-style knowledge-graph layer is optional, and it is kept only where the quality checks show it beats hybrid retrieval. fastRag returns evidence (passages, citations, evidence signals) and leaves answer generation, guardrails and hand-off to a human to the consuming agent.

**`docs/design-log.md` is the source of truth for design decisions.** Read it before making architectural changes, and add to it when a decision is made or reversed. Key constraints recorded there:
- The pilot is a Vietnamese real-estate business, so Vietnamese-language handling is a first-class concern.
- Spreadsheet listings use a separate, typed path with filter queries.
- Data is organised as tenant → collections.
- Ingestion is connector-based sync (Google Sheets/Drive, Excel).

## Design constraints implied by the goal

- **English only:** all product and code is written in English, whatever the market or client. This covers UI and display text, docs, README and API examples, comments, identifiers and test names. Add another language, even a single display string, only when the owner explicitly asks for it. Never infer a UI language from the pilot client or its data.
- **Programming language:** Python (implied by the `.gitignore`).
- **Easy install and run** for non-expert operators. Prefer a single-command start, such as `docker compose up` or a `pip install` plus a CLI. Use sensible defaults and keep required configuration to a minimum.
- **Agent-platform pluggability:** expose retrieval through stable, platform-neutral interfaces, for example an HTTP API and/or an MCP server or tool schema. Don't couple the code to one agent framework.
- **Swappable backends:** keep the LLM provider, embedding model, reranker, and storage behind interfaces, so an SMB can choose hosted or self-hosted options.
- **Multi-tenant awareness:** one deployment may serve several businesses or knowledge bases, so isolate data per tenant or collection.

## Commands

Python 3.11+ managed with `uv`. Database tests need Postgres with pgvector. The test suite drops and recreates its own database.

```bash
uv sync                                        # install with dev tools (add --extra mcp etc. as needed)
docker run -d --name fastrag-db -p 5432:5432 -e POSTGRES_USER=fastrag -e POSTGRES_PASSWORD=fastrag \
  -e POSTGRES_DB=fastrag pgvector/pgvector:pg17
export FASTRAG_TEST_DATABASE_URL=postgresql+asyncpg://fastrag:fastrag@localhost:5432/fastrag_test
uv run pytest                                  # full suite (DB tests are skipped if Postgres is unreachable)
uv run pytest tests/test_values.py::test_parse_price     # single test
uv run ruff check src tests && uv run ruff format src tests
uv run mypy
uv run fastrag migrate                         # apply Alembic migrations (src/fastrag/db/migrations)
uv run fastrag serve                           # API on :8000
uv run fastrag worker                          # ingestion worker + sync scheduler
uv run fastrag bootstrap pilot                 # tenant, two collections, upload sources, three keys
uv run fastrag ingest-settings <tenant> <collection> [--set JSON]   # per-collection parser/chunking/context (D25)
uv run fastrag import <source-id> <folder> [--dry-run|--inline]     # bulk import with cost estimate and report
uv run fastrag chunk-preview <file> [--tenant T --collection C] [--set JSON]
uv run fastrag compare <tenant> <collection> refs.jsonl --set JSON  # trial copy with other settings, eval on both
uv run fastrag eval <tenant> refs.jsonl        # release checks (exit 1 on failure)
uv run fastrag calibrate <tenant> refs.jsonl   # fit reranker score calibration (D21)
uv run python scripts/bench_search.py --chunks 50000   # DB-side latency benchmark (D22); --reuse to re-query
docker compose up -d                           # full stack incl. TEI embeddings/reranker (x86-64 images)
(cd integrations/dsh-tools && node --test test/*.test.mjs)   # DeepSeek Harness tools (Node >= 20.3)
```

To add a migration, change `db/models.py` and run `uv run python -c "from fastrag.db.migrate import revision; revision('msg')"` against a migrated database, then review the generated file. The vector dimension comes from `FASTRAG_EMBEDDING_DIM`.

## Architecture

The flow is: connectors/upload → `documents` row and stored file → worker job → parse → normalize (Vietnamese) → chunk → context header → hygiene flags → keyword terms + embedding → `chunks`. Retrieval then runs BM25 ∥ vector → RRF → rerank → evidence signals. Spreadsheets take a separate path: sheet → column mapping → typed `listings` rows → `query_listings`.

- `text/vietnamese.py`: NFC, TCVN3 repair, canonical tone placement, segmentation, and `keyword_terms`. Every word is indexed as the segmented word, its syllables, and an accent-free `~` copy. Index time and query time must use the same functions.
- `text/values.py` + `gazetteer.yaml`: deterministic parsing of prices, areas, districts and wards. It raises `ValueError` and never guesses (D13). The same code runs on spreadsheet cells and on filter arguments. District codes also become `@code` keyword terms.
- `ingest/settings.py` (D25): per-collection `settings.ingest` overrides resolved against the env defaults into an `IngestConfig` with a fingerprint. Each document stores it in `report.ingest_fingerprint`; a different fingerprint means stale. `ingest_document` → `prepare_document` (parse/OCR/normalize/chunk, writes nothing) → `chunk_contexts`, and the dry run, chunk preview and `eval/compare.py` reuse both. `ingest/bulk.py`: folder scan, dry-run plan, import (queued or inline) and progress. Trial collections carry `settings.trial_of` and are left out of default collection selection.
- `ingest/`: `parsers.py` (file → sections), `chunking.py` (token budgets; `structure` or `window` strategy), `contextual.py` (title/heading header plus an optional LLM summary), `hygiene.py` (flags), `documents.py` (pipeline), `index.py` (writes chunks and keeps the BM25 stats `term_stats` and `collection_stats` in step; always go through it), `listings.py` + `sheets.py` (snapshot import), `sync.py` (file registration and source sync).
- `retrieval/search.py`: BM25 is computed in SQL from the `chunk_postings` inverted index and `term_stats`, within a per-profile postings budget (rarest terms first). Vector search uses pgvector HNSW with iterative scan. Both run concurrently. If the reranker fails, results fall back to fused order and signals report `degraded`. `retrieval/listings.py`: field-level masking and filtering in SQL through `field_groups`.
- **Access control:** `effective_groups` = request ∩ API-key ceiling ∩ profile ceiling. Every SQL query filters on `tenant_id`, `collection_id` and `groups &&`. Never filter results in Python after the query. New content defaults to `staff` (`DEFAULT_GROUPS`).
- `retrieval/profiles.py`: `customer`, `handoff_copilot` and `internal_search` are configuration. A collection may override values but cannot lift `max_groups`.
- `worker/`: Postgres job queue (`FOR UPDATE SKIP LOCKED`, at most one queued job per dedupe key) plus a scheduler for `sources.sync_interval_minutes`.
- `api/`: `retrieval_routes.py` (agent-facing, versioned schema in `schemas.py`), `admin_routes.py`, `auth.py` (hashed `fr_` tokens plus the operator `FASTRAG_ADMIN_TOKEN` with the `X-Tenant` header). `mcp_server.py` wraps the HTTP API as MCP tools. `integrations/dsh-tools/` is the zero-dependency JS package that registers the same tools in DeepSeek Harness, built with the host's own `defineTool` (D24). Keep the tool names, descriptions and request fields in step across MCP, `GET /v1/tools` and this package.
- `providers/reranker.py`: every reranker is wrapped in `CalibratedReranker` (Platt scaling, D21), so profile thresholds share one scale. `eval/calibrate.py` fits the parameters.
- `api/static/admin.html`: the staff admin page served at `/admin`, which calls the admin API. `metrics.py`: Prometheus metrics (API `/metrics`, worker `--metrics-port`).
- Performance rules (D22): never add a GIN index on `chunks.groups` (it makes the planner skip HNSW), keep JIT off, and measure with `scripts/bench_search.py` before and after query changes.
- `services.py`: providers (embedder, reranker, LLM, OCR, storage) built once and injected. Tests swap in `HashEmbedder` and a fake reranker (`tests/conftest.py`).
