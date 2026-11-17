# fastRag

fastRag is a retrieval service for AI agents at small and medium-sized businesses. Your agent sends it a question, and fastRag returns the passages and listings that answer it, along with citations and evidence signals. Writing the answer, guardrails and hand-off to a human stay in your agent.

- **Hybrid document search:** BM25 keyword search over Vietnamese word-segmented text, plus vector search (BGE-M3 on pgvector), merged with reciprocal rank fusion and then reranked.
- **Typed listings:** spreadsheets (Excel, CSV, Google Sheets) become typed rows that you query with fixed filters. Values such as `"3,5 tỷ"`, `"Q7"` and `"72m2"` are standardized. A filter fastRag cannot read is rejected, not guessed.
- **Access control in SQL:** every passage, listing row and listing column has access groups. Customer callers see `public` content only. This is enforced inside the database query.
- **Evidence signals:** each result carries calibrated scores, coverage, possible conflicts, staleness, `nothing_relevant`, and a trace ID.
- **Vietnamese first:** Unicode NFC, repair of TCVN3 legacy text, one tone-mark style, matching with and without diacritics, and district aliases (Q7 = Quận 7 = District 7).
- **Connector sync:** manual upload, a local or shared folder, web pages, and Google Drive/Sheets, on a schedule or via a webhook. Unchanged files are skipped by content hash.

The design decisions behind all of this are in [`docs/design-log.md`](docs/design-log.md). Running it in production (go-live checklist, backups, monitoring, runbook) is covered in [`docs/operations.md`](docs/operations.md).

## Quick start (Docker)

```bash
cp .env.example .env          # set FASTRAG_ADMIN_TOKEN and POSTGRES_PASSWORD
docker compose up -d          # db, migrate, api, worker, embeddings, reranker
docker compose exec api fastrag bootstrap pilot
```

`bootstrap` creates a tenant with a `docs` collection and a `listings` collection, an upload source for each, and three API keys: `customer-agent`, `staff` and `admin`. The tokens are printed once.

The first start downloads BGE-M3 and bge-reranker-v2-m3 (about 4 GB). The TEI model containers are x86-64 only. On Apple Silicon they run under emulation and slowly, so for local development you can point `FASTRAG_EMBEDDING_PROVIDER=openai` at any hosted endpoint serving BGE-M3.

Staff can do everything below in the admin page at **`http://<host>:8000/admin`** (in Vietnamese). It covers uploading files to the public or internal folder, the review queue, marking spreadsheet columns public, document status, jobs, and a test search. The page asks for an admin token. The `curl` equivalents follow.

### Add content

```bash
ADMIN="Authorization: Bearer <admin token>"

# Documents. Files under public/ are visible to customers; everything else is staff-only.
curl -H "$ADMIN" -F file=@faq.pdf -F path=public  localhost:8000/v1/admin/sources/<docs source id>/files
curl -H "$ADMIN" -F file=@chinh-sach-noi-bo.docx  localhost:8000/v1/admin/sources/<docs source id>/files

# Listings spreadsheet
curl -H "$ADMIN" -F file=@bang-gia.xlsx -F path=public localhost:8000/v1/admin/sources/<listings source id>/files
```

Every spreadsheet column starts out **internal**. Open the suggested column mapping, mark the columns customers may see, and confirm it. The affected sheets are then re-imported automatically:

```bash
curl -H "$ADMIN" "localhost:8000/v1/admin/mappings?collection=listings"
curl -H "$ADMIN" -X PUT localhost:8000/v1/admin/mappings/<id> -H 'content-type: application/json' -d '{
  "columns": {
    "Mã căn":        {"field": "listing_code", "groups": ["public", "staff"]},
    "Giá bán (tỷ)":  {"field": "price_vnd",    "groups": ["public", "staff"], "unit": "ty"},
    "Quận":          {"field": "district_code", "groups": ["public", "staff"]},
    "Giá sàn":       {"field": "attr:gia-san",  "groups": ["staff"]}
  }}'
```

### Query

```bash
CUSTOMER="Authorization: Bearer <customer-agent token>"

curl -H "$CUSTOMER" localhost:8000/v1/search -H 'content-type: application/json' \
  -d '{"query": "Phí quản lý chung cư bao nhiêu?", "profile": "customer"}'

curl -H "$CUSTOMER" localhost:8000/v1/listings/query -H 'content-type: application/json' \
  -d '{"filters": {"district": "Q7", "price_max": "3,5 tỷ", "bedrooms_min": 2}, "sort": "price_asc"}'
```

`GET /v1/tools` returns JSON-Schema tool definitions (`search_documents`, `query_listings`, `get_listings`) that you can register in any function-calling agent framework. The full API is listed at `/docs`.

### Connect an agent

The same three tools are available in three ways:

| Agent platform | Use |
|---|---|
| DeepSeek Harness (`dsh`) and hosts built on `@deepseek-ai/dsh-tools`, such as any-agent | [`integrations/dsh-tools`](integrations/dsh-tools/README.md): a zero-dependency JS package. Call `createFastragTools(defineTool, config)` inside your tools plugin, or load its `plugin.mjs` as a profile row. |
| MCP clients (Claude Desktop, IDE agents) | `fastrag mcp` with `FASTRAG_MCP_API_URL` and `FASTRAG_MCP_TOKEN` (needs the `mcp` extra). |
| Anything with function calling | `GET /v1/tools` for the schemas, then call the HTTP endpoints. |

## Concepts

| Concept | What it is |
|---|---|
| **Tenant** | One business. API keys belong to one tenant. |
| **Collection** | Either `documents` (PDF, DOCX, HTML, Markdown, text) or `listings` (spreadsheets). Each collection has its own settings. |
| **Source** | Where files come from: `upload`, `local_folder`, `web` or `google_drive`. It has default access groups, plus `folder_groups` that map sub-folders to other groups. |
| **Access groups** | For example `public` and `staff`. Content is visible when its groups overlap the caller's groups. Anything without a mapping is `staff`. |
| **API key** | Carries a ceiling on access groups and, optionally, on profiles. A key issued with `["public"]` can never read staff content, whatever the request asks for. |
| **Profile** | Retrieval settings per surface: `customer` (public only, strict, quarantined passages hidden), `handoff_copilot`, and `internal_search` (broader recall). A collection can override the settings under `settings.profiles`. |

### Response shape (`schema_version` 1.0)

```json
{
  "trace_id": "…", "profile": "customer", "groups": ["public"],
  "passages": [{
    "id": "chunk:42", "text": "…", "context": "FAQ › Phí quản lý", "score": 0.91,
    "scores": {"rerank": 0.91, "similarity": 0.62, "keyword_rank": 1, "vector_rank": 2, "fused": 0.032},
    "source": {"document_id": "…", "title": "FAQ", "uri": "…", "version": 3, "page_start": 2, "modified_at": "…"},
    "flags": [], "age_days": 12, "stale": false
  }],
  "signals": {
    "nothing_relevant": false, "top_score": 0.91, "score_gap": 0.3, "coverage": 0.8,
    "distinct_documents": 2, "conflicts": [], "stale_passages": 0, "reranker": "ok", "degraded": []
  }
}
```

fastRag reports facts. Your agent decides what they mean: it may refuse when `nothing_relevant` is true, hand off when `conflicts` is non-empty, and so on. Listing results always include `as_of`, so the agent can say how fresh a price or status is. For a follow-up such as "that unit", pass the listing `id` back to `/v1/listings/lookup`. The ID stays the same across syncs.

### Hygiene and review

Ingestion flags three kinds of passage: possible prompt injection, personal data (phone numbers, emails, ID and bank numbers), and internal business terms ("giá chốt", "hoa hồng") found in public content. Flagged passages are hidden from the `customer` profile until an admin reviews them with `GET /v1/admin/review/chunks` and `POST /v1/admin/review/chunks/clear`. Sensitive cells in public spreadsheet columns stay internal.

## Quality checks

```bash
fastrag eval pilot references.jsonl --baseline ragflow-results.jsonl --out report.json
```

The command runs three release checks and exits with status 1 if any of them fails:

1. **Access safety:** no customer-scope result may carry a non-public group.
2. **Listing accuracy:** every listing filter check gives the expected standardized values.
3. **Document retrieval:** Recall@k and MRR. If you pass a baseline (for example RAGFlow results), fastRag must not score lower.

The docstring in `src/fastrag/eval/runner.py` describes the file format.

Reranker scores are calibrated, so that thresholds such as `min_score` mean the same across models. Refit the calibration on your reference set whenever you change the reranker:

```bash
fastrag calibrate pilot references.jsonl     # prints FASTRAG_RERANKER_CALIBRATION=a,b
```

Measured so far (design log D21, D22): with BGE-M3 and bge-reranker-v2-m3, held-out Recall@10 is 0.875, and database-side latency for customer searches at 50,000 chunks is 68 ms (p95). Run `scripts/bench_search.py` to measure your own hardware. Thumbs-down feedback (`POST /v1/feedback`) and hand-offs are logged with their trace ID, and `GET /v1/admin/search-logs?feedback=down` lists them as candidate reference questions.

## Loading a large corpus

Every document goes through the same steps: parse, OCR scanned pages, normalize the Vietnamese text, chunk, add context to each chunk, flag problems, then index it for keyword and vector search. Each documents collection can choose how parsing, chunking and context work. Here is a typical workflow:

```bash
# 1. Choose settings for the collection (only the keys you give are overridden).
fastrag ingest-settings pilot docs --set '{
  "chunking": {"strategy": "structure", "target_tokens": 250, "max_tokens": 400, "overlap_tokens": 30},
  "context":  {"mode": "llm", "llm": "cheap", "instructions": "Name the project the passage is about."}
}'

# 2. See how a file would be chunked with them. Nothing is written.
fastrag chunk-preview handbook.pdf --tenant pilot --collection docs --show 5

# 3. Dry run: new, changed and unchanged files, estimated chunks, embedding tokens and LLM cost.
fastrag import <docs source id> ./company-docs --dry-run

# 4. Import. Sub-folders become paths in the source, so folder_groups decide access (public/... is customer-visible).
fastrag import <docs source id> ./company-docs --report import-report.json            # queued for the worker
fastrag import <docs source id> ./company-docs --inline --concurrency 8                # or process here, no worker needed
```

- **Re-running an import is safe.** Unchanged files are skipped by content hash, so an interrupted import continues where it stopped. Files that are unchanged but were built with older settings are re-processed; pass `--no-reprocess-stale` to skip them.
- **Filters and folders:** `--include` / `--exclude` take glob patterns, and `--prefix` imports into a sub-folder of the source. Unsupported files are listed in the report, not uploaded.
- **Settings changes re-process automatically.** Changing a collection's ingestion settings (CLI or `PATCH /v1/admin/collections/<slug>`) queues a re-process of only the documents built with other settings. `GET /v1/admin/collections/<slug>/ingest` shows the effective settings and how many documents are stale.

| Setting (`settings.ingest`) | Values | Default |
|---|---|---|
| `parser` | `builtin`, `docling` | `FASTRAG_PARSER` |
| `ocr` | `true`/`false`: send scanned pages to OCR when it is configured | `true` |
| `chunking.strategy` | `structure`: a chunk never crosses a heading. `window`: pack text across headings, for documents split into many tiny sections. | `structure` |
| `chunking.target_tokens` / `max_tokens` / `overlap_tokens` | token budgets | `FASTRAG_CHUNK_*` (350 / 512 / 40) |
| `context.mode` | `none` (chunk text only), `header` (title and heading path), `llm` (header plus an LLM summary), `auto` (`llm` if an LLM is configured) | `auto` |
| `context.llm` | name of an LLM provider (see `FASTRAG_LLM_PROVIDERS`) | `default` |
| `context.instructions` | extra guidance for the summaries | none |
| `context.summary_max_tokens` | summary length | 120 |

### Compare settings before switching

```bash
fastrag compare pilot docs references.jsonl --set '{"chunking": {"max_tokens": 300}}'
```

`compare` re-processes the collection's files into a hidden trial copy with the new settings, then runs the reference questions on both. It prints Recall@k, MRR, chunk counts and the questions whose ranking changed. The trial shares the stored files, is never returned to live searches, and is dropped afterwards. Use `--keep` to keep it, and `fastrag drop-trial` to remove it later. Trials that use LLM summaries need `--yes`, because every document is summarized again. If the trial wins, the command prints the `ingest-settings` line that adopts its settings.

## Configuration

All settings are environment variables prefixed with `FASTRAG_`; see `src/fastrag/config.py` and `.env.example`. The most important ones:

| Variable | Default | Notes |
|---|---|---|
| `ADMIN_TOKEN` | `change-me-admin` | Operator token. The API refuses to start in `prod` with the default. |
| `EMBEDDING_PROVIDER` | `tei` | `tei`, `openai` (any compatible endpoint), `local` (sentence-transformers), or `hash` (tests only). |
| `EMBEDDING_DIM` | `1024` | Fixed when the database is created. |
| `RERANKER_PROVIDER` | `tei` | `tei`, `cohere`, `jina`, `local` or `none`. If the reranker fails or times out, fastRag falls back to fused ranking. |
| `RERANKER_TIMEOUT_S` | `2.0` | Keep this low for the customer surface. |
| `RERANKER_CALIBRATION` | per-model default | `a,b` from `fastrag calibrate`. |
| `LLM_URL` / `LLM_MODEL` | unset | Optional. Used only at ingestion, for contextual chunk summaries. This is the provider named `default`. |
| `LLM_PROVIDERS` | none | More LLM providers by name, as JSON: `{"cheap": {"url": "...", "model": "...", "api_key_env": "CHEAP_LLM_KEY", "input_cost_per_mtok": 0.15, "output_cost_per_mtok": 0.6}}`. A collection picks one with `settings.ingest.context.llm`. The prices are used only for import cost estimates. |
| `CHUNK_TARGET_TOKENS` / `CHUNK_MAX_TOKENS` / `CHUNK_OVERLAP_TOKENS` | `350` / `512` / `40` | Defaults; each collection can override them. |
| `OCR_PROVIDER` | `none` | `vlm` sends scanned PDF pages to a vision model. |
| `PARSER` | `builtin` | `docling` gives layout-aware PDF and DOCX parsing (install the `docling` extra). Check it on your own files first: in testing it dropped a heading at the top of a page as a page header (design log D23). |
| `SEGMENTER` | `underthesea` | `pyvi` or `builtin` as alternatives. |

## Development

```bash
uv sync                                   # install, including dev tools
docker run -d --name fastrag-db -p 5432:5432 -e POSTGRES_USER=fastrag \
  -e POSTGRES_PASSWORD=fastrag -e POSTGRES_DB=fastrag pgvector/pgvector:pg17
export FASTRAG_TEST_DATABASE_URL=postgresql+asyncpg://fastrag:fastrag@localhost:5432/fastrag_test

uv run pytest                             # full suite; database tests skip if Postgres is down
uv run pytest tests/test_values.py::test_parse_price   # one test
uv run ruff check src tests && uv run ruff format src tests
uv run mypy
uv run fastrag migrate && uv run fastrag serve           # API on :8000
uv run fastrag worker                                    # ingestion worker
```

## License

Apache-2.0
