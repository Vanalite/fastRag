# fastRag design log

A running record of design decisions, with their reasons and the questions still open. Add a new entry for each new decision. If a decision is reversed, mark it superseded instead of deleting it.

_Last updated: 2026-11-07_

## Context

- **Product:** fastRag, a retrieval service for small and medium-sized businesses that plugs into agent platforms.
- **First consumer:** our own agent backend, which runs in the cloud and owns the LLM, guardrails, settings, refusal and hand-off to a human. Three surfaces sit on top of it, and all three use fastRag:
  1. **Customer AI agent:** the real-estate company's customers chat with the AI.
  2. **Hand-off inbox (admin dashboard):** sales or support staff take over conversations from the AI.
  3. **Internal knowledge (admin dashboard):** sales staff look up internal documents for selling and advertising.
- **Pilot client:** a Vietnamese real-estate business.
  - About 99% of content and customer questions are in Vietnamese; 1% English.
  - Documents are PDFs, web pages, and many spreadsheets.
  - Staff are sales people with minimal technical skills.
  - No documents have been received yet.

## Decisions

### D1. The knowledge graph is a hypothesis, not a requirement
- **Backbone:** hybrid retrieval, meaning keyword search (BM25) plus vector search, followed by a reranker, with contextual chunking (each chunk gets a short summary of its surrounding document before indexing).
- **Graph:** an optional, LightRAG-style layer for entities and relationships. A query router turns it on only for multi-hop or entity-specific questions.
- **Rule:** the graph ships only where our quality checks show it beats hybrid retrieval. If it costs more than it delivers, drop it and keep plain hybrid RAG.
- **Most promising place for the graph:** the address and project hierarchy (city → district → ward → project → unit).

### D2. v1 surfaces (corrected 2026-10-22)
- **Customer AI agent.** End customers of the real-estate company.
  - Retrieval in roughly 300 ms or less; first useful output in about 1–2 seconds.
  - Strictest precision; shows customer-visible content only.
- **Hand-off inbox.** Staff take over conversations from the AI.
  - fastRag acts as an assistant: it retrieves from the conversation so far, not just a single question.
  - Staff may see internal content.
- **Internal knowledge.** Sales and advertising staff search internal documents.
  - Speed matters less; broader recall is preferred.
- The same documents and the same ingestion serve all three surfaces; they differ only in retrieval profile and access (D11).
- The retrieval core stays reusable for automation pipelines later.

### D3. fastRag returns evidence; the agent writes the answer
- **v1 is retrieval-only.** An optional `/answer` endpoint is deferred.
- **Rule:** fastRag states facts about the evidence; the agent makes the decisions. fastRag does not return a `should_escalate` flag.

| Layer | What it validates |
|---|---|
| **fastRag (evidence)** | Tenant and access filtering inside the database query, never after retrieval. Ingestion hygiene: prompt-injection scan, personal-data scan, deduplication, source/version/date metadata. Evidence signals on every response: calibrated reranker scores, coverage, agreement/conflict/staleness, and an explicit "nothing relevant found". Source IDs, versions and trace IDs. |
| **Agent (answer and policy)** | Groundedness of the generated answer against the citations, refusal, hand-off to a human, tone, personal data in the output, business rules. Turns fastRag's signals into decisions. |
| **Shared** | A versioned response schema and a shared trace ID. Optional later: a `/verify(claims, passages)` helper. |

### D4. Pilot data and quality checks
- Design and tune against the real-estate pilot client, not generated questions.
- **Requested from the client:** about 20 sample files, including spreadsheets, and about 30 real customer questions copied exactly as typed (from Zalo, Facebook or phone notes).
- These become the first reference question set.
- Every later choice (chunking, embedding model, reranker, whether to keep the graph) is decided by measurement against this set.

### D5. Vietnamese is a first-class design constraint
- **Unicode:** convert all text to NFC at ingest. Detect and convert legacy encodings (TCVN3, VNI) found in old PDFs.
- **Keyword search:**
  - Run a Vietnamese word segmenter (`underthesea`, `pyvi` or VnCoreNLP) at both index time and query time, because spaces separate syllables, not words.
  - Index text both with and without diacritics. Matches with diacritics rank higher; accent-free matches are the fallback.
- **Embeddings and reranker:** multilingual or Vietnamese-tuned models. Candidates:
  - embeddings: BGE-M3, `AITeamVN/Vietnamese_Embedding`, Qwen3-Embedding, Cohere multilingual
  - rerankers: `bge-reranker-v2-m3`, Qwen3-Reranker
  - choose using the pilot's reference questions, not public benchmarks
- **OCR for scans** (contracts, land certificates): PaddleOCR, VietOCR or a vision-language model. Not Tesseract, which is weak on Vietnamese diacritics.
- **Standardizing values:**
  - addresses: "Q.7", "Quận 7" and "District 7" all map to one district code; "P." means phường (ward)
  - prices: "3,5 tỷ" and "850 triệu" become numbers in VND
  - areas: "m2" and "m²"
- **Token budgets:** measure chunk sizes and costs in tokens, because Vietnamese uses more tokens per character than English.

### D6. Spreadsheets get their own structured path
- Listing and inventory spreadsheets are loaded into **typed database tables**: price, area, cleaned-up address, status and description.
- They are served by a separate tool, `query_listings(filters)`, using **predefined filters only**. No free-form text-to-SQL in v1.
- PDFs and web pages go through hybrid search.
- The agent chooses which tool to call.
- **Data:** listings and inventory with prices, estimated areas, addresses and sales descriptions. No transaction or customer records. Updated daily to weekly.

### D7. Data model: tenant → collections
- This follows the enterprise "shared platform with separate collections" pattern, rather than one mixed index.
- Each collection has a type (`documents` or `listings`), its own settings and its own reference questions.
- **Pilot setup:** one tenant with two collections: (1) policy and project documents, (2) listings.

### D8. Ingestion is connector-based sync
- **Ways in:** manual upload, a Google Sheets/Drive connector with scheduled sync, and a webhook for "sync now". Excel files are supported as well.
- **Upload-only is rejected:** the client's staff are not technical, so freshness can't depend on someone remembering to upload.
- **Listings:** each sync replaces the sheet's rows as a complete snapshot and stamps them with `as_of`. Every listing result returns `as_of`, so the agent can state how fresh it is.
- **Documents:** compared by content hash, so only changed files are re-processed.
- **Future expansion:** a direct connection to the client's database or CRM. The client has mentioned this.

### D9. Deployment: a private service next to the agent backend
- fastRag runs as its own containers (API and ingestion workers) beside the agent backend, in the same cloud and private network.
- It is never exposed to the internet.
- **Interface:** REST between the backend and fastRag, carrying a service token, the tenant, the caller's scope and a trace ID. MCP can come later for outside platforms.
- **Why separate:** ingestion load is heavy and spiky (OCR, embeddings) and must not slow down chat responses.
- **Deployed together:** one Compose file or one Kubernetes namespace, so self-hosting stays cheap to add later.
- **Storage:** Postgres (pgvector, full-text search, listing tables) plus object storage for raw files.

### D10. Build a thin core; RAGFlow is the baseline to beat
- Don't fork a full system. The parts that make fastRag valuable (Vietnamese handling, the typed listings path, evidence signals, `as_of` sync) would have to be built anyway.
- **Borrow proven libraries:**
  - parsing: docling, PaddleOCR
  - embeddings and reranking: BGE-M3, bge-reranker-v2-m3
  - vector search: pgvector
  - model providers: LiteLLM
  - API: FastAPI
- **Write ourselves:** chunking, hybrid ranking with reciprocal rank fusion, and the evidence signals.
- **Baseline:** run RAGFlow on the pilot's reference questions alongside fastRag. If it clearly wins on prose documents, reuse its document parser.
- **Rejected:**
  - LightRAG: built around the graph, which conflicts with D1
  - R2R: maintenance status unclear
  - Onyx: heavy, tuned for English
  - Forking RAGFlow wholesale: too large a fork to maintain

### D11. One body of documents, several audiences
- Every document, passage and listing column carries a **visibility** label: `public` (safe for customers) or `internal`.
- **The caller's scope decides what it can see:**
  - the customer agent sees `public` only
  - staff surfaces see `public` and `internal`
- **Enforcement:** filters run inside the database query, never after retrieval. Tests must prove that the customer scope can never return internal content.
- **Retrieval profiles are configuration, not separate code paths:**
  - `customer`: strict, public only
  - `handoff_copilot`: public and internal (caller builds the query, see D13)
  - `internal_search`: broader recall, internal included

### D12. Access groups are inherited from folders, with "internal" as the default
- This is permission inheritance from the source, the same pattern large companies use for department folders.
- **Folders map to access groups** (for example `public`, `staff`, later `managers`). Every passage and listing column inherits its folder's groups.
- **Callers send their groups:**
  - customer agent: `[public]`
  - staff: `[public, staff]`
- **Filtering:** keep only content whose groups overlap the caller's groups. The filter runs inside the database query.
- **Spreadsheet columns:**
  - The first time a sheet syncs, an admin marks which columns are public. The mapping is saved per sheet layout.
  - New or unmapped columns default to internal.
- **Safety net:** the ingestion scan flags phone numbers, "giá chốt", "hoa hồng" and similar patterns found in public content. An LLM classifier may suggest labels but never decides them.
- **v1 mapping lives in fastRag's admin settings,** not copied from Google Drive sharing, because small businesses often share files as "anyone with the link".

### D13. The caller interprets the question; fastRag standardizes values
- **The agent's LLM, or the dashboard backend:**
  - rewrites follow-up messages into questions that stand on their own
  - extracts filters into `query_listings` arguments
  - chooses which tool to call
- **fastRag deterministically standardizes values:** prices ("3,5 tỷ", "850tr" → numbers in VND), districts and wards ("Q7", "Quận 7", "District 7" → one code), areas. It rejects filters it cannot interpret instead of guessing.
- **No LLM on fastRag's query path.** LLMs are used only at ingestion: contextual chunking, cleaning up spreadsheet headers, optional entity extraction.
- **This replaces part of D11:** the caller builds the query, and fastRag never reads conversations.
- **Recommended, to confirm:** the agent keeps track of the listing IDs it has shown and passes them back, so a follow-up like "that unit" becomes an exact lookup by ID with a fresh `as_of`.
- **Fallback:** if recall on short Vietnamese questions is poor, add query expansion in the agent, not in fastRag.
- **Product definition:**
  - fastRag is a retrieval engine for callers that already have an LLM and the conversation.
  - It is not a search product for callers without an LLM; those are outside its target users.
  - Managed search products such as Vertex AI Search and Bedrock Knowledge Bases put an LLM inside because their callers may lack one. That reasoning does not apply here.
  - Multi-hop questions are handled by the agent calling the tools repeatedly (agentic RAG), not by fastRag.

### D14. Models: open embedding model, hosted reranker
- **Embeddings: BGE-M3** (open weights).
  - Changing the embedding model later means re-indexing every document, so we avoid vendor lock-in here.
  - Run it on CPU workers at pilot volume. If CPU capacity becomes a problem, use a hosted inference provider serving the same model.
- **Reranker: a hosted multilingual reranker,** to meet the speed target without a GPU. It can be swapped at any time with no re-indexing.
- **Move the reranker to our own GPU** only when traffic or cost justifies it.
- **Before committing:** benchmark two or three embedding and reranker combinations against the pilot's reference questions (D4).
- Both models sit behind interfaces (see CLAUDE.md, "Swappable backends").

### D15. BM25 is computed inside Postgres (decided 2026-11-07, implementation)
- Each chunk stores its term frequencies (`chunks.tf`, `chunks.terms` with a GIN index). Document frequencies per collection live in `term_stats` and `collection_stats`, and these are updated in the same transaction that writes or deletes chunks.
- The keyword query computes true BM25 (k1 = 1.2, b = 0.75, with IDF) in SQL, with the tenant, collection, group and quarantine filters in the same `WHERE` clause.
- **Rejected:**
  - `ts_rank`: it is not BM25, has no IDF, and Postgres has no Vietnamese text-search configuration.
  - ParadeDB `pg_search` or VectorChord-BM25: these need a non-standard Postgres image, which is harder for SMB operators to run.
- **Terms:** each word is indexed as the segmented word (`chung_cư`), its syllables, an accent-free copy (`~chung_cu`, weighted 0.5), and district codes found by the gazetteer (`@hcm-q7`). This makes "Q7" match "Quận 7" and accent-free questions still match.
- **Revisit** if a collection grows past roughly 1M chunks: the `term_stats` upserts and the per-candidate scoring are the first things to measure.
- **Revised 2026-11-07 (migration 0002):** the first version read each candidate chunk's term-frequency JSON and took over 3 s at 50,000 chunks when question terms were common.
  - Keyword search now uses an inverted index, `chunk_postings`, whose primary key includes `tf`, so queries are index-only scans.
  - Each profile has a postings budget. The rarest question terms are used first, and the most common ones (lowest IDF) are skipped once the budget is spent. The budget is 40k postings for `customer` and 120k for `internal_search`.

### D16. The job queue is a Postgres table (decided 2026-11-07, implementation)
- The queue uses `FOR UPDATE SKIP LOCKED`, keeps at most one queued job per dedupe key (for example one pending sync per source), retries with exponential backoff, and re-queues jobs whose worker died.
- A job that fails while an identical job is already queued is marked `superseded`.
- **Why:** there is no Redis or broker to install. Ingestion volume at SMB scale is small.
- Syncs of the same source are serialized with an advisory lock.

### D17. Degrade instead of failing on the query path (decided 2026-11-07, implementation)
- Keyword search and vector search run concurrently. If one of them fails, the other still answers, and `signals.degraded` says so.
- If the reranker errors or exceeds `FASTRAG_RERANKER_TIMEOUT_S`, results keep fused (RRF) order and report `signals.reranker = "unavailable"`.
- Without reranker scores, a passage is kept when its cosine similarity **or** the share of question content words it contains reaches the profile's `min_similarity`. A clear keyword match must not vanish because the embedding model is weak on accent-free text.
- Question words and particles ("bao nhiêu", "gì", "phải không", a trailing "không") are excluded from word-match and coverage. They are still searched.
- **Questions typed without diacritics:** bge-reranker-v2-m3 still ranks the right passage first, but scores it low in absolute terms. For these questions, a passage is also kept when it contains at least `min_similarity` of the question's content words. `signals.accentless_query` reports that this rule applied.

### D18. Listing visibility is per field, enforced in SQL (decided 2026-11-07, implementation)
- `listings.field_groups` holds the groups of each field, for each row.
  - Masked fields come back as `NULL` from SQL.
  - Filters and sorts on a field the caller cannot see match nothing; they are not silently ignored.
  - Keyword terms are stored with an access-group prefix (`public|căn_hộ`), so a keyword filter can only match text the caller may see.
- A public cell that the hygiene scan flags (phone number, "giá chốt") is downgraded to internal for that row.
- Listing IDs are UUIDv5 values of (collection, sheet, listing code). They stay the same across syncs, so the agent can look a unit up again (D13). Rows without a code column get a code hashed from the row content; such a code changes if the row changes.
- When a sync finds an unchanged file, it refreshes `as_of` without re-importing.

### D19. Ingestion hygiene quarantines content; an admin clears it (decided 2026-11-07, implementation)
- Flags: `prompt_injection`, `pii:*` (phone, email, ID number, bank account), and `sensitive_in_public` (personal data or internal terms in public content).
- The `customer` profile excludes `prompt_injection` and `sensitive_in_public` passages. `handoff_copilot` excludes only `prompt_injection`.
- An admin clears flags in the review queue. The decision is stored by passage text hash, so it survives re-ingestion until the text changes.
- Documents with flags or warnings get status `needs_review`. They stay searchable, except for their quarantined passages.

### D20. Vietnamese data notes (decided 2026-11-07, implementation)
- **District codes follow the pre-July-2025 district layer.** The 2025 reform removed districts, but listings and customers still use the old names.
  - Thủ Đức is the parent of the former Quận 2 and Quận 9, so a filter on Thủ Đức includes both.
  - Tenants can add aliases (for example "Phú Mỹ Hưng" → `hcm-q7`) under the collection's `settings.gazetteer`.
- **TCVN3 text is converted. VNI text is only detected and flagged for review,** because a wrong conversion would corrupt text silently.
- **Tone marks are normalized to one style** ("hòa", not "hoà"), so text typed in either style matches.
- **Price shorthand follows spoken usage:** "3 tỷ 5" is 3.5 tỷ and "3 tỷ 200" is 3.2 tỷ. A bare number under 100,000 has no unit and is rejected.

### D21. Reranker scores are calibrated per model (decided 2026-11-07, measured)
- bge-reranker-v2-m3's raw 0–1 scores are not probabilities. Correct answers scored 0.03–0.28 and unrelated passages about 0.000, so a "0.5 means relevant" threshold dropped most correct answers.
- Every reranker therefore passes through Platt scaling: `sigmoid(a · logit(raw) + b)`. Profile thresholds assume the calibrated scale.
- **Default for bge-reranker-v2-m3:** a = 0.63, b = 2.15. These were fitted with `fastrag calibrate` on 15 Vietnamese real-estate questions (150 labelled candidates, log-loss 0.10). After calibration, relevant passages had a median score of 0.8 and others 0.013.
- **Results with BGE-M3 and bge-reranker-v2-m3:**
  - On the set used for fitting: Recall@10 1.0, MRR 1.0, "nothing relevant" correct in 3 of 3 cases. This is optimistic because it is in-sample.
  - On 11 held-out questions: Recall@10 0.875, MRR 0.875, "nothing relevant" correct in 3 of 3.
  - Access safety and listing checks passed in both.
- **Must be refit on the pilot's reference set** (`fastrag calibrate`, then set `FASTRAG_RERANKER_CALIBRATION`) and again whenever the reranker changes.
- **Known gap:** English questions against Vietnamese text sometimes score too low to keep (for example "Can I keep a cat?"). Following D13, the agent translates questions into the documents' language, and the tool descriptions say so.

### D22. Query-path performance (decided 2026-11-07, measured with `scripts/bench_search.py`)
- **Benchmark:** 50,000 synthetic Vietnamese chunks with 1024-dimensional vectors; database time only.
  - `customer`: p50 56 ms, p95 68 ms.
  - `internal_search`: p50 211 ms, p95 240 ms.
  - Adding query embedding and a hosted reranker should keep the customer path under the 300 ms target (D2). This still needs measuring against the chosen reranker.
- **Fixes the benchmark found:**
  1. The GIN index on `chunks.groups` made the planner skip HNSW when the caller's groups were a query parameter: 107 ms down to 15 ms after dropping it (migration 0003).
  2. Postgres JIT cost ten times the query itself, so it is off on fastRag connections (`FASTRAG_DB_JIT`) and in Compose.
  3. `shared_buffers` must hold the HNSW index. Compose defaults to 1 GB; set it to about 25% of the host's RAM.
  4. The query vector is parsed once, in a CTE.

### D23. Builtin parser stays the default; docling is opt-in (decided 2026-11-07, measured)
- Tested docling 2.148 on a generated two-page Vietnamese policy PDF and a DOCX.
- **What works:** it parses Vietnamese correctly. The layout models load once per process (8 s for the first file, 0.3 s after that), and DOCX output matches the builtin parser.
- **Problems on the PDF:**
  - It dropped the line "Điều 4. Bàn giao" at the top of page 2, treating it as a page header. That is lost text, not just a lost heading.
  - It marks every heading at the same Markdown level.
  - It ignores the PDF metadata title.
- **Fixes on our side:** the docling path now re-detects Vietnamese headings (Chương / Mục / Điều) in docling's Markdown, and takes the title, page count and scanned-page detection from pypdf.
- **The dropped-text problem stays.** Use docling only for table-heavy PDFs, and only after comparing chunk counts with the builtin parser on the pilot's files.

### D24. Native DeepSeek Harness tools for the agent backend (decided 2026-11-17, implementation)
- **Context:** the agent backend (any-agent) runs its agents on DeepSeek Harness (`dsh`). Its agent profiles disable every MCP, shell and file row on purpose, because customer messages are untrusted. Agents get only tools that are registered by a Cordis plugin through `defineTool` from `@deepseek-ai/dsh-tools`. So the MCP server is not a way in for this backend.
- **Decision:** ship `integrations/dsh-tools` (`@fastrag/dsh-tools`), a zero-dependency ES module that builds `search_documents`, `query_listings` and `get_listings` as dsh tool definitions. They call the REST API (D9) with a tenant API key.
  - **The host passes in its `defineTool`.** This avoids module-resolution problems when the package is bundled into the host's own plugin, which is how any-agent builds `business-tools.mjs`. A standalone `plugin.mjs` covers plain dsh profiles.
  - **Config is plain JSON in the profile patch.** The API key is named by an environment variable and never written into the patch, matching the backend's rule for its LLM gateway key.
  - **What the model sees is evidence, not an answer (D3):** numbered passages, conflict and staleness notes, and an explicit "do not answer" when `nothing_relevant` is set. Listing prices are formatted for reading, with `as_of`. The `trace_id` goes into the presentation metadata for feedback.
  - **Failures become tool errors,** not failed turns. Rejected filters (D13) are passed back word for word, so the model can ask the customer to clarify.
- **Rejected:**
  - Enabling MCP in the backend's agent profiles. It would undo a deliberate lockdown.
  - Having the backend write its own HTTP tool code. Every agent backend would then repeat the rendering rules for the evidence signals.
- **Tool names are shared** with the MCP server and `GET /v1/tools`. Renaming one means renaming all three.

### D25. Ingestion is configured per collection and measured before switching (decided 2026-11-17, implementation)
- **Context:** the write path existed, but its parser, chunk budgets and context LLM were deployment-wide environment settings. Loading a large corpus had no dry run, no cost estimate and no progress report. D2 says chunking is decided by measurement, yet there was no way to compare two settings.
- **Per-collection settings:** documents collections can override the deployment defaults under `settings.ingest` with partial overrides. These cover the parser, OCR on/off, chunking (`strategy`, target/max/overlap tokens) and context (`mode` none/header/llm/auto, LLM provider, extra instructions, summary length).
  - Settings are validated when saved. A missing LLM is an error at save time, but at ingest time it falls back to header context with a warning, so a removed provider never stops ingestion.
- **Named LLM providers:** `FASTRAG_LLM_PROVIDERS` adds endpoints by name next to the `default` one. Keys are read from named environment variables. Optional per-token prices feed only the dry-run estimate.
- **Fingerprint:** each document records the fingerprint of its effective settings, which also covers the pipeline version, tokenizer and embedding model.
  - A changed fingerprint counts as a change, so re-syncing an unchanged file under new settings re-processes it.
  - Changing a collection's settings queues a re-process of only the stale documents.
  - Legacy documents without a fingerprint count as stale, but nothing re-processes them automatically.
- **Chunking strategies:** `structure` (unchanged default) and `window`, which packs text across headings and keeps later headings as lines. It is for documents with many tiny sections. No other strategies until the reference set shows a need (D2).
- **Bulk import:** `fastrag import <source> <folder>` walks a tree.
  - **Paths and access:** paths inside the source come from the folders, so `folder_groups` decide access.
  - **Dry run:** `--dry-run` classifies files as new/changed/unchanged/stale/skipped. It parses and chunks new and changed files to estimate chunks, embedding tokens and LLM calls, tokens and cost. It writes nothing and calls no LLM.
  - **Running it:** the import queues work for the worker and follows it to the end, or `--inline` processes it in the process with bounded concurrency. Re-running continues an interrupted import.
- **Preview and compare:**
  - `fastrag chunk-preview` shows the chunks for a file without writing.
  - `fastrag compare` re-processes a collection into a trial copy (`settings.trial_of`), runs the reference questions on both and reports the difference.
  - Trial collections share stored files and are excluded from default collection selection (API and eval), so live traffic never sees them. They are dropped afterwards unless `--keep` is given.
- **Rejected:**
  - Per-source settings: access is per source, but retrieval tuning and the BM25 statistics are per collection.
  - Speculative chunking strategies such as semantic or fixed windows.
  - Re-processing on every deployment-default change: that could run up LLM costs without warning.

### D26. Reference questions can be drafted from the documents, but only people approve them (decided 2026-11-18, implementation)
- **Context:** Q12 proposes LLM-drafted reference questions that a person approves. Without a reference set, `eval`, `calibrate` and `compare` have nothing to measure.
- **Decision:** `fastrag eval-generate` samples passages and asks an LLM for a question plus the exact text that answers it. It writes drafts into the normal reference file. `fastrag eval-review` approves, edits or rejects them one at a time.
- **Drafts are never scored.** Entries with `status` `draft` or `rejected` are skipped by `eval`, `calibrate` and `compare`. A file with nothing else is an error, so it cannot pass the release checks by measuring nothing.
- **Checks that do not rely on the LLM:** the quote must be in the passage (same comparison as `contains` at scoring time), the question must not contain the quote or copy the passage, and it must not refer to "this document".
- **What reaches the LLM:** only passages from `ready` documents, in the requested groups (default `public,staff`), with no hygiene flags. Flagged passages may hold prompt injection or personal data.
- **Coverage:** passages are picked round-robin across documents with a seed, so long documents cannot fill the set. Internal passages get the `internal_search` profile, because the customer scope cannot find them.
- **Growing the set:** a new run appends. It skips passages and questions already in the file, including rejected ones, and existing lines are not changed.
- **Limits:** questions drafted from documents test wording that follows the documents, so they tend to be easier than real customer questions. Q12's real chats, hand-offs and thumbs-down feedback remain the better source and should be added to the same file. "Unanswerable" drafts are the LLM's guess until a reviewer confirms that no document answers them.

## Open questions

- Does each project have one master sheet, or does each sales person keep their own copy? Separate copies would require deduplication and choosing which copy wins.
- Where do the Excel files live: OneDrive/SharePoint, Google Drive, or local machines and email? This decides the second connector.

- Expected pilot traffic: tens or thousands of customer conversations a day? This decides whether CPU embedding and a hosted reranker remain viable.

- **Q12, pending answer.** What counts as "good enough to launch", and who maintains the reference questions? Proposed:
  - **Three release checks:**
    1. Access safety: zero internal content returned to the customer scope. This blocks a release.
    2. Listing accuracy: 100% correct filters and value standardization on a fixed test set. This blocks a release.
    3. Document retrieval quality: Recall@10 and MRR. Targets are set after the first baseline run against RAGFlow; after that, no release may score lower.
  - **Who maintains the reference questions:** we draft them with an LLM from real customer chats, a sales person approves them in a review screen (about 30 minutes a week), and the set grows from hand-offs and thumbs-down feedback.

## Build order

**Status 2026-11-07:** steps 1–6 are implemented in v0.1, with 136 tests against a real Postgres. BGE-M3 and bge-reranker-v2-m3 have been measured (D21) and latency benchmarked (D22). Step 7 stays deferred, as D1 requires. The release checks follow the Q12 proposal, but the targets still need the pilot's answer. Not yet verified against live services:
- the Google Drive connector (tested with a mocked Drive client);
- OCR through a vision model;
- docling parsing;
- the hosted Cohere/Jina rerankers.

Original plan:
1. **Project skeleton:** Python project, FastAPI service, Postgres with pgvector in Docker Compose, a worker process, lint, type-check and tests in CI. Update the Commands section of CLAUDE.md.
2. **Data model:** tenants, collections, documents, chunks and listing tables, all carrying access groups (D7, D11, D12). Write access-safety tests first: customer scope never sees internal content.
3. **Listings path:** Excel and Google Sheets import, column mapping with internal as the default, value standardization (prices, districts, areas), `query_listings` and lookup by ID returning `as_of` (D6, D8, D13).
4. **Documents path:**
   - parsing (docling, plus OCR for scans)
   - Vietnamese pipeline: Unicode NFC, word segmentation, text with and without diacritics
   - contextual chunking, BGE-M3 embeddings
   - hybrid search with reciprocal rank fusion, then the hosted reranker
   - evidence signals (D3, D5, D14)
5. **Quality checks:** reference-question format, an evaluation command that reports Recall@10 and MRR, and a RAGFlow baseline run (D4, D10, Q12).
6. **Sync:** Google Drive/Sheets connector, scheduled sync, content-hash comparison, webhook (D8).
7. **Later:** the knowledge-graph layer, only if the quality checks justify it (D1).

## Next actions

- [ ] Get sample documents and real customer questions from the pilot client.
- [ ] Answer Q12 and the open questions above.
- [x] Start implementation locally at step 1 of the proposed build order (v0.1 done 2026-11-07).
- [ ] Load the pilot files, write the first reference set, and run `fastrag eval` alongside a RAGFlow baseline.
- [ ] Benchmark embedding and reranker combinations (D14) on the pilot set, run `fastrag calibrate`, and check the profile thresholds (D21).
- [ ] Choose the hosted reranker and measure customer-profile latency against the 300 ms target (D2).
- [ ] Verify the Google Drive connector with a real service account.
