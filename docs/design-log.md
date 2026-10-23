# fastRag design log

A running record of design decisions, with their reasons and the questions still open. Add a new entry for each new decision. If a decision is reversed, mark it superseded instead of deleting it.

_Last updated: 2026-10-22_

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

## Open questions

- Does each project have one master sheet, or does each sales person keep their own copy? Separate copies would require deduplication and choosing which copy wins.
- Where do the Excel files live: OneDrive/SharePoint, Google Drive, or local machines and email? This decides the second connector.

- Expected pilot traffic: tens or thousands of customer conversations a day? This decides whether CPU embedding and a hosted reranker remain viable.

## Next actions

- [ ] Get sample documents and real customer questions from the pilot client.
