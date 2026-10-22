# fastRag design log

A running record of design decisions, with their reasons and the questions still open. Add a new entry for each new decision. If a decision is reversed, mark it superseded instead of deleting it.

_Last updated: 2026-10-22_

## Context

- **Product:** fastRag, a retrieval service for small and medium-sized businesses that plugs into agent platforms.
- **First consumer:** our own business agent, built on top of fastRag. The agent owns the LLM, guardrails, settings, refusal and hand-off to a human.
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

### D2. v1 serves customer-service chat first
- **Speed target:** retrieval in roughly 300 ms or less; first useful output to the customer in about 1–2 seconds.
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

## Open questions

- Does each project have one master sheet, or does each sales person keep their own copy? Separate copies would require deduplication and choosing which copy wins.
- Where do the Excel files live: OneDrive/SharePoint, Google Drive, or local machines and email? This decides the second connector.
- How is fastRag deployed: managed (we host it, shared by many tenants) or installed by each client? See Q7 in the discussion.

## Next actions

- [ ] Get sample documents and real customer questions from the pilot client.
