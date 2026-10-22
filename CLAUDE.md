# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

The repository is at its initial commit: only `README.md`, `LICENSE` (Apache-2.0), and a standard Python `.gitignore` exist. There is no source code, build tooling, test suite, or dependency manifest yet. When you add these, update the **Commands** and **Architecture** sections below so they describe what actually exists, not what is planned.

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

- **Language:** Python (implied by the `.gitignore`).
- **Easy install and run** for non-expert operators. Prefer a single-command start, such as `docker compose up` or a `pip install` plus a CLI. Use sensible defaults and keep required configuration to a minimum.
- **Agent-platform pluggability:** expose retrieval through stable, platform-neutral interfaces, for example an HTTP API and/or an MCP server or tool schema. Don't couple the code to one agent framework.
- **Swappable backends:** keep the LLM provider, embedding model, reranker, and storage behind interfaces, so an SMB can choose hosted or self-hosted options.
- **Multi-tenant awareness:** one deployment may serve several businesses or knowledge bases, so isolate data per tenant or collection.

## Commands

None yet. Add the commands for install, dev server, lint/format, type-check, the full test suite, and a single test once the tooling is chosen.

## Architecture

Not yet implemented. Once code exists, document the ingestion → graph/vector indexing → hybrid retrieval → answer/serving flow here. Include the module boundaries that only make sense after reading several files.
