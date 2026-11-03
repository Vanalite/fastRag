# ADR 0001: One source of truth, with rebuildable derived indexes

- **Status:** Accepted
- **Date:** 2026-11-03

## Context

A RAG system keeps several copies of the same data: the text, the embeddings, a keyword index, and an answer cache. Each copy can drift. The most serious drift is a delete that reaches one copy and not another. A deleted policy then stays searchable, or comes back from a cache. Enterprise deployments need to answer "what did we store, and is it gone?" with one place to check.

## Decision

- `SQLiteDocumentStore` (`src/fastrag/storage/sqlite.py`) is the **system of record**. It holds documents, chunks, embeddings, ACLs, editor lists and a per-tenant generation counter.
- The vector index (`storage/vector.py`) and the keyword index (`storage/bm25.py`) are **projections**. `SearchIndexes` (`storage/indexes.py`) groups them, so they change together under one lock.
- Every write is one SQLite transaction (`replace_document`, `delete_document`). Each transaction covers the document rows and chunks only. The tenant generation is bumped separately, after the indexes are updated.
- On SQLite startup, `RAGService.from_settings` rebuilds both indexes from the store (`SearchIndexes.rebuild_from`). Nothing persists them. With Postgres, the indexes are the store's own tables, so nothing is rebuilt (ADR 0007).
- The ingest path commits to the store first, applies the change to the indexes, and then bumps the generation (`IngestService.ingest`). The store commit is the point of truth. The generation is the signal that tells caches to move on, so it comes last.

## Consequences

Good:

- Deletes and re-ingests are correct in one place. Restarting the process repairs any index drift.
- The data model is small enough to read in one sitting.
- Schema changes are in-place migrations recorded in `meta` (`SQLiteDocumentStore._migrate`), run inside one immediate transaction. A database from a newer release is refused rather than downgraded.

Costs and risks:

- **Startup time** is linear in the corpus, because every chunk is reloaded and reindexed.
- **Memory** is linear in the corpus. Embeddings and text both live in memory.
- **Stale window (SQLite).** Between the commit and the index update, the in-memory indexes disagree with the store. Queries in that window can see the old content, but their answers are cached under the old generation, so they are never served after the change. If the index update fails, the indexes stay stale until restart. `SearchIndexes.consistent_with` exists, but only tests call it.
- **Single writer.** SQLite serializes writes (`BEGIN IMMEDIATE`). Several processes would each hold their own in-memory indexes, and an ingest on one would not reach the others. ADR 0007 removes this limit for deployments that need it.

## Alternatives considered

- **Dual writes to the store and a vector database, with no transaction.** Rejected. A crash between the two writes leaves them disagreeing with no way to tell which one is right.
- **Vector database as the system of record.** Viable at scale, but then the ACL rules, the text and the deletes all depend on that database's semantics. Deletes by filter and per-tenant partitions vary between products, so the guarantee would be weaker and harder to test.
- **Postgres with pgvector, storing vectors beside the documents in one transaction.** This is the intended production path. It makes the store and the vector index one thing, which removes the in-memory index and the stale window. The interfaces in `storage/base.py` made it a new adapter, not a rewrite. It is implemented in `storage/postgres.py` (ADR 0007).
- **Event log with projections (CDC).** Adds an operational component (a log and consumers) that a reference implementation does not need yet.
