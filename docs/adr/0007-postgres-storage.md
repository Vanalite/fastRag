# ADR 0007: PostgreSQL + pgvector as the production system of record and search index

- **Status:** Accepted. Verification is partial: the `tests/integration` suite, which runs the adapter against a real pgvector database, has not yet passed end to end. It ran 11 of 17 cases before the last fixes. CI's `integration` job is the first full run.
- **Date:** 2026-11-03

## Context

The reference deployment stores documents in SQLite and serves search from in-memory indexes (`NumpyVectorIndex`, `BM25Index`) that each process rebuilds from the store at startup (ADR 0001). That works for one process. It does not work for several API replicas behind a load balancer:

- Each replica holds its own copy of the indexes. A document ingested through replica A is invisible to replica B until B restarts, and a delete is not honoured on B at all. A delete that only some replicas honour is a data-protection failure, not a staleness issue.
- Startup time and memory grow with the corpus, because every replica reloads every vector.
- SQLite has a single writer and lives on one host's disk.

We need one shared, transactional system of record that every replica also *searches*, with the access-control pre-filter (ADR 0002) enforced inside the database query.

## Decision

`src/fastrag/storage/postgres.py` provides drop-in implementations of the three storage protocols in `storage/base.py`:

- `PostgresDocumentStore(dsn, *, dim, pool_size=10)` implements `DocumentStore` with the same semantics as `SQLiteDocumentStore`: single-transaction `replace_document` and `delete_document`, a separate `bump_generation` (the ingest service calls it after the indexes reflect the change), embedder pinning in `meta`, and refusal to run against a schema newer than the code.
- `PgVectorIndex(store)` and `PgKeywordIndex(store)` implement `VectorIndex` and `KeywordIndex` by querying the store's `chunks` table. `add` and `remove` are no-ops, because the store already wrote the rows inside its transaction. `postgres_indexes(store)` wraps them in a `SearchIndexes`.

**Schema.** `documents` holds the record (`acl text[]`, `editors text[]`, `metadata jsonb`). `chunks` holds `chunk_id`, `tenant_id`, `doc_id`, `version`, `ordinal`, `text`, `acl text[]` (denormalized from the document), `title`, `source_id`, `embedding vector(dim)` and `tsv tsvector`. It has a foreign key to `documents` with `ON DELETE CASCADE`. The indexes are:

- btree on `(tenant_id, doc_id)`
- GIN on `acl`
- GIN on `tsv`
- HNSW on `embedding` with `vector_cosine_ops`. It is skipped above 2000 dimensions, pgvector's ANN limit, and search then falls back to an exact scan.

`tenant_generation` holds the cache-invalidation counter.

**ACL pre-filter in SQL.** Both searches put `tenant_id = %s AND acl && %s::text[]` in the `WHERE` clause. The array is the caller's `AccessFilter.tokens`. Rows the caller may not read are never ranked, never counted against `LIMIT k`, and never leave the database. This matches `AccessFilter.allows`: the tenant must match and the sets must intersect. An empty token set or an empty ACL denies access. The retriever's post-hoc `access.allows` check stays in place as defence in depth.

**Dense search.** `ORDER BY embedding <=> %s LIMIT k` (cosine distance). The returned score is `1 - distance`, which is cosine similarity, the same scale as `NumpyVectorIndex`.

**Keyword search.** Postgres full-text search, but over fastrag's own terms. `chunks.tsv` is built by the store from `bm25.tokenize(title + "\n" + text)`, with stop words removed, simple plurals folded, and positions kept for `ts_rank_cd`. The query is the OR of the quoted `tokenize(query)` terms, cast to `tsquery`. Candidates are `tsv @@ query`, pre-filtered by tenant and ACL, ranked by `ts_rank_cd` and limited to k. `coverage` and `matched` are computed in Python exactly as `BM25Index` computes them: the IDF-weighted share of distinct query terms present, with terms absent from the corpus weighted maximally. The document frequencies come from one tenant-wide aggregate query. Like BM25, these statistics are per tenant and ignore ACLs.

*Deviation from a `GENERATED ALWAYS AS (to_tsvector('simple', ...))` column:* Postgres' parser and the `simple` dictionary neither drop fastrag's stop words nor fold plurals. A generated column would make `laptops` unfindable by `laptop` and break coverage parity with the confidence gate (ADR 0003). The trade-off is that only fastrag may write `chunks`. A manual SQL backfill must compute `tsv` the same way.

**Migrations.** The steps are versioned (`_MIGRATIONS`), and the applied version is recorded in `meta.schema_version`. All steps run in one transaction after `pg_advisory_xact_lock`, so replicas starting together cannot race, and a failed step rolls back completely. A schema newer than the code raises `ConfigurationError`. The embedding dimension is pinned in `meta.embedding_dim`.

**Connections.** A `psycopg_pool.ConnectionPool` with a health check on checkout and the pgvector adapter registered per connection. All values are bound parameters. The one exception is the `vector(dim)` type modifier in DDL, which cannot be a parameter. It is a validated integer rendered with `psycopg.sql.Literal`.

**Concurrency.** Writes to one document take `pg_advisory_xact_lock(hashtextextended(tenant/doc))`, which serializes them across replicas. The ingest service's striped locks only cover one process. Each write also carries the version it read (`expected_version`), and `_check_version` compares it under that lock. A mismatch raises `ConcurrentModification`, which the ingest service retries with fresh state (up to 3 attempts, re-checking permissions).

## HNSW and filtered search

HNSW returns the `hnsw.ef_search` nearest candidates (default 40), and the `WHERE` clause filters *those*. If a selective filter, such as a small tenant or a narrow group, removes most of them, the query returns fewer than k rows or none. That is a recall loss, not a leak. Mitigations, applied per query with `set_config(..., true)` (transaction-local, so pooled connections are not affected):

- `hnsw.iterative_scan = strict_order` (pgvector ≥ 0.8). The index keeps scanning until k rows pass the filter, bounded by `hnsw.max_scan_tuples` (default 20 000).
- `hnsw.ef_search = min(1000, max(ef_search, k))`, where `ef_search` comes from `FASTRAG_POSTGRES_EF_SEARCH` (default 100).

For very selective tenants, consider partial HNSW indexes per large tenant, list partitioning of `chunks` by `tenant_id`, or exact search for small tenants, where the planner already prefers the btree. Watch the proportion of dense searches returning fewer than k rows.

## Consequences

Good:

- Every replica searches the same rows. Ingest and delete are visible everywhere as soon as they commit. No index rebuild at startup, and no cross-replica invalidation bus for the indexes. `tenant_generation` also lives in Postgres, so cache keys move in step.
- Access control is enforced where the data lives. The pre-filter is in the query plan, not in application code that a new adapter could forget.
- One backup covers documents, embeddings and indexes.

Costs and caveats:

- `ts_rank_cd` is not BM25. It has no corpus IDF and a different length normalization, so keyword *ordering* can differ from the in-memory index. Coverage and matched counts, which the confidence gate uses, are identical. If ranking parity matters, use a BM25 extension (ParadeDB `pg_search`, VectorChord-BM25) or an external engine.
- The keyword search's statistics query scans the tenant's matching postings on every query. That is fine to millions of chunks per tenant. Beyond that, cache document frequencies per `generation`.
- `SearchIndexes.rebuild_from` still works, because `add` is a no-op, but it streams the whole corpus for nothing. Skip it when using Postgres indexes. `consistent_with` is trivially true.
- The next version number is read before the write, as on SQLite. Two replicas re-ingesting the same document serialize on the advisory lock, and the second one's expected version no longer matches, so it gets `ConcurrentModification` and retries. No write is lost silently. If all attempts fail, the client receives 409.
- The audit log stays a JSONL file per host, outside the database. Each replica needs its own file, and each host's chain is verified separately.

## Operations

- **Pooling.** Each process opens up to 10 connections (`pool_size` is a constructor argument with no `FASTRAG_` setting yet). Keep replicas × 10 under `max_connections`, or put PgBouncer in transaction mode in front. The per-query `set_config(..., true)` settings are transaction-local and safe with PgBouncer. Set `statement_timeout` on the role or in the DSN (`options=-c statement_timeout=5s`).
- **Migrations.** By default they run at startup under the advisory lock, which needs DDL rights. In production, run `fastrag db migrate` as a deploy step with a schema-owner role and start the app with `FASTRAG_POSTGRES_MIGRATE_ON_START=false`: it then only checks the schema version and embedding dimension (`PostgresDocumentStore._check_schema`), so the app role can be DML-only. `CREATE EXTENSION vector` needs privileges, so have the DBA pre-create the extension. Never edit a released migration step; append a new one.
- **Backups.** Use continuous WAL archiving with point-in-time recovery (pgBackRest, WAL-G, or the managed service's PITR) plus periodic restore drills. The store is the system of record, so there is nothing else to back up.
- **Right to be forgotten.** `delete_document` removes the rows in one transaction, and the cascade deletes chunks and embeddings. The old tuples remain in heap pages, index pages and WAL until `VACUUM` reclaims them and the WAL segments age out. For an erasure request, run `VACUUM (INDEX_CLEANUP ON) chunks, documents` after the delete. Note that backups and PITR archives still hold the data until they expire, so retention for them must fit the erasure SLA. Autovacuum covers routine deletes. HNSW indexes reclaim deleted entries during vacuum, so tune `autovacuum_vacuum_scale_factor` on `chunks` for delete-heavy tenants.

### Defence in depth: Row-Level Security

The pre-filter can additionally be enforced by the database itself. Enable RLS on `documents` and `chunks` with a policy such as:

```sql
CREATE POLICY tenant_isolation ON chunks
  USING (tenant_id = current_setting('app.tenant', true)
         AND acl && string_to_array(current_setting('app.acl_tokens', true), ','));
```

Then run each request's queries in a transaction that starts with `SELECT set_config('app.tenant', %s, true), set_config('app.acl_tokens', %s, true)`, which is the equivalent of `SET LOCAL app.tenant = ...`. Connect as a non-owner role without `BYPASSRLS`. A query that forgot its `WHERE` clause then returns nothing instead of another tenant's data. This is not enabled by default because migrations, `iter_chunks` and tenant-wide statistics need a role that may see across tenants, and policy predicates can stop the planner from using the HNSW index efficiently. Benchmark before enabling it.

## Alternatives considered

- **Keep in-memory indexes, add an invalidation bus.** Every replica still holds the whole corpus, and a missed message means a document survives deletion on one replica. Rejected.
- **A dedicated vector database (Qdrant, Weaviate, Pinecone) or OpenSearch next to Postgres.** These scale further and have better filtered ANN search, but they add a second system of record that must stay transactionally consistent with the first. A delete becomes a distributed operation that needs an outbox and retries. Worth it beyond tens of millions of chunks. The `VectorIndex` and `KeywordIndex` protocols leave that door open.
- **IVFFlat instead of HNSW.** Faster to build and smaller, but recall depends on `lists` and `probes` tuned to a corpus size that changes per tenant, and it needs data present before the index is built. HNSW needs no training and handles incremental inserts.
- **Postgres FTS with a `GENERATED` tsvector.** Simpler to keep consistent, but it breaks term parity with the BM25 tokenizer, as explained above.
