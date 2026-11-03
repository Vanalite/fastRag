# ADR 0002: Access control with Principals, ACL tokens and mandatory access filters

- **Status:** Accepted (in-process indexes). The pgvector choice below is a decision for the planned adapter.
- **Date:** 2026-11-03

## Context

Every RAG leak in the wild has the same root cause: retrieval ran over everything, and access was checked later, if at all. Access needs to be decided at the point where data leaves the index, and the design should make it hard to skip that check.

## Decision

**Identity and tokens.** `Principal` (`src/fastrag/domain.py:43-51`) carries `tenant_id`, `user_id`, `groups` and `roles`. `Principal.acl_tokens()` expands it into a set of tokens: `tenant:*`, `user:<id>` and one `group:<g>` per group. A document's ACL is a set of the same tokens (`ACL_TOKEN_PATTERN`, `domain.py:22`).

**Rule.** A chunk is visible to a principal when the tenant matches **and** the two token sets intersect (`security/acl.py:26-28`). An empty ACL matches nothing. Ingest rejects an empty ACL (`ingest/service.py:73-74`) and any token outside the pattern (`:75-77`).

**Every chunk carries its access data.** `Chunk` has `tenant_id` and `acl` (`domain.py:91-101`), copied from the document at ingest. The index can therefore decide without a join.

**Mandatory access argument.** `VectorIndex.search` and `KeywordIndex.search` take `access` as a keyword-only, required argument (`src/fastrag/storage/base.py:50-52`, `:62-64`). There is no way to query an index without naming the caller. `SearchIndexes.dense` and `.keyword` pass it through (`storage/indexes.py:54-62`).

**Pre-filter, not post-filter.** The in-memory adapters apply the filter before ranking:

- Dense: the ACL mask is built over the tenant partition before the matrix product (`storage/vector.py:61-73`).
- Keyword: each posting is skipped unless the chunk is allowed, before scoring (`storage/bm25.py:100`).

**Tenant-partitioned indexes.** Each adapter keeps one partition per tenant (`vector.py:58`, `bm25.py:82`). BM25 term statistics are also per tenant, so one tenant's corpus cannot change another tenant's scores.

**Defense in depth.** The retriever re-checks each result (`src/fastrag/retrieval/retriever.py:57-61`). A chunk that fails the check is dropped and logged at critical level. This protects against a buggy adapter, not against a missing filter in the caller.

**Deny by default** applies at each level: unknown tenant, empty ACL, unmatched token, unknown role.

### pgvector: WHERE clause or Postgres row-level security

When the vector index moves to Postgres, there are two ways to enforce the same rule. The recommended combination is to use both.

| Option | How it works | Strengths | Weaknesses |
|---|---|---|---|
| **A. WHERE clause in every query** | `WHERE tenant_id = $1 AND acl && $2::text[]`, with a GIN index on `acl`. The principal's tokens are bound as a parameter. | Explicit in the code. Works with any connection pooling. The planner can use the GIN index. | A query that forgets the clause leaks. It has to be checked in review and tested. |
| **B. Row-level security** | `CREATE POLICY ... USING (tenant_id = current_setting('app.tenant_id'))`, with the session variable set by `SET LOCAL` in each transaction. | The database enforces it, so a forgotten clause still fails closed. | The table owner bypasses RLS unless `FORCE ROW LEVEL SECURITY` is set. Pooled connections must reset the setting. Matching ACL arrays in a policy is possible but hard to read and to tune. |

**Recommendation:** use A for the ACL (the planner needs it for performance), and B for tenant isolation as a backstop. The app role must not own the tables.

### Post-filter is rejected

Retrieving the top k over everything and then filtering can return fewer than k results. It can also let scores computed over other tenants influence the result. BM25 IDF is the clearest example: a post-filter would compute term statistics across tenants.

## Consequences

- Access is checked in one place per adapter, and the type signature forces callers to supply it. Reviewing a new adapter means checking one argument.
- The test suite exercises isolation end to end (`tests/security/test_isolation.py`).
- Costs: the ACL mask is a Python loop over every chunk in the tenant partition (`vector.py:61-65`). Per-query work grows with corpus size, not with the number of results. That is acceptable up to the sizes noted in `docs/operations.md`, and is the first thing to move into the database.
- ACL data is a snapshot taken at ingest. Permission changes in the source system take effect only after the document is re-ingested.
- Group membership comes from the credential at request time, so removing a user from a group takes effect on their next request. The cache key includes the token set (`acl.py:30-33`, `cache.py:40-43`), so the user stops hitting entries built for the old set straight away. Those entries stay in memory until their TTL expires, and they match again if the group is restored.

## Alternatives considered

- **Per-user indexes.** Simple to reason about, but storage grows with users and every ACL change means copying data. Rejected.
- **Application-level filtering after retrieval.** Rejected, as above.
- **An external policy engine (OPA, Cedar) evaluated per chunk.** Powerful, but a network call per candidate is too slow for the candidate pool. Revisit it for coarse rules, such as which tenants a user may query at all.
- **Permissions looked up from the source system at query time.** Gives freshest permissions, but adds a dependency on every source system in the query path. Left as a later option, and noted as a gap in `docs/security.md`.
