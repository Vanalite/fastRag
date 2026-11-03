# ADR 0002: Access control with Principals, ACL tokens, editor lists and mandatory access filters

- **Status:** Accepted. The pgvector adapter (ADR 0007) implements option A below; option B is not enabled. The write rule is accepted. Its sharing rule was tightened after the first version (see Consequences).
- **Date:** 2026-11-03

## Context

Every RAG leak in the wild has the same root cause: retrieval ran over everything, and access was checked later, if at all. Access needs to be decided at the point where data leaves the index, and the design should make it hard to skip that check. Writes need the same care: a reader who can overwrite a document can change what everyone else is told.

## Decision

**Identity and tokens.** `Principal` (`src/fastrag/domain.py`) carries `tenant_id`, `user_id`, `groups` and `roles`. `Principal.acl_tokens()` expands it into a set of tokens: `tenant:*`, `user:<id>` and one `group:<g>` per group. A document's ACL is a set of the same tokens (`ACL_TOKEN_PATTERN`).

**Read rule.** A chunk is visible to a principal when the tenant matches **and** the two token sets intersect (`security/acl.py`, `AccessFilter.allows`). An empty ACL matches nothing. Ingest rejects an empty ACL and any token outside the pattern.

**Write rule.** Each document also has `editors`, a set of the same tokens. Replacing or deleting a document is allowed when:

- the caller is an admin of the document's tenant, **or**
- the caller can read the document **and** matches one of its `editors`.

Reading does not imply writing. Changing an existing document's `acl` or `editors` is admin-only, so an editor cannot publish a restricted document to a wider audience. A new document's editors default to its uploader (`user:<id>`). An update that names no editors keeps the existing ones. A document migrated from schema v1 has no editors, so only admins can change it. The check (`RAGService._can_modify`) runs under the document's lock, before anything else is revealed. Delete returns the same 404 whether the document is absent or not modifiable.

**Every chunk carries its access data.** `Chunk` has `tenant_id` and `acl` (`domain.py`), copied from the document at ingest. The index can therefore decide without a join.

**Mandatory access argument.** `VectorIndex.search` and `KeywordIndex.search` take `access` as a keyword-only, required argument (`src/fastrag/storage/base.py`). There is no way to query an index without naming the caller. `SearchIndexes.dense` and `.keyword` pass it through.

**Pre-filter, not post-filter.** The in-memory adapters apply the filter before ranking:

- Dense: the ACL mask is built over the tenant partition before the matrix product (`storage/vector.py`). On Postgres, the same condition is in the SQL `WHERE` clause (`storage/postgres.py`).
- Keyword: each posting is skipped unless the chunk is allowed, before scoring (`storage/bm25.py`). On Postgres, the full-text query has the same `WHERE` condition.

**Tenant-partitioned indexes.** Each adapter keeps one partition per tenant. BM25 term statistics are also per tenant, so one tenant's corpus cannot change another tenant's scores.

**Defense in depth.** The retriever re-checks each result (`retrieval/retriever.py`, `Retriever.retrieve`). A chunk that fails the check is dropped and logged at critical level. This protects against a buggy adapter, not against a missing filter in the caller.

**Deny by default** applies at each level: unknown tenant, empty ACL, unmatched token, unknown role. A token with no known fastRag role is refused.

**Group membership** comes from the bearer token at request time, so removing a user from a group takes effect on their next request. API keys carry their groups in the key record, so a group change means a new key.

### pgvector: WHERE clause or Postgres row-level security

When the vector index moves to Postgres, there are two ways to enforce the same rule. The recommended combination is to use both.

| Option | How it works | Strengths | Weaknesses |
|---|---|---|---|
| **A. WHERE clause in every query** | `WHERE tenant_id = $1 AND acl && $2::text[]`, with a GIN index on `acl`. The principal's tokens are bound as a parameter. | Explicit in the code. Works with any connection pooling. The planner can use the GIN index. | A query that forgets the clause leaks. It has to be checked in review and tested. |
| **B. Row-level security** | `CREATE POLICY ... USING (tenant_id = current_setting('app.tenant_id'))`, with the session variable set by `SET LOCAL` in each transaction. | The database enforces it, so a forgotten clause still fails closed. | The table owner bypasses RLS unless `FORCE ROW LEVEL SECURITY` is set. Pooled connections must reset the setting. Matching ACL arrays in a policy is possible but hard to read and to tune. |

**Recommendation:** use A for the ACL (the planner needs it for performance), and B for tenant isolation as a backstop. The app role must not own the tables. The implementation uses A only; the RLS policy and its trade-offs are in `docs/operations.md`.

### Post-filter is rejected

Retrieving the top k over everything and then filtering can return fewer than k results. It can also let scores computed over other tenants influence the result. BM25 IDF is the clearest example: a post-filter would compute term statistics across tenants.

## Consequences

- Access is checked in one place per adapter, and the type signature forces callers to supply it. Reviewing a new adapter means checking one argument.
- The test suite exercises isolation end to end (`tests/security/test_isolation.py`) and the write rule (`tests/security/test_http_security.py`).
- Costs: the ACL mask is a Python loop over every chunk in the tenant partition. Per-query work grows with corpus size, not with the number of results. That is acceptable up to the sizes noted in `docs/operations.md`, and is the first thing to move into the database.
- ACL data is a snapshot taken at ingest. Permission changes in the source system take effect only after the document is re-ingested.
- **Editors cannot widen an existing document.** `IngestService.ingest` refuses a non-admin's replace that changes `acl` or `editors` (`can_change_sharing`; `tests/security/test_http_security.py`, `test_editors_cannot_change_who_may_read_or_edit`). Creating a document still sets its ACL freely, and the uploader becomes its first editor.
- Removing a user from a group takes effect on their next token. The cache key includes the token set (`AccessFilter.scope_key`, `AnswerCache.key`), so the user stops hitting entries built for the old set straight away. Those entries stay in memory until their TTL expires, and they match again if the group is restored.

## Alternatives considered

- **Per-user indexes.** Simple to reason about, but storage grows with users and every ACL change means copying data. Rejected.
- **Application-level filtering after retrieval.** Rejected, as above.
- **An external policy engine (OPA, Cedar) evaluated per chunk.** Powerful, but a network call per candidate is too slow for the candidate pool. Revisit it for coarse rules, such as which tenants a user may query at all.
- **Permissions looked up from the source system at query time.** Gives freshest permissions, but adds a dependency on every source system in the query path. Left as a later option, and noted as a gap in `docs/security.md`.
- **Writes allowed to anyone who can read.** Rejected. A reader could overwrite or delete documents it can see, which is how a published policy gets quietly changed. The editor list is the narrower grant.
