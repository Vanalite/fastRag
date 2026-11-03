# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/) (pre-1.0: minor versions may break).

## [Unreleased]

### Added
- Postgres + pgvector storage (`FASTRAG_POSTGRES_DSN`, `fastrag[postgres]`): documents,
  vectors (HNSW) and keyword search (full-text over fastrag's own terms) live in the
  database, with the access filter in every query's WHERE clause, so all replicas see
  the same data. Schema migrations run under an advisory lock. A CI job runs the
  integration tests against a real pgvector database.
- Optimistic concurrency: writes carry the version they expect to replace; a stale write
  is rejected (`409 ConcurrentModification`) and the ingest service retries with fresh
  state, re-checking permissions. Two processes or replicas writing one document can no
  longer produce duplicate versions.
- Background ingestion jobs: `POST /v1/ingest-jobs` (up to 500 documents, 202 +
  `Location`) and `GET /v1/ingest-jobs/{id}` for progress and per-document results.
  Each document is authorized and audited as the submitter; tenants have a cap on active
  jobs; status is shared through Redis when configured.
- Connector sync (`fastrag sync`, `SyncService`, `Connector`): permissions follow the
  source. A manifest-based connector maps folders to ACLs (deny by default); re-syncs
  update content and permissions and prune documents deleted at the source, never
  after a failed listing. Metadata keys starting with `_` are now reserved. Manifest
  globs are folder-aware (`*` never crosses `/`), large prunes need `--force-prune`, and
  documents the connector didn't create are not adopted without `--adopt`.
- Jobs heartbeat (an abandoned job stops counting against the tenant cap after 15
  minutes), stop cleanly on shutdown, are paced per tenant, and each replica bounds the
  jobs and text it holds in memory.
- `fastrag db migrate`, and `FASTRAG_POSTGRES_MIGRATE_ON_START=false` so production
  replicas can run with a role that can't change the schema. Pool size and HNSW
  `ef_search` are configurable.
- Redis backends (`FASTRAG_REDIS_URL`, `fastrag[redis]`): rate-limit buckets (an atomic
  token-bucket script on the Redis clock) and the answer cache are shared by every
  replica. Cached answers are stored as JSON, never pickle.
- Streaming answers: `POST /v1/ask/stream` (server-sent events) and
  `RAGService.ask_stream`. Errors before the stream opens are ordinary HTTP errors;
  errors after it opens arrive as an `error` event. Abandoned streams are audited as
  `cancelled`.
- Optional reranker (`FASTRAG_RERANKER=cross-encoder[:<model>]`, needs `fastrag[st]`)
  over the top `FASTRAG_RERANK_POOL` candidates. Rerankers can reorder, never add.
- OpenTelemetry spans per operation (`fastrag[otel]`), with timings and counts but no
  questions, document text or user ids.
- PII in questions is redacted before they are sent to the LLM
  (`FASTRAG_REDACT_QUESTIONS`, on by default); retrieval still uses the original.

### Changed
- Changing a tenant's corpus purges that tenant's cached answers immediately, rather
  than leaving them unreachable until their TTL expires.
- An answer is not cached if the tenant's corpus changed while it was being generated.
- Redis outages degrade instead of failing requests: cache misses, and rate limits fail
  open with an error log (alert on it). Unreadable cache entries count as misses.
- A failing reranker falls back to the fused order; a tracing failure never blocks the
  audit entry.

### Fixed
- A streaming client that disconnected was audited only when the garbage collector got
  to it, on another thread and without its request id; one that disconnected right after
  receiving the answer was audited as `cancelled`. Both are now recorded correctly.
- Purging a tenant's cached answers is atomic in Redis, so an entry written during the
  purge can't survive untracked.
- `rerank_score` was lost when an answer came from the Redis cache.

## [0.3.0] - 2026-11-03

### Upgrade notes
- **Reading no longer implies writing.** Documents now have `editors` in addition to
  their read `acl`. Replacing or deleting a document needs a tenant admin, or a listed
  editor who can also read it. Documents created before this release have no editors, so
  only admins can change them until editors are set (the database migrates in place).
- **Throttled addresses wait.** After too many failed logins from one address, requests
  from it get `429` before credentials are checked, including requests with valid
  credentials, until the budget refills. Behind a proxy, run uvicorn with
  `--forwarded-allow-ips` so each client gets its own budget.
- **Re-ingest re-chunks.** Chunking changed (see below) and `PIPELINE_VERSION` is now 3,
  so re-ingesting an unchanged document creates a new version instead of returning
  "unchanged".
- **Editors change content, not sharing.** Replacing a document with a different `acl`
  or `editors` needs a tenant admin.
- Bearer tokens must carry `iat`, and lifetimes over 24 hours are rejected.

### Added
- OIDC sign-in: tokens are verified against the identity provider's JWKS (RS256, ES256,
  PS256), with issuer and audience required and claim names configurable.
- Production mode (`FASTRAG_ENVIRONMENT=production`) refuses to start with an unsafe
  configuration. Validation errors never echo raw settings, so secrets can't reach logs.
- API key lifecycle: key ids, expiry (`keys create --ttl-days`), `keys list` and
  `keys revoke`. Changes reach a running server without a restart.
- Per-user rate limits inside each tenant's budget, and failed-login throttling per client
  address, audited as one aggregated entry per minute.
- `editors` on documents (API and storage), with an in-place schema migration (v2).
- `grounded` on answers; answers citing nothing are counted and audited as ungrounded.
- Pagination for `GET /v1/documents` (`limit`, `offset`, `X-Total-Count`).
- `scripts/claude_smoke.py`: a live check of the Claude path.

### Changed
- Chunking keeps paragraph and line breaks, and overlap carries whole sentences.
- Exact duplicate chunks are dropped from results, so copies don't crowd out evidence.
  Near copies are kept: they may differ in the one fact that matters.
- PII redaction also covers metadata, including keys; metadata nesting is limited to 8.
- The audit log appends under a file lock and links to the file's last line, so several
  processes can share one log. A line torn by a crash is repaired; a corrupt complete
  line stops writes until someone investigates.
- Missing Anthropic credentials raise a configuration error instead of a bare `TypeError`.

### Fixed
- An empty cache or rate limiter passed to `RAGService` was silently replaced by a
  default one, because both define `__len__` and an empty one is falsy.
- `MemoryAuditSink` failed verification once its size cap dropped old entries.
- Revoking an already revoked key overwrote the original revocation time.
- Operator CLI commands are no longer subject to the HTTP rate limits, so a bulk
  `fastrag ingest` doesn't stop at the burst size.
- `fastrag token` uses the configured claim names.
- A database with a newer schema than the code supports is refused instead of being
  rewritten.

## [0.2.0] - 2026-11-03

### Added
- Enterprise architecture: tenants, RBAC, document ACLs enforced inside every index,
  SQLite system of record with rebuildable indexes, hybrid BM25 + dense retrieval with
  reciprocal rank fusion, evidence-gated abstention, citation validation,
  prompt-injection flagging, PII redaction, hash-chained audit log, answer cache scoped
  by access, per-tenant rate limits, metrics, structured logs, Docker image, CI.
- Security acceptance tests and a golden-set retrieval evaluation gate.
- Architecture, security and operations docs, and six ADRs.

## [0.1.0] - 2026-11-03

### Added
- First scaffold: chunking, hashing embedder, in-memory vector store, extractive and
  Claude generators, CLI and HTTP API.
