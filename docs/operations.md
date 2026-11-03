# Operations

**Last updated:** 2026-11-03
**Entry points:** `src/fastrag/config.py` (settings), `src/fastrag/api/app.py` (HTTP), `src/fastrag/cli.py` (operator commands)

This page describes the code as it is now. Where a default changed, the table says so.

## Configuration reference

All settings come from environment variables named `FASTRAG_<FIELD>` (`config.py`, `Settings`), or from a `.env` file in the working directory. Environment variables win over the file. The `ANTHROPIC_API_KEY` variable is read by the Anthropic SDK, not by fastRag.

### Storage and retrieval

| Variable | Default | Meaning |
|---|---|---|
| `FASTRAG_POSTGRES_DSN` | unset | Postgres with pgvector for documents, vectors and keyword search, so every replica shares them. Needs `fastrag[postgres]`. When set, `FASTRAG_DATABASE_PATH` is not used. SecretStr. |
| `FASTRAG_POSTGRES_POOL_SIZE` | `10` | Connections per replica. |
| `FASTRAG_POSTGRES_EF_SEARCH` | `100` | HNSW candidates per query (`hnsw.ef_search`); higher means better recall under selective filters, slower queries. |
| `FASTRAG_POSTGRES_MIGRATE_ON_START` | `true` | Apply schema migrations at startup. Set `false` in production and run `fastrag db migrate` during deploys. |
| `FASTRAG_DATABASE_PATH` | `.fastrag/fastrag.db` | SQLite file, used when no Postgres DSN is set. `:memory:` gives an ephemeral store. |
| `FASTRAG_EMBEDDER` | `hashing` | `hashing` or `hashing:<dim>` (dim 1024 by default), or `st:<model>` with the `st` extra. The store records the name and refuses to start with a different one. |
| `FASTRAG_CHUNK_SIZE` | `800` | Characters per chunk, 100 to 8000. |
| `FASTRAG_CHUNK_OVERLAP` | `100` | Maximum characters of whole trailing sentences carried into the next chunk. Must be smaller than the chunk size. |
| `FASTRAG_MAX_DOCUMENT_CHARS` | `2000000` | Largest accepted document. The API schema also caps text at 2,000,000 characters. |
| `FASTRAG_REDACT_PII` | `true` | Regex redaction of email, SSN, card numbers and phone numbers in text, title and metadata strings, at ingest. |
| `FASTRAG_REDACT_QUESTIONS` | `true` | Applies the same redaction to the question sent to the LLM. Retrieval uses the original question locally. Independent of `FASTRAG_REDACT_PII`. |
| `FASTRAG_TOP_K` | `4` | Chunks returned and shown, 1 to 20. |
| `FASTRAG_CANDIDATE_POOL` | `20` | Candidates fetched from each index before fusion, 1 to 200. |
| `FASTRAG_RRF_K` | `60` | Reciprocal-rank-fusion constant. |
| `FASTRAG_RERANKER` | unset | Second pass over the fused candidates: `cross-encoder` or `cross-encoder:<model>`. Needs `fastrag[st]`. The model loads at startup. Reranking reorders; it never adds chunks. |
| `FASTRAG_RERANK_POOL` | `20` | Candidates given to the reranker, 1 to 200. Used only when a reranker is set. |
| `FASTRAG_MIN_DENSE_SCORE` | unset (`None`) | Dense cosine threshold for evidence. Unset uses the embedder's default: 0.25 for hashing, 0.35 for sentence-transformers. |
| `FASTRAG_MIN_KEYWORD_COVERAGE` | `0.3` | IDF-weighted share of the query's terms a chunk must contain for the keyword signal to count. Unless every query term matched, at least two must. Calibrated on `eval/golden`. |

### Generation

| Variable | Default | Meaning |
|---|---|---|
| `FASTRAG_GENERATOR` | `extractive` | `extractive` quotes passages with no model call. `claude` calls the Messages API. |
| `FASTRAG_CLAUDE_MODEL` | `claude-opus-5-5` | Model name for the `claude` generator. |
| `FASTRAG_CLAUDE_EFFORT` | `medium` | One of `low`, `medium`, `high`, `xhigh`, `max`. |
| `FASTRAG_MAX_QUESTION_CHARS` | `2000` | Longest accepted question. The API schema also caps it at 2000. |
| `FASTRAG_DROP_SUSPICIOUS_CHUNKS` | `false` | Remove chunks that match the injection heuristic before generation. |

### Security

| Variable | Default | Meaning |
|---|---|---|
| `FASTRAG_ENVIRONMENT` | `development` | `production` makes startup fail on unsafe settings (see "Production mode"). |
| `FASTRAG_API_KEYS_FILE` | `.fastrag/api_keys.json` | Hashed API keys. At least one authentication method is required. |
| `FASTRAG_JWT_SECRET` | unset | HS256 shared secret, at least 32 characters. For service tokens and demos. |
| `FASTRAG_OIDC_JWKS_URL` | unset | The identity provider's JWKS URL, https only. Enables RS256, ES256 and PS256 bearer tokens. Requires `FASTRAG_JWT_ISSUER` and `FASTRAG_JWT_AUDIENCE`. |
| `FASTRAG_JWT_ISSUER` | unset | Expected `iss`. Required with OIDC, and with a shared secret in production. |
| `FASTRAG_JWT_AUDIENCE` | unset | Expected `aud`. Required with OIDC, and with a shared secret in production. |
| `FASTRAG_TOKEN_TENANT_CLAIM` | `tenant` | Claim that carries the tenant id. |
| `FASTRAG_TOKEN_GROUPS_CLAIM` | `groups` | Claim that carries group names. |
| `FASTRAG_TOKEN_ROLES_CLAIM` | `roles` | Claim that carries fastRag roles. Unknown values are ignored, and at least one known role is required. |

### Operations

| Variable | Default | Meaning |
|---|---|---|
| `FASTRAG_REDIS_URL` | unset | Shares the tenant, user and auth-failure rate-limit buckets and the answer cache across processes. Needs `fastrag[redis]`. Pinged at startup. The URL carries a password: SecretStr. |
| `FASTRAG_AUDIT_LOG_PATH` | `.fastrag/audit.jsonl` | Audit file. Unset keeps entries in memory, up to 100,000, and they are lost on restart. |
| `FASTRAG_AUDIT_STORE_QUERY_TEXT` | `false` | Store question text in audit entries. The default stores only a SHA-256. |
| `FASTRAG_CACHE_TTL_SECONDS` | `300` | Answer cache lifetime. `0` disables the cache. |
| `FASTRAG_CACHE_MAX_ENTRIES` | `1000` | Cache size limit for the in-process cache. `0` disables the cache. |
| `FASTRAG_RATE_LIMIT_PER_MINUTE` | `120` | Token-bucket refill rate per **tenant**, applied to every service operation (queries, ingest, delete, listing, audit reads). `0` disables. |
| `FASTRAG_RATE_LIMIT_BURST` | `30` | Tenant bucket size. `0` disables. |
| `FASTRAG_USER_RATE_LIMIT_PER_MINUTE` | `60` | Refill rate per **user** within the tenant. Checked before the tenant bucket. `0` disables. |
| `FASTRAG_USER_RATE_LIMIT_BURST` | `15` | User bucket size. `0` disables. |
| `FASTRAG_AUTH_FAILURES_PER_MINUTE` | `30` | Failed authentications per client address. Past the limit the response is 429. `0` disables. |
| `FASTRAG_AUTH_FAILURE_BURST` | `10` | Failed-authentication bucket size. `0` disables. |
| `FASTRAG_LOG_LEVEL` | `INFO` | One of `DEBUG`, `INFO`, `WARNING`, `ERROR`. |
| `FASTRAG_LOG_JSON` | `true` | JSON logs to stderr, with `request_id` on each line. |
| `FASTRAG_EXPOSE_METRICS` | `false` | Serves `/metrics`. Off by default because the endpoint is unauthenticated and its labels include tenant ids. |

Tracing has no `FASTRAG_` setting. Install `fastrag[otel]` and an OpenTelemetry SDK, and configure export the usual way (`OTEL_*`, or `opentelemetry-instrument`).

### Production mode

With `FASTRAG_ENVIRONMENT=production`, startup fails when any of these hold: the store is SQLite and `database_path` is `:memory:` (a Postgres DSN satisfies this check); the audit log path is unset; no authentication method is configured; `FASTRAG_JWT_SECRET` is set without issuer and audience; `FASTRAG_LOG_JSON=false`; the tenant or auth-failure rate limit is `0`. Redis is not required. Settings validation errors never print the submitted value.

## Endpoints

Authentication is an `Authorization: Bearer <jwt>` header or an `X-API-Key: <key>` header. The OpenAPI document at `/docs` declares both schemes. Every response carries `X-Request-ID` (echoed from the request if it matches `[A-Za-z0-9._-]{1,128}`, otherwise generated) and `Cache-Control: no-store`.

| Method and path | Auth | Permission | Success | Notes |
|---|---|---|---|---|
| `GET /healthz` | no | none | 200 | Liveness. Returns `{"status": "ok"}` whenever the process answers. |
| `GET /readyz` | no | none | 200 or 503 | Runs `store.ping()` only (the SQLite file or the Postgres pool). |
| `GET /metrics` | no | none | 200 | Only when `FASTRAG_EXPOSE_METRICS=true`. Prometheus text format. |
| `GET /v1/me` | yes | none | 200 | Returns the caller's tenant, user, groups and roles. |
| `POST /v1/documents` | yes | `ingest` (ingester, admin) | 201 | Body: `source_id`, `text`, `acl` (1 to 256 tokens), optional `title`, `media_type`, `metadata` (16 KiB serialized), `editors` (up to 256 tokens). Re-ingesting a `source_id` makes a new version. Replacing needs an admin of the tenant, or a caller that can read the document and is in its `editors`. Only admins may change an existing document's `acl` or `editors`. 409 if the document changed concurrently and retries ran out. |
| `GET /v1/documents?limit=N&offset=M` | yes | `query` | 200 | `limit` 1 to 500 (default 100), `offset` 0 or more. The `X-Total-Count` header holds the total. Admins see every document's metadata. Everyone else sees the documents their ACL allows. Each item includes `acl` and `editors`. |
| `DELETE /v1/documents/{doc_id}` | yes | `delete` (ingester, admin) | 204 | Same modify rule as replacement. 404 if the document is absent or not modifiable. 409 as for `POST`. |
| `POST /v1/ingest-jobs` | yes | `ingest` (ingester, admin) | 202 + `Location` | Body `{"documents": [...]}`: 1 to 500 `POST /v1/documents` bodies, at most 20,000,000 characters in total. Returns the job at once (`queued`); worker threads ingest each document as the caller, with the usual permission checks and one audit entry per document (tagged with `job_id`). 429 when the tenant already has `FASTRAG_MAX_ACTIVE_JOBS_PER_TENANT` active jobs. |
| `GET /v1/ingest-jobs/{job_id}` | yes | `ingest` | 200 | Job status (`queued`, `running`, `succeeded`, `partial`, `failed`, `cancelled`), per-status `counts`, and per-document results with `doc_id`, `version` or `error`. Visible to the submitter and tenant admins; 404 for anyone else. Kept 24 hours. |
| `POST /v1/search` | yes | `query` | 200 | Body: `question`, optional `k`. Returns chunks (`rerank_score` is null without a reranker) and the `confident` flag. Does not flag or drop suspicious chunks. |
| `POST /v1/ask` | yes | `query` | 200 | Body: `question`, optional `k`. Returns `answer`, `abstained`, `grounded`, `cached`, `citations`, `sources` and token `usage`. |
| `POST /v1/ask/stream` | yes | `query` | 200 `text/event-stream` | Same body. Events: `delta` (`{"text"}`, raw model output, not validated), then one `answer` event with the same shape as `/v1/ask`. Keep the `answer` event as the result. A failure after streaming starts is an `error` event (`{"error": {"type", "message"}, "request_id"}`) with a generic message, and the stream ends. Errors before the first event are normal HTTP errors. |
| `GET /v1/audit?limit=N` | yes | `read_audit` (admin) | 200 | Entries for the caller's tenant, newest last, up to 1000. Reads are audited as `read_audit`. |

Error responses use one envelope: `{"error": {"type": "...", "message": "..."}, "request_id": "..."}`. The mapping is in `src/fastrag/api/app.py` (`_STATUS`, `domain_error`):

| Exception | Status | Notes |
|---|---|---|
| `AuthenticationError` | 401 | Generic message, `WWW-Authenticate: Bearer`. |
| `PermissionDenied` | 403 | |
| `NotFound` | 404 | |
| `ConcurrentModification` | 409 | The message names the document id and its versions. Retried internally first (up to 3 attempts), so clients see it only under heavy contention. |
| `InvalidInput`, request validation | 422 | Submitted input is not echoed back. |
| `RateLimited` | 429 | `Retry-After` in seconds. Used for user and tenant buckets and for failed authentications. |
| `GenerationError` | 502 | LLM failed or refused. |
| `ConfigurationError` and anything else | 500 | Generic message. The exception is logged with the request id. A missing Anthropic credential is a `ConfigurationError`. |

Failed authentication is audited with `tenant_id: null` and counted as `fastrag_requests_total{outcome="unauthenticated"}`. Attempts past the per-address limit get 429 before their credentials are checked, are counted as `outcome="throttled"`, and are audited as one aggregated `throttled` entry per client per minute (with an `attempts` count).

## Background ingestion and connector sync

**Jobs** (`src/fastrag/jobs.py`) run in worker threads inside the API process (`FASTRAG_INGEST_WORKERS`, default 2). Job status is in memory, or in Redis when `FASTRAG_REDIS_URL` is set, so any replica can answer a poll. On a clean shutdown, queued jobs are cancelled and running ones stop after their current document (status `cancelled`, finished documents kept). Workers heartbeat on every document: a job with no heartbeat for 15 minutes (its process died) stops counting against the tenant's cap and is reported `failed`, with unprocessed documents marked for resubmission. Background documents are paced per tenant (`FASTRAG_JOB_DOCUMENTS_PER_MINUTE`), and each replica bounds what it holds in memory (`FASTRAG_MAX_QUEUED_JOBS`, `FASTRAG_MAX_QUEUED_CHARS`; over either, 429). Queued jobs run in submission order, so one tenant's large backlog can delay others on the same replica; for strict fairness or delivery guarantees across restarts, put an external queue behind the same `submit`/`get` interface. A job runs with the permissions its submitter had at submission time.

| Setting | Default | Meaning |
| --- | --- | --- |
| `FASTRAG_INGEST_WORKERS` | `2` | Worker threads per replica. |
| `FASTRAG_MAX_ACTIVE_JOBS_PER_TENANT` | `10` | Queued or running jobs per tenant; more get 429. |
| `FASTRAG_MAX_DOCUMENTS_PER_JOB` | `500` | Documents per job. |
| `FASTRAG_MAX_QUEUED_JOBS` | `100` | Jobs one replica holds (queued or running). |
| `FASTRAG_MAX_QUEUED_CHARS` | `100000000` | Characters of document text one replica holds for queued jobs. |
| `FASTRAG_JOB_DOCUMENTS_PER_MINUTE` / `_BURST` | `600` / `50` | Background ingestion pace per tenant; workers wait, they don't fail. |

**Connector sync** keeps a tenant's corpus, and its permissions, in step with a source (`src/fastrag/ingest/sync.py`). The shipped `ManifestConnector` reads a folder plus a JSON manifest of ACL rules; connectors for SharePoint, Confluence or Drive implement the same `Connector` interface (`name`, `documents()`):

```bash
fastrag sync ./share --tenant acme --manifest acl.json
```

```json
{"rules": [
  {"pattern": "finance/**", "acl": ["group:finance"], "editors": ["group:finance-leads"]},
  {"pattern": "**", "acl": ["tenant:*"]}
]}
```

- Patterns are folder-aware globs on the path: `*` and `?` stay within one folder, `**` spans folders. So `*.md` matches only top-level files and `finance/**` everything under `finance/`. (Unlike shell `fnmatch`, where `*` would silently match `hr/payroll.md`.)
- The first matching rule wins; a manifest with rules after a `**` catch-all is rejected, because they could never match. Files no rule matches are skipped (deny by default) and reported.
- A re-sync updates changed documents and re-stamps permissions that changed at the source.
- Documents that disappeared from the source are deleted (`--no-prune` keeps them). Only documents this connector created (metadata `_connector` equal to its name) are ever pruned, and only after a complete listing: a source that fails halfway prunes nothing. A prune that would delete more than half of a connector's documents (when it owns 10 or more) is refused unless `--force-prune` is given, since a half-mounted source looks like a mass deletion.
- An existing document with the same path that the connector didn't create (e.g. uploaded by hand) is left alone and reported; `--adopt` takes it over (it then becomes prunable). Editing a synced document over the API keeps its connector ownership.
- Connector names must be unique within a tenant: two connectors with the same name would prune each other's documents.
- Sync runs as a tenant admin, because permission changes on existing documents need one. Every change is audited. Metadata keys starting with `_` are reserved and rejected by the API.

**Migrations.** `fastrag db migrate` applies pending schema migrations and exits (see "Scaling").

## Health, readiness and metrics

- **Liveness:** `/healthz` only shows that the process answers HTTP. Use it for restarts.
- **Readiness:** `/readyz` pings the store. It does not check Redis, that the indexes match the store, that the embedder loaded, or that the LLM is reachable.
- **Metrics** (`src/fastrag/observability/metrics.py`):

| Metric | Labels | Meaning |
|---|---|---|
| `fastrag_requests_total` | `action`, `outcome`, `tenant` | One per service call. Outcomes: `success`, `abstained`, `denied`, `rate_limited`, `not_found`, `invalid`, `generation_error`, `error`, `cancelled` (stream abandoned by the client), `unauthenticated`, `throttled` (`action="authenticate"`, `tenant="-"`). |
| `fastrag_stage_seconds` | `stage` | Histogram for `retrieval`, `generation` and each action. |
| `fastrag_llm_tokens_total` | `tenant`, `kind` | Input, output and cache tokens from generation. |
| `fastrag_injection_flags_total` | `tenant` | Chunks flagged by the injection heuristic. |
| `fastrag_invalid_citations_total` | `tenant` | Citation markers that pointed at unshown sources. |
| `fastrag_ungrounded_answers_total` | `tenant` | Generated answers that cite none of their sources. |
| `fastrag_pii_redactions_total` | `kind` | Redactions at ingest, and in questions sent to the LLM, by kind. |

Keep label cardinality low. Tenant ids are acceptable. Never add user ids or query text.

## Audit log

Each line of `.fastrag/audit.jsonl` is one entry: `ts`, `request_id`, `action`, `outcome`, `tenant_id`, `user_id`, `details`, `prev_hash`, `hash` (`src/fastrag/observability/audit.py`). An `ask` records `question_redactions` (counts by kind) when the LLM question was redacted, and `streamed: true` for streamed calls.

**Writers.** Each append takes an exclusive `flock` on the file, repairs a torn final line (one cut off before its newline, which can only come from a crash mid-write), reads the last entry's hash, and appends. So several workers and CLI calls on one host extend one chain. The lock is POSIX-only. On other platforms run one writer. Keep the file on local disk, because `flock` is not reliable on network filesystems. A last line that parses but is not a valid entry stops appends with a `ConfigurationError`.

**Readers** do not take the lock. A read during an append can see a partial last line. Retry the read.

**Verify:** `fastrag audit verify` prints `N entries; chain OK` or `chain BROKEN`, and exits 1 when broken. It checks the chain only. It cannot detect a truncated tail or a forged file (see `docs/security.md`).

**Failure behavior.** An audit write that fails is logged and never replaces the original exception. If the call itself succeeded, the request fails (fail closed).

## Keys and tokens

**API keys** are the default for services. The `fastrag keys` commands edit `FASTRAG_API_KEYS_FILE` directly.

```sh
fastrag keys create --tenant northwind --user svc-ingest --role ingester --group engineering --ttl-days 90
fastrag keys list --tenant northwind
fastrag keys revoke 3fa9c1d2
```

- `create` prints the key once, to stdout. The file stores only its SHA-256. The key id is the first 12 hex characters of that hash, and is shown in listings and the audit log. Options: `--tenant` and `--user` (both required), `--group` (repeatable), `--role` (repeatable: `reader`, `ingester` or `admin`; default `reader`), `--ttl-days` (optional, positive).
- `list` shows id, state (`active`, `expired` or `revoked`), owner, roles and groups. It never shows keys.
- `revoke` takes a key id prefix of at least 6 characters, which must match exactly one key. A running server sees the change on its next request, without a restart.
- Writers take an exclusive `flock` on `api_keys.lock`, re-read the file, then write it through `api_keys.tmp` created 0600.
- Keys are not rotated automatically. Rotate by creating a new key, updating the caller, and revoking the old one.
- Group membership is stored in the key record. Changing a service's groups means issuing a new key.
- Replicas that read the same keys file see revocations. Each replica must have the file, so use shared storage or distribute the file.

**Bearer tokens.** Production tokens come from the identity provider. Set `FASTRAG_OIDC_JWKS_URL`, `FASTRAG_JWT_ISSUER` and `FASTRAG_JWT_AUDIENCE`. The server accepts RS256, ES256 and PS256 tokens whose `exp`, `iat`, `sub`, `iss` and `aud` claims are present and valid, with 30 seconds of leeway. Keys are cached for 5 minutes, and an unknown `kid` triggers a refetch. Tokens longer than 24 hours (`exp` minus `iat`) are rejected, with HS256 as well.

**Demo tokens** (`fastrag token`) need `FASTRAG_JWT_SECRET` and are for local use only: `fastrag token --tenant northwind --user alice --group finance --role reader --ttl 3600`. The demo minter always writes the default claim names, so if `FASTRAG_TOKEN_*_CLAIM` is changed its tokens will not verify. Keep `--ttl` at or below 86400.

**CLI commands bypass HTTP authentication.** `ingest` and `delete` run as an admin operator named `cli-operator`. `ask`, `search` and `documents` run as the principal you name. Only run the CLI where you could read the data files directly. The `ingest` default ACL is `tenant:*`. Pass `--acl` for anything narrower. The CLI is not rate limited; it is still authorized and audited.

## Scaling

Two shapes are supported. Pick one.

**1. SQLite, one process per host (default).** Set nothing extra. Run one process (`fastrag serve` uses one worker). Each process loads its own indexes, so an ingest through one process is invisible to any other process on the same file until restart. A delete is not honoured there either. Do not run a second process against the same file, and do not run several workers in one instance.

**2. Postgres with pgvector, several replicas.** Set `FASTRAG_POSTGRES_DSN`, and `FASTRAG_REDIS_URL` if rate limits and the cache must be shared. Every replica queries the same rows, so ingest and delete are visible everywhere as soon as they commit. Before starting replicas:

- Create the `vector` extension yourself, or give the migration role the right to. Migrations run at startup under an advisory lock.
- Set `statement_timeout` on the role or in the DSN (`options=-c statement_timeout=5s`).
- Connections: each process opens up to 10 (`PostgresDocumentStore(pool_size=10)`). No setting exposes this yet, so budget `replicas × 10` against `max_connections`, or put PgBouncer in transaction mode in front (the per-query search settings are transaction-local, so this is safe).
- Run migrations as a deploy step with a role that may change the schema: `fastrag db migrate`. Then start the replicas with `FASTRAG_POSTGRES_MIGRATE_ON_START=false`, so the app can run as a DML-only role: it only checks that the schema is current (and refuses to start if not). The default (`true`) migrates at startup under an advisory lock, which needs DDL rights. Never edit a released migration step; append a new one.
- Backups: continuous WAL archiving with point-in-time recovery, plus restore drills. The tables are the whole system of record.
- Erasure: `delete_document` removes rows and cascaded chunks in one transaction, but dead tuples, WAL and backups keep the content until they are reclaimed or expire. For an erasure request, run `VACUUM (INDEX_CLEANUP ON) chunks, documents` after the delete, and make backup retention fit the erasure SLA.
- Row-level security is not enabled. It is a backstop for a forgotten `WHERE` clause and would need a role that can see across tenants for migrations and statistics. Benchmark before enabling it.
- The integration suite (`tests/integration`) has not yet passed end to end. CI's `integration` job, with a pgvector service, is the first full run. See ADR 0007.

**Redis.** Treat Redis like the document store: cached answers contain document text and ACL tokens, and anyone who can write to Redis can plant answers. Use `rediss://`, authentication, and a private network. Entries are JSON, never pickle. Each key expires, and each tenant's keys are tracked for purge. If Redis is unreachable during a request, the call fails (500, fail closed); the limiters do not fall back to memory. `connect_redis` pings at startup, so a bad URL stops the process.

| Component | SQLite deployment | Postgres (+ Redis) deployment |
|---|---|---|
| API process | `fastrag serve`, one worker. | One process per instance behind a gateway. Do not run several workers in one instance. |
| Documents and indexes | SQLite WAL and in-memory indexes, rebuilt at startup (linear in the corpus). | Postgres tables. Dense search is HNSW with iterative scans; keyword search is Postgres full-text over fastrag's own terms. No rebuild. |
| ACL filtering | Python mask over the tenant partition, per query (`storage/vector.py`). | `WHERE tenant_id = ... AND acl && ...` in SQL (ADR 0007). |
| Answer cache | Per process. | Redis, or per process (see `docs/architecture.md`, "What is per process"). |
| Rate limits and auth-failure throttle | Per process. | Redis, or per process. Per process, N replicas allow N times the rate. |
| API keys | One file, read by every process that has it. | Shared storage for the keys file, or distribute it on change. |
| Audit log | JSONL, one file per host. | Same. Ship the lines to a log pipeline or WORM store. Anchor each host's head hash out of band. |
| Ingestion | Inline in the request. Large batches tie up request workers. | Same. Run batch ingest with the CLI (mind the per-user burst), or add a queue (not implemented). |

Related: [architecture.md](architecture.md), [security.md](security.md), [adr/0005-audit-log.md](adr/0005-audit-log.md), [adr/0006-offline-first-defaults.md](adr/0006-offline-first-defaults.md), [adr/0007-postgres-storage.md](adr/0007-postgres-storage.md).
