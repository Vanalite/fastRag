# Operations

**Last updated:** 2026-11-03
**Entry points:** `src/fastrag/config.py` (settings), `src/fastrag/api/app.py` (HTTP), `src/fastrag/cli.py` (operator commands)

This page describes the code as it is after the pending changes listed in `docs/architecture.md` land. Where a default changed, the table says so.

## Configuration reference

All settings come from environment variables named `FASTRAG_<FIELD>` (`config.py:13`), or from a `.env` file in the working directory. Environment variables win over the file. The `ANTHROPIC_API_KEY` variable is read by the Anthropic SDK, not by fastRag.

### Storage and retrieval

| Variable | Default | Meaning |
|---|---|---|
| `FASTRAG_DATABASE_PATH` | `.fastrag/fastrag.db` | SQLite file. `:memory:` gives an ephemeral store. |
| `FASTRAG_EMBEDDER` | `hashing` | `hashing` or `hashing:<dim>` (dim 1024 by default), or `st:<model>` with the `st` extra. The store records the name and refuses to start with a different one. |
| `FASTRAG_CHUNK_SIZE` | `800` | Characters per chunk, 100 to 8000. |
| `FASTRAG_CHUNK_OVERLAP` | `100` | Characters shared by consecutive chunks. Must be smaller than the chunk size. |
| `FASTRAG_MAX_DOCUMENT_CHARS` | `2000000` | Largest accepted document. The API schema also caps text at 2,000,000 characters. |
| `FASTRAG_REDACT_PII` | `true` | Regex redaction of email, SSN, card numbers and phone numbers at ingest. |
| `FASTRAG_TOP_K` | `4` | Chunks returned and shown, 1 to 20. |
| `FASTRAG_CANDIDATE_POOL` | `20` | Candidates fetched from each index before fusion, 1 to 200. |
| `FASTRAG_RRF_K` | `60` | Reciprocal-rank-fusion constant. |
| `FASTRAG_MIN_DENSE_SCORE` | unset (`None`) | Dense cosine threshold for confidence. Unset uses the embedder's own default: 0.25 for hashing, 0.35 for sentence-transformers. |
| `FASTRAG_MIN_KEYWORD_COVERAGE` | `0.3` | IDF-weighted share of the query's terms a chunk must contain for the keyword signal to count (rare, informative terms weigh more than common ones). Unless every query term matched, at least two must. Calibrated on `eval/golden`: 0.4 removes one false answer but adds two false abstentions. |

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
| `FASTRAG_API_KEYS_FILE` | `.fastrag/api_keys.json` | Hashed API keys. The app needs at least one authentication method (keys or JWT), or it refuses to start. |
| `FASTRAG_JWT_SECRET` | unset | HS256 shared secret, at least 32 characters. Enables bearer tokens. |
| `FASTRAG_JWT_ISSUER` | unset | Required `iss` value, checked when set. |
| `FASTRAG_JWT_AUDIENCE` | unset | Required `aud` value. Checked only when set. |

### Operations

| Variable | Default | Meaning |
|---|---|---|
| `FASTRAG_AUDIT_LOG_PATH` | `.fastrag/audit.jsonl` | Audit file. Unset keeps entries in memory (100,000 maximum). |
| `FASTRAG_AUDIT_STORE_QUERY_TEXT` | `false` | Store question text in audit entries. The default stores only a SHA-256. |
| `FASTRAG_CACHE_TTL_SECONDS` | `300` | Answer cache lifetime. `0` disables the cache. |
| `FASTRAG_CACHE_MAX_ENTRIES` | `1000` | Cache size limit. `0` disables the cache. |
| `FASTRAG_RATE_LIMIT_PER_MINUTE` | `120` | Token-bucket refill rate per tenant, for queries only. `0` disables. |
| `FASTRAG_RATE_LIMIT_BURST` | `30` | Bucket size per tenant. `0` disables. |
| `FASTRAG_LOG_LEVEL` | `INFO` | Root log level. |
| `FASTRAG_LOG_JSON` | `true` | JSON logs to stderr, with `request_id` on each line. |
| `FASTRAG_EXPOSE_METRICS` | `false` (changed) | Serves `/metrics`. Off by default because the endpoint is unauthenticated and its labels include tenant ids. Enable it only on an internal network. |

## Endpoints

Authentication is an `Authorization: Bearer <jwt>` header or an `X-API-Key: <key>` header. The OpenAPI document at `/docs` declares both schemes. Every response carries `X-Request-ID` (echoed from the request if it matches `[A-Za-z0-9._-]{1,128}`, otherwise generated) and `Cache-Control: no-store`.

| Method and path | Auth | Permission | Success | Notes |
|---|---|---|---|---|
| `GET /healthz` | no | none | 200 | Liveness. Returns `{"status": "ok"}` whenever the process answers. |
| `GET /readyz` | no | none | 200 or 503 | Runs `store.ping()` only. |
| `GET /metrics` | no | none | 200 | Only when `FASTRAG_EXPOSE_METRICS=true`. Prometheus text format. |
| `GET /v1/me` | yes | none | 200 | Returns the caller's tenant, user, groups and roles. |
| `POST /v1/documents` | yes | `ingest` (ingester, admin) | 201 | Body: `source_id`, `text`, `acl` (1 to 256 tokens), optional `title`, `media_type`, `metadata`. Re-ingesting a `source_id` makes a new version. |
| `GET /v1/documents` | yes | `query` | 200 | Lists documents. Admins see every document in the tenant, and everyone else sees documents their ACL allows. Metadata only. |
| `DELETE /v1/documents/{doc_id}` | yes | `delete` (ingester, admin) | 204 | 404 if the document is absent or not visible to the caller. |
| `POST /v1/search` | yes | `query` | 200 | Body: `question`, optional `k`. Returns chunks and the `confident` flag. Does not flag or drop suspicious chunks. |
| `POST /v1/ask` | yes | `query` | 200 | Body: `question`, optional `k`. Returns the answer, citations, shown sources and token usage. |
| `GET /v1/audit?limit=N` | yes | `read_audit` (admin) | 200 | Entries for the caller's tenant only, newest last, up to 1000. |

Error responses use one envelope: `{"error": {"type": "...", "message": "..."}}`, and include the request id. The mapping, in `src/fastrag/api/app.py`:

| Exception | Status | Notes |
|---|---|---|
| `AuthenticationError` | 401 | Generic message, `WWW-Authenticate: Bearer`. |
| `PermissionDenied` | 403 | |
| `NotFound` | 404 | |
| `InvalidInput`, request validation | 422 | Submitted input is not echoed back. |
| `RateLimited` | 429 | `Retry-After` in seconds. |
| `GenerationError` | 502 | LLM failed or refused. |
| Anything else | 500 | Generic message. The exception is logged with the request id. |

Requests that fail authentication are audited with `tenant_id: null`, and count toward `fastrag_requests_total{outcome="unauthenticated"}`.

## Health, readiness and metrics

- **Liveness:** `/healthz` only shows that the process answers HTTP. Use it for restarts.
- **Readiness:** `/readyz` pings the SQLite store. It does not check that the indexes match the store, that the embedder loaded, or that the LLM is reachable. Add those checks before relying on it for routing.
- **Metrics** (`src/fastrag/observability/metrics.py`):

| Metric | Labels | Meaning |
|---|---|---|
| `fastrag_requests_total` | `action`, `outcome`, `tenant` | One per service call. Outcomes: `success`, `abstained` (via the `outcome` detail), `denied`, `rate_limited`, `not_found`, `invalid`, `generation_error`, `error`, `unauthenticated`. |
| `fastrag_stage_seconds` | `stage` | Histogram for `retrieval`, `generation` and each action. |
| `fastrag_llm_tokens_total` | `tenant`, `kind` | Input, output and cache tokens from generation. |
| `fastrag_injection_flags_total` | `tenant` | Chunks flagged by the injection heuristic. |
| `fastrag_invalid_citations_total` | `tenant` | Citation markers that pointed at unshown sources. |
| `fastrag_pii_redactions_total` | `kind` | Redactions at ingest, by kind. |

Keep label cardinality low. Tenant ids are acceptable. Never add user ids or query text.

## Audit log

Each line of `.fastrag/audit.jsonl` is one entry: `ts`, `request_id`, `action`, `outcome`, `tenant_id`, `user_id`, `details`, `prev_hash`, `hash` (`src/fastrag/observability/audit.py`).

**Verify:**

```sh
fastrag audit verify        # uses FASTRAG_AUDIT_LOG_PATH
```

It prints `N entries; chain OK` or `chain BROKEN`, and exits 1 when broken. It runs the chain check only. It cannot detect a truncated tail or a forged file, as `docs/security.md` explains. To detect those, record the last hash out of band (see the scaling notes).

**Failure behavior.** An audit write that fails is logged and never replaces the original exception. If the call itself succeeded, the request fails, so a successful answer is never returned without its audit record (fail closed).

## Keys and tokens

**API keys** (`fastrag keys create`) are the default for services:

```sh
fastrag keys create --tenant northwind --user svc-ingest --role ingester --group engineering
```

The key is printed once to stdout. Only its SHA-256 goes into `FASTRAG_API_KEYS_FILE` (mode 0600). Options: `--tenant` (required), `--user` (default `cli-operator`), `--group` (repeatable), `--role` (repeatable: `reader`, `ingester` or `admin`; default `reader`).

- **Revoking a key:** there is no command. Remove its record from the JSON file and restart the server. The server loads keys once at startup.
- **New keys:** a running server does not see keys created after it started. Restart it.
- **Keys cannot expire.** Rotate by creating a new key, updating the caller, then removing the old record.

**JWT tokens** (`fastrag token`) are for local demos. They need `FASTRAG_JWT_SECRET`:

```sh
fastrag token --tenant northwind --user alice --group finance --role reader --ttl 3600
```

Production tokens come from the identity provider. Until JWKS verification exists, the secret is shared with every verifier, and changing it invalidates all tokens. Keep `exp` short, and set `FASTRAG_JWT_ISSUER` and `FASTRAG_JWT_AUDIENCE`.

**CLI commands bypass HTTP authentication.** `ingest` and `delete` run as an admin operator. `ask`, `search` and `documents` run as the principal you name. Only run the CLI where you could read the data files directly. The `ingest` default ACL is `tenant:*`, which makes documents visible to every user in the tenant. Pass `--acl` for anything narrower.

## Scaling

The reference is one process, one SQLite file, and in-memory indexes. Each piece scales differently.

| Component | Today | How to scale |
|---|---|---|
| API process | `uvicorn.run` with one worker (`cli.py:110`). | Run one process per instance behind a gateway. **Do not use multiple workers in one instance.** Each worker would load its own indexes and keep its own audit chain. |
| Document store | SQLite WAL, one writer. | Move to Postgres (`storage/base.py` is the interface). Until then, one instance writes. |
| Vector and keyword indexes | In memory, rebuilt from the store at startup. Dense vectors use 4 bytes per dimension: about 410 MB for 100,000 chunks at the default 1024 dimensions, before text and Python overhead. | pgvector or OpenSearch as a new adapter. Until then, size the instance for the corpus. |
| Startup | Linear in the number of chunks, because every chunk is reloaded and reindexed. | Keep the corpus per instance small, or add a persisted index. |
| ACL filtering | A Python loop over every chunk in the tenant partition, on each query (`storage/vector.py:61-65`). | Move the filter into the index engine (ADR 0002). |
| Answer cache | Per process. Replicas do not share hits. | Redis keyed the same way, or accept the lower hit rate. |
| Rate limits | Per process. With N replicas, a tenant gets up to N times the configured rate. | Enforce the limit at the gateway, or in Redis. |
| Index freshness across replicas | An ingest updates only the instance that handled it. Other instances serve stale results until they restart. | Route all writes to one instance, or move the index into a shared store. Do not run replicas with writes on each one. |
| Audit log | JSONL, one writer per file. | Give each instance its own file, and ship the lines to a log pipeline or WORM store. Anchor each chain's head hash out of band and verify each chain separately. |
| Ingestion | Inline in the request. Large batches tie up request workers. | Run batch ingest with the CLI, or add a queue (not implemented). |

**Before running more than one instance,** do the following first: move the store and index to shared storage, move the rate limit and cache to shared storage, and ship the audit log. Without those changes, the replicas disagree about what exists and how often a tenant has called the API.

Related: [architecture.md](architecture.md), [security.md](security.md), [adr/0005-audit-log.md](adr/0005-audit-log.md), [adr/0006-offline-first-defaults.md](adr/0006-offline-first-defaults.md).
