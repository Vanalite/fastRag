# fastRag

A reference implementation of an **enterprise-grade RAG system**: one that answers questions from your organization's documents while respecting who is allowed to see what, keeps an audit trail, and proves its own properties with tests.

Most RAG demos stop at "embed, search, prompt". The hard parts in an enterprise are everything around that loop. This repo implements them in small, readable modules, and documents why each one exists.

| Concern | What this repo does | Where |
| --- | --- | --- |
| Identity | API keys (stored hashed) and JWT bearer tokens resolve to a `Principal` (tenant, user, groups, roles) | `security/auth.py` |
| Permissions | Roles (reader, ingester, admin) decide what a caller may *do* | `security/rbac.py` |
| Document access | Every chunk carries its tenant and ACL. Every index search *requires* an access filter and applies it before ranking | `security/acl.py`, `storage/` |
| Consistency | SQLite is the single source of truth; vector and keyword indexes are rebuilt from it. Re-ingest creates a new version; delete removes the document from every index and the cache | `storage/`, `ingest/service.py` |
| Retrieval quality | Hybrid search: BM25 + dense vectors, fused with reciprocal rank fusion | `retrieval/` |
| Grounding | Abstains without calling the LLM when evidence is weak; validates that citations point at documents that were shown | `service.py`, `generation/citations.py` |
| Prompt injection | Documents are escaped, delimited and labeled untrusted; suspicious ones are flagged (optionally dropped) and audited | `generation/prompt.py` |
| Privacy | PII (emails, phone numbers, SSNs, card numbers) is redacted at ingest, before anything is embedded or sent to an LLM | `ingest/redact.py` |
| Governance | Append-only, hash-chained audit log of who asked what and which documents they were shown (ids and hashes only, never content) | `observability/audit.py` |
| Operations | Per-tenant rate limits, an access-scoped answer cache, Prometheus metrics, JSON logs with request ids, health and readiness probes | `ratelimit.py`, `cache.py`, `observability/` |
| Evaluation | A golden dataset and a retrieval-quality gate in CI (recall, MRR, nDCG, abstention accuracy, zero ACL leaks) | `eval/`, `src/fastrag/eval/` |

Everything runs offline with zero infrastructure: a hashing embedder, an extractive answer mode and in-process stores. Answers from Claude, semantic embeddings and external stores plug in behind interfaces.

## Quick start

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Load the demo corpus: ten internal documents of a fictional company, Northwind Labs. Most are readable by everyone in the tenant; the Q3 board summary is restricted to the `finance` group and the compensation bands to `hr`:

```bash
uv run fastrag seed-demo
```

Ask as different people and see access control at work:

```bash
uv run fastrag ask "What was Q3 revenue?" --tenant northwind --user bob --group sales
```

```bash
uv run fastrag ask "What was Q3 revenue?" --tenant northwind --user carol --group finance
```

Bob gets "I couldn't find information about that in the documents available to you", and the LLM is never called. Carol gets the answer with a citation. Both requests are in the audit log (`uv run fastrag audit verify` checks its hash chain).

To index your own files: `uv run fastrag ingest ./docs --tenant acme --acl group:engineering` (repeat `--acl` to allow several groups; the default is everyone in the tenant).

### Answers from Claude

The default answer mode quotes the best passages. For written answers with inline `[n]` citations, set `ANTHROPIC_API_KEY` and:

```bash
FASTRAG_GENERATOR=claude uv run fastrag ask "How do I file a travel expense?" --tenant northwind --user bob
```

This uses `claude-opus-5-5`. If Claude declines a request on safety grounds, the API retries it on a fallback model automatically.

### HTTP API

```bash
uv run fastrag keys create --tenant northwind --user carol --group finance
```

```bash
uv run fastrag serve --port 8000
```

```bash
curl -s localhost:8000/v1/ask -H "X-API-Key: <key>" -H "content-type: application/json" -d '{"question": "What was Q3 revenue?"}'
```

| Method | Path | Who | Purpose |
| --- | --- | --- | --- |
| POST | `/v1/ask` | reader+ | Grounded answer with citations |
| POST | `/v1/search` | reader+ | Ranked chunks with dense and keyword scores |
| GET | `/v1/documents` | reader+ | Documents visible to the caller |
| POST | `/v1/documents` | ingester+ | Create or replace a document (with its ACL) |
| DELETE | `/v1/documents/{doc_id}` | ingester+ | Delete a document everywhere |
| GET | `/v1/audit` | admin | The tenant's audit trail |
| GET | `/v1/me` | any | Who the credentials resolve to |
| GET | `/healthz`, `/readyz`, `/metrics` | none | Probes and Prometheus metrics |

Interactive API docs are at `/docs`. JWTs work too: set `FASTRAG_JWT_SECRET` and send `Authorization: Bearer <token>` (`fastrag token ...` mints one for testing).

### Docker

```bash
docker compose up --build
```

The image runs as a non-root user with a read-only filesystem; state lives on the `fastrag-data` volume.

## Tests and quality gates

```bash
uv run pytest
```

```bash
uv run fastrag eval
```

- `tests/unit/`: each component on its own.
- `tests/security/`: acceptance tests that try to break the enterprise properties: cross-tenant leaks, ACL bypass through search, cache or audit, deleted documents coming back after a restart, prompt injection, forged or expired tokens, and rate-limit fairness between tenants.
- `fastrag eval`: runs the golden question set and fails if retrieval quality drops below the recorded baseline or any answer leaks a document the asker can't see. With the default hashing embedder the scores are a regression baseline, not a quality claim.

## Learn more

Start with [docs/README.md](docs/README.md) for a reading order. Key documents:

- [Architecture](docs/architecture.md): components, request lifecycle and data model
- [Security](docs/security.md): threat model and production hardening checklist
- [Architecture decisions](docs/adr/): why each design choice was made, and the alternatives
- [Operations](docs/operations.md): every setting, endpoint and operational task

## What is deliberately not here

These are documented extension points, not implemented: OIDC/JWKS token verification, Postgres/pgvector storage, a cross-encoder reranker, an asynchronous ingestion queue, streaming answers, distributed rate limiting and OpenTelemetry tracing. [docs/architecture.md](docs/architecture.md) explains where each would plug in.

## Development

```bash
uv run ruff check . && uv run ruff format --check .
```

## License

Apache 2.0. See [LICENSE](LICENSE).
