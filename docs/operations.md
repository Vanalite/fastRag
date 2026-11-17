# Operating fastRag

This guide is for the person who runs fastRag for a client. The design reasons are in `design-log.md`.

## Before going live

- [ ] Set `FASTRAG_ENV=prod`. The API refuses to start in `prod` with the default admin token or the test-only hash embedder.
- [ ] Set `FASTRAG_ADMIN_TOKEN` (generate it with `openssl rand -hex 32`) and `POSTGRES_PASSWORD`.
- [ ] Bind the API to a private address only (`FASTRAG_API_BIND`). fastRag has no public-internet hardening such as rate limits or TLS, by design (D9). Put it next to the agent backend on the private network.
- [ ] Give the customer agent a key limited to `public` and the `customer` profile: `fastrag create-key <tenant> --name customer-agent --group public --profile customer`. Never give it a staff or admin key.
- [ ] Confirm the column mapping of every listings sheet (`GET /v1/admin/mappings`). Until it is confirmed, customers see no listing fields.
- [ ] Empty the review queue (`GET /v1/admin/review/chunks`).
- [ ] Run `fastrag eval <tenant> references.jsonl`. It must exit 0. Keep the report with the release.
- [ ] Choose the reranker. The bundled CPU reranker is too slow for the 300 ms customer target, so use a hosted one (`cohere` or `jina`) or a GPU.
- [ ] Set up backups and alerts (below).

## Backups and restore

All state is in Postgres plus the raw files volume (`files`, mounted at `/data/files`).

```bash
# nightly
docker compose exec -T db pg_dump -U fastrag -Fc fastrag > fastrag-$(date +%F).dump
docker run --rm -v fastrag_files:/f -v "$PWD":/b alpine tar czf /b/files-$(date +%F).tgz -C /f .

# restore
docker compose exec -T db pg_restore -U fastrag -d fastrag --clean --if-exists < fastrag-2026-11-07.dump
```

Raw files are content-addressed, so restoring a files backup that is older than the database only loses files uploaded since that backup. Re-upload them, or re-run the source sync.

## Upgrades

```bash
git pull && docker compose build && docker compose up -d
```

The `migrate` service applies schema migrations before the API and worker start. Take a database backup first.

## Changing models

- **Reranker:** change `FASTRAG_RERANKER_*` and restart the API. No re-indexing is needed. Re-run `fastrag eval` and recheck the profile thresholds (`min_score`, `relevance_threshold`), because each reranker has its own score scale.
- **Embedding model:** the vector size is fixed in the schema, and every chunk must be re-embedded.
  - With the same dimension: change the model, then call `POST /v1/admin/collections/<slug>/reindex` for each collection.
  - With a different dimension: start a new database with the new `FASTRAG_EMBEDDING_DIM`, migrate, and re-sync.
- **Ingestion settings (parser, chunking, context LLM):** these are per collection (D25).
  - Changing a collection's settings re-processes only the documents built with other settings. Changing a deployment default (`FASTRAG_CHUNK_*`, `FASTRAG_PARSER`, the default LLM model) makes documents stale in every collection that doesn't override it, but nothing is re-processed automatically.
  - `fastrag ingest-settings <tenant> <collection>` shows the stale count. Then call `POST /v1/admin/collections/<slug>/reindex?only_stale=true`.
  - Measure first with `fastrag compare`, and when LLM summaries are on, estimate the cost with `fastrag import --dry-run`.

## Keys

- List keys: `GET /v1/admin/keys` (this shows `last_used_at`).
- Revoke a key: `DELETE /v1/admin/keys/<id>`. It takes effect immediately on the API process that handled the request, and within 30 seconds on other API processes.
- Rotate a key: create the new key, deploy it to the agent, then revoke the old one.

## Monitoring

The API exposes Prometheus metrics at `/metrics`. The worker exposes them on port 9100 (`--metrics-port`). Suggested alerts:

| Alert | Expression (PromQL sketch) | Meaning |
|---|---|---|
| Customer search slow | `histogram_quantile(0.95, rate(fastrag_http_request_seconds_bucket{route="/v1/search"}[5m])) > 1` | Check the embedding and reranker latency in `fastrag_retrieval_stage_seconds`. |
| Reranker failing | `rate(fastrag_reranker_total{state="unavailable"}[10m]) > 0` | Results fall back to fused ranking, so quality drops. |
| Searches degraded | `rate(fastrag_degraded_total[10m]) > 0` | Keyword search, vector search or the reranker is failing. |
| Queue backlog | `fastrag_jobs_queued > 100` for 30 min | The worker is down or too slow. Add worker concurrency or replicas. |
| Failed jobs | `fastrag_jobs_failed > 0` | Look in `GET /v1/admin/jobs?status=failed`, fix the cause, then `POST /v1/admin/jobs/<id>/retry`. |
| Review needed | `fastrag_documents_needing_review > 0` for 1 day | Someone has to clear or fix flagged content. |
| Many "nothing relevant" answers | rising `fastrag_retrieval_total{outcome="nothing_relevant"}` share | Content gaps. Look at the search logs with thumbs-down feedback. |

Logs are JSON lines on stdout, and every line from a request carries its `trace_id`. The agent backend should send its own `X-Trace-Id`, so one conversation turn can be followed across both systems.

## Runbook

- **A document is stuck in `processing` or `failed`:** read `GET /v1/admin/documents/<id>`, which includes the error and the ingestion report. Stuck jobs are re-queued after 30 minutes. Re-run a document with `POST /v1/admin/documents/<id>/reindex`.
- **A spreadsheet row is missing or a value is wrong:** the snapshot report (`documents/<id>` → `report.sheets`) lists the cells that could not be read. Those values are kept as text attributes instead of typed fields.
- **Customers saw something they shouldn't have:**
  1. Revoke the key if it was the wrong key.
  2. Check the folder's groups (`folder_groups` on the source) and the sheet's column mapping.
  3. Fix them, and the sync re-applies the access groups.
  4. Add the case to the reference set: access safety is release check 1.
- **Personal-data deletion request:** delete the document (`DELETE /v1/admin/documents/<id>`); this removes its chunks. Remove the file from the source so the next sync does not bring it back. Search logs are purged after `FASTRAG_SEARCH_LOG_RETENTION_DAYS` (90 by default).

## Scaling

- **API:** it is stateless, so add replicas or raise `FASTRAG_API_WORKERS`. Each process holds up to `db_pool_size + db_max_overflow` database connections.
- **Worker:** raise `FASTRAG_WORKER_CONCURRENCY` or run more worker containers. They coordinate through the job table.
- **Postgres:** the vector index is HNSW (m 16, ef_construction 64), and searches use iterative scans so that filtered queries still return enough rows. `scripts/bench_search.py` measures database-side latency at a given corpus size.
