"""Prometheus metrics. The API serves them at /metrics; the worker on
--metrics-port. Keep both on the private network (D9)."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

LATENCY_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 2.0, 5.0, 10.0)

HTTP_REQUESTS = Counter("fastrag_http_requests_total", "HTTP requests", ["method", "route", "status"])
HTTP_LATENCY = Histogram("fastrag_http_request_seconds", "HTTP request latency", ["route"], buckets=LATENCY_BUCKETS)
RETRIEVAL_STAGE = Histogram(
    "fastrag_retrieval_stage_seconds", "Time per retrieval stage", ["stage"], buckets=LATENCY_BUCKETS
)
RETRIEVAL_RESULTS = Counter("fastrag_retrieval_total", "Retrieval calls by outcome", ["endpoint", "profile", "outcome"])
RERANKER_STATE = Counter("fastrag_reranker_total", "Reranker outcome per search", ["state"])
DEGRADED = Counter("fastrag_degraded_total", "Searches served in a degraded mode", ["reason"])
JOBS = Counter("fastrag_jobs_total", "Worker jobs by result", ["kind", "result"])
JOB_SECONDS = Histogram(
    "fastrag_job_seconds", "Worker job duration", ["kind"], buckets=(0.1, 0.5, 1, 5, 15, 60, 300, 1800)
)
QUEUE_DEPTH = Gauge("fastrag_jobs_queued", "Jobs waiting in the queue")
JOBS_FAILED = Gauge("fastrag_jobs_failed", "Jobs that exhausted their retries")
DOCS_NEED_REVIEW = Gauge("fastrag_documents_needing_review", "Documents with status needs_review or failed")


def observe_search(endpoint: str, profile: str, signals: dict[str, object], timings_ms: dict[str, float]) -> None:
    for stage, ms in timings_ms.items():
        RETRIEVAL_STAGE.labels(stage.removesuffix("_ms")).observe(ms / 1000)
    outcome = "nothing_relevant" if signals.get("nothing_relevant") else "answered"
    RETRIEVAL_RESULTS.labels(endpoint, profile, outcome).inc()
    RERANKER_STATE.labels(str(signals.get("reranker"))).inc()
    for reason in signals.get("degraded") or []:  # type: ignore[attr-defined]
        DEGRADED.labels(str(reason)).inc()
