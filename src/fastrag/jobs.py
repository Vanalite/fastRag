"""Asynchronous ingestion jobs.

Bulk uploads shouldn't hold an HTTP request open while hundreds of documents are
parsed, redacted and embedded. A job accepts the batch, returns at once with an id, and
worker threads ingest it in the background; clients poll the job for progress.

Each document still goes through ``RAGService`` as the submitter: permissions, editors,
redaction and the audit log apply exactly as for a single upload. Submitting a job is
rate limited; the documents inside it are not (the job is the unit of admission), but
each tenant may only have a bounded number of active jobs.

Workers run in the API process. If a replica stops, its unfinished jobs are marked
``cancelled`` (best effort) and must be resubmitted. For guaranteed delivery across
restarts, put an external queue (SQS, RabbitMQ, a Postgres-backed queue) behind the same
``submit`` / ``get`` interface. Job status can be kept in Redis so any replica can
answer a status poll.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Protocol

from fastrag.domain import DocumentIn, IngestResult, Principal, Role
from fastrag.errors import FastRagError, InvalidInput, NotFound, RateLimited
from fastrag.observability.logging import request_id_var

log = logging.getLogger(__name__)

JobStatus = Literal["queued", "running", "succeeded", "partial", "failed", "cancelled"]
FINISHED: frozenset[str] = frozenset({"succeeded", "partial", "failed", "cancelled"})
# A job whose record hasn't been touched for this long has lost its worker (the process
# died): it no longer counts against the tenant's cap and is reported as failed.
STALE_AFTER_SECONDS = 900.0


def is_active(job: IngestJob, now: float) -> bool:
    return job.status not in FINISHED and now - job.updated_at < STALE_AFTER_SECONDS


@dataclass
class JobItem:
    source_id: str
    status: str = "pending"  # pending | created | updated | unchanged | failed | cancelled
    doc_id: str | None = None
    version: int | None = None
    error: str | None = None


@dataclass
class IngestJob:
    id: str
    tenant_id: str
    user_id: str
    status: JobStatus
    items: list[JobItem]
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None
    request_id: str | None = None
    counts: dict[str, int] = field(default_factory=dict)
    updated_at: float = 0.0  # heartbeat: refreshed on every save by the worker

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, raw: str | bytes) -> IngestJob:
        data = json.loads(raw)
        data["items"] = [JobItem(**item) for item in data["items"]]
        return cls(**data)

    def recount(self) -> None:
        counts: dict[str, int] = {}
        for item in self.items:
            counts[item.status] = counts.get(item.status, 0) + 1
        self.counts = counts


class JobStore(Protocol):
    def save(self, job: IngestJob) -> None: ...

    def get(self, job_id: str) -> IngestJob | None: ...

    def active_count(self, tenant_id: str) -> int: ...


class MemoryJobStore:
    """In-process job status. Finished jobs are kept for ``retention_seconds``."""

    def __init__(
        self,
        retention_seconds: float = 86_400,
        max_jobs: int = 10_000,
        clock: Callable[[], float] = time.time,
    ):
        self._jobs: OrderedDict[str, IngestJob] = OrderedDict()
        self._retention = retention_seconds
        self._max_jobs = max_jobs
        self._clock = clock
        self._lock = threading.Lock()

    def save(self, job: IngestJob) -> None:
        snapshot = _copy(job)  # callers keep mutating theirs
        with self._lock:
            self._jobs[job.id] = snapshot
            self._jobs.move_to_end(job.id)
            self._evict()

    def get(self, job_id: str) -> IngestJob | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return _copy(job) if job else None

    def active_count(self, tenant_id: str) -> int:
        now = self._clock()
        with self._lock:
            return sum(
                1 for j in self._jobs.values() if j.tenant_id == tenant_id and is_active(j, now)
            )

    def _evict(self) -> None:
        cutoff = self._clock() - self._retention
        for jid in [
            j for j, job in self._jobs.items() if job.finished_at and job.finished_at < cutoff
        ]:
            del self._jobs[jid]
        # Over the cap, drop the oldest *finished* jobs only: never an active one.
        for jid in [j for j, job in self._jobs.items() if job.status in FINISHED]:
            if len(self._jobs) <= self._max_jobs:
                break
            del self._jobs[jid]


class RedisJobStore:
    """Job status in Redis, so a status poll can land on any replica.

    Active jobs are tracked in a per-tenant sorted set scored by their last heartbeat, so
    a job abandoned by a crashed replica ages out of the tenant's cap on its own.
    """

    def __init__(
        self,
        client: Any,
        retention_seconds: int = 86_400,
        prefix: str = "fastrag:job:",
        clock: Callable[[], float] = time.time,
    ):
        self._redis = client
        self._ttl = retention_seconds
        self._prefix = prefix
        self._clock = clock

    def _active_key(self, tenant_id: str) -> str:
        return f"{self._prefix}active:{tenant_id}"

    def save(self, job: IngestJob) -> None:
        active = self._active_key(job.tenant_id)
        pipe = self._redis.pipeline(transaction=True)
        pipe.set(self._prefix + job.id, job.to_json(), ex=self._ttl)
        if job.status in FINISHED:
            pipe.zrem(active, job.id)
        else:
            pipe.zadd(active, {job.id: job.updated_at})
            pipe.expire(active, self._ttl)
        pipe.execute()

    def get(self, job_id: str) -> IngestJob | None:
        raw = self._redis.get(self._prefix + job_id)
        return IngestJob.from_json(raw) if raw is not None else None

    def active_count(self, tenant_id: str) -> int:
        active = self._active_key(tenant_id)
        self._redis.zremrangebyscore(active, "-inf", self._clock() - STALE_AFTER_SECONDS)
        return int(self._redis.zcard(active))


def _copy(job: IngestJob) -> IngestJob:
    return IngestJob.from_json(job.to_json())


IngestFn = Callable[[Principal, DocumentIn, str], IngestResult]


class IngestJobRunner:
    """Runs jobs on a small worker pool, with admission control.

    Admission: per-tenant active-job cap, plus global caps on queued jobs and queued
    characters, so the documents held in memory while waiting stay bounded. Inside a job,
    ``pace(tenant_id)`` (a blocking per-tenant rate limit) spreads the embedding work.
    """

    def __init__(
        self,
        ingest: IngestFn,
        store: JobStore | None = None,
        *,
        workers: int = 2,
        max_active_per_tenant: int = 10,
        max_documents: int = 500,
        max_queued_jobs: int = 100,
        max_queued_chars: int = 100_000_000,
        pace: Callable[[str, threading.Event], None] | None = None,
        clock: Callable[[], float] = time.time,
    ):
        self._ingest = ingest
        self.store = store if store is not None else MemoryJobStore()
        self.max_active_per_tenant = max_active_per_tenant
        self.max_documents = max_documents
        self.max_queued_jobs = max_queued_jobs
        self.max_queued_chars = max_queued_chars
        self._pace = pace
        self._clock = clock
        self._executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ingest-job")
        self._pending: dict[str, int] = {}  # job id -> characters held for it
        self._lock = threading.Lock()
        self._stop = threading.Event()

    def submit(self, principal: Principal, documents: list[DocumentIn]) -> IngestJob:
        if not documents:
            raise InvalidInput("a job needs at least one document")
        if len(documents) > self.max_documents:
            raise InvalidInput(f"a job may contain at most {self.max_documents} documents")
        chars = sum(len(d.text) for d in documents)
        with self._lock:
            if self._stop.is_set():
                raise RateLimited(retry_after=5.0)  # shutting down: try another replica
            queued_chars = sum(self._pending.values())
            if (
                len(self._pending) >= self.max_queued_jobs
                or queued_chars + chars > self.max_queued_chars
            ):
                raise RateLimited(retry_after=10.0)  # this replica's queue is full
            if self.store.active_count(principal.tenant_id) >= self.max_active_per_tenant:
                raise RateLimited(retry_after=10.0)
            job = IngestJob(
                id=uuid.uuid4().hex,
                tenant_id=principal.tenant_id,
                user_id=principal.user_id,
                status="queued",
                items=[JobItem(source_id=d.source_id) for d in documents],
                created_at=self._clock(),
                request_id=request_id_var.get(),
            )
            job.recount()
            self._save(job)
            self._pending[job.id] = chars
            self._executor.submit(self._run, job, principal, list(documents))
        return _copy(job)

    def get(self, principal: Principal, job_id: str) -> IngestJob:
        job = self.store.get(job_id)
        visible = job is not None and job.tenant_id == principal.tenant_id
        if visible and Role.ADMIN not in principal.roles:
            visible = job.user_id == principal.user_id
        if not visible:
            raise NotFound(f"job {job_id!r} not found")  # same answer for foreign jobs
        if job.status not in FINISHED and not is_active(job, self._clock()):
            # No heartbeat for a long time: the process running it is gone.
            self._finish(job, "failed", "worker stopped; resubmit the document")
        return job

    def shutdown(self, timeout: float = 30.0) -> None:
        """Stop accepting jobs, cancel queued ones, and let running ones stop cleanly
        after their current document (marked ``cancelled``, with what was done kept)."""
        self._stop.set()
        self._executor.shutdown(wait=False, cancel_futures=True)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                pending = dict(self._pending)
            if not pending:
                return
            for job_id in pending:
                job = self.store.get(job_id)
                if job is not None and job.status == "queued":
                    self._finish(job, "cancelled", "server shutting down; resubmit")
                    with self._lock:
                        self._pending.pop(job_id, None)
            time.sleep(0.01)

    def _run(self, job: IngestJob, principal: Principal, documents: list[DocumentIn]) -> None:
        with self._lock:
            if job.id not in self._pending or self._stop.is_set():
                return  # cancelled (by shutdown) before it started
        token = request_id_var.set(job.request_id)  # ties each document's audit to the job
        try:
            job.status, job.started_at = "running", self._clock()
            self._save(job)
            for item, doc in zip(job.items, documents, strict=True):
                if self._stop.is_set():
                    self._finish(job, "cancelled", "server shutting down; resubmit")
                    return
                try:
                    if self._pace is not None:
                        self._pace(job.tenant_id, self._stop)
                    result = self._ingest(principal, doc, job.id)
                    item.status, item.doc_id, item.version = (
                        result.status,
                        result.doc_id,
                        result.version,
                    )
                except FastRagError as exc:
                    item.status, item.error = "failed", f"{type(exc).__name__}: {exc}"
                except Exception:  # noqa: BLE001 - one bad document must not sink the job
                    log.exception("ingest job item failed", extra={"job_id": job.id})
                    item.status, item.error = "failed", "internal error"
                job.recount()
                self._save(job)  # progress is visible while the job runs (and a heartbeat)
            failed = job.counts.get("failed", 0)
            status = (
                "succeeded" if failed == 0 else "failed" if failed == len(job.items) else "partial"
            )
            self._finish(job, status)
        except Exception:  # noqa: BLE001
            log.exception("ingest job crashed", extra={"job_id": job.id})
            self._finish(job, "failed", "internal error")
        finally:
            with self._lock:
                self._pending.pop(job.id, None)
            request_id_var.reset(token)

    def _finish(self, job: IngestJob, status: JobStatus, pending_error: str | None = None) -> None:
        job.status, job.finished_at = status, self._clock()
        for item in job.items:
            if item.status == "pending":
                item.status, item.error = (
                    ("cancelled" if status == "cancelled" else "failed"),
                    pending_error,
                )
        job.recount()
        self._save(job)

    def _save(self, job: IngestJob) -> None:
        job.updated_at = self._clock()
        self.store.save(job)

    def wait(self, job_id: str, timeout: float = 30.0) -> None:
        """Block until a job finishes (for tests and the CLI)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            job = self.store.get(job_id)
            if job is not None and job.status in FINISHED:
                return
            time.sleep(0.01)
        raise TimeoutError(f"job {job_id} did not finish within {timeout}s")
