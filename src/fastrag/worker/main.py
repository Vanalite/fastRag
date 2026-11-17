"""Ingestion worker: runs queued jobs and schedules source syncs.

Runs as its own container so OCR and embedding load never slows down
retrieval (D9).
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import socket
import traceback
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from fastrag import metrics
from fastrag.config import get_settings
from fastrag.db.engine import session_scope
from fastrag.db.models import Collection, Document, SearchLog, Source
from fastrag.ingest.documents import REPROCESSABLE, delete_document, ingest_document, stale_clause
from fastrag.ingest.settings import resolve_ingest
from fastrag.ingest.sync import enqueue_processing, import_spreadsheet_document, sync_source
from fastrag.log import log_event
from fastrag.services import Services
from fastrag.worker import queue

logger = logging.getLogger(__name__)

Handler = Callable[[AsyncSession, Services, dict[str, Any]], Awaitable[dict[str, Any]]]


async def _ingest(session: AsyncSession, services: Services, p: dict[str, Any]) -> dict[str, Any]:
    return await ingest_document(session, services, uuid.UUID(p["document_id"]), bool(p.get("force")))


async def _import(session: AsyncSession, services: Services, p: dict[str, Any]) -> dict[str, Any]:
    return await import_spreadsheet_document(session, services, uuid.UUID(p["document_id"]))


async def _sync(session: AsyncSession, services: Services, p: dict[str, Any]) -> dict[str, Any]:
    return await sync_source(session, services, uuid.UUID(p["source_id"]))


async def _delete(session: AsyncSession, services: Services, p: dict[str, Any]) -> dict[str, Any]:
    await delete_document(session, uuid.UUID(p["document_id"]))
    return {"deleted": p["document_id"]}


async def _reindex(session: AsyncSession, services: Services, p: dict[str, Any]) -> dict[str, Any]:
    """Re-process the documents of a collection: all of them (e.g. after
    changing the embedding model), or with `only_stale` just the ones not built
    with the collection's current ingestion settings (D25)."""
    cid = uuid.UUID(p["collection_id"])
    collection = await session.get(Collection, cid)
    assert collection is not None
    stmt = select(Document).where(Document.collection_id == cid, Document.status != "deleted")
    if p.get("only_stale") and collection.type == "documents":
        cfg = resolve_ingest(collection.settings, services)
        stmt = select(Document).where(
            Document.collection_id == cid, Document.status.in_(REPROCESSABLE), stale_clause(cfg.fingerprint)
        )
    docs = (await session.execute(stmt)).scalars().all()
    for d in docs:
        d.status = "pending"
        await enqueue_processing(session, d, collection.type)
    return {"queued": len(docs)}


HANDLERS: dict[str, Handler] = {
    "ingest_document": _ingest,
    "import_sheet": _import,
    "sync_source": _sync,
    "delete_document": _delete,
    "reindex_collection": _reindex,
}


async def run_one(services: Services, worker_id: str) -> bool:
    """Claim and run one job. Returns False when the queue is empty."""
    async with session_scope() as session:
        job = await queue.claim(session, worker_id)
        if job is None:
            return False
        job_id, kind, payload = job.id, job.kind, dict(job.payload)
        attempts, max_attempts = job.attempts, job.max_attempts
    handler = HANDLERS.get(kind)
    started = datetime.now(UTC)
    try:
        if handler is None:
            raise ValueError(f"unknown job kind {kind!r}")
        async with session_scope() as session:
            result = await handler(session, services, payload)
        async with session_scope() as session:
            await queue.complete(session, job_id, result)
        metrics.JOBS.labels(kind, "done").inc()
        metrics.JOB_SECONDS.labels(kind).observe((datetime.now(UTC) - started).total_seconds())
        log_event(
            logger,
            "job done",
            job_id=job_id,
            kind=kind,
            ms=int((datetime.now(UTC) - started).total_seconds() * 1000),
        )
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=5)}"
        logger.error("job %s (%s) failed: %s", job_id, kind, exc)
        metrics.JOBS.labels(kind, "failed" if attempts >= max_attempts else "retry").inc()
        async with session_scope() as session:
            await queue.fail(session, job_id, attempts, max_attempts, err)
            if attempts >= max_attempts and "document_id" in payload:
                doc = await session.get(Document, uuid.UUID(payload["document_id"]))
                if doc is not None:
                    doc.status, doc.error = "failed", str(exc)[:2000]
    return True


async def schedule_due_syncs() -> int:
    now = datetime.now(UTC)
    n = 0
    async with session_scope() as session:
        due = (
            (
                await session.execute(
                    select(Source)
                    .where(Source.sync_interval_minutes.is_not(None))
                    .where((Source.next_sync_at.is_(None)) | (Source.next_sync_at <= now))
                    .with_for_update(skip_locked=True)
                )
            )
            .scalars()
            .all()
        )
        for src in due:
            await queue.enqueue(
                session,
                "sync_source",
                {"source_id": str(src.id)},
                tenant_id=src.tenant_id,
                dedupe_key=f"sync:{src.id}",
            )
            src.next_sync_at = now + timedelta(minutes=src.sync_interval_minutes or 60)
            n += 1
        await queue.recover_stale(session)
        await session.execute(
            delete(SearchLog).where(
                SearchLog.created_at < now - timedelta(days=get_settings().search_log_retention_days)
            )
        )
    return n


async def run_worker(
    services: Services, concurrency: int | None = None, once: bool = False, metrics_port: int | None = None
) -> None:
    s = services.settings
    if metrics_port:
        from prometheus_client import start_http_server

        start_http_server(metrics_port)
    concurrency = concurrency or s.worker_concurrency
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, stop.set)
        except (NotImplementedError, RuntimeError):  # pragma: no cover - non-main thread / Windows
            pass
    base_id = f"{socket.gethostname()}:{os.getpid()}"

    async def loop_jobs(i: int) -> None:
        wid = f"{base_id}:{i}"
        while not stop.is_set():
            try:
                busy = await run_one(services, wid)
            except Exception:
                logger.exception("worker loop error")
                busy = False
            if once and not busy:
                return
            if not busy:
                try:
                    await asyncio.wait_for(stop.wait(), timeout=s.worker_poll_interval_s)
                except TimeoutError:
                    pass

    async def loop_scheduler() -> None:
        while not stop.is_set():
            try:
                n = await schedule_due_syncs()
                if n:
                    log_event(logger, "syncs scheduled", count=n)
            except Exception:
                logger.exception("scheduler error")
            try:
                await asyncio.wait_for(stop.wait(), timeout=s.scheduler_interval_s)
            except TimeoutError:
                pass

    log_event(logger, "worker started", concurrency=concurrency)
    tasks = [asyncio.create_task(loop_jobs(i)) for i in range(concurrency)]
    if not once:
        tasks.append(asyncio.create_task(loop_scheduler()))
        await stop.wait()
    await asyncio.gather(*tasks, return_exceptions=True)
    log_event(logger, "worker stopped")
