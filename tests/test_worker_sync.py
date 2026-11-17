import os
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import select, text

from factories import make_collection, make_source, make_tenant, xlsx
from fastrag.db.engine import get_sessionmaker, session_scope
from fastrag.db.models import Document, Job, Listing, Source
from fastrag.ingest.sync import sync_source
from fastrag.worker import queue
from fastrag.worker.main import run_worker, schedule_due_syncs

pytestmark = pytest.mark.db


async def test_queue_dedupe_claim_and_retry(db):
    async with session_scope() as s:
        a = await queue.enqueue(s, "sync_source", {"source_id": "x"}, dedupe_key="sync:x")
        b = await queue.enqueue(s, "sync_source", {"source_id": "x"}, dedupe_key="sync:x")
        assert a is not None and b is None
    async with session_scope() as s:
        job = await queue.claim(s, "w1")
        assert job is not None and job.attempts == 1
        # While it runs, one more sync may be queued.
        assert await queue.enqueue(s, "sync_source", {"source_id": "x"}, dedupe_key="sync:x") is not None
    async with session_scope() as s:
        # An identical job is already queued, so this one is superseded, not requeued.
        await queue.fail(s, job.id, 1, 3, "boom")
        assert (await s.get(Job, job.id, populate_existing=True)).status == "superseded"
        retry = await queue.enqueue(s, "ingest_document", {"document_id": "y"}, dedupe_key="ingest:y")
        await s.execute(text("UPDATE jobs SET status='running' WHERE id=:i"), {"i": retry})
        await queue.fail(s, retry, 1, 3, "boom")
        j = await s.get(Job, retry, populate_existing=True)
        assert j.status == "queued" and j.run_after > datetime.now(UTC)
        await queue.fail(s, retry, 3, 3, "boom")
        assert (await s.get(Job, retry, populate_existing=True)).status == "failed"


async def test_recover_stale(db):
    async with session_scope() as s:
        jid = await queue.enqueue(s, "noop", {})
        await s.execute(
            text("UPDATE jobs SET status='running', locked_at = now() - interval '2 hours' WHERE id=:i"), {"i": jid}
        )
        assert await queue.recover_stale(s) == 1


async def test_unknown_job_fails_cleanly(db, services):
    async with session_scope() as s:
        jid = await queue.enqueue(s, "mystery", {}, max_attempts=1)
    await run_worker(services, 1, once=True)
    async with get_sessionmaker()() as s:
        j = await s.get(Job, jid)
        assert j.status == "failed" and "unknown job kind" in j.last_error


async def test_local_folder_sync_with_folder_groups(db, services):
    root = Path(os.environ["FASTRAG_LOCAL_SOURCES_ROOT"]) / uuid.uuid4().hex
    (root / "public").mkdir(parents=True)
    (root / "sales").mkdir()
    (root / "public" / "faq.md").write_text("# FAQ\n\n## Giờ mở cửa\nVăn phòng mở cửa 8h-17h.", encoding="utf-8")
    (root / "sales" / "noibo.md").write_text("# Nội bộ\n\n## Thưởng\nThưởng quý 10 triệu.", encoding="utf-8")
    tid = await make_tenant()
    cid = await make_collection(tid, "docs", "documents")
    sid = await make_source(
        tid, cid, kind="local_folder", config={"path": str(root)}, folder_groups={"public": ["public"]}
    )
    async with session_scope() as s:
        stats = await sync_source(s, services, sid)
    assert stats["seen"] == 2 and stats["changed"] == 2
    await run_worker(services, 1, once=True)
    async with get_sessionmaker()() as s:
        docs = {d.external_id: d for d in (await s.execute(select(Document))).scalars()}
    assert docs["public/faq.md"].groups == ["public"] and docs["public/faq.md"].status == "ready"
    assert docs["sales/noibo.md"].groups == ["staff"]

    # Second sync: nothing changed. Then one file removed.
    async with session_scope() as s:
        stats = await sync_source(s, services, sid)
    assert stats["unchanged"] == 2 and stats["changed"] == 0
    (root / "sales" / "noibo.md").unlink()
    async with session_scope() as s:
        stats = await sync_source(s, services, sid)
    assert stats["removed"] == 1
    async with get_sessionmaker()() as s:
        d = await s.get(Document, docs["sales/noibo.md"].id)
        assert d.status == "deleted"


async def test_local_folder_must_stay_inside_root(db, services):
    tid = await make_tenant()
    cid = await make_collection(tid, "docs", "documents")
    sid = await make_source(tid, cid, kind="local_folder", config={"path": "/etc"})
    async with session_scope() as s:
        with pytest.raises(ValueError):
            await sync_source(s, services, sid)


async def test_listings_sync_refreshes_as_of(db, services):
    root = Path(os.environ["FASTRAG_LOCAL_SOURCES_ROOT"]) / uuid.uuid4().hex
    root.mkdir(parents=True)
    (root / "gia.xlsx").write_bytes(xlsx([["Mã", "Giá"], ["A1", "3 tỷ"]]))
    tid = await make_tenant()
    cid = await make_collection(tid, "listings", "listings")
    sid = await make_source(tid, cid, kind="local_folder", config={"path": str(root)}, groups=["public"])
    async with session_scope() as s:
        await sync_source(s, services, sid)
    await run_worker(services, 1, once=True)
    async with get_sessionmaker()() as s:
        first = (await s.execute(select(Listing))).scalar_one()
    async with session_scope() as s:
        stats = await sync_source(s, services, sid)
    assert stats["unchanged"] == 1
    async with get_sessionmaker()() as s:
        again = (await s.execute(select(Listing))).scalar_one()
    assert again.id == first.id and again.as_of > first.as_of


async def test_scheduler_enqueues_due_sources(db, services):
    tid = await make_tenant()
    cid = await make_collection(tid, "docs", "documents")
    sid = await make_source(tid, cid)
    async with session_scope() as s:
        src = await s.get(Source, sid)
        src.sync_interval_minutes = 30
    assert await schedule_due_syncs() == 1
    assert await schedule_due_syncs() == 0
    async with get_sessionmaker()() as s:
        src = await s.get(Source, sid)
        assert src.next_sync_at > datetime.now(UTC) + timedelta(minutes=25)
