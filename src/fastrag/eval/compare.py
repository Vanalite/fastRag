"""Compare ingestion settings before adopting them (D25).

`compare_settings` builds a trial copy of a documents collection with other
ingestion settings: the same stored files and access groups, re-processed
with the new parser, chunking and context. It then runs the reference
questions against both and reports the difference. The trial collection is
only searched when named, so live traffic never sees it, and it is dropped
afterwards unless `keep` is set. Stored files are shared, never copied.
"""

from __future__ import annotations

import secrets
import time
import uuid
from collections.abc import Callable
from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from fastrag.db.models import Chunk, Collection, Document, Source
from fastrag.eval.runner import EvalReport, run_eval
from fastrag.ingest.bulk import ImportItem, process_documents
from fastrag.ingest.documents import ingest_status
from fastrag.ingest.settings import (
    IngestSettingsError,
    merge_settings,
    resolve_ingest,
    validate_collection_settings,
)
from fastrag.services import Services

TRIAL_KEY = "trial_of"
COPY_STATUSES = ("ready", "needs_review")

Progress = Callable[[str], None]


async def build_trial(
    sessionmaker: async_sessionmaker[AsyncSession],
    services: Services,
    base: Collection,
    overrides: dict[str, Any],
    *,
    concurrency: int = 4,
    progress: Progress | None = None,
) -> tuple[Collection, dict[str, str]]:
    """Create and fill the trial collection. Returns it and a map from trial
    document id to base document id."""
    settings = merge_settings(base.settings, {"ingest": overrides})
    settings[TRIAL_KEY] = str(base.id)
    validate_collection_settings("documents", settings, services)

    trial = Collection(
        id=uuid.uuid4(),
        tenant_id=base.tenant_id,
        slug=f"{base.slug[:40]}--trial-{secrets.token_hex(3)}",
        name=f"Trial of {base.name}",
        type="documents",
        settings=settings,
    )
    doc_map: dict[str, str] = {}
    work: list[tuple[ImportItem, uuid.UUID]] = []
    async with sessionmaker() as s, s.begin():
        # Explicit flushes: the models have no ORM relationships, so the
        # unit of work does not order these inserts by foreign key.
        s.add(trial)
        await s.flush()
        docs = (
            (
                await s.execute(
                    select(Document).where(
                        Document.collection_id == base.id,
                        Document.status.in_(COPY_STATUSES),
                        Document.storage_key.is_not(None),
                    )
                )
            )
            .scalars()
            .all()
        )
        trial_sources: dict[uuid.UUID, uuid.UUID] = {}
        for d in docs:
            if d.source_id not in trial_sources:
                src = await s.get(Source, d.source_id)
                assert src is not None
                copy_src = Source(
                    id=uuid.uuid4(),
                    tenant_id=base.tenant_id,
                    collection_id=trial.id,
                    kind="upload",
                    name=f"Trial of {src.name}",
                    config={},
                    groups=list(src.groups),
                )
                s.add(copy_src)
                await s.flush()
                trial_sources[d.source_id] = copy_src.id
            copy = Document(
                id=uuid.uuid4(),
                tenant_id=d.tenant_id,
                collection_id=trial.id,
                source_id=trial_sources[d.source_id],
                external_id=d.external_id,
                uri=d.uri,
                title=d.title,
                mime_type=d.mime_type,
                groups=list(d.groups),
                status="pending",
                storage_key=d.storage_key,
                source_modified_at=d.source_modified_at,
                meta=dict(d.meta or {}),
            )
            s.add(copy)
            doc_map[str(copy.id)] = str(d.id)
            work.append((ImportItem(d.external_id, "new", document_id=str(copy.id)), copy.id))
        await s.flush()
    if progress:
        progress(f"trial collection {trial.slug}: processing {len(work)} document(s)")
    await process_documents(sessionmaker, services, "documents", work, concurrency, progress)
    return trial, doc_map


async def drop_trial(sessionmaker: async_sessionmaker[AsyncSession], collection_id: uuid.UUID) -> bool:
    """Delete a trial collection (chunks, documents and sources cascade).
    Refuses anything that is not a trial; stored files stay, they belong to
    the original collection."""
    async with sessionmaker() as s, s.begin():
        res = await s.execute(
            delete(Collection).where(Collection.id == collection_id, Collection.settings.has_key(TRIAL_KEY))
        )
    return bool(res.rowcount)  # type: ignore[attr-defined]


async def _chunk_stats(sessionmaker: async_sessionmaker[AsyncSession], collection_id: uuid.UUID) -> dict[str, Any]:
    async with sessionmaker() as s:
        n, avg = (
            await s.execute(
                select(func.count(), func.avg(Chunk.token_count)).where(Chunk.collection_id == collection_id)
            )
        ).one()
    return {"chunks": int(n), "avg_tokens": round(float(avg), 1) if avg is not None else 0.0}


def _metrics(rep: EvalReport) -> dict[str, Any]:
    return {
        "questions": rep.questions,
        "recall_at_k": round(rep.recall_at_k, 4),
        "mrr": round(rep.mrr, 4),
        "nothing_accuracy": rep.nothing_accuracy,
        "access_violations": len(rep.access_violations),
    }


async def compare_settings(
    sessionmaker: async_sessionmaker[AsyncSession],
    services: Services,
    tenant_id: uuid.UUID,
    collection_slug: str,
    refs: list[dict[str, Any]],
    overrides: dict[str, Any],
    *,
    k: int = 10,
    concurrency: int = 4,
    keep: bool = False,
    progress: Progress | None = None,
) -> dict[str, Any]:
    async with sessionmaker() as s:
        base = (
            await s.execute(
                select(Collection).where(Collection.tenant_id == tenant_id, Collection.slug == collection_slug)
            )
        ).scalar_one_or_none()
        if base is None or base.type != "documents":
            raise LookupError(f"documents collection {collection_slug!r} not found")
        if TRIAL_KEY in (base.settings or {}):
            raise IngestSettingsError("compare against the original collection, not a trial")
        base_cfg = resolve_ingest(base.settings, services)
        base_status = await ingest_status(s, base.id, base_cfg.fingerprint)

    doc_refs = [
        r for r in refs if r.get("type") != "listings" and (not r.get("collections") or base.slug in r["collections"])
    ]
    if not doc_refs:
        raise ValueError(f"no document reference questions target collection {base.slug!r}")

    started = time.monotonic()
    trial, doc_map = await build_trial(
        sessionmaker, services, base, overrides, concurrency=concurrency, progress=progress
    )
    build_s = round(time.monotonic() - started, 1)
    try:
        trial_cfg = resolve_ingest(trial.settings, services)
        if progress:
            progress(f"running {len(doc_refs)} reference question(s) on both")
        base_rep = await run_eval(
            sessionmaker, services, tenant_id, [r | {"collections": [base.slug]} for r in doc_refs], k
        )
        trial_rep = await run_eval(
            sessionmaker,
            services,
            tenant_id,
            [r | {"collections": [trial.slug]} for r in doc_refs],
            k,
            document_id_map=doc_map,
        )
        async with sessionmaker() as s:
            trial_status = await ingest_status(s, trial.id, trial_cfg.fingerprint)
        base_m, trial_m = _metrics(base_rep), _metrics(trial_rep)
        changed = []
        for b, t in zip(base_rep.per_question, trial_rep.per_question, strict=True):
            if b.get("rr") != t.get("rr") or b.get("expect_nothing_ok") != t.get("expect_nothing_ok"):
                changed.append(
                    {
                        "id": b["id"],
                        "query": b["query"],
                        "base": {x: b.get(x) for x in ("recall", "rr", "expect_nothing_ok", "top")},
                        "trial": {x: t.get(x) for x in ("recall", "rr", "expect_nothing_ok", "top")},
                    }
                )
        warnings = []
        if base_status["stale"]:
            warnings.append(
                f"{base_status['stale']} base document(s) were built with other settings than the current ones; "
                "the base side mixes settings"
            )
        failed = trial_status["documents"].get("failed", 0)
        if failed:
            warnings.append(f"{failed} trial document(s) failed to process")
        return {
            "collection": base.slug,
            "k": k,
            "base": {"settings": base_cfg.describe(), "metrics": base_m, **await _chunk_stats(sessionmaker, base.id)},
            "trial": {
                "collection": trial.slug,
                "kept": keep,
                "settings": trial_cfg.describe(),
                "metrics": trial_m,
                "build_seconds": build_s,
                **await _chunk_stats(sessionmaker, trial.id),
            },
            "delta": {
                "recall_at_k": round(trial_m["recall_at_k"] - base_m["recall_at_k"], 4),
                "mrr": round(trial_m["mrr"] - base_m["mrr"], 4),
            },
            "changed_questions": changed,
            "warnings": warnings,
        }
    finally:
        if not keep:
            await drop_trial(sessionmaker, trial.id)
