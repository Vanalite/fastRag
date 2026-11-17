"""Register files and sync sources (D8).

Every file, document or spreadsheet, gets a row in `documents` with its
content hash, so unchanged files are skipped and removed files are cleaned up.
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import delete, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from fastrag.connectors.base import groups_for_path
from fastrag.connectors.registry import build_connector
from fastrag.db.models import Collection, Document, Listing, Source
from fastrag.ingest.documents import delete_document
from fastrag.ingest.listings import import_sheet
from fastrag.ingest.parsers import guess_mime
from fastrag.ingest.sheets import read_spreadsheet
from fastrag.log import log_event
from fastrag.services import Services
from fastrag.storage import content_key
from fastrag.worker.queue import enqueue

logger = logging.getLogger(__name__)

SHEET_MIMES = {
    "text/csv",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.ms-excel.sheet.macroEnabled.12",
}


async def register_file(
    session: AsyncSession,
    services: Services,
    source: Source,
    *,
    external_id: str,
    filename: str,
    data: bytes,
    path: str = "",
    uri: str | None = None,
    modified_at: datetime | None = None,
    version_tag: str | None = None,
    mime: str | None = None,
    title: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> tuple[Document, bool]:
    """Store the file and upsert its document row. Returns (document, changed)."""
    collection = await session.get(Collection, source.collection_id)
    assert collection is not None
    mime = guess_mime(filename, mime)
    is_sheet = mime in SHEET_MIMES
    if collection.type == "listings" and not is_sheet:
        raise ValueError("a listings collection only accepts spreadsheets (.xlsx, .csv)")
    if collection.type == "documents" and is_sheet:
        raise ValueError("spreadsheets go to a listings collection (D6)")
    digest = hashlib.sha256(data).hexdigest()
    groups = groups_for_path(list(source.groups), source.config or {}, path)
    doc = (
        await session.execute(
            select(Document).where(Document.source_id == source.id, Document.external_id == external_id)
        )
    ).scalar_one_or_none()
    meta = {"filename": filename, "path": path, "version_tag": version_tag} | (metadata or {})
    if title:
        meta["title_locked"] = True
    if doc is None:
        doc = Document(
            id=uuid.uuid4(),
            tenant_id=source.tenant_id,
            collection_id=source.collection_id,
            source_id=source.id,
            external_id=external_id,
            title=title or filename.rsplit(".", 1)[0],
            groups=groups,
            status="pending",
        )
        session.add(doc)
    changed = doc.content_hash != digest or doc.status not in ("ready", "imported", "needs_review")
    groups_changed = sorted(doc.groups or []) != sorted(groups)
    key = content_key(str(source.tenant_id), str(source.collection_id), data, "." + filename.rsplit(".", 1)[-1])
    if changed:
        await services.storage.put(key, data)
        doc.storage_key = key
        doc.status = "pending"
    doc.uri, doc.mime_type, doc.meta = uri, mime, (doc.meta or {}) | meta
    doc.source_modified_at = modified_at or doc.source_modified_at or datetime.now(UTC)
    if title:
        doc.title = title
    if groups_changed:
        doc.groups = groups
        if not changed and collection.type == "documents":
            # Access changed without content change: update chunks in place.
            await session.execute(
                text("UPDATE chunks SET groups = :g WHERE document_id = :d"), {"g": groups, "d": doc.id}
            )
    await session.flush()
    return doc, changed or (groups_changed and collection.type == "listings")


async def import_spreadsheet_document(
    session: AsyncSession, services: Services, document_id: uuid.UUID
) -> dict[str, Any]:
    doc = await session.get(Document, document_id, with_for_update=True)
    if doc is None or doc.storage_key is None:
        raise LookupError(f"document {document_id} not found")
    source = await session.get(Source, doc.source_id)
    assert source is not None
    data = await services.storage.get(doc.storage_key)
    sheets = read_spreadsheet(data, doc.meta.get("filename", "sheet.xlsx"))
    results = {}
    seen: list[str] = []
    for sheet in sheets:
        key = f"{doc.external_id}#{sheet.name}"
        seen.append(key)
        results[sheet.name] = await import_sheet(
            session,
            services,
            source,
            key,
            sheet.rows,
            list(doc.groups),
            as_of=datetime.now(UTC),
            storage_key=doc.storage_key,
        )
    # Tabs that disappeared from the workbook.
    await session.execute(
        delete(Listing).where(
            Listing.source_id == source.id,
            Listing.sheet_key.like(f"{doc.external_id}#%"),
            Listing.sheet_key.not_in(seen) if seen else text("true"),
        )
    )
    doc.status = "imported"
    doc.content_hash = hashlib.sha256(data).hexdigest()
    doc.ingested_at = datetime.now(UTC)
    doc.version = (doc.version or 0) + 1
    doc.report = {"sheets": results}
    return {"sheets": results}


async def enqueue_processing(session: AsyncSession, doc: Document, collection_type: str) -> None:
    kind = "import_sheet" if collection_type == "listings" else "ingest_document"
    await enqueue(session, kind, {"document_id": str(doc.id)}, tenant_id=doc.tenant_id, dedupe_key=f"{kind}:{doc.id}")


async def sync_source(session: AsyncSession, services: Services, source_id: uuid.UUID) -> dict[str, Any]:
    source = await session.get(Source, source_id)
    if source is None:
        raise LookupError(f"source {source_id} not found")
    # One sync per source at a time across all workers.
    locked = (
        await session.execute(text("SELECT pg_try_advisory_xact_lock(hashtext(:k))"), {"k": f"sync:{source_id}"})
    ).scalar()
    if not locked:
        return {"status": "skipped", "reason": "another sync of this source is running"}
    collection = await session.get(Collection, source.collection_id)
    assert collection is not None
    connector = build_connector(source, services.settings)
    if connector is None:
        return {"status": "skipped", "reason": "upload sources have nothing to sync"}

    stats: dict[str, int] = {"seen": 0, "changed": 0, "unchanged": 0, "removed": 0}
    errors: list[dict[str, str]] = []
    items = await connector.list_items()
    existing = {
        d.external_id: d
        for d in (
            await session.execute(select(Document).where(Document.source_id == source.id, Document.status != "deleted"))
        ).scalars()
    }
    now = datetime.now(UTC)
    for item in items:
        stats["seen"] += 1
        doc = existing.pop(item.external_id, None)
        groups = groups_for_path(list(source.groups), source.config or {}, item.path)
        if (
            doc is not None
            and item.version_tag
            and doc.meta.get("version_tag") == item.version_tag
            and sorted(doc.groups) == sorted(groups)
            and doc.status in ("ready", "imported", "needs_review")
        ):
            stats["unchanged"] += 1
            if collection.type == "listings":
                # Still current: refresh as_of without re-importing (D8).
                await session.execute(
                    update(Listing)
                    .where(Listing.source_id == source.id, Listing.sheet_key.like(f"{item.external_id}#%"))
                    .values(as_of=now)
                )
            continue
        try:
            data, filename = await connector.fetch(item)
            doc, changed = await register_file(
                session,
                services,
                source,
                external_id=item.external_id,
                filename=filename,
                data=data,
                path=item.path,
                uri=item.uri,
                modified_at=item.modified_at,
                version_tag=item.version_tag,
            )
        except Exception as exc:
            logger.warning("sync item failed: %s: %s", item.external_id, exc)
            errors.append({"item": item.external_id, "error": str(exc)[:300]})
            continue
        if changed:
            stats["changed"] += 1
            await enqueue_processing(session, doc, collection.type)
        else:
            stats["unchanged"] += 1
            if collection.type == "listings":
                await session.execute(
                    update(Listing)
                    .where(Listing.source_id == source.id, Listing.sheet_key.like(f"{item.external_id}#%"))
                    .values(as_of=now)
                )
    # Files removed at the source.
    for doc in existing.values():
        stats["removed"] += 1
        if collection.type == "listings":
            await session.execute(
                delete(Listing).where(Listing.source_id == source.id, Listing.sheet_key.like(f"{doc.external_id}#%"))
            )
            doc.status = "deleted"
        else:
            await delete_document(session, doc.id)
    source.last_synced_at = now
    source.last_sync_status = "ok" if not errors else f"{len(errors)} error(s)"
    log_event(logger, "source synced", source_id=str(source.id), **stats)
    return {**stats, "errors": errors}
