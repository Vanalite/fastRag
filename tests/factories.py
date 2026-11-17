"""Helpers to create tenants, collections and content in tests."""

from __future__ import annotations

import io
import uuid
from typing import Any

from fastrag.db.engine import session_scope
from fastrag.db.models import Collection, Source, Tenant
from fastrag.ingest.documents import ingest_document
from fastrag.ingest.sync import import_spreadsheet_document, register_file
from fastrag.services import Services


async def make_tenant(slug: str = "acme") -> uuid.UUID:
    async with session_scope() as s:
        t = Tenant(id=uuid.uuid4(), slug=slug, name=slug)
        s.add(t)
    return t.id


async def make_collection(
    tenant_id: uuid.UUID, slug: str, ctype: str, settings: dict[str, Any] | None = None
) -> uuid.UUID:
    async with session_scope() as s:
        c = Collection(id=uuid.uuid4(), tenant_id=tenant_id, slug=slug, name=slug, type=ctype, settings=settings or {})
        s.add(c)
    return c.id


async def make_source(
    tenant_id: uuid.UUID,
    collection_id: uuid.UUID,
    groups: list[str] | None = None,
    folder_groups: dict[str, list[str]] | None = None,
    kind: str = "upload",
    config: dict[str, Any] | None = None,
) -> uuid.UUID:
    cfg = dict(config or {})
    if folder_groups is not None:
        cfg["folder_groups"] = folder_groups
    async with session_scope() as s:
        src = Source(
            id=uuid.uuid4(),
            tenant_id=tenant_id,
            collection_id=collection_id,
            kind=kind,
            name="src",
            config=cfg,
            groups=groups or ["staff"],
        )
        s.add(src)
    return src.id


async def add_document(
    services: Services, source_id: uuid.UUID, filename: str, content: str | bytes, path: str = ""
) -> tuple[uuid.UUID, dict[str, Any]]:
    data = content.encode("utf-8") if isinstance(content, str) else content
    async with session_scope() as s:
        src = await s.get(Source, source_id)
        assert src is not None
        doc, changed = await register_file(
            s, services, src, external_id=f"{path}/{filename}".strip("/"), filename=filename, data=data, path=path
        )
        doc_id = doc.id
    if not changed:
        return doc_id, {"status": "unchanged"}
    async with session_scope() as s:
        col = await s.get(Collection, src.collection_id)
        assert col is not None
        if col.type == "listings":
            return doc_id, await import_spreadsheet_document(s, services, doc_id)
        return doc_id, await ingest_document(s, services, doc_id)


def xlsx(rows: list[list[Any]], sheet: str = "Sheet1") -> bytes:
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.title = sheet
    for r in rows:
        ws.append(r)
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()
