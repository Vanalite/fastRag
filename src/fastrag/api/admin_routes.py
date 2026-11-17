"""Admin endpoints: tenants, keys, collections, sources, files, mappings,
review queue and jobs. Tenant admin keys manage their own tenant; the
operator token manages all tenants (X-Tenant header selects one)."""

from __future__ import annotations

import hmac
import re
import secrets
import uuid
from datetime import UTC, datetime
from typing import Any, Literal

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from fastrag.api.auth import (
    Principal,
    clear_auth_cache,
    hash_token,
    new_token,
    require_admin,
    require_operator,
)
from fastrag.api.auth import resolve_tenant as _resolve_tenant
from fastrag.connectors.registry import KINDS
from fastrag.db.engine import get_sessionmaker, session_scope
from fastrag.db.models import (
    DEFAULT_GROUPS,
    ApiKey,
    Chunk,
    Collection,
    Document,
    Job,
    Listing,
    ListingSnapshot,
    SearchLog,
    SheetMapping,
    Source,
    Tenant,
)
from fastrag.ingest.documents import ingest_status, text_hash
from fastrag.ingest.hygiene import QUARANTINE_FLAGS
from fastrag.ingest.listings import FIELDS
from fastrag.ingest.parsers import UnsupportedFile
from fastrag.ingest.settings import IngestSettingsError, resolve_ingest, validate_collection_settings
from fastrag.ingest.sync import enqueue_processing, register_file
from fastrag.services import Services
from fastrag.worker.queue import enqueue

router = APIRouter(prefix="/v1/admin", tags=["admin"])
hooks = APIRouter(prefix="/v1/hooks", tags=["webhooks"])

SLUG = r"^[a-z0-9][a-z0-9-]{0,62}$"
GROUP = re.compile(r"^[a-z0-9_-]{1,32}$")


def _groups(values: list[str]) -> list[str]:
    for g in values:
        if not GROUP.match(g):
            raise HTTPException(422, f"invalid group name {g!r}")
    return sorted(set(values))


async def _tenant(session: AsyncSession, principal: Principal, x_tenant: str | None) -> uuid.UUID:
    return await _resolve_tenant(session, principal, x_tenant)


async def _owned(session: AsyncSession, model: Any, obj_id: uuid.UUID, tenant_id: uuid.UUID) -> Any:
    obj = await session.get(model, obj_id)
    if obj is None or getattr(obj, "tenant_id", tenant_id) != tenant_id:
        raise HTTPException(404, f"{model.__name__.lower()} not found")
    return obj


async def _collection(session: AsyncSession, tenant_id: uuid.UUID, ref: str) -> Collection:
    stmt = select(Collection).where(Collection.tenant_id == tenant_id)
    try:
        stmt = stmt.where(Collection.id == uuid.UUID(ref))
    except ValueError:
        stmt = stmt.where(Collection.slug == ref)
    col = (await session.execute(stmt)).scalar_one_or_none()
    if col is None:
        raise HTTPException(404, f"collection {ref!r} not found")
    return col


# --- Tenants and keys ---------------------------------------------------------


class TenantIn(BaseModel):
    slug: str = Field(pattern=SLUG)
    name: str = Field(min_length=1, max_length=200)


@router.post("/tenants", status_code=201)
async def create_tenant(body: TenantIn, _: Principal = Depends(require_operator)) -> dict[str, Any]:
    async with session_scope() as s:
        if (await s.execute(select(Tenant).where(Tenant.slug == body.slug))).scalar_one_or_none():
            raise HTTPException(409, "tenant exists")
        t = Tenant(id=uuid.uuid4(), slug=body.slug, name=body.name)
        s.add(t)
    return {"id": str(t.id), "slug": t.slug, "name": t.name}


@router.get("/tenants")
async def list_tenants(_: Principal = Depends(require_operator)) -> list[dict[str, Any]]:
    async with get_sessionmaker()() as s:
        rows = (await s.execute(select(Tenant).order_by(Tenant.slug))).scalars()
        return [{"id": str(t.id), "slug": t.slug, "name": t.name} for t in rows]


class KeyIn(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    allowed_groups: list[str] = Field(description='e.g. ["public"] for the customer agent')
    allowed_profiles: list[Literal["customer", "handoff_copilot", "internal_search"]] | None = None
    is_admin: bool = False


@router.post("/keys", status_code=201)
async def create_key(
    body: KeyIn, principal: Principal = Depends(require_admin), x_tenant: str | None = Header(default=None)
) -> dict[str, Any]:
    async with session_scope() as s:
        tid = await _tenant(s, principal, x_tenant)
        token, prefix, h = new_token()
        key = ApiKey(
            id=uuid.uuid4(),
            tenant_id=tid,
            name=body.name,
            prefix=prefix,
            key_hash=h,
            allowed_groups=_groups(body.allowed_groups),
            allowed_profiles=body.allowed_profiles,
            is_admin=body.is_admin,
        )
        s.add(key)
    return {"id": str(key.id), "token": token, "note": "Store this token now; it is not shown again."}


@router.get("/keys")
async def list_keys(
    principal: Principal = Depends(require_admin), x_tenant: str | None = Header(default=None)
) -> list[dict[str, Any]]:
    async with get_sessionmaker()() as s:
        tid = await _tenant(s, principal, x_tenant)
        rows = (await s.execute(select(ApiKey).where(ApiKey.tenant_id == tid).order_by(ApiKey.created_at))).scalars()
        return [
            {
                "id": str(k.id),
                "name": k.name,
                "prefix": k.prefix,
                "allowed_groups": k.allowed_groups,
                "allowed_profiles": k.allowed_profiles,
                "is_admin": k.is_admin,
                "revoked": k.revoked_at is not None,
                "last_used_at": k.last_used_at.isoformat() if k.last_used_at else None,
            }
            for k in rows
        ]


@router.delete("/keys/{key_id}", status_code=204)
async def revoke_key(
    key_id: uuid.UUID,
    principal: Principal = Depends(require_admin),
    x_tenant: str | None = Header(default=None),
) -> None:
    async with session_scope() as s:
        tid = await _tenant(s, principal, x_tenant)
        key = await _owned(s, ApiKey, key_id, tid)
        key.revoked_at = datetime.now(UTC)
    clear_auth_cache()


# --- Collections ----------------------------------------------------------------


_SETTINGS_HELP = (
    'Collection settings. "profiles" overrides retrieval profiles; "ingest" (documents collections) overrides '
    "parser, chunking and context for ingestion (D25)."
)


class CollectionIn(BaseModel):
    slug: str = Field(pattern=SLUG)
    name: str = Field(min_length=1, max_length=200)
    type: Literal["documents", "listings"]
    settings: dict[str, Any] = Field(default_factory=dict, description=_SETTINGS_HELP)


class CollectionPatch(BaseModel):
    name: str | None = None
    settings: dict[str, Any] | None = Field(default=None, description=_SETTINGS_HELP + " Replaces all settings.")


def _validate_settings(services: Services, ctype: str, settings: dict[str, Any]) -> None:
    try:
        validate_collection_settings(ctype, settings, services)
    except IngestSettingsError as exc:
        raise HTTPException(422, str(exc)) from exc


def _col_out(c: Collection) -> dict[str, Any]:
    return {
        "id": str(c.id),
        "slug": c.slug,
        "name": c.name,
        "type": c.type,
        "settings": c.settings,
        "embedding_model": c.embedding_model,
    }


@router.post("/collections", status_code=201)
async def create_collection(
    body: CollectionIn,
    request: Request,
    principal: Principal = Depends(require_admin),
    x_tenant: str | None = Header(default=None),
) -> dict[str, Any]:
    _validate_settings(request.app.state.services, body.type, body.settings)
    async with session_scope() as s:
        tid = await _tenant(s, principal, x_tenant)
        exists = await s.execute(select(Collection).where(Collection.tenant_id == tid, Collection.slug == body.slug))
        if exists.scalar_one_or_none():
            raise HTTPException(409, "collection exists")
        c = Collection(
            id=uuid.uuid4(),
            tenant_id=tid,
            slug=body.slug,
            name=body.name,
            type=body.type,
            settings=body.settings,
        )
        s.add(c)
    return _col_out(c)


@router.get("/collections")
async def list_collections(
    principal: Principal = Depends(require_admin), x_tenant: str | None = Header(default=None)
) -> list[dict[str, Any]]:
    async with get_sessionmaker()() as s:
        tid = await _tenant(s, principal, x_tenant)
        rows = (
            await s.execute(select(Collection).where(Collection.tenant_id == tid).order_by(Collection.slug))
        ).scalars()
        return [_col_out(c) for c in rows]


@router.patch("/collections/{ref}")
async def update_collection(
    ref: str,
    body: CollectionPatch,
    request: Request,
    principal: Principal = Depends(require_admin),
    x_tenant: str | None = Header(default=None),
) -> dict[str, Any]:
    """Changing ingestion settings queues a re-process of the documents that
    were built with other settings (only those)."""
    services: Services = request.app.state.services
    job: int | None = None
    async with session_scope() as s:
        tid = await _tenant(s, principal, x_tenant)
        c = await _collection(s, tid, ref)
        if body.name:
            c.name = body.name
        if body.settings is not None:
            _validate_settings(services, c.type, body.settings)
            before = _fingerprint(services, c)
            c.settings = body.settings
            if c.type == "documents" and _fingerprint(services, c) != before:
                job = await _enqueue_reindex(s, tid, c, only_stale=True)
    return _col_out(c) | {"reindex_job_id": job}


def _fingerprint(services: Services, c: Collection) -> str | None:
    if c.type != "documents":
        return None
    try:
        return resolve_ingest(c.settings, services).fingerprint
    except IngestSettingsError:
        return None


async def _enqueue_reindex(s: AsyncSession, tid: uuid.UUID, c: Collection, only_stale: bool) -> int | None:
    kind_key = "reindex-stale" if only_stale else "reindex"
    return await enqueue(
        s,
        "reindex_collection",
        {"collection_id": str(c.id), "only_stale": only_stale},
        tenant_id=tid,
        dedupe_key=f"{kind_key}:{c.id}",
    )


@router.get("/collections/{ref}/ingest")
async def collection_ingest_settings(
    ref: str,
    request: Request,
    principal: Principal = Depends(require_admin),
    x_tenant: str | None = Header(default=None),
) -> dict[str, Any]:
    """Effective ingestion settings (collection overrides on top of the
    deployment defaults), their fingerprint, and how many documents are stale."""
    services: Services = request.app.state.services
    async with get_sessionmaker()() as s:
        tid = await _tenant(s, principal, x_tenant)
        c = await _collection(s, tid, ref)
        if c.type != "documents":
            raise HTTPException(422, "ingest settings apply to documents collections only")
        try:
            cfg = resolve_ingest(c.settings, services)
        except IngestSettingsError as exc:
            raise HTTPException(422, str(exc)) from exc
        status = await ingest_status(s, c.id, cfg.fingerprint)
    overrides = (c.settings or {}).get("ingest") or {}
    return {"collection": c.slug, "overrides": overrides, "effective": cfg.describe()} | status


@router.post("/collections/{ref}/reindex", status_code=202)
async def reindex_collection(
    ref: str,
    only_stale: bool = False,
    principal: Principal = Depends(require_admin),
    x_tenant: str | None = Header(default=None),
) -> dict[str, Any]:
    """Re-process all documents, or with `only_stale=true` only those built
    with other ingestion settings."""
    async with session_scope() as s:
        tid = await _tenant(s, principal, x_tenant)
        c = await _collection(s, tid, ref)
        job = await _enqueue_reindex(s, tid, c, only_stale)
    return {"job_id": job}


# --- Sources --------------------------------------------------------------------


class SourceIn(BaseModel):
    collection: str
    kind: Literal["upload", "local_folder", "web", "google_drive"]
    name: str = Field(min_length=1, max_length=200)
    config: dict[str, Any] = Field(
        default_factory=dict,
        description='local_folder: {"path"}; web: {"urls": [...]}; google_drive: {"folder_id"}. '
        'Any kind: {"folder_groups": {"Public": ["public"]}}',
    )
    groups: list[str] = Field(default_factory=lambda: list(DEFAULT_GROUPS))
    sync_interval_minutes: int | None = Field(default=None, ge=5, le=60 * 24 * 7)


class SourcePatch(BaseModel):
    name: str | None = None
    config: dict[str, Any] | None = None
    groups: list[str] | None = None
    sync_interval_minutes: int | None = Field(default=None, ge=5, le=60 * 24 * 7)


def _src_out(src: Source) -> dict[str, Any]:
    return {
        "id": str(src.id),
        "collection_id": str(src.collection_id),
        "kind": src.kind,
        "name": src.name,
        "config": src.config,
        "groups": src.groups,
        "sync_interval_minutes": src.sync_interval_minutes,
        "last_synced_at": src.last_synced_at.isoformat() if src.last_synced_at else None,
        "last_sync_status": src.last_sync_status,
    }


def _check_source_config(kind: str, config: dict[str, Any]) -> None:
    required = {"local_folder": "path", "web": "urls", "google_drive": "folder_id"}.get(kind)
    if required and not config.get(required):
        raise HTTPException(422, f"{kind} sources need config.{required}")
    for groups in (config.get("folder_groups") or {}).values():
        _groups(list(groups))


@router.post("/sources", status_code=201)
async def create_source(
    body: SourceIn, principal: Principal = Depends(require_admin), x_tenant: str | None = Header(default=None)
) -> dict[str, Any]:
    if body.kind not in KINDS:
        raise HTTPException(422, "unknown source kind")
    _check_source_config(body.kind, body.config)
    secret = secrets.token_urlsafe(24)
    async with session_scope() as s:
        tid = await _tenant(s, principal, x_tenant)
        col = await _collection(s, tid, body.collection)
        src = Source(
            id=uuid.uuid4(),
            tenant_id=tid,
            collection_id=col.id,
            kind=body.kind,
            name=body.name,
            config=body.config,
            groups=_groups(body.groups),
            sync_interval_minutes=body.sync_interval_minutes,
            webhook_secret_hash=hash_token(secret),
        )
        s.add(src)
    out = _src_out(src)
    out["webhook"] = {
        "url": f"/v1/hooks/sources/{src.id}/sync",
        "secret": secret,
        "note": "Send the secret in the X-Webhook-Secret header. Shown once.",
    }
    return out


@router.get("/sources")
async def list_sources(
    principal: Principal = Depends(require_admin), x_tenant: str | None = Header(default=None)
) -> list[dict[str, Any]]:
    async with get_sessionmaker()() as s:
        tid = await _tenant(s, principal, x_tenant)
        rows = (await s.execute(select(Source).where(Source.tenant_id == tid).order_by(Source.created_at))).scalars()
        return [_src_out(x) for x in rows]


@router.patch("/sources/{source_id}")
async def update_source(
    source_id: uuid.UUID,
    body: SourcePatch,
    principal: Principal = Depends(require_admin),
    x_tenant: str | None = Header(default=None),
) -> dict[str, Any]:
    async with session_scope() as s:
        tid = await _tenant(s, principal, x_tenant)
        src = await _owned(s, Source, source_id, tid)
        if body.name:
            src.name = body.name
        if body.config is not None:
            _check_source_config(src.kind, body.config)
            src.config = body.config
        if body.groups is not None:
            src.groups = _groups(body.groups)
        if "sync_interval_minutes" in body.model_fields_set:
            src.sync_interval_minutes = body.sync_interval_minutes
            src.next_sync_at = None
        # Access rules changed: re-sync so documents pick up the new groups.
        if body.groups is not None or body.config is not None:
            await enqueue(s, "sync_source", {"source_id": str(src.id)}, tenant_id=tid, dedupe_key=f"sync:{src.id}")
    return _src_out(src)


@router.post("/sources/{source_id}/sync", status_code=202)
async def sync_now(
    source_id: uuid.UUID,
    principal: Principal = Depends(require_admin),
    x_tenant: str | None = Header(default=None),
) -> dict[str, Any]:
    async with session_scope() as s:
        tid = await _tenant(s, principal, x_tenant)
        src = await _owned(s, Source, source_id, tid)
        job = await enqueue(s, "sync_source", {"source_id": str(src.id)}, tenant_id=tid, dedupe_key=f"sync:{src.id}")
    return {"job_id": job, "queued": job is not None}


@hooks.post("/sources/{source_id}/sync", status_code=202)
async def sync_webhook(source_id: uuid.UUID, x_webhook_secret: str = Header()) -> dict[str, Any]:
    """ "Sync now" for Apps Script, Zapier or a cron job (D8)."""
    async with session_scope() as s:
        src = await s.get(Source, source_id)
        if (
            src is None
            or not src.webhook_secret_hash
            or not hmac.compare_digest(src.webhook_secret_hash, hash_token(x_webhook_secret))
        ):
            raise HTTPException(404, "source not found")
        job = await enqueue(
            s, "sync_source", {"source_id": str(src.id)}, tenant_id=src.tenant_id, dedupe_key=f"sync:{src.id}"
        )
    return {"queued": job is not None}


@router.post("/sources/{source_id}/files", status_code=202)
async def upload_file(
    source_id: uuid.UUID,
    request: Request,
    file: UploadFile = File(...),
    path: str = Form(default="", description="Folder inside the source, for folder_groups"),
    title: str | None = Form(default=None),
    external_id: str | None = Form(default=None),
    principal: Principal = Depends(require_admin),
    x_tenant: str | None = Header(default=None),
) -> dict[str, Any]:
    services: Services = request.app.state.services
    limit = services.settings.max_upload_mb * 1024 * 1024
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(413, f"file larger than {services.settings.max_upload_mb} MB")
    if not data:
        raise HTTPException(422, "empty file")
    filename = (file.filename or "upload").replace("\\", "/").rsplit("/", 1)[-1]
    async with session_scope() as s:
        tid = await _tenant(s, principal, x_tenant)
        src = await _owned(s, Source, source_id, tid)
        col = await s.get(Collection, src.collection_id)
        assert col is not None
        try:
            doc, changed = await register_file(
                s,
                services,
                src,
                external_id=external_id or f"{path.strip('/')}/{filename}".strip("/"),
                filename=filename,
                data=data,
                path=path,
                title=title,
                mime=file.content_type,
            )
        except (UnsupportedFile, ValueError) as exc:
            raise HTTPException(422, str(exc)) from exc
        if changed:
            await enqueue_processing(s, doc, col.type)
    return {"document_id": str(doc.id), "queued": changed, "groups": doc.groups}


# --- Documents and review ---------------------------------------------------------


def _doc_out(d: Document) -> dict[str, Any]:
    return {
        "id": str(d.id),
        "collection_id": str(d.collection_id),
        "source_id": str(d.source_id),
        "external_id": d.external_id,
        "title": d.title,
        "status": d.status,
        "groups": d.groups,
        "version": d.version,
        "error": d.error,
        "uri": d.uri,
        "ingested_at": d.ingested_at.isoformat() if d.ingested_at else None,
        "source_modified_at": d.source_modified_at.isoformat() if d.source_modified_at else None,
        "report": d.report,
    }


@router.get("/documents")
async def list_documents(
    collection: str | None = None,
    status: str | None = None,
    limit: int = 100,
    offset: int = 0,
    principal: Principal = Depends(require_admin),
    x_tenant: str | None = Header(default=None),
) -> list[dict[str, Any]]:
    async with get_sessionmaker()() as s:
        tid = await _tenant(s, principal, x_tenant)
        stmt = select(Document).where(Document.tenant_id == tid)
        if collection:
            stmt = stmt.where(Document.collection_id == (await _collection(s, tid, collection)).id)
        if status:
            stmt = stmt.where(Document.status == status)
        rows = (
            await s.execute(stmt.order_by(Document.created_at.desc()).limit(min(limit, 500)).offset(offset))
        ).scalars()
        return [_doc_out(d) for d in rows]


@router.get("/documents/{document_id}")
async def get_document(
    document_id: uuid.UUID,
    principal: Principal = Depends(require_admin),
    x_tenant: str | None = Header(default=None),
) -> dict[str, Any]:
    async with get_sessionmaker()() as s:
        tid = await _tenant(s, principal, x_tenant)
        d = await _owned(s, Document, document_id, tid)
        out = _doc_out(d)
        out["chunks"] = (await s.execute(select(func.count()).where(Chunk.document_id == d.id))).scalar()
        return out


@router.delete("/documents/{document_id}", status_code=202)
async def remove_document(
    document_id: uuid.UUID,
    principal: Principal = Depends(require_admin),
    x_tenant: str | None = Header(default=None),
) -> dict[str, Any]:
    async with session_scope() as s:
        tid = await _tenant(s, principal, x_tenant)
        d = await _owned(s, Document, document_id, tid)
        job = await enqueue(
            s, "delete_document", {"document_id": str(d.id)}, tenant_id=tid, dedupe_key=f"delete:{d.id}"
        )
    return {"job_id": job}


@router.post("/documents/{document_id}/reindex", status_code=202)
async def reindex_document(
    document_id: uuid.UUID,
    principal: Principal = Depends(require_admin),
    x_tenant: str | None = Header(default=None),
) -> dict[str, Any]:
    async with session_scope() as s:
        tid = await _tenant(s, principal, x_tenant)
        d = await _owned(s, Document, document_id, tid)
        col = await s.get(Collection, d.collection_id)
        assert col is not None
        kind = "import_sheet" if col.type == "listings" else "ingest_document"
        job = await enqueue(
            s, kind, {"document_id": str(d.id), "force": True}, tenant_id=tid, dedupe_key=f"{kind}:{d.id}"
        )
    return {"job_id": job}


@router.get("/review/chunks")
async def flagged_chunks(
    collection: str | None = None,
    limit: int = 100,
    principal: Principal = Depends(require_admin),
    x_tenant: str | None = Header(default=None),
) -> list[dict[str, Any]]:
    """Passages held back from customers: possible prompt injection, or
    personal data and internal terms in public content."""
    async with get_sessionmaker()() as s:
        tid = await _tenant(s, principal, x_tenant)
        stmt = (
            select(Chunk.id, Chunk.document_id, Chunk.text, Chunk.flags, Chunk.groups, Document.title)
            .join(Document, Document.id == Chunk.document_id)
            .where(Chunk.tenant_id == tid, Chunk.flags.overlap(list(QUARANTINE_FLAGS)))
        )
        if collection:
            stmt = stmt.where(Chunk.collection_id == (await _collection(s, tid, collection)).id)
        rows = (await s.execute(stmt.order_by(Chunk.id).limit(min(limit, 500)))).all()
        return [
            {
                "chunk_id": r.id,
                "document_id": str(r.document_id),
                "title": r.title,
                "flags": r.flags,
                "groups": r.groups,
                "text": r.text[:1500],
            }
            for r in rows
        ]


class ClearFlags(BaseModel):
    chunk_ids: list[int] = Field(min_length=1, max_length=500)


@router.post("/review/chunks/clear")
async def clear_flags(
    body: ClearFlags,
    principal: Principal = Depends(require_admin),
    x_tenant: str | None = Header(default=None),
) -> dict[str, Any]:
    """Mark passages as reviewed. The decision survives re-ingestion as long
    as the passage text is unchanged."""
    async with session_scope() as s:
        tid = await _tenant(s, principal, x_tenant)
        rows = (
            (await s.execute(select(Chunk).where(Chunk.tenant_id == tid, Chunk.id.in_(body.chunk_ids)))).scalars().all()
        )
        by_doc: dict[uuid.UUID, list[str]] = {}
        for ch in rows:
            ch.flags = [f for f in ch.flags if f not in QUARANTINE_FLAGS] + ["reviewed"]
            by_doc.setdefault(ch.document_id, []).append(text_hash(ch.text))
        for doc_id, hashes in by_doc.items():
            d = await s.get(Document, doc_id)
            if d is not None:
                d.meta = d.meta | {"cleared_hashes": sorted(set(d.meta.get("cleared_hashes", [])) | set(hashes))}
                remaining = (
                    await s.execute(
                        select(func.count()).where(
                            Chunk.document_id == doc_id,
                            Chunk.flags.overlap(list(QUARANTINE_FLAGS)),
                            Chunk.id.not_in(body.chunk_ids),
                        )
                    )
                ).scalar()
                if not remaining and d.status == "needs_review" and not (d.report or {}).get("warnings"):
                    d.status = "ready"
    return {"cleared": len(rows)}


# --- Spreadsheet mappings ---------------------------------------------------------


class ColumnSpec(BaseModel):
    field: str = Field(description=f"one of {', '.join(FIELDS)}, 'attr:<name>' or 'ignore'")
    groups: list[str] = Field(default_factory=lambda: list(DEFAULT_GROUPS))
    unit: Literal["ty", "trieu"] | None = None


class MappingUpdate(BaseModel):
    columns: dict[str, ColumnSpec]
    confirmed: bool = True


@router.get("/mappings")
async def list_mappings(
    collection: str,
    principal: Principal = Depends(require_admin),
    x_tenant: str | None = Header(default=None),
) -> list[dict[str, Any]]:
    async with get_sessionmaker()() as s:
        tid = await _tenant(s, principal, x_tenant)
        col = await _collection(s, tid, collection)
        rows = (await s.execute(select(SheetMapping).where(SheetMapping.collection_id == col.id))).scalars()
        out = []
        for m in rows:
            sample = (
                await s.execute(
                    select(Listing.attributes, Listing.title, Listing.price_vnd, Listing.district_code)
                    .join(ListingSnapshot, ListingSnapshot.id == Listing.snapshot_id)
                    .where(
                        Listing.collection_id == col.id,
                        ListingSnapshot.report["mapping_id"].astext == str(m.id),
                    )
                    .limit(3)
                )
            ).all()
            out.append(
                {
                    "id": str(m.id),
                    "headers": m.headers,
                    "columns": m.columns,
                    "confirmed": m.confirmed,
                    "updated_at": m.updated_at.isoformat() if m.updated_at else None,
                    "sample_rows": [dict(r._mapping) for r in sample],
                }
            )
        return out


@router.put("/mappings/{mapping_id}")
async def update_mapping(
    mapping_id: uuid.UUID,
    body: MappingUpdate,
    principal: Principal = Depends(require_admin),
    x_tenant: str | None = Header(default=None),
) -> dict[str, Any]:
    """Confirm the column mapping and mark public columns (D12). Re-imports
    the affected sheets so the new access rules apply immediately."""
    for header, spec in body.columns.items():
        if spec.field not in FIELDS and spec.field != "ignore" and not re.match(r"^attr:[a-z0-9-]{1,64}$", spec.field):
            raise HTTPException(422, f"column {header!r}: unknown field {spec.field!r}")
        _groups(spec.groups)
    targets = [s.field for s in body.columns.values() if s.field in FIELDS]
    dupes = sorted({t for t in targets if targets.count(t) > 1})
    if dupes:
        raise HTTPException(422, f"several columns map to {', '.join(dupes)}")
    async with session_scope() as s:
        tid = await _tenant(s, principal, x_tenant)
        m = await s.get(SheetMapping, mapping_id)
        col = await s.get(Collection, m.collection_id) if m else None
        if m is None or col is None or col.tenant_id != tid:
            raise HTTPException(404, "mapping not found")
        unknown = set(body.columns) - set(m.headers)
        if unknown:
            raise HTTPException(422, f"unknown column(s): {', '.join(sorted(unknown))}")
        m.columns = {
            h: (body.columns[h].model_dump(exclude_none=True) if h in body.columns else m.columns.get(h))
            for h in m.headers
            if h
        }
        m.confirmed = body.confirmed
        doc_ids = (
            (
                await s.execute(
                    text(
                        "SELECT DISTINCT d.id FROM documents d JOIN listing_snapshots ls ON ls.source_id = d.source_id "
                        "WHERE d.collection_id = :cid AND ls.report->>'mapping_id' = :mid AND d.status = 'imported'"
                    ),
                    {"cid": col.id, "mid": str(m.id)},
                )
            )
            .scalars()
            .all()
        )
        for did in doc_ids:
            await enqueue(s, "import_sheet", {"document_id": str(did)}, tenant_id=tid, dedupe_key=f"import_sheet:{did}")
    return {"id": str(m.id), "confirmed": m.confirmed, "columns": m.columns, "reimport_queued": len(doc_ids)}


# --- Jobs, stats, search logs -----------------------------------------------------


@router.get("/jobs")
async def list_jobs(
    status: str | None = None,
    limit: int = 50,
    principal: Principal = Depends(require_admin),
    x_tenant: str | None = Header(default=None),
) -> list[dict[str, Any]]:
    async with get_sessionmaker()() as s:
        tid = await _tenant(s, principal, x_tenant)
        stmt = select(Job).where(Job.tenant_id == tid)
        if status:
            stmt = stmt.where(Job.status == status)
        rows = (await s.execute(stmt.order_by(Job.id.desc()).limit(min(limit, 500)))).scalars()
        return [
            {
                "id": j.id,
                "kind": j.kind,
                "status": j.status,
                "attempts": j.attempts,
                "payload": j.payload,
                "last_error": (j.last_error or "")[:500] or None,
                "result": j.result,
                "created_at": j.created_at.isoformat(),
                "finished_at": j.finished_at.isoformat() if j.finished_at else None,
            }
            for j in rows
        ]


@router.post("/jobs/{job_id}/retry", status_code=202)
async def retry_job(
    job_id: int, principal: Principal = Depends(require_admin), x_tenant: str | None = Header(default=None)
) -> dict[str, Any]:
    async with session_scope() as s:
        tid = await _tenant(s, principal, x_tenant)
        j = await s.get(Job, job_id)
        if j is None or j.tenant_id != tid:
            raise HTTPException(404, "job not found")
        if j.status != "failed":
            raise HTTPException(409, "only failed jobs can be retried")
        new_id = await enqueue(s, j.kind, dict(j.payload), tenant_id=tid, dedupe_key=j.dedupe_key)
    return {"job_id": new_id, "queued": new_id is not None}


@router.get("/stats")
async def stats(
    principal: Principal = Depends(require_admin), x_tenant: str | None = Header(default=None)
) -> dict[str, Any]:
    async with get_sessionmaker()() as s:
        tid = await _tenant(s, principal, x_tenant)
        docs = (
            await s.execute(
                select(Document.status, func.count()).where(Document.tenant_id == tid).group_by(Document.status)
            )
        ).all()
        jobs = (
            await s.execute(select(Job.status, func.count()).where(Job.tenant_id == tid).group_by(Job.status))
        ).all()
        chunks = (await s.execute(select(func.count()).where(Chunk.tenant_id == tid))).scalar()
        listings = (await s.execute(select(func.count()).where(Listing.tenant_id == tid))).scalar()
        flagged = (
            await s.execute(
                select(func.count()).where(Chunk.tenant_id == tid, Chunk.flags.overlap(list(QUARANTINE_FLAGS)))
            )
        ).scalar()
        return {
            "documents": dict(docs),
            "jobs": dict(jobs),
            "chunks": chunks,
            "listings": listings,
            "chunks_held_for_review": flagged,
        }


@router.get("/search-logs")
async def search_logs(
    feedback: Literal["up", "down", "any"] | None = None,
    limit: int = 100,
    principal: Principal = Depends(require_admin),
    x_tenant: str | None = Header(default=None),
) -> list[dict[str, Any]]:
    """Recent retrieval calls; feedback=down lists candidates for the
    reference question set."""
    async with get_sessionmaker()() as s:
        tid = await _tenant(s, principal, x_tenant)
        stmt = select(SearchLog).where(SearchLog.tenant_id == tid)
        if feedback == "any":
            stmt = stmt.where(SearchLog.feedback.is_not(None))
        elif feedback:
            stmt = stmt.where(SearchLog.feedback["rating"].astext == feedback)
        rows = (await s.execute(stmt.order_by(SearchLog.id.desc()).limit(min(limit, 1000)))).scalars()
        return [
            {
                "trace_id": r.trace_id,
                "endpoint": r.endpoint,
                "profile": r.profile,
                "request": r.request,
                "result_ids": r.result_ids,
                "signals": r.signals,
                "latency_ms": r.latency_ms,
                "feedback": r.feedback,
                "created_at": r.created_at.isoformat(),
            }
            for r in rows
        ]
