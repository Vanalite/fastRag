"""Document ingestion: file -> sections -> chunks -> context -> flags ->
keyword terms + embeddings -> database. Unchanged files are skipped by
content hash (D8)."""

from __future__ import annotations

import hashlib
import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from fastrag.db.models import Collection, Document
from fastrag.ingest import contextual, hygiene
from fastrag.ingest.chunking import ChunkDraft, chunk_sections
from fastrag.ingest.index import add_chunks, remove_document_chunks
from fastrag.ingest.parsers import ParsedDocument, Section, parse
from fastrag.ingest.settings import PIPELINE_VERSION, IngestConfig, llm_for, resolve_ingest
from fastrag.log import log_event
from fastrag.services import Services
from fastrag.text.vietnamese import keyword_terms, normalize_text

logger = logging.getLogger(__name__)

__all__ = ["PIPELINE_VERSION", "PreparedDocument", "chunk_contexts", "ingest_document", "prepare_document"]


def text_hash(text: str) -> str:
    return hashlib.sha1(text.encode(), usedforsecurity=False).hexdigest()


def alias_terms(services: Services, text: str) -> list[str]:
    """Standardized place codes as extra keyword terms, so "Q7" in a question
    matches "Quận 7" in a brochure."""
    return ["@" + code for code in services.gazetteer.find_in_text(text)]


async def _ocr_scanned(services: Services, parsed: ParsedDocument, report: dict[str, Any], enabled: bool) -> None:
    if not parsed.scanned_pages:
        return
    report["scanned_pages"] = parsed.scanned_pages
    if not enabled:
        report.setdefault("warnings", []).append(
            f"{len(parsed.scanned_pages)} page(s) look scanned and OCR is off for this collection"
        )
        return
    if services.ocr is None:
        report.setdefault("warnings", []).append(
            f"{len(parsed.scanned_pages)} page(s) look scanned and OCR is not configured"
        )
        return
    for page in parsed.scanned_pages:
        images = parsed.page_images.get(page) or []
        if not images:
            continue
        try:
            text = await services.ocr.page_text(images)
        except Exception as exc:
            report.setdefault("warnings", []).append(f"OCR failed on page {page}: {exc}")
            continue
        parsed.sections.append(Section(text, ["OCR"], page))
    report["ocr_pages"] = len(parsed.scanned_pages)


@dataclass
class PreparedDocument:
    """A parsed and chunked document, before context, embedding and storage."""

    title: str
    drafts: list[ChunkDraft]
    report: dict[str, Any]
    scanned_pages: list[int]


async def prepare_document(
    services: Services,
    cfg: IngestConfig,
    data: bytes,
    filename: str,
    mime: str,
    fallback_title: str,
    *,
    title_locked: bool = False,
    run_ocr: bool = True,
) -> PreparedDocument:
    """Parse, OCR, normalize and chunk with a collection's settings. Writes
    nothing; shared by ingestion, the import dry run, chunk preview and
    settings comparison (D25)."""
    report: dict[str, Any] = {"pipeline_version": PIPELINE_VERSION, "ingest_fingerprint": cfg.fingerprint}
    if cfg.warnings:
        report["settings_warnings"] = list(cfg.warnings)
    parsed = parse(data, filename, mime, cfg.parser)
    if run_ocr:
        await _ocr_scanned(services, parsed, report, cfg.ocr)
    elif parsed.scanned_pages:
        report["scanned_pages"] = parsed.scanned_pages

    encodings: set[str] = set()
    sections: list[Section] = []
    for sec in parsed.sections:
        text, enc = normalize_text(sec.text)
        encodings.add(enc.encoding)
        if text:
            sections.append(Section(text, [normalize_text(h)[0] for h in sec.heading], sec.page, sec.kind))
    report["encodings"] = sorted(encodings)
    if "vni-suspected" in encodings:
        report.setdefault("warnings", []).append("text may use the VNI legacy encoding; check the source file")

    title = normalize_text(fallback_title if title_locked else parsed.title or fallback_title)[0]
    drafts = chunk_sections(
        sections, services.count_tokens, cfg.target_tokens, cfg.max_tokens, cfg.overlap_tokens, cfg.strategy
    )
    report.update(parsed.meta)
    return PreparedDocument(title, drafts, report, list(parsed.scanned_pages))


async def chunk_contexts(
    services: Services, cfg: IngestConfig, title: str, drafts: list[ChunkDraft]
) -> list[str | None]:
    """The context stored and embedded with each chunk: none, the title and
    heading path, or that plus an LLM summary."""
    if cfg.context_mode == "none":
        return [None] * len(drafts)
    summaries: list[str | None] = [None] * len(drafts)
    if cfg.context_mode == "llm":
        summaries = await contextual.describe(
            llm_for(services, cfg),
            title,
            contextual.outline(drafts),
            drafts,
            instructions=cfg.instructions,
            max_tokens=cfg.summary_max_tokens,
        )
    out: list[str | None] = []
    for draft, summary in zip(drafts, summaries, strict=True):
        head = contextual.header(title, draft.heading)
        out.append(head + (f"\n{summary}" if summary else ""))
    return out


async def ingest_document(
    session: AsyncSession, services: Services, document_id: uuid.UUID, force: bool = False
) -> dict[str, Any]:
    doc = await session.get(Document, document_id, with_for_update=True)
    if doc is None:
        raise LookupError(f"document {document_id} not found")
    if doc.storage_key is None:
        raise ValueError("document has no stored file")
    collection = await session.get(Collection, doc.collection_id)
    assert collection is not None
    s = services.settings
    cfg = resolve_ingest(collection.settings, services)

    data = await services.storage.get(doc.storage_key)
    digest = hashlib.sha256(data).hexdigest()
    built_with = (doc.report or {}).get("ingest_fingerprint")
    if not force and doc.content_hash == digest and doc.status == "ready" and built_with == cfg.fingerprint:
        return {"status": "unchanged"}

    # The same file already indexed elsewhere in this collection: skip it.
    dup = (
        await session.execute(
            select(Document.id, Document.title).where(
                Document.collection_id == doc.collection_id,
                Document.content_hash == digest,
                Document.id != doc.id,
                Document.status == "ready",
            )
        )
    ).first()
    if dup is not None:
        await remove_document_chunks(session, doc.id)
        doc.status, doc.content_hash = "duplicate", digest
        doc.report = {"duplicate_of": str(dup.id), "duplicate_title": dup.title}
        return {"status": "duplicate", "duplicate_of": str(dup.id)}

    doc.status = "processing"
    await session.flush()

    prepared = await prepare_document(
        services,
        cfg,
        data,
        doc.meta.get("filename", doc.title),
        doc.mime_type or "",
        doc.title,
        title_locked=bool(doc.meta.get("title_locked")),
    )
    title, drafts, report = prepared.title, prepared.drafts, prepared.report
    contexts = await chunk_contexts(services, cfg, title, drafts)

    cleared = set(doc.meta.get("cleared_hashes") or [])
    rows: list[dict[str, Any]] = []
    flag_counts: dict[str, int] = {}
    embed_inputs: list[str] = []
    for i, (draft, context) in enumerate(zip(drafts, contexts, strict=True)):
        head = contextual.header(title, draft.heading)
        flags = hygiene.scan(draft.text, doc.groups)
        if flags and text_hash(draft.text) in cleared:
            # An admin reviewed this exact text and cleared it.
            flags = [f for f in flags if f not in hygiene.QUARANTINE_FLAGS] + ["reviewed"]
        for f in flags:
            flag_counts[f] = flag_counts.get(f, 0) + 1
        prefix = f"{context}\n" if context else ""
        terms = keyword_terms(prefix + draft.text, s.segmenter, alias_terms(services, head + " " + draft.text))
        embed_inputs.append((f"{context}\n\n" if context else "") + draft.text)
        rows.append(
            {
                "tenant_id": doc.tenant_id,
                "collection_id": doc.collection_id,
                "document_id": doc.id,
                "ordinal": i,
                "text": draft.text,
                "context": context,
                "heading": " › ".join(draft.heading) or None,
                "page_start": draft.page_start,
                "page_end": draft.page_end,
                "token_count": draft.tokens,
                "groups": list(doc.groups),
                "terms": sorted(terms.freqs),
                "tf": terms.freqs,
                "length": max(terms.length, 1),
                "flags": flags,
            }
        )

    vectors = await services.embedder.embed_documents(embed_inputs) if embed_inputs else []
    for row, vec in zip(rows, vectors, strict=True):
        row["embedding"] = vec

    await remove_document_chunks(session, doc.id)
    await add_chunks(session, doc.collection_id, rows)

    report.update(
        {
            "chunks": len(rows),
            "flags": flag_counts,
            "title": title,
            "ingest": {"strategy": cfg.strategy, "context": cfg.context_mode, "llm": cfg.llm_name},
        }
    )
    if not rows:
        report.setdefault("warnings", []).append("no text extracted")
    needs_review = bool(set(flag_counts) & set(hygiene.QUARANTINE_FLAGS) or report.get("warnings"))
    doc.title = title
    doc.content_hash = digest
    doc.version = (doc.version or 0) + 1
    doc.status = "needs_review" if needs_review else "ready"
    doc.error = None
    doc.ingested_at = datetime.now(UTC)
    doc.report = report
    if collection.embedding_model is None:
        collection.embedding_model = services.embedder.model
    log_event(logger, "document ingested", document_id=str(doc.id), chunks=len(rows), status=doc.status)
    return {"status": doc.status, "chunks": len(rows), "flags": flag_counts}


async def delete_document(session: AsyncSession, document_id: uuid.UUID) -> None:
    await remove_document_chunks(session, document_id)
    await session.execute(
        update(Document).where(Document.id == document_id).values(status="deleted", content_hash=None)
    )


# Statuses a settings change re-processes. "pending"/"processing" are queued
# already; "duplicate" and "deleted" have nothing to rebuild.
REPROCESSABLE = ("ready", "needs_review", "failed")


def stale_clause(fingerprint: str) -> Any:
    """Documents not built with these settings (including ones built before
    settings were tracked)."""
    return Document.report["ingest_fingerprint"].astext.is_distinct_from(fingerprint)


async def ingest_status(session: AsyncSession, collection_id: uuid.UUID, fingerprint: str) -> dict[str, Any]:
    """Document counts for a collection: by status, and how many reprocessable
    documents are current or stale under `fingerprint`."""
    by_status = dict(
        (
            await session.execute(
                select(Document.status, func.count())
                .where(Document.collection_id == collection_id, Document.status != "deleted")
                .group_by(Document.status)
            )
        ).all()
    )
    stale = (
        await session.scalar(
            select(func.count()).where(
                Document.collection_id == collection_id,
                Document.status.in_(REPROCESSABLE),
                stale_clause(fingerprint),
            )
        )
    ) or 0
    reprocessable = sum(by_status.get(st, 0) for st in REPROCESSABLE)
    return {"documents": by_status, "current": reprocessable - stale, "stale": stale}
