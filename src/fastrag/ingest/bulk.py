"""Bulk import of a folder tree into a source (D25).

`plan_import` is the dry run: it compares every file with what is indexed
and, for documents collections, parses and chunks the new and changed files
with the collection's settings to estimate chunks, embedding tokens and LLM
summary cost. Nothing is written.

`run_import` registers the files (unchanged ones are skipped by content hash)
and either queues them for the worker or processes them in this process with
bounded concurrency. `wait_for_documents` follows queued work to the end.

Re-running an import is safe: it picks up where a previous run stopped.
"""

from __future__ import annotations

import asyncio
import fnmatch
import hashlib
import time
import uuid
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from fastrag.db.models import Collection, Document, Source
from fastrag.ingest import contextual
from fastrag.ingest.documents import (
    REPROCESSABLE,
    chunk_contexts,
    ingest_document,
    prepare_document,
)
from fastrag.ingest.parsers import MIME_BY_EXT, guess_mime
from fastrag.ingest.settings import IngestConfig, llm_for, resolve_ingest
from fastrag.ingest.sync import SHEET_MIMES, enqueue_processing, import_spreadsheet_document, register_file
from fastrag.services import Services

# Rough size of the fixed summary prompt (system rules and tags), in tokens.
_SUMMARY_PROMPT_OVERHEAD = 120

Progress = Callable[[str], None]


@dataclass
class FileEntry:
    path: Path
    rel: str  # external id inside the source, POSIX style
    folder: str  # folder part of `rel`, used for folder_groups
    size: int


@dataclass
class ImportItem:
    rel: str
    action: str  # new | changed | unchanged | stale | skipped | failed
    reason: str | None = None
    document_id: str | None = None
    status: str | None = None
    chunks: int | None = None
    error: str | None = None
    estimate: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class ImportReport:
    source_id: str
    collection: str
    mode: str
    settings: dict[str, Any] | None
    items: list[ImportItem] = field(default_factory=list)
    totals: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "collection": self.collection,
            "mode": self.mode,
            "settings": self.settings,
            "totals": self.totals,
            "items": [i.to_dict() for i in self.items],
        }


def scan_folder(
    root: Path,
    *,
    prefix: str = "",
    include: Sequence[str] = (),
    exclude: Sequence[str] = (),
    recursive: bool = True,
) -> tuple[list[FileEntry], list[ImportItem]]:
    """Files under `root` that fastRag can read. Hidden files and Office lock
    files are ignored; other unsupported files are reported as skipped.
    `include`/`exclude` are glob patterns matched against the relative path
    and the file name."""
    root = root.resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"{root} is not a folder")
    prefix = prefix.strip("/")
    entries: list[FileEntry] = []
    skipped: list[ImportItem] = []
    candidates = root.rglob("*") if recursive else root.glob("*")
    for p in sorted(candidates):
        rel_parts = p.relative_to(root).parts
        if any(part.startswith((".", "~$")) for part in rel_parts) or not p.is_file():
            continue
        rel = "/".join(filter(None, [prefix, *rel_parts]))
        if include and not _matches(rel, include):
            continue
        if exclude and _matches(rel, exclude):
            continue
        if p.suffix.lower() not in MIME_BY_EXT:
            skipped.append(ImportItem(rel, "skipped", reason=f"unsupported file type {p.suffix or '(none)'}"))
            continue
        folder = str(PurePosixPath(rel).parent)
        entries.append(FileEntry(p, rel, "" if folder == "." else folder, p.stat().st_size))
    return entries, skipped


def _matches(rel: str, patterns: Sequence[str]) -> bool:
    name = PurePosixPath(rel).name
    return any(fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(name, pat) for pat in patterns)


@dataclass
class _Target:
    source: Source
    collection: Collection
    cfg: IngestConfig | None


async def _target(sessionmaker: async_sessionmaker[AsyncSession], services: Services, source_id: uuid.UUID) -> _Target:
    async with sessionmaker() as s:
        source = await s.get(Source, source_id)
        if source is None:
            raise LookupError(f"source {source_id} not found")
        collection = await s.get(Collection, source.collection_id)
        assert collection is not None
    cfg = resolve_ingest(collection.settings, services) if collection.type == "documents" else None
    return _Target(source, collection, cfg)


def _type_problem(collection_type: str, filename: str) -> str | None:
    is_sheet = guess_mime(filename) in SHEET_MIMES
    if collection_type == "listings" and not is_sheet:
        return "a listings collection only accepts spreadsheets"
    if collection_type == "documents" and is_sheet:
        return "spreadsheets go to a listings collection"
    return None


async def plan_import(
    sessionmaker: async_sessionmaker[AsyncSession],
    services: Services,
    source_id: uuid.UUID,
    entries: list[FileEntry],
    *,
    skipped: Sequence[ImportItem] = (),
    estimate: bool = True,
    reprocess_stale: bool = True,
    progress: Progress | None = None,
) -> ImportReport:
    """Dry run. Reads every file but writes nothing."""
    t = await _target(sessionmaker, services, source_id)
    async with sessionmaker() as s:
        existing = {
            d.external_id: d
            for d in (
                await s.execute(select(Document).where(Document.source_id == t.source.id, Document.status != "deleted"))
            ).scalars()
        }
    report = ImportReport(
        str(t.source.id), t.collection.slug, "dry-run", t.cfg.describe() if t.cfg else None, list(skipped)
    )
    limit = services.settings.max_upload_mb * 1024 * 1024
    too_large = f"larger than {services.settings.max_upload_mb} MB (FASTRAG_MAX_UPLOAD_MB)"
    for n, e in enumerate(entries, 1):
        if progress:
            progress(f"[{n}/{len(entries)}] {e.rel}")
        if problem := _type_problem(t.collection.type, e.path.name):
            report.items.append(ImportItem(e.rel, "skipped", reason=problem))
            continue
        if e.size > limit:
            report.items.append(ImportItem(e.rel, "skipped", reason=too_large))
            continue
        data = await asyncio.to_thread(e.path.read_bytes)
        digest = hashlib.sha256(data).hexdigest()
        doc = existing.get(e.rel)
        item = ImportItem(e.rel, "new" if doc is None else "changed", document_id=str(doc.id) if doc else None)
        if doc is not None and doc.content_hash == digest and doc.status in ("ready", "imported", "needs_review"):
            item.action = "unchanged"
            stale = t.cfg is not None and (doc.report or {}).get("ingest_fingerprint") != t.cfg.fingerprint
            if stale and reprocess_stale and doc.status in REPROCESSABLE:
                item.action = "stale"
        if estimate and t.cfg is not None and item.action in ("new", "changed", "stale"):
            try:
                item.estimate = await _estimate(services, t.cfg, data, e.path.name)
            except Exception as exc:
                item.action, item.error = "failed", f"{type(exc).__name__}: {exc}"
        report.items.append(item)
    report.totals = _plan_totals(report.items, services, t.cfg)
    return report


async def _estimate(services: Services, cfg: IngestConfig, data: bytes, filename: str) -> dict[str, Any]:
    prepared = await prepare_document(
        services, cfg, data, filename, guess_mime(filename), filename.rsplit(".", 1)[0], run_ocr=False
    )
    count = services.count_tokens
    chunk_tokens = sum(d.tokens for d in prepared.drafts)
    heads = [contextual.header(prepared.title, d.heading) for d in prepared.drafts]
    context_tokens = 0 if cfg.context_mode == "none" else sum(count(h) for h in heads)
    out: dict[str, Any] = {
        "chunks": len(prepared.drafts),
        "chunk_tokens": chunk_tokens,
        "embedding_tokens": chunk_tokens + context_tokens,
        "scanned_pages": len(prepared.scanned_pages),
    }
    if cfg.context_mode == "llm" and prepared.drafts:
        outline_tokens = count(contextual.outline(prepared.drafts)[:3000])
        title_tokens = count(prepared.title)
        per_call = _SUMMARY_PROMPT_OVERHEAD + outline_tokens + title_tokens
        out["llm_calls"] = len(prepared.drafts)
        out["llm_input_tokens"] = per_call * len(prepared.drafts) + sum(count(d.text[:4000]) for d in prepared.drafts)
        out["llm_output_tokens_max"] = cfg.summary_max_tokens * len(prepared.drafts)
        out["embedding_tokens"] += out["llm_output_tokens_max"]
    return out


def _plan_totals(items: list[ImportItem], services: Services, cfg: IngestConfig | None) -> dict[str, Any]:
    totals: dict[str, Any] = {"files": len(items), "by_action": dict(Counter(i.action for i in items))}
    keys = (
        "chunks",
        "chunk_tokens",
        "embedding_tokens",
        "scanned_pages",
        "llm_calls",
        "llm_input_tokens",
        "llm_output_tokens_max",
    )
    est = {k: sum((i.estimate or {}).get(k, 0) for i in items) for k in keys}
    totals["estimate"] = est
    llm = llm_for(services, cfg) if cfg else None
    if llm and est["llm_calls"]:
        totals["estimate"]["llm"] = f"{llm.name} ({llm.model})"
        if llm.input_cost_per_mtok is not None and llm.output_cost_per_mtok is not None:
            cost = (
                est["llm_input_tokens"] * llm.input_cost_per_mtok
                + est["llm_output_tokens_max"] * llm.output_cost_per_mtok
            ) / 1_000_000
            totals["estimate"]["llm_cost_usd_max"] = round(cost, 4)
    if est["scanned_pages"] and cfg is not None:
        totals["estimate"]["ocr_pages"] = est["scanned_pages"] if cfg.ocr and services.ocr else 0
    return totals


async def run_import(
    sessionmaker: async_sessionmaker[AsyncSession],
    services: Services,
    source_id: uuid.UUID,
    entries: list[FileEntry],
    *,
    skipped: Sequence[ImportItem] = (),
    inline: bool = False,
    concurrency: int = 4,
    reprocess_stale: bool = True,
    progress: Progress | None = None,
) -> ImportReport:
    """Register every file. Queued mode hands processing to the worker;
    inline mode processes here, `concurrency` documents at a time."""
    t = await _target(sessionmaker, services, source_id)
    report = ImportReport(
        str(t.source.id), t.collection.slug, "inline" if inline else "queued", t.cfg.describe() if t.cfg else None
    )
    report.items.extend(skipped)
    limit = services.settings.max_upload_mb * 1024 * 1024
    too_large = f"larger than {services.settings.max_upload_mb} MB (FASTRAG_MAX_UPLOAD_MB)"
    to_process: list[tuple[ImportItem, uuid.UUID]] = []
    async with sessionmaker() as s:
        known = set(
            (
                await s.execute(
                    select(Document.external_id).where(Document.source_id == t.source.id, Document.status != "deleted")
                )
            ).scalars()
        )

    for n, e in enumerate(entries, 1):
        if problem := _type_problem(t.collection.type, e.path.name):
            report.items.append(ImportItem(e.rel, "skipped", reason=problem))
            continue
        if e.size > limit:
            report.items.append(ImportItem(e.rel, "skipped", reason=too_large))
            continue
        try:
            data = await asyncio.to_thread(e.path.read_bytes)
            async with sessionmaker() as s, s.begin():
                source = await s.get(Source, t.source.id)
                assert source is not None
                doc, changed = await register_file(
                    s,
                    services,
                    source,
                    external_id=e.rel,
                    filename=e.path.name,
                    data=data,
                    path=e.folder,
                    uri=e.path.as_uri(),
                )
                item = ImportItem(e.rel, "changed" if e.rel in known else "new", document_id=str(doc.id))
                if not changed:
                    item.action, item.status = "unchanged", doc.status
                    built_with = (doc.report or {}).get("ingest_fingerprint")
                    if (
                        t.cfg is not None
                        and reprocess_stale
                        and doc.status in REPROCESSABLE
                        and built_with != t.cfg.fingerprint
                    ):
                        item.action, changed = "stale", True
                if changed:
                    # Also for stale documents, so `wait_for_documents` follows them.
                    doc.status = item.status = "pending"
                    if not inline:
                        await enqueue_processing(s, doc, t.collection.type)
                    to_process.append((item, doc.id))
        except Exception as exc:
            item = ImportItem(e.rel, "failed", error=f"{type(exc).__name__}: {exc}"[:500])
        report.items.append(item)
        if progress:
            progress(f"[{n}/{len(entries)}] {item.action}: {e.rel}")

    if inline and to_process:
        await process_documents(sessionmaker, services, t.collection.type, to_process, concurrency, progress)
    report.totals = import_totals(report.items)
    return report


async def process_documents(
    sessionmaker: async_sessionmaker[AsyncSession],
    services: Services,
    collection_type: str,
    work: list[tuple[ImportItem, uuid.UUID]],
    concurrency: int,
    progress: Progress | None,
) -> None:
    sem = asyncio.Semaphore(max(1, concurrency))
    done = 0

    async def one(item: ImportItem, doc_id: uuid.UUID) -> None:
        nonlocal done
        async with sem:
            try:
                async with sessionmaker() as s, s.begin():
                    if collection_type == "listings":
                        await import_spreadsheet_document(s, services, doc_id)
                        item.status = "imported"
                    else:
                        res = await ingest_document(s, services, doc_id)
                        item.status, item.chunks = res.get("status"), res.get("chunks")
            except Exception as exc:
                item.status, item.error = "failed", f"{type(exc).__name__}: {exc}"[:500]
                async with sessionmaker() as s, s.begin():
                    doc = await s.get(Document, doc_id)
                    if doc is not None:
                        doc.status, doc.error = "failed", item.error
            done += 1
            if progress:
                progress(f"processed {done}/{len(work)}: {item.status} {item.rel}")

    await asyncio.gather(*(one(i, d) for i, d in work))


async def wait_for_documents(
    sessionmaker: async_sessionmaker[AsyncSession],
    report: ImportReport,
    *,
    poll_s: float = 2.0,
    timeout_s: float | None = None,
    stall_s: float = 120.0,
    progress: Progress | None = None,
) -> bool:
    """Follow queued documents until none is pending or processing. Fills in
    each item's final status. Returns False on timeout."""
    ids = {uuid.UUID(i.document_id): i for i in report.items if i.document_id and i.status == "pending"}
    if not ids:
        return True
    start = last_change = time.monotonic()
    last: dict[str, int] = {}
    warned = False
    while True:
        async with sessionmaker() as s:
            rows = (
                await s.execute(
                    select(Document.id, Document.status, Document.report, Document.error).where(Document.id.in_(ids))
                )
            ).all()
        for r in rows:
            item = ids[r.id]
            item.status = r.status
            item.chunks = (r.report or {}).get("chunks")
            item.error = r.error
        counts = dict(Counter(ids[r.id].status or "" for r in rows))
        if counts != last:
            last, last_change, warned = counts, time.monotonic(), False
            if progress:
                progress("status: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
        if not counts.get("pending") and not counts.get("processing"):
            report.totals = import_totals(report.items)
            return True
        now = time.monotonic()
        if not warned and now - last_change > stall_s and progress:
            progress(f"no progress for {int(stall_s)} s; is `fastrag worker` running?")
            warned = True
        if timeout_s is not None and now - start > timeout_s:
            report.totals = import_totals(report.items)
            return False
        await asyncio.sleep(poll_s)


def import_totals(items: list[ImportItem]) -> dict[str, Any]:
    return {
        "files": len(items),
        "by_action": dict(Counter(i.action for i in items)),
        "by_status": dict(Counter(i.status for i in items if i.status)),
        "chunks": sum(i.chunks or 0 for i in items),
        "errors": [{"file": i.rel, "error": i.error} for i in items if i.error][:50],
    }


async def preview_chunks(
    services: Services,
    cfg: IngestConfig,
    path: Path,
    *,
    with_context: bool = False,
    context_limit: int = 5,
) -> dict[str, Any]:
    """How a file would be chunked with these settings. Writes nothing; with
    `with_context` the first `context_limit` chunks also get their context
    (this calls the LLM when the context mode is 'llm')."""
    data = await asyncio.to_thread(path.read_bytes)
    prepared = await prepare_document(services, cfg, data, path.name, guess_mime(path.name), path.stem, run_ocr=False)
    contexts: list[str | None] = [None] * len(prepared.drafts)
    if with_context and prepared.drafts:
        head = prepared.drafts[:context_limit]
        contexts[: len(head)] = await chunk_contexts(services, cfg, prepared.title, head)
    tokens = [d.tokens for d in prepared.drafts]
    return {
        "file": str(path),
        "title": prepared.title,
        "settings": cfg.describe(),
        "stats": {
            "chunks": len(tokens),
            "tokens_min": min(tokens, default=0),
            "tokens_avg": round(sum(tokens) / len(tokens), 1) if tokens else 0,
            "tokens_max": max(tokens, default=0),
            "scanned_pages": prepared.scanned_pages,
        },
        "warnings": prepared.report.get("warnings", []),
        "chunks": [
            {
                "ordinal": i,
                "heading": " › ".join(d.heading) or None,
                "pages": [d.page_start, d.page_end] if d.page_start else None,
                "tokens": d.tokens,
                "context": ctx,
                "text": d.text,
            }
            for i, (d, ctx) in enumerate(zip(prepared.drafts, contexts, strict=True))
        ],
    }
