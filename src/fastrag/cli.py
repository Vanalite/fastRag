"""fastrag command line: run the services and manage tenants without the API."""

from __future__ import annotations

import asyncio
import json
import sys
import uuid
from pathlib import Path
from typing import Any

import typer

app = typer.Typer(help="fastRag: retrieval service for AI agents.", no_args_is_help=True)


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def _print(obj: Any) -> None:
    typer.echo(json.dumps(obj, ensure_ascii=False, indent=2, default=str))


@app.command()
def migrate(revision: str = "head") -> None:
    """Create or upgrade the database schema."""
    from fastrag.db.migrate import upgrade

    upgrade(revision=revision)
    typer.echo("database is up to date")


@app.command()
def serve(host: str = "0.0.0.0", port: int = 8000, workers: int = 1) -> None:  # noqa: S104
    """Run the HTTP API."""
    import uvicorn

    from fastrag.config import get_settings
    from fastrag.log import configure_logging

    s = get_settings()
    configure_logging(s.log_level, s.log_json)
    uvicorn.run(
        "fastrag.api.app:create_app",
        factory=True,
        host=host,
        port=port,
        workers=workers,
        log_config=None,
        proxy_headers=True,
    )


@app.command()
def worker(
    concurrency: int = 0,
    once: bool = typer.Option(False, help="Exit when the queue is empty."),
    metrics_port: int = typer.Option(0, help="Serve Prometheus metrics on this port (0 = off)."),
) -> None:
    """Run the ingestion worker and sync scheduler."""
    from fastrag.config import get_settings
    from fastrag.log import configure_logging
    from fastrag.services import build_services
    from fastrag.worker.main import run_worker

    s = get_settings()
    configure_logging(s.log_level, s.log_json)

    async def main() -> None:
        services = build_services(s)
        try:
            await run_worker(services, concurrency or None, once=once, metrics_port=metrics_port or None)
        finally:
            await services.aclose()
            from fastrag.db.engine import dispose_engine

            await dispose_engine()

    _run(main())


async def _tenant_id(session: Any, slug: str) -> uuid.UUID:
    from sqlalchemy import select

    from fastrag.db.models import Tenant

    tid = (await session.execute(select(Tenant.id).where(Tenant.slug == slug))).scalar_one_or_none()
    if tid is None:
        typer.echo(f"tenant {slug!r} not found", err=True)
        raise typer.Exit(1)
    return tid  # type: ignore[no-any-return]


@app.command("create-tenant")
def create_tenant(slug: str, name: str = typer.Option(None)) -> None:
    from fastrag.db.engine import session_scope
    from fastrag.db.models import Tenant

    async def main() -> None:
        async with session_scope() as s:
            t = Tenant(id=uuid.uuid4(), slug=slug, name=name or slug)
            s.add(t)
        _print({"id": str(t.id), "slug": slug})

    _run(main())


@app.command("create-collection")
def create_collection(
    tenant: str,
    slug: str,
    type: str = typer.Option("documents", help="documents | listings"),
    name: str = typer.Option(None),
    ingest: str = typer.Option(
        None, help='Ingestion settings as JSON, e.g. \'{"chunking": {"max_tokens": 256}}\' (see ingest-settings).'
    ),
) -> None:
    from fastrag.db.engine import session_scope
    from fastrag.db.models import Collection

    if type not in ("documents", "listings"):
        raise typer.BadParameter("type must be documents or listings")
    settings: dict[str, Any] = {"ingest": _json_option(ingest, "--ingest")} if ingest else {}
    if settings:
        _check_ingest(type, settings)

    async def main() -> None:
        async with session_scope() as s:
            tid = await _tenant_id(s, tenant)
            c = Collection(id=uuid.uuid4(), tenant_id=tid, slug=slug, name=name or slug, type=type, settings=settings)
            s.add(c)
        _print({"id": str(c.id), "slug": slug, "type": type})

    _run(main())


@app.command("create-key")
def create_key(
    tenant: str,
    name: str = typer.Option(..., help="e.g. customer-agent"),
    groups: list[str] = typer.Option(["public"], "--group", help="Repeat for several groups."),
    profile: list[str] = typer.Option([], "--profile", help="Restrict to profiles; repeatable."),
    admin: bool = False,
) -> None:
    """Create a tenant API key. The token is printed once."""
    from fastrag.api.auth import new_token
    from fastrag.db.engine import session_scope
    from fastrag.db.models import ApiKey

    async def main() -> None:
        async with session_scope() as s:
            tid = await _tenant_id(s, tenant)
            token, prefix, h = new_token()
            s.add(
                ApiKey(
                    id=uuid.uuid4(),
                    tenant_id=tid,
                    name=name,
                    prefix=prefix,
                    key_hash=h,
                    allowed_groups=sorted(set(groups)),
                    allowed_profiles=profile or None,
                    is_admin=admin,
                )
            )
        _print({"token": token, "groups": sorted(set(groups)), "profiles": profile or "all", "admin": admin})

    _run(main())


@app.command("create-source")
def create_source(
    tenant: str,
    collection: str,
    kind: str = typer.Option("upload", help="upload | local_folder | web | google_drive"),
    name: str = typer.Option("uploads"),
    config: str = typer.Option("{}", help='JSON, e.g. {"path": "inbox", "folder_groups": {"public": ["public"]}}'),
    groups: list[str] = typer.Option(["staff"], "--group"),
    interval: int = typer.Option(0, help="Sync interval in minutes; 0 = manual."),
) -> None:
    from sqlalchemy import select

    from fastrag.db.engine import session_scope
    from fastrag.db.models import Collection, Source

    async def main() -> None:
        async with session_scope() as s:
            tid = await _tenant_id(s, tenant)
            col = (
                await s.execute(select(Collection).where(Collection.tenant_id == tid, Collection.slug == collection))
            ).scalar_one()
            src = Source(
                id=uuid.uuid4(),
                tenant_id=tid,
                collection_id=col.id,
                kind=kind,
                name=name,
                config=json.loads(config),
                groups=sorted(set(groups)),
                sync_interval_minutes=interval or None,
            )
            s.add(src)
        _print({"id": str(src.id), "kind": kind, "groups": src.groups})

    _run(main())


@app.command()
def ingest(
    source_id: str,
    files: list[Path],
    path: str = typer.Option("", help="Folder inside the source (for folder_groups)."),
) -> None:
    """Upload files to a source and process them now (no worker needed)."""
    from fastrag.db.engine import dispose_engine, session_scope
    from fastrag.db.models import Collection, Source
    from fastrag.ingest.documents import ingest_document
    from fastrag.ingest.sync import import_spreadsheet_document, register_file
    from fastrag.services import build_services

    async def main() -> None:
        services = build_services()
        try:
            for f in files:
                async with session_scope() as s:
                    src = await s.get(Source, uuid.UUID(source_id))
                    if src is None:
                        raise typer.BadParameter("source not found")
                    col = await s.get(Collection, src.collection_id)
                    assert col is not None
                    doc, changed = await register_file(
                        s,
                        services,
                        src,
                        external_id=f"{path.strip('/')}/{f.name}".strip("/"),
                        filename=f.name,
                        data=await asyncio.to_thread(f.read_bytes),
                        path=path,
                    )
                    doc_id = doc.id
                if not changed:
                    typer.echo(f"{f.name}: unchanged")
                    continue
                async with session_scope() as s:
                    if col.type == "listings":
                        res = await import_spreadsheet_document(s, services, doc_id)
                    else:
                        res = await ingest_document(s, services, doc_id)
                typer.echo(f"{f.name}: {json.dumps(res, ensure_ascii=False, default=str)}")
        finally:
            await services.aclose()
            await dispose_engine()

    _run(main())


@app.command()
def sync(source_id: str) -> None:
    """Sync a source now and process the changed files."""
    from fastrag.db.engine import dispose_engine, session_scope
    from fastrag.ingest.sync import sync_source
    from fastrag.services import build_services
    from fastrag.worker.main import run_worker

    async def main() -> None:
        services = build_services()
        try:
            async with session_scope() as s:
                _print(await sync_source(s, services, uuid.UUID(source_id)))
            await run_worker(services, 1, once=True)
        finally:
            await services.aclose()
            await dispose_engine()

    _run(main())


@app.command()
def search(
    tenant: str,
    query: str,
    profile: str = "internal_search",
    top_k: int = 5,
    group: list[str] = typer.Option(["public", "staff"], "--group"),
) -> None:
    """Try a search from the terminal."""
    from sqlalchemy import select

    from fastrag.db.engine import dispose_engine, get_sessionmaker
    from fastrag.db.models import Collection
    from fastrag.retrieval.profiles import effective_groups, resolve_profile
    from fastrag.retrieval.search import SearchRequest
    from fastrag.retrieval.search import search as do_search
    from fastrag.services import build_services

    async def main() -> None:
        services = build_services()
        try:
            async with get_sessionmaker()() as s:
                tid = await _tenant_id(s, tenant)
                cols = list(
                    (
                        await s.execute(
                            select(Collection).where(Collection.tenant_id == tid, Collection.type == "documents")
                        )
                    ).scalars()
                )
            prof = resolve_profile(profile)
            req = SearchRequest(tid, [c.id for c in cols], query, effective_groups(None, group, prof), prof, top_k)
            res = await do_search(get_sessionmaker(), services, req)
            _print(
                {
                    "signals": res.signals,
                    "timings_ms": res.timings_ms,
                    "passages": [
                        {"score": p.score, "title": p.title, "heading": p.heading, "text": p.text[:300]}
                        for p in res.passages
                    ],
                }
            )
        finally:
            await services.aclose()
            await dispose_engine()

    _run(main())


@app.command("eval")
def eval_cmd(
    tenant: str,
    references: Path,
    k: int = 10,
    baseline: Path | None = typer.Option(None, help="Another system's results (JSONL) to compare with."),
    out: Path | None = typer.Option(None, help="Write the full report as JSON."),
) -> None:
    """Run the release checks against a reference question file. Exits 1 if a check fails."""
    from fastrag.db.engine import dispose_engine, get_sessionmaker
    from fastrag.eval.runner import load_references, run_eval
    from fastrag.services import build_services

    async def main() -> bool:
        services = build_services()
        try:
            async with get_sessionmaker()() as s:
                tid = await _tenant_id(s, tenant)
            rep = await run_eval(
                get_sessionmaker(),
                services,
                tid,
                load_references(references),
                k,
                str(baseline) if baseline else None,
            )
            data = rep.to_dict()
            if out:
                payload = json.dumps(data, ensure_ascii=False, indent=2, default=str)
                await asyncio.to_thread(out.write_text, payload, encoding="utf-8")
            summary = {k2: v for k2, v in data.items() if k2 != "per_question"}
            _print(summary)
            return rep.passed
        finally:
            await services.aclose()
            await dispose_engine()

    if not _run(main()):
        sys.exit(1)


@app.command()
def calibrate(tenant: str, references: Path) -> None:
    """Fit reranker score calibration on reference questions and print the
    FASTRAG_RERANKER_CALIBRATION value to use."""
    from fastrag.db.engine import dispose_engine, get_sessionmaker
    from fastrag.eval.calibrate import collect_pairs, fit_platt, summarize
    from fastrag.eval.runner import load_references
    from fastrag.services import build_services

    async def main() -> None:
        services = build_services()
        try:
            async with get_sessionmaker()() as s:
                tid = await _tenant_id(s, tenant)
            pairs = await collect_pairs(get_sessionmaker(), services, tid, load_references(references))
            if sum(y for _, y in pairs) < 3 or len(pairs) - sum(y for _, y in pairs) < 3:
                typer.echo("not enough labelled candidates; add more reference questions", err=True)
                raise typer.Exit(1)
            a, b = fit_platt([x for x, _ in pairs], [y for _, y in pairs])
            out = summarize(pairs, a, b)
            out["env"] = f"FASTRAG_RERANKER_CALIBRATION={a:.3f},{b:.3f}"
            _print(out)
        finally:
            await services.aclose()
            await dispose_engine()

    _run(main())


@app.command()
def mcp() -> None:
    """Run an MCP server (stdio) for agent platforms. Needs the `mcp` extra."""
    from fastrag.mcp_server import main

    main()


@app.command()
def bootstrap(
    tenant: str = typer.Argument("pilot"),
    name: str = typer.Option("Pilot tenant"),
) -> None:
    """One-step setup: a tenant, a documents and a listings collection with
    upload sources, and keys for the customer agent, staff and an admin."""
    from fastrag.api.auth import new_token
    from fastrag.db.engine import session_scope
    from fastrag.db.models import ApiKey, Collection, Source, Tenant

    async def main() -> None:
        out: dict[str, Any] = {"tenant": tenant, "keys": {}, "sources": {}}
        async with session_scope() as s:
            t = Tenant(id=uuid.uuid4(), slug=tenant, name=name)
            s.add(t)
            await s.flush()
            for slug, ctype in (("docs", "documents"), ("listings", "listings")):
                c = Collection(id=uuid.uuid4(), tenant_id=t.id, slug=slug, name=slug, type=ctype, settings={})
                s.add(c)
                await s.flush()
                src = Source(
                    id=uuid.uuid4(),
                    tenant_id=t.id,
                    collection_id=c.id,
                    kind="upload",
                    name=f"{slug}-uploads",
                    config={"folder_groups": {"public": ["public"]}},
                    groups=["staff"],
                )
                s.add(src)
                out["sources"][slug] = str(src.id)
            for key_name, groups, profiles, admin in (
                ("customer-agent", ["public"], ["customer"], False),
                ("staff", ["public", "staff"], None, False),
                ("admin", ["public", "staff"], None, True),
            ):
                token, prefix, h = new_token()
                s.add(
                    ApiKey(
                        id=uuid.uuid4(),
                        tenant_id=t.id,
                        name=key_name,
                        prefix=prefix,
                        key_hash=h,
                        allowed_groups=groups,
                        allowed_profiles=profiles,
                        is_admin=admin,
                    )
                )
                out["keys"][key_name] = token
        out["note"] = (
            "Tokens are shown once. Files uploaded with path 'public/...' are visible to customers; "
            "everything else is staff-only until you change folder_groups."
        )
        _print(out)

    _run(main())


# --- Ingestion settings, bulk import, preview and comparison (D25) -----------------


def _json_option(value: str | None, flag: str) -> dict[str, Any]:
    if not value:
        return {}
    try:
        data = json.loads(value)
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(f"{flag} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise typer.BadParameter(f"{flag} must be a JSON object")
    return data


def _check_ingest(collection_type: str, settings: dict[str, Any]) -> None:
    from fastrag.ingest.settings import IngestSettingsError, validate_collection_settings
    from fastrag.services import build_services

    async def check() -> None:
        services = build_services()
        try:
            validate_collection_settings(collection_type, settings, services)
        finally:
            await services.aclose()

    try:
        _run(check())
    except IngestSettingsError as exc:
        typer.echo(f"invalid ingest settings: {exc}", err=True)
        raise typer.Exit(2) from exc


async def _find_collection(session: Any, tenant: str, slug: str) -> Any:
    from sqlalchemy import select

    from fastrag.db.models import Collection

    tid = await _tenant_id(session, tenant)
    c = (
        await session.execute(select(Collection).where(Collection.tenant_id == tid, Collection.slug == slug))
    ).scalar_one_or_none()
    if c is None:
        typer.echo(f"collection {slug!r} not found", err=True)
        raise typer.Exit(1)
    return c


def _progress(line: str) -> None:
    typer.echo(line, err=True)


@app.command("ingest-settings")
def ingest_settings(
    tenant: str,
    collection: str,
    set_: str = typer.Option(
        None,
        "--set",
        help='Merge these overrides into the collection, e.g. \'{"context": {"mode": "header"}}\'. null removes a key.',
    ),
    replace: bool = typer.Option(False, help="Replace all overrides with --set instead of merging."),
    reindex: bool = typer.Option(True, help="Queue a re-process of documents built with other settings."),
) -> None:
    """Show a documents collection's effective ingestion settings and how many
    documents are stale; with --set, change them."""
    from fastrag.db.engine import dispose_engine, session_scope
    from fastrag.ingest.documents import ingest_status
    from fastrag.ingest.settings import IngestSettingsError, merge_settings, resolve_ingest
    from fastrag.ingest.settings import validate_collection_settings as validate
    from fastrag.services import build_services
    from fastrag.worker.queue import enqueue

    patch = _json_option(set_, "--set")

    async def main() -> dict[str, Any]:
        services = build_services()
        try:
            async with session_scope() as s:
                c = await _find_collection(s, tenant, collection)
                if c.type != "documents":
                    typer.echo("ingest settings apply to documents collections only", err=True)
                    raise typer.Exit(2)
                out: dict[str, Any] = {"collection": c.slug}
                if set_ is not None:
                    before = resolve_ingest(c.settings, services).fingerprint
                    current = {} if replace else dict((c.settings or {}).get("ingest") or {})
                    settings = dict(c.settings or {}) | {"ingest": merge_settings(current, patch)}
                    if not settings["ingest"]:
                        settings.pop("ingest")
                    validate("documents", settings, services)
                    c.settings = settings
                    if reindex and resolve_ingest(settings, services).fingerprint != before:
                        out["reindex_job_id"] = await enqueue(
                            s,
                            "reindex_collection",
                            {"collection_id": str(c.id), "only_stale": True},
                            tenant_id=c.tenant_id,
                            dedupe_key=f"reindex-stale:{c.id}",
                        )
                cfg = resolve_ingest(c.settings, services)
                out |= {"overrides": (c.settings or {}).get("ingest") or {}, "effective": cfg.describe()}
                out |= await ingest_status(s, c.id, cfg.fingerprint)
                return out
        except IngestSettingsError as exc:
            typer.echo(f"invalid ingest settings: {exc}", err=True)
            raise typer.Exit(2) from exc
        finally:
            await services.aclose()
            await dispose_engine()

    _print(_run(main()))


@app.command("import")
def import_cmd(
    source_id: str,
    folder: Path,
    prefix: str = typer.Option("", help="Folder inside the source to import into (for folder_groups)."),
    include: list[str] = typer.Option([], "--include", help="Glob on path or file name; repeatable."),
    exclude: list[str] = typer.Option([], "--exclude", help="Glob on path or file name; repeatable."),
    recursive: bool = typer.Option(True, help="Walk sub-folders."),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Report what would happen and estimate costs; write nothing."
    ),
    inline: bool = typer.Option(False, help="Process here instead of queueing for the worker."),
    concurrency: int = typer.Option(4, min=1, max=32, help="Documents processed at once with --inline."),
    wait: bool = typer.Option(True, help="Queued mode: follow processing until it finishes."),
    timeout: float = typer.Option(None, help="Stop waiting after this many seconds."),
    reprocess_stale: bool = typer.Option(
        True, help="Also re-process unchanged files built with other ingestion settings."
    ),
    report: Path = typer.Option(None, help="Write the full per-file report as JSON."),
) -> None:
    """Import a folder tree into a source. Sub-folders become paths inside the
    source, so the source's folder_groups decide access (e.g. public/...).
    Unchanged files are skipped, so re-running continues an interrupted import."""
    from fastrag.db.engine import dispose_engine, get_sessionmaker
    from fastrag.ingest.bulk import plan_import, run_import, scan_folder, wait_for_documents
    from fastrag.services import build_services

    try:
        entries, skipped = scan_folder(folder, prefix=prefix, include=include, exclude=exclude, recursive=recursive)
    except NotADirectoryError as exc:
        raise typer.BadParameter(str(exc)) from exc
    _progress(f"{len(entries)} file(s) to consider, {len(skipped)} skipped")

    async def main() -> tuple[dict[str, Any], bool]:
        services = build_services()
        sm = get_sessionmaker()
        try:
            sid = uuid.UUID(source_id)
            if dry_run:
                rep = await plan_import(
                    sm, services, sid, entries, skipped=skipped, reprocess_stale=reprocess_stale, progress=_progress
                )
                return rep.to_dict(), True
            rep = await run_import(
                sm,
                services,
                sid,
                entries,
                skipped=skipped,
                inline=inline,
                concurrency=concurrency,
                reprocess_stale=reprocess_stale,
                progress=_progress,
            )
            finished = True
            if not inline and wait:
                finished = await wait_for_documents(sm, rep, timeout_s=timeout, progress=_progress)
            return rep.to_dict(), finished
        except (LookupError, ValueError) as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(1) from exc
        finally:
            await services.aclose()
            await dispose_engine()

    data, finished = _run(main())
    if report:
        report.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    _print({k: v for k, v in data.items() if k != "items"})
    failed = data["totals"].get("by_status", {}).get("failed", 0) + data["totals"]["by_action"].get("failed", 0)
    if not finished or failed:
        sys.exit(1)


@app.command("chunk-preview")
def chunk_preview(
    file: Path,
    tenant: str = typer.Option(None, help="Use this tenant's collection settings (with --collection)."),
    collection: str = typer.Option(None, help="Documents collection whose settings to use."),
    set_: str = typer.Option(None, "--set", help="Ingest overrides to try on top, as JSON."),
    context: bool = typer.Option(False, help="Also build the context of the first chunks (calls the LLM in llm mode)."),
    show: int = typer.Option(10, help="Chunks to print in full; 0 prints only statistics."),
    as_json: bool = typer.Option(False, "--json", help="Print everything as JSON."),
) -> None:
    """Show how a file would be chunked, without writing anything."""
    from fastrag.db.engine import dispose_engine, get_sessionmaker
    from fastrag.ingest.bulk import preview_chunks
    from fastrag.ingest.settings import IngestSettingsError, merge_settings, resolve_ingest
    from fastrag.services import build_services

    if not file.is_file():
        raise typer.BadParameter(f"{file} is not a file")
    if (tenant is None) != (collection is None):
        raise typer.BadParameter("give both --tenant and --collection, or neither")

    async def main() -> dict[str, Any]:
        services = build_services()
        try:
            base: dict[str, Any] = {}
            if collection:
                async with get_sessionmaker()() as s:
                    base = dict((await _find_collection(s, tenant, collection)).settings or {})
                await dispose_engine()
            settings = merge_settings(base, {"ingest": _json_option(set_, "--set")}) if set_ else base
            cfg = resolve_ingest(settings, services, strict=bool(set_))
            return await preview_chunks(services, cfg, file, with_context=context, context_limit=max(show, 1))
        except IngestSettingsError as exc:
            typer.echo(f"invalid ingest settings: {exc}", err=True)
            raise typer.Exit(2) from exc
        finally:
            await services.aclose()

    out = _run(main())
    if as_json:
        _print(out)
        return
    eff = out["settings"]
    typer.echo(f"{out['title']}  ({file.name})")
    typer.echo(
        f"settings: parser={eff['parser']} strategy={eff['strategy']} target={eff['target_tokens']} "
        f"max={eff['max_tokens']} overlap={eff['overlap_tokens']} context={eff['context_mode']}"
        + (f" llm={eff['llm_name']}" if eff["llm_name"] else "")
    )
    st = out["stats"]
    typer.echo(
        f"chunks: {st['chunks']}  tokens min/avg/max: {st['tokens_min']}/{st['tokens_avg']}/{st['tokens_max']}"
        + (f"  scanned pages (not OCR'd in preview): {st['scanned_pages']}" if st["scanned_pages"] else "")
    )
    for w in out["warnings"]:
        typer.echo(f"warning: {w}")
    for ch in out["chunks"][:show]:
        pages = f" p.{ch['pages'][0]}-{ch['pages'][1]}" if ch["pages"] else ""
        typer.echo(f"\n--- #{ch['ordinal']} [{ch['tokens']} tok]{pages} {ch['heading'] or ''}")
        if ch["context"]:
            typer.echo(f"context: {ch['context']}")
        typer.echo(ch["text"])
    if show and len(out["chunks"]) > show:
        typer.echo(f"\n... {len(out['chunks']) - show} more chunk(s); use --show or --json")


@app.command()
def compare(
    tenant: str,
    collection: str,
    references: Path,
    set_: str = typer.Option(..., "--set", help="Ingest overrides to try, as JSON."),
    k: int = 10,
    concurrency: int = typer.Option(4, min=1, max=32, help="Documents processed at once."),
    keep: bool = typer.Option(False, help="Keep the trial collection for inspection (drop it with drop-trial)."),
    yes: bool = typer.Option(False, "--yes", help="Do not ask before re-processing with LLM summaries."),
    out: Path = typer.Option(None, help="Write the full comparison as JSON."),
) -> None:
    """Re-process a documents collection with other ingestion settings into a
    hidden trial copy, run the reference questions on both and print the
    difference. Adopt the winner with `fastrag ingest-settings --set`."""
    from fastrag.db.engine import dispose_engine, get_sessionmaker
    from fastrag.eval.compare import compare_settings
    from fastrag.eval.runner import load_references
    from fastrag.ingest.settings import IngestSettingsError, merge_settings, resolve_ingest
    from fastrag.services import build_services

    overrides = _json_option(set_, "--set")
    refs = load_references(references)

    async def main() -> dict[str, Any] | None:
        services = build_services()
        sm = get_sessionmaker()
        try:
            async with sm() as s:
                c = await _find_collection(s, tenant, collection)
                tid = c.tenant_id
                trial_cfg = resolve_ingest(merge_settings(c.settings, {"ingest": overrides}), services, strict=True)
            if trial_cfg.context_mode == "llm" and not yes:
                typer.echo(
                    f"The trial re-processes every document with LLM summaries ({trial_cfg.llm_name}). "
                    "Estimate the cost with `fastrag import --dry-run`, then re-run with --yes.",
                    err=True,
                )
                raise typer.Exit(2)
            return await compare_settings(
                sm,
                services,
                tid,
                collection,
                refs,
                overrides,
                k=k,
                concurrency=concurrency,
                keep=keep,
                progress=_progress,
            )
        except (IngestSettingsError, LookupError, ValueError) as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(2) from exc
        finally:
            await services.aclose()
            await dispose_engine()

    data = _run(main())
    if out:
        out.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    summary = {key: v for key, v in data.items() if key != "changed_questions"}
    summary["changed_questions"] = len(data["changed_questions"])
    _print(summary)
    if (
        data["delta"]["recall_at_k"] >= 0
        and data["delta"]["mrr"] >= 0
        and not data["trial"]["metrics"]["access_violations"]
    ):
        adopt = json.dumps(overrides, ensure_ascii=False)
        typer.echo(f"\nTo adopt: fastrag ingest-settings {tenant} {collection} --set '{adopt}'", err=True)


@app.command("drop-trial")
def drop_trial_cmd(tenant: str, collection: str) -> None:
    """Delete a trial collection kept by `fastrag compare --keep`."""
    from fastrag.db.engine import dispose_engine, get_sessionmaker
    from fastrag.eval.compare import TRIAL_KEY, drop_trial

    async def main() -> bool:
        try:
            async with get_sessionmaker()() as s:
                c = await _find_collection(s, tenant, collection)
            if TRIAL_KEY not in (c.settings or {}):
                typer.echo(f"{collection!r} is not a trial collection", err=True)
                raise typer.Exit(2)
            return await drop_trial(get_sessionmaker(), c.id)
        finally:
            await dispose_engine()

    _print({"dropped": _run(main()), "collection": collection})


if __name__ == "__main__":
    app()
