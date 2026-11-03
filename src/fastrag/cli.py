"""Operator CLI.

``serve`` runs the HTTP API. The other commands act directly on the local store as an
operator (they bypass HTTP authentication, so only run them where you already have
access to the data files) - but they still go through ``RAGService``, so every action is
authorized against the principal you name and recorded in the audit log.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from fastrag.config import Settings
from fastrag.domain import EVERYONE, Principal, Role
from fastrag.errors import ConfigurationError, FastRagError, InvalidInput


def _principal_args(p: argparse.ArgumentParser, user_required: bool = True) -> None:
    p.add_argument("--tenant", required=True)
    p.add_argument("--user", required=user_required, default="cli-operator")
    p.add_argument("--group", action="append", default=[], help="repeatable")
    p.add_argument("--role", action="append", choices=[r.value for r in Role], help="repeatable")


def _principal(args: argparse.Namespace, default_role: Role = Role.READER) -> Principal:
    roles = frozenset(Role(r) for r in args.role) if args.role else frozenset({default_role})
    return Principal(
        tenant_id=args.tenant, user_id=args.user, groups=frozenset(args.group), roles=roles
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fastrag", description="Enterprise RAG reference implementation."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("serve", help="run the HTTP API")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)

    p = sub.add_parser("ingest", help="ingest files or directories into a tenant")
    p.add_argument("paths", nargs="+", type=Path)
    p.add_argument("--tenant", required=True)
    p.add_argument(
        "--acl",
        action="append",
        help=f"who may read the documents (repeatable); default {EVERYONE}",
    )

    for name, help_text in (("ask", "answer a question"), ("search", "show matching chunks")):
        p = sub.add_parser(name, help=f"{help_text} as a given principal")
        p.add_argument("question")
        _principal_args(p)
        p.add_argument("-k", type=int)

    p = sub.add_parser("documents", help="list documents visible to a principal")
    _principal_args(p, user_required=False)

    p = sub.add_parser(
        "sync",
        help="sync a folder into a tenant, permissions from a manifest (adds, updates, prunes)",
    )
    p.add_argument("root", type=Path)
    p.add_argument("--tenant", required=True)
    p.add_argument("--manifest", type=Path, required=True, help="JSON ACL rules (see docs)")
    p.add_argument("--name", help="connector name (default: the folder name)")
    p.add_argument("--no-prune", action="store_true", help="keep documents gone from the source")
    p.add_argument(
        "--force-prune",
        action="store_true",
        help="allow deleting most of the connector's documents",
    )
    p.add_argument(
        "--adopt", action="store_true", help="take over existing documents it didn't create"
    )

    p = sub.add_parser("delete", help="delete a document by id")
    p.add_argument("doc_id")
    p.add_argument("--tenant", required=True)

    p = sub.add_parser(
        "seed-demo", help="load the Northwind demo corpus, with its per-document ACLs"
    )
    p.add_argument("--dataset", type=Path, default=Path("eval/golden"))

    keys = sub.add_parser("keys", help="manage API keys").add_subparsers(
        dest="keys_command", required=True
    )
    p = keys.add_parser("create", help="create an API key (printed once)")
    _principal_args(p)
    p.add_argument("--ttl-days", type=float, help="expire the key after this many days")
    p = keys.add_parser("list", help="list API keys (ids and owners, never the keys)")
    p.add_argument("--tenant", help="only this tenant")
    p = keys.add_parser("revoke", help="revoke an API key by id (takes effect immediately)")
    p.add_argument("key_id")

    p = sub.add_parser("token", help="mint a demo JWT (needs FASTRAG_JWT_SECRET)")
    _principal_args(p)
    p.add_argument("--ttl", type=int, default=3600, help="seconds")

    audit = sub.add_parser("audit", help="audit log tools").add_subparsers(
        dest="audit_command", required=True
    )
    audit.add_parser("verify", help="verify the audit log hash chain")

    db = sub.add_parser("db", help="database administration").add_subparsers(
        dest="db_command", required=True
    )
    db.add_parser("migrate", help="apply pending schema migrations, then exit")

    sub.add_parser(
        "eval", help="run the retrieval evaluation gate (see `fastrag eval --help`)", add_help=False
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["eval"]:
        from fastrag.eval import cli_main

        return cli_main(argv[1:])

    args = build_parser().parse_args(argv)
    try:
        settings = Settings()
        return _dispatch(args, settings)
    except (FastRagError, ConfigurationError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _dispatch(args: argparse.Namespace, settings: Settings) -> int:
    if args.command == "serve":
        import uvicorn

        from fastrag.api import create_app_from_settings

        uvicorn.run(create_app_from_settings(settings), host=args.host, port=args.port)
        return 0

    if args.command == "keys":
        from fastrag.security.auth import ApiKeyStore, hash_api_key, key_id

        if settings.api_keys_file is None:
            raise ConfigurationError("FASTRAG_API_KEYS_FILE is not set")
        store = ApiKeyStore(settings.api_keys_file)
        if args.keys_command == "create":
            ttl = args.ttl_days * 86400 if args.ttl_days is not None else None
            try:
                key = store.create(_principal(args), ttl_seconds=ttl)
            except ValueError as exc:
                raise InvalidInput(str(exc)) from exc
            print(key)
            print(
                f"Stored a hash of this key (id {key_id(hash_api_key(key))}) in "
                f"{settings.api_keys_file}. It won't be shown again.",
                file=sys.stderr,
            )
        elif args.keys_command == "list":
            now = time.time()
            for r in store.records():
                if args.tenant and r.principal.tenant_id != args.tenant:
                    continue
                state = "revoked" if r.revoked_at else "expired" if not r.active(now) else "active"
                roles = ",".join(sorted(role.value for role in r.principal.roles))
                groups = ",".join(sorted(r.principal.groups)) or "-"
                print(
                    f"{r.key_id}  {state:8} {r.principal.tenant_id}/{r.principal.user_id}"
                    f"  roles={roles} groups={groups}"
                )
        else:
            try:
                revoked = store.revoke(args.key_id)
            except ValueError as exc:
                raise InvalidInput(str(exc)) from exc
            print(f"revoked {revoked.key_id} ({revoked.principal.user_id})")
        return 0

    if args.command == "token":
        from fastrag.security.auth import ClaimMapping, JwtVerifier

        if settings.jwt_secret is None:
            raise ConfigurationError("FASTRAG_JWT_SECRET is not set")
        verifier = JwtVerifier(
            settings.jwt_secret.get_secret_value(),
            settings.jwt_issuer,
            settings.jwt_audience,
            claims=ClaimMapping(
                tenant=settings.token_tenant_claim,
                groups=settings.token_groups_claim,
                roles=settings.token_roles_claim,
            ),
        )
        print(verifier.issue(_principal(args), ttl_seconds=args.ttl))
        return 0

    if args.command == "db":
        from fastrag.service import migrate_database

        print(migrate_database(settings))
        return 0

    if args.command == "audit":
        from fastrag.observability import verify_chain
        from fastrag.observability.audit import JsonlAuditSink

        if settings.audit_log_path is None:
            raise ConfigurationError("FASTRAG_AUDIT_LOG_PATH is not set")
        entries = JsonlAuditSink(settings.audit_log_path).read()
        ok = verify_chain(entries)
        print(f"{len(entries)} entries; chain {'OK' if ok else 'BROKEN'}")
        return 0 if ok else 1

    # The CLI acts as a local operator on files it can already read, so the HTTP rate
    # limits (meant for remote callers) don't apply: a bulk ingest must not stop at the
    # burst size.
    from fastrag.ratelimit import RateLimiter
    from fastrag.service import RAGService

    unlimited = RateLimiter(0, 0)
    service = RAGService.from_settings(settings, limiter=unlimited, user_limiter=unlimited)

    if args.command == "ingest":
        from fastrag.ingest import FilesystemConnector

        operator = Principal(args.tenant, "cli-operator", roles=frozenset({Role.ADMIN}))
        acl = frozenset(args.acl or [EVERYONE])
        counts: dict[str, int] = {}
        for path in args.paths:
            for doc in FilesystemConnector(path, acl).documents():
                result = service.ingest(operator, doc)
                counts[result.status] = counts.get(result.status, 0) + 1
                status = f"{result.status:9} v{result.version}"
                print(f"{status}  {doc.source_id}  ({result.chunks} chunks)")
        print(json.dumps(counts))
        return 0

    if args.command == "seed-demo":
        from fastrag.domain import DocumentIn
        from fastrag.eval.runner import load_dataset

        dataset = load_dataset(args.dataset)
        operator = Principal(dataset.tenant_id, "cli-operator", roles=frozenset({Role.ADMIN}))
        for d in dataset.docs:
            result = service.ingest(
                operator,
                DocumentIn(source_id=d.source_id, text=d.text, acl=d.acl, media_type=d.media_type),
            )
            print(f"{result.status:9} {d.source_id:32} acl={sorted(d.acl)}")
        print(f"tenant: {dataset.tenant_id}")
        return 0

    if args.command == "sync":
        from fastrag.ingest import ManifestConnector, SyncService

        operator = Principal(args.tenant, "cli-operator", roles=frozenset({Role.ADMIN}))
        connector = ManifestConnector.from_manifest(
            args.name or args.root.resolve().name, args.root, args.manifest
        )
        report = SyncService(service).sync(
            operator,
            connector,
            prune=not args.no_prune,
            force_prune=args.force_prune,
            adopt_existing=args.adopt,
        )
        for source_id, error in report.failed:
            print(f"failed   {source_id}: {error}", file=sys.stderr)
        for source_id in report.skipped:
            print(f"skipped  {source_id} (no ACL rule matches)", file=sys.stderr)
        for source_id in report.adopt_refused:
            print(f"skipped  {source_id} (exists, not created by this connector)", file=sys.stderr)
        print(json.dumps(report.summary()))
        return 1 if report.failed else 0

    if args.command == "delete":
        operator = Principal(args.tenant, "cli-operator", roles=frozenset({Role.ADMIN}))
        service.delete(operator, args.doc_id)
        print(f"deleted {args.doc_id}")
        return 0

    if args.command == "documents":
        for d in service.list_documents(_principal(args)):
            print(f"{d.doc_id}  v{d.version}  {d.source_id}  acl={sorted(d.acl)}")
        return 0

    principal = _principal(args)
    if args.command == "search":
        result = service.search(principal, args.question, args.k)
        print(f"confident: {result.confident}")
        for i, s in enumerate(result.chunks, start=1):
            scores = f"rrf={s.score:.4f} dense={s.dense_score} bm25={s.keyword_score}"
            print(f"\n[{i}] {s.chunk.source_id}  {scores}")
            print(s.chunk.text)
        return 0

    answer = service.ask(principal, args.question, args.k)
    print(answer.text)
    if answer.citations:
        print("\nSources:")
        for c in answer.citations:
            print(f"  [{c.index}] {c.source_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
