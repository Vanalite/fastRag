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
from pathlib import Path

from fastrag.config import Settings
from fastrag.domain import EVERYONE, Principal, Role
from fastrag.errors import ConfigurationError, FastRagError


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

    p = sub.add_parser("token", help="mint a demo JWT (needs FASTRAG_JWT_SECRET)")
    _principal_args(p)
    p.add_argument("--ttl", type=int, default=3600, help="seconds")

    audit = sub.add_parser("audit", help="audit log tools").add_subparsers(
        dest="audit_command", required=True
    )
    audit.add_parser("verify", help="verify the audit log hash chain")

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
        from fastrag.security.auth import ApiKeyStore

        if settings.api_keys_file is None:
            raise ConfigurationError("FASTRAG_API_KEYS_FILE is not set")
        key = ApiKeyStore(settings.api_keys_file).create(_principal(args))
        print(key)
        print(
            f"Stored a hash of this key in {settings.api_keys_file}. It won't be shown again.",
            file=sys.stderr,
        )
        return 0

    if args.command == "token":
        from fastrag.security.auth import JwtVerifier

        if settings.jwt_secret is None:
            raise ConfigurationError("FASTRAG_JWT_SECRET is not set")
        verifier = JwtVerifier(
            settings.jwt_secret.get_secret_value(), settings.jwt_issuer, settings.jwt_audience
        )
        print(verifier.issue(_principal(args), ttl_seconds=args.ttl))
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

    from fastrag.service import RAGService

    service = RAGService.from_settings(settings)

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
