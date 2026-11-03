"""Connector sync: keep a tenant's corpus - and its permissions - in step with a source.

The rule enterprise RAG lives by is **permissions follow the source**. A connector maps
each source document's permissions (SharePoint groups, Confluence space restrictions, a
Drive share list, file-system groups) onto ACL tokens. ``SyncService.sync`` then:

* ingests new documents and new versions of changed ones;
* re-stamps every chunk when only permissions changed (the content hash covers the ACL);
* deletes documents that disappeared from the source (``prune``), so access revoked by
  deleting a file is revoked here too.

Every action goes through ``RAGService`` as the syncing principal, so it is authorized
and audited like any other write. Synced documents carry ``_connector`` in their
metadata, which is how pruning knows which documents a connector owns.

This module ships ``ManifestConnector`` (a directory plus a permissions manifest) as a
working reference. Connectors for real systems implement the same two members.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from fastrag.domain import DocumentIn, Principal, document_id, is_valid_acl_token
from fastrag.errors import ConfigurationError, FastRagError
from fastrag.ingest.connectors import FilesystemConnector

if TYPE_CHECKING:
    from fastrag.service import RAGService

log = logging.getLogger(__name__)

CONNECTOR_KEY = "_connector"


class Connector(Protocol):
    name: str  # stable and unique per tenant: it decides which documents a sync owns

    def documents(self) -> Iterator[DocumentIn]:
        """Every document currently in the source, each with its mapped ACL."""
        ...


@dataclass(frozen=True)
class AclRule:
    """``pattern`` is a path glob on the source_id: ``*`` and ``?`` stay within one
    folder, ``**`` spans any number of folders. ``*.md`` matches only top-level files;
    ``finance/**`` matches everything under ``finance/``."""

    pattern: str
    acl: frozenset[str]
    editors: frozenset[str] = frozenset()

    def matches(self, source_id: str) -> bool:
        return _glob_regex(self.pattern).fullmatch(source_id) is not None


def _glob_regex(pattern: str) -> re.Pattern[str]:
    # fnmatch's "*" also crosses "/", so "*.md" would silently match "hr/payroll.md" and
    # widen access to a whole subtree. Translate with folder-aware semantics instead.
    out, i = [], 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif pattern[i] == "*":
            out.append("[^/]*")
            i += 1
        elif pattern[i] == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(pattern[i]))
            i += 1
    return re.compile("".join(out))


class ManifestConnector:
    """Files under ``root``, with permissions from a manifest.

    Manifest (JSON): ``{"rules": [{"pattern": "finance/**", "acl": ["group:finance"],
    "editors": ["group:finance-leads"]}, {"pattern": "**", "acl": ["tenant:*"]}]}``.
    The first matching rule wins. Files no rule matches are **skipped** (deny by
    default): an unmapped document must never become readable by accident.
    """

    def __init__(self, name: str, root: str | Path, rules: list[AclRule]):
        if not name:
            raise ConfigurationError("a connector needs a name")
        self.name = name
        self.root = Path(root)
        self.rules = rules
        self.skipped: list[str] = []

    @classmethod
    def from_manifest(cls, name: str, root: str | Path, manifest: str | Path) -> ManifestConnector:
        try:
            data = json.loads(Path(manifest).read_text())
            rules = [
                AclRule(
                    pattern=str(r["pattern"]),
                    acl=frozenset(r["acl"]),
                    editors=frozenset(r.get("editors", [])),
                )
                for r in data["rules"]
            ]
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ConfigurationError(f"invalid ACL manifest {manifest}: {exc}") from exc
        bad = sorted(t for r in rules for t in r.acl | r.editors if not is_valid_acl_token(t))
        if bad or any(not r.acl for r in rules):
            raise ConfigurationError(f"manifest {manifest} has empty ACLs or bad tokens {bad}")
        catch_all = next((i for i, r in enumerate(rules) if r.pattern == "**"), None)
        if catch_all is not None and catch_all != len(rules) - 1:
            # First match wins, so anything after "**" is dead - and usually was meant
            # to be narrower than the catch-all that now shadows it.
            raise ConfigurationError(
                f"manifest {manifest}: rules after the catch-all '**' can never match"
            )
        return cls(name, root, rules)

    def rule_for(self, source_id: str) -> AclRule | None:
        return next((rule for rule in self.rules if rule.matches(source_id)), None)

    def documents(self) -> Iterator[DocumentIn]:
        self.skipped = []
        # The filesystem connector does the walking and parsing; ACLs come from rules.
        for doc in FilesystemConnector(self.root, acl=frozenset({"tenant:*"})).documents():
            rule = self.rule_for(doc.source_id)
            if rule is None:
                self.skipped.append(doc.source_id)
                log.warning("no ACL rule matches %s; skipped (deny by default)", doc.source_id)
                continue
            yield replace(doc, acl=rule.acl, editors=rule.editors)


@dataclass
class SyncReport:
    connector: str
    created: list[str] = field(default_factory=list)
    adopt_refused: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, int]:
        return {
            "created": len(self.created),
            "updated": len(self.updated),
            "unchanged": len(self.unchanged),
            "deleted": len(self.deleted),
            "failed": len(self.failed),
            "skipped": len(self.skipped),
            "not_owned": len(self.adopt_refused),
        }


class SyncService:
    # Refuse a prune that would delete more than this share of a connector's documents
    # (when it owns at least MIN_PRUNE_GUARD of them) unless forced: an empty or
    # truncated listing from a misconfigured source looks exactly like a mass deletion.
    MAX_PRUNE_FRACTION = 0.5
    MIN_PRUNE_GUARD = 10

    def __init__(self, service: RAGService):
        self.service = service

    def sync(
        self,
        principal: Principal,
        connector: Connector,
        *,
        prune: bool = True,
        force_prune: bool = False,
        adopt_existing: bool = False,
    ) -> SyncReport:
        """Bring the tenant in step with ``connector``.

        ``adopt_existing``: take over documents with the same source_id that this
        connector didn't create (e.g. uploaded by hand). Off by default, because adopting
        a document makes it prunable.
        """
        report = SyncReport(connector=connector.name)
        seen: set[str] = set()
        listing_ok = True
        try:
            for doc in connector.documents():
                seen.add(doc.source_id)
                existing = self.service.store.get_document(
                    principal.tenant_id, document_id(principal.tenant_id, doc.source_id)
                )
                owner = existing.metadata.get(CONNECTOR_KEY) if existing else None
                if existing is not None and owner != connector.name and not adopt_existing:
                    report.adopt_refused.append(doc.source_id)
                    continue
                tagged = replace(doc, metadata={**doc.metadata, CONNECTOR_KEY: connector.name})
                try:
                    result = self.service.ingest(principal, tagged)
                except FastRagError as exc:
                    report.failed.append((doc.source_id, f"{type(exc).__name__}: {exc}"))
                    continue
                getattr(report, result.status).append(doc.source_id)
        except Exception as exc:  # noqa: BLE001 - a broken source must not trigger pruning
            listing_ok = False
            log.exception("connector %s failed while listing documents", connector.name)
            report.failed.append(("<listing>", f"{type(exc).__name__}: {exc}"))
        report.skipped = list(getattr(connector, "skipped", []))

        # Prune only after a *complete* listing: a source that failed halfway (an outage,
        # an expired token) would otherwise look like a mass deletion.
        if prune and listing_ok:
            owned = [
                r
                for r in self.service.list_documents(principal)
                if r.metadata.get(CONNECTOR_KEY) == connector.name
            ]
            doomed = [r for r in owned if r.source_id not in seen]
            if (
                not force_prune
                and len(owned) >= self.MIN_PRUNE_GUARD
                and len(doomed) > self.MAX_PRUNE_FRACTION * len(owned)
            ):
                report.failed.append(
                    (
                        "<prune>",
                        f"refusing to delete {len(doomed)} of {len(owned)} documents; "
                        "check the source, then re-run with force_prune",
                    )
                )
                return report
            for record in doomed:
                try:
                    self.service.delete(principal, record.doc_id)
                    report.deleted.append(record.source_id)
                except FastRagError as exc:
                    report.failed.append((record.source_id, f"{type(exc).__name__}: {exc}"))
        return report
