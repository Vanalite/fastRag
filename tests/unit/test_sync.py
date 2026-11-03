"""Connector sync: permissions follow the source, and deletions propagate."""

import json

import pytest

from fastrag import Principal, RAGService, Role, Settings
from fastrag.errors import ConfigurationError
from fastrag.ingest import AclRule, ManifestConnector, SyncService

ADMIN = Principal("acme", "sync-bot", roles=frozenset({Role.ADMIN}))
ALICE = Principal("acme", "alice", frozenset({"finance"}))
BOB = Principal("acme", "bob")


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "share"
    (root / "finance").mkdir(parents=True)
    (root / "finance" / "q3.md").write_text("# Q3\n\nQ3 revenue was 18 million.")
    (root / "handbook.md").write_text("# Handbook\n\nVacation is 25 days per year.")
    (root / "secret.txt").write_text("Unmapped file with no rule.")
    return root


RULES = [
    AclRule("finance/**", frozenset({"group:finance"})),
    AclRule("*.md", frozenset({"tenant:*"})),
]


def service() -> RAGService:
    return RAGService.from_settings(
        Settings.for_testing(rate_limit_per_minute=0, user_rate_limit_per_minute=0)
    )


def visible(svc, principal, query) -> set[str]:
    return {s.chunk.source_id for s in svc.search(principal, query).chunks}


def test_sync_maps_permissions_and_skips_unmapped_files(source):
    svc = service()
    report = SyncService(svc).sync(ADMIN, ManifestConnector("share", source, RULES))
    assert report.summary() == {
        "created": 2, "updated": 0, "unchanged": 0, "deleted": 0, "failed": 0, "skipped": 1,
        "not_owned": 0,
    }  # fmt: skip
    assert report.skipped == ["secret.txt"]  # deny by default
    assert "finance/q3.md" in visible(svc, ALICE, "Q3 revenue")
    assert "finance/q3.md" not in visible(svc, BOB, "Q3 revenue")


def test_resync_is_idempotent_and_permission_changes_propagate(source):
    svc = service()
    sync = SyncService(svc)
    sync.sync(ADMIN, ManifestConnector("share", source, RULES))
    again = sync.sync(ADMIN, ManifestConnector("share", source, RULES))
    assert again.summary()["unchanged"] == 2
    # The source opens the finance folder to everyone: Bob can now read it.
    opened = [AclRule("finance/**", frozenset({"tenant:*"})), RULES[1]]
    changed = sync.sync(ADMIN, ManifestConnector("share", source, opened))
    assert changed.updated == ["finance/q3.md"]
    assert "finance/q3.md" in visible(svc, BOB, "Q3 revenue")


def test_documents_deleted_at_the_source_are_pruned(source):
    svc = service()
    sync = SyncService(svc)
    sync.sync(ADMIN, ManifestConnector("share", source, RULES))
    (source / "finance" / "q3.md").unlink()
    report = sync.sync(ADMIN, ManifestConnector("share", source, RULES))
    assert report.deleted == ["finance/q3.md"]
    assert visible(svc, ALICE, "Q3 revenue") == set() or "finance/q3.md" not in visible(
        svc, ALICE, "Q3 revenue"
    )
    # Documents owned by another connector, or uploaded by hand, are never pruned.
    assert [d.source_id for d in svc.list_documents(ADMIN)] == ["handbook.md"]


def test_a_failing_source_never_triggers_pruning(source):
    svc = service()
    sync = SyncService(svc)
    sync.sync(ADMIN, ManifestConnector("share", source, RULES))

    class Outage:
        name = "share"

        def documents(self):
            yield from ()
            raise ConnectionError("token expired")

    report = sync.sync(ADMIN, Outage())
    assert report.deleted == [] and report.failed[0][0] == "<listing>"
    assert len(svc.list_documents(ADMIN)) == 2


def test_manifest_validation(tmp_path, source):
    good = tmp_path / "acl.json"
    good.write_text(json.dumps({"rules": [{"pattern": "**", "acl": ["tenant:*"]}]}))
    assert ManifestConnector.from_manifest("share", source, good).rules[0].acl == {"tenant:*"}
    for bad in (
        {"rules": [{"pattern": "**", "acl": []}]},
        {"rules": [{"pattern": "**", "acl": ["everyone"]}]},
        {},
    ):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(bad))
        with pytest.raises(ConfigurationError):
            ManifestConnector.from_manifest("share", source, path)


def test_reserved_metadata_keys_are_rejected_over_http():
    from fastapi.testclient import TestClient

    from fastrag.api import create_app
    from fastrag.security.auth import ApiKeyStore, Authenticator

    keys = ApiKeyStore()
    key = keys.create(ADMIN)
    client = TestClient(create_app(service(), Authenticator(api_keys=keys)))
    body = {"source_id": "x", "text": "x", "acl": ["tenant:*"], "metadata": {"_connector": "share"}}
    assert client.post("/v1/documents", headers={"X-API-Key": key}, json=body).status_code == 422


@pytest.mark.parametrize(
    ("pattern", "path", "expected"),
    [
        ("*.md", "handbook.md", True),
        ("*.md", "hr/payroll.md", False),  # "*" never crosses a folder
        ("**/*.md", "hr/payroll.md", True),
        ("**/*.md", "top.md", True),
        ("finance/**", "finance/q3/board.md", True),
        ("finance/**", "finance-old/q3.md", False),
        ("public/*.md", "public/hr/secret.md", False),
        ("**", "anything/at/all.txt", True),
    ],
)
def test_globs_are_folder_aware(pattern, path, expected):
    assert AclRule(pattern, frozenset({"tenant:*"})).matches(path) is expected


def test_rules_after_a_catch_all_are_rejected(tmp_path, source):
    manifest = tmp_path / "acl.json"
    manifest.write_text(
        json.dumps({"rules": [{"pattern": "**", "acl": ["tenant:*"]},
                              {"pattern": "hr/**", "acl": ["group:hr"]}]})
    )  # fmt: skip
    with pytest.raises(ConfigurationError):
        ManifestConnector.from_manifest("share", source, manifest)


def test_sync_does_not_adopt_documents_it_did_not_create(source):
    from fastrag import DocumentIn

    svc = service()
    svc.ingest(ADMIN, DocumentIn("handbook.md", "Hand-written handbook.", frozenset({"tenant:*"})))
    report = SyncService(svc).sync(ADMIN, ManifestConnector("share", source, RULES))
    assert report.adopt_refused == ["handbook.md"]
    (source / "handbook.md").unlink()
    SyncService(svc).sync(ADMIN, ManifestConnector("share", source, RULES))
    assert "handbook.md" in {d.source_id for d in svc.list_documents(ADMIN)}  # never pruned
    adopted = SyncService(svc).sync(
        ADMIN, ManifestConnector("share", source, RULES), adopt_existing=True
    )
    assert adopted.adopt_refused == []


def test_mass_prune_needs_force(tmp_path):
    root = tmp_path / "big"
    root.mkdir()
    for i in range(12):
        (root / f"doc{i}.md").write_text(f"Document {i} body.")
    svc = service()
    rules = [AclRule("*.md", frozenset({"tenant:*"}))]
    SyncService(svc).sync(ADMIN, ManifestConnector("big", root, rules))
    for path in list(root.iterdir())[:10]:
        path.unlink()  # e.g. a mount that came up half-empty
    report = SyncService(svc).sync(ADMIN, ManifestConnector("big", root, rules))
    assert report.deleted == [] and report.failed[0][0] == "<prune>"
    assert len(svc.list_documents(ADMIN)) == 12
    forced = SyncService(svc).sync(ADMIN, ManifestConnector("big", root, rules), force_prune=True)
    assert len(forced.deleted) == 10


def test_api_edits_keep_connector_ownership(source):
    from fastrag import DocumentIn

    svc = service()
    SyncService(svc).sync(ADMIN, ManifestConnector("share", source, RULES))
    # An admin edits a synced document through the API (which can't send "_" keys).
    svc.ingest(ADMIN, DocumentIn("handbook.md", "Edited by hand.", frozenset({"tenant:*"})))
    (record,) = [d for d in svc.list_documents(ADMIN) if d.source_id == "handbook.md"]
    assert record.metadata["_connector"] == "share"


def test_symlinked_folders_cannot_pull_in_outside_files(tmp_path):
    from fastrag.ingest import FilesystemConnector

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "passwords.md").write_text("root: hunter2")
    root = tmp_path / "share"
    root.mkdir()
    (root / "ok.md").write_text("Fine.")
    (root / "link").symlink_to(outside, target_is_directory=True)
    sources = [d.source_id for d in FilesystemConnector(root, frozenset({"tenant:*"})).documents()]
    assert sources == ["ok.md"]
