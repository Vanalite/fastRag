"""Per-collection ingestion settings, bulk import and settings comparison
against a real Postgres (D25)."""

import dataclasses
import uuid

import httpx
import pytest
from sqlalchemy import func, select

from factories import add_document, make_collection, make_source, make_tenant
from fastrag.api.app import create_app
from fastrag.api.common import collections_for
from fastrag.db.engine import get_sessionmaker, session_scope
from fastrag.db.models import Chunk, Collection, Document, Job
from fastrag.eval.compare import compare_settings, drop_trial
from fastrag.ingest.bulk import plan_import, run_import, scan_folder, wait_for_documents
from fastrag.ingest.documents import ingest_document, ingest_status
from fastrag.ingest.settings import resolve_ingest
from fastrag.worker.main import _reindex, run_one

pytestmark = pytest.mark.db

OP = {"Authorization": "Bearer test-operator-token", "X-Tenant": "acme"}


def long_doc(title: str, sections: int = 6, words: int = 120) -> str:
    parts = [f"# {title}"]
    for i in range(sections):
        body = " ".join(f"nội dung{i} câu số {j}." for j in range(words // 4))
        parts.append(f"## Mục {i}\n{body}")
    return "\n\n".join(parts)


class FakeLLM:
    name = "cheap"
    model = "fake-small"
    concurrency = 2
    input_cost_per_mtok = 1.0
    output_cost_per_mtok = 2.0

    def __init__(self) -> None:
        self.systems: list[str] = []

    async def complete(self, system: str, user: str, max_tokens: int = 200) -> str:
        self.systems.append(system)
        return f"Tóm tắt ({max_tokens})"

    async def aclose(self) -> None:
        return None


async def _chunks(document_id):
    async with get_sessionmaker()() as s:
        return list(
            (await s.execute(select(Chunk).where(Chunk.document_id == document_id).order_by(Chunk.ordinal))).scalars()
        )


async def test_collection_chunking_settings_apply(db, services):
    tid = await make_tenant()
    default = await make_collection(tid, "docs", "documents")
    small = await make_collection(
        tid, "faq", "documents", {"ingest": {"chunking": {"target_tokens": 60, "max_tokens": 90, "overlap_tokens": 0}}}
    )
    text = long_doc("Chính sách")
    d1, r1 = await add_document(services, await make_source(tid, default), "a.md", text)
    d2, r2 = await add_document(services, await make_source(tid, small), "a.md", text)
    assert r2["chunks"] > r1["chunks"]
    assert all(c.token_count <= 90 for c in await _chunks(d2))
    async with get_sessionmaker()() as s:
        doc = await s.get(Document, d2)
        col = await s.get(Collection, small)
        cfg = resolve_ingest(col.settings, services)
        assert doc.report["ingest_fingerprint"] == cfg.fingerprint
        assert doc.report["ingest"] == {"strategy": "structure", "context": "header", "llm": None}
        assert (await ingest_status(s, small, cfg.fingerprint))["current"] == 1


async def test_context_modes(db, services):
    tid = await make_tenant()
    none = await make_collection(tid, "bare", "documents", {"ingest": {"context": {"mode": "none"}}})
    doc_id, _ = await add_document(services, await make_source(tid, none), "a.md", long_doc("Tiêu đề", 1, 20))
    assert all(c.context is None for c in await _chunks(doc_id))

    llm = FakeLLM()
    with_llm = dataclasses.replace(services, llms={"cheap": llm})
    col = await make_collection(
        tid,
        "rich",
        "documents",
        {"ingest": {"context": {"llm": "cheap", "instructions": "Nêu tên dự án.", "summary_max_tokens": 60}}},
    )
    doc_id, _ = await add_document(with_llm, await make_source(tid, col), "a.md", long_doc("Dự án Sunrise", 2, 20))
    chunks = await _chunks(doc_id)
    assert chunks and all(c.context.startswith("Dự án Sunrise") and "Tóm tắt (60)" in c.context for c in chunks)
    assert llm.systems and all(sys.endswith("\nNêu tên dự án.") for sys in llm.systems)


async def test_settings_change_reprocesses_only_stale_documents(db, services):
    tid = await make_tenant()
    cid = await make_collection(tid, "docs", "documents")
    other = await make_collection(tid, "other", "documents")
    sid = await make_source(tid, cid)
    a, _ = await add_document(services, sid, "a.md", long_doc("A"))
    b, _ = await add_document(services, sid, "b.md", long_doc("B"))
    await add_document(services, await make_source(tid, other), "c.md", long_doc("C"))

    async with session_scope() as s:
        col = await s.get(Collection, cid)
        col.settings = {"ingest": {"chunking": {"strategy": "window"}}}
    async with get_sessionmaker()() as s:
        cfg = resolve_ingest((await s.get(Collection, cid)).settings, services)
        assert (await ingest_status(s, cid, cfg.fingerprint))["stale"] == 2

    # A re-sync of the unchanged file is no longer "unchanged".
    async with session_scope() as s:
        assert (await ingest_document(s, services, a))["status"] == "ready"
    async with session_scope() as s:
        assert (await ingest_document(s, services, a))["status"] == "unchanged"
        res = await _reindex(s, services, {"collection_id": str(cid), "only_stale": True})
    assert res == {"queued": 1}
    async with get_sessionmaker()() as s:
        jobs = (await s.execute(select(Job.payload))).scalars().all()
    assert [j["document_id"] for j in jobs] == [str(b)]


@pytest.fixture
async def client(db, services):
    app = create_app(services)
    app.state.services = services
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
        yield c


async def test_admin_api_validates_and_reindexes(client, services):
    r = await client.post("/v1/admin/tenants", json={"slug": "acme", "name": "ACME"}, headers=OP)
    assert r.status_code == 201
    bad = {"slug": "x", "name": "x", "type": "documents", "settings": {"ingest": {"chunking": {"max_tokens": 5}}}}
    r = await client.post("/v1/admin/collections", headers=OP, json=bad)
    assert r.status_code == 422 and "max_tokens" in r.text
    r = await client.post(
        "/v1/admin/collections",
        headers=OP,
        json={"slug": "l", "name": "l", "type": "listings", "settings": {"ingest": {}}},
    )
    assert r.status_code == 422 and "documents collections only" in r.text

    r = await client.post("/v1/admin/collections", headers=OP, json={"slug": "docs", "name": "d", "type": "documents"})
    assert r.status_code == 201
    r = await client.get("/v1/admin/collections/docs/ingest", headers=OP)
    assert r.status_code == 200 and r.json()["effective"]["strategy"] == "structure"
    fp = r.json()["effective"]["fingerprint"]

    # Renaming changes nothing to re-process; new chunking does.
    r = await client.patch("/v1/admin/collections/docs", headers=OP, json={"name": "Docs"})
    assert r.json()["reindex_job_id"] is None
    patch = {"settings": {"ingest": {"chunking": {"max_tokens": 300, "target_tokens": 200}}}}
    r = await client.patch("/v1/admin/collections/docs", headers=OP, json=patch)
    assert r.status_code == 200 and r.json()["reindex_job_id"] is not None
    r = await client.get("/v1/admin/collections/docs/ingest", headers=OP)
    assert r.json()["effective"]["max_tokens"] == 300 and r.json()["effective"]["fingerprint"] != fp
    r = await client.patch("/v1/admin/collections/docs", headers=OP, json={"settings": {"ingest": {"ocr": "maybe"}}})
    assert r.status_code == 422
    r = await client.post("/v1/admin/collections/docs/reindex?only_stale=true", headers=OP)
    assert r.status_code == 202


def _tree(root):
    (root / "public").mkdir()
    (root / "sales").mkdir()
    (root / ".git").mkdir()
    (root / "public" / "faq.md").write_text(long_doc("Hỏi đáp", 3), encoding="utf-8")
    (root / "public" / "gui-xe.txt").write_text("Phí gửi xe ô tô là 1,5 triệu đồng mỗi tháng.", encoding="utf-8")
    (root / "sales" / "hoa-hong.md").write_text(long_doc("Hoa hồng", 2), encoding="utf-8")
    (root / "sales" / "bang-gia.xlsx").write_bytes(b"not really a spreadsheet")
    (root / "notes.bin").write_bytes(b"\x00\x01")
    (root / ".git" / "config").write_text("x")
    (root / "~$faq.md").write_text("lock file")


async def test_scan_folder(tmp_path):
    _tree(tmp_path)
    entries, skipped = scan_folder(tmp_path, prefix="import/")
    assert [e.rel for e in entries] == [
        "import/public/faq.md",
        "import/public/gui-xe.txt",
        "import/sales/bang-gia.xlsx",
        "import/sales/hoa-hong.md",
    ]
    assert entries[0].folder == "import/public"
    assert [(i.rel, i.action) for i in skipped] == [("import/notes.bin", "skipped")]
    only_md, _ = scan_folder(tmp_path, include=["*.md"], exclude=["sales/*"])
    assert [e.rel for e in only_md] == ["public/faq.md"]
    flat, _ = scan_folder(tmp_path / "public", recursive=False)
    assert len(flat) == 2


async def test_import_dry_run_then_inline_then_resume(db, services, tmp_path):
    _tree(tmp_path)
    tid = await make_tenant()
    cid = await make_collection(tid, "docs", "documents")
    sid = await make_source(tid, cid, groups=["staff"], folder_groups={"public": ["public"]})
    entries, skipped = scan_folder(tmp_path)
    sm = get_sessionmaker()

    llm = FakeLLM()
    with_llm = dataclasses.replace(services, llms={"cheap": llm})
    async with session_scope() as s:
        (await s.get(Collection, cid)).settings = {"ingest": {"context": {"llm": "cheap"}}}

    plan = await plan_import(sm, with_llm, sid, entries, skipped=skipped)
    by = {i.rel: i for i in plan.items}
    assert by["public/faq.md"].action == "new" and by["public/faq.md"].estimate["chunks"] >= 1
    assert by["sales/bang-gia.xlsx"].action == "skipped" and "listings" in by["sales/bang-gia.xlsx"].reason
    assert by["notes.bin"].action == "skipped"
    est = plan.totals["estimate"]
    assert est["llm_calls"] == est["chunks"] and est["llm_output_tokens_max"] == 120 * est["chunks"]
    assert est["llm"] == "cheap (fake-small)" and est["llm_cost_usd_max"] > 0
    assert not llm.systems  # the dry run never calls the LLM
    async with sm() as s:
        assert await s.scalar(select(func.count()).select_from(Document)) == 0

    rep = await run_import(sm, with_llm, sid, entries, skipped=skipped, inline=True, concurrency=2)
    assert rep.totals["by_status"] == {"ready": 3}
    assert rep.totals["chunks"] == sum(i.chunks or 0 for i in rep.items) > 0
    async with sm() as s:
        groups = dict((await s.execute(select(Document.external_id, Document.groups))).all())
    assert groups == {"public/faq.md": ["public"], "public/gui-xe.txt": ["public"], "sales/hoa-hong.md": ["staff"]}

    # Re-running continues: nothing changed.
    again = await run_import(sm, with_llm, sid, entries, inline=True)
    assert again.totals["by_action"] == {"unchanged": 3, "skipped": 1}
    (tmp_path / "public" / "gui-xe.txt").write_text("Phí gửi xe ô tô là 2 triệu đồng mỗi tháng.", encoding="utf-8")
    changed = await run_import(sm, with_llm, sid, scan_folder(tmp_path)[0], inline=True)
    assert changed.totals["by_action"] == {"unchanged": 2, "changed": 1, "skipped": 1}

    # New settings: unchanged files built with the old ones are re-processed.
    async with session_scope() as s:
        (await s.get(Collection, cid)).settings = {"ingest": {"context": {"mode": "header"}}}
    plan = await plan_import(sm, with_llm, sid, scan_folder(tmp_path)[0], estimate=False)
    assert plan.totals["by_action"] == {"stale": 3, "skipped": 1}
    stale = await run_import(sm, with_llm, sid, scan_folder(tmp_path)[0], inline=True)
    assert stale.totals["by_action"] == {"stale": 3, "skipped": 1} and stale.totals["by_status"] == {"ready": 3}
    current = await run_import(sm, with_llm, sid, scan_folder(tmp_path)[0], inline=True)
    assert current.totals["by_action"] == {"unchanged": 3, "skipped": 1}


async def test_import_queued_with_worker(db, services, tmp_path):
    _tree(tmp_path)
    tid = await make_tenant()
    cid = await make_collection(tid, "docs", "documents")
    sid = await make_source(tid, cid)
    sm = get_sessionmaker()
    rep = await run_import(sm, services, sid, scan_folder(tmp_path)[0])
    assert rep.mode == "queued" and rep.totals["by_status"] == {"pending": 3}
    assert not await wait_for_documents(sm, rep, poll_s=0.05, timeout_s=0.2)
    while await run_one(services, "test-worker"):
        pass
    assert await wait_for_documents(sm, rep, poll_s=0.05, timeout_s=5)
    assert rep.totals["by_status"] == {"ready": 3} and rep.totals["chunks"] > 0


async def test_import_rejects_unknown_source(db, services, tmp_path):
    _tree(tmp_path)
    with pytest.raises(LookupError):
        await run_import(get_sessionmaker(), services, uuid.uuid4(), scan_folder(tmp_path)[0])


async def test_compare_settings_builds_and_drops_a_hidden_trial(db, services):
    tid = await make_tenant()
    cid = await make_collection(tid, "docs", "documents")
    sid = await make_source(tid, cid, groups=["staff"], folder_groups={"public": ["public"]})
    await add_document(services, sid, "faq.md", long_doc("Phí quản lý", 4), path="public")
    await add_document(services, sid, "hoa-hong.md", long_doc("Hoa hồng", 3), path="sales")
    refs = [
        {"id": "q1", "query": "nội dung1 câu số 3", "relevant": [{"document": "Phí quản lý"}]},
        {
            "id": "q2",
            "query": "nội dung2",
            "collections": ["docs"],
            "profile": "internal_search",
            "relevant": [{"document": "Hoa hồng"}],
        },
        {"id": "l1", "type": "listings", "filters": {}},
    ]
    sm = get_sessionmaker()
    overrides = {"chunking": {"target_tokens": 60, "max_tokens": 80, "overlap_tokens": 0}}

    out = await compare_settings(sm, services, tid, "docs", refs, overrides, keep=True)
    assert out["base"]["metrics"]["questions"] == out["trial"]["metrics"]["questions"] == 2
    assert out["trial"]["chunks"] > out["base"]["chunks"]
    assert out["trial"]["settings"]["max_tokens"] == 80
    assert out["trial"]["metrics"]["access_violations"] == 0
    assert set(out["delta"]) == {"recall_at_k", "mrr"}

    trial_slug = out["trial"]["collection"]
    async with sm() as s:
        trial = (await s.execute(select(Collection).where(Collection.slug == trial_slug))).scalar_one()
        # Live searches without a collection name never see the trial.
        assert [c.slug for c in await collections_for(s, tid, None, "documents")] == ["docs"]
        assert [c.slug for c in await collections_for(s, tid, [trial_slug], "documents")] == [trial_slug]
        keys = select(Document.collection_id, Document.storage_key)
        rows = (await s.execute(keys.where(Document.collection_id.in_([trial.id, cid])))).all()
    storage = {key for col, key in rows if col == trial.id}
    assert storage == {key for col, key in rows if col == cid}  # files are shared, not copied

    assert await drop_trial(sm, trial.id)
    assert not await drop_trial(sm, cid)  # never drops a real collection
    async with sm() as s:
        assert await s.scalar(select(func.count()).select_from(Collection)) == 1
        assert await s.scalar(select(func.count()).where(Document.collection_id == cid)) == 2

    # Default: the trial is dropped automatically.
    await compare_settings(sm, services, tid, "docs", refs, overrides)
    async with sm() as s:
        assert await s.scalar(select(func.count()).select_from(Collection)) == 1

    with pytest.raises(ValueError, match="no document reference questions"):
        await compare_settings(sm, services, tid, "docs", [refs[2]], overrides)
