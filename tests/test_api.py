"""End-to-end through HTTP: admin setup, upload, worker, retrieval."""

import httpx
import pytest

from factories import xlsx
from fastrag.api.app import create_app
from fastrag.worker.main import run_worker

pytestmark = pytest.mark.db
OP = {"Authorization": "Bearer test-operator-token", "X-Tenant": "acme"}

FAQ = "# FAQ\n\n## Phí quản lý\nPhí quản lý là 15.000 đồng/m2/tháng.\n\n## Gửi xe\nPhí gửi ô tô 1,5 triệu/tháng."
SALES = "# Nội bộ\n\n## Hoa hồng\nHoa hồng cho sale là 2% giá trị hợp đồng."


@pytest.fixture
async def client(db, services):
    app = create_app(services)
    app.state.services = services
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def _bootstrap(client, services):
    r = await client.post("/v1/admin/tenants", json={"slug": "acme", "name": "ACME"}, headers=OP)
    assert r.status_code == 201, r.text
    keys = {}
    for name, groups, profiles, admin in (
        ("customer", ["public"], ["customer"], False),
        ("staff", ["public", "staff"], None, False),
        ("admin", ["public", "staff"], None, True),
    ):
        r = await client.post(
            "/v1/admin/keys",
            headers=OP,
            json={"name": name, "allowed_groups": groups, "allowed_profiles": profiles, "is_admin": admin},
        )
        assert r.status_code == 201, r.text
        keys[name] = {"Authorization": f"Bearer {r.json()['token']}"}
    admin = keys["admin"]
    for slug, ctype in (("docs", "documents"), ("listings", "listings")):
        r = await client.post("/v1/admin/collections", headers=admin, json={"slug": slug, "name": slug, "type": ctype})
        assert r.status_code == 201, r.text
    r = await client.post(
        "/v1/admin/sources",
        headers=admin,
        json={
            "collection": "docs",
            "kind": "upload",
            "name": "docs",
            "groups": ["staff"],
            "config": {"folder_groups": {"public": ["public"]}},
        },
    )
    docs_src = r.json()
    r = await client.post(
        "/v1/admin/sources",
        headers=admin,
        json={"collection": "listings", "kind": "upload", "name": "sheets", "groups": ["public"]},
    )
    sheet_src = r.json()
    for fname, body, path in (("faq.md", FAQ, "public"), ("sales.md", SALES, "sales")):
        r = await client.post(
            f"/v1/admin/sources/{docs_src['id']}/files",
            headers=admin,
            files={"file": (fname, body.encode(), "text/markdown")},
            data={"path": path},
        )
        assert r.status_code == 202, r.text
    sheet = xlsx([["Mã", "Giá", "Quận", "Hoa hồng"], ["A1", "3 tỷ", "Q7", "2%"], ["A2", "5 tỷ", "Q2", "1%"]])
    r = await client.post(
        f"/v1/admin/sources/{sheet_src['id']}/files",
        headers=admin,
        files={"file": ("gia.xlsx", sheet, "application/octet-stream")},
    )
    assert r.status_code == 202, r.text
    await run_worker(services, 1, once=True)
    return keys, docs_src, sheet_src


async def test_full_flow(client, services):
    keys, docs_src, sheet_src = await _bootstrap(client, services)
    admin = keys["admin"]

    stats = (await client.get("/v1/admin/stats", headers=admin)).json()
    assert stats["documents"] == {"ready": 2, "imported": 1} and stats["listings"] == 2

    # Customer key: public only, even when it asks for staff content.
    r = await client.post(
        "/v1/search",
        headers=keys["customer"] | {"X-Trace-Id": "trace-123"},
        json={"query": "hoa hồng cho sale", "groups": ["public", "staff"]},
    )
    assert r.status_code == 200 and r.headers["x-trace-id"] == "trace-123"
    body = r.json()
    assert body["schema_version"] == "1.0" and body["groups"] == ["public"]
    assert all("Hoa hồng" not in p["text"] for p in body["passages"])

    r = await client.post("/v1/search", headers=keys["customer"], json={"query": "phí gửi ô tô"})
    top = r.json()["passages"][0]
    assert "gửi ô tô" in top["text"] and top["source"]["title"] == "FAQ" and top["source"]["collection"] == "docs"

    # Customer key may not use staff profiles.
    r = await client.post("/v1/search", headers=keys["customer"], json={"query": "x", "profile": "internal_search"})
    assert r.status_code == 403

    r = await client.post("/v1/search", headers=keys["staff"], json={"query": "hoa hồng", "profile": "internal_search"})
    assert "Hoa hồng" in r.json()["passages"][0]["text"]

    # Listings: columns are internal until the mapping is confirmed.
    r = await client.post("/v1/listings/query", headers=keys["customer"], json={"filters": {"district": "Q7"}})
    assert r.status_code == 200 and r.json()["total"] == 0
    mappings = (await client.get("/v1/admin/mappings", headers=admin, params={"collection": "listings"})).json()
    m = mappings[0]
    cols = {
        h: {"field": e["field"], "groups": ["public", "staff"] if h != "Hoa hồng" else ["staff"]}
        for h, e in m["columns"].items()
    }
    r = await client.put(f"/v1/admin/mappings/{m['id']}", headers=admin, json={"columns": cols})
    assert r.status_code == 200 and r.json()["reimport_queued"] == 1, r.text
    await run_worker(services, 1, once=True)

    r = await client.post(
        "/v1/listings/query", headers=keys["customer"], json={"filters": {"district": "Quận 7", "price_max": "3,5 tỷ"}}
    )
    data = r.json()
    assert data["total"] == 1 and data["items"][0]["listing_code"] == "A1"
    assert data["items"][0]["price_vnd"] == 3_000_000_000 and "attributes" not in data["items"][0]
    assert data["applied_filters"] == {"district": ["hcm-q7"], "price_max": 3_500_000_000} and data["as_of"]
    listing_id = data["items"][0]["id"]

    r = await client.post("/v1/listings/lookup", headers=keys["customer"], json={"ids": [listing_id]})
    assert r.json()["items"][0]["listing_code"] == "A1"

    r = await client.post("/v1/listings/query", headers=keys["customer"], json={"filters": {"price_max": "rẻ"}})
    assert r.status_code == 422 and r.json()["error"] == "invalid_filters"
    assert r.json()["rejected"][0]["filter"] == "price_max"

    # Feedback lands on the search log.
    r = await client.post(
        "/v1/feedback", headers=keys["customer"], json={"trace_id": "trace-123", "rating": "down", "comment": "sai"}
    )
    assert r.status_code == 204
    logs = (await client.get("/v1/admin/search-logs", headers=admin, params={"feedback": "down"})).json()
    assert len(logs) == 1 and logs[0]["request"]["query"] == "hoa hồng cho sale"


async def test_auth_errors(client, services):
    await _bootstrap(client, services)
    assert (await client.post("/v1/search", json={"query": "x"})).status_code == 401
    r = await client.post("/v1/search", headers={"Authorization": "Bearer fr_bad_token"}, json={"query": "x"})
    assert r.status_code == 401
    # Operator token without a tenant.
    r = await client.post("/v1/search", headers={"Authorization": "Bearer test-operator-token"}, json={"query": "x"})
    assert r.status_code == 400
    r = await client.post("/v1/search", headers=OP, json={"query": "", "profile": "customer"})
    assert r.status_code == 422 and r.json()["error"] == "invalid_request"


async def test_key_revocation_and_admin_only(client, services):
    keys, *_ = await _bootstrap(client, services)
    r = await client.get("/v1/admin/stats", headers=keys["staff"])
    assert r.status_code == 403
    key_list = (await client.get("/v1/admin/keys", headers=keys["admin"])).json()
    staff_id = next(k["id"] for k in key_list if k["name"] == "staff")
    assert (await client.delete(f"/v1/admin/keys/{staff_id}", headers=keys["admin"])).status_code == 204
    r = await client.post("/v1/search", headers=keys["staff"], json={"query": "x"})
    assert r.status_code == 401


async def test_review_queue(client, services):
    keys, docs_src, _ = await _bootstrap(client, services)
    admin = keys["admin"]
    bad = "# Tin\n\n## Liên hệ\nGọi ngay 0901234567 để có giá chốt tốt nhất."
    await client.post(
        f"/v1/admin/sources/{docs_src['id']}/files",
        headers=admin,
        files={"file": ("tin.md", bad.encode(), "text/markdown")},
        data={"path": "public"},
    )
    await run_worker(services, 1, once=True)
    r = await client.post("/v1/search", headers=keys["customer"], json={"query": "giá chốt liên hệ"})
    assert all("0901234567" not in p["text"] for p in r.json()["passages"])
    flagged = (await client.get("/v1/admin/review/chunks", headers=admin)).json()
    assert len(flagged) == 1 and "sensitive_in_public" in flagged[0]["flags"]
    r = await client.post("/v1/admin/review/chunks/clear", headers=admin, json={"chunk_ids": [flagged[0]["chunk_id"]]})
    assert r.json() == {"cleared": 1}
    r = await client.post("/v1/search", headers=keys["customer"], json={"query": "giá chốt liên hệ"})
    assert any("0901234567" in p["text"] for p in r.json()["passages"])
    # The decision survives re-indexing.
    doc_id = flagged[0]["document_id"]
    await client.post(f"/v1/admin/documents/{doc_id}/reindex", headers=admin)
    await run_worker(services, 1, once=True)
    assert (await client.get("/v1/admin/review/chunks", headers=admin)).json() == []


async def test_webhook_and_tools(client, services, tmp_path):
    keys, docs_src, _ = await _bootstrap(client, services)
    r = await client.post(f"/v1/hooks/sources/{docs_src['id']}/sync", headers={"X-Webhook-Secret": "wrong"})
    assert r.status_code == 404
    r = await client.post(
        f"/v1/hooks/sources/{docs_src['id']}/sync", headers={"X-Webhook-Secret": docs_src["webhook"]["secret"]}
    )
    assert r.status_code == 202 and r.json()["queued"] is True
    tools = (await client.get("/v1/tools")).json()["tools"]
    assert {t["name"] for t in tools} == {"search_documents", "query_listings", "get_listings"}
    assert (await client.get("/healthz")).json() == {"status": "ok"}
    ready = await client.get("/readyz")
    assert ready.status_code == 200 and ready.json()["checks"]["schema"] == "0003"


async def test_upload_validation(client, services):
    keys, docs_src, sheet_src = await _bootstrap(client, services)
    admin = keys["admin"]
    r = await client.post(
        f"/v1/admin/sources/{docs_src['id']}/files",
        headers=admin,
        files={"file": ("x.exe", b"MZ", "application/octet-stream")},
    )
    assert r.status_code == 422
    r = await client.post(
        f"/v1/admin/sources/{sheet_src['id']}/files",
        headers=admin,
        files={"file": ("faq.md", b"# hi", "text/markdown")},
    )
    assert r.status_code == 422


async def test_metrics(client, services):
    keys, *_ = await _bootstrap(client, services)
    await client.post("/v1/search", headers=keys["customer"], json={"query": "phí gửi ô tô"})
    body = (await client.get("/metrics")).text
    assert 'fastrag_http_requests_total{method="POST",route="/v1/search",status="200"}' in body
    assert "fastrag_retrieval_stage_seconds_bucket" in body and "fastrag_jobs_queued" in body
    assert 'fastrag_reranker_total{state="ok"}' in body


async def test_admin_page(client, services):
    r = await client.get("/admin")
    assert r.status_code == 200 and "fastRag" in r.text
    assert "frame-ancestors 'none'" in r.headers["content-security-policy"]
