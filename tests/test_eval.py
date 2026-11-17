import json

import pytest

from factories import add_document, make_collection, make_source, make_tenant, xlsx
from fastrag.db.engine import get_sessionmaker
from fastrag.eval.runner import load_references, run_eval, score_ranked

pytestmark = pytest.mark.db

FAQ = "# FAQ\n\n## Phí quản lý\nPhí quản lý là 15.000 đồng/m2/tháng.\n\n## Hồ bơi\nHồ bơi mở cửa từ 6h đến 21h."


def test_score_ranked():
    results = [{"external_id": "a", "text": "x"}, {"external_id": "b", "text": "phí quản lý"}]
    assert score_ranked(results, [{"document": "b"}], 10) == (1.0, 0.5)
    assert score_ranked(results, [{"document": "b", "contains": "phi quan ly"}], 10) == (1.0, 0.5)
    assert score_ranked(results, [{"document": "c"}], 10) == (0.0, 0.0)


async def test_run_eval(db, services, tmp_path):
    tid = await make_tenant()
    cid = await make_collection(tid, "docs", "documents")
    lid = await make_collection(tid, "listings", "listings")
    sid = await make_source(tid, cid, folder_groups={"public": ["public"]})
    await add_document(services, sid, "faq.md", FAQ, path="public")
    lsid = await make_source(tid, lid, groups=["public"])
    await add_document(services, lsid, "gia.xlsx", xlsx([["Mã", "Giá"], ["A1", "3 tỷ"]]))
    refs = tmp_path / "refs.jsonl"
    lines = [
        {"id": "q1", "query": "phí quản lý bao nhiêu", "relevant": [{"document": "public/faq.md"}]},
        {"id": "q2", "query": "hồ bơi mấy giờ mở", "relevant": [{"document": "FAQ", "contains": "6h"}]},
        {"id": "q3", "query": "lãi suất ngân hàng kỳ hạn dài nhất", "expect_nothing": True},
        {
            "id": "l1",
            "type": "listings",
            "filters": {"price_max": "3,5 tỷ"},
            "expect_applied": {"price_max": 3_500_000_000},
            "expect_ids": ["A1"],
        },
        {"id": "l2", "type": "listings", "filters": {"district": "Quận 99"}, "expect_rejected": True},
    ]
    refs.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in lines), encoding="utf-8")
    baseline = tmp_path / "ragflow.jsonl"
    baseline.write_text(json.dumps({"id": "q1", "results": [{"document": "other"}, {"document": "public/faq.md"}]}))
    rep = await run_eval(get_sessionmaker(), services, tid, load_references(refs), baseline=str(baseline))
    d = rep.to_dict()
    assert d["documents"]["questions"] == 2 and d["documents"]["recall@10"] == 1.0 and d["documents"]["mrr"] == 1.0
    assert d["documents"]["nothing_relevant_accuracy"] == 1.0
    assert d["access_safety"]["passed"] and d["listings"]["passed"], d["listings"]
    assert d["baseline"]["questions"] == 2 and d["baseline"]["mrr"] == 0.25
    assert rep.passed


async def test_access_check_allows_public_plus_staff(db, services):
    from fastrag.eval.runner import _access_check

    tid = await make_tenant()
    cid = await make_collection(tid, "docs", "documents")
    sid = await make_source(tid, cid, groups=["public", "staff"])
    await add_document(services, sid, "a.md", "# A\n\n## B\nNội dung công khai cho mọi người.")
    sid2 = await make_source(tid, cid, groups=["staff"])
    await add_document(services, sid2, "b.md", "# C\n\n## D\nNội dung chỉ dành cho nhân viên.")
    async with get_sessionmaker()() as s:
        from sqlalchemy import text as sql

        ids = {
            r.groups[0] if len(r.groups) == 1 else "both": r.id
            for r in (await s.execute(sql("SELECT id, groups FROM chunks"))).all()
        }
        assert await _access_check(s, [ids["both"]]) == []
        assert await _access_check(s, [ids["staff"]]) == [ids["staff"]]


def test_fit_platt_is_finite_on_separable_data():
    from fastrag.eval.calibrate import fit_platt
    from fastrag.providers.reranker import sigmoid

    xs = [-3.0, -2.5, -1.0, -9.0, -10.0, -8.5, -11.0, -9.5]
    ys = [1, 1, 1, 0, 0, 0, 0, 0]
    a, b = fit_platt(xs, ys)
    assert abs(a) < 100 and abs(b) < 500
    assert sigmoid(a * -3.0 + b) > 0.5 > sigmoid(a * -9.0 + b)
