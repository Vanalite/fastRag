"""Spreadsheet listings: import, column visibility, filters, lookups."""

import pytest
from sqlalchemy import select

from factories import add_document, make_collection, make_source, make_tenant, xlsx
from fastrag.db.engine import get_sessionmaker, session_scope
from fastrag.db.models import Document, ListingSnapshot, SheetMapping
from fastrag.ingest.sync import import_spreadsheet_document
from fastrag.retrieval import listings as L
from fastrag.text.values import default_gazetteer

pytestmark = pytest.mark.db

HEADER = [
    "Mã căn",
    "Dự án",
    "Loại",
    "Giá bán (tỷ)",
    "Diện tích",
    "PN",
    "Hướng",
    "Quận",
    "Tình trạng",
    "Giá sàn",
    "Ghi chú",
]
ROWS = [
    [
        "A-01",
        "Sunrise Riverside",
        "Căn hộ",
        3.5,
        "72m2",
        "2PN",
        "Đông Nam",
        "Quận 7",
        "Còn hàng",
        "3,1 tỷ",
        "Khách VIP",
    ],
    ["A-02", "Sunrise Riverside", "Căn hộ", 2.8, "56 m²", 1, "Tây Bắc", "Q7", "Đã bán", "2,6 tỷ", ""],
    ["B-01", "Masteri Thảo Điền", "Căn hộ", 5.2, 90, 3, "Nam", "Quận 2", "Còn hàng", "4,9 tỷ", "view sông"],
    ["C-01", "Vinhomes Grand Park", "Căn hộ", 2.1, 47, 1, "Bắc", "Quận 9", "Đã cọc", "2 tỷ", None],
    [
        "D-01",
        "Nhà phố Bình Thạnh",
        "Nhà phố",
        "thỏa thuận",
        "4x15",
        4,
        "Đông",
        "Bình Thạnh",
        "Còn hàng",
        None,
        "hẻm xe hơi",
    ],
]
PUBLIC = ["public", "staff"]


async def _setup(services, sheet=None):
    tid = await make_tenant()
    cid = await make_collection(tid, "listings", "listings")
    sid = await make_source(tid, cid, groups=["public"])
    doc_id, res = await add_document(
        services, sid, "bang-gia.xlsx", xlsx(sheet or [["BẢNG GIÁ THÁNG 11"], HEADER, *ROWS])
    )
    return tid, cid, sid, doc_id, res


async def _confirm_public(cid, public_headers):
    async with session_scope() as s:
        m = (await s.execute(select(SheetMapping).where(SheetMapping.collection_id == cid))).scalar_one()
        cols = dict(m.columns)
        for h in public_headers:
            cols[h] = cols[h] | {"groups": PUBLIC}
        m.columns, m.confirmed = cols, True


async def _reimport(services, doc_id):
    async with session_scope() as s:
        return await import_spreadsheet_document(s, services, doc_id)


async def _query(tid, cid, groups, services, **kw):
    q = L.ListingQuery(tid, [cid], groups, kw.pop("filters", {}), **kw)
    async with get_sessionmaker()() as s:
        return await L.query_listings(s, q, default_gazetteer(), services.settings.segmenter)


async def test_import_report_and_mapping(db, services):
    tid, cid, sid, doc_id, res = await _setup(services)
    sheet = res["sheets"]["Sheet1"]
    assert sheet["rows"] == 5 and sheet["mapping_confirmed"] is False
    assert sheet["cell_errors"] == 1  # "thỏa thuận" is not a price
    async with get_sessionmaker()() as s:
        m = (await s.execute(select(SheetMapping).where(SheetMapping.collection_id == cid))).scalar_one()
        assert m.columns["Giá bán (tỷ)"]["field"] == "price_vnd" and m.columns["Giá bán (tỷ)"]["unit"] == "ty"
        assert m.columns["Giá sàn"]["field"] == "attr:gia-san"
        doc = await s.get(Document, doc_id)
        assert doc.status == "imported"


async def test_unconfirmed_columns_are_internal(db, services):
    tid, cid, *_ = await _setup(services)
    res = await _query(tid, cid, ["public"], services)
    # Rows are visible (source is public) but no field is, until an admin marks columns public.
    assert res["total"] == 5
    assert all(set(i) == {"id", "collection_id", "as_of"} for i in res["items"])
    staff = await _query(tid, cid, ["public", "staff"], services)
    assert staff["items"][0]["price_vnd"] == 3_500_000_000


async def test_public_columns_and_masking(db, services):
    tid, cid, sid, doc_id, _ = await _setup(services)
    await _confirm_public(
        cid, ["Mã căn", "Dự án", "Loại", "Giá bán (tỷ)", "Diện tích", "PN", "Hướng", "Quận", "Tình trạng"]
    )
    await _reimport(services, doc_id)
    res = await _query(tid, cid, ["public"], services, sort="price_asc")
    items = {i["listing_code"]: i for i in res["items"]}
    assert items["A-01"]["price_vnd"] == 3_500_000_000 and items["A-01"]["area_m2"] == 72.0
    assert items["A-01"]["district_code"] == "hcm-q7" and items["A-01"]["direction"] == "SE"
    assert items["A-01"]["city"] == "hcm" and items["A-01"]["status"] == "available"
    # Internal columns never reach the customer scope.
    for i in res["items"]:
        assert "gia-san" not in i.get("attributes", {}) and "ghi-chu" not in i.get("attributes", {})
    staff = await _query(tid, cid, ["public", "staff"], services, filters={"district": "Q7"})
    assert any(i.get("attributes", {}).get("gia-san") == "3,1 tỷ" for i in staff["items"])


async def test_filters(db, services):
    tid, cid, sid, doc_id, _ = await _setup(services)
    await _confirm_public(cid, ["Mã căn", "Dự án", "Giá bán (tỷ)", "Diện tích", "PN", "Quận", "Tình trạng"])
    await _reimport(services, doc_id)

    async def codes(**filters):
        r = await _query(tid, cid, ["public"], services, filters=filters, sort="price_asc")
        return [i["listing_code"] for i in r["items"]]

    assert await codes(price_max="3,5 tỷ") == ["C-01", "A-02", "A-01"]
    assert await codes(price_min="3 tỷ", price_max="6 tỷ") == ["A-01", "B-01"]
    assert await codes(district="Q7") == ["A-02", "A-01"]
    assert await codes(district="District 7", status="còn hàng") == ["A-01"]
    # Thủ Đức covers the former Quận 2 and Quận 9.
    assert await codes(district="Thủ Đức") == ["C-01", "B-01"]
    assert await codes(bedrooms_min=2, area_min="65m2") == ["A-01", "B-01"]
    assert await codes(project="Sunrise Riverside") == ["A-02", "A-01"]
    assert await codes(keywords="masteri thao dien") == ["B-01"]


async def test_filter_on_internal_field_matches_nothing(db, services):
    tid, cid, sid, doc_id, _ = await _setup(services)
    await _confirm_public(cid, ["Mã căn"])
    await _reimport(services, doc_id)
    # Price is internal: a customer-scope price filter must not reveal anything.
    r = await _query(tid, cid, ["public"], services, filters={"price_max": "3 tỷ"})
    assert r["total"] == 0
    r = await _query(tid, cid, ["public"], services, filters={"keywords": "Khách VIP"})
    assert r["total"] == 0
    r = await _query(tid, cid, ["public"], services, sort="price_desc")
    assert all("price_vnd" not in i for i in r["items"])


async def test_unreadable_filters_are_rejected(db, services):
    tid, cid, *_ = await _setup(services)
    with pytest.raises(L.FilterError) as exc:
        await _query(
            tid, cid, ["public", "staff"], services, filters={"price_max": "rẻ", "district": "Quận 99", "colour": "red"}
        )
    names = {r["filter"] for r in exc.value.rejected}
    assert names == {"price_max", "district", "colour"}
    with pytest.raises(L.FilterError):
        await _query(tid, cid, ["public", "staff"], services, filters={"price_max": "35 triệu/m2"})


async def test_snapshot_replace_and_stable_ids(db, services):
    tid, cid, sid, doc_id, _ = await _setup(services)
    first = await _query(tid, cid, ["public", "staff"], services, ids=None, codes=["A-01"])
    a01 = first["items"][0]
    # Next sync: A-02 removed, A-01 price changed.
    new_rows = [r[:] for r in ROWS if r[0] != "A-02"]
    new_rows[0][3] = 3.6
    _, res = await add_document(services, sid, "bang-gia.xlsx", xlsx([HEADER, *new_rows]))
    assert res["sheets"]["Sheet1"]["rows"] == 4
    again = await _query(tid, cid, ["public", "staff"], services, ids=[__import__("uuid").UUID(a01["id"])])
    item = again["items"][0]
    assert item["id"] == a01["id"] and item["price_vnd"] == 3_600_000_000
    assert item["as_of"] >= a01["as_of"]
    gone = await _query(tid, cid, ["public", "staff"], services, codes=["A-02"])
    assert gone["total"] == 0
    async with get_sessionmaker()() as s:
        snaps = (await s.execute(select(ListingSnapshot).where(ListingSnapshot.collection_id == cid))).scalars().all()
        assert len(snaps) == 2


async def test_rows_from_staff_folder_hidden_from_customers(db, services):
    tid = await make_tenant()
    cid = await make_collection(tid, "listings", "listings")
    sid = await make_source(tid, cid, groups=["staff"])
    await add_document(services, sid, "noi-bo.xlsx", xlsx([HEADER, *ROWS]))
    await _confirm_public(cid, HEADER)
    r = await _query(tid, cid, ["public"], services)
    assert r["total"] == 0


async def test_csv_and_duplicate_codes(db, services):
    tid = await make_tenant()
    cid = await make_collection(tid, "listings", "listings")
    sid = await make_source(tid, cid, groups=["public"])
    csv_data = "Mã,Giá,Diện tích\nX1,2 tỷ,50m2\nX1,2.2 tỷ,52m2\n,1 tỷ 9,40\n".encode()
    _, res = await add_document(services, sid, "list.csv", csv_data)
    assert res["sheets"]["Sheet1"]["rows"] == 3
    r = await _query(tid, cid, ["public", "staff"], services, sort="price_asc")
    codes = [i["listing_code"] for i in r["items"]]
    assert codes[0].startswith("row-") and set(codes[1:]) == {"X1", "X1~2"}
