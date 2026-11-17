from datetime import UTC, datetime

import httpx
import pytest
import respx

from fastrag.connectors.base import RemoteItem, groups_for_path
from fastrag.connectors.google_drive import GoogleDriveConnector
from fastrag.connectors.web import WebConnector, check_url


def test_folder_groups_longest_prefix_wins():
    cfg = {"folder_groups": {"Public": ["public"], "Public/Draft": ["staff"], "Managers": ["managers"]}}
    assert groups_for_path(["staff"], cfg, "Public/Brochures") == ["public"]
    assert groups_for_path(["staff"], cfg, "public/draft/v2") == ["staff"]
    assert groups_for_path(["staff"], cfg, "Sales") == ["staff"]
    assert groups_for_path(["staff"], cfg, "PublicX") == ["staff"]


@pytest.mark.parametrize(
    "url",
    ["http://127.0.0.1/admin", "http://169.254.169.254/latest/meta-data", "http://10.0.0.5/", "file:///etc/passwd"],
)
def test_ssrf_guard(url):
    with pytest.raises(ValueError):
        check_url(url, allow_private=False)


@respx.mock
async def test_web_connector_follows_redirects_with_checks():
    respx.get("http://example.test/a").mock(return_value=httpx.Response(301, headers={"location": "/b"}))
    respx.get("http://example.test/b").mock(
        return_value=httpx.Response(200, html="<h1>Hi</h1>", headers={"last-modified": "Wed, 01 Oct 2026 10:00:00 GMT"})
    )
    conn = WebConnector(["http://example.test/a"], allow_private=True)
    [item] = await conn.list_items()
    data, name = await conn.fetch(item)
    assert b"Hi" in data and name.endswith(".html")
    assert item.modified_at == datetime(2026, 10, 1, 10, 0, tzinfo=UTC)


class _Req:
    def __init__(self, result):
        self.result = result

    def execute(self):
        return self.result


class _Files:
    def __init__(self, tree):
        self.tree = tree

    def list(self, q, **kw):
        folder = q.split("'")[1]
        return _Req({"files": self.tree.get(folder, [])})


class _Drive:
    def __init__(self, tree):
        self._files = _Files(tree)

    def files(self):
        return self._files


async def test_google_drive_listing_walks_folders_and_skips_unsupported():
    tree = {
        "root": [
            {
                "id": "f1",
                "name": "Public",
                "mimeType": "application/vnd.google-apps.folder",
                "modifiedTime": "2026-11-01T00:00:00Z",
            },
            {
                "id": "d1",
                "name": "Nội quy",
                "mimeType": "application/vnd.google-apps.document",
                "modifiedTime": "2026-11-02T00:00:00Z",
            },
            {"id": "x1", "name": "photo.jpg", "mimeType": "image/jpeg", "modifiedTime": "2026-11-02T00:00:00Z"},
        ],
        "f1": [
            {
                "id": "s1",
                "name": "Bảng giá",
                "mimeType": "application/vnd.google-apps.spreadsheet",
                "modifiedTime": "2026-11-03T00:00:00Z",
            },
            {
                "id": "p1",
                "name": "brochure.pdf",
                "mimeType": "application/pdf",
                "modifiedTime": "2026-11-03T00:00:00Z",
                "md5Checksum": "abc",
            },
        ],
    }
    conn = GoogleDriveConnector.__new__(GoogleDriveConnector)
    conn.drive, conn.folder_id, conn.max_depth = _Drive(tree), "root", 6
    items = {i.external_id: i for i in await conn.list_items()}
    assert set(items) == {"d1", "s1", "p1"}
    assert items["s1"].path == "Public" and items["d1"].path == ""
    assert items["p1"].version_tag == "abc" and items["s1"].version_tag == "2026-11-03T00:00:00Z"
    assert isinstance(items["p1"], RemoteItem)
