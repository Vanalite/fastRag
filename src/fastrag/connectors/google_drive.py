"""Google Drive folder (documents) and Google Sheets (listings), read with a
service account. Share the folder with the service account's email.

Needs the `google` extra and FASTRAG_GOOGLE_CREDENTIALS_FILE.
"""

from __future__ import annotations

import asyncio
import io
from datetime import datetime
from typing import Any

from fastrag.connectors.base import RemoteItem

FOLDER = "application/vnd.google-apps.folder"
EXPORTS = {
    "application/vnd.google-apps.document": ("text/html", ".html"),
    "application/vnd.google-apps.spreadsheet": (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".xlsx",
    ),
}
SUPPORTED = {
    "application/pdf": ".pdf",
    "text/html": ".html",
    "text/plain": ".txt",
    "text/markdown": ".md",
    "text/csv": ".csv",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
}


class GoogleDriveConnector:
    def __init__(self, folder_id: str, credentials_file: str, max_depth: int = 6):
        from google.oauth2 import service_account
        from googleapiclient.discovery import build

        creds = service_account.Credentials.from_service_account_file(
            credentials_file, scopes=["https://www.googleapis.com/auth/drive.readonly"]
        )
        self.drive: Any = build("drive", "v3", credentials=creds, cache_discovery=False)
        self.folder_id = folder_id
        self.max_depth = max_depth

    def _list(self, folder: str, path: str, depth: int, out: list[RemoteItem]) -> None:
        token = None
        while True:
            resp = (
                self.drive.files()
                .list(
                    q=f"'{folder}' in parents and trashed = false",
                    fields="nextPageToken, files(id, name, mimeType, modifiedTime, md5Checksum, webViewLink)",
                    pageSize=200,
                    pageToken=token,
                    supportsAllDrives=True,
                    includeItemsFromAllDrives=True,
                )
                .execute()
            )
            for f in resp.get("files", []):
                mime = f["mimeType"]
                if mime == FOLDER:
                    if depth < self.max_depth:
                        self._list(f["id"], f"{path}/{f['name']}".strip("/"), depth + 1, out)
                    continue
                if mime not in EXPORTS and mime not in SUPPORTED:
                    continue
                out.append(
                    RemoteItem(
                        external_id=f["id"],
                        name=f["name"],
                        path=path,
                        mime=mime,
                        modified_at=datetime.fromisoformat(f["modifiedTime"].replace("Z", "+00:00")),
                        version_tag=f.get("md5Checksum") or f["modifiedTime"],
                        uri=f.get("webViewLink"),
                    )
                )
            token = resp.get("nextPageToken")
            if not token:
                break

    async def list_items(self) -> list[RemoteItem]:
        out: list[RemoteItem] = []
        await asyncio.to_thread(self._list, self.folder_id, "", 0, out)
        return out

    def _fetch(self, item: RemoteItem) -> tuple[bytes, str]:
        from googleapiclient.http import MediaIoBaseDownload

        if item.mime in EXPORTS:
            export_mime, ext = EXPORTS[item.mime]
            request = self.drive.files().export_media(fileId=item.external_id, mimeType=export_mime)
        else:
            ext = SUPPORTED[item.mime or ""]
            request = self.drive.files().get_media(fileId=item.external_id, supportsAllDrives=True)
        buf = io.BytesIO()
        downloader = MediaIoBaseDownload(buf, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        name = item.name if item.name.lower().endswith(ext) else item.name + ext
        return buf.getvalue(), name

    async def fetch(self, item: RemoteItem) -> tuple[bytes, str]:
        return await asyncio.to_thread(self._fetch, item)
