from __future__ import annotations

from fastrag.config import Settings
from fastrag.connectors.base import Connector
from fastrag.db.models import Source

KINDS = ("upload", "local_folder", "web", "google_drive")


def build_connector(source: Source, s: Settings) -> Connector | None:
    cfg = source.config or {}
    if source.kind == "upload":
        return None
    if source.kind == "local_folder":
        from fastrag.connectors.local_folder import LocalFolderConnector

        return LocalFolderConnector(cfg["path"], s.local_sources_root)
    if source.kind == "web":
        from fastrag.connectors.web import WebConnector

        return WebConnector(list(cfg.get("urls") or []), s.allow_private_urls)
    if source.kind == "google_drive":
        from fastrag.connectors.google_drive import GoogleDriveConnector

        creds = cfg.get("credentials_file") or s.google_credentials_file
        if not creds:
            raise ValueError("google_drive source needs FASTRAG_GOOGLE_CREDENTIALS_FILE")
        return GoogleDriveConnector(cfg["folder_id"], creds)
    raise ValueError(f"unknown source kind {source.kind!r}")
