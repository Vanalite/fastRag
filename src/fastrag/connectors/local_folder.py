"""A folder on disk, e.g. a mounted shared drive with Excel files."""

from __future__ import annotations

import asyncio
import hashlib
from datetime import UTC, datetime
from pathlib import Path

from fastrag.connectors.base import RemoteItem
from fastrag.ingest.parsers import MIME_BY_EXT


class LocalFolderConnector:
    def __init__(self, root: str, allowed_root: str):
        allowed = Path(allowed_root).resolve()
        path = (allowed / root).resolve() if not Path(root).is_absolute() else Path(root).resolve()
        if not path.is_relative_to(allowed):
            raise ValueError(f"local source must be inside {allowed}")
        self.root = path

    async def list_items(self) -> list[RemoteItem]:
        def walk() -> list[RemoteItem]:
            items = []
            for p in sorted(self.root.rglob("*")):
                if not p.is_file() or p.name.startswith((".", "~$")) or p.suffix.lower() not in MIME_BY_EXT:
                    continue
                st = p.stat()
                rel = p.relative_to(self.root)
                tag = hashlib.sha1(f"{st.st_size}:{st.st_mtime_ns}".encode(), usedforsecurity=False).hexdigest()
                items.append(
                    RemoteItem(
                        external_id=str(rel),
                        name=p.name,
                        path=str(rel.parent) if str(rel.parent) != "." else "",
                        mime=MIME_BY_EXT[p.suffix.lower()],
                        modified_at=datetime.fromtimestamp(st.st_mtime, UTC),
                        version_tag=tag,
                        uri=f"file://{p}",
                    )
                )
            return items

        return await asyncio.to_thread(walk)

    async def fetch(self, item: RemoteItem) -> tuple[bytes, str]:
        path = (self.root / item.external_id).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("path escapes source root")
        return await asyncio.to_thread(path.read_bytes), item.name
