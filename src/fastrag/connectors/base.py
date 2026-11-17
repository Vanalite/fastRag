from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol


@dataclass
class RemoteItem:
    external_id: str
    name: str
    path: str  # folder path inside the source, "" for the root, used for folder_groups
    mime: str | None
    modified_at: datetime | None
    version_tag: str | None  # etag / md5 / modifiedTime: skips unchanged files cheaply
    uri: str | None = None


class Connector(Protocol):
    async def list_items(self) -> list[RemoteItem]: ...
    async def fetch(self, item: RemoteItem) -> tuple[bytes, str]:
        """Return (data, filename) — the filename decides the parser."""
        ...


def groups_for_path(source_groups: list[str], config: dict[str, Any], path: str) -> list[str]:
    """Folder-inherited access groups (D12): the longest matching folder prefix
    in config.folder_groups wins; otherwise the source default."""
    mapping: dict[str, list[str]] = config.get("folder_groups") or {}
    norm = path.strip("/").lower()
    best, best_len = list(source_groups), -1
    for prefix, groups in mapping.items():
        p = prefix.strip("/").lower()
        if (norm == p or norm.startswith(p + "/") or p == "") and len(p) > best_len:
            best, best_len = list(groups), len(p)
    return best
