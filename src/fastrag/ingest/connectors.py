"""Connectors pull documents from source systems.

Only a filesystem connector ships here. Real deployments add SharePoint, Confluence,
Google Drive and so on; the important part is that each connector maps the *source
system's* permissions onto ACL tokens, so access rules follow the document into RAG.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from fastrag.domain import DocumentIn

MEDIA_TYPES = {
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".rst": "text/plain",
    ".html": "text/html",
    ".htm": "text/html",
}


class FilesystemConnector:
    def __init__(self, root: str | Path, acl: frozenset[str], max_bytes: int = 5_000_000):
        self.root = Path(root)
        self.acl = acl
        self.max_bytes = max_bytes

    def documents(self) -> Iterator[DocumentIn]:
        if not self.root.exists():
            raise FileNotFoundError(self.root)
        base = self.root if self.root.is_dir() else self.root.parent
        files = [self.root] if self.root.is_file() else sorted(self.root.rglob("*"))
        for path in files:
            media_type = MEDIA_TYPES.get(path.suffix.lower())
            if media_type is None or not path.is_file() or path.is_symlink():
                continue
            relative = path.relative_to(base)
            if any(part.startswith(".") for part in relative.parts):
                continue
            if path.stat().st_size > self.max_bytes:
                continue
            yield DocumentIn(
                source_id=relative.as_posix(),
                text=path.read_text(encoding="utf-8", errors="replace"),
                acl=self.acl,
                media_type=media_type,
            )
