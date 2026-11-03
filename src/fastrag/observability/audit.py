"""Append-only, hash-chained audit log.

Answers the questions an auditor asks: who asked what, which documents were shown to
them, who changed what, and who was refused. Two rules shape it:

* **No content.** Entries hold ids, hashes and counts - never document text, and the
  question only as a SHA-256 by default. Deleting a document ("right to be forgotten")
  therefore never requires rewriting the audit trail.
* **Tamper-evident.** Each entry includes the hash of the previous one, so editing or
  removing a line breaks the chain (``verify_chain``). Ship the log to WORM storage or a
  SIEM in production; the chain makes tampering detectable, not impossible.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from fastrag.domain import Principal
from fastrag.observability.logging import request_id_var

GENESIS = "0" * 64


def _entry_hash(entry: dict[str, Any]) -> str:
    body = {k: v for k, v in entry.items() if k != "hash"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()


def verify_chain(entries: list[dict[str, Any]]) -> bool:
    prev = GENESIS
    for entry in entries:
        if entry.get("prev_hash") != prev or entry.get("hash") != _entry_hash(entry):
            return False
        prev = entry["hash"]
    return True


class AuditSink(Protocol):
    def append(self, entry: dict[str, Any]) -> None: ...

    def last_hash(self) -> str: ...

    def read(self) -> list[dict[str, Any]]: ...


class MemoryAuditSink:
    def __init__(self, max_entries: int = 100_000):
        self._entries: deque[dict[str, Any]] = deque(maxlen=max_entries)
        self._last = GENESIS

    def append(self, entry: dict[str, Any]) -> None:
        self._entries.append(entry)
        self._last = entry["hash"]

    def last_hash(self) -> str:
        return self._last

    def read(self) -> list[dict[str, Any]]:
        return list(self._entries)


class JsonlAuditSink:
    """One JSON object per line, flushed and fsynced on every write."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(mode=0o600, exist_ok=True)
        self._last = GENESIS
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    self._last = json.loads(line)["hash"]

    def append(self, entry: dict[str, Any]) -> None:
        import os

        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, sort_keys=True, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self._last = entry["hash"]

    def last_hash(self) -> str:
        return self._last

    def read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]


class AuditLog:
    def __init__(self, sink: AuditSink | None = None):
        self.sink = sink or MemoryAuditSink()
        self._lock = threading.Lock()

    def record(
        self,
        action: str,
        outcome: str,
        principal: Principal | None = None,
        **details: Any,
    ) -> dict[str, Any]:
        with self._lock:
            entry: dict[str, Any] = {
                "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
                "request_id": request_id_var.get(),
                "action": action,
                "outcome": outcome,
                "tenant_id": principal.tenant_id if principal else None,
                "user_id": principal.user_id if principal else None,
                "details": details,
                "prev_hash": self.sink.last_hash(),
            }
            entry["hash"] = _entry_hash(entry)
            self.sink.append(entry)
            return entry

    def events(self, tenant_id: str, limit: int = 100) -> list[dict[str, Any]]:
        matching = [e for e in self.sink.read() if e.get("tenant_id") == tenant_id]
        return matching[-limit:]

    def verify(self) -> bool:
        return verify_chain(self.sink.read())


def fingerprint(text: str) -> str:
    """Hash for correlating identical questions in the audit log without storing them."""
    return hashlib.sha256(text.encode()).hexdigest()
