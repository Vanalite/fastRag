"""Append-only, hash-chained audit log.

Answers the questions an auditor asks: who asked what, which documents were shown to
them, who changed what, and who was refused. Two rules shape it:

* **No content.** Entries hold ids, hashes and counts - never document text, and the
  question only as a SHA-256 by default. Deleting a document ("right to be forgotten")
  therefore never requires rewriting the audit trail.
* **Tamper-evident.** Each entry includes the hash of the previous one, so editing or
  removing a line breaks the chain (``verify_chain``). Ship the log to WORM storage or a
  SIEM in production; the chain makes tampering detectable, not impossible.

The JSONL sink links each entry to the file's current last line while holding an
exclusive file lock, so several server workers (and the CLI) can append to one log
without forking the chain. File locking is POSIX-only; on other platforms run a single
writer process.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any, Protocol

from fastrag.domain import Principal
from fastrag.errors import ConfigurationError
from fastrag.observability.logging import request_id_var

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore[assignment]

GENESIS = "0" * 64
log = logging.getLogger(__name__)

EntryBuilder = Callable[[str], dict[str, Any]]


def _entry_hash(entry: dict[str, Any]) -> str:
    body = {k: v for k, v in entry.items() if k != "hash"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()


def verify_chain(entries: list[dict[str, Any]], anchor: str = GENESIS) -> bool:
    """Check every entry's hash and its link to the one before.

    ``anchor`` is the hash the first entry must point to: the genesis value for a
    complete log, or the last verified hash when checking a tail.
    """
    prev = anchor
    for entry in entries:
        if entry.get("prev_hash") != prev or entry.get("hash") != _entry_hash(entry):
            return False
        prev = entry["hash"]
    return True


class AuditSink(Protocol):
    def append(self, build: EntryBuilder) -> dict[str, Any]:
        """Atomically: read the chain head, call ``build(prev_hash)``, persist the entry."""

    def read(self) -> list[dict[str, Any]]: ...

    def anchor(self) -> str:
        """The ``prev_hash`` the first entry returned by ``read`` should have."""


class MemoryAuditSink:
    """Bounded in-memory sink for tests and demos. The oldest entries are dropped first."""

    def __init__(self, max_entries: int = 100_000):
        self._entries: deque[dict[str, Any]] = deque(maxlen=max_entries)
        self._last = GENESIS
        self._lock = threading.Lock()

    def append(self, build: EntryBuilder) -> dict[str, Any]:
        with self._lock:
            entry = build(self._last)
            self._entries.append(entry)
            self._last = entry["hash"]
            return entry

    def read(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._entries)

    def anchor(self) -> str:
        with self._lock:
            return self._entries[0]["prev_hash"] if self._entries else GENESIS


class JsonlAuditSink:
    """One JSON object per line; locked, flushed and fsynced on every write."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(mode=0o600, exist_ok=True)
        self._lock = threading.Lock()

    def append(self, build: EntryBuilder) -> dict[str, Any]:
        with self._lock, self.path.open("a+b") as f:
            if fcntl is not None:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                _repair_torn_tail(f, self.path)
                entry = build(_last_hash(f, self.path))
                f.seek(0, os.SEEK_END)
                f.write((json.dumps(entry, sort_keys=True, default=str) + "\n").encode())
                f.flush()
                os.fsync(f.fileno())
                return entry
            finally:
                if fcntl is not None:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def anchor(self) -> str:
        return GENESIS


def _repair_torn_tail(f: IO[bytes], path: Path) -> None:
    """Drop a final line that was cut off mid-write (no trailing newline).

    Entries are written as one line ending in a newline and fsynced before ``record``
    returns, so a missing newline can only mean a crash during the write: that entry
    was never acknowledged, and removing the fragment loses nothing. A complete line
    that doesn't parse is different - that's corruption or tampering, and ``_last_hash``
    refuses to continue the chain past it.
    """
    f.seek(0, os.SEEK_END)
    size = f.tell()
    if size == 0:
        return
    f.seek(size - 1)
    if f.read(1) == b"\n":
        return
    window = 4096
    while True:
        start = max(0, size - window)
        f.seek(start)
        cut = f.read(size - start).rfind(b"\n")
        if cut != -1 or start == 0:
            keep = start + cut + 1 if cut != -1 else 0
            break
        window *= 2
    log.warning("audit log %s ended with a torn line; truncating %d bytes", path, size - keep)
    f.truncate(keep)


def _last_hash(f: IO[bytes], path: Path) -> str:
    """Hash of the last line, read backwards from the end of the file."""
    f.seek(0, os.SEEK_END)
    end = f.tell()
    window = 4096
    while True:
        start = max(0, end - window)
        f.seek(start)
        lines = f.read(end - start).rstrip(b"\n").split(b"\n")
        if len(lines) > 1 or start == 0:
            last = lines[-1].strip()
            if not last:
                return GENESIS
            try:
                return json.loads(last)["hash"]
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ConfigurationError(
                    f"audit log {path} ends with a corrupt entry; investigate before writing"
                ) from exc
        window *= 2


class AuditLog:
    def __init__(self, sink: AuditSink | None = None):
        self.sink = sink if sink is not None else MemoryAuditSink()

    def record(
        self,
        action: str,
        outcome: str,
        principal: Principal | None = None,
        **details: Any,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "request_id": request_id_var.get(),
            "action": action,
            "outcome": outcome,
            "tenant_id": principal.tenant_id if principal else None,
            "user_id": principal.user_id if principal else None,
            "details": details,
        }

        def build(prev_hash: str) -> dict[str, Any]:
            entry = {**body, "prev_hash": prev_hash}
            entry["hash"] = _entry_hash(entry)
            return entry

        return self.sink.append(build)

    def events(self, tenant_id: str, limit: int = 100) -> list[dict[str, Any]]:
        matching = [e for e in self.sink.read() if e.get("tenant_id") == tenant_id]
        return matching[-limit:]

    def verify(self) -> bool:
        return verify_chain(self.sink.read(), self.sink.anchor())


def fingerprint(text: str) -> str:
    """Hash for correlating identical questions in the audit log without storing them."""
    return hashlib.sha256(text.encode()).hexdigest()
