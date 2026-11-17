"""A small job queue on Postgres (FOR UPDATE SKIP LOCKED). No extra broker
to install or operate."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from fastrag.db.models import Job


async def enqueue(
    session: AsyncSession,
    kind: str,
    payload: dict[str, Any],
    *,
    tenant_id: uuid.UUID | None = None,
    dedupe_key: str | None = None,
    run_after: datetime | None = None,
    max_attempts: int = 3,
) -> int | None:
    """Queue a job. Returns its id, or None if an identical job is already queued."""
    row = (
        await session.execute(
            text(
                """
                INSERT INTO jobs (tenant_id, kind, payload, dedupe_key, run_after, max_attempts)
                VALUES (:tenant_id, :kind, CAST(:payload AS jsonb), :dedupe_key,
                        COALESCE(:run_after, now()), :max_attempts)
                ON CONFLICT (dedupe_key) WHERE status = 'queued' AND dedupe_key IS NOT NULL DO NOTHING
                RETURNING id
                """
            ),
            {
                "tenant_id": tenant_id,
                "kind": kind,
                "payload": _json(payload),
                "dedupe_key": dedupe_key,
                "run_after": run_after,
                "max_attempts": max_attempts,
            },
        )
    ).first()
    return int(row.id) if row else None


# A job going back to the queue is "superseded" when an identical job is
# already queued: that one will do the same work.
_REQUEUE_STATUS = (
    "CASE WHEN jobs.dedupe_key IS NOT NULL AND EXISTS ("
    "SELECT 1 FROM jobs j2 WHERE j2.dedupe_key = jobs.dedupe_key AND j2.status = 'queued' AND j2.id <> jobs.id"
    ") THEN 'superseded' ELSE 'queued' END"
)


def _json(v: Any) -> str:
    import json

    return json.dumps(v, default=str)


async def claim(session: AsyncSession, worker_id: str) -> Job | None:
    row = (
        await session.execute(
            text(
                """
                UPDATE jobs SET status = 'running', locked_at = now(), locked_by = :w,
                       attempts = attempts + 1
                WHERE id = (
                    SELECT id FROM jobs
                    WHERE status = 'queued' AND run_after <= now()
                    ORDER BY run_after, id
                    FOR UPDATE SKIP LOCKED
                    LIMIT 1
                )
                RETURNING id
                """
            ),
            {"w": worker_id},
        )
    ).first()
    if row is None:
        return None
    return await session.get(Job, int(row.id))


async def complete(session: AsyncSession, job_id: int, result: dict[str, Any] | None) -> None:
    await session.execute(
        text(
            "UPDATE jobs SET status = 'done', finished_at = now(), result = CAST(:r AS jsonb), "
            "last_error = NULL WHERE id = :id"
        ),
        {"id": job_id, "r": _json(result or {})},
    )


async def fail(session: AsyncSession, job_id: int, attempts: int, max_attempts: int, error: str) -> None:
    if attempts >= max_attempts:
        await session.execute(
            text("UPDATE jobs SET status = 'failed', finished_at = now(), last_error = :e WHERE id = :id"),
            {"id": job_id, "e": error[:4000]},
        )
        return
    delay = timedelta(seconds=30 * (4 ** (attempts - 1)))
    await session.execute(
        text(
            f"UPDATE jobs SET status = {_REQUEUE_STATUS}, locked_at = NULL, locked_by = NULL, last_error = :e, "
            "run_after = :ra WHERE id = :id"
        ),
        {"id": job_id, "e": error[:4000], "ra": datetime.now(UTC) + delay},
    )


async def recover_stale(session: AsyncSession, older_than: timedelta = timedelta(minutes=30)) -> int:
    """Requeue jobs whose worker died mid-run."""
    res = await session.execute(
        text(
            f"UPDATE jobs SET status = {_REQUEUE_STATUS}, locked_at = NULL, locked_by = NULL, "
            "last_error = 'worker lost' WHERE status = 'running' AND locked_at < :cutoff"
        ),
        {"cutoff": datetime.now(UTC) - older_than},
    )
    return res.rowcount or 0  # type: ignore[attr-defined]
