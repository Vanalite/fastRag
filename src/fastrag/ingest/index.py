"""Write and remove chunks, keeping BM25 statistics in step."""

from __future__ import annotations

import uuid
from collections import Counter
from collections.abc import Iterable
from typing import Any

from sqlalchemy import delete, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from fastrag.db.models import Chunk


async def _apply_df(session: AsyncSession, collection_id: uuid.UUID, df: Counter[str], sign: int) -> None:
    if not df:
        return
    terms = sorted(df)  # fixed order avoids deadlocks between workers
    counts = [df[t] * sign for t in terms]
    await session.execute(
        text(
            """
            INSERT INTO term_stats (collection_id, term, df)
            SELECT :cid, t, d FROM unnest(CAST(:terms AS text[]), CAST(:counts AS int[])) AS u(t, d)
            ON CONFLICT (collection_id, term) DO UPDATE SET df = term_stats.df + excluded.df
            """
        ),
        {"cid": collection_id, "terms": terms, "counts": counts},
    )
    if sign < 0:
        await session.execute(
            text("DELETE FROM term_stats WHERE collection_id = :cid AND term = ANY(:terms) AND df <= 0"),
            {"cid": collection_id, "terms": terms},
        )


async def _apply_collection(session: AsyncSession, collection_id: uuid.UUID, chunks: int, length: int) -> None:
    await session.execute(
        text(
            """
            INSERT INTO collection_stats (collection_id, chunk_count, total_length)
            VALUES (:cid, :n, :len)
            ON CONFLICT (collection_id) DO UPDATE SET
              chunk_count = GREATEST(collection_stats.chunk_count + excluded.chunk_count, 0),
              total_length = GREATEST(collection_stats.total_length + excluded.total_length, 0)
            """
        ),
        {"cid": collection_id, "n": chunks, "len": length},
    )


async def remove_document_chunks(session: AsyncSession, document_id: uuid.UUID) -> int:
    rows = (
        await session.execute(
            select(Chunk.collection_id, Chunk.terms, Chunk.length).where(Chunk.document_id == document_id)
        )
    ).all()
    if not rows:
        return 0
    collection_id = rows[0].collection_id
    df: Counter[str] = Counter()
    for r in rows:
        df.update(set(r.terms))
    await _apply_df(session, collection_id, df, -1)
    await _apply_collection(session, collection_id, -len(rows), -sum(r.length for r in rows))
    await session.execute(delete(Chunk).where(Chunk.document_id == document_id))
    return len(rows)


async def add_chunks(session: AsyncSession, collection_id: uuid.UUID, chunks: Iterable[dict[str, Any]]) -> int:
    """Each row is a Chunk's columns plus "tf" (term -> frequency) for the postings."""
    rows = [dict(r) for r in chunks]
    if not rows:
        return 0
    tfs = [r.pop("tf") for r in rows]
    df: Counter[str] = Counter()
    for r in rows:
        df.update(set(r["terms"]))
    objs = [Chunk(**r) for r in rows]
    session.add_all(objs)
    await session.flush()
    ids, terms, counts = [], [], []
    for obj, tf in zip(objs, tfs, strict=True):
        for term, n in tf.items():
            ids.append(obj.id)
            terms.append(term)
            counts.append(min(int(n), 32767))
    for i in range(0, len(ids), 20000):
        await session.execute(
            text(
                "INSERT INTO chunk_postings (collection_id, term, chunk_id, tf) "
                "SELECT :cid, t, c, f FROM unnest(CAST(:terms AS text[]), CAST(:ids AS bigint[]), "
                "CAST(:tfs AS smallint[])) AS u(t, c, f)"
            ),
            {
                "cid": collection_id,
                "terms": terms[i : i + 20000],
                "ids": ids[i : i + 20000],
                "tfs": counts[i : i + 20000],
            },
        )
    await _apply_df(session, collection_id, df, +1)
    await _apply_collection(session, collection_id, len(rows), sum(r["length"] for r in rows))
    return len(rows)
