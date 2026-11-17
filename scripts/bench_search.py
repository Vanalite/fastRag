"""Search latency at scale, without model servers.

Loads N synthetic Vietnamese chunks (Zipf-distributed vocabulary, random unit
vectors of FASTRAG_EMBEDDING_DIM) into a fresh database and times the
database side of the query path: BM25, filtered HNSW, passage load.
Embedding and reranking are not included; add their latency separately.

    FASTRAG_DATABASE_URL=postgresql+asyncpg://.../fastrag_bench \
    uv run python scripts/bench_search.py --chunks 50000 --queries 200
"""

from __future__ import annotations

import argparse
import asyncio
import random
import statistics
import time
import unicodedata
import uuid

import numpy as np
from sqlalchemy import text

from fastrag.config import get_settings
from fastrag.db.engine import get_sessionmaker, session_scope
from fastrag.db.models import Collection, Document, Source, Tenant
from fastrag.ingest.index import add_chunks
from fastrag.retrieval.profiles import resolve_profile
from fastrag.retrieval.search import SearchRequest, _load_passages, keyword_search, rrf, vector_search
from fastrag.text.vietnamese import keyword_terms, query_terms

ONSETS = "b c ch d đ g gi h k kh l m n ng nh p ph qu r s t th tr v x".split()
RIMES = "a an ăn ân anh ang ao âu ay e ê em en eo êt i in im inh o ô ơ ong ông u ư ung ương ươi uy oai uôn iêt".split()


TONES = ["", "\u0300", "\u0301", "\u0309", "\u0303", "\u0323"]


def _syllable(rng: random.Random) -> str:
    rime = rng.choice(RIMES)
    # Put a tone on the first vowel, as Vietnamese does for most rimes.
    for i, ch in enumerate(rime):
        if ch in "aăâeêioôơuưy":
            rime = rime[: i + 1] + rng.choice(TONES) + rime[i + 1 :]
            break
    return unicodedata.normalize("NFC", rng.choice(ONSETS) + rime)


def vocabulary(n: int, rng: random.Random) -> list[str]:
    words: set[str] = set()
    while len(words) < n:
        words.add(" ".join(_syllable(rng) for _ in range(rng.choice((1, 2, 2, 3)))))
    words_list = sorted(words)
    rng.shuffle(words_list)  # Zipf rank is random, not alphabetical
    return words_list


def zipf_sample(vocab: list[str], k: int, rng: np.random.Generator) -> list[str]:
    idx = np.minimum(rng.zipf(1.15, size=k) - 1, len(vocab) - 1)
    return [vocab[i] for i in idx]


async def load(n_chunks: int, dim: int, segmenter: str) -> tuple[uuid.UUID, uuid.UUID, list[str]]:
    rng = random.Random(7)
    nrng = np.random.default_rng(7)
    vocab = vocabulary(20000, rng)
    async with session_scope() as s:
        t = Tenant(id=uuid.uuid4(), slug=f"bench-{uuid.uuid4().hex[:6]}", name="bench")
        s.add(t)
        await s.flush()
        c = Collection(id=uuid.uuid4(), tenant_id=t.id, slug="docs", name="docs", type="documents", settings={})
        s.add(c)
        await s.flush()
        src = Source(
            id=uuid.uuid4(), tenant_id=t.id, collection_id=c.id, kind="upload", name="b", config={}, groups=["staff"]
        )
        s.add(src)
        await s.flush()
    per_doc, batch = 25, 1000
    t0 = time.perf_counter()
    for start in range(0, n_chunks, batch):
        rows = []
        async with session_scope() as s:
            docs = []
            for d in range(batch // per_doc):
                groups = ["public"] if (start // per_doc + d) % 3 == 0 else ["staff"]
                doc = Document(
                    id=uuid.uuid4(),
                    tenant_id=t.id,
                    collection_id=c.id,
                    source_id=src.id,
                    external_id=uuid.uuid4().hex,
                    title="Tài liệu",
                    groups=groups,
                    status="ready",
                )
                docs.append(doc)
            s.add_all(docs)
            await s.flush()
            vecs = nrng.standard_normal((batch, dim)).astype(np.float32)
            vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
            for i in range(batch):
                doc = docs[i // per_doc]
                text = " ".join(zipf_sample(vocab, 120, nrng))
                terms = keyword_terms(text, segmenter)
                rows.append(
                    {
                        "tenant_id": t.id,
                        "collection_id": c.id,
                        "document_id": doc.id,
                        "ordinal": i % per_doc,
                        "text": text,
                        "context": "Tài liệu",
                        "heading": None,
                        "page_start": None,
                        "page_end": None,
                        "token_count": 160,
                        "groups": doc.groups,
                        "terms": sorted(terms.freqs),
                        "tf": terms.freqs,
                        "length": terms.length,
                        "embedding": vecs[i].tolist(),
                        "flags": [],
                    }
                )
            await add_chunks(s, c.id, rows)
        done = start + batch
        if done % 10000 == 0:
            print(f"  loaded {done} chunks ({time.perf_counter() - t0:.0f}s)", flush=True)
    return t.id, c.id, vocab


def pct(values: list[float], p: float) -> float:
    values = sorted(values)
    return values[min(len(values) - 1, int(round(p / 100 * (len(values) - 1))))]


async def run(n_chunks: int, n_queries: int, reuse: bool = False) -> None:
    s = get_settings()
    # The builtin segmenter keeps loading fast; the query path is the same.
    seg = "builtin"
    sm = get_sessionmaker()
    if reuse:
        from sqlalchemy import select

        async with sm() as sess:
            t = (await sess.execute(select(Tenant).where(Tenant.slug.like("bench-%")))).scalars().first()
            if t is None:
                raise SystemExit("no benchmark data to reuse")
            tid = t.id
            cid = (await sess.execute(select(Collection.id).where(Collection.tenant_id == tid))).scalar_one()
            n_chunks = (await sess.execute(text("SELECT count(*) FROM chunks"))).scalar_one()
        vocab = vocabulary(20000, random.Random(7))
    else:
        print(f"loading {n_chunks} chunks, dim {s.embedding_dim}", flush=True)
        tid, cid, vocab = await load(n_chunks, s.embedding_dim, seg)
    async with sm() as sess:
        await sess.execute(text("ANALYZE chunks"))
        await sess.commit()
    rng = random.Random(11)
    nrng = np.random.default_rng(11)
    results: dict[str, dict[str, list[float]]] = {}
    for profile_name, groups in (("customer", ["public"]), ("internal_search", ["public", "staff"])):
        prof = resolve_profile(profile_name)
        times: dict[str, list[float]] = {"keyword": [], "vector": [], "load": [], "total": []}
        for _ in range(n_queries):
            q = " ".join(rng.choice(vocab[:800]) for _ in range(rng.randint(3, 7)))
            qv = nrng.standard_normal(s.embedding_dim).astype(np.float32)
            qv = (qv / np.linalg.norm(qv)).tolist()
            req = SearchRequest(tid, [cid], q, groups, prof)
            qt = query_terms(q, seg)
            t0 = time.perf_counter()

            async def kw() -> list[tuple[int, float]]:
                t = time.perf_counter()
                async with sm() as ss:
                    out = await keyword_search(ss, req, qt, prof.keyword_candidates)
                times["keyword"].append((time.perf_counter() - t) * 1000)
                return out

            async def vec() -> list[tuple[int, float]]:
                t = time.perf_counter()
                async with sm() as ss, ss.begin():
                    out = await vector_search(ss, req, qv, prof.vector_candidates)
                times["vector"].append((time.perf_counter() - t) * 1000)
                return out

            a, b = await asyncio.gather(kw(), vec())
            fused = rrf([[x for x, _ in a], [x for x, _ in b]])
            ids = sorted(fused, key=lambda k: -fused[k])[: prof.rerank_candidates]
            t = time.perf_counter()
            async with sm() as ss:
                await _load_passages(ss, req, ids)
            times["load"].append((time.perf_counter() - t) * 1000)
            times["total"].append((time.perf_counter() - t0) * 1000)
        results[profile_name] = times
    print(f"\n{n_queries} queries per profile, {n_chunks} chunks (latency in ms, database side only)")
    print(f"{'profile':<16}{'stage':<10}{'p50':>8}{'p95':>8}{'p99':>8}")
    for prof_name, times in results.items():
        for stage, vals in times.items():
            print(
                f"{prof_name:<16}{stage:<10}{statistics.median(vals):>8.1f}{pct(vals, 95):>8.1f}{pct(vals, 99):>8.1f}"
            )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=int, default=50000)
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("--reuse", action="store_true", help="Query the data loaded by an earlier run.")
    args = ap.parse_args()
    asyncio.run(run(args.chunks, args.queries, args.reuse))
