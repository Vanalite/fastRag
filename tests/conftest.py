"""Test setup. Database tests need Postgres with pgvector:

    docker compose up -d db   # or any pgvector/pgvector:pg17 container
    FASTRAG_TEST_DATABASE_URL=postgresql+asyncpg://fastrag:fastrag@localhost:5432/fastrag_test pytest

The test database is dropped and recreated on every run.
"""

from __future__ import annotations

import os
import tempfile

TEST_DB = os.environ.get(
    "FASTRAG_TEST_DATABASE_URL", "postgresql+asyncpg://fastrag:fastrag@localhost:5432/fastrag_test"
)
# Settings are read once and cached, so set them before importing fastrag.
os.environ.update(
    {
        "FASTRAG_ENV": "test",
        "FASTRAG_DATABASE_URL": TEST_DB,
        "FASTRAG_EMBEDDING_PROVIDER": "hash",
        "FASTRAG_EMBEDDING_DIM": "256",
        "FASTRAG_RERANKER_PROVIDER": "none",
        "FASTRAG_STORAGE_PATH": tempfile.mkdtemp(prefix="fastrag-test-"),
        "FASTRAG_LOCAL_SOURCES_ROOT": tempfile.mkdtemp(prefix="fastrag-src-"),
        "FASTRAG_ADMIN_TOKEN": "test-operator-token",
        "FASTRAG_LOG_JSON": "false",
        "FASTRAG_SEARCH_LOG_ENABLED": "true",
    }
)

import asyncio  # noqa: E402
import re  # noqa: E402
from collections.abc import AsyncIterator  # noqa: E402

import asyncpg  # noqa: E402
import pytest  # noqa: E402
from sqlalchemy import text  # noqa: E402

from fastrag.config import get_settings  # noqa: E402
from fastrag.services import Services, build_services  # noqa: E402
from fastrag.text.vietnamese import strip_diacritics  # noqa: E402


class OverlapReranker:
    """Deterministic stand-in for a cross-encoder: word overlap with the query."""

    name = "test:overlap"
    fail = False

    async def rerank(self, query: str, texts: list[str]) -> list[float]:
        if self.fail:
            raise RuntimeError("reranker down")

        def words(s: str) -> set[str]:
            return {w for w in re.findall(r"\w+", s.lower()) if len(w) > 1}

        q = words(query)
        out = []
        for t in texts:
            w = words(t)
            plain = {strip_diacritics(x) for x in w}
            exact = len(q & w)
            loose = len({strip_diacritics(x) for x in q - w} & plain)
            out.append((exact + 0.5 * loose) / (len(q) or 1))
        return out

    async def aclose(self) -> None:
        return None


def _db_available() -> bool:
    async def probe() -> bool:
        url = TEST_DB.replace("+asyncpg", "").rsplit("/", 1)[0] + "/postgres"
        try:
            conn = await asyncpg.connect(url, timeout=3)
        except Exception:
            return False
        await conn.close()
        return True

    return asyncio.run(probe())


DB_AVAILABLE = _db_available()


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if DB_AVAILABLE:
        return
    skip = pytest.mark.skip(reason=f"Postgres not reachable at {TEST_DB}")
    for item in items:
        if "db" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def migrated_db() -> str:
    if not DB_AVAILABLE:
        pytest.skip("no database")
    base = TEST_DB.replace("+asyncpg", "")
    admin_url, dbname = base.rsplit("/", 1)

    async def recreate() -> None:
        conn = await asyncpg.connect(admin_url + "/postgres")
        await conn.execute(f'DROP DATABASE IF EXISTS "{dbname}" WITH (FORCE)')
        await conn.execute(f'CREATE DATABASE "{dbname}"')
        await conn.close()

    asyncio.run(recreate())
    from fastrag.db.migrate import upgrade

    upgrade(TEST_DB)
    return TEST_DB


@pytest.fixture(scope="session")
def services(migrated_db: str) -> Services:
    svc = build_services(get_settings())
    svc.reranker = OverlapReranker()
    return svc


TABLES = (
    "search_logs, jobs, listings, listing_snapshots, sheet_mappings, collection_stats, term_stats, chunks, "
    "documents, sources, collections, api_keys, tenants"
)


@pytest.fixture
async def db(migrated_db: str) -> AsyncIterator[None]:
    from fastrag.api.auth import clear_auth_cache
    from fastrag.db.engine import get_engine

    async with get_engine().begin() as conn:
        await conn.execute(text(f"TRUNCATE {TABLES} RESTART IDENTITY CASCADE"))
    clear_auth_cache()
    yield
