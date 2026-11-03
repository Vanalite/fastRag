"""Fixtures for tests against a real PostgreSQL + pgvector.

Set FASTRAG_TEST_POSTGRES_DSN to run them, e.g. with
``docker run -d --rm --name fastrag-pg -e POSTGRES_PASSWORD=fastrag -e POSTGRES_DB=fastrag
-p 127.0.0.1:55432:5432 pgvector/pgvector:pg17`` and
``FASTRAG_TEST_POSTGRES_DSN=postgresql://postgres:fastrag@127.0.0.1:55432/fastrag``.
Each test gets its own schema, dropped afterwards.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Callable, Iterator

import pytest

DSN = os.environ.get("FASTRAG_TEST_POSTGRES_DSN")


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "integration: needs external services (Postgres)")


@pytest.fixture
def pg_dsn() -> Iterator[Callable[..., str]]:
    """A DSN whose search_path points at a fresh, empty schema (then `public` for the
    pgvector extension). Call it with extra libpq settings, e.g. enable_seqscan='off'."""
    if not DSN:
        pytest.skip("FASTRAG_TEST_POSTGRES_DSN is not set")
    psycopg = pytest.importorskip("psycopg")
    from psycopg import sql
    from psycopg.conninfo import make_conninfo

    schema = f"fastrag_test_{uuid.uuid4().hex[:12]}"
    with psycopg.connect(DSN, autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector SCHEMA public")
        conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))

    def make(**settings: str) -> str:
        options = " ".join(
            [f"-c search_path={schema},public"] + [f"-c {k}={v}" for k, v in settings.items()]
        )
        return make_conninfo(DSN, options=options)

    yield make
    with psycopg.connect(DSN, autocommit=True) as conn:
        conn.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))
