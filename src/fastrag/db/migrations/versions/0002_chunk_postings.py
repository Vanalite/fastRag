"""Inverted index table for BM25 (D15).

Keyword search used to read each candidate chunk's term-frequency JSON, which
took seconds once query terms were common. Postings with a covering primary
key make it an index-only scan.

Revision ID: 0002
Revises: 0001
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "chunk_postings",
        sa.Column("collection_id", sa.UUID(), nullable=False),
        sa.Column("term", sa.Text(), nullable=False),
        sa.Column("chunk_id", sa.BigInteger(), nullable=False),
        sa.Column("tf", sa.SmallInteger(), nullable=False),
        sa.ForeignKeyConstraint(["chunk_id"], ["chunks.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["collection_id"], ["collections.id"], ondelete="CASCADE"),
    )
    op.execute(
        "ALTER TABLE chunk_postings ADD CONSTRAINT chunk_postings_pkey "
        "PRIMARY KEY (collection_id, term, chunk_id) INCLUDE (tf)"
    )
    op.create_index("ix_chunk_postings_chunk", "chunk_postings", ["chunk_id"])
    op.execute(
        "INSERT INTO chunk_postings (collection_id, term, chunk_id, tf) "
        "SELECT c.collection_id, t.key, c.id, LEAST((t.value)::int, 32767) FROM chunks c, jsonb_each_text(c.tf) t"
    )
    op.drop_column("chunks", "tf")
    op.drop_index("ix_chunks_terms", table_name="chunks", postgresql_using="gin")


def downgrade() -> None:
    op.create_index("ix_chunks_terms", "chunks", ["terms"], postgresql_using="gin")
    op.add_column("chunks", sa.Column("tf", postgresql.JSONB(), nullable=True))
    op.execute(
        "UPDATE chunks c SET tf = p.tf FROM (SELECT chunk_id, jsonb_object_agg(term, tf) AS tf "
        "FROM chunk_postings GROUP BY chunk_id) p WHERE p.chunk_id = c.id"
    )
    op.execute("UPDATE chunks SET tf = '{}'::jsonb WHERE tf IS NULL")
    op.alter_column("chunks", "tf", nullable=False)
    op.drop_table("chunk_postings")
