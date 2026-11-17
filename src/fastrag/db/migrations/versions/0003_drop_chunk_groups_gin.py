"""Drop the GIN index on chunks.groups.

With the caller's groups passed as a query parameter, the planner chose this
index plus an exact distance sort over every visible chunk instead of the
HNSW index, making vector search 6x slower at 50k chunks (D22). Group
filtering is cheap as a post-filter on HNSW and BM25 candidates.

Revision ID: 0003
Revises: 0002
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_index("ix_chunks_groups", table_name="chunks", postgresql_using="gin")


def downgrade() -> None:
    op.create_index("ix_chunks_groups", "chunks", ["groups"], postgresql_using="gin")
