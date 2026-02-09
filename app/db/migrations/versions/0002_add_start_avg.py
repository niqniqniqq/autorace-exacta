"""add start_avg to race_entries

Revision ID: 0002
Revises: 0001
Create Date: 2026-02-09 00:00:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("race_entries", sa.Column("start_avg", sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column("race_entries", "start_avg")
