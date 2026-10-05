"""initial schema: users, humanization_history, usage

Revision ID: 0001_initial
Revises:
Create Date: 2026-10-04
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001_initial"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("email", sa.String(length=320), nullable=False),
        sa.Column("password_hash", sa.String(length=255), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)

    op.create_table(
        "humanization_history",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("original_text", sa.Text(), nullable=False),
        sa.Column("output_text", sa.Text(), nullable=False),
        sa.Column("tone", sa.String(length=32), nullable=False),
        sa.Column("intensity", sa.String(length=16), nullable=False),
        sa.Column("word_count", sa.Integer(), nullable=False),
        sa.Column("quality", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_humanization_history_user_id", "humanization_history", ["user_id"])
    op.create_index("ix_humanization_history_created_at", "humanization_history", ["created_at"])

    op.create_table(
        "usage",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("request_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("words_processed", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_usage_user_id", "usage", ["user_id"], unique=True)


def downgrade() -> None:
    op.drop_index("ix_usage_user_id", table_name="usage")
    op.drop_table("usage")
    op.drop_index("ix_humanization_history_created_at", table_name="humanization_history")
    op.drop_index("ix_humanization_history_user_id", table_name="humanization_history")
    op.drop_table("humanization_history")
    op.drop_index("ix_users_email", table_name="users")
    op.drop_table("users")
