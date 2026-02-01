"""initial schema

Revision ID: 0001
Revises:
Create Date: 2025-01-01 00:00:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "tracks",
        sa.Column("track_id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("track_code", sa.String(32), nullable=False),
        sa.Column("track_name", sa.String(64), nullable=False),
        sa.PrimaryKeyConstraint("track_id"),
        sa.UniqueConstraint("track_code"),
    )

    op.create_table(
        "race_days",
        sa.Column("race_day_id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("track_id", sa.Integer(), nullable=False),
        sa.Column("race_date", sa.Date(), nullable=False),
        sa.Column("is_cancelled", sa.Boolean(), server_default="false"),
        sa.ForeignKeyConstraint(["track_id"], ["tracks.track_id"]),
        sa.PrimaryKeyConstraint("race_day_id"),
        sa.UniqueConstraint("track_id", "race_date"),
    )

    op.create_table(
        "races",
        sa.Column("race_id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("race_day_id", sa.Integer(), nullable=False),
        sa.Column("race_no", sa.Integer(), nullable=False),
        sa.Column("distance_m", sa.Integer()),
        sa.Column("weather", sa.String(16)),
        sa.Column("air_temp_c", sa.Float()),
        sa.Column("humidity", sa.Float()),
        sa.Column("track_temp_c", sa.Float()),
        sa.Column("track_condition", sa.String(16)),
        sa.Column("start_time", sa.String(8)),
        sa.Column("is_cancelled", sa.Boolean(), server_default="false"),
        sa.ForeignKeyConstraint(["race_day_id"], ["race_days.race_day_id"]),
        sa.PrimaryKeyConstraint("race_id"),
        sa.UniqueConstraint("race_day_id", "race_no"),
    )
    op.create_index("ix_races_day_no", "races", ["race_day_id", "race_no"])

    op.create_table(
        "racers",
        sa.Column("racer_id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("racer_code", sa.String(16)),
        sa.Column("racer_name", sa.String(64), nullable=False),
        sa.Column("home_track", sa.String(32)),
        sa.Column("birth_year", sa.Integer()),
        sa.PrimaryKeyConstraint("racer_id"),
        sa.UniqueConstraint("racer_code"),
    )

    op.create_table(
        "raw_snapshots",
        sa.Column("snapshot_id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("source", sa.String(32), nullable=False),
        sa.Column("url", sa.Text(), nullable=False),
        sa.Column("fetched_at", sa.DateTime(), nullable=False),
        sa.Column("http_status", sa.Integer()),
        sa.Column("content_hash", sa.String(64)),
        sa.Column("content_type", sa.String(64)),
        sa.Column("storage_uri", sa.Text()),
        sa.PrimaryKeyConstraint("snapshot_id"),
        sa.UniqueConstraint("source", "url", "fetched_at"),
    )

    op.create_table(
        "race_entries",
        sa.Column("race_entry_id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("race_id", sa.Integer(), nullable=False),
        sa.Column("car_no", sa.Integer(), nullable=False),
        sa.Column("racer_id", sa.Integer(), nullable=False),
        sa.Column("handicap_m", sa.Integer()),
        sa.Column("machine_name", sa.String(64)),
        sa.Column("trial_time", sa.Float()),
        sa.Column("deviation", sa.Float()),
        sa.Column("quinella_rate", sa.Float()),
        sa.Column("trio_rate", sa.Float()),
        sa.Column("stats_json", postgresql.JSONB()),
        sa.Column("snapshot_id", sa.Integer()),
        sa.ForeignKeyConstraint(["race_id"], ["races.race_id"]),
        sa.ForeignKeyConstraint(["racer_id"], ["racers.racer_id"]),
        sa.ForeignKeyConstraint(["snapshot_id"], ["raw_snapshots.snapshot_id"]),
        sa.PrimaryKeyConstraint("race_entry_id"),
        sa.UniqueConstraint("race_id", "car_no"),
    )
    op.create_index("ix_entries_race", "race_entries", ["race_id"])

    op.create_table(
        "race_results",
        sa.Column("race_id", sa.Integer(), nullable=False),
        sa.Column("decided_at", sa.DateTime()),
        sa.Column("winner_car_no", sa.Integer()),
        sa.Column("second_car_no", sa.Integer()),
        sa.Column("third_car_no", sa.Integer()),
        sa.Column("is_valid", sa.Boolean(), server_default="true"),
        sa.Column("raw_result_json", postgresql.JSONB()),
        sa.ForeignKeyConstraint(["race_id"], ["races.race_id"]),
        sa.PrimaryKeyConstraint("race_id"),
    )

    op.create_table(
        "odds_exacta",
        sa.Column("odds_id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("race_id", sa.Integer(), nullable=False),
        sa.Column("first_car_no", sa.Integer(), nullable=False),
        sa.Column("second_car_no", sa.Integer(), nullable=False),
        sa.Column("odds", sa.Float(), nullable=False),
        sa.Column("captured_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["race_id"], ["races.race_id"]),
        sa.PrimaryKeyConstraint("odds_id"),
        sa.UniqueConstraint("race_id", "first_car_no", "second_car_no", "captured_at"),
    )
    op.create_index("ix_odds_race_time", "odds_exacta", ["race_id", "captured_at"])

    op.create_table(
        "payouts_exacta",
        sa.Column("race_id", sa.Integer(), nullable=False),
        sa.Column("first_car_no", sa.Integer(), nullable=False),
        sa.Column("second_car_no", sa.Integer(), nullable=False),
        sa.Column("payout_yen", sa.Integer(), nullable=False),
        sa.Column("popularity_rank", sa.Integer()),
        sa.Column("decided_at", sa.DateTime()),
        sa.ForeignKeyConstraint(["race_id"], ["races.race_id"]),
        sa.PrimaryKeyConstraint("race_id"),
    )

    op.create_table(
        "predictions_exacta",
        sa.Column("pred_id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("race_id", sa.Integer(), nullable=False),
        sa.Column("first_car_no", sa.Integer(), nullable=False),
        sa.Column("second_car_no", sa.Integer(), nullable=False),
        sa.Column("prob", sa.Float(), nullable=False),
        sa.Column("fair_odds", sa.Float(), nullable=False),
        sa.Column("market_odds", sa.Float()),
        sa.Column("ev", sa.Float()),
        sa.Column("model_version", sa.String(32), nullable=False),
        sa.Column("predicted_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["race_id"], ["races.race_id"]),
        sa.PrimaryKeyConstraint("pred_id"),
        sa.UniqueConstraint(
            "race_id", "first_car_no", "second_car_no", "model_version", "predicted_at"
        ),
    )

    op.create_table(
        "model_runs",
        sa.Column("model_run_id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("model_version", sa.String(32), nullable=False),
        sa.Column("train_from", sa.Date(), nullable=False),
        sa.Column("train_to", sa.Date(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("logloss", sa.Float()),
        sa.Column("brier", sa.Float()),
        sa.Column("n_races", sa.Integer()),
        sa.Column("n_samples", sa.Integer()),
        sa.PrimaryKeyConstraint("model_run_id"),
    )


def downgrade() -> None:
    op.drop_table("model_runs")
    op.drop_table("predictions_exacta")
    op.drop_table("payouts_exacta")
    op.drop_index("ix_odds_race_time", table_name="odds_exacta")
    op.drop_table("odds_exacta")
    op.drop_table("race_results")
    op.drop_index("ix_entries_race", table_name="race_entries")
    op.drop_table("race_entries")
    op.drop_table("raw_snapshots")
    op.drop_table("racers")
    op.drop_index("ix_races_day_no", table_name="races")
    op.drop_table("races")
    op.drop_table("race_days")
    op.drop_table("tracks")
