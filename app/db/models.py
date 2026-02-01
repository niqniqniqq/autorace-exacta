from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class Track(Base):
    __tablename__ = "tracks"

    track_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    track_code: Mapped[str] = mapped_column(String(32), unique=True, nullable=False)
    track_name: Mapped[str] = mapped_column(String(64), nullable=False)

    race_days: Mapped[list[RaceDay]] = relationship(back_populates="track")


class RaceDay(Base):
    __tablename__ = "race_days"
    __table_args__ = (UniqueConstraint("track_id", "race_date"),)

    race_day_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    track_id: Mapped[int] = mapped_column(ForeignKey("tracks.track_id"), nullable=False)
    race_date: Mapped[date] = mapped_column(nullable=False)
    is_cancelled: Mapped[bool] = mapped_column(Boolean, default=False)

    track: Mapped[Track] = relationship(back_populates="race_days")
    races: Mapped[list[Race]] = relationship(back_populates="race_day")


class Race(Base):
    __tablename__ = "races"
    __table_args__ = (
        UniqueConstraint("race_day_id", "race_no"),
        Index("ix_races_day_no", "race_day_id", "race_no"),
    )

    race_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    race_day_id: Mapped[int] = mapped_column(ForeignKey("race_days.race_day_id"), nullable=False)
    race_no: Mapped[int] = mapped_column(Integer, nullable=False)
    distance_m: Mapped[int | None] = mapped_column(Integer)
    weather: Mapped[str | None] = mapped_column(String(16))
    air_temp_c: Mapped[float | None] = mapped_column(Float)
    humidity: Mapped[float | None] = mapped_column(Float)
    track_temp_c: Mapped[float | None] = mapped_column(Float)
    track_condition: Mapped[str | None] = mapped_column(String(16))
    start_time: Mapped[str | None] = mapped_column(String(8))
    is_cancelled: Mapped[bool] = mapped_column(Boolean, default=False)

    race_day: Mapped[RaceDay] = relationship(back_populates="races")
    entries: Mapped[list[RaceEntry]] = relationship(back_populates="race")
    result: Mapped[RaceResult | None] = relationship(back_populates="race", uselist=False)
    odds_exacta: Mapped[list[OddsExacta]] = relationship(back_populates="race")
    payout_exacta: Mapped[PayoutExacta | None] = relationship(back_populates="race", uselist=False)
    predictions_exacta: Mapped[list[PredictionExacta]] = relationship(back_populates="race")


class Racer(Base):
    __tablename__ = "racers"

    racer_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    racer_code: Mapped[str | None] = mapped_column(String(16), unique=True)
    racer_name: Mapped[str] = mapped_column(String(64), nullable=False)
    home_track: Mapped[str | None] = mapped_column(String(32))
    birth_year: Mapped[int | None] = mapped_column(Integer)

    entries: Mapped[list[RaceEntry]] = relationship(back_populates="racer")


class RaceEntry(Base):
    __tablename__ = "race_entries"
    __table_args__ = (
        UniqueConstraint("race_id", "car_no"),
        Index("ix_entries_race", "race_id"),
    )

    race_entry_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    race_id: Mapped[int] = mapped_column(ForeignKey("races.race_id"), nullable=False)
    car_no: Mapped[int] = mapped_column(Integer, nullable=False)
    racer_id: Mapped[int] = mapped_column(ForeignKey("racers.racer_id"), nullable=False)
    handicap_m: Mapped[int | None] = mapped_column(Integer)
    machine_name: Mapped[str | None] = mapped_column(String(64))
    trial_time: Mapped[float | None] = mapped_column(Float)
    deviation: Mapped[float | None] = mapped_column(Float)
    quinella_rate: Mapped[float | None] = mapped_column(Float)
    trio_rate: Mapped[float | None] = mapped_column(Float)
    stats_json: Mapped[dict | None] = mapped_column(JSONB)
    snapshot_id: Mapped[int | None] = mapped_column(ForeignKey("raw_snapshots.snapshot_id"))

    race: Mapped[Race] = relationship(back_populates="entries")
    racer: Mapped[Racer] = relationship(back_populates="entries")


class RaceResult(Base):
    __tablename__ = "race_results"

    race_id: Mapped[int] = mapped_column(ForeignKey("races.race_id"), primary_key=True)
    decided_at: Mapped[datetime | None] = mapped_column(DateTime)
    winner_car_no: Mapped[int | None] = mapped_column(Integer)
    second_car_no: Mapped[int | None] = mapped_column(Integer)
    third_car_no: Mapped[int | None] = mapped_column(Integer)
    is_valid: Mapped[bool] = mapped_column(Boolean, default=True)
    raw_result_json: Mapped[dict | None] = mapped_column(JSONB)

    race: Mapped[Race] = relationship(back_populates="result")


class OddsExacta(Base):
    __tablename__ = "odds_exacta"
    __table_args__ = (
        UniqueConstraint("race_id", "first_car_no", "second_car_no", "captured_at"),
        Index("ix_odds_race_time", "race_id", "captured_at"),
    )

    odds_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    race_id: Mapped[int] = mapped_column(ForeignKey("races.race_id"), nullable=False)
    first_car_no: Mapped[int] = mapped_column(Integer, nullable=False)
    second_car_no: Mapped[int] = mapped_column(Integer, nullable=False)
    odds: Mapped[float] = mapped_column(Float, nullable=False)
    captured_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)

    race: Mapped[Race] = relationship(back_populates="odds_exacta")


class PayoutExacta(Base):
    __tablename__ = "payouts_exacta"

    race_id: Mapped[int] = mapped_column(ForeignKey("races.race_id"), primary_key=True)
    first_car_no: Mapped[int] = mapped_column(Integer, nullable=False)
    second_car_no: Mapped[int] = mapped_column(Integer, nullable=False)
    payout_yen: Mapped[int] = mapped_column(Integer, nullable=False)
    popularity_rank: Mapped[int | None] = mapped_column(Integer)
    decided_at: Mapped[datetime | None] = mapped_column(DateTime)

    race: Mapped[Race] = relationship(back_populates="payout_exacta")


class PredictionExacta(Base):
    __tablename__ = "predictions_exacta"
    __table_args__ = (
        UniqueConstraint(
            "race_id", "first_car_no", "second_car_no", "model_version", "predicted_at"
        ),
    )

    pred_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    race_id: Mapped[int] = mapped_column(ForeignKey("races.race_id"), nullable=False)
    first_car_no: Mapped[int] = mapped_column(Integer, nullable=False)
    second_car_no: Mapped[int] = mapped_column(Integer, nullable=False)
    prob: Mapped[float] = mapped_column(Float, nullable=False)
    fair_odds: Mapped[float] = mapped_column(Float, nullable=False)
    market_odds: Mapped[float | None] = mapped_column(Float)
    ev: Mapped[float | None] = mapped_column(Float)
    model_version: Mapped[str] = mapped_column(String(32), nullable=False)
    predicted_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)

    race: Mapped[Race] = relationship(back_populates="predictions_exacta")


class RawSnapshot(Base):
    __tablename__ = "raw_snapshots"
    __table_args__ = (UniqueConstraint("source", "url", "fetched_at"),)

    snapshot_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    url: Mapped[str] = mapped_column(Text, nullable=False)
    fetched_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    http_status: Mapped[int | None] = mapped_column(Integer)
    content_hash: Mapped[str | None] = mapped_column(String(64))
    content_type: Mapped[str | None] = mapped_column(String(64))
    storage_uri: Mapped[str | None] = mapped_column(Text)



class ModelRun(Base):
    __tablename__ = "model_runs"

    model_run_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_version: Mapped[str] = mapped_column(String(32), nullable=False)
    train_from: Mapped[date] = mapped_column(nullable=False)
    train_to: Mapped[date] = mapped_column(nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    logloss: Mapped[float | None] = mapped_column(Float)
    brier: Mapped[float | None] = mapped_column(Float)
    n_races: Mapped[int | None] = mapped_column(Integer)
    n_samples: Mapped[int | None] = mapped_column(Integer)
