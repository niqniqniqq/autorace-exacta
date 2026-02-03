"""Idempotent upsert operations for all DB entities."""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import (
    ModelRun,
    OddsExacta,
    PayoutExacta,
    PredictionExacta,
    Race,
    RaceDay,
    RaceEntry,
    RaceResult,
    Racer,
    RawSnapshot,
    Track,
)
from app.scraping.parsers.odds_parser import ExactaOddsData
from app.scraping.parsers.program_parser import EntryData
from app.scraping.parsers.results_parser import ExactaPayoutData, ResultData

logger = logging.getLogger(__name__)


def upsert_track(db: Session, track_code: str, track_name: str) -> Track:
    track = db.scalar(select(Track).where(Track.track_code == track_code))
    if track is None:
        track = Track(track_code=track_code, track_name=track_name)
        db.add(track)
        db.flush()
        logger.info("Created track: %s", track_code)
    return track


def upsert_race_day(db: Session, track_id: int, race_date: date) -> RaceDay:
    rd = db.scalar(
        select(RaceDay).where(
            RaceDay.track_id == track_id, RaceDay.race_date == race_date
        )
    )
    if rd is None:
        rd = RaceDay(track_id=track_id, race_date=race_date, is_cancelled=False)
        db.add(rd)
        db.flush()
        logger.info("Created race_day: %s", race_date)
    return rd


def upsert_race(db: Session, race_day_id: int, race_no: int) -> Race:
    race = db.scalar(
        select(Race).where(Race.race_day_id == race_day_id, Race.race_no == race_no)
    )
    if race is None:
        race = Race(race_day_id=race_day_id, race_no=race_no)
        db.add(race)
        db.flush()
    return race


def upsert_racer(db: Session, racer_code: str | None, racer_name: str) -> Racer:
    if racer_code:
        racer = db.scalar(select(Racer).where(Racer.racer_code == racer_code))
        if racer:
            return racer
    racer = Racer(racer_code=racer_code, racer_name=racer_name)
    db.add(racer)
    db.flush()
    return racer


def upsert_entry(
    db: Session, race_id: int, entry: EntryData, snapshot_id: int | None = None
) -> RaceEntry:
    existing = db.scalar(
        select(RaceEntry).where(
            RaceEntry.race_id == race_id, RaceEntry.car_no == entry.car_no
        )
    )
    racer = upsert_racer(db, entry.racer_code, entry.racer_name)

    if existing:
        existing.racer_id = racer.racer_id
        existing.handicap_m = entry.handicap_m
        existing.machine_name = entry.machine_name
        existing.trial_time = entry.trial_time
        existing.deviation = entry.deviation
        existing.quinella_rate = entry.quinella_rate
        existing.trio_rate = entry.trio_rate
        existing.stats_json = entry.stats_json
        if snapshot_id:
            existing.snapshot_id = snapshot_id
        return existing

    re = RaceEntry(
        race_id=race_id,
        car_no=entry.car_no,
        racer_id=racer.racer_id,
        handicap_m=entry.handicap_m,
        machine_name=entry.machine_name,
        trial_time=entry.trial_time,
        deviation=entry.deviation,
        quinella_rate=entry.quinella_rate,
        trio_rate=entry.trio_rate,
        stats_json=entry.stats_json,
        snapshot_id=snapshot_id,
    )
    db.add(re)
    db.flush()
    return re


def upsert_snapshot(
    db: Session,
    source: str,
    url: str,
    fetched_at: datetime,
    http_status: int,
    content_hash: str,
    content_type: str,
    storage_uri: str,
) -> RawSnapshot:
    # Skip if identical content already stored for the same source+url
    existing = db.scalar(
        select(RawSnapshot).where(
            RawSnapshot.source == source,
            RawSnapshot.url == url,
            RawSnapshot.content_hash == content_hash,
        )
    )
    if existing:
        return existing

    snap = RawSnapshot(
        source=source,
        url=url,
        fetched_at=fetched_at,
        http_status=http_status,
        content_hash=content_hash,
        content_type=content_type,
        storage_uri=storage_uri,
    )
    db.add(snap)
    db.flush()
    return snap


def upsert_odds_exacta(
    db: Session, race_id: int, odds_list: list[ExactaOddsData], captured_at: datetime
) -> int:
    count = 0
    for od in odds_list:
        existing = db.scalar(
            select(OddsExacta).where(
                OddsExacta.race_id == race_id,
                OddsExacta.first_car_no == od.first_car_no,
                OddsExacta.second_car_no == od.second_car_no,
                OddsExacta.captured_at == captured_at,
            )
        )
        if existing:
            existing.odds = od.odds
        else:
            db.add(
                OddsExacta(
                    race_id=race_id,
                    first_car_no=od.first_car_no,
                    second_car_no=od.second_car_no,
                    odds=od.odds,
                    captured_at=captured_at,
                )
            )
            count += 1
    db.flush()
    return count


def upsert_result(
    db: Session, race_id: int, result: ResultData
) -> RaceResult:
    existing = db.get(RaceResult, race_id)
    if existing:
        existing.winner_car_no = result.order[0] if len(result.order) > 0 else None
        existing.second_car_no = result.order[1] if len(result.order) > 1 else None
        existing.third_car_no = result.order[2] if len(result.order) > 2 else None
        existing.is_valid = result.is_valid
        existing.raw_result_json = result.raw_json
        return existing

    rr = RaceResult(
        race_id=race_id,
        decided_at=datetime.now(UTC),
        winner_car_no=result.order[0] if len(result.order) > 0 else None,
        second_car_no=result.order[1] if len(result.order) > 1 else None,
        third_car_no=result.order[2] if len(result.order) > 2 else None,
        is_valid=result.is_valid,
        raw_result_json=result.raw_json,
    )
    db.add(rr)
    db.flush()
    return rr


def upsert_payout_exacta(
    db: Session, race_id: int, payout: ExactaPayoutData
) -> PayoutExacta:
    existing = db.get(PayoutExacta, race_id)
    if existing:
        existing.first_car_no = payout.first_car_no
        existing.second_car_no = payout.second_car_no
        existing.payout_yen = payout.payout_yen
        existing.popularity_rank = payout.popularity_rank
        return existing

    pe = PayoutExacta(
        race_id=race_id,
        first_car_no=payout.first_car_no,
        second_car_no=payout.second_car_no,
        payout_yen=payout.payout_yen,
        popularity_rank=payout.popularity_rank,
        decided_at=datetime.now(UTC),
    )
    db.add(pe)
    db.flush()
    return pe


def upsert_prediction_exacta(
    db: Session,
    race_id: int,
    first_car_no: int,
    second_car_no: int,
    prob: float,
    fair_odds: float,
    model_version: str,
    predicted_at: datetime,
    market_odds: float | None = None,
    ev: float | None = None,
) -> PredictionExacta:
    existing = db.scalar(
        select(PredictionExacta).where(
            PredictionExacta.race_id == race_id,
            PredictionExacta.first_car_no == first_car_no,
            PredictionExacta.second_car_no == second_car_no,
            PredictionExacta.model_version == model_version,
            PredictionExacta.predicted_at == predicted_at,
        )
    )
    if existing:
        existing.prob = prob
        existing.fair_odds = fair_odds
        existing.market_odds = market_odds
        existing.ev = ev
        return existing

    pred = PredictionExacta(
        race_id=race_id,
        first_car_no=first_car_no,
        second_car_no=second_car_no,
        prob=prob,
        fair_odds=fair_odds,
        market_odds=market_odds,
        ev=ev,
        model_version=model_version,
        predicted_at=predicted_at,
    )
    db.add(pred)
    db.flush()
    return pred


def upsert_model_run(
    db: Session,
    model_version: str,
    train_from: date,
    train_to: date,
    created_at: datetime,
    logloss: float | None = None,
    brier: float | None = None,
    n_races: int | None = None,
    n_samples: int | None = None,
) -> ModelRun:
    mr = ModelRun(
        model_version=model_version,
        train_from=train_from,
        train_to=train_to,
        created_at=created_at,
        logloss=logloss,
        brier=brier,
        n_races=n_races,
        n_samples=n_samples,
    )
    db.add(mr)
    db.flush()
    return mr
