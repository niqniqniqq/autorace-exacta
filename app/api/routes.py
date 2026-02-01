"""API route definitions."""

from __future__ import annotations

from datetime import date

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import select

from app.api.schemas import (
    EntryOut,
    HealthResponse,
    PredictionOut,
    RaceDayOut,
    RaceDetailOut,
    RaceOut,
)
from app.db.models import PredictionExacta, Race, RaceDay, RaceEntry, Racer, Track
from app.db.session import get_db

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", version="0.1.0")


@router.get("/race-days", response_model=list[RaceDayOut])
def list_race_days(
    track: str = Query(..., description="Track code"),
    from_date: date = Query(..., alias="from", description="Start date"),
    to_date: date = Query(..., alias="to", description="End date"),
) -> list[RaceDayOut]:
    with get_db() as db:
        rows = (
            db.execute(
                select(RaceDay, Track.track_code)
                .join(Track)
                .where(
                    Track.track_code == track,
                    RaceDay.race_date >= from_date,
                    RaceDay.race_date <= to_date,
                )
                .order_by(RaceDay.race_date)
            )
            .all()
        )
        return [
            RaceDayOut(
                race_day_id=rd.race_day_id,
                track_code=tc,
                race_date=rd.race_date,
                is_cancelled=rd.is_cancelled,
            )
            for rd, tc in rows
        ]


@router.get("/races/{race_id}", response_model=RaceDetailOut)
def get_race(race_id: int) -> RaceDetailOut:
    with get_db() as db:
        race = db.get(Race, race_id)
        if not race:
            raise HTTPException(status_code=404, detail="Race not found")

        entries = (
            db.execute(
                select(RaceEntry, Racer.racer_name)
                .join(Racer)
                .where(RaceEntry.race_id == race_id)
                .order_by(RaceEntry.car_no)
            )
            .all()
        )

        return RaceDetailOut(
            race=RaceOut(
                race_id=race.race_id,
                race_no=race.race_no,
                distance_m=race.distance_m,
                weather=race.weather,
                start_time=race.start_time,
                is_cancelled=race.is_cancelled,
            ),
            entries=[
                EntryOut(
                    car_no=e.car_no,
                    racer_name=name,
                    handicap_m=e.handicap_m,
                    trial_time=e.trial_time,
                    deviation=e.deviation,
                    quinella_rate=e.quinella_rate,
                    trio_rate=e.trio_rate,
                )
                for e, name in entries
            ],
        )


@router.get("/races/{race_id}/predictions", response_model=list[PredictionOut])
def get_predictions(
    race_id: int,
    top: int = Query(10, ge=1, le=100),
    min_ev: float | None = Query(None, description="Minimum EV filter"),
) -> list[PredictionOut]:
    with get_db() as db:
        race = db.get(Race, race_id)
        if not race:
            raise HTTPException(status_code=404, detail="Race not found")

        latest_at = db.scalar(
            select(PredictionExacta.predicted_at)
            .where(PredictionExacta.race_id == race_id)
            .order_by(PredictionExacta.predicted_at.desc())
            .limit(1)
        )

        if latest_at is None:
            return []

        q = (
            select(PredictionExacta)
            .where(
                PredictionExacta.race_id == race_id,
                PredictionExacta.predicted_at == latest_at,
            )
            .order_by(PredictionExacta.prob.desc())
            .limit(top)
        )

        preds = db.execute(q).scalars().all()

        result = []
        for p in preds:
            if min_ev is not None and (p.ev is None or p.ev < min_ev):
                continue
            result.append(
                PredictionOut(
                    first_car_no=p.first_car_no,
                    second_car_no=p.second_car_no,
                    prob=p.prob,
                    fair_odds=p.fair_odds,
                    market_odds=p.market_odds,
                    ev=p.ev,
                    model_version=p.model_version,
                    predicted_at=p.predicted_at,
                )
            )
        return result
