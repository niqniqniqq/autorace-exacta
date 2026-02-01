"""Pydantic response schemas for the API."""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    version: str


class RaceDayOut(BaseModel):
    race_day_id: int
    track_code: str
    race_date: date
    is_cancelled: bool

    model_config = {"from_attributes": True}


class RaceOut(BaseModel):
    race_id: int
    race_no: int
    distance_m: int | None = None
    weather: str | None = None
    start_time: str | None = None
    is_cancelled: bool = False

    model_config = {"from_attributes": True}


class EntryOut(BaseModel):
    car_no: int
    racer_name: str
    handicap_m: int | None = None
    trial_time: float | None = None
    deviation: float | None = None
    quinella_rate: float | None = None
    trio_rate: float | None = None

    model_config = {"from_attributes": True}


class RaceDetailOut(BaseModel):
    race: RaceOut
    entries: list[EntryOut]

    model_config = {"from_attributes": True}


class PredictionOut(BaseModel):
    first_car_no: int
    second_car_no: int
    prob: float
    fair_odds: float
    market_odds: float | None = None
    ev: float | None = None
    model_version: str
    predicted_at: datetime

    model_config = {"from_attributes": True}
