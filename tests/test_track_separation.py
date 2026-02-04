from __future__ import annotations

from datetime import date

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import models  # noqa: F401
from app.db.base import Base
from app.services.upsert import upsert_race, upsert_race_day, upsert_track


def make_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def test_tracks_create_distinct_race_days_and_races():
    db = make_session()
    day = date(2026, 2, 4)

    kawaguchi = upsert_track(db, "kawaguchi", "川口")
    kawaguchi2 = upsert_track(db, "kawaguchi2", "川口ナイト")

    rd_day = upsert_race_day(db, kawaguchi.track_id, day)
    rd_night = upsert_race_day(db, kawaguchi2.track_id, day)

    race_day = upsert_race(db, rd_day.race_day_id, 1)
    race_night = upsert_race(db, rd_night.race_day_id, 1)

    assert kawaguchi.track_id != kawaguchi2.track_id
    assert rd_day.race_day_id != rd_night.race_day_id
    assert race_day.race_id != race_night.race_id
