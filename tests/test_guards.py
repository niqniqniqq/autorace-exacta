from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.base import Base
from app.db import models  # noqa: F401
from app.db.models import OddsExacta
from app.scraping.parsers.program_parser import EntryData
from app.services.guards import guard_odds_race, guard_predict_race
from app.services.upsert import upsert_entry, upsert_race, upsert_race_day, upsert_track


def make_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def _seed_entries(db, race_id: int, count: int) -> None:
    for car_no in range(1, count + 1):
        entry = EntryData(
            car_no=car_no,
            racer_code=f"R{car_no:02d}",
            racer_name=f"Racer {car_no}",
            handicap_m=0,
            machine_name=None,
            trial_time=None,
            deviation=None,
            quinella_rate=None,
            trio_rate=None,
            stats_json={},
        )
        upsert_entry(db, race_id, entry)


def test_guard_odds_skips_without_entries():
    db = make_session()
    track = upsert_track(db, "kawaguchi", "川口")
    rd = upsert_race_day(
        db, track.track_id, datetime(2026, 2, 4, tzinfo=timezone.utc).date()
    )
    upsert_race(db, rd.race_day_id, 1)

    guard = guard_odds_race(db, race_day_id=rd.race_day_id, race_no=1)
    assert guard.race is None
    assert guard.reason == "entries_insufficient"


def test_guard_predict_skips_with_insufficient_odds():
    db = make_session()
    track = upsert_track(db, "kawaguchi", "川口")
    rd = upsert_race_day(
        db, track.track_id, datetime(2026, 2, 4, tzinfo=timezone.utc).date()
    )
    race = upsert_race(db, rd.race_day_id, 1)
    _seed_entries(db, race.race_id, 7)

    captured_at = datetime(2026, 2, 4, tzinfo=timezone.utc)
    for idx in range(10):
        db.add(
            OddsExacta(
                race_id=race.race_id,
                first_car_no=1,
                second_car_no=idx + 2,
                odds=5.0,
                captured_at=captured_at,
            )
        )
    db.flush()

    guard = guard_predict_race(db, race_id=race.race_id)
    assert not guard.ok
    assert guard.reason == "odds_insufficient"
