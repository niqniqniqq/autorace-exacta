from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.base import Base
from app.db.models import OddsExacta, Race, RaceDay, Track
from app.services.odds_freshness import has_fresh_odds


def _make_session():
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(
        engine, tables=[Track.__table__, RaceDay.__table__, Race.__table__, OddsExacta.__table__]
    )
    return sessionmaker(bind=engine)()


def test_has_fresh_odds_true():
    session = _make_session()
    track = Track(track_code="kawaguchi", track_name="川口")
    race_day = RaceDay(track=track, race_date=date(2026, 2, 1))
    race = Race(race_day=race_day, race_no=1)
    odds = OddsExacta(
        race=race,
        first_car_no=1,
        second_car_no=2,
        odds=12.3,
        captured_at=datetime.now(timezone.utc) - timedelta(minutes=2),
    )
    session.add_all([track, race_day, race, odds])
    session.commit()

    assert has_fresh_odds(session, "kawaguchi", date(2026, 2, 1)) is True


def test_has_fresh_odds_false_when_stale():
    session = _make_session()
    track = Track(track_code="kawaguchi", track_name="川口")
    race_day = RaceDay(track=track, race_date=date(2026, 2, 1))
    race = Race(race_day=race_day, race_no=1)
    odds = OddsExacta(
        race=race,
        first_car_no=1,
        second_car_no=2,
        odds=12.3,
        captured_at=datetime.now(timezone.utc) - timedelta(minutes=10),
    )
    session.add_all([track, race_day, race, odds])
    session.commit()

    assert has_fresh_odds(session, "kawaguchi", date(2026, 2, 1)) is False
