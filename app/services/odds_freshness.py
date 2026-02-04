"""Helpers for odds freshness checks."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import OddsExacta, Race, RaceDay, Track


def has_fresh_odds(
    db: Session,
    track_code: str,
    race_date: date,
    *,
    freshness_minutes: int = 3,
) -> bool:
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=freshness_minutes)

    latest_capture = (
        db.execute(
            select(func.max(OddsExacta.captured_at))
            .select_from(OddsExacta)
            .join(Race, OddsExacta.race_id == Race.race_id)
            .join(RaceDay, Race.race_day_id == RaceDay.race_day_id)
            .join(Track, RaceDay.track_id == Track.track_id)
            .where(Track.track_code == track_code, RaceDay.race_date == race_date)
        )
        .scalar_one_or_none()
    )

    if latest_capture is None:
        return False

    if latest_capture.tzinfo is None:
        latest_capture = latest_capture.replace(tzinfo=timezone.utc)

    return latest_capture >= cutoff
