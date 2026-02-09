"""Guards to prevent mixing tracks and incomplete data."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import OddsExacta, Race, RaceEntry


@dataclass(frozen=True)
class OddsGuardResult:
    race: Race | None
    reason: str
    entries_count: int = 0


@dataclass(frozen=True)
class PredictGuardResult:
    ok: bool
    reason: str
    entries_count: int = 0
    odds_count: int = 0
    required_odds: int = 0
    captured_at: datetime | None = None


def guard_odds_race(
    db: Session,
    *,
    race_day_id: int,
    race_no: int,
    min_entries: int = 7,
) -> OddsGuardResult:
    race = db.scalar(
        select(Race).where(Race.race_day_id == race_day_id, Race.race_no == race_no)
    )
    if race is None:
        return OddsGuardResult(race=None, reason="race_not_found")

    entries_count = db.scalar(
        select(func.count(RaceEntry.race_entry_id)).where(RaceEntry.race_id == race.race_id)
    )
    if entries_count < min_entries:
        return OddsGuardResult(
            race=None, reason="entries_insufficient", entries_count=entries_count
        )

    return OddsGuardResult(race=race, reason="ok", entries_count=entries_count)


def guard_predict_race(
    db: Session,
    *,
    race_id: int,
    min_entries: int = 7,
) -> PredictGuardResult:
    entries_count = db.scalar(
        select(func.count(RaceEntry.race_entry_id)).where(RaceEntry.race_id == race_id)
    )
    if entries_count < min_entries:
        return PredictGuardResult(
            ok=False,
            reason="entries_insufficient",
            entries_count=entries_count,
        )

    latest_captured = db.scalar(
        select(func.max(OddsExacta.captured_at)).where(OddsExacta.race_id == race_id)
    )
    if latest_captured is None:
        return PredictGuardResult(
            ok=False,
            reason="odds_missing",
            entries_count=entries_count,
        )

    odds_count = db.scalar(
        select(func.count(OddsExacta.odds_id)).where(
            OddsExacta.race_id == race_id, OddsExacta.captured_at == latest_captured
        )
    )

    # Derive active entries from odds: count distinct car_nos in odds
    # This handles absent players who are in entries but not in odds
    active_first = db.scalars(
        select(OddsExacta.first_car_no.distinct()).where(
            OddsExacta.race_id == race_id, OddsExacta.captured_at == latest_captured
        )
    ).all()
    active_entries_count = len(active_first) if active_first else entries_count

    # Check minimum active entries
    if active_entries_count < min_entries:
        return PredictGuardResult(
            ok=False,
            reason="entries_insufficient",
            entries_count=active_entries_count,
        )

    required_odds = active_entries_count * (active_entries_count - 1)
    if odds_count < required_odds:
        return PredictGuardResult(
            ok=False,
            reason="odds_insufficient",
            entries_count=active_entries_count,
            odds_count=odds_count,
            required_odds=required_odds,
            captured_at=latest_captured,
        )

    return PredictGuardResult(
        ok=True,
        reason="ok",
        entries_count=active_entries_count,
        odds_count=odds_count,
        required_odds=required_odds,
        captured_at=latest_captured,
    )
