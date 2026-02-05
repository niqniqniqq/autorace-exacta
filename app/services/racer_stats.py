"""Calculate racer historical statistics from past race results."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

from sqlalchemy import and_, select
from sqlalchemy.orm import Session

from app.db.models import Race, RaceDay, RaceEntry, RaceResult, Racer

logger = logging.getLogger(__name__)


@dataclass
class RacerStats:
    """Historical statistics for a racer."""

    racer_id: int
    race_count: int  # Total races in history
    win_count: int  # 1st place finishes
    second_count: int  # 2nd place finishes
    third_count: int  # 3rd place finishes

    @property
    def win_rate(self) -> float:
        """Win rate (1st place)."""
        return self.win_count / self.race_count if self.race_count > 0 else 0.0

    @property
    def place_rate(self) -> float:
        """Place rate (1st or 2nd)."""
        return (self.win_count + self.second_count) / self.race_count if self.race_count > 0 else 0.0

    @property
    def show_rate(self) -> float:
        """Show rate (1st, 2nd, or 3rd)."""
        return (self.win_count + self.second_count + self.third_count) / self.race_count if self.race_count > 0 else 0.0

    @property
    def avg_finish(self) -> float:
        """Approximate average finish position (lower is better)."""
        if self.race_count == 0:
            return 4.5  # Default to middle for unknown
        # Weighted average: 1st=1, 2nd=2, 3rd=3, other=5 (approximate)
        other = self.race_count - self.win_count - self.second_count - self.third_count
        total = (
            self.win_count * 1
            + self.second_count * 2
            + self.third_count * 3
            + other * 5
        )
        return total / self.race_count


def get_racer_stats(
    db: Session,
    racer_id: int,
    before_date: date,
    lookback_days: int = 90,
) -> RacerStats:
    """Get historical stats for a racer before a given date.

    Args:
        db: Database session
        racer_id: The racer to look up
        before_date: Only count races before this date
        lookback_days: How far back to look (default 90 days)

    Returns:
        RacerStats with aggregated statistics
    """
    from datetime import timedelta

    start_date = before_date - timedelta(days=lookback_days)

    # Get all entries for this racer in the lookback period
    entries_with_results = (
        db.execute(
            select(RaceEntry, RaceResult)
            .join(Race, RaceEntry.race_id == Race.race_id)
            .join(RaceDay, Race.race_day_id == RaceDay.race_day_id)
            .outerjoin(RaceResult, Race.race_id == RaceResult.race_id)
            .where(
                and_(
                    RaceEntry.racer_id == racer_id,
                    RaceDay.race_date >= start_date,
                    RaceDay.race_date < before_date,
                )
            )
        )
        .all()
    )

    race_count = 0
    win_count = 0
    second_count = 0
    third_count = 0

    for entry, result in entries_with_results:
        if result is None or not result.is_valid:
            continue
        race_count += 1
        if result.winner_car_no == entry.car_no:
            win_count += 1
        elif result.second_car_no == entry.car_no:
            second_count += 1
        elif result.third_car_no == entry.car_no:
            third_count += 1

    return RacerStats(
        racer_id=racer_id,
        race_count=race_count,
        win_count=win_count,
        second_count=second_count,
        third_count=third_count,
    )


def get_race_racer_stats(
    db: Session,
    race_id: int,
    race_date: date,
    lookback_days: int = 90,
) -> dict[int, RacerStats]:
    """Get historical stats for all racers in a race.

    Returns:
        Dict mapping car_no -> RacerStats
    """
    entries = (
        db.execute(
            select(RaceEntry)
            .where(RaceEntry.race_id == race_id)
            .order_by(RaceEntry.car_no)
        )
        .scalars()
        .all()
    )

    stats_map: dict[int, RacerStats] = {}
    for entry in entries:
        stats = get_racer_stats(db, entry.racer_id, race_date, lookback_days)
        stats_map[entry.car_no] = stats

    return stats_map
