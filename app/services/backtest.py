"""Backtest service for exacta prediction models."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING

import numpy as np
from sqlalchemy import select, distinct

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

from app.db.models import Race, RaceDay, RaceResult, RaceEntry, OddsExacta, Track

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Result of a single race backtest."""
    race_id: int
    race_no: int
    race_date: date
    track_code: str
    actual_1st: int
    actual_2nd: int
    actual_odds: float
    pred_1st: int
    pred_2nd: int
    pred_prob: float
    pred_odds: float
    hit: bool
    ev_plus_bets: list[tuple[int, int, float, float]]  # (1st, 2nd, prob, odds)
    ev_plus_hit: bool
    profit: float  # Based on EV+ strategy (100 yen per bet)


@dataclass
class BacktestSummary:
    """Summary of backtest results."""
    total_races: int = 0
    top1_hits: int = 0
    ev_plus_races: int = 0  # Races where we had EV+ bets
    ev_plus_hits: int = 0
    total_bets: int = 0
    total_invested: float = 0.0
    total_return: float = 0.0
    results: list[BacktestResult] = field(default_factory=list)

    @property
    def top1_accuracy(self) -> float:
        return self.top1_hits / self.total_races if self.total_races > 0 else 0.0

    @property
    def ev_plus_accuracy(self) -> float:
        return self.ev_plus_hits / self.ev_plus_races if self.ev_plus_races > 0 else 0.0

    @property
    def roi(self) -> float:
        return self.total_return / self.total_invested if self.total_invested > 0 else 0.0

    @property
    def profit(self) -> float:
        return self.total_return - self.total_invested


def get_backtest_races(db: Session) -> list[tuple[Race, RaceResult, RaceDay, Track]]:
    """Get races that have both odds and results."""
    # Get race_ids with odds
    races_with_odds = set(db.scalars(select(distinct(OddsExacta.race_id))).all())

    # Get races with valid results that also have odds
    results = db.execute(
        select(Race, RaceResult, RaceDay, Track)
        .join(RaceResult, Race.race_id == RaceResult.race_id)
        .join(RaceDay, Race.race_day_id == RaceDay.race_day_id)
        .join(Track, RaceDay.track_id == Track.track_id)
        .where(RaceResult.is_valid == True)
        .where(Race.race_id.in_(races_with_odds))
        .order_by(RaceDay.race_date, Race.race_no)
    ).all()

    return [(r.Race, r.RaceResult, r.RaceDay, r.Track) for r in results]


def run_backtest(
    db: Session,
    model,
    feature_extractor,
    bet_amount: float = 100.0,
    min_odds: float = 3.0,
) -> BacktestSummary:
    """Run backtest on all available races.

    Args:
        db: Database session
        model: Model with predict_exacta method
        feature_extractor: Function to extract features from entries
        bet_amount: Amount to bet per combination
        min_odds: Minimum odds for top prediction (skip if below)
    """
    summary = BacktestSummary()
    races = get_backtest_races(db)

    for race, result, race_day, track in races:
        # Get entries
        entries = sorted(race.entries, key=lambda e: e.car_no)
        if len(entries) < 2:
            continue

        # Get odds
        odds_dict: dict[tuple[int, int], float] = {}
        odds_rows = db.scalars(
            select(OddsExacta)
            .where(OddsExacta.race_id == race.race_id)
            .order_by(OddsExacta.captured_at.desc())
        ).all()
        for o in odds_rows:
            key = (o.first_car_no, o.second_car_no)
            if key not in odds_dict:
                odds_dict[key] = o.odds

        # Check if we have enough odds
        required = len(entries) * (len(entries) - 1)
        if len(odds_dict) < required * 0.8:  # At least 80% coverage
            continue

        # Extract features and predict
        car_nos = [e.car_no for e in entries]
        features = np.array([feature_extractor(e) for e in entries])
        preds = model.predict_exacta(features, car_nos)

        if not preds:
            continue

        # Top prediction
        pred_1st, pred_2nd, pred_prob = preds[0]
        pred_odds = odds_dict.get((pred_1st, pred_2nd), 0.0)

        # Actual result
        actual_1st = result.winner_car_no
        actual_2nd = result.second_car_no
        actual_odds = odds_dict.get((actual_1st, actual_2nd), 0.0)

        # Check top1 hit
        hit = (pred_1st == actual_1st and pred_2nd == actual_2nd)

        # Find EV+ bets
        ev_plus_bets = []
        for first, second, prob in preds:
            odds = odds_dict.get((first, second), 0.0)
            if odds > 0:
                ev = prob * odds - 1
                if ev > 0:
                    ev_plus_bets.append((first, second, prob, odds))

        # Check if any EV+ bet hit
        ev_plus_hit = any(
            first == actual_1st and second == actual_2nd
            for first, second, _, _ in ev_plus_bets
        )

        # Calculate profit (EV+ strategy)
        profit = 0.0
        if ev_plus_bets:
            summary.ev_plus_races += 1
            summary.total_bets += len(ev_plus_bets)
            summary.total_invested += bet_amount * len(ev_plus_bets)

            if ev_plus_hit:
                summary.ev_plus_hits += 1
                # Find the winning bet's odds
                for first, second, _, odds in ev_plus_bets:
                    if first == actual_1st and second == actual_2nd:
                        profit = bet_amount * odds - bet_amount * len(ev_plus_bets)
                        summary.total_return += bet_amount * odds
                        break

        # Record result
        br = BacktestResult(
            race_id=race.race_id,
            race_no=race.race_no,
            race_date=race_day.race_date,
            track_code=track.track_code,
            actual_1st=actual_1st,
            actual_2nd=actual_2nd,
            actual_odds=actual_odds,
            pred_1st=pred_1st,
            pred_2nd=pred_2nd,
            pred_prob=pred_prob,
            pred_odds=pred_odds,
            hit=hit,
            ev_plus_bets=ev_plus_bets,
            ev_plus_hit=ev_plus_hit,
            profit=profit,
        )
        summary.results.append(br)
        summary.total_races += 1

        if hit:
            summary.top1_hits += 1

    return summary
