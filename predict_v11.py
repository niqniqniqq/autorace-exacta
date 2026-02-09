"""Quick v11 LightGBM prediction script for today's races."""
from __future__ import annotations

import pickle
import sys
from datetime import date, datetime

import numpy as np
from sqlalchemy import select

from app.db.session import get_db
from app.db.models import Race, RaceDay, RaceEntry, OddsExacta, Track


V11_FEATURES = [
    "handicap_m", "trial_time", "start_avg", "deviation",
    "quinella_rate", "trio_rate", "rank_class", "car_no",
]


def extract_v11_features(entry: RaceEntry) -> np.ndarray:
    stats = entry.stats_json or {}
    rank_str = stats.get("rank", "")
    if rank_str.startswith("S"):
        rank_class = 2.0
    elif rank_str.startswith("A"):
        rank_class = 1.0
    else:
        rank_class = 0.0

    return np.array([
        entry.handicap_m or 0,
        entry.trial_time or 0.0,
        entry.start_avg or 0.15,
        entry.deviation or 50.0,
        entry.quinella_rate or 0.0,
        entry.trio_rate or 0.0,
        rank_class,
        entry.car_no,
    ], dtype=np.float64)


def predict_exacta_pl(model, features: np.ndarray, car_nos: list[int]):
    """Plackett-Luce exacta prediction using LightGBM strengths."""
    proba = model.predict_proba(features)
    strengths = proba[:, 1] if proba.shape[1] == 2 else proba[:, 0]

    exp_u = np.exp(strengths - strengths.max())
    total = exp_u.sum()

    results = []
    for i in range(len(car_nos)):
        p1 = exp_u[i] / total
        remaining = total - exp_u[i]
        if remaining <= 0:
            continue
        for j in range(len(car_nos)):
            if i == j:
                continue
            p2 = exp_u[j] / remaining
            prob = float(p1 * p2)
            results.append((car_nos[i], car_nos[j], prob))

    results.sort(key=lambda x: x[2], reverse=True)
    return results


def get_latest_odds(db, race_id: int) -> dict[tuple[int, int], float]:
    """Get the latest odds for a race."""
    rows = db.execute(
        select(OddsExacta)
        .where(OddsExacta.race_id == race_id)
        .order_by(OddsExacta.captured_at.desc())
    ).scalars().all()

    odds_map: dict[tuple[int, int], float] = {}
    for row in rows:
        key = (row.first_car_no, row.second_car_no)
        if key not in odds_map:
            odds_map[key] = row.odds
    return odds_map


def main():
    track_code = sys.argv[1] if len(sys.argv) > 1 else "iizuka"
    target_date = date.today()

    # Load v11 model
    with open("models/model_v11_lgb.pkl", "rb") as f:
        data = pickle.load(f)
    model = data["model"]
    print(f"Model loaded: {type(model).__name__} (features: {data.get('feature_names', [])})")

    with get_db() as db:
        # Get today's races
        races = db.execute(
            select(Race)
            .join(RaceDay)
            .join(Track)
            .where(Track.track_code == track_code)
            .where(RaceDay.race_date == target_date)
            .order_by(Race.race_no)
        ).scalars().all()

        if not races:
            print(f"No races found for {track_code} on {target_date}")
            return

        print(f"\n{'='*80}")
        print(f"  飯塚オートレース 2連単予測  {target_date}  (model: v11 LightGBM)")
        print(f"{'='*80}")

        for race in races:
            entries = sorted(race.entries, key=lambda e: e.car_no)
            if len(entries) < 2:
                continue

            car_nos = [e.car_no for e in entries]
            features = np.array([extract_v11_features(e) for e in entries])

            # Predict
            exacta_preds = predict_exacta_pl(model, features, car_nos)
            odds_map = get_latest_odds(db, race.race_id)

            print(f"\n--- R{race.race_no} ({len(entries)}車) ---")

            # Show entries
            print(f"  {'車番':>4} {'選手':>10} {'ハンデ':>6} {'試走':>6} {'ST平均':>6} {'偏差値':>6} {'級':>3}")
            for e in entries:
                stats = e.stats_json or {}
                rank = stats.get("rank", "?")
                name = e.racer.racer_name if e.racer else "?"
                print(f"  {e.car_no:>4} {name:>10} {(e.handicap_m or 0):>5}m"
                      f" {(e.trial_time or 0):>6.2f} {(e.start_avg or 0):>6.2f}"
                      f" {(e.deviation or 0):>6.1f} {rank:>3}")

            # Show top predictions with EV
            print(f"\n  {'順位':>4} {'組合せ':>8} {'確率':>8} {'適正':>8} {'市場':>8} {'EV':>8} {'判定':>6}")
            print(f"  {'─'*60}")

            ev_positive = []
            for rank_i, (first, second, prob) in enumerate(exacta_preds[:10]):
                fair_odds = 1.0 / prob if prob > 0 else 9999
                market = odds_map.get((first, second), 0)
                ev = (prob * market - 1) * 100 if market > 0 else -100
                verdict = "BUY" if ev > 0 else ""
                if ev > 0:
                    ev_positive.append((first, second, prob, market, ev))

                print(f"  {rank_i+1:>4} {first}-{second:>2}"
                      f" {prob*100:>7.2f}% {fair_odds:>7.1f}x"
                      f" {market:>7.1f}x {ev:>+7.1f}%"
                      f" {'★BUY' if ev > 0 else '':>6}")

            if ev_positive:
                print(f"\n  >>> EV+推奨: ", end="")
                for f, s, p, m, ev in ev_positive:
                    print(f"{f}-{s}(EV{ev:+.0f}% @{m:.1f}x) ", end="")
                print()
            else:
                print(f"\n  >>> EV+組合せなし → スキップ推奨")



if __name__ == "__main__":
    main()
