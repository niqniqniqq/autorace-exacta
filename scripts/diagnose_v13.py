"""Diagnostic script: why does market baseline beat v13 on LogLoss/Brier?

Tests:
1. Runner-level: P(1着) — model vs market implied win prob
2. Pair-level: P(i→j) — Plackett-Luce vs market direct
3. Calibration curve: are model probabilities well-calibrated?
4. Probability mass: where does model assign mass vs market?
"""

from __future__ import annotations

import logging
import math
import sys
from datetime import date

import numpy as np
from sqlalchemy import func, select

logging.basicConfig(level=logging.WARNING, stream=sys.stderr)

from app.db.models import OddsExacta, Race, RaceDay, RaceEntry, RaceResult
from app.db.session import get_db
from app.services.modeling_v13 import (
    ExactaModelV13,
    compute_runner_odds_stats,
    extract_v13_features,
)


def get_latest_odds(db, race_id):
    latest_captured = db.scalar(
        select(func.max(OddsExacta.captured_at)).where(OddsExacta.race_id == race_id)
    )
    if not latest_captured:
        return {}
    rows = db.scalars(
        select(OddsExacta).where(
            OddsExacta.race_id == race_id,
            OddsExacta.captured_at == latest_captured,
        )
    ).all()
    return {(r.first_car_no, r.second_car_no): r.odds for r in rows}


def main():
    model = ExactaModelV13.load("models/model_v13_lgb.pkl")

    # Test period: 2026-01-01 ~ 2026-01-31
    test_from = date(2026, 1, 1)
    test_to = date(2026, 1, 31)

    with get_db() as db:
        races = (
            db.execute(
                select(Race)
                .join(RaceDay)
                .join(RaceResult)
                .where(
                    RaceDay.race_date >= test_from,
                    RaceDay.race_date <= test_to,
                    RaceResult.is_valid == True,
                    RaceResult.winner_car_no.isnot(None),
                    RaceResult.second_car_no.isnot(None),
                )
                .order_by(RaceDay.race_date, Race.race_no)
            )
            .scalars()
            .all()
        )

        # Accumulators
        # Runner-level
        model_win_logloss = []  # -log(P_model(winner))
        market_win_logloss = []  # -log(P_market_implied(winner))

        # Pair-level
        model_pair_logloss = []
        market_pair_logloss = []
        model_pair_probs_actual = []  # P_model for actual pair
        market_pair_probs_actual = []  # P_market for actual pair
        model_top1_hits = 0
        market_top1_hits = 0

        # Calibration bins for model P(1着)
        cal_bins = {i: {"count": 0, "wins": 0} for i in range(10)}

        # Probability mass analysis
        model_winner_rank = []  # rank of actual winner in model's P(1st)
        market_winner_rank = []  # rank of actual winner in market implied

        n_races = 0
        n_skipped_no_odds = 0

        for race in races:
            entries = sorted(race.entries, key=lambda e: e.car_no)
            if len(entries) < 2:
                continue
            result = race.result
            if not result or not result.is_valid:
                continue

            actual_1st = result.winner_car_no
            actual_2nd = result.second_car_no
            car_nos = [e.car_no for e in entries]
            if actual_1st not in car_nos:
                continue

            odds_dict = get_latest_odds(db, race.race_id)
            if not odds_dict:
                n_skipped_no_odds += 1
                continue

            odds_stats = compute_runner_odds_stats(odds_dict, car_nos)
            feats = np.array([extract_v13_features(e, odds_stats) for e in entries])
            n = len(car_nos)

            # ---- Runner-level analysis ----
            model_probs = model.predict_proba(feats)
            model_prob_sum = model_probs.sum()
            if model_prob_sum <= 0:
                continue

            model_probs_norm = model_probs / model_prob_sum

            # Market implied win probs
            market_win_probs = np.array([
                odds_stats[c]["implied_win_prob"] for c in car_nos
            ])

            # Find winner index
            winner_idx = car_nos.index(actual_1st)

            # Logloss at runner level
            eps = 1e-15
            model_win_logloss.append(-math.log(max(model_probs_norm[winner_idx], eps)))
            market_win_logloss.append(-math.log(max(market_win_probs[winner_idx], eps)))

            # Calibration
            for i, c in enumerate(car_nos):
                p = model_probs_norm[i]
                bin_idx = min(int(p * 10), 9)
                cal_bins[bin_idx]["count"] += 1
                if c == actual_1st:
                    cal_bins[bin_idx]["wins"] += 1

            # Winner rank
            model_ranks = np.argsort(-model_probs_norm)
            market_ranks = np.argsort(-market_win_probs)
            model_winner_rank.append(int(np.where(model_ranks == winner_idx)[0][0]))
            market_winner_rank.append(int(np.where(market_ranks == winner_idx)[0][0]))

            # ---- Pair-level analysis ----
            # Model: Plackett-Luce
            preds = model.predict_exacta(feats, car_nos)
            model_pair_map = {(f, s): p for f, s, p in preds}

            # Market: direct from odds
            total_inv = sum(1.0 / o for o in odds_dict.values() if o > 0)
            if total_inv <= 0:
                continue
            market_pair_map = {
                pair: (1.0 / o) / total_inv
                for pair, o in odds_dict.items()
                if o > 0
            }

            actual_pair = (actual_1st, actual_2nd)

            model_p = model_pair_map.get(actual_pair, eps)
            market_p = market_pair_map.get(actual_pair, eps)

            model_pair_logloss.append(-math.log(max(model_p, eps)))
            market_pair_logloss.append(-math.log(max(market_p, eps)))
            model_pair_probs_actual.append(model_p)
            market_pair_probs_actual.append(market_p)

            # Top-1
            if model_pair_map:
                model_top = max(model_pair_map, key=model_pair_map.get)
                if model_top == actual_pair:
                    model_top1_hits += 1
            if market_pair_map:
                market_top = max(market_pair_map, key=market_pair_map.get)
                if market_top == actual_pair:
                    market_top1_hits += 1

            n_races += 1

        # ---- Print Results ----
        print(f"\n{'='*60}")
        print(f"DIAGNOSTIC REPORT: v13 vs Market Baseline")
        print(f"Period: {test_from} ~ {test_to}")
        print(f"Races: {n_races} (skipped no-odds: {n_skipped_no_odds})")
        print(f"{'='*60}")

        print(f"\n--- 1. Runner-Level: P(1着) ---")
        print(f"  Model  LogLoss(1着): {np.mean(model_win_logloss):.4f}")
        print(f"  Market LogLoss(1着): {np.mean(market_win_logloss):.4f}")
        diff = np.mean(model_win_logloss) - np.mean(market_win_logloss)
        print(f"  Delta: {diff:+.4f} {'(model worse)' if diff > 0 else '(model better)'}")

        print(f"\n--- 2. Pair-Level: P(i→j) ---")
        print(f"  Model  LogLoss(pair): {np.mean(model_pair_logloss):.4f}")
        print(f"  Market LogLoss(pair): {np.mean(market_pair_logloss):.4f}")
        diff = np.mean(model_pair_logloss) - np.mean(market_pair_logloss)
        print(f"  Delta: {diff:+.4f} {'(model worse)' if diff > 0 else '(model better)'}")
        print(f"\n  Model  Top-1: {model_top1_hits}/{n_races} ({model_top1_hits/n_races:.1%})")
        print(f"  Market Top-1: {market_top1_hits}/{n_races} ({market_top1_hits/n_races:.1%})")

        print(f"\n--- 3. Decomposition: Runner vs PL conversion ---")
        runner_gap = np.mean(model_win_logloss) - np.mean(market_win_logloss)
        pair_gap = np.mean(model_pair_logloss) - np.mean(market_pair_logloss)
        pl_penalty = pair_gap - runner_gap
        print(f"  Runner-level gap:     {runner_gap:+.4f}")
        print(f"  Pair-level gap:       {pair_gap:+.4f}")
        print(f"  PL conversion penalty:{pl_penalty:+.4f}")
        print(f"  => {'PL conversion adds significant error' if abs(pl_penalty) > 0.05 else 'Gap mainly at runner level'}")

        print(f"\n--- 4. Calibration: Model P(1着) ---")
        print(f"  {'Bin':>12s}  {'Count':>6s}  {'Wins':>5s}  {'Predicted':>10s}  {'Actual':>8s}")
        for i in range(10):
            cnt = cal_bins[i]["count"]
            wins = cal_bins[i]["wins"]
            pred_mid = (i + 0.5) / 10
            actual_rate = wins / cnt if cnt > 0 else 0
            bar = "#" * int(actual_rate * 50)
            print(f"  {i/10:.1f}-{(i+1)/10:.1f}  {cnt:6d}  {wins:5d}  {pred_mid:10.2f}  {actual_rate:8.3f}  {bar}")

        print(f"\n--- 5. Winner Rank Distribution ---")
        print(f"  {'Rank':>6s}  {'Model':>8s}  {'Market':>8s}")
        for r in range(min(8, max(max(model_winner_rank), max(market_winner_rank)) + 1)):
            m_cnt = sum(1 for x in model_winner_rank if x == r)
            k_cnt = sum(1 for x in market_winner_rank if x == r)
            print(f"  {r+1:6d}  {m_cnt:6d} ({m_cnt/n_races:.1%})  {k_cnt:6d} ({k_cnt/n_races:.1%})")

        print(f"\n--- 6. Prob assigned to actual pair ---")
        m_arr = np.array(model_pair_probs_actual)
        k_arr = np.array(market_pair_probs_actual)
        print(f"  Model  mean={m_arr.mean():.4f}  median={np.median(m_arr):.4f}  std={m_arr.std():.4f}")
        print(f"  Market mean={k_arr.mean():.4f}  median={np.median(k_arr):.4f}  std={k_arr.std():.4f}")
        ratio = m_arr / np.maximum(k_arr, 1e-15)
        print(f"  Ratio (model/market): mean={ratio.mean():.4f}  median={np.median(ratio):.4f}")
        print(f"  Model > Market: {(m_arr > k_arr).sum()}/{n_races} ({(m_arr > k_arr).mean():.1%})")


if __name__ == "__main__":
    main()
