"""Diagnose Plackett-Luce conversion: where does it fail?

Compare:
- Model P(1st) ranking (good) → PL P(i→j) (bad)
- Market P(1st) implied → PL P(i→j) vs Market direct P(i→j)

This tests if PL itself is the problem, or if it's the model's P(1st) distribution.
"""

from __future__ import annotations

import logging
import math
import sys
from datetime import date

import numpy as np
from sqlalchemy import func, select

logging.basicConfig(level=logging.WARNING, stream=sys.stderr)

from app.db.models import OddsExacta, Race, RaceDay, RaceResult
from app.db.session import get_db
from app.services.modeling_v13 import (
    ExactaModelV13,
    compute_runner_odds_stats,
    extract_v13_features,
)


def plackett_luce_pairs(probs, car_nos):
    """Convert P(1st) to P(i→j) using Plackett-Luce."""
    n = len(car_nos)
    prob_sum = probs.sum()
    if prob_sum <= 0:
        prob_sum = 1.0
    result = {}
    for i in range(n):
        p_i = probs[i] / prob_sum
        remaining = probs.copy()
        remaining[i] = 0
        remaining_sum = remaining.sum()
        if remaining_sum <= 0:
            continue
        for j in range(n):
            if i == j:
                continue
            p_j_given_i = probs[j] / remaining_sum
            result[(car_nos[i], car_nos[j])] = float(p_i * p_j_given_i)
    return result


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

        # Compare 4 approaches for pair probabilities:
        # A) Model P(1st) → PL  (current v13)
        # B) Market P(1st) implied → PL  (use market win probs through PL)
        # C) Market direct: 1/odds / sum(1/odds)  (baseline)
        # D) Model P(1st) → PL, with market 2nd place correction

        ll_model_pl = []
        ll_market_pl = []
        ll_market_direct = []

        # Also: analyze the 2nd place prediction
        second_correct_model = 0
        second_correct_market = 0
        n_where_1st_correct = 0

        n_races = 0

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
            if actual_1st not in car_nos or actual_2nd not in car_nos:
                continue

            odds_dict = get_latest_odds(db, race.race_id)
            if not odds_dict:
                continue

            odds_stats = compute_runner_odds_stats(odds_dict, car_nos)
            feats = np.array([extract_v13_features(e, odds_stats) for e in entries])
            n = len(car_nos)

            # A) Model P(1st) → PL
            model_probs = model.predict_proba(feats)
            model_pl = plackett_luce_pairs(model_probs, car_nos)

            # B) Market P(1st) implied → PL
            market_win = np.array([odds_stats[c]["implied_win_prob"] for c in car_nos])
            market_pl = plackett_luce_pairs(market_win, car_nos)

            # C) Market direct
            total_inv = sum(1.0 / o for o in odds_dict.values() if o > 0)
            if total_inv <= 0:
                continue
            market_direct = {
                pair: (1.0 / o) / total_inv for pair, o in odds_dict.items() if o > 0
            }

            actual = (actual_1st, actual_2nd)
            eps = 1e-15

            ll_model_pl.append(-math.log(max(model_pl.get(actual, eps), eps)))
            ll_market_pl.append(-math.log(max(market_pl.get(actual, eps), eps)))
            ll_market_direct.append(-math.log(max(market_direct.get(actual, eps), eps)))

            # 2nd place analysis: when model gets 1st right, how often is 2nd right?
            model_norm = model_probs / model_probs.sum()
            winner_idx = car_nos.index(actual_1st)
            if np.argmax(model_norm) == winner_idx:
                n_where_1st_correct += 1
                # Model's predicted 2nd
                remaining = model_probs.copy()
                remaining[winner_idx] = 0
                model_2nd_idx = np.argmax(remaining)
                if car_nos[model_2nd_idx] == actual_2nd:
                    second_correct_model += 1

                # Market's predicted 2nd
                remaining_m = market_win.copy()
                remaining_m[winner_idx] = 0
                market_2nd_idx = np.argmax(remaining_m)
                if car_nos[market_2nd_idx] == actual_2nd:
                    second_correct_market += 1

            n_races += 1

        print(f"\n{'='*60}")
        print(f"PL CONVERSION ANALYSIS")
        print(f"Period: {test_from} ~ {test_to}, Races: {n_races}")
        print(f"{'='*60}")

        print(f"\n--- Pair-Level LogLoss Comparison ---")
        print(f"  A) Model P(1st) → PL:     {np.mean(ll_model_pl):.4f}")
        print(f"  B) Market P(1st) → PL:    {np.mean(ll_market_pl):.4f}")
        print(f"  C) Market Direct (1/odds): {np.mean(ll_market_direct):.4f}")

        gap_ab = np.mean(ll_model_pl) - np.mean(ll_market_pl)
        gap_bc = np.mean(ll_market_pl) - np.mean(ll_market_direct)
        gap_ac = np.mean(ll_model_pl) - np.mean(ll_market_direct)

        print(f"\n--- Gap Decomposition ---")
        print(f"  A-B (model vs market, same PL): {gap_ab:+.4f}  <- P(1st) quality gap")
        print(f"  B-C (market PL vs market direct): {gap_bc:+.4f}  <- PL conversion loss")
        print(f"  A-C (total model vs baseline):    {gap_ac:+.4f}  <- total gap")
        print(f"\n  P(1st) quality accounts for: {abs(gap_ab)/abs(gap_ac)*100:.0f}%")
        print(f"  PL conversion accounts for:  {abs(gap_bc)/abs(gap_ac)*100:.0f}%")

        print(f"\n--- 2nd Place Prediction (when 1st is correct) ---")
        print(f"  Races where model got 1st right: {n_where_1st_correct}")
        if n_where_1st_correct > 0:
            print(f"  Model 2nd correct: {second_correct_model}/{n_where_1st_correct} ({second_correct_model/n_where_1st_correct:.1%})")
            print(f"  Market 2nd correct: {second_correct_market}/{n_where_1st_correct} ({second_correct_market/n_where_1st_correct:.1%})")

        # Entropy comparison: how spread out are probabilities?
        print(f"\n--- Probability Distribution Shape ---")
        # Collect entropy stats
        model_entropies = []
        market_entropies = []
        model_max_probs = []
        market_max_probs = []

        for ll_m, ll_k, ll_d in zip(ll_model_pl, ll_market_pl, ll_market_direct):
            pass  # already computed above

        # Recompute for entropy
        for race in races:
            entries = sorted(race.entries, key=lambda e: e.car_no)
            if len(entries) < 2:
                continue
            result = race.result
            if not result or not result.is_valid:
                continue
            car_nos = [e.car_no for e in entries]
            odds_dict = get_latest_odds(db, race.race_id)
            if not odds_dict:
                continue
            odds_stats = compute_runner_odds_stats(odds_dict, car_nos)
            feats = np.array([extract_v13_features(e, odds_stats) for e in entries])

            model_probs = model.predict_proba(feats)
            model_norm = model_probs / model_probs.sum()

            total_inv = sum(1.0 / o for o in odds_dict.values() if o > 0)
            if total_inv <= 0:
                continue
            market_direct = {
                pair: (1.0 / o) / total_inv for pair, o in odds_dict.items() if o > 0
            }

            model_pl = plackett_luce_pairs(model_probs, car_nos)

            # Entropy of pair distributions
            m_vals = np.array(list(model_pl.values()))
            m_vals = m_vals[m_vals > 0]
            model_entropies.append(-np.sum(m_vals * np.log(m_vals)))

            k_vals = np.array(list(market_direct.values()))
            k_vals = k_vals[k_vals > 0]
            market_entropies.append(-np.sum(k_vals * np.log(k_vals)))

            model_max_probs.append(max(model_pl.values()))
            market_max_probs.append(max(market_direct.values()))

        print(f"  Model PL  entropy: mean={np.mean(model_entropies):.3f}")
        print(f"  Market    entropy: mean={np.mean(market_entropies):.3f}")
        print(f"  Model PL  max_prob: mean={np.mean(model_max_probs):.4f}")
        print(f"  Market    max_prob: mean={np.mean(market_max_probs):.4f}")

        diff_ent = np.mean(model_entropies) - np.mean(market_entropies)
        print(f"  Entropy diff: {diff_ent:+.3f} {'(model more uniform)' if diff_ent > 0 else '(model more peaked)'}")


if __name__ == "__main__":
    main()
