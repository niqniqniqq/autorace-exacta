"""Test temperature scaling effect on v13 pair-level LogLoss.

Sweep T from 0.1 to 2.0 and find optimal temperature.
Also test market blend as comparison.
"""

from __future__ import annotations

import logging
import math
import sys
from datetime import date

import numpy as np
from scipy.special import expit, logit as scipy_logit
from sqlalchemy import func, select

logging.basicConfig(level=logging.WARNING, stream=sys.stderr)

from app.db.models import OddsExacta, Race, RaceDay, RaceResult
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


def temperature_scale(probs, T):
    """Apply temperature scaling: sharpen (T<1) or soften (T>1) probabilities."""
    eps = 1e-15
    p = np.clip(probs, eps, 1.0 - eps)
    logits = np.log(p / (1 - p))
    scaled = logits / T
    # Softmax-like normalization
    exp_scaled = np.exp(scaled - scaled.max())
    return exp_scaled / exp_scaled.sum()


def plackett_luce_pairs(probs, car_nos):
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


def blend_pairs(model_pairs, market_pairs, alpha):
    """Blend model and market pair probabilities."""
    all_keys = set(model_pairs) | set(market_pairs)
    blended = {}
    for k in all_keys:
        m = model_pairs.get(k, 0.0)
        mk = market_pairs.get(k, 0.0)
        blended[k] = alpha * m + (1 - alpha) * mk
    # Renormalize
    total = sum(blended.values())
    if total > 0:
        blended = {k: v / total for k, v in blended.items()}
    return blended


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

        # Pre-compute race data
        race_data = []
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

            model_probs = model.predict_proba(feats)

            # Market direct
            total_inv = sum(1.0 / o for o in odds_dict.values() if o > 0)
            if total_inv <= 0:
                continue
            market_direct = {
                pair: (1.0 / o) / total_inv for pair, o in odds_dict.items() if o > 0
            }

            race_data.append({
                "car_nos": car_nos,
                "model_probs": model_probs,
                "market_direct": market_direct,
                "actual": (actual_1st, actual_2nd),
            })

        print(f"\nRaces: {len(race_data)}")

        # === Temperature Scaling Sweep ===
        print(f"\n{'='*60}")
        print(f"TEMPERATURE SCALING SWEEP")
        print(f"{'='*60}")
        print(f"  {'T':>5s}  {'LogLoss':>8s}  {'Top1':>6s}  {'MaxProb':>8s}  {'Entropy':>8s}")

        temps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0]
        best_t = 1.0
        best_ll = float("inf")

        for T in temps:
            logloss_sum = 0
            top1_hits = 0
            max_probs = []
            entropies = []

            for rd in race_data:
                scaled = temperature_scale(rd["model_probs"], T)
                pairs = plackett_luce_pairs(scaled, rd["car_nos"])

                eps = 1e-15
                p = pairs.get(rd["actual"], eps)
                logloss_sum += -math.log(max(p, eps))

                if pairs:
                    top = max(pairs, key=pairs.get)
                    if top == rd["actual"]:
                        top1_hits += 1

                vals = np.array(list(pairs.values()))
                vals = vals[vals > 0]
                max_probs.append(vals.max())
                entropies.append(-np.sum(vals * np.log(vals)))

            ll = logloss_sum / len(race_data)
            t1 = top1_hits / len(race_data)
            if ll < best_ll:
                best_ll = ll
                best_t = T

            marker = " <-- best" if T == best_t and T == temps[-1] or (ll == best_ll) else ""
            print(f"  {T:5.2f}  {ll:8.4f}  {t1:5.1%}  {np.mean(max_probs):8.4f}  {np.mean(entropies):8.3f}{marker}")

        # Print baseline for reference
        bl_ll = 0
        bl_t1 = 0
        for rd in race_data:
            eps = 1e-15
            p = rd["market_direct"].get(rd["actual"], eps)
            bl_ll += -math.log(max(p, eps))
            if rd["market_direct"]:
                top = max(rd["market_direct"], key=rd["market_direct"].get)
                if top == rd["actual"]:
                    bl_t1 += 1

        print(f"\n  Market baseline: LogLoss={bl_ll/len(race_data):.4f}  Top1={bl_t1/len(race_data):.1%}")
        print(f"  Best temperature: T={best_t:.2f}  LogLoss={best_ll:.4f}")

        # === Market Blend Sweep ===
        print(f"\n{'='*60}")
        print(f"MARKET BLEND SWEEP (at pair level, using best T={best_t})")
        print(f"{'='*60}")
        print(f"  {'alpha':>6s}  {'LogLoss':>8s}  {'Top1':>6s}")

        alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        best_alpha = 1.0
        best_blend_ll = float("inf")

        for alpha in alphas:
            logloss_sum = 0
            top1_hits = 0

            for rd in race_data:
                scaled = temperature_scale(rd["model_probs"], best_t)
                model_pairs = plackett_luce_pairs(scaled, rd["car_nos"])
                blended = blend_pairs(model_pairs, rd["market_direct"], alpha)

                eps = 1e-15
                p = blended.get(rd["actual"], eps)
                logloss_sum += -math.log(max(p, eps))

                if blended:
                    top = max(blended, key=blended.get)
                    if top == rd["actual"]:
                        top1_hits += 1

            ll = logloss_sum / len(race_data)
            t1 = top1_hits / len(race_data)
            if ll < best_blend_ll:
                best_blend_ll = ll
                best_alpha = alpha

            print(f"  {alpha:6.2f}  {ll:8.4f}  {t1:5.1%}")

        print(f"\n  Best blend: alpha={best_alpha:.2f} LogLoss={best_blend_ll:.4f}")
        print(f"  vs Market only (alpha=0): LogLoss={bl_ll/len(race_data):.4f}")
        print(f"  vs Model only (alpha=1, T={best_t}): LogLoss={best_ll:.4f}")

        improvement = bl_ll/len(race_data) - best_blend_ll
        print(f"\n  Improvement over market: {improvement:+.4f} {'v' if improvement > 0 else 'x'}")


if __name__ == "__main__":
    main()
