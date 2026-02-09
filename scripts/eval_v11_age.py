"""Evaluate v11 + age only (9 features) with time-series CV."""

from __future__ import annotations

import logging
import sys
import warnings
from datetime import date

import numpy as np
from lightgbm import LGBMClassifier
from sqlalchemy import select

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(name)s — %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

from app.db.models import Race, RaceDay
from app.db.session import get_db
from app.services.modeling_v11 import extract_v11_features


def extract_v11_age(entry):
    """v11 + age = 9 features."""
    base = extract_v11_features(entry)
    stats = entry.stats_json or {}
    age = stats.get("age", 35) or 35
    return np.append(base, [age])


FEAT_NAMES = [
    "handicap_m", "trial_time", "start_avg", "deviation",
    "quinella_rate", "trio_rate", "rank_class", "car_no",
    "age",
]


def main():
    d_from = date(2025, 11, 11)
    d_to = date(2026, 2, 9)

    with get_db() as db:
        races = (
            db.execute(
                select(Race).join(RaceDay)
                .where(RaceDay.race_date >= d_from, RaceDay.race_date <= d_to)
                .order_by(RaceDay.race_date, Race.race_no)
            ).scalars().all()
        )

        rows = []
        for race in races:
            entries = sorted(race.entries, key=lambda e: e.car_no)
            if len(entries) < 2:
                continue
            result = race.result
            if result is None or not result.is_valid or result.winner_car_no is None:
                continue
            if result.winner_car_no not in [e.car_no for e in entries]:
                continue
            rd = race.race_day.race_date
            for entry in entries:
                rows.append({
                    "features": extract_v11_age(entry),
                    "label": 1 if entry.car_no == result.winner_car_no else 0,
                    "race_id": race.race_id,
                    "race_date": rd,
                })

        n_races = len(set(r["race_id"] for r in rows))
        logger.info("Built %d rows from %d races", len(rows), n_races)

        # CV splits
        dates = sorted(set(r["race_date"] for r in rows))
        n_folds = 5
        cs = max(len(dates) // (n_folds + 1), 1)
        chunks = [dates[i:i+cs] for i in range(0, len(dates), cs)]
        while len(chunks) > n_folds + 1:
            chunks[-2].extend(chunks[-1])
            chunks.pop()
        d2i: dict = {}
        for i, r in enumerate(rows):
            d2i.setdefault(r["race_date"], []).append(i)

        print(f"\n=== v11+age (9 features) CV Results ===")
        print(f"Races: {n_races}, Samples: {len(rows)}\n")

        fold_ll, fold_t1 = [], []
        for fold in range(1, len(chunks)):
            td = set()
            for c in range(fold):
                td.update(chunks[c])
            vd = set(chunks[fold])
            ti = [i for d in sorted(td) for i in d2i.get(d, [])]
            vi = [i for d in sorted(vd) for i in d2i.get(d, [])]
            if not ti or not vi:
                continue

            Xt = np.array([rows[i]["features"] for i in ti])
            yt = np.array([rows[i]["label"] for i in ti])
            clf = LGBMClassifier(
                n_estimators=100, max_depth=4, num_leaves=15,
                min_child_samples=50, reg_alpha=1.0, reg_lambda=1.0,
                random_state=42, verbose=-1,
            )
            clf.fit(Xt, yt)

            # Eval
            groups: dict[int, list[int]] = {}
            for i in vi:
                groups.setdefault(rows[i]["race_id"], []).append(i)
            ll_sum, hits, n = 0.0, 0, 0
            for rid, idxs in groups.items():
                feats = np.array([rows[i]["features"] for i in idxs])
                labs = [rows[i]["label"] for i in idxs]
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    probs = clf.predict_proba(feats)[:, 1]
                ps = probs.sum()
                if ps <= 0:
                    continue
                wi = next((j for j, l in enumerate(labs) if l == 1), None)
                if wi is None:
                    continue
                n += 1
                ll_sum += -np.log(max(probs[wi] / ps, 1e-15))
                if int(np.argmax(probs)) == wi:
                    hits += 1
            ll = ll_sum / n if n > 0 else float("inf")
            t1 = hits / n if n > 0 else 0
            fold_ll.append(ll)
            fold_t1.append(t1)
            vr = len(groups)
            print(f"  Fold {fold}: train={len(ti)} val={len(vi)} ({vr} races) logloss={ll:.4f} top1={t1:.1%}")

        mll = np.mean(fold_ll)
        mt1 = np.mean(fold_t1)
        print(f"\n  CV Mean LogLoss: {mll:.4f}")
        print(f"  CV Mean Top-1:   {mt1:.1%}")

        # Feature importance
        Xa = np.array([r["features"] for r in rows])
        ya = np.array([r["label"] for r in rows])
        full = LGBMClassifier(
            n_estimators=100, max_depth=4, num_leaves=15,
            min_child_samples=50, reg_alpha=1.0, reg_lambda=1.0,
            random_state=42, verbose=-1,
        )
        full.fit(Xa, ya)
        print(f"\n--- Feature Importances ---")
        for name, imp in sorted(zip(FEAT_NAMES, full.feature_importances_), key=lambda x: -x[1]):
            print(f"  {name:>15s}: {imp:4d} {'#' * min(imp, 50)}")

        # Comparison
        print(f"\n--- Comparison ---")
        print(f"  v11 (8feat):  LogLoss=1.6796  Top-1=39.0%")
        print(f"  v11+age (9):  LogLoss={mll:.4f}  Top-1={mt1:.1%}")
        diff = mt1 - 0.390
        print(f"  Diff:         {'+' if diff >= 0 else ''}{diff*100:.1f}pt")


if __name__ == "__main__":
    main()
