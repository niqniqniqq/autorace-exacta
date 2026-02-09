"""Evaluate v11 model with time-series CV for fair comparison with v12."""

from __future__ import annotations

import logging
import sys
from datetime import date

import numpy as np
from lightgbm import LGBMClassifier
from sqlalchemy import select
from sqlalchemy.orm import Session

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(name)s — %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

from app.db.models import Race, RaceDay, RaceResult
from app.db.session import get_db
from app.services.modeling_v11 import V11_FEATURE_NAMES, extract_v11_features


def build_v11_runner_data(
    db: Session, date_from: date, date_to: date
) -> list[dict]:
    """Build per-runner training data with v11 features."""
    races = (
        db.execute(
            select(Race)
            .join(RaceDay)
            .where(RaceDay.race_date >= date_from, RaceDay.race_date <= date_to)
            .order_by(RaceDay.race_date, Race.race_no)
        )
        .scalars()
        .all()
    )

    rows = []
    for race in races:
        entries = sorted(race.entries, key=lambda e: e.car_no)
        if len(entries) < 2:
            continue
        result = race.result
        if result is None or not result.is_valid or result.winner_car_no is None:
            continue
        car_nos = [e.car_no for e in entries]
        if result.winner_car_no not in car_nos:
            continue

        race_date = race.race_day.race_date
        for entry in entries:
            features = extract_v11_features(entry)
            label = 1 if entry.car_no == result.winner_car_no else 0
            rows.append({
                "features": features,
                "label": label,
                "race_id": race.race_id,
                "race_date": race_date,
            })

    return rows


def date_based_cv(rows, n_folds=5):
    """Same CV logic as v12 training."""
    dates = sorted(set(r["race_date"] for r in rows))
    chunk_size = max(len(dates) // (n_folds + 1), 1)
    date_chunks = []
    for i in range(0, len(dates), chunk_size):
        date_chunks.append(dates[i : i + chunk_size])
    while len(date_chunks) > n_folds + 1:
        date_chunks[-2].extend(date_chunks[-1])
        date_chunks.pop()

    date_to_idx: dict[date, list[int]] = {}
    for i, r in enumerate(rows):
        date_to_idx.setdefault(r["race_date"], []).append(i)

    splits = []
    for fold in range(1, len(date_chunks)):
        train_dates = set()
        for c in range(fold):
            train_dates.update(date_chunks[c])
        val_dates = set(date_chunks[fold])
        train_idx = [i for d in sorted(train_dates) for i in date_to_idx.get(d, [])]
        val_idx = [i for d in sorted(val_dates) for i in date_to_idx.get(d, [])]
        if train_idx and val_idx:
            splits.append((train_idx, val_idx))
    return splits


def evaluate_fold(rows, val_indices, model):
    """Evaluate on validation set."""
    race_groups: dict[int, list[int]] = {}
    for idx in val_indices:
        race_groups.setdefault(rows[idx]["race_id"], []).append(idx)

    logloss_sum = 0.0
    top1_hits = 0
    n_races = 0

    for race_id, indices in race_groups.items():
        features = np.array([rows[i]["features"] for i in indices])
        labels = [rows[i]["label"] for i in indices]

        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names")
            probs = model.predict_proba(features)[:, 1]

        prob_sum = probs.sum()
        if prob_sum <= 0:
            continue

        winner_idx = None
        for j, lab in enumerate(labels):
            if lab == 1:
                winner_idx = j
                break
        if winner_idx is None:
            continue

        n_races += 1
        eps = 1e-15
        winner_prob = max(probs[winner_idx] / prob_sum, eps)
        logloss_sum += -np.log(winner_prob)

        if int(np.argmax(probs)) == winner_idx:
            top1_hits += 1

    if n_races == 0:
        return float("inf"), 0.0
    return logloss_sum / n_races, top1_hits / n_races


def main():
    d_from = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else date(2025, 11, 11)
    d_to = date.fromisoformat(sys.argv[2]) if len(sys.argv) > 2 else date(2026, 2, 9)

    with get_db() as db:
        rows = build_v11_runner_data(db, d_from, d_to)
        logger.info("Built %d v11 rows from %d races",
                     len(rows), len(set(r["race_id"] for r in rows)))

        splits = date_based_cv(rows, n_folds=5)
        logger.info("Running %d-fold CV for v11", len(splits))

        fold_results = []
        for fold_i, (train_idx, val_idx) in enumerate(splits):
            X_train = np.array([rows[i]["features"] for i in train_idx])
            y_train = np.array([rows[i]["label"] for i in train_idx])

            clf = LGBMClassifier(
                n_estimators=100, max_depth=4, num_leaves=15,
                min_child_samples=50, reg_alpha=1.0, reg_lambda=1.0,
                random_state=42, verbose=-1,
            )
            clf.fit(X_train, y_train)

            logloss, top1 = evaluate_fold(rows, val_idx, clf)
            val_races = len(set(rows[i]["race_id"] for i in val_idx))
            fold_results.append((logloss, top1, val_races, len(train_idx), len(val_idx)))
            logger.info("Fold %d: train=%d val=%d(%d races) logloss=%.4f top1=%.1f%%",
                        fold_i + 1, len(train_idx), len(val_idx), val_races, logloss, top1 * 100)

        mean_ll = np.mean([r[0] for r in fold_results])
        mean_top1 = np.mean([r[1] for r in fold_results])

        print(f"\n=== v11 CV Results (same data as v12) ===")
        print(f"Races: {len(set(r['race_id'] for r in rows))}, Samples: {len(rows)}")
        for i, (ll, t1, vr, ts, vs) in enumerate(fold_results):
            print(f"  Fold {i+1}: train={ts} val={vs} ({vr} races) logloss={ll:.4f} top1={t1:.1%}")
        print(f"\n  CV Mean LogLoss: {mean_ll:.4f}")
        print(f"  CV Mean Top-1:   {mean_top1:.1%}")

        # Feature importances from full model
        X_all = np.array([r["features"] for r in rows])
        y_all = np.array([r["label"] for r in rows])
        full_clf = LGBMClassifier(
            n_estimators=100, max_depth=4, num_leaves=15,
            min_child_samples=50, reg_alpha=1.0, reg_lambda=1.0,
            random_state=42, verbose=-1,
        )
        full_clf.fit(X_all, y_all)
        imps = sorted(zip(V11_FEATURE_NAMES, full_clf.feature_importances_),
                       key=lambda x: x[1], reverse=True)
        print(f"\n--- Feature Importances ---")
        for name, imp in imps:
            print(f"  {name:>15s}: {imp:4d} {'#' * min(imp, 50)}")


if __name__ == "__main__":
    main()
