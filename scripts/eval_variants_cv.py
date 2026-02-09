"""Evaluate multiple feature variants with time-series CV."""

from __future__ import annotations

import logging
import sys
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
from app.services.racer_stats import get_racer_stats


# --- Feature extractors for each variant ---

def extract_v11(entry, racer_st, race, n_runners):
    """v11 baseline: 8 features."""
    return extract_v11_features(entry)


def extract_c(entry, racer_st, race, n_runners):
    """Case C: v11 + hist_place_rate = 9 features."""
    base = extract_v11_features(entry)
    place_rate = racer_st.place_rate if racer_st else 0.0
    return np.append(base, [place_rate])


def extract_c_age(entry, racer_st, race, n_runners):
    """Case C + age: v11 + hist_place_rate + age = 10 features."""
    base = extract_v11_features(entry)
    place_rate = racer_st.place_rate if racer_st else 0.0
    stats = entry.stats_json or {}
    age = stats.get("age", 35) or 35
    return np.append(base, [place_rate, age])


def extract_c_age_count(entry, racer_st, race, n_runners):
    """Case C + age + race_count: v11 + hist_place_rate + age + hist_race_count = 11 features."""
    base = extract_v11_features(entry)
    place_rate = racer_st.place_rate if racer_st else 0.0
    race_count = min(racer_st.race_count / 20.0, 1.0) if racer_st else 0.0
    stats = entry.stats_json or {}
    age = stats.get("age", 35) or 35
    return np.append(base, [place_rate, age, race_count])


VARIANTS = {
    "v11 (8feat)": extract_v11,
    "C: v11+place_rate (9)": extract_c,
    "C+age (10)": extract_c_age,
    "C+age+count (11)": extract_c_age_count,
}


def build_data(db, date_from, date_to, extractor):
    """Build runner-level data with given extractor."""
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
        n_runners = len(entries)
        for entry in entries:
            rs = get_racer_stats(db, entry.racer_id, race_date)
            features = extractor(entry, rs, race, n_runners)
            label = 1 if entry.car_no == result.winner_car_no else 0
            rows.append({
                "features": features,
                "label": label,
                "race_id": race.race_id,
                "race_date": race_date,
            })
    return rows


def date_cv_splits(rows, n_folds=5):
    dates = sorted(set(r["race_date"] for r in rows))
    chunk_size = max(len(dates) // (n_folds + 1), 1)
    chunks = []
    for i in range(0, len(dates), chunk_size):
        chunks.append(dates[i : i + chunk_size])
    while len(chunks) > n_folds + 1:
        chunks[-2].extend(chunks[-1])
        chunks.pop()

    d2i: dict[date, list[int]] = {}
    for i, r in enumerate(rows):
        d2i.setdefault(r["race_date"], []).append(i)

    splits = []
    for fold in range(1, len(chunks)):
        td = set()
        for c in range(fold):
            td.update(chunks[c])
        vd = set(chunks[fold])
        ti = [i for d in sorted(td) for i in d2i.get(d, [])]
        vi = [i for d in sorted(vd) for i in d2i.get(d, [])]
        if ti and vi:
            splits.append((ti, vi))
    return splits


def eval_fold(rows, val_idx, clf):
    groups: dict[int, list[int]] = {}
    for i in val_idx:
        groups.setdefault(rows[i]["race_id"], []).append(i)

    ll_sum = 0.0
    hits = 0
    n = 0
    import warnings
    for rid, idxs in groups.items():
        feats = np.array([rows[i]["features"] for i in idxs])
        labs = [rows[i]["label"] for i in idxs]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names")
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
    if n == 0:
        return float("inf"), 0.0
    return ll_sum / n, hits / n


def main():
    d_from = date(2025, 11, 11)
    d_to = date(2026, 2, 9)

    with get_db() as db:
        print(f"\n{'='*60}")
        print(f"  Feature Variant Comparison (CV 5-fold, {d_from}~{d_to})")
        print(f"{'='*60}\n")

        results_summary = []
        for name, extractor in VARIANTS.items():
            logger.info("Building data for: %s", name)
            rows = build_data(db, d_from, d_to, extractor)
            n_races = len(set(r["race_id"] for r in rows))
            logger.info("  %d rows, %d races", len(rows), n_races)

            splits = date_cv_splits(rows, 5)
            fold_ll = []
            fold_t1 = []
            for fi, (ti, vi) in enumerate(splits):
                Xt = np.array([rows[i]["features"] for i in ti])
                yt = np.array([rows[i]["label"] for i in ti])
                clf = LGBMClassifier(
                    n_estimators=100, max_depth=4, num_leaves=15,
                    min_child_samples=50, reg_alpha=1.0, reg_lambda=1.0,
                    random_state=42, verbose=-1,
                )
                clf.fit(Xt, yt)
                ll, t1 = eval_fold(rows, vi, clf)
                fold_ll.append(ll)
                fold_t1.append(t1)

            mean_ll = np.mean(fold_ll)
            mean_t1 = np.mean(fold_t1)
            results_summary.append((name, mean_ll, mean_t1, n_races, len(rows)))

            # Feature importance
            Xa = np.array([r["features"] for r in rows])
            ya = np.array([r["label"] for r in rows])
            full = LGBMClassifier(
                n_estimators=100, max_depth=4, num_leaves=15,
                min_child_samples=50, reg_alpha=1.0, reg_lambda=1.0,
                random_state=42, verbose=-1,
            )
            full.fit(Xa, ya)
            imps = full.feature_importances_
            logger.info("  LogLoss=%.4f Top1=%.1f%% imps=%s", mean_ll, mean_t1 * 100, list(imps))

        # Summary table
        print(f"\n{'Variant':<30} {'LogLoss':>8} {'Top-1':>7} {'Races':>6} {'Samples':>8}")
        print("-" * 65)
        for name, ll, t1, nr, ns in results_summary:
            print(f"{name:<30} {ll:>8.4f} {t1:>6.1%} {nr:>6} {ns:>8}")

        # Best
        best = min(results_summary, key=lambda x: -x[2])
        print(f"\n>>> Best: {best[0]} (Top-1={best[2]:.1%}, LogLoss={best[1]:.4f})")


if __name__ == "__main__":
    main()
