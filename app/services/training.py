"""Training pipeline for v12/v13/v14/v16/v17/v18 models with time-series cross-validation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import OddsExacta, Race, RaceDay, RaceEntry, RaceResult
from app.services.modeling_v12 import V12_FEATURE_NAMES, ExactaModelV12, extract_v12_features

logger = logging.getLogger(__name__)


@dataclass
class TrainingRow:
    """Single training sample: one runner in one race."""

    features: np.ndarray
    label: int  # 1 = winner, 0 = other
    race_id: int
    race_date: date


@dataclass
class FoldResult:
    """Metrics for one CV fold."""

    fold: int
    train_size: int
    val_size: int
    val_races: int
    logloss: float
    top1_accuracy: float


@dataclass
class TrainingReport:
    """Summary of training run."""

    fold_results: list[FoldResult] = field(default_factory=list)
    feature_importances: list[tuple[str, int]] = field(default_factory=list)
    total_samples: int = 0
    total_races: int = 0

    @property
    def cv_mean_logloss(self) -> float:
        if not self.fold_results:
            return float("inf")
        return float(np.mean([f.logloss for f in self.fold_results]))

    @property
    def cv_mean_top1(self) -> float:
        if not self.fold_results:
            return 0.0
        return float(np.mean([f.top1_accuracy for f in self.fold_results]))


def build_v12_training_data(
    db: Session,
    date_from: date,
    date_to: date,
) -> list[TrainingRow]:
    """Build per-runner training data with v12 features.

    Returns list of TrainingRow (one per runner per race).
    """
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

    rows: list[TrainingRow] = []
    skipped = 0

    for race in races:
        entries = sorted(race.entries, key=lambda e: e.car_no)
        if len(entries) < 2:
            continue

        result = race.result
        if result is None or not result.is_valid or result.winner_car_no is None:
            continue

        car_nos = [e.car_no for e in entries]
        if result.winner_car_no not in car_nos:
            skipped += 1
            continue

        race_date = race.race_day.race_date

        for entry in entries:
            features = extract_v12_features(entry)
            label = 1 if entry.car_no == result.winner_car_no else 0
            rows.append(TrainingRow(
                features=features,
                label=label,
                race_id=race.race_id,
                race_date=race_date,
            ))

    logger.info(
        "Built %d training rows from %d races (skipped %d)",
        len(rows), len(set(r.race_id for r in rows)), skipped,
    )
    return rows


def _date_based_cv_splits(
    rows: list[TrainingRow],
    n_folds: int = 5,
) -> list[tuple[list[int], list[int]]]:
    """Create time-series CV splits based on race dates.

    Each fold: train on earlier dates, validate on later dates.
    """
    dates = sorted(set(r.race_date for r in rows))
    if len(dates) < n_folds + 1:
        n_folds = max(len(dates) - 1, 1)

    chunk_size = max(len(dates) // (n_folds + 1), 1)
    date_chunks: list[list[date]] = []
    for i in range(0, len(dates), chunk_size):
        date_chunks.append(dates[i : i + chunk_size])

    while len(date_chunks) > n_folds + 1:
        date_chunks[-2].extend(date_chunks[-1])
        date_chunks.pop()

    date_to_indices: dict[date, list[int]] = {}
    for i, row in enumerate(rows):
        date_to_indices.setdefault(row.race_date, []).append(i)

    splits: list[tuple[list[int], list[int]]] = []
    for fold in range(1, len(date_chunks)):
        train_dates = set()
        for c in range(fold):
            train_dates.update(date_chunks[c])
        val_dates = set(date_chunks[fold])

        train_idx = []
        for d in sorted(train_dates):
            train_idx.extend(date_to_indices.get(d, []))
        val_idx = []
        for d in sorted(val_dates):
            val_idx.extend(date_to_indices.get(d, []))

        if train_idx and val_idx:
            splits.append((train_idx, val_idx))

    return splits


def _evaluate_fold(
    rows: list[TrainingRow],
    val_indices: list[int],
    model,
) -> tuple[float, float]:
    """Evaluate a trained model on validation data.

    Returns (logloss, top1_accuracy).
    """
    race_groups: dict[int, list[int]] = {}
    for idx in val_indices:
        race_groups.setdefault(rows[idx].race_id, []).append(idx)

    logloss_sum = 0.0
    top1_hits = 0
    n_races = 0

    for race_id, indices in race_groups.items():
        features = np.array([rows[i].features for i in indices])
        labels = [rows[i].label for i in indices]

        probs = model.predict_proba(features)
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

        pred_winner = int(np.argmax(probs))
        if pred_winner == winner_idx:
            top1_hits += 1

    if n_races == 0:
        return float("inf"), 0.0

    return logloss_sum / n_races, top1_hits / n_races


def train_v12_model(
    db: Session,
    date_from: date,
    date_to: date,
    n_folds: int = 5,
) -> tuple[ExactaModelV12, TrainingReport]:
    """Train v12 model with time-series CV.

    Returns (fitted model, training report).
    """
    from lightgbm import LGBMClassifier

    rows = build_v12_training_data(db, date_from, date_to)
    if not rows:
        logger.warning("No training data found")
        return ExactaModelV12(), TrainingReport()

    report = TrainingReport(
        total_samples=len(rows),
        total_races=len(set(r.race_id for r in rows)),
    )

    # --- Cross-validation ---
    splits = _date_based_cv_splits(rows, n_folds)
    logger.info("Running %d-fold time-series CV", len(splits))

    for fold_i, (train_idx, val_idx) in enumerate(splits):
        X_train = np.array([rows[i].features for i in train_idx])
        y_train = np.array([rows[i].label for i in train_idx])

        clf = LGBMClassifier(
            n_estimators=100,
            max_depth=4,
            num_leaves=15,
            min_child_samples=50,
            reg_alpha=1.0,
            reg_lambda=1.0,
            random_state=42,
            verbose=-1,
        )
        clf.fit(X_train, y_train)

        fold_model = ExactaModelV12(model=clf)
        logloss, top1 = _evaluate_fold(rows, val_idx, fold_model)

        val_races = len(set(rows[i].race_id for i in val_idx))
        fold_result = FoldResult(
            fold=fold_i + 1,
            train_size=len(train_idx),
            val_size=len(val_idx),
            val_races=val_races,
            logloss=logloss,
            top1_accuracy=top1,
        )
        report.fold_results.append(fold_result)
        logger.info(
            "Fold %d: train=%d val=%d(%d races) logloss=%.4f top1=%.1f%%",
            fold_i + 1, len(train_idx), len(val_idx), val_races,
            logloss, top1 * 100,
        )

    logger.info(
        "CV mean: logloss=%.4f top1=%.1f%%",
        report.cv_mean_logloss, report.cv_mean_top1 * 100,
    )

    # --- Train final model on all data ---
    X_all = np.array([r.features for r in rows])
    y_all = np.array([r.label for r in rows])

    final_clf = LGBMClassifier(
        n_estimators=100,
        max_depth=4,
        num_leaves=15,
        min_child_samples=50,
        reg_alpha=1.0,
        reg_lambda=1.0,
        random_state=42,
        verbose=-1,
    )
    final_clf.fit(X_all, y_all)

    final_model = ExactaModelV12(model=final_clf)
    report.feature_importances = final_model.get_feature_importance()

    logger.info("Feature importances:")
    for name, imp in report.feature_importances:
        logger.info("  %s: %d", name, imp)

    return final_model, report


# -------------------------------------------------------------------
# v13 training
# -------------------------------------------------------------------


def _get_latest_odds_for_race(
    db: Session,
    race_id: int,
) -> dict[tuple[int, int], float]:
    """Get the latest odds for a race as {(first, second): odds}."""
    latest_captured = db.scalar(
        select(func.max(OddsExacta.captured_at)).where(OddsExacta.race_id == race_id)
    )
    if latest_captured is None:
        return {}

    rows = db.scalars(
        select(OddsExacta).where(
            OddsExacta.race_id == race_id,
            OddsExacta.captured_at == latest_captured,
        )
    ).all()

    return {(r.first_car_no, r.second_car_no): r.odds for r in rows}


def build_v13_training_data(
    db: Session,
    date_from: date,
    date_to: date,
) -> list[TrainingRow]:
    """Build per-runner training data with v13 features (v12 + odds).

    Races without odds are skipped.
    """
    from app.services.modeling_v13 import compute_runner_odds_stats, extract_v13_features

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

    rows: list[TrainingRow] = []
    skipped = 0
    no_odds = 0

    for race in races:
        entries = sorted(race.entries, key=lambda e: e.car_no)
        if len(entries) < 2:
            continue

        result = race.result
        if result is None or not result.is_valid or result.winner_car_no is None:
            continue

        car_nos = [e.car_no for e in entries]
        if result.winner_car_no not in car_nos:
            skipped += 1
            continue

        # Get odds — skip race if no odds
        odds_dict = _get_latest_odds_for_race(db, race.race_id)
        if not odds_dict:
            no_odds += 1
            continue

        odds_stats = compute_runner_odds_stats(odds_dict, car_nos)
        race_date = race.race_day.race_date

        for entry in entries:
            features = extract_v13_features(entry, odds_stats)
            label = 1 if entry.car_no == result.winner_car_no else 0
            rows.append(TrainingRow(
                features=features,
                label=label,
                race_id=race.race_id,
                race_date=race_date,
            ))

    logger.info(
        "Built %d v13 training rows from %d races (skipped %d, no_odds %d)",
        len(rows), len(set(r.race_id for r in rows)), skipped, no_odds,
    )
    return rows


def train_v13_model(
    db: Session,
    date_from: date,
    date_to: date,
    n_folds: int = 5,
    calibrate: bool = True,
) -> tuple:
    """Train v13 model with time-series CV + optional Platt calibration.

    Returns (ExactaModelV13, TrainingReport).
    """
    from lightgbm import LGBMClassifier

    from app.services.modeling_v13 import ExactaModelV13

    rows = build_v13_training_data(db, date_from, date_to)
    if not rows:
        logger.warning("No v13 training data found")
        return ExactaModelV13(), TrainingReport()

    report = TrainingReport(
        total_samples=len(rows),
        total_races=len(set(r.race_id for r in rows)),
    )

    # --- Cross-validation ---
    splits = _date_based_cv_splits(rows, n_folds)
    logger.info("Running %d-fold time-series CV (v13)", len(splits))

    last_val_idx: list[int] = []

    for fold_i, (train_idx, val_idx) in enumerate(splits):
        X_train = np.array([rows[i].features for i in train_idx])
        y_train = np.array([rows[i].label for i in train_idx])

        clf = LGBMClassifier(
            n_estimators=100,
            max_depth=4,
            num_leaves=15,
            min_child_samples=50,
            reg_alpha=1.0,
            reg_lambda=1.0,
            random_state=42,
            verbose=-1,
        )
        clf.fit(X_train, y_train)

        fold_model = ExactaModelV13(model=clf)
        logloss, top1 = _evaluate_fold(rows, val_idx, fold_model)

        val_races = len(set(rows[i].race_id for i in val_idx))
        fold_result = FoldResult(
            fold=fold_i + 1,
            train_size=len(train_idx),
            val_size=len(val_idx),
            val_races=val_races,
            logloss=logloss,
            top1_accuracy=top1,
        )
        report.fold_results.append(fold_result)
        last_val_idx = val_idx
        logger.info(
            "Fold %d: train=%d val=%d(%d races) logloss=%.4f top1=%.1f%%",
            fold_i + 1, len(train_idx), len(val_idx), val_races,
            logloss, top1 * 100,
        )

    logger.info(
        "CV mean: logloss=%.4f top1=%.1f%%",
        report.cv_mean_logloss, report.cv_mean_top1 * 100,
    )

    # --- Train final model on all data ---
    X_all = np.array([r.features for r in rows])
    y_all = np.array([r.label for r in rows])

    final_clf = LGBMClassifier(
        n_estimators=100,
        max_depth=4,
        num_leaves=15,
        min_child_samples=50,
        reg_alpha=1.0,
        reg_lambda=1.0,
        random_state=42,
        verbose=-1,
    )
    final_clf.fit(X_all, y_all)

    final_model = ExactaModelV13(model=final_clf)
    report.feature_importances = final_model.get_feature_importance()

    # --- Platt calibration on last fold's validation data ---
    if calibrate and last_val_idx:
        race_groups: dict[int, list[int]] = {}
        for idx in last_val_idx:
            race_groups.setdefault(rows[idx].race_id, []).append(idx)

        val_feats_list: list[np.ndarray] = []
        val_labels_list: list[np.ndarray] = []
        for race_id, indices in race_groups.items():
            feats = np.array([rows[i].features for i in indices])
            labels = np.array([rows[i].label for i in indices])
            val_feats_list.append(feats)
            val_labels_list.append(labels)

        if val_feats_list:
            final_model.fit_calibrator(val_feats_list, val_labels_list)

    # --- Fit market blend alpha on last fold's validation data ---
    if calibrate and last_val_idx:
        from app.services.modeling_v13 import compute_runner_odds_stats, extract_v13_features

        val_race_ids = sorted(set(rows[i].race_id for i in last_val_idx))
        val_blend_data: list[tuple] = []

        for race_id in val_race_ids:
            race = db.get(Race, race_id)
            if not race or not race.result or not race.result.is_valid:
                continue
            result = race.result
            if result.winner_car_no is None or result.second_car_no is None:
                continue

            entries = sorted(race.entries, key=lambda e: e.car_no)
            car_nos = [e.car_no for e in entries]
            actual_pair = (result.winner_car_no, result.second_car_no)

            odds_dict = _get_latest_odds_for_race(db, race_id)
            if not odds_dict:
                continue

            odds_stats = compute_runner_odds_stats(odds_dict, car_nos)
            feats = np.array([extract_v13_features(e, odds_stats) for e in entries])

            total_inv = sum(1.0 / o for o in odds_dict.values() if o > 0)
            if total_inv <= 0:
                continue
            market_pair_probs = {
                pair: (1.0 / o) / total_inv
                for pair, o in odds_dict.items() if o > 0
            }

            val_blend_data.append((feats, car_nos, market_pair_probs, actual_pair))

        if val_blend_data:
            final_model.fit_market_alpha(val_blend_data)

    logger.info("Feature importances (v13):")
    for name, imp in report.feature_importances:
        logger.info("  %s: %d", name, imp)

    return final_model, report


# -------------------------------------------------------------------
# v14 training
# -------------------------------------------------------------------


def build_v14_training_data(
    db: Session,
    date_from: date,
    date_to: date,
) -> list[TrainingRow]:
    """Build per-runner training data with v14 features (v13 + racer history + home track).

    Races without odds are skipped.
    """
    from app.services.modeling_v13 import compute_runner_odds_stats
    from app.services.modeling_v14 import extract_v14_features
    from app.services.racer_stats import get_racer_stats

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

    rows: list[TrainingRow] = []
    skipped = 0
    no_odds = 0

    for race in races:
        entries = sorted(race.entries, key=lambda e: e.car_no)
        if len(entries) < 2:
            continue

        result = race.result
        if result is None or not result.is_valid or result.winner_car_no is None:
            continue

        car_nos = [e.car_no for e in entries]
        if result.winner_car_no not in car_nos:
            skipped += 1
            continue

        # Get odds — skip race if no odds
        odds_dict = _get_latest_odds_for_race(db, race.race_id)
        if not odds_dict:
            no_odds += 1
            continue

        odds_stats = compute_runner_odds_stats(odds_dict, car_nos)
        race_date = race.race_day.race_date

        for entry in entries:
            # Racer history (before_date = race_date for leak prevention)
            stats = get_racer_stats(db, entry.racer_id, before_date=race_date)
            features = extract_v14_features(entry, odds_stats, racer_stats=stats)
            label = 1 if entry.car_no == result.winner_car_no else 0
            rows.append(TrainingRow(
                features=features,
                label=label,
                race_id=race.race_id,
                race_date=race_date,
            ))

    logger.info(
        "Built %d v14 training rows from %d races (skipped %d, no_odds %d)",
        len(rows), len(set(r.race_id for r in rows)), skipped, no_odds,
    )
    return rows


def train_v14_model(
    db: Session,
    date_from: date,
    date_to: date,
    n_folds: int = 5,
    calibrate: bool = True,
) -> tuple:
    """Train v14 model with time-series CV + Platt calibration + market blend.

    Returns (ExactaModelV14, TrainingReport).
    """
    from lightgbm import LGBMClassifier

    from app.services.modeling_v14 import ExactaModelV14

    rows = build_v14_training_data(db, date_from, date_to)
    if not rows:
        logger.warning("No v14 training data found")
        return ExactaModelV14(), TrainingReport()

    report = TrainingReport(
        total_samples=len(rows),
        total_races=len(set(r.race_id for r in rows)),
    )

    # --- Cross-validation ---
    splits = _date_based_cv_splits(rows, n_folds)
    logger.info("Running %d-fold time-series CV (v14)", len(splits))

    last_val_idx: list[int] = []

    lgb_params = dict(
        n_estimators=100,
        max_depth=4,
        num_leaves=15,
        min_child_samples=50,
        reg_alpha=1.0,
        reg_lambda=1.0,
        random_state=42,
        verbose=-1,
    )

    for fold_i, (train_idx, val_idx) in enumerate(splits):
        X_train = np.array([rows[i].features for i in train_idx])
        y_train = np.array([rows[i].label for i in train_idx])

        clf = LGBMClassifier(**lgb_params)
        clf.fit(X_train, y_train)

        fold_model = ExactaModelV14(model=clf)
        logloss, top1 = _evaluate_fold(rows, val_idx, fold_model)

        val_races = len(set(rows[i].race_id for i in val_idx))
        fold_result = FoldResult(
            fold=fold_i + 1,
            train_size=len(train_idx),
            val_size=len(val_idx),
            val_races=val_races,
            logloss=logloss,
            top1_accuracy=top1,
        )
        report.fold_results.append(fold_result)
        last_val_idx = val_idx
        logger.info(
            "Fold %d: train=%d val=%d(%d races) logloss=%.4f top1=%.1f%%",
            fold_i + 1, len(train_idx), len(val_idx), val_races,
            logloss, top1 * 100,
        )

    logger.info(
        "CV mean: logloss=%.4f top1=%.1f%%",
        report.cv_mean_logloss, report.cv_mean_top1 * 100,
    )

    # --- Train final model on all data ---
    X_all = np.array([r.features for r in rows])
    y_all = np.array([r.label for r in rows])

    final_clf = LGBMClassifier(**lgb_params)
    final_clf.fit(X_all, y_all)

    final_model = ExactaModelV14(model=final_clf)
    report.feature_importances = final_model.get_feature_importance()

    # --- Platt calibration on last fold's validation data ---
    if calibrate and last_val_idx:
        race_groups: dict[int, list[int]] = {}
        for idx in last_val_idx:
            race_groups.setdefault(rows[idx].race_id, []).append(idx)

        val_feats_list: list[np.ndarray] = []
        val_labels_list: list[np.ndarray] = []
        for race_id, indices in race_groups.items():
            feats = np.array([rows[i].features for i in indices])
            labels = np.array([rows[i].label for i in indices])
            val_feats_list.append(feats)
            val_labels_list.append(labels)

        if val_feats_list:
            final_model.fit_calibrator(val_feats_list, val_labels_list)

    # --- Fit market blend alpha on last fold's validation data ---
    if calibrate and last_val_idx:
        from app.services.modeling_v13 import compute_runner_odds_stats
        from app.services.modeling_v14 import extract_v14_features
        from app.services.racer_stats import get_racer_stats

        val_race_ids = sorted(set(rows[i].race_id for i in last_val_idx))
        val_blend_data: list[tuple] = []

        for race_id in val_race_ids:
            race = db.get(Race, race_id)
            if not race or not race.result or not race.result.is_valid:
                continue
            result = race.result
            if result.winner_car_no is None or result.second_car_no is None:
                continue

            entries = sorted(race.entries, key=lambda e: e.car_no)
            car_nos = [e.car_no for e in entries]
            actual_pair = (result.winner_car_no, result.second_car_no)

            odds_dict = _get_latest_odds_for_race(db, race_id)
            if not odds_dict:
                continue

            odds_stats = compute_runner_odds_stats(odds_dict, car_nos)
            race_date = race.race_day.race_date

            feats = np.array([
                extract_v14_features(
                    e, odds_stats,
                    racer_stats=get_racer_stats(db, e.racer_id, before_date=race_date),
                )
                for e in entries
            ])

            total_inv = sum(1.0 / o for o in odds_dict.values() if o > 0)
            if total_inv <= 0:
                continue
            market_pair_probs = {
                pair: (1.0 / o) / total_inv
                for pair, o in odds_dict.items() if o > 0
            }

            val_blend_data.append((feats, car_nos, market_pair_probs, actual_pair))

        if val_blend_data:
            final_model.fit_market_alpha(val_blend_data)

    logger.info("Feature importances (v14):")
    for name, imp in report.feature_importances:
        logger.info("  %s: %d", name, imp)

    return final_model, report


# -------------------------------------------------------------------
# v15 training (pairwise)
# -------------------------------------------------------------------


@dataclass
class PairTrainingRow:
    """Single training sample: one (i->j) pair in one race."""

    features: np.ndarray  # (37,)
    label: int  # 1 = actual exacta pair, 0 = other
    race_id: int
    race_date: date


def build_v15_training_data(
    db: Session,
    date_from: date,
    date_to: date,
) -> list[PairTrainingRow]:
    """Build pair-level training data with v15 features.

    Each race produces N*(N-1) pairs.  label=1 for the actual exacta result.
    Races without odds or without a valid second_car_no are skipped.
    """
    from app.services.modeling_v13 import compute_runner_odds_stats
    from app.services.modeling_v14 import extract_v14_features
    from app.services.modeling_v15 import build_pair_features
    from app.services.racer_stats import get_racer_stats

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

    rows: list[PairTrainingRow] = []
    skipped = 0
    no_odds = 0

    for race in races:
        entries = sorted(race.entries, key=lambda e: e.car_no)
        if len(entries) < 2:
            continue

        result = race.result
        if result is None or not result.is_valid:
            continue
        if result.winner_car_no is None or result.second_car_no is None:
            continue

        car_nos = [e.car_no for e in entries]
        if result.winner_car_no not in car_nos or result.second_car_no not in car_nos:
            skipped += 1
            continue

        # Get odds — skip race if no odds
        odds_dict = _get_latest_odds_for_race(db, race.race_id)
        if not odds_dict:
            no_odds += 1
            continue

        odds_stats = compute_runner_odds_stats(odds_dict, car_nos)
        race_date = race.race_day.race_date
        actual_pair = (result.winner_car_no, result.second_car_no)

        # Build runner features (n, 15)
        runner_feats = np.array([
            extract_v14_features(
                e, odds_stats,
                racer_stats=get_racer_stats(db, e.racer_id, before_date=race_date),
            )
            for e in entries
        ])

        # Build pair features (n*(n-1), 37)
        pair_feats, pairs = build_pair_features(runner_feats, car_nos)

        for k, pair in enumerate(pairs):
            label = 1 if pair == actual_pair else 0
            rows.append(PairTrainingRow(
                features=pair_feats[k],
                label=label,
                race_id=race.race_id,
                race_date=race_date,
            ))

    logger.info(
        "Built %d v15 pair-training rows from %d races (skipped %d, no_odds %d)",
        len(rows), len(set(r.race_id for r in rows)), skipped, no_odds,
    )
    return rows


def _evaluate_fold_v15(
    rows: list[PairTrainingRow],
    val_indices: list[int],
    model,
) -> tuple[float, float]:
    """Evaluate v15 pairwise model on validation data.

    Groups pairs by race, applies softmax, computes pair LogLoss and top-1 accuracy.
    Returns (logloss, top1_accuracy).
    """
    from app.services.modeling_v15 import _softmax

    race_groups: dict[int, list[int]] = {}
    for idx in val_indices:
        race_groups.setdefault(rows[idx].race_id, []).append(idx)

    logloss_sum = 0.0
    top1_hits = 0
    n_races = 0

    for race_id, indices in race_groups.items():
        features = np.array([rows[i].features for i in indices])
        labels = np.array([rows[i].label for i in indices])

        # Check that we have exactly one positive label
        if labels.sum() != 1:
            continue

        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names")
            raw_scores = model.predict_proba(features)[:, 1]

        probs = _softmax(raw_scores)

        actual_idx = int(np.argmax(labels))
        eps = 1e-15
        actual_prob = max(probs[actual_idx], eps)
        logloss_sum += -np.log(actual_prob)

        pred_idx = int(np.argmax(probs))
        if pred_idx == actual_idx:
            top1_hits += 1
        n_races += 1

    if n_races == 0:
        return float("inf"), 0.0

    return logloss_sum / n_races, top1_hits / n_races


def train_v15_model(
    db: Session,
    date_from: date,
    date_to: date,
    n_folds: int = 5,
) -> tuple:
    """Train v15 pairwise model with time-series CV + market blend.

    Returns (ExactaModelV15, TrainingReport).
    """
    from lightgbm import LGBMClassifier

    from app.services.modeling_v15 import ExactaModelV15

    rows = build_v15_training_data(db, date_from, date_to)
    if not rows:
        logger.warning("No v15 training data found")
        return ExactaModelV15(), TrainingReport()

    report = TrainingReport(
        total_samples=len(rows),
        total_races=len(set(r.race_id for r in rows)),
    )

    # --- Cross-validation ---
    # Reuse _date_based_cv_splits (works on any object with .race_date)
    splits = _date_based_cv_splits(rows, n_folds)
    logger.info("Running %d-fold time-series CV (v15 pairwise)", len(splits))

    last_val_idx: list[int] = []

    lgb_params = dict(
        n_estimators=200,
        max_depth=5,
        num_leaves=31,
        min_child_samples=100,
        reg_alpha=1.0,
        reg_lambda=1.0,
        scale_pos_weight=55,
        random_state=42,
        verbose=-1,
    )

    for fold_i, (train_idx, val_idx) in enumerate(splits):
        X_train = np.array([rows[i].features for i in train_idx])
        y_train = np.array([rows[i].label for i in train_idx])

        clf = LGBMClassifier(**lgb_params)
        clf.fit(X_train, y_train)

        logloss, top1 = _evaluate_fold_v15(rows, val_idx, clf)

        val_races = len(set(rows[i].race_id for i in val_idx))
        fold_result = FoldResult(
            fold=fold_i + 1,
            train_size=len(train_idx),
            val_size=len(val_idx),
            val_races=val_races,
            logloss=logloss,
            top1_accuracy=top1,
        )
        report.fold_results.append(fold_result)
        last_val_idx = val_idx
        logger.info(
            "Fold %d: train=%d val=%d(%d races) logloss=%.4f top1=%.1f%%",
            fold_i + 1, len(train_idx), len(val_idx), val_races,
            logloss, top1 * 100,
        )

    logger.info(
        "CV mean: logloss=%.4f top1=%.1f%%",
        report.cv_mean_logloss, report.cv_mean_top1 * 100,
    )

    # --- Train final model on all data ---
    X_all = np.array([r.features for r in rows])
    y_all = np.array([r.label for r in rows])

    final_clf = LGBMClassifier(**lgb_params)
    final_clf.fit(X_all, y_all)

    final_model = ExactaModelV15(model=final_clf)
    report.feature_importances = final_model.get_feature_importance()

    # --- Fit market blend alpha on last fold's validation data ---
    if last_val_idx:
        from app.services.modeling_v13 import compute_runner_odds_stats
        from app.services.modeling_v14 import extract_v14_features
        from app.services.racer_stats import get_racer_stats

        val_race_ids = sorted(set(rows[i].race_id for i in last_val_idx))
        val_blend_data: list[tuple] = []

        for race_id in val_race_ids:
            race = db.get(Race, race_id)
            if not race or not race.result or not race.result.is_valid:
                continue
            result = race.result
            if result.winner_car_no is None or result.second_car_no is None:
                continue

            entries = sorted(race.entries, key=lambda e: e.car_no)
            car_nos = [e.car_no for e in entries]
            actual_pair = (result.winner_car_no, result.second_car_no)

            odds_dict = _get_latest_odds_for_race(db, race_id)
            if not odds_dict:
                continue

            odds_stats = compute_runner_odds_stats(odds_dict, car_nos)
            race_date = race.race_day.race_date

            feats = np.array([
                extract_v14_features(
                    e, odds_stats,
                    racer_stats=get_racer_stats(db, e.racer_id, before_date=race_date),
                )
                for e in entries
            ])

            total_inv = sum(1.0 / o for o in odds_dict.values() if o > 0)
            if total_inv <= 0:
                continue
            market_pair_probs = {
                pair: (1.0 / o) / total_inv
                for pair, o in odds_dict.items() if o > 0
            }

            val_blend_data.append((feats, car_nos, market_pair_probs, actual_pair))

        if val_blend_data:
            final_model.fit_market_alpha(val_blend_data)

    logger.info("Feature importances (v15):")
    for name, imp in report.feature_importances[:20]:
        logger.info("  %s: %d", name, imp)

    return final_model, report


# -------------------------------------------------------------------
# v16 training
# -------------------------------------------------------------------


def build_v16_training_data(
    db: Session,
    date_from: date,
    date_to: date,
) -> list[TrainingRow]:
    """Build per-runner training data with v16 features (v14 + API stats + race context).

    Races without odds are skipped.
    """
    from app.services.modeling_v13 import compute_runner_odds_stats
    from app.services.modeling_v16 import extract_v16_features
    from app.services.racer_stats import get_racer_stats

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

    rows: list[TrainingRow] = []
    skipped = 0
    no_odds = 0

    for race in races:
        entries = sorted(race.entries, key=lambda e: e.car_no)
        if len(entries) < 2:
            continue

        result = race.result
        if result is None or not result.is_valid or result.winner_car_no is None:
            continue

        car_nos = [e.car_no for e in entries]
        if result.winner_car_no not in car_nos:
            skipped += 1
            continue

        # Get odds — skip race if no odds
        odds_dict = _get_latest_odds_for_race(db, race.race_id)
        if not odds_dict:
            no_odds += 1
            continue

        odds_stats = compute_runner_odds_stats(odds_dict, car_nos)
        race_date = race.race_day.race_date

        for entry in entries:
            stats = get_racer_stats(db, entry.racer_id, before_date=race_date)
            features = extract_v16_features(
                entry, odds_stats, racer_stats=stats,
            )
            label = 1 if entry.car_no == result.winner_car_no else 0
            rows.append(TrainingRow(
                features=features,
                label=label,
                race_id=race.race_id,
                race_date=race_date,
            ))

    logger.info(
        "Built %d v16 training rows from %d races (skipped %d, no_odds %d)",
        len(rows), len(set(r.race_id for r in rows)), skipped, no_odds,
    )
    return rows


def train_v16_model(
    db: Session,
    date_from: date,
    date_to: date,
    n_folds: int = 5,
    calibrate: bool = True,
) -> tuple:
    """Train v16 model with time-series CV + Platt calibration + market blend.

    Returns (ExactaModelV16, TrainingReport).
    """
    from lightgbm import LGBMClassifier

    from app.services.modeling_v16 import ExactaModelV16

    rows = build_v16_training_data(db, date_from, date_to)
    if not rows:
        logger.warning("No v16 training data found")
        return ExactaModelV16(), TrainingReport()

    report = TrainingReport(
        total_samples=len(rows),
        total_races=len(set(r.race_id for r in rows)),
    )

    # --- Cross-validation ---
    splits = _date_based_cv_splits(rows, n_folds)
    logger.info("Running %d-fold time-series CV (v16)", len(splits))

    last_val_idx: list[int] = []

    lgb_params = dict(
        n_estimators=100,
        max_depth=4,
        num_leaves=15,
        min_child_samples=50,
        reg_alpha=1.0,
        reg_lambda=1.0,
        random_state=42,
        verbose=-1,
    )

    for fold_i, (train_idx, val_idx) in enumerate(splits):
        X_train = np.array([rows[i].features for i in train_idx])
        y_train = np.array([rows[i].label for i in train_idx])

        clf = LGBMClassifier(**lgb_params)
        clf.fit(X_train, y_train)

        fold_model = ExactaModelV16(model=clf)
        logloss, top1 = _evaluate_fold(rows, val_idx, fold_model)

        val_races = len(set(rows[i].race_id for i in val_idx))
        fold_result = FoldResult(
            fold=fold_i + 1,
            train_size=len(train_idx),
            val_size=len(val_idx),
            val_races=val_races,
            logloss=logloss,
            top1_accuracy=top1,
        )
        report.fold_results.append(fold_result)
        last_val_idx = val_idx
        logger.info(
            "Fold %d: train=%d val=%d(%d races) logloss=%.4f top1=%.1f%%",
            fold_i + 1, len(train_idx), len(val_idx), val_races,
            logloss, top1 * 100,
        )

    logger.info(
        "CV mean: logloss=%.4f top1=%.1f%%",
        report.cv_mean_logloss, report.cv_mean_top1 * 100,
    )

    # --- Train final model on all data ---
    X_all = np.array([r.features for r in rows])
    y_all = np.array([r.label for r in rows])

    final_clf = LGBMClassifier(**lgb_params)
    final_clf.fit(X_all, y_all)

    final_model = ExactaModelV16(model=final_clf)
    report.feature_importances = final_model.get_feature_importance()

    # --- Platt calibration on last fold's validation data ---
    if calibrate and last_val_idx:
        race_groups: dict[int, list[int]] = {}
        for idx in last_val_idx:
            race_groups.setdefault(rows[idx].race_id, []).append(idx)

        val_feats_list: list[np.ndarray] = []
        val_labels_list: list[np.ndarray] = []
        for race_id, indices in race_groups.items():
            feats = np.array([rows[i].features for i in indices])
            labels = np.array([rows[i].label for i in indices])
            val_feats_list.append(feats)
            val_labels_list.append(labels)

        if val_feats_list:
            final_model.fit_calibrator(val_feats_list, val_labels_list)

    # --- Fit market blend alpha on last fold's validation data ---
    if calibrate and last_val_idx:
        from app.services.modeling_v13 import compute_runner_odds_stats
        from app.services.modeling_v16 import extract_v16_features
        from app.services.racer_stats import get_racer_stats

        val_race_ids = sorted(set(rows[i].race_id for i in last_val_idx))
        val_blend_data: list[tuple] = []

        for race_id in val_race_ids:
            race = db.get(Race, race_id)
            if not race or not race.result or not race.result.is_valid:
                continue
            result = race.result
            if result.winner_car_no is None or result.second_car_no is None:
                continue

            entries = sorted(race.entries, key=lambda e: e.car_no)
            car_nos = [e.car_no for e in entries]
            actual_pair = (result.winner_car_no, result.second_car_no)

            odds_dict = _get_latest_odds_for_race(db, race_id)
            if not odds_dict:
                continue

            odds_stats = compute_runner_odds_stats(odds_dict, car_nos)
            race_date = race.race_day.race_date

            feats = np.array([
                extract_v16_features(
                    e, odds_stats,
                    racer_stats=get_racer_stats(db, e.racer_id, before_date=race_date),
                )
                for e in entries
            ])

            total_inv = sum(1.0 / o for o in odds_dict.values() if o > 0)
            if total_inv <= 0:
                continue
            market_pair_probs = {
                pair: (1.0 / o) / total_inv
                for pair, o in odds_dict.items() if o > 0
            }

            val_blend_data.append((feats, car_nos, market_pair_probs, actual_pair))

        if val_blend_data:
            final_model.fit_market_alpha(val_blend_data)

    logger.info("Feature importances (v16):")
    for name, imp in report.feature_importances:
        logger.info("  %s: %d", name, imp)

    return final_model, report


# -------------------------------------------------------------------
# v17 training (odds-free fundamental)
# -------------------------------------------------------------------


def build_v17_training_data(
    db: Session,
    date_from: date,
    date_to: date,
) -> list[TrainingRow]:
    """Build per-runner training data with v17 features (odds-free).

    Unlike v13-v16, races without odds are NOT skipped for training data
    (odds are not used as features). However, races still need valid results.
    """
    from app.services.modeling_v17 import extract_v17_features
    from app.services.racer_stats import get_racer_stats

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

    rows: list[TrainingRow] = []
    skipped = 0

    for race in races:
        entries = sorted(race.entries, key=lambda e: e.car_no)
        if len(entries) < 2:
            continue

        result = race.result
        if result is None or not result.is_valid or result.winner_car_no is None:
            continue

        car_nos = [e.car_no for e in entries]
        if result.winner_car_no not in car_nos:
            skipped += 1
            continue

        race_date = race.race_day.race_date

        for entry in entries:
            stats = get_racer_stats(db, entry.racer_id, before_date=race_date)
            features = extract_v17_features(entry, racer_stats=stats)
            label = 1 if entry.car_no == result.winner_car_no else 0
            rows.append(TrainingRow(
                features=features,
                label=label,
                race_id=race.race_id,
                race_date=race_date,
            ))

    logger.info(
        "Built %d v17 training rows from %d races (skipped %d)",
        len(rows), len(set(r.race_id for r in rows)), skipped,
    )
    return rows


def train_v17_model(
    db: Session,
    date_from: date,
    date_to: date,
    n_folds: int = 5,
    calibrate: bool = True,
) -> tuple:
    """Train v17 odds-free model with time-series CV + Platt calibration + market blend.

    Returns (ExactaModelV17, TrainingReport).
    """
    from lightgbm import LGBMClassifier

    from app.services.modeling_v17 import ExactaModelV17

    rows = build_v17_training_data(db, date_from, date_to)
    if not rows:
        logger.warning("No v17 training data found")
        return ExactaModelV17(), TrainingReport()

    report = TrainingReport(
        total_samples=len(rows),
        total_races=len(set(r.race_id for r in rows)),
    )

    # --- Cross-validation ---
    splits = _date_based_cv_splits(rows, n_folds)
    logger.info("Running %d-fold time-series CV (v17)", len(splits))

    last_val_idx: list[int] = []

    lgb_params = dict(
        n_estimators=100,
        max_depth=4,
        num_leaves=15,
        min_child_samples=50,
        reg_alpha=1.0,
        reg_lambda=1.0,
        random_state=42,
        verbose=-1,
    )

    for fold_i, (train_idx, val_idx) in enumerate(splits):
        X_train = np.array([rows[i].features for i in train_idx])
        y_train = np.array([rows[i].label for i in train_idx])

        clf = LGBMClassifier(**lgb_params)
        clf.fit(X_train, y_train)

        fold_model = ExactaModelV17(model=clf)
        logloss, top1 = _evaluate_fold(rows, val_idx, fold_model)

        val_races = len(set(rows[i].race_id for i in val_idx))
        fold_result = FoldResult(
            fold=fold_i + 1,
            train_size=len(train_idx),
            val_size=len(val_idx),
            val_races=val_races,
            logloss=logloss,
            top1_accuracy=top1,
        )
        report.fold_results.append(fold_result)
        last_val_idx = val_idx
        logger.info(
            "Fold %d: train=%d val=%d(%d races) logloss=%.4f top1=%.1f%%",
            fold_i + 1, len(train_idx), len(val_idx), val_races,
            logloss, top1 * 100,
        )

    logger.info(
        "CV mean: logloss=%.4f top1=%.1f%%",
        report.cv_mean_logloss, report.cv_mean_top1 * 100,
    )

    # --- Train final model on all data ---
    X_all = np.array([r.features for r in rows])
    y_all = np.array([r.label for r in rows])

    final_clf = LGBMClassifier(**lgb_params)
    final_clf.fit(X_all, y_all)

    final_model = ExactaModelV17(model=final_clf)
    report.feature_importances = final_model.get_feature_importance()

    # --- Platt calibration on last fold's validation data ---
    if calibrate and last_val_idx:
        race_groups: dict[int, list[int]] = {}
        for idx in last_val_idx:
            race_groups.setdefault(rows[idx].race_id, []).append(idx)

        val_feats_list: list[np.ndarray] = []
        val_labels_list: list[np.ndarray] = []
        for race_id, indices in race_groups.items():
            feats = np.array([rows[i].features for i in indices])
            labels = np.array([rows[i].label for i in indices])
            val_feats_list.append(feats)
            val_labels_list.append(labels)

        if val_feats_list:
            final_model.fit_calibrator(val_feats_list, val_labels_list)

    # --- Fit market blend alpha on last fold's validation data ---
    # Note: v17 features are odds-free, but alpha fitting still needs odds
    # to compute market_pair_probs for the blend target.
    if calibrate and last_val_idx:
        from app.services.modeling_v17 import extract_v17_features
        from app.services.racer_stats import get_racer_stats

        val_race_ids = sorted(set(rows[i].race_id for i in last_val_idx))
        val_blend_data: list[tuple] = []

        for race_id in val_race_ids:
            race = db.get(Race, race_id)
            if not race or not race.result or not race.result.is_valid:
                continue
            result = race.result
            if result.winner_car_no is None or result.second_car_no is None:
                continue

            entries = sorted(race.entries, key=lambda e: e.car_no)
            car_nos = [e.car_no for e in entries]
            actual_pair = (result.winner_car_no, result.second_car_no)

            odds_dict = _get_latest_odds_for_race(db, race_id)
            if not odds_dict:
                continue

            race_date = race.race_day.race_date

            # v17 features: no odds needed
            feats = np.array([
                extract_v17_features(
                    e,
                    racer_stats=get_racer_stats(db, e.racer_id, before_date=race_date),
                )
                for e in entries
            ])

            total_inv = sum(1.0 / o for o in odds_dict.values() if o > 0)
            if total_inv <= 0:
                continue
            market_pair_probs = {
                pair: (1.0 / o) / total_inv
                for pair, o in odds_dict.items() if o > 0
            }

            val_blend_data.append((feats, car_nos, market_pair_probs, actual_pair))

        if val_blend_data:
            final_model.fit_market_alpha(val_blend_data)

    logger.info("Feature importances (v17):")
    for name, imp in report.feature_importances:
        logger.info("  %s: %d", name, imp)

    return final_model, report


# -------------------------------------------------------------------
# v18 training (race-relative + interactions)
# -------------------------------------------------------------------


def build_v18_training_data(
    db: Session,
    date_from: date,
    date_to: date,
) -> list[TrainingRow]:
    """Build per-runner training data with v18 features (odds-free + race context).

    Like v17, races without odds are NOT skipped (odds are not features).
    Difference from v17: computes race_context per race and passes it to
    extract_v18_features.
    """
    from app.services.modeling_v18 import compute_race_context, extract_v18_features
    from app.services.racer_stats import get_racer_stats

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

    rows: list[TrainingRow] = []
    skipped = 0

    for race in races:
        entries = sorted(race.entries, key=lambda e: e.car_no)
        if len(entries) < 2:
            continue

        result = race.result
        if result is None or not result.is_valid or result.winner_car_no is None:
            continue

        car_nos = [e.car_no for e in entries]
        if result.winner_car_no not in car_nos:
            skipped += 1
            continue

        race_date = race.race_day.race_date
        race_ctx = compute_race_context(entries)

        for entry in entries:
            stats = get_racer_stats(db, entry.racer_id, before_date=race_date)
            features = extract_v18_features(
                entry, racer_stats=stats, race_context=race_ctx,
            )
            label = 1 if entry.car_no == result.winner_car_no else 0
            rows.append(TrainingRow(
                features=features,
                label=label,
                race_id=race.race_id,
                race_date=race_date,
            ))

    logger.info(
        "Built %d v18 training rows from %d races (skipped %d)",
        len(rows), len(set(r.race_id for r in rows)), skipped,
    )
    return rows


def build_v20_training_data(
    db: Session,
    date_from: date,
    date_to: date,
) -> dict[str, list[TrainingRow]]:
    """Build per-runner training data split by track_code.

    Returns a dict: {track_code: [TrainingRow, ...], "_all": [TrainingRow, ...]}.
    "_all" contains all rows regardless of track (for fallback model).
    """
    from app.db.models import Track
    from app.services.modeling_v18 import compute_race_context, extract_v18_features
    from app.services.racer_stats import get_racer_stats

    races = (
        db.execute(
            select(Race)
            .join(RaceDay)
            .join(Track, RaceDay.track_id == Track.track_id)
            .where(RaceDay.race_date >= date_from, RaceDay.race_date <= date_to)
            .order_by(RaceDay.race_date, Race.race_no)
        )
        .scalars()
        .all()
    )

    by_track: dict[str, list[TrainingRow]] = {"_all": []}
    skipped = 0

    for race in races:
        entries = sorted(race.entries, key=lambda e: e.car_no)
        if len(entries) < 2:
            continue

        result = race.result
        if result is None or not result.is_valid or result.winner_car_no is None:
            continue

        car_nos = [e.car_no for e in entries]
        if result.winner_car_no not in car_nos:
            skipped += 1
            continue

        race_date = race.race_day.race_date
        track_code = race.race_day.track.track_code
        race_ctx = compute_race_context(entries)

        for entry in entries:
            stats = get_racer_stats(db, entry.racer_id, before_date=race_date)
            features = extract_v18_features(
                entry, racer_stats=stats, race_context=race_ctx,
            )
            label = 1 if entry.car_no == result.winner_car_no else 0
            row = TrainingRow(
                features=features,
                label=label,
                race_id=race.race_id,
                race_date=race_date,
            )
            by_track.setdefault(track_code, []).append(row)
            by_track["_all"].append(row)

    for tc, rows in by_track.items():
        races_count = len(set(r.race_id for r in rows))
        logger.info("Track %s: %d rows, %d races", tc, len(rows), races_count)
    logger.info("Total skipped: %d", skipped)
    return by_track


def _train_single_v19(
    db: Session,
    rows: list[TrainingRow],
    n_folds: int,
    calibrate: bool,
    label: str = "",
) -> ExactaModelV19:
    """Train one ExactaModelV19 on the given rows. Shared by v19 and v20 training."""
    from lightgbm import LGBMClassifier
    from app.services.modeling_v19 import ExactaModelV19

    lgb_params = dict(
        n_estimators=100,
        max_depth=4,
        num_leaves=15,
        min_child_samples=50,
        reg_alpha=1.0,
        reg_lambda=1.0,
        random_state=42,
        verbose=-1,
    )

    splits = _date_based_cv_splits(rows, n_folds)
    oof_features: list[np.ndarray] = []
    oof_labels: list[np.ndarray] = []
    last_val_idx: list[int] = []

    for fold_i, (train_idx, val_idx) in enumerate(splits):
        X_train = np.array([rows[i].features for i in train_idx])
        y_train = np.array([rows[i].label for i in train_idx])
        clf = LGBMClassifier(**lgb_params)
        clf.fit(X_train, y_train)

        if calibrate:
            race_groups: dict[int, list[int]] = {}
            for idx in val_idx:
                race_groups.setdefault(rows[idx].race_id, []).append(idx)
            for race_id, indices in race_groups.items():
                feats = np.array([rows[i].features for i in indices])
                lbls = np.array([rows[i].label for i in indices])
                oof_features.append(feats)
                oof_labels.append(lbls)

        last_val_idx = val_idx

    # Final model on all data
    X_all = np.array([r.features for r in rows])
    y_all = np.array([r.label for r in rows])
    final_clf = LGBMClassifier(**lgb_params)
    final_clf.fit(X_all, y_all)
    final_model = ExactaModelV19(model=final_clf)

    # Isotonic calibration
    if calibrate and oof_features:
        final_model.fit_calibrator(oof_features, oof_labels)

    # Conditional alpha on last fold validation races
    if calibrate and last_val_idx:
        from app.services.modeling_v18 import compute_race_context, extract_v18_features
        from app.services.racer_stats import get_racer_stats

        val_race_ids = sorted(set(rows[i].race_id for i in last_val_idx))
        val_blend_data: list[tuple] = []
        val_odds_dicts: list[dict] = []

        for race_id in val_race_ids:
            race = db.get(Race, race_id)
            if not race or not race.result or not race.result.is_valid:
                continue
            result = race.result
            if result.winner_car_no is None or result.second_car_no is None:
                continue
            entries = sorted(race.entries, key=lambda e: e.car_no)
            car_nos = [e.car_no for e in entries]
            actual_pair = (result.winner_car_no, result.second_car_no)
            odds_dict = _get_latest_odds_for_race(db, race_id)
            if not odds_dict:
                continue
            race_date = race.race_day.race_date
            race_ctx = compute_race_context(entries)
            feats = np.array([
                extract_v18_features(
                    e,
                    racer_stats=get_racer_stats(db, e.racer_id, before_date=race_date),
                    race_context=race_ctx,
                )
                for e in entries
            ])
            total_inv = sum(1.0 / o for o in odds_dict.values() if o > 0)
            if total_inv <= 0:
                continue
            market_pair_probs = {
                pair: (1.0 / o) / total_inv
                for pair, o in odds_dict.items() if o > 0
            }
            val_blend_data.append((feats, car_nos, market_pair_probs, actual_pair))
            val_odds_dicts.append(odds_dict)

        if val_blend_data:
            final_model.fit_market_alpha(val_blend_data, odds_dicts=val_odds_dicts)

    logger.info(
        "Trained single v19 model [%s]: %d rows, alpha=%.2f, alpha_map=%s",
        label, len(rows), final_model.market_alpha,
        {b: f"{a:.2f}" for b, a in final_model.alpha_map.items()},
    )
    return final_model


def train_v20_model(
    db: Session,
    date_from: date,
    date_to: date,
    n_folds: int = 5,
    calibrate: bool = True,
    min_track_races: int = 100,
) -> tuple:
    """Train v20 multi-track model.

    For each track_code with >= min_track_races races, trains a dedicated
    ExactaModelV19. Tracks below the threshold use a global fallback model.

    Returns (ExactaModelV20, TrainingReport).
    """
    from app.services.modeling_v20 import ExactaModelV20, _FALLBACK_KEY

    by_track = build_v20_training_data(db, date_from, date_to)
    all_rows = by_track.get("_all", [])

    if not all_rows:
        logger.warning("No v20 training data found")
        from app.services.modeling_v19 import ExactaModelV19
        return ExactaModelV20(track_models={_FALLBACK_KEY: ExactaModelV19()}), TrainingReport()

    report = TrainingReport(
        total_samples=len(all_rows),
        total_races=len(set(r.race_id for r in all_rows)),
    )

    track_models: dict[str, ExactaModelV19] = {}

    # Train per-track models
    for track_code, rows in by_track.items():
        if track_code == "_all":
            continue
        n_races = len(set(r.race_id for r in rows))
        if n_races < min_track_races:
            logger.info(
                "Track %s: only %d races (< %d) — will use fallback",
                track_code, n_races, min_track_races,
            )
            continue

        actual_folds = min(n_folds, max(2, n_races // 50))
        logger.info(
            "Training track model [%s]: %d rows, %d races, %d folds",
            track_code, len(rows), n_races, actual_folds,
        )
        track_models[track_code] = _train_single_v19(
            db, rows, actual_folds, calibrate, label=track_code
        )

    # Train global fallback on all data
    logger.info("Training fallback model on all %d rows", len(all_rows))
    fallback_folds = min(n_folds, 5)
    track_models[_FALLBACK_KEY] = _train_single_v19(
        db, all_rows, fallback_folds, calibrate, label="fallback"
    )

    # Feature importances from fallback
    report.feature_importances = track_models[_FALLBACK_KEY].get_feature_importance()

    trained_tracks = [k for k in track_models if k != _FALLBACK_KEY]
    logger.info(
        "v20 trained: %d track models (%s) + fallback",
        len(trained_tracks), trained_tracks,
    )

    return ExactaModelV20(track_models=track_models), report


def train_v18_model(
    db: Session,
    date_from: date,
    date_to: date,
    n_folds: int = 5,
    calibrate: bool = True,
) -> tuple:
    """Train v18 model with time-series CV + Platt calibration + market blend.

    Returns (ExactaModelV18, TrainingReport).
    """
    from lightgbm import LGBMClassifier

    from app.services.modeling_v18 import ExactaModelV18

    rows = build_v18_training_data(db, date_from, date_to)
    if not rows:
        logger.warning("No v18 training data found")
        return ExactaModelV18(), TrainingReport()

    report = TrainingReport(
        total_samples=len(rows),
        total_races=len(set(r.race_id for r in rows)),
    )

    # --- Cross-validation ---
    splits = _date_based_cv_splits(rows, n_folds)
    logger.info("Running %d-fold time-series CV (v18)", len(splits))

    last_val_idx: list[int] = []

    lgb_params = dict(
        n_estimators=100,
        max_depth=4,
        num_leaves=15,
        min_child_samples=50,
        reg_alpha=1.0,
        reg_lambda=1.0,
        random_state=42,
        verbose=-1,
    )

    for fold_i, (train_idx, val_idx) in enumerate(splits):
        X_train = np.array([rows[i].features for i in train_idx])
        y_train = np.array([rows[i].label for i in train_idx])

        clf = LGBMClassifier(**lgb_params)
        clf.fit(X_train, y_train)

        fold_model = ExactaModelV18(model=clf)
        logloss, top1 = _evaluate_fold(rows, val_idx, fold_model)

        val_races = len(set(rows[i].race_id for i in val_idx))
        fold_result = FoldResult(
            fold=fold_i + 1,
            train_size=len(train_idx),
            val_size=len(val_idx),
            val_races=val_races,
            logloss=logloss,
            top1_accuracy=top1,
        )
        report.fold_results.append(fold_result)
        last_val_idx = val_idx
        logger.info(
            "Fold %d: train=%d val=%d(%d races) logloss=%.4f top1=%.1f%%",
            fold_i + 1, len(train_idx), len(val_idx), val_races,
            logloss, top1 * 100,
        )

    logger.info(
        "CV mean: logloss=%.4f top1=%.1f%%",
        report.cv_mean_logloss, report.cv_mean_top1 * 100,
    )

    # --- Train final model on all data ---
    X_all = np.array([r.features for r in rows])
    y_all = np.array([r.label for r in rows])

    final_clf = LGBMClassifier(**lgb_params)
    final_clf.fit(X_all, y_all)

    final_model = ExactaModelV18(model=final_clf)
    report.feature_importances = final_model.get_feature_importance()

    # --- Platt calibration on last fold's validation data ---
    if calibrate and last_val_idx:
        race_groups: dict[int, list[int]] = {}
        for idx in last_val_idx:
            race_groups.setdefault(rows[idx].race_id, []).append(idx)

        val_feats_list: list[np.ndarray] = []
        val_labels_list: list[np.ndarray] = []
        for race_id, indices in race_groups.items():
            feats = np.array([rows[i].features for i in indices])
            labels = np.array([rows[i].label for i in indices])
            val_feats_list.append(feats)
            val_labels_list.append(labels)

        if val_feats_list:
            final_model.fit_calibrator(val_feats_list, val_labels_list)

    # --- Fit market blend alpha on last fold's validation data ---
    # v18 features are odds-free, but alpha fitting still needs odds
    # to compute market_pair_probs for the blend target.
    if calibrate and last_val_idx:
        from app.services.modeling_v18 import compute_race_context, extract_v18_features
        from app.services.racer_stats import get_racer_stats

        val_race_ids = sorted(set(rows[i].race_id for i in last_val_idx))
        val_blend_data: list[tuple] = []

        for race_id in val_race_ids:
            race = db.get(Race, race_id)
            if not race or not race.result or not race.result.is_valid:
                continue
            result = race.result
            if result.winner_car_no is None or result.second_car_no is None:
                continue

            entries = sorted(race.entries, key=lambda e: e.car_no)
            car_nos = [e.car_no for e in entries]
            actual_pair = (result.winner_car_no, result.second_car_no)

            odds_dict = _get_latest_odds_for_race(db, race_id)
            if not odds_dict:
                continue

            race_date = race.race_day.race_date
            race_ctx = compute_race_context(entries)

            # v18 features: no odds needed
            feats = np.array([
                extract_v18_features(
                    e,
                    racer_stats=get_racer_stats(db, e.racer_id, before_date=race_date),
                    race_context=race_ctx,
                )
                for e in entries
            ])

            total_inv = sum(1.0 / o for o in odds_dict.values() if o > 0)
            if total_inv <= 0:
                continue
            market_pair_probs = {
                pair: (1.0 / o) / total_inv
                for pair, o in odds_dict.items() if o > 0
            }

            val_blend_data.append((feats, car_nos, market_pair_probs, actual_pair))

        if val_blend_data:
            final_model.fit_market_alpha(val_blend_data)

    logger.info("Feature importances (v18):")
    for name, imp in report.feature_importances:
        logger.info("  %s: %d", name, imp)

    return final_model, report


# -------------------------------------------------------------------
# v19 training (isotonic calibration + conditional alpha)
# -------------------------------------------------------------------


def train_v19_model(
    db: Session,
    date_from: date,
    date_to: date,
    n_folds: int = 5,
    calibrate: bool = True,
) -> tuple:
    """Train v19 model with isotonic calibration + conditional alpha.

    Key differences from v18 training:
    - OOF (out-of-fold) calibration data from ALL folds (not just last)
    - Isotonic regression instead of Platt sigmoid
    - Conditional alpha fitting by odds bucket

    Returns (ExactaModelV19, TrainingReport).
    """
    from lightgbm import LGBMClassifier

    from app.services.modeling_v19 import ExactaModelV19

    rows = build_v18_training_data(db, date_from, date_to)
    if not rows:
        logger.warning("No v19 training data found")
        return ExactaModelV19(), TrainingReport()

    report = TrainingReport(
        total_samples=len(rows),
        total_races=len(set(r.race_id for r in rows)),
    )

    # --- Cross-validation with OOF collection ---
    splits = _date_based_cv_splits(rows, n_folds)
    logger.info("Running %d-fold time-series CV (v19)", len(splits))

    lgb_params = dict(
        n_estimators=100,
        max_depth=4,
        num_leaves=15,
        min_child_samples=50,
        reg_alpha=1.0,
        reg_lambda=1.0,
        random_state=42,
        verbose=-1,
    )

    # Collect OOF predictions from all folds for isotonic calibration
    oof_features: list[np.ndarray] = []  # per-race feature arrays
    oof_labels: list[np.ndarray] = []    # per-race label arrays
    last_val_idx: list[int] = []

    for fold_i, (train_idx, val_idx) in enumerate(splits):
        X_train = np.array([rows[i].features for i in train_idx])
        y_train = np.array([rows[i].label for i in train_idx])

        clf = LGBMClassifier(**lgb_params)
        clf.fit(X_train, y_train)

        fold_model = ExactaModelV19(model=clf)
        logloss, top1 = _evaluate_fold(rows, val_idx, fold_model)

        val_races = len(set(rows[i].race_id for i in val_idx))
        fold_result = FoldResult(
            fold=fold_i + 1,
            train_size=len(train_idx),
            val_size=len(val_idx),
            val_races=val_races,
            logloss=logloss,
            top1_accuracy=top1,
        )
        report.fold_results.append(fold_result)
        last_val_idx = val_idx
        logger.info(
            "Fold %d: train=%d val=%d(%d races) logloss=%.4f top1=%.1f%%",
            fold_i + 1, len(train_idx), len(val_idx), val_races,
            logloss, top1 * 100,
        )

        # Collect OOF data for calibration (this fold's model on this fold's val)
        if calibrate:
            race_groups: dict[int, list[int]] = {}
            for idx in val_idx:
                race_groups.setdefault(rows[idx].race_id, []).append(idx)
            for race_id, indices in race_groups.items():
                feats = np.array([rows[i].features for i in indices])
                labels = np.array([rows[i].label for i in indices])
                # Store raw predictions from THIS fold's model
                raw_preds = clf.predict_proba(feats)[:, 1]
                oof_features.append(feats)
                oof_labels.append(labels)

    logger.info(
        "CV mean: logloss=%.4f top1=%.1f%%",
        report.cv_mean_logloss, report.cv_mean_top1 * 100,
    )

    # --- Train final model on all data ---
    X_all = np.array([r.features for r in rows])
    y_all = np.array([r.label for r in rows])

    final_clf = LGBMClassifier(**lgb_params)
    final_clf.fit(X_all, y_all)

    final_model = ExactaModelV19(model=final_clf)
    report.feature_importances = final_model.get_feature_importance()

    # --- Isotonic calibration on ALL OOF data ---
    if calibrate and oof_features:
        logger.info("Fitting isotonic calibrator on %d OOF race groups", len(oof_features))
        final_model.fit_calibrator(oof_features, oof_labels)

    # --- Fit conditional alpha on last fold's validation data ---
    if calibrate and last_val_idx:
        from app.services.modeling_v18 import compute_race_context, extract_v18_features
        from app.services.racer_stats import get_racer_stats

        val_race_ids = sorted(set(rows[i].race_id for i in last_val_idx))
        val_blend_data: list[tuple] = []
        val_odds_dicts: list[dict] = []

        for race_id in val_race_ids:
            race = db.get(Race, race_id)
            if not race or not race.result or not race.result.is_valid:
                continue
            result = race.result
            if result.winner_car_no is None or result.second_car_no is None:
                continue

            entries = sorted(race.entries, key=lambda e: e.car_no)
            car_nos = [e.car_no for e in entries]
            actual_pair = (result.winner_car_no, result.second_car_no)

            odds_dict = _get_latest_odds_for_race(db, race_id)
            if not odds_dict:
                continue

            race_date = race.race_day.race_date
            race_ctx = compute_race_context(entries)

            feats = np.array([
                extract_v18_features(
                    e,
                    racer_stats=get_racer_stats(db, e.racer_id, before_date=race_date),
                    race_context=race_ctx,
                )
                for e in entries
            ])

            total_inv = sum(1.0 / o for o in odds_dict.values() if o > 0)
            if total_inv <= 0:
                continue
            market_pair_probs = {
                pair: (1.0 / o) / total_inv
                for pair, o in odds_dict.items() if o > 0
            }

            val_blend_data.append((feats, car_nos, market_pair_probs, actual_pair))
            val_odds_dicts.append(odds_dict)

        if val_blend_data:
            final_model.fit_market_alpha(val_blend_data, odds_dicts=val_odds_dicts)

    logger.info("Feature importances (v19):")
    for name, imp in report.feature_importances:
        logger.info("  %s: %d", name, imp)

    return final_model, report
