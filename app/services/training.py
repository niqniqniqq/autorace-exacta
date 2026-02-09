"""Training pipeline for v12 model with time-series cross-validation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Race, RaceDay, RaceEntry, RaceResult
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
