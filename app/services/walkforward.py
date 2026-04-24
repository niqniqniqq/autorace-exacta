"""Walk-forward evaluation engine with market baseline comparison."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import OddsExacta, Race, RaceDay, RaceEntry, RaceResult

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardSplit:
    """Time boundaries for one walk-forward split."""

    train_from: date
    train_to: date  # exclusive
    val_from: date
    val_to: date  # exclusive
    test_from: date
    test_to: date  # exclusive


@dataclass
class SplitResult:
    """Evaluation results for one split."""

    split_idx: int
    split: WalkForwardSplit
    # Model metrics (2連単ペアレベル)
    model_logloss: float
    model_brier: float
    model_top1: float
    # Market baseline metrics
    baseline_logloss: float
    baseline_brier: float
    baseline_top1: float
    # Meta
    n_races: int
    n_pairs: int


@dataclass
class WalkForwardReport:
    """Aggregated walk-forward results."""

    splits: list[SplitResult] = field(default_factory=list)

    def _metric_stats(self, attr: str) -> tuple[float, float]:
        vals = [getattr(s, attr) for s in self.splits if not np.isinf(getattr(s, attr))]
        if not vals:
            return float("inf"), 0.0
        return float(np.mean(vals)), float(np.std(vals))

    @property
    def model_logloss(self) -> tuple[float, float]:
        return self._metric_stats("model_logloss")

    @property
    def model_brier(self) -> tuple[float, float]:
        return self._metric_stats("model_brier")

    @property
    def model_top1(self) -> tuple[float, float]:
        return self._metric_stats("model_top1")

    @property
    def baseline_logloss(self) -> tuple[float, float]:
        return self._metric_stats("baseline_logloss")

    @property
    def baseline_brier(self) -> tuple[float, float]:
        return self._metric_stats("baseline_brier")

    @property
    def baseline_top1(self) -> tuple[float, float]:
        return self._metric_stats("baseline_top1")


def generate_splits(
    date_from: date,
    date_to: date,
    train_days: int = 60,
    val_days: int = 7,
    test_days: int = 7,
    step_days: int = 7,
) -> list[WalkForwardSplit]:
    """Generate rolling walk-forward splits.

    For each step:
        train = [t - train_days - val_days, t - val_days)
        val   = [t - val_days, t)
        test  = [t, t + test_days)

    Starting t = date_from + train_days + val_days, stepping by step_days.
    """
    splits: list[WalkForwardSplit] = []
    window = train_days + val_days
    t = date_from + timedelta(days=window)

    while t + timedelta(days=test_days) <= date_to + timedelta(days=1):
        split = WalkForwardSplit(
            train_from=t - timedelta(days=window),
            train_to=t - timedelta(days=val_days),
            val_from=t - timedelta(days=val_days),
            val_to=t,
            test_from=t,
            test_to=t + timedelta(days=test_days),
        )
        splits.append(split)
        t += timedelta(days=step_days)

    return splits


def _get_races_in_period(
    db: Session,
    date_from: date,
    date_to: date,
) -> list[Race]:
    """Get races with valid results in [date_from, date_to)."""
    return (
        db.execute(
            select(Race)
            .join(RaceDay)
            .join(RaceResult)
            .where(
                RaceDay.race_date >= date_from,
                RaceDay.race_date < date_to,
                RaceResult.is_valid == True,
                RaceResult.winner_car_no.isnot(None),
                RaceResult.second_car_no.isnot(None),
            )
            .order_by(RaceDay.race_date, Race.race_no)
        )
        .scalars()
        .all()
    )


def _get_latest_odds_dict(
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


def compute_market_baseline(
    db: Session,
    race_id: int,
) -> dict[tuple[int, int], float]:
    """Compute market-implied exacta probabilities from odds.

    P_market(i→j) = (1/odds_ij) / Σ(1/odds_kl)
    """
    odds_dict = _get_latest_odds_dict(db, race_id)
    if not odds_dict:
        return {}

    total_inv = sum(1.0 / o for o in odds_dict.values() if o > 0)
    if total_inv <= 0:
        return {}

    return {
        pair: (1.0 / odds_val) / total_inv
        for pair, odds_val in odds_dict.items()
        if odds_val > 0
    }


def _evaluate_race_metrics(
    prob_map: dict[tuple[int, int], float],
    actual: tuple[int, int],
) -> tuple[float, float, bool]:
    """Compute (logloss, brier, top1_hit) for one race.

    logloss = -log(P(actual pair))
    brier = Σ_pairs (P - outcome)²
    top1 = argmax(P) == actual
    """
    eps = 1e-15

    if not prob_map:
        return float("inf"), float("inf"), False

    # LogLoss
    actual_prob = prob_map.get(actual, eps)
    logloss = -np.log(max(actual_prob, eps))

    # Brier
    brier = 0.0
    for pair, prob in prob_map.items():
        outcome = 1.0 if pair == actual else 0.0
        brier += (prob - outcome) ** 2

    # Top-1
    top_pair = max(prob_map, key=prob_map.get)
    top1_hit = top_pair == actual

    return float(logloss), float(brier), top1_hit


def evaluate_split(
    db: Session,
    split: WalkForwardSplit,
    split_idx: int,
    build_training_data_fn,
    train_model_fn,
    extract_features_fn,
) -> SplitResult | None:
    """Evaluate one walk-forward split.

    Args:
        db: Database session
        split: Time boundaries
        split_idx: Index of this split
        build_training_data_fn: (db, date_from, date_to) -> list[TrainingRow]
        train_model_fn: (rows) -> model  — trains a model from training rows
        extract_features_fn: (db, race, entries, odds_dict) -> (car_nos, features)
    """
    from app.services.modeling_v12 import extract_v12_features
    from app.services.modeling_v13 import compute_runner_odds_stats, extract_v13_features

    # 1. Build training data and train model
    train_rows = build_training_data_fn(db, split.train_from, split.train_to)
    if not train_rows:
        logger.warning("Split %d: no training data — skipping", split_idx)
        return None

    model = train_model_fn(train_rows)

    # 2. Fit calibrator on validation data if model supports it
    val_races = _get_races_in_period(db, split.val_from, split.val_to)
    if hasattr(model, "fit_calibrator") and val_races:
        val_feats_list: list[np.ndarray] = []
        val_labels_list: list[np.ndarray] = []
        for race in val_races:
            entries = sorted(race.entries, key=lambda e: e.car_no)
            if len(entries) < 2:
                continue
            result = race.result
            if result is None or not result.is_valid:
                continue
            odds_dict = _get_latest_odds_dict(db, race.race_id)
            car_nos, feats = extract_features_fn(db, race, entries, odds_dict)
            if len(car_nos) < 2:
                continue
            labels = np.array([
                1 if e.car_no == result.winner_car_no else 0
                for e in entries
                if e.car_no in car_nos
            ])
            if len(labels) == len(car_nos):
                val_feats_list.append(feats)
                val_labels_list.append(labels)

        if val_feats_list:
            model.fit_calibrator(val_feats_list, val_labels_list)

    # 2b. Fit market blend alpha on validation data
    if hasattr(model, "fit_market_alpha") and val_races:
        val_blend_data: list[tuple] = []
        for race in val_races:
            entries = sorted(race.entries, key=lambda e: e.car_no)
            if len(entries) < 2:
                continue
            result = race.result
            if result is None or not result.is_valid:
                continue
            if result.winner_car_no is None or result.second_car_no is None:
                continue
            actual = (result.winner_car_no, result.second_car_no)

            odds_dict = _get_latest_odds_dict(db, race.race_id)
            if not odds_dict:
                continue
            car_nos, feats = extract_features_fn(db, race, entries, odds_dict)
            if len(car_nos) < 2:
                continue

            total_inv = sum(1.0 / o for o in odds_dict.values() if o > 0)
            if total_inv <= 0:
                continue
            market_pair_probs = {
                pair: (1.0 / o) / total_inv
                for pair, o in odds_dict.items() if o > 0
            }

            val_blend_data.append((feats, car_nos, market_pair_probs, actual))

        if val_blend_data:
            model.fit_market_alpha(val_blend_data)

    # 3. Evaluate on test period
    test_races = _get_races_in_period(db, split.test_from, split.test_to)
    if not test_races:
        logger.warning("Split %d: no test races — skipping", split_idx)
        return None

    model_logloss_sum = 0.0
    model_brier_sum = 0.0
    model_top1_hits = 0
    baseline_logloss_sum = 0.0
    baseline_brier_sum = 0.0
    baseline_top1_hits = 0
    n_races = 0
    n_pairs = 0

    for race in test_races:
        entries = sorted(race.entries, key=lambda e: e.car_no)
        if len(entries) < 2:
            continue

        result = race.result
        if result is None or not result.is_valid:
            continue
        actual = (result.winner_car_no, result.second_car_no)

        odds_dict = _get_latest_odds_dict(db, race.race_id)

        # Model predictions
        car_nos, feats = extract_features_fn(db, race, entries, odds_dict)
        if len(car_nos) < 2:
            continue

        # Market baseline (also used for blend)
        baseline_prob_map = compute_market_baseline(db, race.race_id)
        if not baseline_prob_map:
            # Skip race if no baseline available
            continue

        # Model predictions (with blend if model supports it)
        if hasattr(model, "market_alpha"):
            preds = model.predict_exacta(feats, car_nos, market_pair_probs=baseline_prob_map)
        else:
            preds = model.predict_exacta(feats, car_nos)
        model_prob_map = {(f, s): p for f, s, p in preds}

        m_ll, m_br, m_top1 = _evaluate_race_metrics(model_prob_map, actual)

        b_ll, b_br, b_top1 = _evaluate_race_metrics(baseline_prob_map, actual)

        model_logloss_sum += m_ll
        model_brier_sum += m_br
        if m_top1:
            model_top1_hits += 1

        baseline_logloss_sum += b_ll
        baseline_brier_sum += b_br
        if b_top1:
            baseline_top1_hits += 1

        n_races += 1
        n_pairs += len(model_prob_map)

    if n_races == 0:
        logger.warning("Split %d: no evaluable test races", split_idx)
        return None

    return SplitResult(
        split_idx=split_idx,
        split=split,
        model_logloss=model_logloss_sum / n_races,
        model_brier=model_brier_sum / n_races,
        model_top1=model_top1_hits / n_races,
        baseline_logloss=baseline_logloss_sum / n_races,
        baseline_brier=baseline_brier_sum / n_races,
        baseline_top1=baseline_top1_hits / n_races,
        n_races=n_races,
        n_pairs=n_pairs,
    )


def run_walkforward(
    db: Session,
    splits: list[WalkForwardSplit],
    build_training_data_fn,
    train_model_fn,
    extract_features_fn,
) -> WalkForwardReport:
    """Run walk-forward evaluation over all splits.

    Args:
        db: Database session
        splits: List of WalkForwardSplit
        build_training_data_fn: (db, date_from, date_to) -> list[TrainingRow]
        train_model_fn: (rows) -> model
        extract_features_fn: (db, race, entries, odds_dict) -> (car_nos, features)
    """
    report = WalkForwardReport()

    for i, split in enumerate(splits):
        logger.info(
            "Split %d/%d: train=[%s, %s) val=[%s, %s) test=[%s, %s)",
            i + 1, len(splits),
            split.train_from, split.train_to,
            split.val_from, split.val_to,
            split.test_from, split.test_to,
        )
        result = evaluate_split(
            db, split, i + 1,
            build_training_data_fn,
            train_model_fn,
            extract_features_fn,
        )
        if result is not None:
            report.splits.append(result)
            logger.info(
                "  Model:    LogLoss=%.3f  Brier=%.4f  Top1=%.1f%%  (%d races)",
                result.model_logloss, result.model_brier,
                result.model_top1 * 100, result.n_races,
            )
            logger.info(
                "  Baseline: LogLoss=%.3f  Brier=%.4f  Top1=%.1f%%",
                result.baseline_logloss, result.baseline_brier,
                result.baseline_top1 * 100,
            )

    return report
