"""LightGBM model for exacta prediction (v13).

Extends v12 (9 features) with odds-derived features (3) = 12 features.
Includes optional Platt calibration for probability estimates.
"""

from __future__ import annotations

import logging
import math
import pickle
from pathlib import Path

import numpy as np

from app.services.modeling_v12 import ExactaModelV12, V12_FEATURE_NAMES, extract_v12_features

logger = logging.getLogger(__name__)

V13_FEATURE_NAMES = [
    *V12_FEATURE_NAMES,
    "implied_win_prob",
    "log_implied_win_odds",
    "odds_rank",
]

# Defaults when odds are missing
_DEFAULT_N_RUNNERS = 8


def compute_runner_odds_stats(
    odds_dict: dict[tuple[int, int], float],
    car_nos: list[int],
) -> dict[int, dict[str, float]]:
    """Compute per-runner odds statistics from exacta odds.

    For each runner i, implied_win_prob is the share of total inverse odds
    where runner i is first:
        implied_win_prob(i) = sum_j(1/odds(i,j)) / sum_all(1/odds(k,l))

    Args:
        odds_dict: {(first_car_no, second_car_no): odds_value}
        car_nos: list of active car numbers in this race

    Returns:
        {car_no: {"implied_win_prob": float, "log_implied_win_odds": float, "odds_rank": float}}
    """
    n = len(car_nos)
    if n < 2:
        return {
            c: {
                "implied_win_prob": 1.0 / max(n, 1),
                "log_implied_win_odds": math.log(max(n, 1)),
                "odds_rank": 0.5,
            }
            for c in car_nos
        }

    # Sum of 1/odds for each runner as first
    inv_sums: dict[int, float] = {c: 0.0 for c in car_nos}
    total_inv = 0.0
    for (first, second), odds_val in odds_dict.items():
        if first in inv_sums and odds_val > 0:
            inv = 1.0 / odds_val
            inv_sums[first] += inv
            total_inv += inv

    if total_inv <= 0:
        # No valid odds — return defaults
        return {
            c: {
                "implied_win_prob": 1.0 / n,
                "log_implied_win_odds": math.log(n),
                "odds_rank": 0.5,
            }
            for c in car_nos
        }

    # Compute implied_win_prob
    probs: dict[int, float] = {}
    for c in car_nos:
        probs[c] = inv_sums[c] / total_inv

    # Sort by prob descending for ranking
    sorted_cars = sorted(car_nos, key=lambda c: probs[c], reverse=True)
    rank_map: dict[int, int] = {}
    for rank_i, c in enumerate(sorted_cars):
        rank_map[c] = rank_i

    result: dict[int, dict[str, float]] = {}
    for c in car_nos:
        p = probs[c]
        eps = 1e-15
        result[c] = {
            "implied_win_prob": p,
            "log_implied_win_odds": -math.log(max(p, eps)),
            "odds_rank": rank_map[c] / max(n - 1, 1),  # 0=favorite, 1=longshot
        }

    return result


def extract_v13_features(
    entry,
    odds_stats: dict[int, dict[str, float]] | None = None,
) -> np.ndarray:
    """Extract 12 features from a RaceEntry + odds statistics."""
    base = extract_v12_features(entry)

    if odds_stats and entry.car_no in odds_stats:
        stats = odds_stats[entry.car_no]
        odds_feats = np.array([
            stats["implied_win_prob"],
            stats["log_implied_win_odds"],
            stats["odds_rank"],
        ], dtype=np.float64)
    else:
        # Default: uniform probability for n=8 runners
        n = _DEFAULT_N_RUNNERS
        odds_feats = np.array([
            1.0 / n,
            math.log(n),
            0.5,
        ], dtype=np.float64)

    return np.concatenate([base, odds_feats])


class ExactaModelV13(ExactaModelV12):
    """LightGBM v13 model — v12 + odds features + Platt calibration + market blend."""

    def __init__(
        self,
        model=None,
        calibrator: tuple[float, float] | None = None,
        market_alpha: float = 0.0,
    ) -> None:
        super().__init__(model)
        self.feature_names = V13_FEATURE_NAMES
        self.n_features = len(V13_FEATURE_NAMES)
        self.calibrator = calibrator  # (a, b) for Platt: P_cal = 1/(1+exp(a*f+b))
        self.market_alpha = market_alpha  # blend weight: alpha*model + (1-alpha)*market

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Predict win probability, optionally with Platt calibration."""
        raw = super().predict_proba(features)
        if self.calibrator is not None:
            a, b = self.calibrator
            raw = 1.0 / (1.0 + np.exp(a * raw + b))
        return raw

    def fit_calibrator(
        self,
        val_features_list: list[np.ndarray],
        val_labels_list: list[np.ndarray],
    ) -> None:
        """Fit Platt scaling parameters (a, b) on validation data.

        Args:
            val_features_list: list of feature arrays, one per race
            val_labels_list: list of label arrays (1=winner), one per race
        """
        from scipy.optimize import minimize

        all_raw: list[float] = []
        all_labels: list[float] = []

        for feats, labels in zip(val_features_list, val_labels_list):
            raw = super().predict_proba(feats)
            all_raw.extend(raw.tolist())
            all_labels.extend(labels.tolist())

        if not all_raw:
            logger.warning("No validation data for calibrator — skipping")
            return

        raw_arr = np.array(all_raw)
        label_arr = np.array(all_labels)

        def neg_log_likelihood(params: np.ndarray) -> float:
            a, b = params
            p = 1.0 / (1.0 + np.exp(a * raw_arr + b))
            eps = 1e-15
            p = np.clip(p, eps, 1.0 - eps)
            return float(-np.sum(
                label_arr * np.log(p) + (1.0 - label_arr) * np.log(1.0 - p)
            ))

        result = minimize(neg_log_likelihood, x0=np.array([0.0, 0.0]), method="Nelder-Mead")
        self.calibrator = (float(result.x[0]), float(result.x[1]))
        logger.info("Platt calibrator fitted: a=%.4f, b=%.4f", *self.calibrator)

    def predict_exacta(
        self,
        features: np.ndarray,
        car_nos: list[int],
        market_pair_probs: dict[tuple[int, int], float] | None = None,
    ) -> list[tuple[int, int, float]]:
        """Predict exacta probabilities, optionally blending with market.

        When market_alpha > 0 and market_pair_probs is provided:
            P_blend = alpha * P_model_PL + (1 - alpha) * P_market
        """
        preds = super().predict_exacta(features, car_nos)

        if market_pair_probs is None:
            return preds

        model_map = {(f, s): p for f, s, p in preds}
        alpha = self.market_alpha
        all_keys = set(model_map) | set(market_pair_probs)

        blended: dict[tuple[int, int], float] = {}
        for k in all_keys:
            m = model_map.get(k, 0.0)
            mk = market_pair_probs.get(k, 0.0)
            blended[k] = alpha * m + (1 - alpha) * mk

        # Renormalize
        total = sum(blended.values())
        if total > 0:
            blended = {k: v / total for k, v in blended.items()}

        result = [(f, s, p) for (f, s), p in blended.items()]
        result.sort(key=lambda x: x[2], reverse=True)
        return result

    def fit_market_alpha(
        self,
        val_races: list[tuple[np.ndarray, list[int], dict[tuple[int, int], float], tuple[int, int]]],
    ) -> None:
        """Fit optimal market blend alpha on validation races.

        Sweeps alpha from 0.0 to 1.0 and picks the value that minimizes
        pair-level LogLoss on validation data.

        Args:
            val_races: list of (features, car_nos, market_pair_probs, actual_pair)
        """
        if not val_races:
            logger.warning("No validation races for market alpha — skipping")
            return

        alphas = [i * 0.05 for i in range(21)]  # 0.00, 0.05, ..., 1.00
        best_alpha = 0.0
        best_ll = float("inf")

        for alpha in alphas:
            ll_sum = 0.0
            n = 0
            for feats, car_nos, market_probs, actual in val_races:
                # Get model PL predictions (uses calibrated probs)
                preds = super().predict_exacta(feats, car_nos)
                model_map = {(f, s): p for f, s, p in preds}

                # Blend
                all_keys = set(model_map) | set(market_probs)
                blended: dict[tuple[int, int], float] = {}
                for k in all_keys:
                    m = model_map.get(k, 0.0)
                    mk = market_probs.get(k, 0.0)
                    blended[k] = alpha * m + (1 - alpha) * mk

                total_p = sum(blended.values())
                if total_p > 0:
                    blended = {k: v / total_p for k, v in blended.items()}

                eps = 1e-15
                p = blended.get(actual, eps)
                ll_sum += -math.log(max(p, eps))
                n += 1

            if n > 0:
                ll = ll_sum / n
                if ll < best_ll:
                    best_ll = ll
                    best_alpha = alpha

        self.market_alpha = best_alpha
        logger.info("Market alpha fitted: %.2f (LogLoss=%.4f)", best_alpha, best_ll)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "feature_names": self.feature_names,
                "model_type": "v13_lgb",
                "calibrator": self.calibrator,
                "market_alpha": self.market_alpha,
            }, f)
        logger.info("Model v13 saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> ExactaModelV13:
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls(
            model=data["model"],
            calibrator=data.get("calibrator"),
            market_alpha=data.get("market_alpha", 0.0),
        )
        obj.feature_names = data.get("feature_names", V13_FEATURE_NAMES)
        logger.info("Model v13 loaded from %s", path)
        return obj

    @classmethod
    def is_v13_model(cls, path: str | Path) -> bool:
        """Check if the pickle file is a v13 model."""
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            return data.get("model_type") == "v13_lgb"
        except Exception:
            return False
