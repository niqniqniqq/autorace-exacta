"""LightGBM model for exacta prediction (v19).

Extends v18 (22 features) with improved calibration and blending:
- Isotonic regression calibration (replaces Platt sigmoid)
- Conditional alpha blending by odds bucket (replaces uniform alpha)
- OOF calibration data (all folds, not just last fold)

Same 22 features as v18 (race-relative + interactions).
"""

from __future__ import annotations

import logging
import math
import pickle
from pathlib import Path

import numpy as np

from app.services.modeling_v13 import ExactaModelV13
from app.services.modeling_v18 import (
    V18_FEATURE_NAMES,
    ExactaModelV18,
    RaceContext,
    compute_race_context,
    extract_v18_features,
)

logger = logging.getLogger(__name__)

V19_FEATURE_NAMES = V18_FEATURE_NAMES  # Same 22 features

# Odds bucket boundaries for conditional alpha
_ODDS_BUCKETS = {
    "low": (0.0, 20.0),       # favorites: trust market
    "mid": (20.0, 80.0),      # mid-range
    "high": (80.0, 300.0),    # longshots: model may have edge
    "extreme": (300.0, float("inf")),  # extreme longshots
}


def _odds_bucket(odds: float) -> str:
    """Classify odds into a bucket."""
    for name, (lo, hi) in _ODDS_BUCKETS.items():
        if lo <= odds < hi:
            return name
    return "extreme"


class ExactaModelV19(ExactaModelV13):
    """LightGBM v19 model — isotonic calibration + conditional alpha.

    Same 22 features as v18. Improvements are in calibration and blending.
    """

    def __init__(
        self,
        model=None,
        calibrator=None,
        market_alpha: float = 0.0,
        alpha_map: dict[str, float] | None = None,
    ) -> None:
        super().__init__(model, calibrator=None, market_alpha=market_alpha)
        self.feature_names = V19_FEATURE_NAMES
        self.n_features = len(V19_FEATURE_NAMES)
        self._isotonic = calibrator  # IsotonicRegression object or None
        self.alpha_map = alpha_map or {}  # {bucket_name: alpha}

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Predict win probability with isotonic calibration."""
        # Get raw probabilities from LightGBM (skip v13's Platt)
        raw = self._raw_predict_proba(features)
        if self._isotonic is not None:
            raw = self._isotonic.transform(raw)
            raw = np.clip(raw, 1e-6, 1.0 - 1e-6)
        return raw

    def _raw_predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Get raw LightGBM probabilities (no calibration)."""
        if self.model is None:
            n = features.shape[0] if features.ndim > 1 else 1
            return np.ones(n) / n
        if features.ndim == 1:
            features = features.reshape(1, -1)
        proba = self.model.predict_proba(features)
        return proba[:, 1] if proba.ndim == 2 else proba

    def fit_calibrator(
        self,
        val_features_list: list[np.ndarray],
        val_labels_list: list[np.ndarray],
    ) -> None:
        """Fit isotonic regression calibrator on validation data."""
        from sklearn.isotonic import IsotonicRegression

        all_raw: list[float] = []
        all_labels: list[float] = []

        for feats, labels in zip(val_features_list, val_labels_list):
            raw = self._raw_predict_proba(feats)
            all_raw.extend(raw.tolist())
            all_labels.extend(labels.tolist())

        if not all_raw:
            logger.warning("No validation data for isotonic calibrator — skipping")
            return

        ir = IsotonicRegression(y_min=1e-6, y_max=1.0 - 1e-6, out_of_bounds="clip")
        ir.fit(all_raw, all_labels)
        self._isotonic = ir
        logger.info(
            "Isotonic calibrator fitted on %d samples (unique X: %d)",
            len(all_raw),
            len(set(round(x, 6) for x in all_raw)),
        )

    def predict_exacta(
        self,
        features: np.ndarray,
        car_nos: list[int],
        market_pair_probs: dict[tuple[int, int], float] | None = None,
        odds_dict: dict[tuple[int, int], float] | None = None,
    ) -> list[tuple[int, int, float]]:
        """Predict exacta with conditional alpha blending.

        When alpha_map is populated and odds_dict is provided, each pair gets
        a different blend weight based on its market odds bucket.
        Falls back to uniform market_alpha when alpha_map is empty.
        """
        from app.services.modeling_v11 import ExactaModelV11

        # Get model PL predictions (uses isotonic-calibrated probs)
        preds = ExactaModelV11.predict_exacta(self, features, car_nos)

        if market_pair_probs is None:
            return preds

        model_map = {(f, s): p for f, s, p in preds}
        all_keys = set(model_map) | set(market_pair_probs)

        use_conditional = bool(self.alpha_map) and odds_dict is not None

        blended: dict[tuple[int, int], float] = {}
        for k in all_keys:
            m = model_map.get(k, 0.0)
            mk = market_pair_probs.get(k, 0.0)

            if use_conditional:
                pair_odds = odds_dict.get(k, 0.0)
                bucket = _odds_bucket(pair_odds) if pair_odds > 0 else "low"
                alpha = self.alpha_map.get(bucket, self.market_alpha)
            else:
                alpha = self.market_alpha

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
        odds_dicts: list[dict[tuple[int, int], float]] | None = None,
    ) -> None:
        """Fit conditional alpha map by odds bucket.

        First fits a global alpha (like v13), then fits per-bucket alphas.
        """
        if not val_races:
            logger.warning("No validation races for market alpha — skipping")
            return

        # Step 1: Fit global alpha (same as v13)
        super().fit_market_alpha(val_races)

        # Step 2: Fit per-bucket alpha if odds_dicts provided
        if odds_dicts is None or not odds_dicts:
            logger.info("No odds_dicts for conditional alpha — using global alpha=%.2f", self.market_alpha)
            return

        from app.services.modeling_v11 import ExactaModelV11

        # Collect per-pair data: (bucket, model_prob, market_prob, is_actual)
        pair_data: dict[str, list[tuple[float, float, bool]]] = {
            b: [] for b in _ODDS_BUCKETS
        }

        for (feats, car_nos, market_probs, actual), odds_dict in zip(val_races, odds_dicts):
            preds = ExactaModelV11.predict_exacta(self, feats, car_nos)
            model_map = {(f, s): p for f, s, p in preds}
            all_keys = set(model_map) | set(market_probs)

            for k in all_keys:
                m_p = model_map.get(k, 0.0)
                mk_p = market_probs.get(k, 0.0)
                pair_odds = odds_dict.get(k, 0.0)
                bucket = _odds_bucket(pair_odds) if pair_odds > 0 else "low"
                is_actual = (k == actual)
                pair_data[bucket].append((m_p, mk_p, is_actual))

        # For each bucket, sweep alpha to minimize LogLoss
        alpha_map: dict[str, float] = {}
        for bucket, data in pair_data.items():
            if len(data) < 50:
                alpha_map[bucket] = self.market_alpha
                logger.info("  Bucket %s: too few samples (%d), using global alpha", bucket, len(data))
                continue

            best_alpha = self.market_alpha
            best_ll = float("inf")
            eps = 1e-15

            for a_i in range(21):  # 0.00 to 1.00
                alpha = a_i * 0.05
                ll_sum = 0.0
                for m_p, mk_p, is_actual in data:
                    p = alpha * m_p + (1 - alpha) * mk_p
                    p = max(p, eps)
                    if is_actual:
                        ll_sum += -math.log(p)
                    else:
                        ll_sum += -math.log(max(1 - p, eps))

                ll = ll_sum / len(data)
                if ll < best_ll:
                    best_ll = ll
                    best_alpha = alpha

            alpha_map[bucket] = best_alpha
            logger.info("  Bucket %s: alpha=%.2f (%d pairs)", bucket, best_alpha, len(data))

        self.alpha_map = alpha_map
        logger.info(
            "Conditional alpha map: %s",
            {b: f"{a:.2f}" for b, a in alpha_map.items()},
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "feature_names": self.feature_names,
                "model_type": "v19_lgb",
                "calibrator_isotonic": self._isotonic,
                "market_alpha": self.market_alpha,
                "alpha_map": self.alpha_map,
            }, f)
        logger.info("Model v19 saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> ExactaModelV19:
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls(
            model=data["model"],
            calibrator=data.get("calibrator_isotonic"),
            market_alpha=data.get("market_alpha", 0.0),
            alpha_map=data.get("alpha_map", {}),
        )
        obj.feature_names = data.get("feature_names", V19_FEATURE_NAMES)
        logger.info("Model v19 loaded from %s", path)
        return obj

    @classmethod
    def is_v19_model(cls, path: str | Path) -> bool:
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            return data.get("model_type") == "v19_lgb"
        except Exception:
            return False
