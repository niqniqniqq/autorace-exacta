"""LightGBM model for exacta prediction (v11).

Uses 8 raw features without custom transformations.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# v11 uses raw features only - no relative/adjusted features
V11_FEATURE_NAMES = [
    "handicap_m",
    "trial_time",
    "start_avg",
    "deviation",
    "quinella_rate",
    "trio_rate",
    "rank_class",
    "car_no",
]


def extract_v11_features(entry) -> np.ndarray:
    """Extract 8 raw features from a RaceEntry object."""
    stats = entry.stats_json or {}
    rank_str = stats.get("rank", "")
    rank_class = 2 if rank_str.startswith("S") else (1 if rank_str.startswith("A") else 0)

    return np.array([
        entry.handicap_m or 0,
        entry.trial_time or 0,
        entry.start_avg or 0.15,
        entry.deviation or 50,
        entry.quinella_rate or 0,
        entry.trio_rate or 0,
        rank_class,
        entry.car_no,
    ], dtype=np.float64)


class ExactaModelV11:
    """LightGBM-based model for exacta prediction."""

    def __init__(self, model=None) -> None:
        self.model = model
        self.feature_names = V11_FEATURE_NAMES
        self.n_features = len(V11_FEATURE_NAMES)
        self.is_fitted = model is not None

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Predict win probability for each runner."""
        if not self.is_fitted:
            n = features.shape[0]
            return np.ones(n) / n
        return self.model.predict_proba(features)[:, 1]

    def predict_exacta(
        self, features: np.ndarray, car_nos: list[int]
    ) -> list[tuple[int, int, float]]:
        """Predict exacta probabilities using Plackett-Luce.

        Returns sorted list of (first, second, prob) - highest prob first.
        """
        probs = self.predict_proba(features)
        n = len(car_nos)

        # Normalize to get P(i wins)
        prob_sum = probs.sum()
        if prob_sum <= 0:
            prob_sum = 1.0

        results: list[tuple[int, int, float]] = []
        for i in range(n):
            p_i = probs[i] / prob_sum
            # P(j is 2nd | i wins)
            remaining = probs.copy()
            remaining[i] = 0
            remaining_sum = remaining.sum()
            if remaining_sum <= 0:
                continue

            for j in range(n):
                if i == j:
                    continue
                p_j_given_i = probs[j] / remaining_sum
                prob = float(p_i * p_j_given_i)
                results.append((car_nos[i], car_nos[j], prob))

        results.sort(key=lambda x: x[2], reverse=True)
        return results

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "feature_names": self.feature_names,
                "model_type": "v11_lgb",
            }, f)
        logger.info("Model saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> ExactaModelV11:
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls(model=data["model"])
        obj.feature_names = data.get("feature_names", V11_FEATURE_NAMES)
        logger.info("Model v11 loaded from %s", path)
        return obj

    @classmethod
    def is_v11_model(cls, path: str | Path) -> bool:
        """Check if the pickle file is a v11 model."""
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            return data.get("model_type") == "v11_lgb"
        except Exception:
            return False

    def get_feature_importance(self) -> list[tuple[str, int]]:
        """Return feature names and their importance scores."""
        if not self.is_fitted:
            return []
        importances = self.model.feature_importances_
        return sorted(
            zip(self.feature_names, importances),
            key=lambda x: x[1],
            reverse=True,
        )
