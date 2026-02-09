"""LightGBM model for exacta prediction (v12).

Extends v11 (8 raw features) with age = 9 features.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np

from app.services.modeling_v11 import ExactaModelV11

logger = logging.getLogger(__name__)

V12_FEATURE_NAMES = [
    # v11 base features (8)
    "handicap_m",
    "trial_time",
    "start_avg",
    "deviation",
    "quinella_rate",
    "trio_rate",
    "rank_class",
    "car_no",
    # additional (1)
    "age",
]


def extract_v12_features(entry) -> np.ndarray:
    """Extract 9 features from a RaceEntry object."""
    stats = entry.stats_json or {}
    rank_str = stats.get("rank", "")
    rank_class = 2 if rank_str.startswith("S") else (1 if rank_str.startswith("A") else 0)
    age = stats.get("age", 35) or 35

    return np.array([
        entry.handicap_m or 0,
        entry.trial_time or 0,
        entry.start_avg or 0.15,
        entry.deviation or 50,
        entry.quinella_rate or 0,
        entry.trio_rate or 0,
        rank_class,
        entry.car_no,
        age,
    ], dtype=np.float64)


class ExactaModelV12(ExactaModelV11):
    """LightGBM v12 model — v11 + age."""

    def __init__(self, model=None) -> None:
        super().__init__(model)
        self.feature_names = V12_FEATURE_NAMES
        self.n_features = len(V12_FEATURE_NAMES)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "feature_names": self.feature_names,
                "model_type": "v12_lgb",
            }, f)
        logger.info("Model v12 saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> ExactaModelV12:
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls(model=data["model"])
        obj.feature_names = data.get("feature_names", V12_FEATURE_NAMES)
        logger.info("Model v12 loaded from %s", path)
        return obj

    @classmethod
    def is_v12_model(cls, path: str | Path) -> bool:
        """Check if the pickle file is a v12 model."""
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            return data.get("model_type") == "v12_lgb"
        except Exception:
            return False
