"""LightGBM model for exacta prediction (v14).

Extends v13 (12 features) with racer history (3) = 15 features.
Inherits Platt calibration and market blend from v13.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np

from app.services.modeling_v13 import ExactaModelV13, V13_FEATURE_NAMES, extract_v13_features
from app.services.racer_stats import RacerStats

logger = logging.getLogger(__name__)

V14_FEATURE_NAMES = [
    *V13_FEATURE_NAMES,
    "win_rate",
    "place_rate",
    "race_count",
]


def extract_v14_features(
    entry,
    odds_stats: dict[int, dict[str, float]] | None = None,
    racer_stats: RacerStats | None = None,
) -> np.ndarray:
    """Extract 15 features from a RaceEntry + odds + racer history."""
    base = extract_v13_features(entry, odds_stats)

    if racer_stats is not None:
        extra = [racer_stats.win_rate, racer_stats.place_rate, float(racer_stats.race_count)]
    else:
        extra = [0.0, 0.0, 0.0]

    return np.concatenate([base, np.array(extra, dtype=np.float64)])


class ExactaModelV14(ExactaModelV13):
    """LightGBM v14 model — v13 + racer history (15 features)."""

    def __init__(
        self,
        model=None,
        calibrator: tuple[float, float] | None = None,
        market_alpha: float = 0.0,
    ) -> None:
        super().__init__(model, calibrator, market_alpha)
        self.feature_names = V14_FEATURE_NAMES
        self.n_features = len(V14_FEATURE_NAMES)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "feature_names": self.feature_names,
                "model_type": "v14_lgb",
                "calibrator": self.calibrator,
                "market_alpha": self.market_alpha,
            }, f)
        logger.info("Model v14 saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> ExactaModelV14:
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls(
            model=data["model"],
            calibrator=data.get("calibrator"),
            market_alpha=data.get("market_alpha", 0.0),
        )
        obj.feature_names = data.get("feature_names", V14_FEATURE_NAMES)
        logger.info("Model v14 loaded from %s", path)
        return obj

    @classmethod
    def is_v14_model(cls, path: str | Path) -> bool:
        """Check if the pickle file is a v14 model."""
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            return data.get("model_type") == "v14_lgb"
        except Exception:
            return False
