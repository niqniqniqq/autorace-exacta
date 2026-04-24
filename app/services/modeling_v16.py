"""LightGBM model for exacta prediction (v16).

Extends v14 (15 features) with API-derived stats (4) = 19 features.
Inherits Platt calibration and market blend from v13/v14.

New features:
  - good_track_trial_avg: 良走路試走平均 (latest90List)
  - good_track_race_avg: 良走路レース平均 (latest90List)
  - career_win_rate: 通算勝率 (winList)
  - career_place_rate: 通算複勝率 (winList)
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np

from app.services.modeling_v14 import ExactaModelV14, V14_FEATURE_NAMES, extract_v14_features
from app.services.racer_stats import RacerStats

logger = logging.getLogger(__name__)

V16_FEATURE_NAMES = [
    *V14_FEATURE_NAMES,
    "good_track_trial_avg",
    "good_track_race_avg",
    "career_win_rate",
    "career_place_rate",
]


def extract_v16_features(
    entry,
    odds_stats: dict[int, dict[str, float]] | None = None,
    racer_stats: RacerStats | None = None,
) -> np.ndarray:
    """Extract 19 features from a RaceEntry + odds + racer history."""
    base = extract_v14_features(entry, odds_stats, racer_stats)

    stats = entry.stats_json or {}

    extra = np.array([
        float(stats.get("good_track_trial_avg") or 0.0),
        float(stats.get("good_track_race_avg") or 0.0),
        float(stats.get("career_win_rate") or 0.0),
        float(stats.get("career_place_rate") or 0.0),
    ], dtype=np.float64)

    return np.concatenate([base, extra])


class ExactaModelV16(ExactaModelV14):
    """LightGBM v16 model — v14 + API stats (19 features)."""

    def __init__(
        self,
        model=None,
        calibrator: tuple[float, float] | None = None,
        market_alpha: float = 0.0,
    ) -> None:
        super().__init__(model, calibrator, market_alpha)
        self.feature_names = V16_FEATURE_NAMES
        self.n_features = len(V16_FEATURE_NAMES)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "feature_names": self.feature_names,
                "model_type": "v16_lgb",
                "calibrator": self.calibrator,
                "market_alpha": self.market_alpha,
            }, f)
        logger.info("Model v16 saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> ExactaModelV16:
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls(
            model=data["model"],
            calibrator=data.get("calibrator"),
            market_alpha=data.get("market_alpha", 0.0),
        )
        obj.feature_names = data.get("feature_names", V16_FEATURE_NAMES)
        logger.info("Model v16 loaded from %s", path)
        return obj

    @classmethod
    def is_v16_model(cls, path: str | Path) -> bool:
        """Check if the pickle file is a v16 model."""
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            return data.get("model_type") == "v16_lgb"
        except Exception:
            return False
