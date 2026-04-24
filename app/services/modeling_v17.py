"""LightGBM model for exacta prediction (v17).

Odds-free fundamental model: removes all odds-derived features
(implied_win_prob, log_implied_win_odds, odds_rank) to produce
a signal orthogonal to the market.

Uses v12 base (9) + v14 racer stats (3) + v16 API stats (4) = 16 features.
Inherits Platt calibration and market blend from v13.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np

from app.services.modeling_v12 import extract_v12_features
from app.services.modeling_v13 import ExactaModelV13
from app.services.racer_stats import RacerStats

logger = logging.getLogger(__name__)

V17_FEATURE_NAMES = [
    # v12 base (9)
    "handicap_m",
    "trial_time",
    "start_avg",
    "deviation",
    "quinella_rate",
    "trio_rate",
    "rank_class",
    "car_no",
    "age",
    # v14 racer stats (3) — 90-day history
    "win_rate",
    "place_rate",
    "race_count",
    # v16 API stats (4) — stats_json
    "good_track_trial_avg",
    "good_track_race_avg",
    "career_win_rate",
    "career_place_rate",
]


def extract_v17_features(
    entry,
    racer_stats: RacerStats | None = None,
) -> np.ndarray:
    """Extract 16 features from a RaceEntry + racer history.

    No odds_stats parameter — this model is odds-free.
    Bypasses v13/v14 extraction to avoid any odds dependency.
    """
    # v12 base: 9 features
    base = extract_v12_features(entry)

    # v14 racer stats: 3 features
    if racer_stats is not None:
        racer_feats = np.array([
            racer_stats.win_rate,
            racer_stats.place_rate,
            float(racer_stats.race_count),
        ], dtype=np.float64)
    else:
        racer_feats = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    # v16 API stats: 4 features
    stats = entry.stats_json or {}
    api_stats = np.array([
        float(stats.get("good_track_trial_avg") or 0.0),
        float(stats.get("good_track_race_avg") or 0.0),
        float(stats.get("career_win_rate") or 0.0),
        float(stats.get("career_place_rate") or 0.0),
    ], dtype=np.float64)

    return np.concatenate([base, racer_feats, api_stats])


class ExactaModelV17(ExactaModelV13):
    """LightGBM v17 model — odds-free fundamental (16 features).

    Inherits Platt calibration and market blend from v13.
    """

    def __init__(
        self,
        model=None,
        calibrator: tuple[float, float] | None = None,
        market_alpha: float = 0.0,
    ) -> None:
        super().__init__(model, calibrator, market_alpha)
        self.feature_names = V17_FEATURE_NAMES
        self.n_features = len(V17_FEATURE_NAMES)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "feature_names": self.feature_names,
                "model_type": "v17_lgb",
                "calibrator": self.calibrator,
                "market_alpha": self.market_alpha,
            }, f)
        logger.info("Model v17 saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> ExactaModelV17:
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls(
            model=data["model"],
            calibrator=data.get("calibrator"),
            market_alpha=data.get("market_alpha", 0.0),
        )
        obj.feature_names = data.get("feature_names", V17_FEATURE_NAMES)
        logger.info("Model v17 loaded from %s", path)
        return obj

    @classmethod
    def is_v17_model(cls, path: str | Path) -> bool:
        """Check if the pickle file is a v17 model."""
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            return data.get("model_type") == "v17_lgb"
        except Exception:
            return False
