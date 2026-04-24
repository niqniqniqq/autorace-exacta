"""LightGBM model for exacta prediction (v18).

Adds race-relative features and nonlinear interactions to the v17 odds-free base.
22 features = v17 base (16) + race context (4) + interactions (2).

Race-relative features capture how a runner compares to the field:
- trial_time_rel: trial_time minus field mean (negative = faster than average)
- deviation_rel: deviation minus field mean (positive = stronger than average)
- field_strength: mean deviation of all runners (high = tough field)
- trial_time_best_diff: gap to fastest trial time (0 = fastest)

Interaction features capture nonlinear relationships:
- form_delta: trial_time minus 90-day good-track avg (negative = improving form)
- handicap_deviation_ratio: handicap_m / (deviation + 1) (low = strong for handicap)

Inherits Platt calibration and market blend from v13.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.services.modeling_v13 import ExactaModelV13
from app.services.modeling_v17 import V17_FEATURE_NAMES, extract_v17_features
from app.services.racer_stats import RacerStats

logger = logging.getLogger(__name__)

V18_FEATURE_NAMES = [
    *V17_FEATURE_NAMES,
    # Race-relative (4)
    "trial_time_rel",
    "deviation_rel",
    "field_strength",
    "trial_time_best_diff",
    # Interactions (2)
    "form_delta",
    "handicap_deviation_ratio",
]


@dataclass
class RaceContext:
    """Race-level statistics computed once per race from all entries."""

    trial_time_avg: float
    trial_time_min: float
    deviation_avg: float


def compute_race_context(entries: list) -> RaceContext:
    """Compute race-level statistics from all entries.

    Args:
        entries: List of RaceEntry objects for this race.

    Returns:
        RaceContext with field-level aggregates.
    """
    trial_times = [
        e.trial_time for e in entries
        if e.trial_time is not None and e.trial_time > 0
    ]
    deviations = [
        e.deviation for e in entries
        if e.deviation is not None and e.deviation > 0
    ]
    return RaceContext(
        trial_time_avg=float(np.mean(trial_times)) if trial_times else 0.0,
        trial_time_min=float(np.min(trial_times)) if trial_times else 0.0,
        deviation_avg=float(np.mean(deviations)) if deviations else 50.0,
    )


def extract_v18_features(
    entry,
    racer_stats: RacerStats | None = None,
    race_context: RaceContext | None = None,
) -> np.ndarray:
    """Extract 22 features: v17 base (16) + race context (4) + interactions (2).

    Args:
        entry: RaceEntry ORM object.
        racer_stats: 90-day racer history (optional).
        race_context: Race-level stats from compute_race_context (optional).

    Returns:
        numpy array of shape (22,).
    """
    # v17 base: 16 features
    base = extract_v17_features(entry, racer_stats)

    # Race context defaults
    if race_context is None:
        race_context = RaceContext(
            trial_time_avg=0.0, trial_time_min=0.0, deviation_avg=50.0,
        )

    trial_time = float(entry.trial_time or 0.0)
    deviation = float(entry.deviation or 0.0)

    # Race-relative features (4)
    if race_context.trial_time_avg > 0 and trial_time > 0:
        trial_time_rel = trial_time - race_context.trial_time_avg
        trial_time_best_diff = trial_time - race_context.trial_time_min
    else:
        trial_time_rel = 0.0
        trial_time_best_diff = 0.0

    if race_context.deviation_avg > 0 and deviation > 0:
        deviation_rel = deviation - race_context.deviation_avg
    else:
        deviation_rel = 0.0

    field_strength = race_context.deviation_avg

    context_feats = np.array([
        trial_time_rel,
        deviation_rel,
        field_strength,
        trial_time_best_diff,
    ], dtype=np.float64)

    # Interaction features (2)
    stats = entry.stats_json or {}
    good_track_trial_avg = float(stats.get("good_track_trial_avg") or 0.0)
    handicap_m = float(entry.handicap_m or 0.0)

    if good_track_trial_avg > 0 and trial_time > 0:
        form_delta = trial_time - good_track_trial_avg
    else:
        form_delta = 0.0

    handicap_deviation_ratio = handicap_m / (deviation + 1.0)

    interaction_feats = np.array([
        form_delta,
        handicap_deviation_ratio,
    ], dtype=np.float64)

    return np.concatenate([base, context_feats, interaction_feats])


class ExactaModelV18(ExactaModelV13):
    """LightGBM v18 model — race-relative + interactions (22 features).

    Inherits Platt calibration and market blend from v13.
    """

    def __init__(
        self,
        model=None,
        calibrator: tuple[float, float] | None = None,
        market_alpha: float = 0.0,
    ) -> None:
        super().__init__(model, calibrator, market_alpha)
        self.feature_names = V18_FEATURE_NAMES
        self.n_features = len(V18_FEATURE_NAMES)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "feature_names": self.feature_names,
                "model_type": "v18_lgb",
                "calibrator": self.calibrator,
                "market_alpha": self.market_alpha,
            }, f)
        logger.info("Model v18 saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> ExactaModelV18:
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls(
            model=data["model"],
            calibrator=data.get("calibrator"),
            market_alpha=data.get("market_alpha", 0.0),
        )
        obj.feature_names = data.get("feature_names", V18_FEATURE_NAMES)
        logger.info("Model v18 loaded from %s", path)
        return obj

    @classmethod
    def is_v18_model(cls, path: str | Path) -> bool:
        """Check if the pickle file is a v18 model."""
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            return data.get("model_type") == "v18_lgb"
        except Exception:
            return False
