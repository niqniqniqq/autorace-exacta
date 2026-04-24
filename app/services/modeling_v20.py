"""LightGBM model for exacta prediction (v20).

Track-specific models: each track_code gets its own ExactaModelV19 instance
trained only on that track's data, capturing track-specific dynamics
(e.g., sanyou #6 dominance, kawaguchi #8 dominance).

Falls back to a global model for tracks with insufficient training data.
Same 22 features as v18/v19.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np

from app.services.modeling_v19 import ExactaModelV19

logger = logging.getLogger(__name__)

# Minimum races per track to train a dedicated model (otherwise use fallback)
MIN_TRACK_RACES = 100

_FALLBACK_KEY = "_fallback"


class ExactaModelV20:
    """Multi-track container: one ExactaModelV19 per track_code."""

    def __init__(
        self,
        track_models: dict[str, ExactaModelV19] | None = None,
    ) -> None:
        self.track_models: dict[str, ExactaModelV19] = track_models or {}

    def get_model(self, track_code: str | None) -> ExactaModelV19 | None:
        """Return per-track model, falling back to global model."""
        if track_code and track_code in self.track_models:
            return self.track_models[track_code]
        return self.track_models.get(_FALLBACK_KEY)

    def predict_exacta(
        self,
        features: np.ndarray,
        car_nos: list[int],
        track_code: str | None = None,
        market_pair_probs: dict[tuple[int, int], float] | None = None,
        odds_dict: dict[tuple[int, int], float] | None = None,
    ) -> list[tuple[int, int, float]]:
        model = self.get_model(track_code)
        if model is None:
            return []
        return model.predict_exacta(
            features, car_nos,
            market_pair_probs=market_pair_probs,
            odds_dict=odds_dict,
        )

    def track_codes(self) -> list[str]:
        return [k for k in self.track_models if k != _FALLBACK_KEY]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model_type": "v20_lgb",
                "track_models": self.track_models,
            }, f)
        logger.info("Model v20 saved to %s (tracks: %s)", path, self.track_codes())

    @classmethod
    def load(cls, path: str | Path) -> ExactaModelV20:
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls(track_models=data["track_models"])
        logger.info(
            "Model v20 loaded from %s (tracks: %s)",
            path, obj.track_codes(),
        )
        return obj

    @classmethod
    def is_v20_model(cls, path: str | Path) -> bool:
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            return data.get("model_type") == "v20_lgb"
        except Exception:
            return False
