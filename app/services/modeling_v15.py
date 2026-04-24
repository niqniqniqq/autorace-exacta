"""Direct pairwise model for exacta prediction (v15).

Instead of predicting individual win probabilities and deriving pair probs
via Plackett-Luce, this model directly scores every (i->j) pair.

Pair features (37 dims):
  first_*  : runner i's v14 features (15)
  second_* : runner j's v14 features (15)
  diff_*   : i - j for 7 key features
"""

from __future__ import annotations

import logging
import math
import pickle
from pathlib import Path

import numpy as np

from app.services.modeling_v14 import V14_FEATURE_NAMES

logger = logging.getLogger(__name__)

# Indices into V14 feature vector for diff features
# V14: handicap_m(0), trial_time(1), start_avg(2), deviation(3),
#      quinella_rate(4), trio_rate(5), rank_class(6), car_no(7),
#      age(8), implied_win_prob(9), log_implied_win_odds(10), odds_rank(11),
#      win_rate(12), place_rate(13), race_count(14)
V15_DIFF_INDICES = [1, 0, 3, 2, 9, 12, 8]
# trial_time, handicap_m, deviation, start_avg, implied_win_prob, win_rate, age

V15_DIFF_NAMES = [V14_FEATURE_NAMES[i] for i in V15_DIFF_INDICES]

V15_FEATURE_NAMES = (
    [f"first_{n}" for n in V14_FEATURE_NAMES]
    + [f"second_{n}" for n in V14_FEATURE_NAMES]
    + [f"diff_{n}" for n in V15_DIFF_NAMES]
)

N_RUNNER_FEATURES = len(V14_FEATURE_NAMES)  # 15
N_PAIR_FEATURES = len(V15_FEATURE_NAMES)  # 37


def build_pair_features(
    runner_features: np.ndarray,
    car_nos: list[int],
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Build pair-level features from runner features.

    Args:
        runner_features: (n, 15) array of v14 features, one row per runner.
        car_nos: list of car numbers, aligned with runner_features rows.

    Returns:
        pair_features: (n*(n-1), 37) array
        pairs: list of (first_car_no, second_car_no) in same order
    """
    n = len(car_nos)
    n_pairs = n * (n - 1)

    pair_feats = np.empty((n_pairs, N_PAIR_FEATURES), dtype=np.float64)
    pairs: list[tuple[int, int]] = []

    idx = 0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            fi = runner_features[i]
            fj = runner_features[j]
            diff = fi[V15_DIFF_INDICES] - fj[V15_DIFF_INDICES]
            pair_feats[idx] = np.concatenate([fi, fj, diff])
            pairs.append((car_nos[i], car_nos[j]))
            idx += 1

    return pair_feats, pairs


class ExactaModelV15:
    """Direct pairwise exacta model — no Plackett-Luce."""

    def __init__(
        self,
        model=None,
        market_alpha: float = 0.0,
    ) -> None:
        self.model = model
        self.market_alpha = market_alpha
        self.feature_names = V15_FEATURE_NAMES
        self.n_features = N_PAIR_FEATURES
        self.is_fitted = model is not None

    def predict_exacta(
        self,
        features: np.ndarray,
        car_nos: list[int],
        market_pair_probs: dict[tuple[int, int], float] | None = None,
    ) -> list[tuple[int, int, float]]:
        """Predict exacta probabilities from runner features.

        Args:
            features: (n, 15) runner-level v14 features.
            car_nos: list of car numbers.
            market_pair_probs: optional market probabilities for blending.

        Returns:
            Sorted list of (first, second, prob), highest prob first.
        """
        pair_feats, pairs = build_pair_features(features, car_nos)

        if not self.is_fitted:
            # Uniform
            n_pairs = len(pairs)
            prob = 1.0 / max(n_pairs, 1)
            return [(f, s, prob) for f, s in pairs]

        # Raw scores from LightGBM
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names")
            raw_scores = self.model.predict_proba(pair_feats)[:, 1]

        # Softmax normalization
        model_probs = _softmax(raw_scores)

        model_map: dict[tuple[int, int], float] = {}
        for k, pair in enumerate(pairs):
            model_map[pair] = float(model_probs[k])

        # Market blend
        if market_pair_probs is not None:
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
            prob_map = blended
        else:
            prob_map = model_map

        result = [(f, s, p) for (f, s), p in prob_map.items()]
        result.sort(key=lambda x: x[2], reverse=True)
        return result

    def fit_market_alpha(
        self,
        val_races: list[
            tuple[np.ndarray, list[int], dict[tuple[int, int], float], tuple[int, int]]
        ],
    ) -> None:
        """Fit optimal market blend alpha on validation races.

        Sweeps alpha 0.00~1.00 and picks the value minimizing pair LogLoss.
        """
        if not val_races:
            logger.warning("No validation races for market alpha — skipping")
            return

        # Pre-compute model-only predictions for each val race
        race_model_maps: list[dict[tuple[int, int], float]] = []
        for feats, car_nos, _market_probs, _actual in val_races:
            preds = self.predict_exacta(feats, car_nos)
            race_model_maps.append({(f, s): p for f, s, p in preds})

        alphas = [i * 0.05 for i in range(21)]  # 0.00 ~ 1.00
        best_alpha = 0.0
        best_ll = float("inf")

        for alpha in alphas:
            ll_sum = 0.0
            n = 0
            for idx, (_feats, _car_nos, market_probs, actual) in enumerate(val_races):
                model_map = race_model_maps[idx]
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
            pickle.dump(
                {
                    "model": self.model,
                    "feature_names": self.feature_names,
                    "model_type": "v15_pair",
                    "market_alpha": self.market_alpha,
                },
                f,
            )
        logger.info("Model v15 saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> ExactaModelV15:
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls(
            model=data["model"],
            market_alpha=data.get("market_alpha", 0.0),
        )
        obj.feature_names = data.get("feature_names", V15_FEATURE_NAMES)
        logger.info("Model v15 loaded from %s", path)
        return obj

    @classmethod
    def is_v15_model(cls, path: str | Path) -> bool:
        """Check if the pickle file is a v15 model."""
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            return data.get("model_type") == "v15_pair"
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


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    x_shifted = x - np.max(x)
    exp_x = np.exp(x_shifted)
    return exp_x / exp_x.sum()
