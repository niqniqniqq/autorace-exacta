"""ML model for Plackett-Luce style exacta prediction."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from app.services.features import FEATURE_NAMES

logger = logging.getLogger(__name__)


class ExactaModel:
    """Simple model: learn per-runner strength, then Plackett-Luce for pairs."""

    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.model = LogisticRegression(max_iter=1000)
        self.n_features = len(FEATURE_NAMES)
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray, meta: list[dict]) -> None:
        """Train the model.

        X: (n_races, max_runners * n_features) — padded feature matrix
        y: (n_races,) — winner index among sorted car_nos
        """
        if len(X) == 0:
            logger.warning("No training data, skipping fit.")
            return

        runner_features: list[np.ndarray] = []
        runner_labels: list[int] = []

        for i, m in enumerate(meta):
            n_runners = m["n_runners"]
            race_feats = X[i].reshape(-1, self.n_features)[:n_runners]
            winner_idx = y[i]

            for j in range(n_runners):
                runner_features.append(race_feats[j])
                runner_labels.append(1 if j == winner_idx else 0)

        Xr = np.array(runner_features)
        yr = np.array(runner_labels)

        Xr_scaled = self.scaler.fit_transform(Xr)
        self.model.fit(Xr_scaled, yr)
        self.is_fitted = True

        acc = self.model.score(Xr_scaled, yr)
        logger.info("Model trained: %d samples, accuracy=%.3f", len(yr), acc)

    def predict_strengths(self, features: np.ndarray) -> np.ndarray:
        """Given (n_runners, n_features), return log-strength scores."""
        if not self.is_fitted:
            return np.zeros(features.shape[0])
        X_scaled = self.scaler.transform(features)
        proba = self.model.predict_proba(X_scaled)
        if proba.shape[1] == 2:
            return proba[:, 1]
        return proba[:, 0]

    def predict_exacta(
        self, features: np.ndarray, car_nos: list[int]
    ) -> list[tuple[int, int, float]]:
        """Predict exacta probabilities using Plackett-Luce.

        Returns sorted list of (first, second, prob) — highest prob first.
        """
        strengths = self.predict_strengths(features)
        n = len(car_nos)

        exp_u = np.exp(strengths - strengths.max())
        total = exp_u.sum()

        results: list[tuple[int, int, float]] = []
        for i in range(n):
            p1 = exp_u[i] / total
            remaining = total - exp_u[i]
            if remaining <= 0:
                continue
            for j in range(n):
                if i == j:
                    continue
                p2_given_1 = exp_u[j] / remaining
                prob = float(p1 * p2_given_1)
                results.append((car_nos[i], car_nos[j], prob))

        results.sort(key=lambda x: x[2], reverse=True)
        return results

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"scaler": self.scaler, "model": self.model, "is_fitted": self.is_fitted}, f)
        logger.info("Model saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> ExactaModel:
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls()
        obj.scaler = data["scaler"]
        obj.model = data["model"]
        obj.is_fitted = data["is_fitted"]
        logger.info("Model loaded from %s", path)
        return obj
