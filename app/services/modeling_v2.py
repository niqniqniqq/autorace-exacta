"""Separate models for 1st and 2nd place prediction."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from app.services.features import FEATURE_NAMES

logger = logging.getLogger(__name__)


class ExactaModelV2:
    """Separate models for 1st and 2nd place, combined for exacta."""

    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.model_1st = LogisticRegression(max_iter=2000, C=0.5)
        self.model_2nd = LogisticRegression(max_iter=2000, C=0.5)
        self.n_features = len(FEATURE_NAMES)
        self.is_fitted = False
        self.feature_names = FEATURE_NAMES

    def fit(self, X: np.ndarray, y: np.ndarray, meta: list[dict]) -> None:
        """Train separate models for 1st and 2nd place.

        X: (n_races, max_runners * n_features)
        y: (n_races,) — winner index (unused, we use meta instead)
        meta: list of dicts with race_id, car_nos, n_runners, winner, second
        """
        if len(X) == 0:
            logger.warning("No training data, skipping fit.")
            return

        # Collect features and labels for 1st and 2nd place models
        features_1st: list[np.ndarray] = []
        labels_1st: list[int] = []
        features_2nd: list[np.ndarray] = []
        labels_2nd: list[int] = []

        for i, m in enumerate(meta):
            n_runners = m["n_runners"]
            race_feats = X[i].reshape(-1, self.n_features)[:n_runners]
            car_nos = m["car_nos"]
            winner = m["winner"]
            second = m.get("second")

            if winner is None:
                continue

            winner_idx = car_nos.index(winner) if winner in car_nos else None
            second_idx = car_nos.index(second) if second and second in car_nos else None

            # 1st place model: binary classification (won or not)
            for j in range(n_runners):
                features_1st.append(race_feats[j])
                labels_1st.append(1 if j == winner_idx else 0)

            # 2nd place model: binary classification (2nd or not), excluding winner
            if second_idx is not None:
                for j in range(n_runners):
                    if j == winner_idx:
                        continue  # Skip winner for 2nd place model
                    features_2nd.append(race_feats[j])
                    labels_2nd.append(1 if j == second_idx else 0)

        # Fit scaler on all features
        all_features = np.array(features_1st)
        self.scaler.fit(all_features)

        # Train 1st place model
        X1 = self.scaler.transform(np.array(features_1st))
        y1 = np.array(labels_1st)
        self.model_1st.fit(X1, y1)
        acc1 = self.model_1st.score(X1, y1)
        logger.info("1st place model: %d samples, accuracy=%.3f", len(y1), acc1)

        # Train 2nd place model
        X2 = self.scaler.transform(np.array(features_2nd))
        y2 = np.array(labels_2nd)
        self.model_2nd.fit(X2, y2)
        acc2 = self.model_2nd.score(X2, y2)
        logger.info("2nd place model: %d samples, accuracy=%.3f", len(y2), acc2)

        self.is_fitted = True

    def predict_probs(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict 1st and 2nd place probabilities.

        Returns (p_1st, p_2nd) arrays of shape (n_runners,).
        """
        if not self.is_fitted:
            n = features.shape[0]
            return np.ones(n) / n, np.ones(n) / n

        X_scaled = self.scaler.transform(features)

        # 1st place probabilities
        proba_1st = self.model_1st.predict_proba(X_scaled)
        p_1st = proba_1st[:, 1] if proba_1st.shape[1] == 2 else proba_1st[:, 0]

        # 2nd place probabilities (conditional on not winning)
        proba_2nd = self.model_2nd.predict_proba(X_scaled)
        p_2nd = proba_2nd[:, 1] if proba_2nd.shape[1] == 2 else proba_2nd[:, 0]

        return p_1st, p_2nd

    def predict_exacta(
        self, features: np.ndarray, car_nos: list[int]
    ) -> list[tuple[int, int, float]]:
        """Predict exacta probabilities using separate 1st/2nd models.

        P(i-j) = P(i wins) × P(j is 2nd | i wins)

        For P(j is 2nd | i wins), we use the 2nd place model scores
        normalized over remaining runners.
        """
        p_1st, p_2nd = self.predict_probs(features)
        n = len(car_nos)

        # Normalize 1st place probabilities
        p_1st_norm = p_1st / p_1st.sum()

        results: list[tuple[int, int, float]] = []
        for i in range(n):
            # P(i wins)
            p_i_wins = p_1st_norm[i]

            # For 2nd place, normalize 2nd place scores excluding i
            remaining_indices = [j for j in range(n) if j != i]
            remaining_p2 = p_2nd[remaining_indices]

            if remaining_p2.sum() > 0:
                remaining_p2_norm = remaining_p2 / remaining_p2.sum()
            else:
                remaining_p2_norm = np.ones(len(remaining_indices)) / len(remaining_indices)

            for k, j in enumerate(remaining_indices):
                # P(j is 2nd | i wins)
                p_j_second = remaining_p2_norm[k]
                # P(i-j exacta)
                prob = float(p_i_wins * p_j_second)
                results.append((car_nos[i], car_nos[j], prob))

        results.sort(key=lambda x: x[2], reverse=True)
        return results

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "scaler": self.scaler,
                "model_1st": self.model_1st,
                "model_2nd": self.model_2nd,
                "is_fitted": self.is_fitted,
                "n_features": self.n_features,
                "feature_names": self.feature_names,
            }, f)
        logger.info("Model saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> ExactaModelV2:
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls()
        obj.scaler = data["scaler"]
        obj.model_1st = data["model_1st"]
        obj.model_2nd = data["model_2nd"]
        obj.is_fitted = data["is_fitted"]
        obj.n_features = data.get("n_features", len(FEATURE_NAMES))
        obj.feature_names = data.get("feature_names", FEATURE_NAMES)
        logger.info("Model loaded from %s", path)
        return obj

    def get_feature_importance(self) -> dict[str, list[tuple[str, float]]]:
        """Return feature importance for both models."""
        if not self.is_fitted:
            return {"1st": [], "2nd": []}

        coefs_1st = self.model_1st.coef_[0]
        coefs_2nd = self.model_2nd.coef_[0]

        importance_1st = sorted(
            zip(self.feature_names, coefs_1st),
            key=lambda x: abs(x[1]),
            reverse=True,
        )
        importance_2nd = sorted(
            zip(self.feature_names, coefs_2nd),
            key=lambda x: abs(x[1]),
            reverse=True,
        )

        return {"1st": importance_1st, "2nd": importance_2nd}
