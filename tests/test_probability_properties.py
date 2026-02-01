"""Property tests: exacta probabilities must sum to ~1 and be in (0,1)."""

from __future__ import annotations

import numpy as np
import pytest

from app.services.modeling import ExactaModel


class TestPlackettLuceProperties:
    """Verify that Plackett-Luce exacta probabilities satisfy basic axioms."""

    def _make_model_with_strengths(self, n_runners: int) -> tuple[ExactaModel, np.ndarray, list[int]]:
        """Create a model and synthetic features for n_runners."""
        model = ExactaModel()
        # Simulate fitted model: use random strengths directly
        rng = np.random.RandomState(42)
        features = rng.randn(n_runners, 5)  # 5 features
        car_nos = list(range(1, n_runners + 1))
        return model, features, car_nos

    def test_probabilities_are_valid(self):
        """All probabilities must be in (0, 1)."""
        model, features, car_nos = self._make_model_with_strengths(8)
        # Use unfitted model -> uniform strengths -> still valid
        preds = model.predict_exacta(features, car_nos)

        assert len(preds) > 0
        for first, second, prob in preds:
            assert 0 < prob < 1, f"Invalid prob {prob} for {first}-{second}"
            assert first != second

    def test_probabilities_sum_to_one(self):
        """Sum of all exacta probabilities must be close to 1."""
        model, features, car_nos = self._make_model_with_strengths(8)
        preds = model.predict_exacta(features, car_nos)

        total = sum(prob for _, _, prob in preds)
        assert abs(total - 1.0) < 1e-6, f"Total probability = {total}, expected ~1.0"

    def test_correct_combination_count(self):
        """n*(n-1) exacta combinations for n runners."""
        for n in [2, 4, 6, 8]:
            model, features, car_nos = self._make_model_with_strengths(n)
            preds = model.predict_exacta(features, car_nos)
            expected = n * (n - 1)
            assert len(preds) == expected, (
                f"Expected {expected} combinations for {n} runners, got {len(preds)}"
            )

    def test_marginal_win_probability(self):
        """Sum of p(i->j) over all j should equal p1(i) = exp(u_i)/sum(exp(u_k))."""
        model, features, car_nos = self._make_model_with_strengths(6)
        preds = model.predict_exacta(features, car_nos)

        strengths = model.predict_strengths(features)
        exp_u = np.exp(strengths - strengths.max())
        total = exp_u.sum()

        for idx, car in enumerate(car_nos):
            marginal = sum(prob for f, s, prob in preds if f == car)
            expected = float(exp_u[idx] / total)
            assert abs(marginal - expected) < 1e-6, (
                f"Marginal win prob for car {car}: {marginal} vs expected {expected}"
            )

    def test_two_runners(self):
        """With 2 runners, should have exactly 2 combinations summing to 1."""
        model, features, car_nos = self._make_model_with_strengths(2)
        preds = model.predict_exacta(features, car_nos)

        assert len(preds) == 2
        total = sum(p for _, _, p in preds)
        assert abs(total - 1.0) < 1e-10
