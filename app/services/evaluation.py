"""Model evaluation metrics — LogLoss and Brier score for exacta predictions."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def compute_logloss(
    predictions: list[list[tuple[int, int, float]]],
    actuals: list[tuple[int, int]],
    eps: float = 1e-15,
) -> float:
    """Compute log loss across races.

    predictions: list of [(first, second, prob), ...] per race
    actuals: list of (actual_first, actual_second) per race
    """
    losses: list[float] = []
    for preds, (act_first, act_second) in zip(predictions, actuals):
        prob_map = {(f, s): p for f, s, p in preds}
        prob = prob_map.get((act_first, act_second), eps)
        prob = max(prob, eps)
        losses.append(-np.log(prob))

    if not losses:
        return float("inf")
    return float(np.mean(losses))


def compute_brier(
    predictions: list[list[tuple[int, int, float]]],
    actuals: list[tuple[int, int]],
) -> float:
    """Compute Brier score across races."""
    scores: list[float] = []
    for preds, (act_first, act_second) in zip(predictions, actuals):
        for first, second, prob in preds:
            outcome = 1.0 if (first == act_first and second == act_second) else 0.0
            scores.append((prob - outcome) ** 2)

    if not scores:
        return float("inf")
    return float(np.mean(scores))
