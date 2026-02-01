"""Feature extraction for race entries."""

from __future__ import annotations

import logging

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Race, RaceDay, RaceEntry, RaceResult

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "handicap_m",
    "trial_time",
    "deviation",
    "quinella_rate",
    "trio_rate",
]


def entry_to_features(entry: RaceEntry) -> np.ndarray:
    """Extract feature vector from a single race entry."""
    vals = [
        entry.handicap_m or 0,
        entry.trial_time or 0.0,
        entry.deviation or 50.0,
        entry.quinella_rate or 0.0,
        entry.trio_rate or 0.0,
    ]
    return np.array(vals, dtype=np.float64)


def build_training_data(
    db: Session, date_from: str, date_to: str
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Build training dataset from historical races.

    Returns:
        X: feature matrix (n_samples, n_runners * n_features)
        y: label vector — index of winner among sorted car_nos
        meta: list of metadata dicts with race_id, car_nos, etc.
    """
    from datetime import date as date_type

    d_from = date_type.fromisoformat(date_from)
    d_to = date_type.fromisoformat(date_to)

    races = (
        db.execute(
            select(Race)
            .join(RaceDay)
            .where(RaceDay.race_date >= d_from, RaceDay.race_date <= d_to)
            .order_by(RaceDay.race_date, Race.race_no)
        )
        .scalars()
        .all()
    )

    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    meta_list: list[dict] = []

    for race in races:
        entries = sorted(race.entries, key=lambda e: e.car_no)
        if len(entries) < 2:
            continue

        result = race.result
        if result is None or not result.is_valid or result.winner_car_no is None:
            continue

        car_nos = [e.car_no for e in entries]
        if result.winner_car_no not in car_nos:
            continue

        features = np.array([entry_to_features(e) for e in entries])
        winner_idx = car_nos.index(result.winner_car_no)

        X_list.append(features.flatten())
        y_list.append(winner_idx)
        meta_list.append(
            {
                "race_id": race.race_id,
                "car_nos": car_nos,
                "n_runners": len(entries),
                "winner": result.winner_car_no,
                "second": result.second_car_no,
            }
        )

    if not X_list:
        return np.array([]), np.array([]), []

    max_len = max(x.shape[0] for x in X_list)
    X_padded = np.zeros((len(X_list), max_len))
    for i, x in enumerate(X_list):
        X_padded[i, : x.shape[0]] = x

    return X_padded, np.array(y_list), meta_list


def get_race_features(
    db: Session, race_id: int
) -> tuple[list[int], np.ndarray]:
    """Get car_nos and feature matrix for a single race."""
    entries = (
        db.execute(
            select(RaceEntry).where(RaceEntry.race_id == race_id).order_by(RaceEntry.car_no)
        )
        .scalars()
        .all()
    )
    car_nos = [e.car_no for e in entries]
    features = np.array([entry_to_features(e) for e in entries])
    return car_nos, features
