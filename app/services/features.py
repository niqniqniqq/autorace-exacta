"""Feature extraction for race entries."""

from __future__ import annotations

import logging

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Race, RaceDay, RaceEntry, RaceResult

logger = logging.getLogger(__name__)

# Base features (per-entry raw values)
BASE_FEATURE_NAMES = [
    "handicap_m",
    "trial_time",
    "deviation",
    "quinella_rate",
    "trio_rate",
]

# Relative features (computed within race context)
RELATIVE_FEATURE_NAMES = [
    "relative_handicap",      # handicap - min_handicap in race
    "relative_trial_time",    # trial_time - min_trial_time (lower is better)
    "car_position",           # normalized car_no position (1=inside, 0=outside)
    "handicap_advantage",     # (max_handicap - handicap) / range, higher = more advantage
]

FEATURE_NAMES = BASE_FEATURE_NAMES + RELATIVE_FEATURE_NAMES


def entry_to_base_features(entry: RaceEntry) -> np.ndarray:
    """Extract base feature vector from a single race entry."""
    vals = [
        entry.handicap_m or 0,
        entry.trial_time or 0.0,
        entry.deviation or 50.0,
        entry.quinella_rate or 0.0,
        entry.trio_rate or 0.0,
    ]
    return np.array(vals, dtype=np.float64)


def compute_relative_features(entries: list[RaceEntry]) -> np.ndarray:
    """Compute relative features for all entries in a race.

    Returns (n_entries, n_relative_features) array.
    """
    n = len(entries)
    if n == 0:
        return np.zeros((0, len(RELATIVE_FEATURE_NAMES)))

    # Extract raw values
    handicaps = np.array([e.handicap_m or 0 for e in entries], dtype=np.float64)
    trial_times = np.array([e.trial_time or 0.0 for e in entries], dtype=np.float64)
    car_nos = np.array([e.car_no for e in entries], dtype=np.float64)

    # Relative handicap (0 = minimum handicap in race)
    min_handicap = handicaps.min()
    relative_handicap = handicaps - min_handicap

    # Relative trial time (0 = fastest in race)
    # Handle case where trial_time is 0 (not yet available)
    valid_times = trial_times[trial_times > 0]
    if len(valid_times) > 0:
        min_trial = valid_times.min()
        relative_trial = np.where(trial_times > 0, trial_times - min_trial, 0.0)
    else:
        relative_trial = np.zeros(n)

    # Car position advantage (1番車 = 1.0, 8番車 = 0.0)
    # Inside position (lower car_no) is generally advantageous
    max_car = car_nos.max()
    min_car = car_nos.min()
    if max_car > min_car:
        car_position = 1.0 - (car_nos - min_car) / (max_car - min_car)
    else:
        car_position = np.ones(n) * 0.5

    # Handicap advantage (higher handicap = disadvantage, so invert)
    max_handicap = handicaps.max()
    handicap_range = max_handicap - min_handicap
    if handicap_range > 0:
        handicap_advantage = (max_handicap - handicaps) / handicap_range
    else:
        handicap_advantage = np.ones(n) * 0.5

    # Stack all relative features
    relative_feats = np.column_stack([
        relative_handicap,
        relative_trial,
        car_position,
        handicap_advantage,
    ])

    return relative_feats


def entries_to_features(entries: list[RaceEntry]) -> np.ndarray:
    """Extract full feature matrix for a list of entries.

    Returns (n_entries, n_features) array combining base and relative features.
    """
    if not entries:
        return np.zeros((0, len(FEATURE_NAMES)))

    # Base features
    base_feats = np.array([entry_to_base_features(e) for e in entries])

    # Relative features
    relative_feats = compute_relative_features(entries)

    # Combine
    return np.hstack([base_feats, relative_feats])


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

        features = entries_to_features(entries)
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
    features = entries_to_features(list(entries))
    return car_nos, features
