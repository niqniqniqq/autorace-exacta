"""Feature extraction for race entries."""

from __future__ import annotations

import logging
from datetime import date

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
    "start_avg",  # 90日間スタート平均タイム
]

# Relative features (computed within race context)
RELATIVE_FEATURE_NAMES = [
    "relative_handicap",      # handicap - min_handicap in race
    "relative_trial_time",    # trial_time - min_trial_time (lower is better)
    "car_position",           # normalized car_no position (1=inside, 0=outside)
    "handicap_advantage",     # (max_handicap - handicap) / range, higher = more advantage
    "relative_start",         # start_avg - min_start_avg in race (lower is better)
]

# Interaction features (v6)
INTERACTION_FEATURE_NAMES = [
    "adjusted_time",          # trial_time + handicap * 0.001 (10m = 0.01s rule)
    "adjusted_time_rank",     # rank of adjusted_time in race (1=best, normalized)
    "trial_rank",             # rank of trial_time in race (1=fastest, normalized)
]

# Racer historical features
RACER_HISTORY_FEATURE_NAMES = [
    "hist_win_rate",          # Historical win rate (90 days)
    "hist_place_rate",        # Historical 1st/2nd rate
    "hist_show_rate",         # Historical 1st/2nd/3rd rate
    "hist_avg_finish",        # Historical average finish position
    "hist_race_count",        # Number of races in history (experience)
]

# Profile features (v10)
PROFILE_FEATURE_NAMES = [
    "rank_class",             # S=2, A=1, B=0
    "is_young",               # age < 35 → 1.0
    "is_home",                # racing at home track → 1.0
    "handicap_start_interaction",  # relative_handicap * (1 - relative_start_norm)
]

FEATURE_NAMES = BASE_FEATURE_NAMES + RELATIVE_FEATURE_NAMES + INTERACTION_FEATURE_NAMES + RACER_HISTORY_FEATURE_NAMES + PROFILE_FEATURE_NAMES


def entry_to_base_features(entry: RaceEntry) -> np.ndarray:
    """Extract base feature vector from a single race entry."""
    vals = [
        entry.handicap_m or 0,
        entry.trial_time or 0.0,
        entry.deviation or 50.0,
        entry.quinella_rate or 0.0,
        entry.trio_rate or 0.0,
        entry.start_avg or 0.15,  # Default to average start time
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
    start_avgs = np.array([e.start_avg or 0.15 for e in entries], dtype=np.float64)

    # Relative handicap (0 = minimum handicap in race)
    min_handicap = handicaps.min()
    relative_handicap = handicaps - min_handicap

    # Relative trial time (0 = fastest in race)
    valid_times = trial_times[trial_times > 0]
    if len(valid_times) > 0:
        min_trial = valid_times.min()
        relative_trial = np.where(trial_times > 0, trial_times - min_trial, 0.0)
    else:
        relative_trial = np.zeros(n)

    # Car position advantage (1番車 = 1.0, 8番車 = 0.0)
    max_car = car_nos.max()
    min_car = car_nos.min()
    if max_car > min_car:
        car_position = 1.0 - (car_nos - min_car) / (max_car - min_car)
    else:
        car_position = np.ones(n) * 0.5

    # Handicap advantage
    max_handicap = handicaps.max()
    handicap_range = max_handicap - min_handicap
    if handicap_range > 0:
        handicap_advantage = (max_handicap - handicaps) / handicap_range
    else:
        handicap_advantage = np.ones(n) * 0.5

    # Relative start (0 = fastest starter in race)
    valid_starts = start_avgs[start_avgs > 0]
    if len(valid_starts) > 0:
        min_start = valid_starts.min()
        relative_start = np.where(start_avgs > 0, start_avgs - min_start, 0.0)
    else:
        relative_start = np.zeros(n)

    relative_feats = np.column_stack([
        relative_handicap,
        relative_trial,
        car_position,
        handicap_advantage,
        relative_start,
    ])

    return relative_feats


def compute_interaction_features(entries: list[RaceEntry]) -> np.ndarray:
    """Compute interaction features for all entries in a race.

    Returns (n_entries, n_interaction_features) array.
    """
    n = len(entries)
    if n == 0:
        return np.zeros((0, len(INTERACTION_FEATURE_NAMES)))

    handicaps = np.array([e.handicap_m or 0 for e in entries], dtype=np.float64)
    trial_times = np.array([e.trial_time or 0.0 for e in entries], dtype=np.float64)

    # Adjusted time: trial + handicap * 0.001 (10m = 0.01s rule)
    adjusted_time = np.where(
        trial_times > 0,
        trial_times + handicaps * 0.001,
        0.0
    )

    # Adjusted time rank (1=best, normalized to 0-1)
    valid_adjusted = adjusted_time[adjusted_time > 0]
    if len(valid_adjusted) > 0:
        # Rank: lower adjusted time = better rank
        ranks = np.zeros(n)
        valid_indices = np.where(adjusted_time > 0)[0]
        sorted_indices = valid_indices[np.argsort(adjusted_time[valid_indices])]
        for rank, idx in enumerate(sorted_indices):
            ranks[idx] = rank + 1
        # Normalize: 1=best -> 1.0, last -> 0.0
        max_rank = len(valid_adjusted)
        adjusted_time_rank = np.where(
            adjusted_time > 0,
            1.0 - (ranks - 1) / max(max_rank - 1, 1),
            0.5
        )
    else:
        adjusted_time_rank = np.ones(n) * 0.5

    # Trial time rank (1=fastest, normalized to 0-1)
    valid_trials = trial_times[trial_times > 0]
    if len(valid_trials) > 0:
        ranks = np.zeros(n)
        valid_indices = np.where(trial_times > 0)[0]
        sorted_indices = valid_indices[np.argsort(trial_times[valid_indices])]
        for rank, idx in enumerate(sorted_indices):
            ranks[idx] = rank + 1
        max_rank = len(valid_trials)
        trial_rank = np.where(
            trial_times > 0,
            1.0 - (ranks - 1) / max(max_rank - 1, 1),
            0.5
        )
    else:
        trial_rank = np.ones(n) * 0.5

    interaction_feats = np.column_stack([
        adjusted_time,
        adjusted_time_rank,
        trial_rank,
    ])

    return interaction_feats


def compute_racer_history_features(
    db: Session,
    entries: list[RaceEntry],
    race_date: date,
    lookback_days: int = 90,
) -> np.ndarray:
    """Compute racer historical features for all entries.

    Returns (n_entries, n_history_features) array.
    """
    from app.services.racer_stats import get_racer_stats

    n = len(entries)
    if n == 0:
        return np.zeros((0, len(RACER_HISTORY_FEATURE_NAMES)))

    history_feats = []
    for entry in entries:
        stats = get_racer_stats(db, entry.racer_id, race_date, lookback_days)
        history_feats.append([
            stats.win_rate,
            stats.place_rate,
            stats.show_rate,
            stats.avg_finish,
            min(stats.race_count / 20.0, 1.0),  # Normalize: 20+ races = 1.0
        ])

    return np.array(history_feats, dtype=np.float64)


def compute_profile_features(
    entries: list[RaceEntry],
    track_code: str | None = None,
    relative_handicap: np.ndarray | None = None,
    relative_start: np.ndarray | None = None,
) -> np.ndarray:
    """Compute profile-based features for all entries.

    Returns (n_entries, n_profile_features) array.
    """
    n = len(entries)
    if n == 0:
        return np.zeros((0, len(PROFILE_FEATURE_NAMES)))

    # Track name mappings for home detection
    track_place_map = {
        "kawaguchi": "川口",
        "kawaguchi2": "川口",
        "hamamatsu": "浜松",
        "iizuka": "飯塚",
        "isesaki": "伊勢崎",
        "sanyou": "山陽",
    }
    home_place = track_place_map.get(track_code, "") if track_code else ""

    rank_class = []
    is_young = []
    is_home = []

    for e in entries:
        stats = e.stats_json or {}

        # Rank class: S=2, A=1, B=0
        rank_str = stats.get("rank", "")
        if rank_str.startswith("S"):
            rank_class.append(2.0)
        elif rank_str.startswith("A"):
            rank_class.append(1.0)
        else:
            rank_class.append(0.0)

        # Is young: age < 35
        age = stats.get("age", 40)
        is_young.append(1.0 if age and age < 35 else 0.0)

        # Is home track
        place_name = stats.get("place_name", "").replace("\u3000", "").strip()
        is_home.append(1.0 if home_place and home_place in place_name else 0.0)

    rank_class = np.array(rank_class, dtype=np.float64)
    is_young = np.array(is_young, dtype=np.float64)
    is_home = np.array(is_home, dtype=np.float64)

    # Handicap × Start interaction
    # Higher value = heavy handicap + fast start (advantageous combination)
    if relative_handicap is not None and relative_start is not None:
        # Normalize relative_start to 0-1 range (0 = fastest)
        max_rel_start = relative_start.max() if relative_start.max() > 0 else 1.0
        rel_start_norm = relative_start / max_rel_start
        # Interaction: high handicap (relative_handicap) + fast start (1 - rel_start_norm)
        handicap_start_interaction = relative_handicap * (1.0 - rel_start_norm)
    else:
        handicap_start_interaction = np.zeros(n)

    return np.column_stack([
        rank_class,
        is_young,
        is_home,
        handicap_start_interaction,
    ])


def entries_to_features(
    entries: list[RaceEntry],
    db: Session | None = None,
    race_date: date | None = None,
    track_code: str | None = None,
) -> np.ndarray:
    """Extract full feature matrix for a list of entries.

    Args:
        entries: List of race entries
        db: Database session (required for racer history features)
        race_date: Date of the race (required for racer history features)
        track_code: Track code for home track detection

    Returns (n_entries, n_features) array combining all features.
    """
    if not entries:
        return np.zeros((0, len(FEATURE_NAMES)))

    # Base features
    base_feats = np.array([entry_to_base_features(e) for e in entries])

    # Relative features
    relative_feats = compute_relative_features(entries)

    # Interaction features
    interaction_feats = compute_interaction_features(entries)

    # Racer history features
    if db is not None and race_date is not None:
        history_feats = compute_racer_history_features(db, entries, race_date)
    else:
        # Default values when history not available
        n = len(entries)
        history_feats = np.array([
            [0.125, 0.25, 0.375, 4.5, 0.0]  # Default: uniform distribution
            for _ in range(n)
        ])

    # Profile features (v10)
    # Extract relative_handicap and relative_start from relative_feats for interaction
    relative_handicap = relative_feats[:, 0]  # First column
    relative_start = relative_feats[:, 4]     # Fifth column
    profile_feats = compute_profile_features(
        entries, track_code, relative_handicap, relative_start
    )

    # Combine
    return np.hstack([base_feats, relative_feats, interaction_feats, history_feats, profile_feats])


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

        # Get race date and track for history lookup
        race_date = race.race_day.race_date
        track_code = race.race_day.track.track_code

        features = entries_to_features(entries, db=db, race_date=race_date, track_code=track_code)
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
    db: Session, race_id: int, *, active_car_nos: list[int] | None = None
) -> tuple[list[int], np.ndarray]:
    """Get car_nos and feature matrix for a single race.

    Args:
        db: Database session
        race_id: Race ID
        active_car_nos: If provided, filter to only these car numbers (for absent player handling)
    """
    race = db.execute(
        select(Race).where(Race.race_id == race_id)
    ).scalar_one()

    entries = (
        db.execute(
            select(RaceEntry).where(RaceEntry.race_id == race_id).order_by(RaceEntry.car_no)
        )
        .scalars()
        .all()
    )

    # Filter to active entries if specified (handles absent players)
    if active_car_nos is not None:
        active_set = set(active_car_nos)
        entries = [e for e in entries if e.car_no in active_set]

    car_nos = [e.car_no for e in entries]
    race_date = race.race_day.race_date
    track_code = race.race_day.track.track_code
    features = entries_to_features(list(entries), db=db, race_date=race_date, track_code=track_code)

    return car_nos, features
