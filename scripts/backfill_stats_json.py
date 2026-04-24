#!/usr/bin/env python3
"""Backfill stats_json with latest90List / winList data from disk snapshots.

Reads program snapshots from data/snapshots/program/{track}/{date}/*.json.gz
and merges new fields into RaceEntry.stats_json (no API calls needed).
"""

from __future__ import annotations

import gzip
import json
import logging
from pathlib import Path

from sqlalchemy import select

from app.db.models import Race, Racer, RaceDay, RaceEntry, Track
from app.db.session import get_db
from app.scraping.parsers.program_parser import _safe_float, _safe_int

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SNAPSHOT_ROOT = Path("data/snapshots/program")


def _extract_stats_from_snapshot(body: dict) -> dict[str, dict]:
    """Extract per-player stats from a program snapshot body.

    Returns {player_code: {field: value, ...}} for new stats_json fields.
    """
    latest90 = body.get("latest90List") or {}
    win_list = body.get("winList") or {}
    player_list = body.get("playerList") or []

    result: dict[str, dict] = {}
    for p in player_list:
        player_code = str(p.get("playerCode", ""))
        if not player_code:
            continue

        l90 = latest90.get(player_code, {})
        win = win_list.get(player_code, {})

        result[player_code] = {
            "good_track_trial_avg": _safe_float(l90.get("goodTrackTraialAve")),
            "good_track_race_avg": _safe_float(l90.get("goodTrackRaceAve")),
            "run_count_90d": _safe_int(l90.get("runCount")),
            "win_count_90d": _safe_int(l90.get("winCount")),
            "career_win_rate": _safe_float(win.get("rate1")),
            "career_place_rate": _safe_float(win.get("rate2")),
            "career_trio_rate": _safe_float(win.get("rate3")),
            "career_total_wins": _safe_int(win.get("totalWinCount")),
        }

    return result


def backfill_stats_json() -> None:
    """Walk disk snapshots and merge new stats_json fields into DB."""
    if not SNAPSHOT_ROOT.exists():
        logger.error("Snapshot root not found: %s", SNAPSHOT_ROOT)
        return

    updated_total = 0
    skipped_total = 0
    files_processed = 0

    with get_db() as db:
        # Build mapping: (track_code, race_date, race_no) -> race_id  (lazily)
        # and (race_id, racer_code) -> entry  (per snapshot)

        for track_dir in sorted(SNAPSHOT_ROOT.iterdir()):
            if not track_dir.is_dir():
                continue
            track_code = track_dir.name

            # Resolve track in DB
            track = db.scalar(
                select(Track).where(Track.track_code == track_code)
            )
            if track is None:
                logger.debug("Track %s not in DB, skipping", track_code)
                continue

            for date_dir in sorted(track_dir.iterdir()):
                if not date_dir.is_dir():
                    continue

                try:
                    race_date_str = date_dir.name
                    from datetime import date
                    race_date = date.fromisoformat(race_date_str)
                except ValueError:
                    continue

                # Find race_day
                race_day = db.scalar(
                    select(RaceDay).where(
                        RaceDay.track_id == track.track_id,
                        RaceDay.race_date == race_date,
                    )
                )
                if race_day is None:
                    continue

                # Process each snapshot file for this date
                snapshot_files = sorted(date_dir.glob("*.json.gz"))
                date_updated = 0

                for gz_path in snapshot_files:
                    try:
                        with gzip.open(gz_path, "rt") as f:
                            data = json.load(f)
                    except Exception:
                        logger.debug("Failed to read %s", gz_path)
                        continue

                    body = data.get("body", {})
                    if not body.get("playerList"):
                        continue

                    files_processed += 1
                    stats_map = _extract_stats_from_snapshot(body)
                    if not stats_map:
                        continue

                    # Determine race_no from the snapshot
                    # Each snapshot is one race — match via playerList car_nos
                    player_list = body.get("playerList", [])
                    player_codes = {
                        str(p["playerCode"])
                        for p in player_list
                        if p.get("playerCode")
                    }
                    car_nos_in_snap = {int(p.get("carNo", 0)) for p in player_list}

                    # Find matching race by entries
                    races = db.scalars(
                        select(Race).where(Race.race_day_id == race_day.race_day_id)
                    ).all()

                    for race in races:
                        entries = race.entries
                        if not entries:
                            continue

                        entry_codes = {
                            e.racer.racer_code for e in entries
                            if e.racer and e.racer.racer_code
                        }
                        # Match: at least half the player codes overlap
                        overlap = player_codes & entry_codes
                        if len(overlap) < max(len(player_codes) // 2, 1):
                            continue

                        # Matched race — update entries
                        for entry in entries:
                            racer_code = entry.racer.racer_code if entry.racer else None
                            if not racer_code:
                                continue
                            new_stats = stats_map.get(racer_code)
                            if not new_stats:
                                continue

                            # Merge into existing stats_json
                            existing = entry.stats_json or {}
                            needs_update = False
                            for key, val in new_stats.items():
                                if key not in existing and val is not None:
                                    needs_update = True
                                    break

                            if not needs_update:
                                skipped_total += 1
                                continue

                            merged = {**existing}
                            for key, val in new_stats.items():
                                if key not in merged and val is not None:
                                    merged[key] = val

                            entry.stats_json = merged
                            date_updated += 1

                        break  # Matched this snapshot to a race

                if date_updated > 0:
                    db.commit()
                    updated_total += date_updated
                    logger.info(
                        "  %s %s: updated %d entries",
                        track_code, race_date_str, date_updated,
                    )

    logger.info(
        "Done! Processed %d files, updated %d entries, skipped %d (already filled)",
        files_processed, updated_total, skipped_total,
    )


if __name__ == "__main__":
    backfill_stats_json()
