"""Batch fetch historical programs and results for training data."""

from __future__ import annotations

import logging
import sys
import time
from datetime import date, timedelta

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(name)s — %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

TRACKS = ["sanyou", "iizuka", "kawaguchi", "isesaki", "hamamatsu", "kawaguchi2"]

TRACK_NAMES = {
    "kawaguchi": "川口",
    "kawaguchi2": "川口ナイト",
    "isesaki": "伊勢崎",
    "hamamatsu": "浜松",
    "iizuka": "飯塚",
    "sanyou": "山陽",
}


def fetch_day(track: str, race_date: date) -> tuple[int, int]:
    """Fetch program + results for one track/date. Returns (programs, results)."""
    from datetime import datetime, timezone

    from app.db.session import get_db
    from app.scraping.http import AutoraceClient
    from app.scraping.parsers.program_parser import parse_program
    from app.scraping.parsers.results_parser import parse_exacta_payout, parse_result
    from app.scraping.sources.autorace_program import fetch_all_programs
    from app.scraping.sources.autorace_results import fetch_all_results
    from app.services.storage import save_json_snapshot
    from app.services.upsert import (
        upsert_entry,
        upsert_payout_exacta,
        upsert_race,
        upsert_race_day,
        upsert_result,
        upsert_snapshot,
        upsert_track,
    )

    dt_str = race_date.isoformat()
    n_programs = 0
    n_results = 0

    try:
        client = AutoraceClient(init_track_code=track)
    except Exception as e:
        logger.debug("CSRF init failed for %s: %s", track, e)
        return 0, 0

    # --- Fetch programs ---
    try:
        programs = fetch_all_programs(client, track, race_date)
    except Exception as e:
        logger.debug("Program fetch failed %s %s: %s", track, dt_str, e)
        programs = []

    if not programs:
        return 0, 0

    with get_db() as db:
        t = upsert_track(db, track, TRACK_NAMES.get(track, track))
        rd = upsert_race_day(db, t.track_id, race_date)

        for race_no, prog_data in programs:
            storage_uri, chash = save_json_snapshot("program", track, dt_str, prog_data)
            snap = upsert_snapshot(
                db,
                source="program",
                url=f"autorace.jp/program/{track}/{dt_str}/R{race_no}",
                fetched_at=datetime.now(timezone.utc),
                http_status=200,
                content_hash=chash,
                content_type="application/json",
                storage_uri=storage_uri,
            )
            race = upsert_race(db, rd.race_day_id, race_no)
            entries = parse_program(prog_data, race_no)
            for entry in entries:
                upsert_entry(db, race.race_id, entry, snap.snapshot_id)
            n_programs += 1

    # --- Fetch results ---
    race_nos = [rno for rno, _ in programs]
    try:
        results_data = fetch_all_results(client, track, race_date, race_nos=race_nos)
    except Exception as e:
        logger.debug("Results fetch failed %s %s: %s", track, dt_str, e)
        results_data = []

    if not results_data:
        return n_programs, 0

    with get_db() as db:
        t = upsert_track(db, track, TRACK_NAMES.get(track, track))
        rd = upsert_race_day(db, t.track_id, race_date)

        for rno, data in results_data:
            storage_uri, chash = save_json_snapshot("results", track, dt_str, data)
            upsert_snapshot(
                db,
                source="results",
                url=f"autorace.jp/results/{track}/{dt_str}/R{rno}",
                fetched_at=datetime.now(timezone.utc),
                http_status=200,
                content_hash=chash,
                content_type="application/json",
                storage_uri=storage_uri,
            )
            race = upsert_race(db, rd.race_day_id, rno)

            result = parse_result(data)
            if result:
                upsert_result(db, race.race_id, result)
                n_results += 1

            payout = parse_exacta_payout(data)
            if payout:
                upsert_payout_exacta(db, race.race_id, payout)

    return n_programs, n_results


def main() -> None:
    lookback_days = int(sys.argv[1]) if len(sys.argv) > 1 else 90
    tracks = sys.argv[2].split(",") if len(sys.argv) > 2 else TRACKS

    today = date.today()
    start = today - timedelta(days=lookback_days)

    total_programs = 0
    total_results = 0
    days_with_data = 0

    logger.info(
        "Batch fetch: %s ~ %s, tracks=%s (%d days)",
        start, today, tracks, lookback_days,
    )

    current = start
    while current < today:
        for track in tracks:
            try:
                n_prog, n_res = fetch_day(track, current)
            except Exception as e:
                logger.warning("Error %s %s: %s", track, current, e)
                n_prog, n_res = 0, 0

            if n_prog > 0:
                total_programs += n_prog
                total_results += n_res
                days_with_data += 1
                logger.info(
                    "%s %s: %d programs, %d results",
                    track, current, n_prog, n_res,
                )

        current += timedelta(days=1)

    logger.info(
        "=== Done: %d days with data, %d programs, %d results ===",
        days_with_data, total_programs, total_results,
    )


if __name__ == "__main__":
    main()
