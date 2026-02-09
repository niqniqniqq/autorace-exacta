#!/usr/bin/env python
"""Backfill start_avg for historical race entries."""

from __future__ import annotations

import logging
import time
from datetime import date

from sqlalchemy import select

from app.db.models import Race, RaceDay, RaceEntry, Track
from app.db.session import get_db
from app.scraping.http import AutoraceClient
from app.scraping.sources.autorace_program import fetch_program
from app.scraping.parsers.program_parser import parse_program

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def backfill_start_avg():
    """Re-fetch program data and update start_avg for all entries."""
    client = AutoraceClient()
    updated_total = 0
    skipped_total = 0

    with get_db() as db:
        # Get all race days ordered by date
        race_days = db.execute(
            select(RaceDay, Track)
            .join(Track)
            .order_by(RaceDay.race_date)
        ).all()

        logger.info("Processing %d race days", len(race_days))

        for rd, track in race_days:
            logger.info("Processing %s %s", track.track_code, rd.race_date)

            # Get all races for this day
            races = db.scalars(
                select(Race)
                .where(Race.race_day_id == rd.race_day_id)
                .order_by(Race.race_no)
            ).all()

            for race in races:
                try:
                    # Fetch program from API
                    raw = fetch_program(client, track.track_code, rd.race_date, race.race_no)
                    if not raw:
                        logger.debug("R%d: empty response", race.race_no)
                        continue

                    entries_data = parse_program(raw, race.race_no)
                    if not entries_data:
                        continue

                    # Update entries with start_avg
                    updated = 0
                    for ed in entries_data:
                        if ed.start_avg is None:
                            continue

                        entry = db.scalar(
                            select(RaceEntry).where(
                                RaceEntry.race_id == race.race_id,
                                RaceEntry.car_no == ed.car_no
                            )
                        )
                        if entry and entry.start_avg is None:
                            entry.start_avg = ed.start_avg
                            updated += 1

                    if updated > 0:
                        logger.info("  R%d: updated %d entries", race.race_no, updated)
                        updated_total += updated
                    else:
                        skipped_total += 1

                    # Rate limiting
                    time.sleep(0.5)

                except Exception as e:
                    logger.error("  R%d: error %s", race.race_no, e)
                    continue

            # Commit after each race day
            db.commit()
            logger.info("  Committed %s", rd.race_date)

    logger.info("Done! Updated %d entries, skipped %d races", updated_total, skipped_total)


if __name__ == "__main__":
    backfill_start_avg()
