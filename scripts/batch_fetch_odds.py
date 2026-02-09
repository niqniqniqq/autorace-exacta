"""Batch fetch odds for races that already have programs and results."""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone

from sqlalchemy import select, distinct

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(name)s — %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

from app.db.models import Race, RaceDay, RaceResult, Track, OddsExacta
from app.db.session import get_db


def main() -> None:
    from app.scraping.http import AutoraceClient
    from app.scraping.parsers.odds_parser import parse_exacta_odds
    from app.scraping.sources.autorace_odds import fetch_all_odds
    from app.services.storage import save_json_snapshot
    from app.services.upsert import upsert_odds_exacta, upsert_snapshot

    # Get race_ids that already have odds
    with get_db() as db:
        existing_odds_races = set(
            db.scalars(select(distinct(OddsExacta.race_id))).all()
        )

    # Get all track/date combos that have results but no odds
    with get_db() as db:
        rows = db.execute(
            select(
                Track.track_code,
                RaceDay.race_date,
                Race.race_no,
                Race.race_id,
            )
            .join(RaceDay, Race.race_day_id == RaceDay.race_day_id)
            .join(Track, RaceDay.track_id == Track.track_id)
            .join(RaceResult, RaceResult.race_id == Race.race_id)
            .where(RaceResult.is_valid == True)
            .order_by(RaceDay.race_date, Track.track_code, Race.race_no)
        ).all()

    # Group by (track, date)
    groups: dict[tuple[str, str], list[int]] = {}
    skipped = 0
    for track_code, race_date, race_no, race_id in rows:
        if race_id in existing_odds_races:
            skipped += 1
            continue
        key = (track_code, race_date.isoformat())
        groups.setdefault(key, []).append(race_no)

    logger.info(
        "Need odds for %d track-days (%d races), skipping %d with existing odds",
        len(groups),
        sum(len(v) for v in groups.values()),
        skipped,
    )

    total_odds = 0
    for (track_code, dt_str), race_nos in sorted(groups.items()):
        try:
            client = AutoraceClient(init_track_code=track_code)
        except Exception as e:
            logger.debug("CSRF failed %s: %s", track_code, e)
            continue

        race_date_obj = __import__("datetime").date.fromisoformat(dt_str)
        try:
            odds_data = fetch_all_odds(client, track_code, race_date_obj, race_nos=race_nos)
        except Exception as e:
            logger.warning("Odds fetch failed %s %s: %s", track_code, dt_str, e)
            continue

        if not odds_data:
            continue

        captured_at = datetime.now(timezone.utc)
        with get_db() as db:
            day_odds = 0
            for rno, data in odds_data:
                # Find race_id
                race = db.execute(
                    select(Race)
                    .join(RaceDay)
                    .join(Track)
                    .where(
                        Track.track_code == track_code,
                        RaceDay.race_date == race_date_obj,
                        Race.race_no == rno,
                    )
                ).scalar_one_or_none()
                if race is None:
                    continue

                storage_uri, chash = save_json_snapshot("odds", track_code, dt_str, data)
                upsert_snapshot(
                    db,
                    source="odds",
                    url=f"autorace.jp/odds/{track_code}/{dt_str}/R{rno}",
                    fetched_at=captured_at,
                    http_status=200,
                    content_hash=chash,
                    content_type="application/json",
                    storage_uri=storage_uri,
                )
                odds_list = parse_exacta_odds(data)
                cnt = upsert_odds_exacta(db, race.race_id, odds_list, captured_at)
                day_odds += cnt

            total_odds += day_odds
            logger.info("%s %s: %d odds entries", track_code, dt_str, day_odds)

    logger.info("=== Done: %d total odds entries ===", total_odds)


if __name__ == "__main__":
    main()
