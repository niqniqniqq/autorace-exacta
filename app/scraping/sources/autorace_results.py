"""Fetch race results and payouts from autorace.jp API."""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

from app.scraping.http import AutoraceClient, resolve_place_code

logger = logging.getLogger(__name__)

RESULT_PATH = "/race_info/RaceResult"
REFUND_PATH = "/race_info/RaceRefund"


def fetch_result(
    client: AutoraceClient, track_code: str, race_date: date, race_no: int
) -> dict[str, Any]:
    """Fetch result for a single race."""
    place_code = resolve_place_code(track_code)
    payload = {
        "placeCode": place_code,
        "raceDate": race_date.isoformat(),
        "raceNo": race_no,
    }
    data = client.post_api(RESULT_PATH, payload)
    if data.get("result") != "Success":
        logger.warning("Result API returned non-success: %s", data.get("errors"))
    return data


def fetch_all_results(
    client: AutoraceClient,
    track_code: str,
    race_date: date,
    race_nos: list[int] | None = None,
    max_race_no: int = 12,
) -> list[tuple[int, dict[str, Any]]]:
    """Fetch results for specified races (or all). Returns [(race_no, data)]."""
    if race_nos is None:
        race_nos = list(range(1, max_race_no + 1))

    results: list[tuple[int, dict[str, Any]]] = []
    for race_no in race_nos:
        try:
            data = fetch_result(client, track_code, race_date, race_no)
            body = data.get("body") or {}
            race_result = body.get("raceResult") or []
            if not race_result:
                logger.info(
                    "No results for %s %s R%d — skipping.", track_code, race_date, race_no
                )
                continue
            results.append((race_no, data))
            logger.info("Fetched results: %s %s R%d", track_code, race_date, race_no)
        except Exception:
            logger.exception("Failed to fetch result R%d", race_no)
    return results
