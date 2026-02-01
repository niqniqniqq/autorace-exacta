"""Fetch race program data from autorace.jp API."""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

from app.scraping.http import PLACE_CODES, AutoraceClient

logger = logging.getLogger(__name__)

API_PATH = "/race_info/Program"


def fetch_program(
    client: AutoraceClient, track_code: str, race_date: date, race_no: int
) -> dict[str, Any]:
    """Fetch program for a single race. Returns the raw API body."""
    place_code = PLACE_CODES[track_code]
    payload = {
        "placeCode": place_code,
        "raceDate": race_date.isoformat(),
        "raceNo": race_no,
    }
    data = client.post_api(API_PATH, payload)
    if data.get("result") != "Success":
        logger.warning("Program API returned non-success: %s", data.get("errors"))
    return data


def fetch_all_programs(
    client: AutoraceClient, track_code: str, race_date: date, max_race_no: int = 12
) -> list[dict[str, Any]]:
    """Fetch programs for all races of the day. Returns list of raw API responses."""
    results: list[dict[str, Any]] = []
    for race_no in range(1, max_race_no + 1):
        try:
            data = fetch_program(client, track_code, race_date, race_no)
            body = data.get("body") or {}
            player_list = body.get("playerList") or []
            if not player_list:
                logger.info("No players for %s R%d — stopping.", race_date, race_no)
                break
            results.append(data)
            logger.info(
                "Fetched program: %s %s R%d (%d entries)",
                track_code, race_date, race_no, len(player_list),
            )
        except Exception:
            logger.exception("Failed to fetch program R%d", race_no)
            break
    return results
