"""Fetch exacta odds from autorace.jp API."""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

from app.scraping.http import AutoraceClient, resolve_place_code

logger = logging.getLogger(__name__)

API_PATH = "/race_info/Odds"


def fetch_odds(
    client: AutoraceClient, track_code: str, race_date: date, race_no: int
) -> dict[str, Any]:
    """Fetch odds for a single race. Returns raw API body."""
    place_code = resolve_place_code(track_code)
    payload = {
        "placeCode": place_code,
        "raceDate": race_date.isoformat(),
        "raceNo": race_no,
    }
    data = client.post_api(API_PATH, payload)
    if data.get("result") != "Success":
        logger.warning("Odds API returned non-success: %s", data.get("errors"))
    return data


def fetch_all_odds(
    client: AutoraceClient,
    track_code: str,
    race_date: date,
    race_nos: list[int] | None = None,
    max_race_no: int = 12,
) -> list[tuple[int, dict[str, Any]]]:
    """Fetch odds for specified races (or all). Returns [(race_no, data)]."""
    if race_nos is None:
        race_nos = list(range(1, max_race_no + 1))

    results: list[tuple[int, dict[str, Any]]] = []
    for race_no in race_nos:
        try:
            data = fetch_odds(client, track_code, race_date, race_no)
            body = data.get("body") or {}
            rtw = body.get("rtwOddsList") or {}
            if not rtw:
                logger.info(
                    "No exacta odds for %s %s R%d — skipping.",
                    track_code,
                    race_date,
                    race_no,
                )
                continue
            results.append((race_no, data))
            logger.info("Fetched odds: %s %s R%d", track_code, race_date, race_no)
        except Exception:
            logger.exception("Failed to fetch odds R%d", race_no)
    return results
