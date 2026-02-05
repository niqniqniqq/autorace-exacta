"""Fetch race program data from autorace.jp API."""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

from app.scraping.http import AutoraceClient, resolve_place_code

logger = logging.getLogger(__name__)

API_PATH = "/race_info/Program"


def fetch_program(
    client: AutoraceClient, track_code: str, race_date: date, race_no: int
) -> dict[str, Any]:
    """Fetch program for a single race. Returns the raw API body."""
    place_code = resolve_place_code(track_code)
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
    client: AutoraceClient,
    track_code: str,
    race_date: date,
    race_nos: list[int] | None = None,
    max_race_no: int = 14,
    preloaded: dict[int, dict[str, Any]] | None = None,
) -> list[tuple[int, dict[str, Any]]]:
    """Fetch programs for specified races (or all). Returns [(race_no, data)].

    When *race_nos* is given, only those races are fetched (skipping empties
    rather than stopping).  When ``None``, probes 1..max_race_no and stops
    on the first empty response (legacy behaviour).

    *preloaded* is a dict of ``{race_no: raw_data}`` already fetched (e.g.
    by :func:`resolve_active_race_nos`).  Entries present in *preloaded* are
    reused without an additional API call.
    """
    if preloaded is None:
        preloaded = {}

    if race_nos is not None:
        # Explicit list — iterate all, skip empties.
        results: list[tuple[int, dict[str, Any]]] = []
        for race_no in race_nos:
            try:
                data = preloaded.get(race_no)
                if data is None:
                    data = fetch_program(client, track_code, race_date, race_no)
                body = data.get("body") or {}
                player_list = body.get("playerList") or []
                if not player_list:
                    logger.info(
                        "No players for %s %s R%d — skipping.",
                        track_code,
                        race_date,
                        race_no,
                    )
                    continue
                results.append((race_no, data))
                logger.info(
                    "Fetched program: %s %s R%d (%d entries)",
                    track_code,
                    race_date,
                    race_no,
                    len(player_list),
                )
            except Exception:
                logger.exception("Failed to fetch program R%d", race_no)
        return results

    # Legacy path: probe 1..max_race_no, stop on first empty.
    results = []
    for race_no in range(1, max_race_no + 1):
        try:
            data = preloaded.get(race_no)
            if data is None:
                data = fetch_program(client, track_code, race_date, race_no)
            body = data.get("body") or {}
            player_list = body.get("playerList") or []
            if not player_list:
                logger.info(
                    "No players for %s %s R%d — stopping.",
                    track_code,
                    race_date,
                    race_no,
                )
                break
            results.append((race_no, data))
            logger.info(
                "Fetched program: %s %s R%d (%d entries)",
                track_code,
                race_date,
                race_no,
                len(player_list),
            )
        except Exception:
            logger.exception("Failed to fetch program R%d", race_no)
            break
    return results
