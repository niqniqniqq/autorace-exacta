"""Resolve which race numbers are active for a given track + date."""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

from app.scraping.http import AutoraceClient
from app.scraping.sources.autorace_program import fetch_program

logger = logging.getLogger(__name__)


def resolve_active_race_nos(
    client: AutoraceClient,
    track_code: str,
    race_date: date,
    *,
    max_race_no: int = 14,
    program_cache: dict[int, dict[str, Any]] | None = None,
) -> list[int]:
    """Probe Program API for race_no 1..max_race_no and return active race numbers.

    A race is considered active when its ``playerList`` is non-empty.
    Probing stops after 3 consecutive empty responses as an optimisation.

    If *program_cache* is provided, raw API responses for active races are
    stored there so the caller can reuse them without re-fetching.
    """
    active: list[int] = []
    consecutive_empty = 0

    for race_no in range(1, max_race_no + 1):
        try:
            data = fetch_program(client, track_code, race_date, race_no)
            body = data.get("body") or {}
            player_list = body.get("playerList") or []

            if not player_list:
                consecutive_empty += 1
                if consecutive_empty >= 3:
                    logger.debug(
                        "3 consecutive empty races at R%d — stopping probe.",
                        race_no,
                    )
                    break
                continue

            # Active race found — reset counter.
            consecutive_empty = 0
            active.append(race_no)
            if program_cache is not None:
                program_cache[race_no] = data

        except Exception:
            logger.exception(
                "Error probing program R%d for %s %s — treating as empty",
                race_no,
                track_code,
                race_date,
            )
            consecutive_empty += 1
            if consecutive_empty >= 3:
                break

    logger.info(
        "active_race_nos=%s track=%s date=%s",
        active,
        track_code,
        race_date,
    )
    return active
