"""Resolve race dates with meet-day probing."""

from __future__ import annotations

import json
import logging
from datetime import date, timedelta

from app.scraping.http import AutoraceClient
from app.scraping.sources.autorace_program import fetch_program

logger = logging.getLogger(__name__)

_DATE_KEYWORDS = {"auto", "latest", "today"}


def _today() -> date:
    return date.today()


def _probe_meet_day(client: AutoraceClient, track: str, race_date: date) -> bool | None:
    """Return True if meet exists, False if not, None if resolution failed."""
    try:
        data = fetch_program(client, track, race_date, race_no=1)
    except Exception:
        logger.exception("Date resolution failed probing program: %s %s", track, race_date)
        return None

    body = data.get("body") or {}
    player_list = body.get("playerList") or []
    if player_list:
        return True

    body_text = json.dumps(body, ensure_ascii=False)
    if "中止" in body_text or "開催なし" in body_text:
        logger.info("No meet (cancellation marker) for %s %s", track, race_date)
        return False

    logger.info("No meet (empty program) for %s %s", track, race_date)
    return False


def resolve_date_with_reason(
    track: str,
    date_str: str,
    mode: str,
    lookback_days: int = 14,
    *,
    client: AutoraceClient | None = None,
) -> tuple[date | None, str | None]:
    """Resolve race date; return (date, reason) where reason is for None cases."""
    if date_str not in _DATE_KEYWORDS:
        return date.fromisoformat(date_str), None

    probe_client = client or AutoraceClient()
    today = _today()

    if date_str == "today":
        status = _probe_meet_day(probe_client, track, today)
        if status is None:
            logger.warning("Date resolution failed for %s (%s)", track, mode)
            return None, "resolution_failed"
        if status:
            return today, None
        logger.info("No meet today for %s (%s)", track, mode)
        return None, "no_meet"

    for offset in range(0, lookback_days + 1):
        candidate = today - timedelta(days=offset)
        status = _probe_meet_day(probe_client, track, candidate)
        if status is None:
            logger.warning("Date resolution failed for %s (%s)", track, mode)
            return None, "resolution_failed"
        if status:
            if offset:
                logger.info(
                    "Resolved %s (%s) to %s (lookback %d days)",
                    track,
                    mode,
                    candidate,
                    offset,
                )
            return candidate, None

    logger.info(
        "No meet found within %d days for %s (%s)", lookback_days, track, mode
    )
    return None, "no_meet"


def resolve_date(
    track: str, date_str: str, mode: str, lookback_days: int = 14
) -> date | None:
    resolved, _ = resolve_date_with_reason(track, date_str, mode, lookback_days)
    return resolved
