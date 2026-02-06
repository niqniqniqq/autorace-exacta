"""Parse program API response into structured data."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class EntryData:
    car_no: int
    racer_code: str | None
    racer_name: str
    handicap_m: int | None
    machine_name: str | None
    trial_time: float | None
    deviation: float | None
    quinella_rate: float | None
    trio_rate: float | None
    start_avg: float | None  # 90日間スタート平均タイム
    stats_json: dict[str, Any] = field(default_factory=dict)


def _safe_float(v: Any) -> float | None:
    if v is None or v == "" or v == "-":
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def _safe_int(v: Any) -> int | None:
    if v is None or v == "" or v == "-":
        return None
    try:
        return int(v)
    except (ValueError, TypeError):
        return None


def parse_program(api_response: dict[str, Any], race_no: int) -> list[EntryData]:
    """Parse program API response body into list of EntryData."""
    body = api_response.get("body") or {}
    player_list = body.get("playerList") or []
    if not player_list:
        return []

    # Extract 90-day stats for start average
    latest90 = body.get("latest90List") or {}

    entries: list[EntryData] = []
    for p in player_list:
        try:
            player_code = str(p["playerCode"]) if p.get("playerCode") else None

            # Get start average from latest90List (keyed by player code)
            start_avg = None
            if player_code and player_code in latest90:
                start_avg = _safe_float(latest90[player_code].get("stAve"))

            entry = EntryData(
                car_no=int(p.get("carNo", 0)),
                racer_code=player_code,
                racer_name=p.get("playerName", ""),
                handicap_m=_safe_int(p.get("handicap")),
                machine_name=p.get("bikeName"),
                trial_time=_safe_float(p.get("trialRunTime")),
                deviation=_safe_float(p.get("raceDev")),
                quinella_rate=_safe_float(p.get("rate2")),
                trio_rate=_safe_float(p.get("rate3")),
                start_avg=start_avg,
                stats_json={
                    "rank": p.get("rank"),
                    "age": p.get("age"),
                    "bike_class": p.get("bikeClass"),
                    "graduation_code": p.get("graduationCode"),
                    "place_name": p.get("placeName"),
                },
            )
            entries.append(entry)
        except Exception:
            logger.exception("Failed to parse player entry: %s", p)
    return entries


def parse_race_count(api_response: dict[str, Any]) -> int:
    """Try to determine total race count from a program response. Default 12."""
    return 12
