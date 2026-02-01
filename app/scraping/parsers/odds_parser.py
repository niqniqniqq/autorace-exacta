"""Parse odds API response into structured exacta odds."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ExactaOddsData:
    first_car_no: int
    second_car_no: int
    odds: float


def _safe_float(v: Any) -> float | None:
    if v is None or v == "" or v == "-" or v == "0" or v == "0.0":
        return None
    try:
        f = float(v)
        return f if f > 0 else None
    except (ValueError, TypeError):
        return None


def parse_exacta_odds(api_response: dict[str, Any]) -> list[ExactaOddsData]:
    """Parse exacta (2連単 / rtwOddsList) from odds API response.

    The rtwOddsList is a dict keyed by first_car_no (str), each value
    is a dict keyed by second_car_no (str) with odds value.
    """
    body = api_response.get("body") or {}
    rtw = body.get("rtwOddsList") or {}
    if not rtw:
        return []

    odds_list: list[ExactaOddsData] = []
    for first_str, inner in rtw.items():
        if not isinstance(inner, dict):
            continue
        first = int(first_str)
        for second_str, odds_val in inner.items():
            second = int(second_str)
            if first == second:
                continue
            odds = _safe_float(odds_val)
            if odds is None:
                continue
            odds_list.append(ExactaOddsData(first_car_no=first, second_car_no=second, odds=odds))

    logger.debug("Parsed %d exacta odds entries", len(odds_list))
    return odds_list
