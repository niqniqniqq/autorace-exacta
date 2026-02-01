"""Parse race result and payout API responses."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ResultData:
    """Finishing order for a race."""

    order: list[int]  # car_nos in finishing order
    is_valid: bool
    raw_json: dict[str, Any]


@dataclass
class ExactaPayoutData:
    first_car_no: int
    second_car_no: int
    payout_yen: int
    popularity_rank: int | None


def _safe_int(v: Any) -> int | None:
    if v is None or v == "" or v == "-":
        return None
    try:
        return int(v)
    except (ValueError, TypeError):
        return None


def parse_result(api_response: dict[str, Any]) -> ResultData | None:
    """Parse race result into finishing order."""
    body = api_response.get("body") or {}
    race_result = body.get("raceResult") or []
    if not race_result:
        return None

    order: list[int] = []
    has_valid = True

    sorted_result = sorted(race_result, key=lambda x: int(x.get("order") or 99))
    for entry in sorted_result:
        car_no = _safe_int(entry.get("carNo"))
        accident = entry.get("accidentCode")
        if car_no is not None and (not accident or str(accident).strip() == ""):
            order.append(car_no)
        elif car_no is not None:
            order.append(car_no)

    if len(order) < 2:
        has_valid = False

    return ResultData(order=order, is_valid=has_valid, raw_json=body)


def parse_exacta_payout(api_response: dict[str, Any]) -> ExactaPayoutData | None:
    """Parse exacta payout from result or refund response."""
    body = api_response.get("body") or {}

    refund_info = body.get("refundInfo") or {}
    rtw = refund_info.get("rtw") or {}

    if not rtw:
        return None

    first = _safe_int(rtw.get("1thCarNo"))
    second = _safe_int(rtw.get("2thCarNo"))
    payout = _safe_int(rtw.get("refund"))
    pop = _safe_int(rtw.get("pop"))

    if first is None or second is None or payout is None:
        return None

    return ExactaPayoutData(
        first_car_no=first,
        second_car_no=second,
        payout_yen=payout,
        popularity_rank=pop,
    )
