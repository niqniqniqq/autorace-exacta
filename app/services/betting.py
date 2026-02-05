"""Betting strategy calculations using Kelly Criterion."""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BetRecommendation:
    """Single bet recommendation."""

    race_no: int
    first_car_no: int
    second_car_no: int
    prob: float
    market_odds: float
    ev: float
    kelly_fraction: float  # Raw Kelly percentage (0.0-1.0)
    recommended_stake: int  # Yen amount


@dataclass
class PurchasePlan:
    """Overall purchase plan for a day."""

    bets: list[BetRecommendation]
    total_stake: int
    expected_return: float
    expected_profit: float
    bankroll: int
    kelly_multiplier: float

    @property
    def roi(self) -> float:
        """Expected ROI as percentage."""
        if self.total_stake == 0:
            return 0.0
        return (self.expected_profit / self.total_stake) * 100


def calc_kelly_fraction(prob: float, odds: float) -> float:
    """Calculate Kelly Criterion fraction.

    Kelly formula: f* = (bp - q) / b
    where:
        b = odds - 1 (net odds, what you win per unit bet)
        p = probability of winning
        q = 1 - p (probability of losing)

    Simplified: f* = (p * odds - 1) / (odds - 1)

    Returns 0 if EV is negative (don't bet).
    """
    if odds <= 1.0:
        return 0.0

    ev = prob * odds - 1
    if ev <= 0:
        return 0.0

    kelly = ev / (odds - 1)
    # Cap at 100% (shouldn't happen with valid inputs)
    return min(kelly, 1.0)


def generate_purchase_plan(
    predictions: list[dict],
    bankroll: int,
    kelly_multiplier: float = 0.25,
    min_ev: float = 0.0,
    min_stake: int = 100,
    max_stake_pct: float = 0.10,
) -> PurchasePlan:
    """Generate optimal purchase plan using Kelly Criterion.

    Args:
        predictions: List of prediction dicts with keys:
            race_no, first_car_no, second_car_no, prob, market_odds, ev
        bankroll: Total available bankroll in Yen
        kelly_multiplier: Fraction of Kelly to use (0.25 = quarter Kelly, safer)
        min_ev: Minimum EV threshold to consider betting
        min_stake: Minimum bet amount in Yen (usually 100)
        max_stake_pct: Maximum single bet as % of bankroll

    Returns:
        PurchasePlan with recommended bets and summary stats
    """
    bets: list[BetRecommendation] = []
    max_stake = int(bankroll * max_stake_pct)

    for pred in predictions:
        prob = pred["prob"]
        odds = pred.get("market_odds")
        ev = pred.get("ev")

        # Skip if no market odds
        if odds is None or odds <= 1.0:
            continue

        # Skip if EV below threshold
        if ev is None or ev < min_ev:
            continue

        kelly = calc_kelly_fraction(prob, odds)
        if kelly <= 0:
            continue

        # Apply fractional Kelly
        adjusted_kelly = kelly * kelly_multiplier

        # Calculate stake
        raw_stake = bankroll * adjusted_kelly
        stake = int(round(raw_stake / 100) * 100)  # Round to 100 yen

        # Apply min/max constraints
        if stake < min_stake:
            continue
        stake = min(stake, max_stake)

        bets.append(
            BetRecommendation(
                race_no=pred["race_no"],
                first_car_no=pred["first_car_no"],
                second_car_no=pred["second_car_no"],
                prob=prob,
                market_odds=odds,
                ev=ev,
                kelly_fraction=kelly,
                recommended_stake=stake,
            )
        )

    # Sort by EV descending
    bets.sort(key=lambda b: b.ev, reverse=True)

    # Calculate totals
    total_stake = sum(b.recommended_stake for b in bets)
    expected_return = sum(b.recommended_stake * b.prob * b.market_odds for b in bets)
    expected_profit = expected_return - total_stake

    return PurchasePlan(
        bets=bets,
        total_stake=total_stake,
        expected_return=expected_return,
        expected_profit=expected_profit,
        bankroll=bankroll,
        kelly_multiplier=kelly_multiplier,
    )


def format_purchase_plan(plan: PurchasePlan) -> str:
    """Format purchase plan as human-readable string."""
    lines = []
    lines.append("=" * 50)
    lines.append("購入推奨プラン (Kelly Criterion)")
    lines.append("=" * 50)
    lines.append(f"資金: ¥{plan.bankroll:,}")
    lines.append(f"Kelly係数: {plan.kelly_multiplier:.0%} (フラクショナル)")
    lines.append("")

    if not plan.bets:
        lines.append("推奨なし (EV > 0 の買い目がありません)")
        return "\n".join(lines)

    lines.append("【推奨購入】")
    for bet in plan.bets:
        lines.append(
            f"  R{bet.race_no:2d}: {bet.first_car_no}-{bet.second_car_no}  "
            f"¥{bet.recommended_stake:,}  "
            f"(EV={bet.ev:+.2f}, odds={bet.market_odds:.1f}, "
            f"Kelly={bet.kelly_fraction:.1%})"
        )

    lines.append("")
    lines.append("【サマリー】")
    lines.append(f"  購入点数: {len(plan.bets)}点")
    lines.append(f"  合計投資: ¥{plan.total_stake:,}")
    lines.append(f"  期待リターン: ¥{plan.expected_return:,.0f}")
    lines.append(f"  期待収支: ¥{plan.expected_profit:+,.0f} ({plan.roi:+.1f}%)")

    return "\n".join(lines)
