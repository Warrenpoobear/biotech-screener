"""Deterministic transaction-cost model for strategy backtests.

Uses PIT-safe ADV (avg_volume * price) from archive market_data.json to
estimate spread + market-impact costs.  All inputs are per-trade; the model
is stateless and has zero external dependencies.

Conventions
-----------
- Costs are in **basis points** (bps): 1 bp = 0.01%.
- ``estimate_trade_cost`` returns a *one-way* cost (spread + impact).
  Multiply by 2 for round-trip.
- ``participation_pct`` = trade_value / adv_dollars.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, fields as dc_fields
from typing import Tuple


@dataclass(frozen=True)
class CostSchedule:
    """Immutable, versioned cost-model parameters.

    ``spread_schedule`` maps (adv_floor_dollars, half_spread_bps) tuples.
    The schedule is scanned top-down; the first entry whose floor is <=
    the ticker's ADV determines the spread.
    """

    spread_schedule: Tuple[Tuple[int, int], ...] = (
        (10_000_000, 3),    # ADV >= $10M
        (5_000_000, 5),     # ADV >= $5M
        (1_000_000, 10),    # ADV >= $1M
        (500_000, 18),      # ADV >= $500K
        (0, 25),            # ADV < $500K
    )
    impact_coeff: float = 0.10
    impact_cap_bps: float = 200.0
    aum_dollars: float = 50_000_000

    @property
    def schedule_id(self) -> str:
        """Deterministic 8-char SHA-256 of all fields."""
        parts = []
        for f in dc_fields(self):
            parts.append(f"{f.name}={getattr(self, f.name)}")
        blob = "|".join(parts).encode()
        return hashlib.sha256(blob).hexdigest()[:8]


DEFAULT_SCHEDULE = CostSchedule()


@dataclass(frozen=True)
class CostEstimate:
    """Result of ``estimate_trade_cost``."""

    spread_bps: float
    impact_bps: float
    one_way_bps: float
    round_trip_bps: float
    participation_pct: float
    adv_dollars: float


def _lookup_spread(adv_dollars: float, schedule: Tuple[Tuple[int, int], ...]) -> float:
    """Walk the spread schedule and return half-spread bps for *adv_dollars*."""
    for floor, bps in schedule:
        if adv_dollars >= floor:
            return float(bps)
    # Fallback: last entry (floor=0) should always match, but be safe
    return float(schedule[-1][1])


def estimate_trade_cost(
    weight_pct: float,
    adv_dollars: float,
    schedule: CostSchedule = DEFAULT_SCHEDULE,
) -> CostEstimate:
    """Estimate one-way and round-trip trading cost for a single position.

    Parameters
    ----------
    weight_pct : float
        Portfolio weight in percent (e.g. 5.0 = 5%).
    adv_dollars : float
        Average daily volume in dollars (avg_volume * price).
    schedule : CostSchedule
        Cost-model parameters.

    Returns
    -------
    CostEstimate
        Spread, impact, one-way, round-trip costs in bps,
        plus participation rate and ADV.
    """
    if weight_pct <= 0 or adv_dollars <= 0:
        return CostEstimate(
            spread_bps=0.0,
            impact_bps=0.0,
            one_way_bps=0.0,
            round_trip_bps=0.0,
            participation_pct=0.0,
            adv_dollars=adv_dollars,
        )

    trade_value = schedule.aum_dollars * weight_pct / 100.0
    participation = trade_value / adv_dollars

    spread_bps = _lookup_spread(adv_dollars, schedule.spread_schedule)
    impact_bps = min(
        schedule.impact_coeff * math.sqrt(participation) * 10_000,
        schedule.impact_cap_bps,
    )
    one_way = spread_bps + impact_bps
    round_trip = 2.0 * one_way

    return CostEstimate(
        spread_bps=round(spread_bps, 2),
        impact_bps=round(impact_bps, 2),
        one_way_bps=round(one_way, 2),
        round_trip_bps=round(round_trip, 2),
        participation_pct=round(participation * 100, 4),
        adv_dollars=adv_dollars,
    )
