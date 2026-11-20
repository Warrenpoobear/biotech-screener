"""Straddle mispricing — cheap/rich vol score vs historical event magnitude.

Compares the market-implied move (from ATM IV) against the empirical
distribution of abs_gap outcomes for the same event type. A cheap_vol_score
> 1.0 means the straddle is cheap relative to history.

Purely diagnostic — no ranking impact. Wire into review_queue for
action generation once the study validates.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

from common.event_move_lookup import indication_bucket, lookup_event_move, phase_bucket

# An event move priced above this (150% of the share price) is not a usable
# market signal: it comes from junk quotes on thin chains (e.g. SABS ATM IV
# 26.8 = 2,680% annualised on a 9-DTE contract) stretched by sqrt(days/365) to
# a far-dated catalyst. 41/260 rows on 2026-11-11 priced moves above 200%,
# up to 2,641%. Such rows get no cheap_vol_score / priced_move_pct.
MAX_PLAUSIBLE_IMPLIED_MOVE = 1.5


def compute_cheap_vol_score(
    opt_atm_iv: float,
    catalyst_days: int,
    catalyst_family: str,
    lead_program_phase: str,
    therapeutic_area: str,
    event_move_table: Dict[str, Dict[str, Any]],
    actual_straddle_price: Optional[float] = None,
    underlying_price: Optional[float] = None,
) -> Dict[str, Any]:
    """Compare market-implied move to historical empirical distribution.

    Args:
        opt_atm_iv: Annualized ATM implied volatility.
        catalyst_days: Days to catalyst event.
        catalyst_family: REGULATORY / CLINICAL.
        lead_program_phase: Raw phase string (e.g. "3.0").
        therapeutic_area: Raw indication string.
        event_move_table: Loaded from event_move_table.json.
        actual_straddle_price: ATM straddle close price from Massive chain
            (optional). When provided with underlying_price, used instead
            of the IV approximation for more accurate implied move.
        underlying_price: Current underlying price (optional).

    Returns:
        Dict with cheap_vol_score, vol_classification, and context.
    """
    empty = {
        "implied_move": None,
        "historical_p50": None,
        "historical_p25": None,
        "historical_p75": None,
        "cheap_vol_score": None,
        "vol_classification": "",
        "table_confidence": "",
        "n_obs": 0,
        "lookup_key": "",
    }

    if catalyst_days <= 0:
        return empty

    # Market-implied move: prefer actual straddle price, fall back to IV approx
    implied_move = None
    move_source = "iv_approximation"
    if (
        actual_straddle_price is not None
        and underlying_price is not None
        and actual_straddle_price > 0
        and underlying_price > 0
    ):
        implied_move = actual_straddle_price / underlying_price
        move_source = "actual_straddle"
    elif not math.isnan(opt_atm_iv) and opt_atm_iv > 0:
        t = catalyst_days / 365.0
        implied_move = opt_atm_iv * math.sqrt(t)
    else:
        return empty

    if implied_move > MAX_PLAUSIBLE_IMPLIED_MOVE:
        return empty

    # Historical fair move from lookup table
    phase = phase_bucket(lead_program_phase)
    ind = indication_bucket(therapeutic_area)
    hist = lookup_event_move(catalyst_family, phase, ind, event_move_table)

    p50 = hist.get("p50")
    if p50 is None or p50 <= 0:
        return empty

    cheap_vol_score = p50 / implied_move if implied_move > 0 else float("nan")

    # Classification
    confidence = hist.get("confidence", "ok")
    if math.isnan(cheap_vol_score):
        classification = ""
    elif confidence in ("low_confidence", "insufficient"):
        # Suppress strong classifications on thin data
        if cheap_vol_score >= 1.15:
            classification = "SLIGHTLY_CHEAP"
        elif cheap_vol_score <= 0.85:
            classification = "SLIGHTLY_RICH"
        else:
            classification = "FAIR"
    else:
        if cheap_vol_score >= 1.4:
            classification = "CHEAP"
        elif cheap_vol_score >= 1.15:
            classification = "SLIGHTLY_CHEAP"
        elif cheap_vol_score <= 0.65:
            classification = "RICH"
        elif cheap_vol_score <= 0.85:
            classification = "SLIGHTLY_RICH"
        else:
            classification = "FAIR"

    return {
        "implied_move": round(implied_move, 4),
        "move_source": move_source,
        "historical_p50": p50,
        "historical_p25": hist.get("p25"),
        "historical_p75": hist.get("p75"),
        "cheap_vol_score": round(cheap_vol_score, 4),
        "vol_classification": classification,
        "table_confidence": confidence,
        "n_obs": hist.get("n", 0),
        "lookup_key": hist.get("lookup_key", ""),
    }
