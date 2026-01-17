#!/usr/bin/env python3
"""
Manager Momentum Engine (v1) - VALIDATION ONLY

Tracks Q/Q position changes for elite biotech managers to detect:
- Fresh convictions (NEW positions)
- Conviction increases (ADD)
- Conviction decreases (TRIM)
- Position exits (EXIT)
- Coordinated activity (multiple managers acting in same direction)

USAGE MODE: VALIDATION ONLY
---------------------------
This module is used for validation and monitoring purposes only.
Momentum signals are NOT used for scoring adjustments in the pipeline.
The momentum_score and related metrics are for analyst review and
audit trails, not for automated score modifications.

Design Philosophy (Governed Over Smart):
- DETERMINISTIC: No datetime.now(), no randomness
- STDLIB-ONLY: No external dependencies (uses existing 13F infrastructure)
- DECIMAL-ONLY: Pure Decimal arithmetic for all calculations
- FAIL LOUDLY: Clear error states
- AUDITABLE: Full provenance chain

DETERMINISM CONTRACT:
----------------------
This module guarantees deterministic output for identical inputs:
1. All arithmetic uses Decimal with explicit quantization rules
2. Position changes computed from explicit Q/Q comparisons
3. No floating-point intermediate calculations
4. Output field ordering is stable

Data Sources:
- holdings_snapshots.json: Current and historical 13F positions
- manager_registry.json: Elite manager definitions

Output Integration:
- Validation layer for analyst review
- Provides momentum_score for monitoring (not scoring)
- Flags coordinated activity for alerts

Author: Wake Robin Capital Management
Version: 1.0.0
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from common.provenance import create_provenance

__version__ = "1.0.0"
RULESET_VERSION = "1.0.0-MOMENTUM"
SCHEMA_VERSION = "v1.0"

logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

SCORE_PRECISION = Decimal("0.01")
WEIGHT_PRECISION = Decimal("0.0001")
SHARE_CHANGE_THRESHOLD = Decimal("0.10")  # 10% change to qualify as ADD/TRIM
VALUE_CHANGE_THRESHOLD = Decimal("0.15")  # 15% value change threshold

# Scoring weights
NEW_POSITION_SCORE = Decimal("10")
ADD_POSITION_SCORE = Decimal("5")
HOLD_POSITION_SCORE = Decimal("2")
TRIM_POSITION_SCORE = Decimal("-3")
EXIT_POSITION_SCORE = Decimal("-8")

# Coordinated activity thresholds
COORDINATED_ADD_MIN = 3  # At least 3 managers adding = coordinated
COORDINATED_NEW_MIN = 2  # At least 2 managers initiating = strong signal
CROWDING_THRESHOLD = 6   # 6+ managers holding = crowded

# Manager tier weights
ELITE_CORE_WEIGHT = Decimal("1.5")
CONDITIONAL_WEIGHT = Decimal("1.0")


class ConvictionChange(str, Enum):
    """Position change classification."""
    NEW = "NEW"           # Fresh position (wasn't held prior quarter)
    ADD = "ADD"           # Increased position by >10%
    HOLD = "HOLD"         # Position unchanged (+/- 10%)
    TRIM = "TRIM"         # Decreased position by >10%
    EXIT = "EXIT"         # Completely exited position
    UNKNOWN = "UNKNOWN"   # Missing data


class CrowdingLevel(str, Enum):
    """Position crowding classification."""
    LOW = "LOW"           # 1-2 managers
    MODERATE = "MODERATE" # 3-5 managers
    HIGH = "HIGH"         # 6+ managers (crowded)


# ============================================================================
# DATACLASSES
# ============================================================================

@dataclass
class ManagerPosition:
    """Single manager's position in a ticker."""
    manager_cik: str
    manager_name: str
    manager_tier: str  # "elite_core" or "conditional"
    quarter_end: str
    shares: int
    value_kusd: int
    prior_shares: Optional[int]
    prior_value_kusd: Optional[int]
    change: ConvictionChange
    share_change_pct: Optional[Decimal]
    value_change_pct: Optional[Decimal]

    def to_dict(self) -> dict:
        return {
            "manager_cik": self.manager_cik,
            "manager_name": self.manager_name,
            "manager_tier": self.manager_tier,
            "quarter_end": self.quarter_end,
            "shares": self.shares,
            "value_kusd": self.value_kusd,
            "prior_shares": self.prior_shares,
            "prior_value_kusd": self.prior_value_kusd,
            "change": self.change.value,
            "share_change_pct": str(self.share_change_pct) if self.share_change_pct else None,
            "value_change_pct": str(self.value_change_pct) if self.value_change_pct else None,
        }


@dataclass
class TickerMomentum:
    """Aggregated momentum signals for a ticker."""
    ticker: str
    total_managers: int
    elite_core_count: int
    conditional_count: int
    crowding_level: CrowdingLevel
    positions: List[ManagerPosition]

    # Change counts
    new_count: int
    add_count: int
    hold_count: int
    trim_count: int
    exit_count: int

    # Derived signals
    net_conviction: int  # (NEW + ADD) - (TRIM + EXIT)
    momentum_score: Decimal
    coordinated_buying: bool
    coordinated_selling: bool
    fresh_conviction_signal: bool

    # Provenance
    determinism_hash: str

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "total_managers": self.total_managers,
            "elite_core_count": self.elite_core_count,
            "conditional_count": self.conditional_count,
            "crowding_level": self.crowding_level.value,
            "new_count": self.new_count,
            "add_count": self.add_count,
            "hold_count": self.hold_count,
            "trim_count": self.trim_count,
            "exit_count": self.exit_count,
            "net_conviction": self.net_conviction,
            "momentum_score": str(self.momentum_score),
            "coordinated_buying": self.coordinated_buying,
            "coordinated_selling": self.coordinated_selling,
            "fresh_conviction_signal": self.fresh_conviction_signal,
            "determinism_hash": self.determinism_hash,
            "positions": [p.to_dict() for p in self.positions],
        }


# ============================================================================
# HELPERS
# ============================================================================

def _to_decimal(value: Any, default: Optional[Decimal] = None) -> Optional[Decimal]:
    """Convert value to Decimal safely."""
    if value is None:
        return default
    try:
        if isinstance(value, Decimal):
            return value
        if isinstance(value, (int, float)):
            return Decimal(str(value))
        if isinstance(value, str):
            stripped = value.strip()
            return Decimal(stripped) if stripped else default
        return default
    except (InvalidOperation, ValueError):
        return default


def _quantize_score(value: Decimal) -> Decimal:
    """Quantize score to standard precision."""
    return value.quantize(SCORE_PRECISION, rounding=ROUND_HALF_UP)


def _compute_change_pct(current: int, prior: int) -> Optional[Decimal]:
    """Compute percentage change from prior to current."""
    if prior == 0:
        return None  # Can't compute % change from zero
    change = Decimal(str(current - prior)) / Decimal(str(prior))
    return _quantize_score(change * Decimal("100"))


def _classify_change(
    current_shares: int,
    prior_shares: Optional[int],
    current_value: int,
    prior_value: Optional[int],
) -> Tuple[ConvictionChange, Optional[Decimal], Optional[Decimal]]:
    """
    Classify position change between quarters.

    Uses shares as primary signal, falls back to value if shares are zero.

    Returns:
        (change_type, share_change_pct, value_change_pct)
    """
    # Determine if we should use value-based analysis (when shares are 0 but value exists)
    use_value = (current_shares == 0 and current_value > 0) or \
                (prior_shares == 0 and prior_value and prior_value > 0)

    if use_value:
        # Value-based change detection
        if prior_value is None or prior_value == 0:
            if current_value > 0:
                return ConvictionChange.NEW, None, None
            return ConvictionChange.UNKNOWN, None, None

        if current_value == 0:
            return ConvictionChange.EXIT, None, Decimal("-100")

        value_pct = _compute_change_pct(current_value, prior_value)
        if value_pct is None:
            return ConvictionChange.UNKNOWN, None, None

        threshold_pct = VALUE_CHANGE_THRESHOLD * Decimal("100")

        if value_pct > threshold_pct:
            return ConvictionChange.ADD, None, value_pct
        elif value_pct < -threshold_pct:
            return ConvictionChange.TRIM, None, value_pct
        else:
            return ConvictionChange.HOLD, None, value_pct

    # Share-based change detection (original logic)
    if prior_shares is None or prior_shares == 0:
        if current_shares > 0:
            return ConvictionChange.NEW, None, None
        return ConvictionChange.UNKNOWN, None, None

    if current_shares == 0:
        return ConvictionChange.EXIT, Decimal("-100"), Decimal("-100")

    share_pct = _compute_change_pct(current_shares, prior_shares)
    value_pct = _compute_change_pct(current_value, prior_value) if prior_value else None

    if share_pct is None:
        return ConvictionChange.UNKNOWN, None, value_pct

    threshold_pct = SHARE_CHANGE_THRESHOLD * Decimal("100")

    if share_pct > threshold_pct:
        return ConvictionChange.ADD, share_pct, value_pct
    elif share_pct < -threshold_pct:
        return ConvictionChange.TRIM, share_pct, value_pct
    else:
        return ConvictionChange.HOLD, share_pct, value_pct


def _compute_determinism_hash(
    ticker: str,
    positions: List[ManagerPosition],
    momentum_score: Decimal,
) -> str:
    """Compute determinism hash for audit trail."""
    payload = {
        "ticker": ticker,
        "version": SCHEMA_VERSION,
        "positions": sorted([
            {
                "cik": p.manager_cik,
                "change": p.change.value,
                "shares": p.shares,
            }
            for p in positions
        ], key=lambda x: x["cik"]),
        "momentum_score": str(momentum_score),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


# ============================================================================
# CORE ANALYSIS
# ============================================================================

def analyze_ticker_momentum(
    ticker: str,
    holdings_data: Dict,
    manager_registry: Dict,
    prior_quarter: Optional[str] = None,
) -> TickerMomentum:
    """
    Analyze Q/Q momentum for a single ticker.

    Args:
        ticker: Stock ticker
        holdings_data: Holdings for this ticker from holdings_snapshots.json
        manager_registry: Manager definitions
        prior_quarter: Prior quarter to compare (auto-detected if None)

    Returns:
        TickerMomentum with full analysis
    """
    # Build manager lookup
    elite_ciks = {m["cik"]: m for m in manager_registry.get("elite_core", [])}
    conditional_ciks = {m["cik"]: m for m in manager_registry.get("conditional", [])}
    all_managers = {**elite_ciks, **conditional_ciks}

    holdings = holdings_data.get("holdings", {})
    current_holdings = holdings.get("current", {})
    prior_holdings = holdings.get("prior", {})

    positions = []

    # Process each manager
    for cik, current in current_holdings.items():
        if cik not in all_managers:
            continue  # Not an elite manager

        manager = all_managers[cik]
        tier = "elite_core" if cik in elite_ciks else "conditional"

        current_shares = current.get("shares", 0)
        current_value = current.get("value_kusd", 0)
        quarter_end = current.get("quarter_end", "")

        # Get prior quarter data
        prior = prior_holdings.get(cik, {})
        prior_shares = prior.get("shares") if prior else None
        prior_value = prior.get("value_kusd") if prior else None

        # Classify change
        change, share_pct, value_pct = _classify_change(
            current_shares, prior_shares, current_value, prior_value
        )

        positions.append(ManagerPosition(
            manager_cik=cik,
            manager_name=manager.get("name", "Unknown"),
            manager_tier=tier,
            quarter_end=quarter_end,
            shares=current_shares,
            value_kusd=current_value,
            prior_shares=prior_shares,
            prior_value_kusd=prior_value,
            change=change,
            share_change_pct=share_pct,
            value_change_pct=value_pct,
        ))

    # Check for exits (managers who had position but now don't)
    for cik, prior in prior_holdings.items():
        if cik not in all_managers:
            continue
        if cik in current_holdings:
            continue  # Already processed

        manager = all_managers[cik]
        tier = "elite_core" if cik in elite_ciks else "conditional"

        positions.append(ManagerPosition(
            manager_cik=cik,
            manager_name=manager.get("name", "Unknown"),
            manager_tier=tier,
            quarter_end=prior.get("quarter_end", ""),
            shares=0,
            value_kusd=0,
            prior_shares=prior.get("shares", 0),
            prior_value_kusd=prior.get("value_kusd", 0),
            change=ConvictionChange.EXIT,
            share_change_pct=Decimal("-100"),
            value_change_pct=Decimal("-100"),
        ))

    # Aggregate counts
    elite_core_count = sum(1 for p in positions if p.manager_tier == "elite_core" and p.shares > 0)
    conditional_count = sum(1 for p in positions if p.manager_tier == "conditional" and p.shares > 0)
    total_managers = elite_core_count + conditional_count

    new_count = sum(1 for p in positions if p.change == ConvictionChange.NEW)
    add_count = sum(1 for p in positions if p.change == ConvictionChange.ADD)
    hold_count = sum(1 for p in positions if p.change == ConvictionChange.HOLD)
    trim_count = sum(1 for p in positions if p.change == ConvictionChange.TRIM)
    exit_count = sum(1 for p in positions if p.change == ConvictionChange.EXIT)

    # Crowding level
    if total_managers >= CROWDING_THRESHOLD:
        crowding = CrowdingLevel.HIGH
    elif total_managers >= 3:
        crowding = CrowdingLevel.MODERATE
    else:
        crowding = CrowdingLevel.LOW

    # Compute momentum score
    momentum_score = Decimal("0")
    for p in positions:
        weight = ELITE_CORE_WEIGHT if p.manager_tier == "elite_core" else CONDITIONAL_WEIGHT
        if p.change == ConvictionChange.NEW:
            momentum_score += NEW_POSITION_SCORE * weight
        elif p.change == ConvictionChange.ADD:
            momentum_score += ADD_POSITION_SCORE * weight
        elif p.change == ConvictionChange.HOLD:
            momentum_score += HOLD_POSITION_SCORE * weight
        elif p.change == ConvictionChange.TRIM:
            momentum_score += TRIM_POSITION_SCORE * weight
        elif p.change == ConvictionChange.EXIT:
            momentum_score += EXIT_POSITION_SCORE * weight

    momentum_score = _quantize_score(momentum_score)

    # Coordinated signals
    coordinated_buying = (new_count + add_count) >= COORDINATED_ADD_MIN
    coordinated_selling = (trim_count + exit_count) >= COORDINATED_ADD_MIN
    fresh_conviction_signal = new_count >= COORDINATED_NEW_MIN

    # Net conviction
    net_conviction = (new_count + add_count) - (trim_count + exit_count)

    # Determinism hash
    determinism_hash = _compute_determinism_hash(ticker, positions, momentum_score)

    return TickerMomentum(
        ticker=ticker,
        total_managers=total_managers,
        elite_core_count=elite_core_count,
        conditional_count=conditional_count,
        crowding_level=crowding,
        positions=positions,
        new_count=new_count,
        add_count=add_count,
        hold_count=hold_count,
        trim_count=trim_count,
        exit_count=exit_count,
        net_conviction=net_conviction,
        momentum_score=momentum_score,
        coordinated_buying=coordinated_buying,
        coordinated_selling=coordinated_selling,
        fresh_conviction_signal=fresh_conviction_signal,
        determinism_hash=determinism_hash,
    )


# ============================================================================
# MAIN COMPUTATION
# ============================================================================

def compute_manager_momentum(
    holdings_snapshots: Dict[str, Any],
    manager_registry: Dict[str, Any],
    as_of_date: str,
    target_tickers: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compute manager momentum signals for all tickers.

    Args:
        holdings_snapshots: Full holdings data keyed by ticker
        manager_registry: Manager definitions
        as_of_date: Analysis date (YYYY-MM-DD)
        target_tickers: Optional list to filter (analyzes all if None)

    Returns:
        Dict with momentum analysis for each ticker
    """
    logger.info(f"Computing manager momentum for {as_of_date}")

    tickers = target_tickers or list(holdings_snapshots.keys())

    results = {}
    summary = {
        "coordinated_buys": [],
        "coordinated_sells": [],
        "fresh_convictions": [],
        "crowded_positions": [],
    }

    for ticker in tickers:
        if ticker not in holdings_snapshots:
            continue

        holdings_data = holdings_snapshots[ticker]
        momentum = analyze_ticker_momentum(ticker, holdings_data, manager_registry)
        results[ticker] = momentum.to_dict()

        # Track summary signals
        if momentum.coordinated_buying:
            summary["coordinated_buys"].append(ticker)
        if momentum.coordinated_selling:
            summary["coordinated_sells"].append(ticker)
        if momentum.fresh_conviction_signal:
            summary["fresh_convictions"].append(ticker)
        if momentum.crowding_level == CrowdingLevel.HIGH:
            summary["crowded_positions"].append(ticker)

    # Sort results by momentum score descending
    sorted_tickers = sorted(
        results.keys(),
        key=lambda t: Decimal(results[t]["momentum_score"]),
        reverse=True
    )

    return {
        "as_of_date": as_of_date,
        "schema_version": SCHEMA_VERSION,
        "tickers_analyzed": len(results),
        "summary": {
            "coordinated_buys": sorted(summary["coordinated_buys"]),
            "coordinated_sells": sorted(summary["coordinated_sells"]),
            "fresh_convictions": sorted(summary["fresh_convictions"]),
            "crowded_positions": sorted(summary["crowded_positions"]),
            "coordinated_buy_count": len(summary["coordinated_buys"]),
            "coordinated_sell_count": len(summary["coordinated_sells"]),
            "fresh_conviction_count": len(summary["fresh_convictions"]),
            "crowded_count": len(summary["crowded_positions"]),
        },
        "rankings": [
            {"ticker": t, "momentum_score": results[t]["momentum_score"]}
            for t in sorted_tickers[:20]
        ],
        "signals": results,
        "provenance": create_provenance(
            RULESET_VERSION,
            {"tickers": len(results), "as_of_date": as_of_date},
            as_of_date,
        ),
    }


# ============================================================================
# VALIDATION HELPERS (NOT USED FOR SCORING)
# ============================================================================

def get_momentum_validation(
    ticker: str,
    momentum_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Get momentum validation data for analyst review.

    NOTE: This is for VALIDATION ONLY - not used for scoring adjustments.
    Momentum signals are monitored but do not modify composite scores.

    Returns dict with:
        - momentum_score: Raw momentum score (for monitoring)
        - momentum_confidence: Confidence in signal (based on manager coverage)
        - momentum_flags: List of signal flags for alerts
        - momentum_adjustment: Suggested adjustment (FOR REFERENCE ONLY, not applied)
    """
    signals = momentum_result.get("signals", {})
    if ticker not in signals:
        return {
            "momentum_score": Decimal("0"),
            "momentum_confidence": Decimal("0"),
            "momentum_flags": [],
            "momentum_adjustment": Decimal("0"),
        }

    data = signals[ticker]
    score = _to_decimal(data["momentum_score"], Decimal("0"))
    total = data["total_managers"]

    # Confidence based on manager coverage
    if total >= 5:
        confidence = Decimal("1.0")
    elif total >= 3:
        confidence = Decimal("0.7")
    elif total >= 1:
        confidence = Decimal("0.4")
    else:
        confidence = Decimal("0")

    # Build flags
    flags = []
    if data.get("coordinated_buying"):
        flags.append("coordinated_buying")
    if data.get("coordinated_selling"):
        flags.append("coordinated_selling")
    if data.get("fresh_conviction_signal"):
        flags.append("fresh_conviction")
    if data.get("crowding_level") == "HIGH":
        flags.append("crowded_position")

    # Compute adjustment (scaled to +/- 5 points max)
    # Positive momentum adds points, negative subtracts
    adjustment = (score / Decimal("20")).quantize(SCORE_PRECISION)
    adjustment = max(Decimal("-5"), min(Decimal("5"), adjustment))

    return {
        "momentum_score": score,
        "momentum_confidence": confidence,
        "momentum_flags": flags,
        "momentum_adjustment": adjustment,
    }


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Load data
    data_dir = Path("production_data")

    with open(data_dir / "holdings_snapshots.json") as f:
        holdings = json.load(f)

    with open(data_dir / "manager_registry.json") as f:
        registry = json.load(f)

    # Run analysis
    result = compute_manager_momentum(holdings, registry, "2026-01-12")

    # Print summary
    print("=" * 70)
    print("MANAGER MOMENTUM ANALYSIS")
    print("=" * 70)
    print(f"Tickers analyzed: {result['tickers_analyzed']}")
    print(f"\nCoordinated buys ({result['summary']['coordinated_buy_count']}): "
          f"{', '.join(result['summary']['coordinated_buys'][:10])}")
    print(f"Coordinated sells ({result['summary']['coordinated_sell_count']}): "
          f"{', '.join(result['summary']['coordinated_sells'][:10])}")
    print(f"Fresh convictions ({result['summary']['fresh_conviction_count']}): "
          f"{', '.join(result['summary']['fresh_convictions'][:10])}")
    print(f"Crowded positions ({result['summary']['crowded_count']}): "
          f"{', '.join(result['summary']['crowded_positions'][:10])}")

    print("\nTop 20 by momentum score:")
    for r in result["rankings"]:
        print(f"  {r['ticker']}: {r['momentum_score']}")

    # Save
    with open("momentum_results.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nSaved to momentum_results.json")
