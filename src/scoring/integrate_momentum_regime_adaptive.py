#!/usr/bin/env python3
"""
Regime-Adaptive Momentum Integration

KEY INSIGHT: Momentum works great in bull markets, fails in bear markets.
Your validation proved this:
- Risk-ON (Oct-Dec 2025): +87.60% spread, 0.713 IC
- Risk-OFF (Jan-Mar 2025): -2.97% spread, -0.078 IC

Solution: Adjust momentum weight based on XBI regime.
"""

from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Union
from typing_extensions import TypedDict
import json
from pathlib import Path


# =============================================================================
# TYPE DEFINITIONS
# =============================================================================

class MomentumSignal(TypedDict, total=False):
    """Momentum signal data for a ticker."""
    composite_momentum_score: str
    confidence_tier: str
    confidence_multiplier: str
    multi_horizon_sharpe: Dict[str, str]
    relative_strength_vs_xbi: Dict[str, str]
    idiosyncratic_momentum: Dict[str, Union[str, int]]
    drawdown_gate: Dict[str, str]


class MomentumMetadata(TypedDict):
    """Metadata for momentum integration."""
    score: str
    confidence_tier: str
    confidence_multiplier: str
    weight_applied: str
    contribution: str


class MomentumIntegrationResult(TypedDict, total=False):
    """Result of momentum integration."""
    status: str
    reason: Optional[str]
    regime: str
    xbi_return_90d: str
    momentum_weight_used: str
    tickers_enriched: int
    rationale: str


class BlendingMetadata(TypedDict):
    """Metadata from blending momentum into composite."""
    original_composite: str
    momentum_score: str
    momentum_weight: str
    regime: str
    confidence_multiplier: str
    contribution_from_momentum: str


def load_momentum_signals(path: str = "outputs/momentum_signals.json") -> Dict[str, MomentumSignal]:
    """Load pre-calculated momentum signals."""
    with open(path) as f:
        data = json.load(f)
    return data.get('signals', {})


def load_xbi_returns(path: str = "data/returns/returns_db_daily.json") -> Dict[str, Decimal]:
    """Load XBI daily returns for regime calculation."""
    with open(path) as f:
        data = json.load(f)

    xbi_data = data.get('benchmark', {}).get('XBI', {})
    return {date: Decimal(str(ret)) for date, ret in xbi_data.items()}


def calculate_xbi_regime(
    xbi_returns: Dict[str, Decimal],
    calc_date: str,
    lookback_days: int = 90
) -> Tuple[str, Decimal]:
    """
    Classify current regime based on XBI performance.

    Returns: (regime, xbi_return_90d)
        regime: "risk_on" | "neutral" | "risk_off"
    """
    # Get dates up to calc_date
    dates_sorted = sorted([d for d in xbi_returns.keys() if d <= calc_date])

    if len(dates_sorted) < lookback_days:
        return "neutral", Decimal("0.0")

    window_dates = dates_sorted[-lookback_days:]

    # Calculate cumulative XBI return
    cum_return = Decimal("1.0")
    for date in window_dates:
        cum_return *= (Decimal("1.0") + xbi_returns[date])

    xbi_return_90d = cum_return - Decimal("1.0")

    # Classify regime
    if xbi_return_90d > Decimal("0.10"):  # >10% in 90 days
        regime = "risk_on"
    elif xbi_return_90d < Decimal("-0.05"):  # <-5% in 90 days
        regime = "risk_off"
    else:
        regime = "neutral"

    return regime, xbi_return_90d


# Regime-adaptive momentum weights
REGIME_MOMENTUM_WEIGHTS = {
    "risk_on": Decimal("0.25"),    # Bull market: momentum works great
    "neutral": Decimal("0.15"),    # Sideways: momentum uncertain
    "risk_off": Decimal("0.05"),   # Bear market: momentum often fails
}


def get_momentum_weight(regime: str, confidence_multiplier: Decimal = Decimal("1.0")) -> Decimal:
    """
    Get effective momentum weight based on regime and confidence.

    Args:
        regime: "risk_on" | "neutral" | "risk_off"
        confidence_multiplier: From momentum signals (0.25 to 1.0)

    Returns:
        Effective momentum weight (0 to 0.25)
    """
    base_weight = REGIME_MOMENTUM_WEIGHTS.get(regime, Decimal("0.15"))
    return base_weight * confidence_multiplier


def apply_momentum_to_composite(
    composite_score: Decimal,
    momentum_score: Decimal,
    regime: str,
    confidence_multiplier: Decimal = Decimal("1.0")
) -> Tuple[Decimal, BlendingMetadata]:
    """
    Blend momentum into existing composite score.

    The existing composite score is treated as (1 - momentum_weight) of the final,
    and momentum is added with momentum_weight.

    Args:
        composite_score: Existing composite score (0-100)
        momentum_score: Momentum score (0-100)
        regime: Market regime for weight selection
        confidence_multiplier: Momentum confidence (0.25-1.0)

    Returns:
        (final_score, metadata_dict)
    """
    momentum_weight = get_momentum_weight(regime, confidence_multiplier)
    base_weight = Decimal("1.0") - momentum_weight

    final_score = (composite_score * base_weight) + (momentum_score * momentum_weight)

    # Cap at 0-100
    final_score = min(Decimal("100"), max(Decimal("0"), final_score))

    metadata = {
        "original_composite": str(composite_score),
        "momentum_score": str(momentum_score),
        "momentum_weight": str(momentum_weight),
        "regime": regime,
        "confidence_multiplier": str(confidence_multiplier),
        "contribution_from_momentum": str(momentum_score * momentum_weight),
    }

    return final_score, metadata


def enrich_module5_with_momentum(
    module5_output: Dict[str, object],
    momentum_signals_path: str = "outputs/momentum_signals.json",
    returns_path: str = "data/returns/returns_db_daily.json",
    calc_date: Optional[str] = None
) -> Dict[str, object]:
    """
    Enrich Module 5 output with regime-adaptive momentum signals.

    This is the main integration function. Call after compute_module_5_composite().

    Args:
        module5_output: Output from compute_module_5_composite()
        momentum_signals_path: Path to pre-calculated momentum signals
        returns_path: Path to daily returns for regime calculation
        calc_date: Override calculation date (default: use module5's as_of_date)

    Returns:
        Enriched module5_output with momentum-adjusted scores
    """
    # Load momentum signals
    momentum_path = Path(momentum_signals_path)
    if not momentum_path.exists():
        # No momentum signals - return original
        module5_output["momentum_integration"] = {
            "status": "skipped",
            "reason": f"Momentum signals not found at {momentum_signals_path}"
        }
        return module5_output

    momentum_signals = load_momentum_signals(momentum_signals_path)

    # Load XBI returns for regime
    returns_path_obj = Path(returns_path)
    if not returns_path_obj.exists():
        module5_output["momentum_integration"] = {
            "status": "skipped",
            "reason": f"Returns data not found at {returns_path}"
        }
        return module5_output

    xbi_returns = load_xbi_returns(returns_path)

    # Get calculation date
    if calc_date is None:
        calc_date = module5_output.get("provenance", {}).get("as_of_date", "2026-01-14")

    # Calculate regime
    regime, xbi_return_90d = calculate_xbi_regime(xbi_returns, calc_date)

    # Enrich each security
    enriched_count = 0
    for security in module5_output.get("ranked_securities", []):
        ticker = security.get("ticker")

        if ticker not in momentum_signals:
            security["momentum"] = None
            continue

        mom_sig = momentum_signals[ticker]
        momentum_score = Decimal(str(mom_sig["composite_momentum_score"]))
        confidence_mult = Decimal(str(mom_sig["confidence_multiplier"]))

        # Get original composite score
        original_score = security.get("composite_score", Decimal("50"))
        if isinstance(original_score, (int, float, str)):
            original_score = Decimal(str(original_score))

        # Apply momentum
        final_score, metadata = apply_momentum_to_composite(
            composite_score=original_score,
            momentum_score=momentum_score,
            regime=regime,
            confidence_multiplier=confidence_mult
        )

        # Update security record
        security["composite_score_pre_momentum"] = str(original_score)
        security["composite_score"] = final_score.quantize(Decimal("0.01"))
        security["momentum"] = {
            "score": str(momentum_score),
            "confidence_tier": mom_sig["confidence_tier"],
            "confidence_multiplier": str(confidence_mult),
            "weight_applied": metadata["momentum_weight"],
            "contribution": metadata["contribution_from_momentum"],
        }
        enriched_count += 1

    # Re-rank by new composite score
    ranked = sorted(
        module5_output.get("ranked_securities", []),
        key=lambda x: x.get("composite_score", Decimal("0")),
        reverse=True  # Higher is better after momentum integration
    )

    for i, security in enumerate(ranked):
        security["composite_rank"] = i + 1

    module5_output["ranked_securities"] = ranked

    # Add integration metadata
    module5_output["momentum_integration"] = {
        "status": "applied",
        "regime": regime,
        "xbi_return_90d": str(xbi_return_90d),
        "momentum_weight_used": str(REGIME_MOMENTUM_WEIGHTS[regime]),
        "tickers_enriched": enriched_count,
        "rationale": {
            "risk_on": "25% weight - Momentum highly predictive in bull markets",
            "neutral": "15% weight - Momentum moderately predictive",
            "risk_off": "5% weight - Momentum unreliable in bear markets"
        }[regime]
    }

    return module5_output


# Convenience function for standalone use
def integrate_momentum(
    module5_output_path: str,
    output_path: Optional[str] = None,
    momentum_signals_path: str = "outputs/momentum_signals.json",
    returns_path: str = "data/returns/returns_db_daily.json"
) -> Dict[str, object]:
    """
    Load Module 5 output, integrate momentum, save result.

    Usage:
        python -c "from src.scoring.integrate_momentum_regime_adaptive import integrate_momentum; integrate_momentum('outputs/module5.json', 'outputs/module5_with_momentum.json')"
    """
    with open(module5_output_path) as f:
        module5_output = json.load(f)

    enriched = enrich_module5_with_momentum(
        module5_output,
        momentum_signals_path,
        returns_path
    )

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(enriched, f, indent=2, default=str)
        print(f"Saved momentum-enriched output to: {output_path}")

    return enriched


if __name__ == "__main__":
    # Quick test
    import sys

    if len(sys.argv) > 1:
        integrate_momentum(
            module5_output_path=sys.argv[1],
            output_path=sys.argv[2] if len(sys.argv) > 2 else None
        )
    else:
        print("Usage: python integrate_momentum_regime_adaptive.py <module5_output.json> [output.json]")
