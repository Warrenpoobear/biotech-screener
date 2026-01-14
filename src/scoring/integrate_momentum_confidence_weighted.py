#!/usr/bin/env python3
"""
Wake Robin - Confidence-Weighted Momentum Integration

Integrates Morningstar momentum signals into composite scoring with
confidence-tier adjustments.

GOVERNANCE:
- Decimal arithmetic for financial calculations
- Confidence tiers reduce weight for incomplete data
- Deterministic and auditable
- No fail-loud on missing momentum data

Integration formula:
    final_score = (old_composite * old_weight) + (momentum_score * momentum_weight * confidence_multiplier)

Where:
    - old_weight = 0.85 (85% baseline scoring)
    - momentum_weight = 0.15 (15% max for momentum)
    - confidence_multiplier = 0.25 to 1.0 based on data quality
"""

from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime
import json
from pathlib import Path


# Weight allocation
BASELINE_WEIGHT = Decimal("0.85")  # Original composite components
MOMENTUM_WEIGHT = Decimal("0.15")  # Maximum momentum contribution

# Verify weights sum to 1.0
assert BASELINE_WEIGHT + MOMENTUM_WEIGHT == Decimal("1.0"), "Weights must sum to 1.0"


def integrate_momentum_score(
    ticker: str,
    original_composite: Decimal,
    momentum_signals: dict | None,
    scale_to_10: bool = True
) -> dict:
    """
    Integrate momentum signals into composite score with confidence weighting.

    Args:
        ticker: Stock symbol
        original_composite: Original composite score (0-10 or 0-100 scale)
        momentum_signals: Output from MorningstarMomentumSignals.calculate_all_signals()
                         or None if momentum data unavailable
        scale_to_10: If True, assumes original_composite is 0-10 scale

    Returns:
        dict: {
            "ticker": str,
            "original_composite": Decimal,
            "integrated_composite": Decimal,
            "momentum_contribution": Decimal,
            "confidence_tier": str,
            "confidence_multiplier": Decimal,
            "calculation_details": {...}
        }
    """
    # Convert original composite to 0-100 scale if needed
    if scale_to_10:
        original_100 = original_composite * Decimal("10")
    else:
        original_100 = original_composite

    # Handle missing momentum data gracefully
    if momentum_signals is None:
        return _no_momentum_result(ticker, original_composite, original_100, scale_to_10)

    # Extract momentum components
    momentum_score = momentum_signals.get("composite_momentum_score", Decimal("50.0"))
    confidence_tier = momentum_signals.get("confidence_tier", "UNKNOWN")
    confidence_multiplier = momentum_signals.get("confidence_multiplier", Decimal("0.25"))

    # Ensure Decimal types
    if not isinstance(momentum_score, Decimal):
        momentum_score = Decimal(str(momentum_score))
    if not isinstance(confidence_multiplier, Decimal):
        confidence_multiplier = Decimal(str(confidence_multiplier))

    # Calculate effective momentum weight (reduced by confidence)
    effective_momentum_weight = MOMENTUM_WEIGHT * confidence_multiplier
    adjusted_baseline_weight = Decimal("1.0") - effective_momentum_weight

    # Integrated score calculation
    integrated_100 = (
        (original_100 * adjusted_baseline_weight) +
        (momentum_score * effective_momentum_weight)
    )

    # Ensure bounds
    integrated_100 = max(Decimal("0.0"), min(Decimal("100.0"), integrated_100))

    # Calculate momentum's actual contribution
    momentum_contribution = integrated_100 - (original_100 * adjusted_baseline_weight)

    # Convert back to original scale if needed
    if scale_to_10:
        integrated_final = (integrated_100 / Decimal("10")).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
        original_final = original_composite
    else:
        integrated_final = integrated_100.quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
        original_final = original_100

    return {
        "ticker": ticker,
        "original_composite": original_final,
        "integrated_composite": integrated_final,
        "momentum_contribution": momentum_contribution.quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        ),
        "confidence_tier": confidence_tier,
        "confidence_multiplier": confidence_multiplier,
        "calculation_details": {
            "baseline_weight": adjusted_baseline_weight,
            "momentum_weight": effective_momentum_weight,
            "momentum_score_raw": momentum_score,
            "sharpe_composite": momentum_signals.get("multi_horizon_sharpe", {}).get("composite"),
            "rs_composite": momentum_signals.get("relative_strength_vs_xbi", {}).get("composite"),
            "idio_sharpe": momentum_signals.get("idiosyncratic_momentum", {}).get("idio_sharpe"),
            "risk_penalty": momentum_signals.get("drawdown_gate", {}).get("risk_penalty")
        }
    }


def _no_momentum_result(ticker, original_composite, original_100, scale_to_10):
    """Return result when momentum data is unavailable."""
    return {
        "ticker": ticker,
        "original_composite": original_composite,
        "integrated_composite": original_composite,  # No change
        "momentum_contribution": Decimal("0.0"),
        "confidence_tier": "NONE",
        "confidence_multiplier": Decimal("0.0"),
        "calculation_details": {
            "baseline_weight": Decimal("1.0"),
            "momentum_weight": Decimal("0.0"),
            "momentum_score_raw": None,
            "note": "No momentum data available - using baseline score only"
        }
    }


def batch_integrate_momentum(
    composite_scores: dict,
    momentum_results: dict,
    scale_to_10: bool = True
) -> dict:
    """
    Batch integrate momentum for all tickers in composite scores.

    Args:
        composite_scores: {ticker: Decimal(score), ...}
        momentum_results: {ticker: momentum_signals_dict, ...}
        scale_to_10: Scale assumption for composite scores

    Returns:
        dict: {
            "integrated_scores": {ticker: Decimal, ...},
            "details": {ticker: integration_result, ...},
            "summary": {
                "total_tickers": int,
                "with_momentum": int,
                "without_momentum": int,
                "by_confidence_tier": {...}
            }
        }
    """
    integrated_scores = {}
    details = {}
    tier_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0, "UNKNOWN": 0, "NONE": 0}

    for ticker, original_score in composite_scores.items():
        # Ensure Decimal
        if not isinstance(original_score, Decimal):
            original_score = Decimal(str(original_score))

        # Get momentum signals if available
        momentum_signals = momentum_results.get(ticker)

        # Integrate
        result = integrate_momentum_score(
            ticker=ticker,
            original_composite=original_score,
            momentum_signals=momentum_signals,
            scale_to_10=scale_to_10
        )

        integrated_scores[ticker] = result["integrated_composite"]
        details[ticker] = result
        tier_counts[result["confidence_tier"]] += 1

    # Summary statistics
    with_momentum = sum(1 for t, d in details.items() if d["confidence_tier"] != "NONE")

    return {
        "integrated_scores": integrated_scores,
        "details": details,
        "summary": {
            "total_tickers": len(composite_scores),
            "with_momentum": with_momentum,
            "without_momentum": len(composite_scores) - with_momentum,
            "by_confidence_tier": tier_counts
        }
    }


def save_integration_results(results: dict, output_path: str | Path):
    """Save integration results to JSON with Decimal serialization."""

    def decimal_serializer(obj):
        if isinstance(obj, Decimal):
            return str(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=decimal_serializer)

    return str(output_path)


# ===== ACCEPTANCE TESTS =====

def run_acceptance_tests():
    """Run integration acceptance tests."""

    print("=" * 70)
    print("MOMENTUM INTEGRATION ACCEPTANCE TESTS")
    print("=" * 70)

    # Test 1: Integration with full confidence
    print("\n[Test 1] Full Confidence Integration")

    mock_momentum = {
        "composite_momentum_score": Decimal("75.0"),  # Above average momentum
        "confidence_tier": "HIGH",
        "confidence_multiplier": Decimal("1.0"),
        "multi_horizon_sharpe": {"composite": Decimal("1.5")},
        "relative_strength_vs_xbi": {"composite": Decimal("0.10")},
        "idiosyncratic_momentum": {"idio_sharpe": Decimal("0.8")},
        "drawdown_gate": {"risk_penalty": Decimal("0.95")}
    }

    result = integrate_momentum_score(
        ticker="TEST",
        original_composite=Decimal("7.0"),  # 70 on 0-100 scale
        momentum_signals=mock_momentum,
        scale_to_10=True
    )

    # With 85% baseline + 15% momentum:
    # 70 * 0.85 + 75 * 0.15 = 59.5 + 11.25 = 70.75 -> 7.08 on 0-10 scale
    assert result["integrated_composite"] > Decimal("7.0"), \
        f"FAIL: Positive momentum should increase score, got {result['integrated_composite']}"
    assert result["confidence_tier"] == "HIGH", "FAIL: Wrong confidence tier"
    print(f"  PASS: Original={result['original_composite']}, "
          f"Integrated={result['integrated_composite']}, "
          f"Contribution={result['momentum_contribution']}")

    # Test 2: Low confidence reduces momentum weight
    print("\n[Test 2] Low Confidence Reduces Impact")

    mock_momentum_low = {
        "composite_momentum_score": Decimal("90.0"),  # Very high momentum
        "confidence_tier": "LOW",
        "confidence_multiplier": Decimal("0.5"),  # Only 50% weight
        "multi_horizon_sharpe": {"composite": Decimal("2.0")},
        "relative_strength_vs_xbi": {"composite": Decimal("0.20")},
        "idiosyncratic_momentum": {"idio_sharpe": Decimal("1.5")},
        "drawdown_gate": {"risk_penalty": Decimal("1.0")}
    }

    result_low = integrate_momentum_score(
        ticker="TEST2",
        original_composite=Decimal("7.0"),
        momentum_signals=mock_momentum_low,
        scale_to_10=True
    )

    # Effective momentum weight = 0.15 * 0.5 = 0.075
    # Baseline weight = 0.925
    # 70 * 0.925 + 90 * 0.075 = 64.75 + 6.75 = 71.5 -> 7.15

    # Compare to full confidence with same momentum score
    mock_momentum_high = dict(mock_momentum_low)
    mock_momentum_high["confidence_tier"] = "HIGH"
    mock_momentum_high["confidence_multiplier"] = Decimal("1.0")

    result_high = integrate_momentum_score(
        ticker="TEST3",
        original_composite=Decimal("7.0"),
        momentum_signals=mock_momentum_high,
        scale_to_10=True
    )

    assert result_low["integrated_composite"] < result_high["integrated_composite"], \
        "FAIL: Low confidence should have less impact than high confidence"
    print(f"  PASS: Low conf={result_low['integrated_composite']}, "
          f"High conf={result_high['integrated_composite']}")

    # Test 3: No momentum data returns original
    print("\n[Test 3] No Momentum -> Return Original")

    result_none = integrate_momentum_score(
        ticker="TEST4",
        original_composite=Decimal("8.5"),
        momentum_signals=None,
        scale_to_10=True
    )

    assert result_none["integrated_composite"] == Decimal("8.5"), \
        "FAIL: No momentum should return original score"
    assert result_none["confidence_tier"] == "NONE", "FAIL: Should be NONE tier"
    print(f"  PASS: Original preserved = {result_none['integrated_composite']}")

    # Test 4: Batch integration
    print("\n[Test 4] Batch Integration")

    composite_scores = {
        "TICK1": Decimal("6.0"),
        "TICK2": Decimal("8.0"),
        "TICK3": Decimal("5.0")
    }

    momentum_results = {
        "TICK1": mock_momentum,  # Has momentum
        "TICK3": mock_momentum_low  # Has momentum with low confidence
        # TICK2 has no momentum data
    }

    batch_result = batch_integrate_momentum(
        composite_scores=composite_scores,
        momentum_results=momentum_results,
        scale_to_10=True
    )

    assert batch_result["summary"]["total_tickers"] == 3
    assert batch_result["summary"]["with_momentum"] == 2
    assert batch_result["summary"]["without_momentum"] == 1
    print(f"  PASS: {batch_result['summary']}")

    print("\n" + "=" * 70)
    print("ALL INTEGRATION TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    run_acceptance_tests()
