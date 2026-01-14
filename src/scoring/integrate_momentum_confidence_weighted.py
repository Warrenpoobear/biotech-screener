#!/usr/bin/env python3
"""
Integrate momentum signals with confidence-based weighting.

GOVERNANCE: Momentum contribution shrinks automatically when confidence is low.
"""

from decimal import Decimal


def integrate_momentum_signals_with_confidence(
    base_scores,
    momentum_signals,
    ticker,
    momentum_max_weight=Decimal("0.20")  # Cap momentum at 20% max
):
    """
    Add momentum signals to composite score with confidence weighting.

    Args:
        base_scores: Dict with existing scores:
            - financial_health_score: Decimal (0-100)
            - clinical_development_score: Decimal (0-100)
            - institutional_signal_score: Decimal (0-100)
        momentum_signals: Output from MorningstarMomentumSignals.calculate_all_signals()
        ticker: Stock symbol
        momentum_max_weight: Maximum weight for momentum (default 20%)

    Returns:
        dict: Final integrated scores with provenance
    """
    # Extract base scores
    financial = base_scores.get("financial_health_score", Decimal("50.0"))
    clinical = base_scores.get("clinical_development_score", Decimal("50.0"))
    institutional = base_scores.get("institutional_signal_score", Decimal("50.0"))

    # Extract momentum score and confidence
    momentum_score = momentum_signals["composite_momentum_score"]
    confidence_mult = momentum_signals["confidence_multiplier"]

    # Effective momentum weight = max_weight * confidence_multiplier
    effective_momentum_weight = momentum_max_weight * confidence_mult

    # Rebalance other weights to maintain sum = 1.0
    remaining_weight = Decimal("1.0") - effective_momentum_weight

    base_weights = {
        "financial_health": Decimal("0.30"),
        "clinical_development": Decimal("0.35"),
        "institutional_signal": Decimal("0.35")
    }
    # Sum = 1.0

    # Scale base weights proportionally
    adjusted_weights = {
        component: weight * remaining_weight
        for component, weight in base_weights.items()
    }

    # Calculate final score
    final_score = (
        financial * adjusted_weights["financial_health"] +
        clinical * adjusted_weights["clinical_development"] +
        institutional * adjusted_weights["institutional_signal"] +
        momentum_score * effective_momentum_weight
    )

    # Build result
    result = {
        "final_score": final_score,
        "component_scores": {
            "financial_health": {
                "score": financial,
                "weight": adjusted_weights["financial_health"],
                "contribution": financial * adjusted_weights["financial_health"]
            },
            "clinical_development": {
                "score": clinical,
                "weight": adjusted_weights["clinical_development"],
                "contribution": clinical * adjusted_weights["clinical_development"]
            },
            "institutional_signal": {
                "score": institutional,
                "weight": adjusted_weights["institutional_signal"],
                "contribution": institutional * adjusted_weights["institutional_signal"]
            },
            "momentum": {
                "score": momentum_score,
                "weight": effective_momentum_weight,
                "contribution": momentum_score * effective_momentum_weight,
                "confidence_tier": momentum_signals["confidence_tier"],
                "confidence_multiplier": str(confidence_mult)
            }
        },
        "momentum_detail": {
            "multi_horizon_sharpe": str(momentum_signals["multi_horizon_sharpe"]["composite"]),
            "relative_strength": str(momentum_signals["relative_strength_vs_xbi"]["composite"]),
            "idiosyncratic_sharpe": str(momentum_signals["idiosyncratic_momentum"]["idio_sharpe"]),
            "drawdown_penalty": str(momentum_signals["drawdown_gate"]["risk_penalty"])
        },
        "provenance": {
            "ticker": ticker,
            "momentum_calc_date": momentum_signals["provenance"]["calc_date"],
            "n_observations": momentum_signals["provenance"]["n_observations"]
        }
    }

    return result


def integrate_momentum_batch(
    tickers_base_scores,
    momentum_calculator,
    returns_db,
    calc_date,
    momentum_max_weight=Decimal("0.20")
):
    """
    Batch integration for multiple tickers.

    Args:
        tickers_base_scores: Dict[ticker, base_scores_dict]
        momentum_calculator: MorningstarMomentumSignals instance
        returns_db: Dict with "tickers" and "benchmark" keys
        calc_date: Calculation date string (YYYY-MM-DD)
        momentum_max_weight: Maximum weight for momentum

    Returns:
        Dict[ticker, integrated_result]
    """
    results = {}

    # Extract XBI benchmark returns
    xbi_returns = returns_db.get("benchmark", {}).get("XBI", {})

    # Convert XBI returns to Decimal if needed
    xbi_returns_decimal = {}
    for date, ret in xbi_returns.items():
        if isinstance(ret, Decimal):
            xbi_returns_decimal[date] = ret
        else:
            xbi_returns_decimal[date] = Decimal(str(ret))

    for ticker, base_scores in tickers_base_scores.items():
        # Get ticker returns
        ticker_returns = returns_db.get("tickers", {}).get(ticker, {})

        if not ticker_returns:
            # No returns data - use base scores only
            results[ticker] = {
                "final_score": (
                    base_scores.get("financial_health_score", Decimal("50.0")) * Decimal("0.30") +
                    base_scores.get("clinical_development_score", Decimal("50.0")) * Decimal("0.35") +
                    base_scores.get("institutional_signal_score", Decimal("50.0")) * Decimal("0.35")
                ),
                "momentum_available": False,
                "provenance": {"ticker": ticker, "reason": "no_returns_data"}
            }
            continue

        # Convert ticker returns to Decimal if needed
        ticker_returns_decimal = {}
        for date, ret in ticker_returns.items():
            if isinstance(ret, Decimal):
                ticker_returns_decimal[date] = ret
            else:
                ticker_returns_decimal[date] = Decimal(str(ret))

        # Calculate momentum signals
        momentum_signals = momentum_calculator.calculate_all_signals(
            ticker,
            ticker_returns_decimal,
            xbi_returns_decimal,
            calc_date
        )

        # Integrate with base scores
        integrated = integrate_momentum_signals_with_confidence(
            base_scores,
            momentum_signals,
            ticker,
            momentum_max_weight
        )
        integrated["momentum_available"] = True

        results[ticker] = integrated

    return results
