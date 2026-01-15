#!/usr/bin/env python3
"""
Pre-deployment portfolio validation script.

Runs comprehensive checks before deploying a portfolio:
- Data freshness
- Position count
- Concentration limits
- Weight integrity
- Price availability
- Beta/volatility sanity
- Momentum score validation
- Rank uniqueness

Usage:
    python scripts/validate_portfolio.py
    python scripts/validate_portfolio.py --portfolio outputs/portfolio_20260115.json
"""

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from collections import Counter

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class ValidationResult:
    def __init__(self, name: str, passed: bool, message: str, severity: str = "error"):
        self.name = name
        self.passed = passed
        self.message = message
        self.severity = severity  # "error", "warning", "info"


def validate_data_freshness(portfolio: dict, max_age_hours: int = 24) -> ValidationResult:
    """Check if portfolio data is recent enough."""
    timestamp = portfolio.get("metadata", {}).get("generation_timestamp")
    if not timestamp:
        return ValidationResult(
            "Data Freshness",
            False,
            "No generation timestamp found",
            "error"
        )

    try:
        gen_time = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        age = datetime.now(gen_time.tzinfo) - gen_time if gen_time.tzinfo else datetime.now() - gen_time
        age_hours = age.total_seconds() / 3600

        if age_hours > max_age_hours:
            return ValidationResult(
                "Data Freshness",
                False,
                f"Portfolio is {age_hours:.1f} hours old (max: {max_age_hours}h)",
                "error"
            )

        return ValidationResult(
            "Data Freshness",
            True,
            f"Portfolio is {age_hours:.1f} hours old",
            "info"
        )
    except Exception as e:
        return ValidationResult(
            "Data Freshness",
            False,
            f"Could not parse timestamp: {e}",
            "error"
        )


def validate_position_count(portfolio: dict, target: int = 60, tolerance: float = 0.1) -> ValidationResult:
    """Check if position count is within tolerance of target."""
    positions = portfolio.get("positions", [])
    count = len(positions)

    min_count = int(target * (1 - tolerance))
    max_count = int(target * (1 + tolerance))

    if min_count <= count <= max_count:
        return ValidationResult(
            "Position Count",
            True,
            f"{count} positions (target: {target})",
            "info"
        )

    return ValidationResult(
        "Position Count",
        False,
        f"{count} positions (expected: {min_count}-{max_count})",
        "error"
    )


def validate_concentration_limits(portfolio: dict, max_position: float = 0.03) -> ValidationResult:
    """Check no position exceeds concentration limit."""
    positions = portfolio.get("positions", [])

    violations = []
    for pos in positions:
        weight = pos.get("weight", 0)
        if weight > max_position * 1.01:  # 1% tolerance for rounding
            violations.append(f"{pos['ticker']}: {weight*100:.2f}%")

    if violations:
        return ValidationResult(
            "Concentration Limits",
            False,
            f"{len(violations)} positions exceed {max_position*100:.0f}% limit: {', '.join(violations[:3])}",
            "error"
        )

    max_weight = max(pos.get("weight", 0) for pos in positions) if positions else 0
    return ValidationResult(
        "Concentration Limits",
        True,
        f"Max position: {max_weight*100:.2f}% (limit: {max_position*100:.0f}%)",
        "info"
    )


def validate_weight_sum(portfolio: dict, tolerance: float = 0.01) -> ValidationResult:
    """Check weights sum to approximately 100%."""
    positions = portfolio.get("positions", [])
    total_weight = sum(pos.get("weight", 0) for pos in positions)

    if abs(total_weight - 1.0) > tolerance:
        return ValidationResult(
            "Weight Sum",
            False,
            f"Weights sum to {total_weight*100:.2f}% (expected: ~100%)",
            "error"
        )

    return ValidationResult(
        "Weight Sum",
        True,
        f"Weights sum to {total_weight*100:.2f}%",
        "info"
    )


def validate_prices(portfolio: dict) -> ValidationResult:
    """Check all positions have valid prices."""
    positions = portfolio.get("positions", [])

    missing_prices = []
    invalid_prices = []

    for pos in positions:
        price = pos.get("price")
        if price is None:
            missing_prices.append(pos["ticker"])
        elif price <= 0:
            invalid_prices.append(f"{pos['ticker']}: ${price}")

    if missing_prices:
        return ValidationResult(
            "Price Data",
            False,
            f"{len(missing_prices)} tickers missing prices: {', '.join(missing_prices[:5])}",
            "error"
        )

    if invalid_prices:
        return ValidationResult(
            "Price Data",
            False,
            f"{len(invalid_prices)} tickers with invalid prices: {', '.join(invalid_prices[:5])}",
            "error"
        )

    return ValidationResult(
        "Price Data",
        True,
        f"All {len(positions)} positions have valid prices",
        "info"
    )


def validate_beta_range(portfolio: dict, min_beta: float = 0.50, max_beta: float = 0.95) -> ValidationResult:
    """Check portfolio beta is within expected range."""
    metrics = portfolio.get("metrics", {})
    beta = metrics.get("portfolio_beta_estimate")

    if beta is None:
        return ValidationResult(
            "Portfolio Beta",
            False,
            "No beta estimate available",
            "warning"
        )

    if beta < min_beta or beta > max_beta:
        return ValidationResult(
            "Portfolio Beta",
            False,
            f"Beta {beta:.3f} outside expected range ({min_beta}-{max_beta})",
            "warning"
        )

    return ValidationResult(
        "Portfolio Beta",
        True,
        f"Beta: {beta:.3f} (target: {min_beta}-{max_beta})",
        "info"
    )


def validate_momentum_scores(portfolio: dict) -> ValidationResult:
    """Check momentum scores are sane (0-100 range, no NaN)."""
    positions = portfolio.get("positions", [])

    invalid_momentum = []
    for pos in positions:
        momentum = pos.get("momentum_score")
        if momentum is None:
            invalid_momentum.append(f"{pos['ticker']}: None")
        elif not (0 <= momentum <= 100):
            invalid_momentum.append(f"{pos['ticker']}: {momentum}")

    if invalid_momentum:
        return ValidationResult(
            "Momentum Scores",
            False,
            f"{len(invalid_momentum)} invalid momentum scores: {', '.join(invalid_momentum[:3])}",
            "error"
        )

    avg_momentum = sum(pos.get("momentum_score", 50) for pos in positions) / len(positions) if positions else 0
    return ValidationResult(
        "Momentum Scores",
        True,
        f"All scores valid, avg: {avg_momentum:.1f}",
        "info"
    )


def validate_rank_uniqueness(portfolio: dict) -> ValidationResult:
    """Check all ranks are unique."""
    positions = portfolio.get("positions", [])
    ranks = [pos.get("rank") for pos in positions]

    rank_counts = Counter(ranks)
    duplicates = {r: c for r, c in rank_counts.items() if c > 1}

    if duplicates:
        return ValidationResult(
            "Rank Uniqueness",
            False,
            f"Duplicate ranks found: {duplicates}",
            "error"
        )

    return ValidationResult(
        "Rank Uniqueness",
        True,
        f"All {len(positions)} ranks are unique",
        "info"
    )


def validate_no_nan_scores(portfolio: dict) -> ValidationResult:
    """Check no NaN values in composite scores."""
    positions = portfolio.get("positions", [])

    nan_scores = []
    for pos in positions:
        score = pos.get("composite_score")
        if score is None or (isinstance(score, float) and str(score).lower() == "nan"):
            nan_scores.append(pos["ticker"])

    if nan_scores:
        return ValidationResult(
            "Score Integrity",
            False,
            f"{len(nan_scores)} tickers with NaN scores: {', '.join(nan_scores[:5])}",
            "error"
        )

    return ValidationResult(
        "Score Integrity",
        True,
        "No NaN scores found",
        "info"
    )


def main():
    parser = argparse.ArgumentParser(description="Validate portfolio before deployment")
    parser.add_argument(
        "--portfolio",
        help="Path to portfolio JSON file (default: most recent in outputs/)"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors"
    )
    args = parser.parse_args()

    # Find portfolio file
    if args.portfolio:
        portfolio_path = Path(args.portfolio)
    else:
        # Find most recent portfolio file
        outputs_dir = Path("outputs")
        portfolio_files = list(outputs_dir.glob("portfolio_*.json"))
        if not portfolio_files:
            print("Error: No portfolio files found in outputs/")
            return 1
        portfolio_path = max(portfolio_files, key=lambda p: p.stat().st_mtime)

    if not portfolio_path.exists():
        print(f"Error: Portfolio file not found: {portfolio_path}")
        return 1

    print(f"Validating: {portfolio_path}")
    print("=" * 70)

    # Load portfolio
    with open(portfolio_path) as f:
        portfolio = json.load(f)

    # Run all validations
    validators = [
        validate_data_freshness,
        validate_position_count,
        validate_concentration_limits,
        validate_weight_sum,
        validate_prices,
        validate_beta_range,
        validate_momentum_scores,
        validate_rank_uniqueness,
        validate_no_nan_scores,
    ]

    results = [v(portfolio) for v in validators]

    # Display results
    passed = 0
    failed = 0
    warnings = 0

    for result in results:
        if result.passed:
            icon = "✅"
            passed += 1
        elif result.severity == "warning":
            icon = "⚠️ "
            warnings += 1
            if args.strict:
                failed += 1
        else:
            icon = "❌"
            failed += 1

        print(f"{icon} {result.name}: {result.message}")

    print("=" * 70)

    # Summary
    total = len(results)
    if failed == 0:
        print(f"\n✅ VALIDATION PASSED - READY TO DEPLOY")
        print(f"   {passed}/{total} checks passed")
        if warnings > 0:
            print(f"   {warnings} warnings (non-blocking)")

        # Show key metrics
        metrics = portfolio.get("metrics", {})
        print(f"\n   Portfolio beta: {metrics.get('portfolio_beta_estimate', 'N/A')}")
        print(f"   Max position: {metrics.get('max_position_pct', 'N/A')}%")
        print(f"   Positions: {metrics.get('num_positions', 'N/A')}")

        return 0
    else:
        print(f"\n❌ VALIDATION FAILED - DO NOT DEPLOY")
        print(f"   {failed}/{total} checks failed")

        # List failures
        print("\n   Failures:")
        for result in results:
            if not result.passed and (result.severity == "error" or args.strict):
                print(f"   - {result.name}: {result.message}")

        return 1


if __name__ == "__main__":
    sys.exit(main())
