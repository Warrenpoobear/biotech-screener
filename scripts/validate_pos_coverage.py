#!/usr/bin/env python3
"""
validate_pos_coverage.py

Validation script for PoS coverage expansion.
Runs the validation checklist:
- Coverage distribution by market cap decile
- False positive audit (20-ticker sample)
- Overall coverage statistics

Author: Wake Robin Capital Management
"""

import json
import random
import sys
from pathlib import Path
from decimal import Decimal
from datetime import date
from typing import Dict, List, Any, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pos_engine import ProbabilityOfSuccessEngine


def load_indication_mapping(path: str = "data/indication_mapping.json") -> Dict[str, Any]:
    """Load the indication mapping configuration."""
    mapping_path = Path(__file__).parent.parent / path
    with open(mapping_path) as f:
        return json.load(f)


def load_universe(path: str = "production_data/universe_mapped.json") -> List[Dict[str, Any]]:
    """Load the production universe."""
    universe_path = Path(__file__).parent.parent / path
    with open(universe_path) as f:
        return json.load(f)


def categorize_market_cap(market_cap_mm: float) -> str:
    """Categorize market cap into deciles/buckets."""
    if market_cap_mm is None or market_cap_mm <= 0:
        return "unknown"
    elif market_cap_mm < 100:
        return "nano (<$100M)"
    elif market_cap_mm < 300:
        return "micro ($100-300M)"
    elif market_cap_mm < 1000:
        return "small ($300M-1B)"
    elif market_cap_mm < 5000:
        return "mid ($1-5B)"
    elif market_cap_mm < 20000:
        return "large ($5-20B)"
    else:
        return "mega (>$20B)"


def get_ticker_indication(
    ticker: str,
    conditions: List[str],
    mapping: Dict[str, Any]
) -> Optional[str]:
    """Determine indication for a ticker using mapping logic."""
    # First check ticker overrides
    if ticker in mapping.get("ticker_overrides", {}):
        return mapping["ticker_overrides"][ticker]

    # Then pattern match against conditions
    condition_text = " ".join(conditions).lower() if conditions else ""

    for category, patterns in mapping.get("condition_patterns", {}).items():
        for pattern in patterns:
            if pattern.lower() in condition_text:
                return category

    return None


def run_coverage_analysis(
    universe: List[Dict[str, Any]],
    mapping: Dict[str, Any]
) -> Dict[str, Any]:
    """Run comprehensive coverage analysis."""

    # Initialize counters
    total = 0
    covered = 0
    by_override = 0
    by_pattern = 0

    coverage_by_mcap = {}
    coverage_by_category = {}
    uncovered_tickers = []
    covered_tickers = []

    for company in universe:
        ticker = company.get("ticker", "UNKNOWN")
        market_cap = company.get("market_data", {}).get("market_cap")
        market_cap_mm = market_cap / 1_000_000 if market_cap else None
        conditions = company.get("clinical", {}).get("conditions", [])

        total += 1
        mcap_bucket = categorize_market_cap(market_cap_mm)

        # Initialize bucket tracking
        if mcap_bucket not in coverage_by_mcap:
            coverage_by_mcap[mcap_bucket] = {"total": 0, "covered": 0}
        coverage_by_mcap[mcap_bucket]["total"] += 1

        # Check coverage
        indication = get_ticker_indication(ticker, conditions, mapping)

        if indication:
            covered += 1
            coverage_by_mcap[mcap_bucket]["covered"] += 1

            # Track coverage source
            if ticker in mapping.get("ticker_overrides", {}):
                by_override += 1
            else:
                by_pattern += 1

            # Track by category
            if indication not in coverage_by_category:
                coverage_by_category[indication] = 0
            coverage_by_category[indication] += 1

            covered_tickers.append({
                "ticker": ticker,
                "indication": indication,
                "market_cap_mm": market_cap_mm,
                "conditions": conditions[:3] if conditions else [],
                "source": "override" if ticker in mapping.get("ticker_overrides", {}) else "pattern"
            })
        else:
            uncovered_tickers.append({
                "ticker": ticker,
                "market_cap_mm": market_cap_mm,
                "conditions": conditions[:3] if conditions else []
            })

    # Calculate coverage percentages
    coverage_pct = covered / total * 100 if total > 0 else 0

    for bucket in coverage_by_mcap:
        bucket_data = coverage_by_mcap[bucket]
        bucket_data["coverage_pct"] = (
            bucket_data["covered"] / bucket_data["total"] * 100
            if bucket_data["total"] > 0 else 0
        )

    return {
        "summary": {
            "total_tickers": total,
            "covered_tickers": covered,
            "uncovered_tickers": total - covered,
            "coverage_pct": round(coverage_pct, 1),
            "by_override": by_override,
            "by_pattern": by_pattern
        },
        "coverage_by_market_cap": coverage_by_mcap,
        "coverage_by_category": dict(sorted(coverage_by_category.items(), key=lambda x: -x[1])),
        "uncovered_sample": uncovered_tickers[:20],
        "covered_tickers": covered_tickers
    }


def run_false_positive_audit(
    covered_tickers: List[Dict[str, Any]],
    sample_size: int = 20,
    seed: int = 42  # Deterministic seed
) -> List[Dict[str, Any]]:
    """
    Sample 20 random tickers for manual verification.
    Uses deterministic seed for reproducibility.
    """
    random.seed(seed)

    if len(covered_tickers) < sample_size:
        sample = covered_tickers
    else:
        sample = random.sample(covered_tickers, sample_size)

    # Sort by ticker for easier review
    sample = sorted(sample, key=lambda x: x["ticker"])

    return sample


def print_report(analysis: Dict[str, Any], audit_sample: List[Dict[str, Any]]) -> None:
    """Print formatted validation report."""
    print("=" * 70)
    print("PoS COVERAGE VALIDATION REPORT")
    print("=" * 70)
    print()

    # Summary
    summary = analysis["summary"]
    print("COVERAGE SUMMARY")
    print("-" * 70)
    print(f"Total Tickers:     {summary['total_tickers']}")
    print(f"Covered Tickers:   {summary['covered_tickers']}")
    print(f"Uncovered:         {summary['uncovered_tickers']}")
    print(f"Coverage Rate:     {summary['coverage_pct']:.1f}%")
    print(f"  - By Override:   {summary['by_override']}")
    print(f"  - By Pattern:    {summary['by_pattern']}")
    print()

    # Coverage by market cap
    print("COVERAGE BY MARKET CAP BUCKET")
    print("-" * 70)
    print(f"{'Bucket':<25} {'Total':>8} {'Covered':>10} {'Coverage':>12}")
    print("-" * 70)

    # Sort buckets by market cap order
    bucket_order = [
        "nano (<$100M)", "micro ($100-300M)", "small ($300M-1B)",
        "mid ($1-5B)", "large ($5-20B)", "mega (>$20B)", "unknown"
    ]

    for bucket in bucket_order:
        if bucket in analysis["coverage_by_market_cap"]:
            data = analysis["coverage_by_market_cap"][bucket]
            print(f"{bucket:<25} {data['total']:>8} {data['covered']:>10} {data['coverage_pct']:>11.1f}%")
    print()

    # Coverage by category
    print("COVERAGE BY THERAPEUTIC AREA")
    print("-" * 70)
    for category, count in analysis["coverage_by_category"].items():
        print(f"  {category:<20} {count:>5}")
    print()

    # Uncovered sample
    print("UNCOVERED TICKERS SAMPLE (first 10)")
    print("-" * 70)
    for item in analysis["uncovered_sample"][:10]:
        conditions_str = ", ".join(item["conditions"][:2]) if item["conditions"] else "no conditions"
        mcap_str = f"${item['market_cap_mm']:.0f}M" if item['market_cap_mm'] else "N/A"
        print(f"  {item['ticker']:<8} mcap={mcap_str:<12} conditions: {conditions_str[:40]}")
    print()

    # False positive audit sample
    print("FALSE POSITIVE AUDIT SAMPLE (20 tickers for manual verification)")
    print("-" * 70)
    print(f"{'Ticker':<8} {'Indication':<18} {'Source':<10} {'Conditions (first 2)'}")
    print("-" * 70)
    for item in audit_sample:
        conditions_str = ", ".join(item["conditions"][:2])[:40] if item["conditions"] else "-"
        print(f"{item['ticker']:<8} {item['indication']:<18} {item['source']:<10} {conditions_str}")
    print()

    # Red flag check
    print("RED FLAG ANALYSIS")
    print("-" * 70)

    # Check if coverage drops sharply in small-cap
    mcap_data = analysis["coverage_by_market_cap"]
    small_coverage = mcap_data.get("small ($300M-1B)", {}).get("coverage_pct", 0)
    mid_coverage = mcap_data.get("mid ($1-5B)", {}).get("coverage_pct", 0)
    large_coverage = mcap_data.get("large ($5-20B)", {}).get("coverage_pct", 0)

    if small_coverage < mid_coverage - 20 or small_coverage < large_coverage - 20:
        print("⚠️  WARNING: Coverage drops significantly for small-cap tickers")
        print(f"   Small: {small_coverage:.1f}% vs Mid: {mid_coverage:.1f}% vs Large: {large_coverage:.1f}%")
        print("   This may indicate bias toward liquid names")
    else:
        print("✓ Coverage is relatively even across market cap buckets")

    print()
    print("=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)


def main():
    """Main entry point."""
    print("Loading data...")

    mapping = load_indication_mapping()
    universe = load_universe()

    print(f"Loaded {len(universe)} tickers from universe")
    print(f"Loaded {len(mapping.get('ticker_overrides', {}))} ticker overrides")
    print()

    print("Running coverage analysis...")
    analysis = run_coverage_analysis(universe, mapping)

    print("Running false positive audit...")
    audit_sample = run_false_positive_audit(analysis["covered_tickers"])

    print()
    print_report(analysis, audit_sample)

    # Save detailed results for later review
    output_path = Path(__file__).parent.parent / "production_data" / "pos_coverage_validation.json"
    with open(output_path, "w") as f:
        # Remove covered_tickers from output (too large)
        output = {k: v for k, v in analysis.items() if k != "covered_tickers"}
        output["audit_sample"] = audit_sample
        json.dump(output, f, indent=2, default=str)

    print(f"\nDetailed results saved to: {output_path}")

    return analysis, audit_sample


if __name__ == "__main__":
    main()
