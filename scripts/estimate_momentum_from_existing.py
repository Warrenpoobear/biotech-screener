#!/usr/bin/env python3
"""
estimate_momentum_from_existing.py - Estimate 60-day momentum from existing market_data fields

This script creates the momentum fields required by Module 5 using existing data in
market_data.json. It estimates:
- return_60d: Derived from returns_3m (63 trading days ≈ 60 days)
- xbi_return_60d: Estimated XBI benchmark return (from reference or median of dataset)
- volatility_252d: Scaled from volatility_90d

This is a fallback approach when live API fetching is not available.
For production use, run the full data collection pipeline instead.

Usage:
    python scripts/estimate_momentum_from_existing.py \
        --market-data production_data/market_data.json \
        --output production_data/market_data.json \
        --xbi-return -0.05

Point-in-Time Note:
    This script preserves the collected_at date from the original data and does not
    introduce lookahead bias since it only uses data that was already in market_data.json.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import sys


def estimate_xbi_return_from_data(market_data: List[Dict]) -> float:
    """
    Estimate XBI return from the median of individual stock returns.

    This is a rough approximation - in production, fetch XBI directly.
    XBI typically correlates with biotech stocks but has less volatility.

    Args:
        market_data: List of market data records with returns_3m

    Returns:
        Estimated XBI 60-day return
    """
    returns = []
    for record in market_data:
        ret_3m = record.get("returns_3m")
        if ret_3m is not None:
            returns.append(ret_3m)

    if not returns:
        return 0.0

    # Use median and dampen by 0.7 (XBI is less volatile than average stock)
    returns.sort()
    median = returns[len(returns) // 2]
    return median * 0.7


def enrich_with_momentum(
    market_data: List[Dict],
    xbi_return_override: Optional[float] = None,
) -> List[Dict]:
    """
    Enrich market data with estimated momentum fields.

    Args:
        market_data: List of market data records
        xbi_return_override: Optional explicit XBI return to use

    Returns:
        Enriched market data
    """
    # Determine XBI return
    if xbi_return_override is not None:
        xbi_return_60d = xbi_return_override
    else:
        xbi_return_60d = estimate_xbi_return_from_data(market_data)

    print(f"Using XBI 60-day return: {xbi_return_60d:.4f} ({xbi_return_60d*100:.2f}%)")

    enriched_count = 0

    for record in market_data:
        # Estimate return_60d from returns_3m (63 trading days ≈ 60 days)
        returns_3m = record.get("returns_3m")
        if returns_3m is not None:
            # Use returns_3m directly as return_60d approximation
            record["return_60d"] = round(returns_3m, 6)
            record["xbi_return_60d"] = round(xbi_return_60d, 6)
            record["benchmark_return_60d"] = round(xbi_return_60d, 6)

            # Calculate alpha (excess return vs benchmark)
            record["alpha_60d"] = round(returns_3m - xbi_return_60d, 6)

            enriched_count += 1
        elif record.get("returns_1m") is not None:
            # Fallback: extrapolate from 1-month return
            # Assume 3-month return is roughly 3x the monthly volatility
            returns_1m = record["returns_1m"]
            estimated_3m = returns_1m * 1.5  # Conservative estimate
            record["return_60d"] = round(estimated_3m, 6)
            record["xbi_return_60d"] = round(xbi_return_60d, 6)
            record["benchmark_return_60d"] = round(xbi_return_60d, 6)
            record["alpha_60d"] = round(estimated_3m - xbi_return_60d, 6)
            record["return_60d_estimated"] = True  # Flag as estimated
            enriched_count += 1

        # Estimate volatility_252d from volatility_90d
        # Annualized vol is relatively stable, so we can use 90d as proxy
        vol_90d = record.get("volatility_90d")
        if vol_90d is not None:
            # 90-day vol is already annualized, just use it
            record["volatility_252d"] = round(vol_90d, 6)
            record["annualized_volatility"] = round(vol_90d, 6)

    return market_data, enriched_count


def main():
    parser = argparse.ArgumentParser(
        description="Estimate 60-day momentum from existing market_data fields"
    )
    parser.add_argument(
        "--market-data",
        type=Path,
        required=True,
        help="Path to market_data.json"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path (defaults to overwriting input)"
    )
    parser.add_argument(
        "--xbi-return",
        type=float,
        help="Override XBI 60-day return (e.g., -0.05 for -5%%)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without making changes"
    )

    args = parser.parse_args()

    # Load existing market data
    if not args.market_data.exists():
        print(f"Error: {args.market_data} not found")
        sys.exit(1)

    print(f"Loading {args.market_data}...")
    with open(args.market_data) as f:
        market_data = json.load(f)

    if not isinstance(market_data, list):
        print("Error: market_data.json must be a list of records")
        sys.exit(1)

    print(f"Loaded {len(market_data)} records")

    # Check current state
    has_return_60d = sum(1 for r in market_data if r.get("return_60d") is not None)
    has_returns_3m = sum(1 for r in market_data if r.get("returns_3m") is not None)
    has_returns_1m = sum(1 for r in market_data if r.get("returns_1m") is not None)
    has_vol_90d = sum(1 for r in market_data if r.get("volatility_90d") is not None)

    print(f"\nCurrent state:")
    print(f"  With return_60d: {has_return_60d}/{len(market_data)}")
    print(f"  With returns_3m (source): {has_returns_3m}/{len(market_data)}")
    print(f"  With returns_1m (fallback): {has_returns_1m}/{len(market_data)}")
    print(f"  With volatility_90d: {has_vol_90d}/{len(market_data)}")

    if args.dry_run:
        print("\n[DRY RUN] Would estimate momentum from existing fields")

        # Show a few sample records
        sample_count = 0
        print("\nSample records that would be enriched:")
        for r in market_data[:10]:
            if r.get("returns_3m") is not None:
                print(f"  {r['ticker']}: returns_3m={r['returns_3m']:.4f}")
                sample_count += 1
                if sample_count >= 3:
                    break

        sys.exit(0)

    # Enrich the data
    print("\n" + "="*60)
    enriched, count = enrich_with_momentum(market_data, args.xbi_return)
    print("="*60)

    # Report final state
    has_return_60d = sum(1 for r in enriched if r.get("return_60d") is not None)
    has_xbi = sum(1 for r in enriched if r.get("xbi_return_60d") is not None)
    has_vol_252d = sum(1 for r in enriched if r.get("volatility_252d") is not None)

    print(f"\nFinal state:")
    print(f"  With return_60d: {has_return_60d}/{len(enriched)}")
    print(f"  With xbi_return_60d: {has_xbi}/{len(enriched)}")
    print(f"  With volatility_252d: {has_vol_252d}/{len(enriched)}")
    print(f"\nEnriched {count} records")

    # Write output
    output_path = args.output or args.market_data
    print(f"\nWriting to {output_path}...")

    with open(output_path, "w") as f:
        json.dump(enriched, f, indent=2, sort_keys=False)

    print("Done!")

    # Print sample records
    print("\nSample enriched records:")
    sample_count = 0
    for r in enriched:
        if r.get("return_60d") is not None:
            alpha = r.get("alpha_60d", "N/A")
            vol = r.get("volatility_252d")
            est = " (estimated)" if r.get("return_60d_estimated") else ""
            print(f"  {r['ticker']}: return_60d={r['return_60d']:.4f}{est}, "
                  f"alpha={alpha:.4f if isinstance(alpha, float) else alpha}, "
                  f"vol={vol:.4f if vol else 'N/A'}")
            sample_count += 1
            if sample_count >= 5:
                break


if __name__ == "__main__":
    main()
