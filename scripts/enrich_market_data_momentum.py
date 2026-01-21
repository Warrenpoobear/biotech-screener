#!/usr/bin/env python3
"""
enrich_market_data_momentum.py - Add multi-window momentum fields to market_data.json

This script enriches the existing market_data.json with momentum fields required
for Module 5 scoring. Supports multiple lookback windows with fallback for
improved coverage.

FIELDS ADDED:
    - return_20d: 20 trading day compounded return (P_t / P_{t-20} - 1)
    - return_60d: 60 trading day compounded return (P_t / P_{t-60} - 1)
    - return_120d: 120 trading day compounded return (P_t / P_{t-120} - 1)
    - xbi_return_20d: XBI benchmark 20-day return
    - xbi_return_60d: XBI benchmark 60-day return
    - xbi_return_120d: XBI benchmark 120-day return
    - volatility_252d: Annualized 252-day volatility
    - trading_days_available: Number of trading days with data

The fields are calculated using yfinance historical price data.

COVERAGE IMPROVEMENT:
    - Multiple windows allow fallback (60d -> 120d -> 20d)
    - Tickers with limited history can still get momentum scores
    - Diagnostic tracking shows coverage by window

DETERMINISM & PIT-SAFETY:
    - Output JSON uses sort_keys=True for reproducible byte hashes
    - Writes to temp file first, then validates before replacing original
    - Includes enrichment metadata for audit trail
    - Benchmark symbol (XBI) and price field (Close) are explicit
    - Only uses prices <= as_of_date (no lookahead)
    - Clips extreme returns (> ±300%) to prevent outlier distortion

Usage:
    python scripts/enrich_market_data_momentum.py \\
        --market-data production_data/market_data.json \\
        --output production_data/market_data.json

Point-in-Time Safety:
    - Uses the collected_at date from each record as the reference date
    - Ensures consistent lookback windows across all tickers
"""

import json
import argparse
import hashlib
import os
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import sys


# =============================================================================
# CONSTANTS FOR DETERMINISM AND SAFETY
# =============================================================================

ENRICHMENT_VERSION = "1.1.0"  # Increment on logic changes
BENCHMARK_SYMBOL = "XBI"
PRICE_FIELD = "Close"  # Explicit price field for audit
MAX_RETURN_CLIP = 3.0  # Clip returns > ±300% to prevent outlier distortion
WINDOWS = [20, 60, 120]  # Trading days


def fetch_historical_prices(
    ticker: str,
    end_date: datetime,
    lookback_days: int = 300,  # Fetch extra days to ensure we have 252 trading days
) -> Optional[List[Dict]]:
    """
    Fetch historical daily prices for a ticker.

    Returns list of {date, close} dicts sorted by date ascending.
    """
    try:
        import yfinance as yf

        start_date = end_date - timedelta(days=lookback_days)
        stock = yf.Ticker(ticker)
        hist = stock.history(start=start_date, end=end_date)

        if hist.empty:
            return None

        prices = []
        for idx, row in hist.iterrows():
            prices.append({
                "date": idx.strftime("%Y-%m-%d"),
                "close": float(row["Close"])
            })

        return prices

    except Exception as e:
        print(f"  Warning: Failed to fetch {ticker}: {e}")
        return None


def calculate_return(
    prices: List[Dict],
    lookback_trading_days: int,
    clip_threshold: float = MAX_RETURN_CLIP,
) -> Tuple[Optional[float], bool]:
    """
    Calculate compounded return over N trading days with outlier clipping.

    Formula: (P_t / P_{t-N}) - 1, clipped to ±clip_threshold

    Args:
        prices: List of {date, close} dicts sorted by date ascending
        lookback_trading_days: Number of trading days to look back (e.g., 60)
        clip_threshold: Maximum absolute return value (default 3.0 = ±300%)

    Returns:
        Tuple of (return, was_clipped):
        - Compounded return as decimal (e.g., 0.15 = 15%), or None if insufficient data
        - Boolean indicating if return was clipped (for flagging)
    """
    if not prices or len(prices) < lookback_trading_days + 1:
        return None, False

    # Get the most recent price and the price N trading days ago
    current_price = prices[-1]["close"]
    past_price = prices[-(lookback_trading_days + 1)]["close"]

    if past_price <= 0:
        return None, False

    raw_return = (current_price / past_price) - 1

    # Clip extreme returns to prevent outlier distortion
    was_clipped = abs(raw_return) > clip_threshold
    if was_clipped:
        clipped_return = max(-clip_threshold, min(clip_threshold, raw_return))
        return clipped_return, True

    return raw_return, False


def calculate_volatility(prices: List[Dict], lookback_trading_days: int = 252) -> Optional[float]:
    """
    Calculate annualized volatility over N trading days.

    Formula: std(daily_returns) * sqrt(252)

    Args:
        prices: List of {date, close} dicts sorted by date ascending
        lookback_trading_days: Number of trading days (default 252 for annual)

    Returns:
        Annualized volatility as decimal (e.g., 0.50 = 50%)
    """
    if not prices or len(prices) < lookback_trading_days + 1:
        return None

    # Use the most recent N trading days
    recent_prices = prices[-(lookback_trading_days + 1):]

    # Calculate daily returns
    returns = []
    for i in range(1, len(recent_prices)):
        if recent_prices[i-1]["close"] > 0:
            daily_return = (recent_prices[i]["close"] / recent_prices[i-1]["close"]) - 1
            returns.append(daily_return)

    if len(returns) < 20:  # Need at least 20 data points
        return None

    # Calculate standard deviation
    mean_return = sum(returns) / len(returns)
    variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
    std_dev = variance ** 0.5

    # Annualize
    annualized_vol = std_dev * (252 ** 0.5)

    return annualized_vol


def enrich_market_data(
    market_data: List[Dict],
    reference_date: Optional[str] = None,
    rate_limit_delay: float = 0.2,
) -> Tuple[List[Dict], Dict[str, Any]]:
    """
    Enrich market data records with multi-window momentum fields.

    Computes returns for multiple windows (20d, 60d, 120d) to enable fallback
    when longer windows aren't available. This maximizes momentum coverage.

    Args:
        market_data: List of market data records
        reference_date: Optional override date (YYYY-MM-DD), otherwise uses collected_at
        rate_limit_delay: Delay between API calls in seconds

    Returns:
        Tuple of (enriched_data, enrichment_metadata):
        - enriched_data: List with momentum fields added
        - enrichment_metadata: Dict with version, stats, and PIT info for audit
    """
    import time

    # Determine reference date
    if reference_date:
        ref_date = datetime.strptime(reference_date, "%Y-%m-%d")
    else:
        # Use the most recent collected_at from the data
        collected_dates = [
            r.get("collected_at") for r in market_data
            if r.get("collected_at")
        ]
        if collected_dates:
            ref_date = datetime.strptime(max(collected_dates), "%Y-%m-%d")
        else:
            ref_date = datetime.now()

    print(f"Reference date: {ref_date.strftime('%Y-%m-%d')}")

    # Fetch XBI (benchmark) prices once - get enough for all windows
    print(f"Fetching {BENCHMARK_SYMBOL} benchmark data...")
    xbi_prices = fetch_historical_prices(BENCHMARK_SYMBOL, ref_date, lookback_days=400)

    xbi_returns = {}
    xbi_clipped_flags = {}
    if xbi_prices:
        for window in WINDOWS:
            ret, was_clipped = calculate_return(xbi_prices, window)
            if ret is not None:
                xbi_returns[window] = ret
                xbi_clipped_flags[window] = was_clipped
                clip_marker = " [CLIPPED]" if was_clipped else ""
                print(f"  {BENCHMARK_SYMBOL} {window}-day return: {ret:.4f} ({ret*100:.2f}%){clip_marker}")
            else:
                print(f"  Warning: Could not calculate {BENCHMARK_SYMBOL} {window}-day return")
    else:
        print(f"  Warning: Could not fetch {BENCHMARK_SYMBOL} prices")

    time.sleep(rate_limit_delay)

    # Track coverage statistics (granular for stable metrics)
    coverage_stats = {
        "total": len(market_data),
        "with_any_window": 0,
        "with_20d": 0,
        "with_60d": 0,
        "with_120d": 0,
        "with_volatility": 0,
        "missing_prices": 0,
        "returns_clipped": 0,  # Track outlier clipping
    }

    # Process each ticker
    enriched = []
    total = len(market_data)

    print(f"\nProcessing {total} tickers...")

    for i, record in enumerate(market_data):
        ticker = record.get("ticker")
        if not ticker:
            enriched.append(record)
            continue

        if (i + 1) % 50 == 0:
            print(f"  Progress: {i + 1}/{total} "
                  f"(60d: {coverage_stats['with_60d']}, "
                  f"any: {coverage_stats['with_any_window']})")

        # Fetch historical prices - get enough for longest window + volatility
        prices = fetch_historical_prices(ticker, ref_date, lookback_days=400)

        if not prices:
            coverage_stats["missing_prices"] += 1
            enriched.append(record)
            time.sleep(rate_limit_delay)
            continue

        # Track trading days available
        record["trading_days_available"] = len(prices)

        # Compute returns for each window
        has_any_window = False
        ticker_was_clipped = False
        for window in WINDOWS:
            ret, was_clipped = calculate_return(prices, window)
            if ret is not None:
                has_any_window = True
                record[f"return_{window}d"] = round(ret, 6)
                # Add benchmark return for this window
                if window in xbi_returns:
                    record[f"xbi_return_{window}d"] = round(xbi_returns[window], 6)
                    # Add alias for backwards compatibility
                    if window == 60:
                        record["benchmark_return_60d"] = record["xbi_return_60d"]
                coverage_stats[f"with_{window}d"] += 1
                # Track if clipped
                if was_clipped:
                    ticker_was_clipped = True
                    record[f"return_{window}d_clipped"] = True

        if has_any_window:
            coverage_stats["with_any_window"] += 1
        if ticker_was_clipped:
            coverage_stats["returns_clipped"] += 1

        # Compute volatility
        volatility_252d = calculate_volatility(prices, 252)
        if volatility_252d is not None:
            record["volatility_252d"] = round(volatility_252d, 6)
            record["annualized_volatility"] = record["volatility_252d"]  # Alias
            coverage_stats["with_volatility"] += 1

        enriched.append(record)
        time.sleep(rate_limit_delay)

    # Print coverage report
    print("\n" + "=" * 60)
    print("COVERAGE REPORT")
    print("=" * 60)
    print(f"Total tickers: {coverage_stats['total']}")
    print(f"With any momentum window: {coverage_stats['with_any_window']} "
          f"({100*coverage_stats['with_any_window']/max(1,coverage_stats['total']):.1f}%)")
    print(f"  - 20d window: {coverage_stats['with_20d']} "
          f"({100*coverage_stats['with_20d']/max(1,coverage_stats['total']):.1f}%)")
    print(f"  - 60d window: {coverage_stats['with_60d']} "
          f"({100*coverage_stats['with_60d']/max(1,coverage_stats['total']):.1f}%)")
    print(f"  - 120d window: {coverage_stats['with_120d']} "
          f"({100*coverage_stats['with_120d']/max(1,coverage_stats['total']):.1f}%)")
    print(f"With volatility_252d: {coverage_stats['with_volatility']} "
          f"({100*coverage_stats['with_volatility']/max(1,coverage_stats['total']):.1f}%)")
    print(f"Missing all prices: {coverage_stats['missing_prices']} "
          f"({100*coverage_stats['missing_prices']/max(1,coverage_stats['total']):.1f}%)")
    print(f"Returns clipped (outliers): {coverage_stats['returns_clipped']} "
          f"({100*coverage_stats['returns_clipped']/max(1,coverage_stats['total']):.1f}%)")
    print("=" * 60)

    # Build enrichment metadata for audit trail
    enrichment_metadata = {
        "momentum_enrichment_version": ENRICHMENT_VERSION,
        "as_of_date_used": ref_date.strftime("%Y-%m-%d"),
        "benchmark_symbol": BENCHMARK_SYMBOL,
        "price_field_used": PRICE_FIELD,
        "max_return_clip": MAX_RETURN_CLIP,
        "windows_computed": WINDOWS,
        "xbi_returns": {f"{k}d": round(v, 6) for k, v in xbi_returns.items()},
        "xbi_returns_clipped": {f"{k}d": v for k, v in xbi_clipped_flags.items() if v},
        "coverage_stats": coverage_stats,
        "enriched_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    return enriched, enrichment_metadata


def main():
    parser = argparse.ArgumentParser(
        description="Enrich market_data.json with 60-day momentum fields"
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
        "--reference-date",
        type=str,
        help="Reference date (YYYY-MM-DD), defaults to collected_at from data"
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=0.2,
        help="Delay between API calls in seconds (default: 0.2)"
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
    has_return_20d = sum(1 for r in market_data if r.get("return_20d") is not None)
    has_return_60d = sum(1 for r in market_data if r.get("return_60d") is not None)
    has_return_120d = sum(1 for r in market_data if r.get("return_120d") is not None)
    has_xbi = sum(1 for r in market_data if r.get("xbi_return_60d") is not None)
    has_vol_252d = sum(1 for r in market_data if r.get("volatility_252d") is not None)
    has_any = sum(1 for r in market_data if any(r.get(f"return_{w}d") is not None for w in [20, 60, 120]))

    print(f"\nCurrent state:")
    print(f"  With any momentum: {has_any}/{len(market_data)}")
    print(f"  With return_20d: {has_return_20d}/{len(market_data)}")
    print(f"  With return_60d: {has_return_60d}/{len(market_data)}")
    print(f"  With return_120d: {has_return_120d}/{len(market_data)}")
    print(f"  With xbi_return_60d: {has_xbi}/{len(market_data)}")
    print(f"  With volatility_252d: {has_vol_252d}/{len(market_data)}")

    if args.dry_run:
        print("\n[DRY RUN] Would enrich market data with momentum fields")
        sys.exit(0)

    # Check for yfinance
    try:
        import yfinance
        print(f"\nyfinance version: {yfinance.__version__}")
    except ImportError:
        print("\nError: yfinance not installed. Run: pip install yfinance")
        sys.exit(1)

    # Enrich the data
    print("\n" + "="*60)
    enriched, enrichment_metadata = enrich_market_data(
        market_data,
        reference_date=args.reference_date,
        rate_limit_delay=args.rate_limit,
    )
    print("="*60)

    # Report final state
    has_return_20d = sum(1 for r in enriched if r.get("return_20d") is not None)
    has_return_60d = sum(1 for r in enriched if r.get("return_60d") is not None)
    has_return_120d = sum(1 for r in enriched if r.get("return_120d") is not None)
    has_xbi = sum(1 for r in enriched if r.get("xbi_return_60d") is not None)
    has_vol_252d = sum(1 for r in enriched if r.get("volatility_252d") is not None)
    has_any = sum(1 for r in enriched if any(r.get(f"return_{w}d") is not None for w in [20, 60, 120]))

    print(f"\nFinal state:")
    print(f"  With any momentum: {has_any}/{len(enriched)}")
    print(f"  With return_20d: {has_return_20d}/{len(enriched)}")
    print(f"  With return_60d: {has_return_60d}/{len(enriched)}")
    print(f"  With return_120d: {has_return_120d}/{len(enriched)}")
    print(f"  With xbi_return_60d: {has_xbi}/{len(enriched)}")
    print(f"  With volatility_252d: {has_vol_252d}/{len(enriched)}")

    # Write output with temp file safety
    output_path = args.output or args.market_data
    print(f"\nWriting to {output_path}...")

    # SAFETY: Write to temp file first, validate, then replace
    output_dir = output_path.parent
    temp_fd, temp_path = tempfile.mkstemp(
        suffix=".tmp.json",
        prefix="market_data_",
        dir=output_dir,
    )

    try:
        # Write to temp file with sort_keys for determinism
        with os.fdopen(temp_fd, "w") as f:
            json.dump(enriched, f, indent=2, sort_keys=True)

        # Validate the temp file
        with open(temp_path) as f:
            validated = json.load(f)

        if len(validated) != len(enriched):
            raise ValueError(f"Validation failed: expected {len(enriched)} records, got {len(validated)}")

        # Compute hash for audit
        with open(temp_path, "rb") as f:
            content_hash = hashlib.sha256(f.read()).hexdigest()
        enrichment_metadata["output_hash"] = f"sha256:{content_hash}"

        # Replace the original file atomically
        os.replace(temp_path, output_path)
        print(f"  Output hash: sha256:{content_hash[:16]}...")

        # Write enrichment metadata alongside the data
        meta_path = output_path.with_suffix(".momentum_meta.json")
        with open(meta_path, "w") as f:
            json.dump(enrichment_metadata, f, indent=2, sort_keys=True)
        print(f"  Metadata written to: {meta_path}")

    except Exception as e:
        # Clean up temp file on error
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise RuntimeError(f"Failed to write output: {e}")

    print("Done!")


if __name__ == "__main__":
    main()
