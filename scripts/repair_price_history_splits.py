#!/usr/bin/env python3
"""Repair split artifacts in price_history.csv.

Detects single-day price discontinuities (reverse/forward splits) and
retroactively adjusts all *prior* prices by the split factor so that
returns computed across the split boundary are continuous.

The adjustment is cumulative: if a ticker has two splits, all prices
before the first split are multiplied by both factors.

CLI:
  python scripts/repair_price_history_splits.py \
    [--prices production_data/price_history.csv] \
    [--out production_data/price_history_split_adj.csv] \
    [--jump-threshold 3.0] \
    [--drop-threshold -0.75]
"""
from __future__ import annotations

import argparse
import csv
import sys
from datetime import date
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

JUMP_THRESHOLD = 3.0     # +300% single-day → reverse split
DROP_THRESHOLD = -0.75   # -75% single-day → forward split

# Quantization for output prices (6 decimal places)
PRICE_QUANT = Decimal("0.000001")


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def detect_splits(
    prices: Dict[str, Dict[date, Decimal]],
    jump_threshold: float = JUMP_THRESHOLD,
    drop_threshold: float = DROP_THRESHOLD,
) -> Dict[str, List[Dict[str, Any]]]:
    """Detect split events per ticker.

    Returns {ticker: [{date, factor, pct_change, flag_type}, ...]}
    sorted by date ascending within each ticker.

    The factor is curr/prev — the ratio by which the price jumped.
    To make pre-split prices continuous, multiply them by this factor.
    """
    splits: Dict[str, List[Dict[str, Any]]] = {}
    for ticker, ticker_prices in prices.items():
        sorted_dates = sorted(ticker_prices.keys())
        for i in range(1, len(sorted_dates)):
            prev = ticker_prices[sorted_dates[i - 1]]
            curr = ticker_prices[sorted_dates[i]]
            if prev <= 0:
                continue
            pct = float(curr / prev) - 1.0
            if pct >= jump_threshold or pct <= drop_threshold:
                factor = curr / prev  # Decimal
                splits.setdefault(ticker, []).append({
                    "date": sorted_dates[i],
                    "factor": factor,
                    "pct_change": round(pct, 6),
                    "flag_type": "reverse_split" if pct > 0 else "forward_split",
                })
    # Sort events per ticker by date ascending
    for ticker in splits:
        splits[ticker].sort(key=lambda e: e["date"])
    return splits


def compute_adjusted_prices(
    prices: Dict[str, Dict[date, Decimal]],
    splits: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Dict[date, Decimal]]:
    """Compute split-adjusted prices.

    For each ticker with splits, multiply all prices *before* each split
    date by the split factor. Factors are cumulative (applied in reverse
    chronological order so that the most recent prices remain unchanged).
    """
    adjusted: Dict[str, Dict[date, Decimal]] = {}

    for ticker, ticker_prices in prices.items():
        if ticker not in splits:
            # No adjustment needed — copy as-is
            adjusted[ticker] = dict(ticker_prices)
            continue

        events = splits[ticker]
        # Work with sorted dates
        sorted_dates = sorted(ticker_prices.keys())
        adj = {d: ticker_prices[d] for d in sorted_dates}

        # Apply each split: multiply all prices before the split date by factor.
        # Process splits from earliest to latest. Each split's factor applies
        # to all dates strictly before it.
        #
        # For cumulative correctness: process from latest to earliest so that
        # a later split doesn't double-adjust dates already adjusted by an
        # earlier split. Actually, the correct approach:
        #
        # Each split says: on split_date, price jumped by factor.
        # To make the series continuous, multiply all prices < split_date by factor.
        # If there are multiple splits, process them from latest to earliest,
        # so each adjustment cascades to all prior dates.
        for event in reversed(events):
            split_date = event["date"]
            factor = event["factor"]
            for d in sorted_dates:
                if d < split_date:
                    adj[d] = adj[d] * factor
                else:
                    break  # sorted, so no more dates < split_date

        adjusted[ticker] = adj

    return adjusted


def write_adjusted_csv(
    adjusted: Dict[str, Dict[date, Decimal]],
    output_path: Path,
) -> int:
    """Write adjusted prices to CSV.

    Returns the number of rows written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect all rows, sorted by (ticker, date)
    rows: List[Tuple[str, date, Decimal]] = []
    for ticker in sorted(adjusted.keys()):
        for d in sorted(adjusted[ticker].keys()):
            rows.append((ticker, d, adjusted[ticker][d]))

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ticker", "date", "close"])
        for ticker, d, price in rows:
            writer.writerow([ticker, d.isoformat(), str(price.quantize(PRICE_QUANT, rounding=ROUND_HALF_UP))])

    return len(rows)


# ---------------------------------------------------------------------------
# High-level orchestrator
# ---------------------------------------------------------------------------

def repair_price_history(
    prices_path: Path,
    output_path: Path,
    jump_threshold: float = JUMP_THRESHOLD,
    drop_threshold: float = DROP_THRESHOLD,
) -> Dict[str, Any]:
    """Full repair pipeline.

    Returns summary dict with statistics.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from backtest.returns_provider import CSVReturnsProvider

    # Load prices
    provider = CSVReturnsProvider(prices_path, price_col="close")
    prices = provider._prices

    # Detect splits
    splits = detect_splits(prices, jump_threshold, drop_threshold)

    total_events = sum(len(events) for events in splits.values())
    tickers_affected = sorted(splits.keys())

    # Compute adjusted prices
    adjusted = compute_adjusted_prices(prices, splits)

    # Write output
    n_rows = write_adjusted_csv(adjusted, output_path)

    return {
        "input_path": str(prices_path),
        "output_path": str(output_path),
        "tickers_in_input": len(prices),
        "tickers_with_splits": len(tickers_affected),
        "total_split_events": total_events,
        "tickers_affected": tickers_affected,
        "split_details": {
            ticker: [
                {
                    "date": e["date"].isoformat(),
                    "factor": round(float(e["factor"]), 6),
                    "pct_change": e["pct_change"],
                    "flag_type": e["flag_type"],
                }
                for e in events
            ]
            for ticker, events in splits.items()
        },
        "rows_written": n_rows,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Repair split artifacts in price_history.csv")
    parser.add_argument(
        "--prices",
        default="production_data/price_history.csv",
        help="Input price_history.csv path",
    )
    parser.add_argument(
        "--out",
        default="production_data/price_history_split_adj.csv",
        help="Output split-adjusted CSV path",
    )
    parser.add_argument(
        "--jump-threshold", type=float, default=JUMP_THRESHOLD,
        help=f"Reverse split detection threshold (default: {JUMP_THRESHOLD})",
    )
    parser.add_argument(
        "--drop-threshold", type=float, default=DROP_THRESHOLD,
        help=f"Forward split detection threshold (default: {DROP_THRESHOLD})",
    )
    args = parser.parse_args()

    result = repair_price_history(
        prices_path=Path(args.prices),
        output_path=Path(args.out),
        jump_threshold=args.jump_threshold,
        drop_threshold=args.drop_threshold,
    )

    print(f"Split-adjusted price history written to: {result['output_path']}")
    print(f"  Input tickers:     {result['tickers_in_input']}")
    print(f"  Tickers with splits: {result['tickers_with_splits']}")
    print(f"  Total split events:  {result['total_split_events']}")
    print(f"  Rows written:        {result['rows_written']}")

    if result["split_details"]:
        print(f"\nSplit events:")
        for ticker, events in sorted(result["split_details"].items()):
            for e in events:
                print(f"  {ticker:6s}  {e['date']}  {e['flag_type']:14s}  "
                      f"factor={e['factor']:.4f}  ({e['pct_change']:+.2%})")


if __name__ == "__main__":
    main()
