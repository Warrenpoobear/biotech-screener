#!/usr/bin/env python3
"""Repair split artifacts in price_history.csv.

Detects single-day price discontinuities (reverse/forward splits) and
retroactively adjusts all *prior* prices by the split factor so that
returns computed across the split boundary are continuous.

Only discontinuities CONFIRMED by a split in ``corporate_actions.json`` (same
ticker, within ``CONFIRM_WINDOW_DAYS``) are applied. Biotech names routinely
move -75%/+300% in a day on trial readouts; unconfirmed jumps are real price
moves and are reported, not adjusted (EYPT 2026-10-19 and ABOS 2026-10-30
crashes were previously smoothed away as "forward splits"). If the registry is
missing or unreadable, nothing is adjusted: a price gap alone is not a split.

The adjustment is cumulative: if a ticker has two splits, all prices
before the first split are multiplied by both factors.

Registry ``special_dividend`` entries are back-adjusted too (factor
``(close_before - amount) / close_before`` on prices before the ex-date), but
only when the raw series visibly drops by roughly the amount, mirroring
``run_screen``'s momentum adjustment.

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
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

JUMP_THRESHOLD = 3.0  # +300% single-day → reverse split
DROP_THRESHOLD = -0.75  # -75% single-day → forward split

# A detected jump confirms against a registry split within this many calendar
# days (vendor and registry dates can differ by a day or two).
CONFIRM_WINDOW_DAYS = 5

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
                splits.setdefault(ticker, []).append(
                    {
                        "date": sorted_dates[i],
                        "factor": factor,
                        "pct_change": round(pct, 6),
                        "flag_type": "reverse_split" if pct > 0 else "forward_split",
                    }
                )
    # Sort events per ticker by date ascending
    for ticker in splits:
        splits[ticker].sort(key=lambda e: e["date"])
    return splits


def filter_confirmed_splits(
    splits: Dict[str, List[Dict[str, Any]]],
    registry: Any,
    window_days: int = CONFIRM_WINDOW_DAYS,
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, List[Dict[str, Any]]]]:
    """Split detected events into (confirmed, unconfirmed) by the registry.

    An event is confirmed when the registry has a forward/reverse split for the
    same ticker whose effective_date is within ``window_days`` of the event.
    Confirmed events keep their observed factor (unchanged legacy behaviour).
    """
    from common.corporate_actions import get_splits_only

    confirmed: Dict[str, List[Dict[str, Any]]] = {}
    unconfirmed: Dict[str, List[Dict[str, Any]]] = {}
    for ticker, events in splits.items():
        reg_dates = []
        for a in get_splits_only(ticker, registry):
            try:
                reg_dates.append(date.fromisoformat(a.effective_date))
            except (TypeError, ValueError):
                continue
        for e in events:
            ok = any(abs((rd - e["date"]).days) <= window_days for rd in reg_dates)
            (confirmed if ok else unconfirmed).setdefault(ticker, []).append(e)
    return confirmed, unconfirmed


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


def apply_special_dividends(
    raw_prices: Dict[str, Dict[date, Decimal]],
    adjusted: Dict[str, Dict[date, Decimal]],
    registry: Any,
) -> List[Dict[str, Any]]:
    """Back-adjust registry special cash dividends in place; return what was applied.

    The ex-date drop of a special dividend is a distribution, not a return. For
    each registry ``special_dividend`` the factor ``(close_before - amount) /
    close_before`` (from RAW closes) multiplies every adjusted price strictly
    before the ex-date. Same visibility rule as ``run_screen``: applied only when
    the raw series drops by 0.5x..1.5x the amount across the ex-date, so a
    dividend-adjusted source is never adjusted twice.
    """
    from common.corporate_actions import get_special_dividends

    applied: List[Dict[str, Any]] = []
    for ticker, raw in raw_prices.items():
        if ticker not in adjusted:
            continue
        for a in get_special_dividends(ticker, registry):
            try:
                ex = date.fromisoformat(a.effective_date)
                amount = Decimal(str(a.amount)) if a.amount else None
            except (TypeError, ValueError, ArithmeticError):
                continue
            if not amount or amount <= 0:
                continue
            dates = sorted(raw)
            before = next((raw[d] for d in reversed(dates) if d < ex), None)
            after = next((raw[d] for d in dates if d >= ex), None)
            if not before or not after or amount >= before:
                continue
            drop = before - after
            if not (amount / 2 <= drop <= amount * Decimal("1.5")):
                continue
            factor = (before - amount) / before
            adj = adjusted[ticker]
            for d in adj:
                if d < ex:
                    adj[d] = adj[d] * factor
            applied.append(
                {
                    "ticker": ticker,
                    "ex_date": ex.isoformat(),
                    "amount": float(amount),
                    "factor": round(float(factor), 6),
                }
            )
    return applied


# ---------------------------------------------------------------------------
# High-level orchestrator
# ---------------------------------------------------------------------------


def repair_price_history(
    prices_path: Path,
    output_path: Path,
    jump_threshold: float = JUMP_THRESHOLD,
    drop_threshold: float = DROP_THRESHOLD,
    registry_path: Path | None = None,
) -> Dict[str, Any]:
    """Full repair pipeline.

    ``registry_path`` defaults to ``corporate_actions.json`` next to the input
    prices. Returns summary dict with statistics.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from backtest.returns_provider import CSVReturnsProvider

    # Load prices
    provider = CSVReturnsProvider(prices_path, price_col="close")
    prices = provider._prices

    # Detect candidate splits, then keep only registry-confirmed ones
    detected = detect_splits(prices, jump_threshold, drop_threshold)
    registry_path = registry_path or (Path(prices_path).parent / "corporate_actions.json")
    unconfirmed: Dict[str, List[Dict[str, Any]]] = {}
    registry = None
    try:
        from common.corporate_actions import load_actions

        if not Path(registry_path).exists():
            raise FileNotFoundError(registry_path)
        registry = load_actions(Path(registry_path))
        splits, unconfirmed = filter_confirmed_splits(detected, registry)
        mode = "registry_confirmed"
    except Exception as exc:
        # A gap is not evidence of a split. EYPT and ABOS crashes were
        # adjusted when this path fell back to the price-gap heuristic.
        print(f"WARNING: corporate-actions registry unavailable ({exc}); not adjusting any price gaps")
        splits = {}
        unconfirmed = detected
        mode = "registry_missing"

    total_events = sum(len(events) for events in splits.values())
    tickers_affected = sorted(splits.keys())

    # Compute adjusted prices
    adjusted = compute_adjusted_prices(prices, splits)
    dividends_applied = apply_special_dividends(prices, adjusted, registry) if registry is not None else []

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
        "split_mode": mode,
        "special_dividends_applied": dividends_applied,
        "unconfirmed_not_applied": {
            ticker: [{"date": e["date"].isoformat(), "pct_change": e["pct_change"]} for e in events]
            for ticker, events in unconfirmed.items()
        },
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
        "--jump-threshold",
        type=float,
        default=JUMP_THRESHOLD,
        help=f"Reverse split detection threshold (default: {JUMP_THRESHOLD})",
    )
    parser.add_argument(
        "--drop-threshold",
        type=float,
        default=DROP_THRESHOLD,
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
        print("\nSplit events:")
        for ticker, events in sorted(result["split_details"].items()):
            for e in events:
                print(
                    f"  {ticker:6s}  {e['date']}  {e['flag_type']:14s}  "
                    f"factor={e['factor']:.4f}  ({e['pct_change']:+.2%})"
                )

    if result["special_dividends_applied"]:
        print("\nSpecial dividends back-adjusted:")
        for d in result["special_dividends_applied"]:
            print(f"  {d['ticker']:6s}  ex {d['ex_date']}  ${d['amount']:.2f}  factor={d['factor']:.4f}")


if __name__ == "__main__":
    main()
