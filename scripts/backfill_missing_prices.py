#!/usr/bin/env python3
"""Backfill missing price history for tickers in the universe.

Compares universe.json against price_history.csv, identifies tickers with
no price data, and fetches daily closes via yfinance.  Appends new rows
to the existing CSV.

WARNING: This is a manual dev-only tool. It is NOT called by CI or
phase2-daily. It uses yfinance which is not a production data source.
Run only for ad-hoc recovery of missing price data.

Modes:
  default        — fetch tickers completely absent from the CSV
  --refill-short — also re-fetch tickers with < MIN_BARS bars (replaces
                   their truncated rows with full history)

CLI:
  python scripts/backfill_missing_prices.py \
    [--universe production_data/universe.json] \
    [--price-history production_data/price_history.csv] \
    [--start-date 2020-01-01] \
    [--refill-short] \
    [--min-bars 126] \
    [--dry-run]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


def _load_universe_tickers(path: Path) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [
        t["ticker"] for t in data
        if not t["ticker"].startswith("_")
    ]


def _load_csv_bar_counts(path: Path) -> Dict[str, int]:
    """Return {ticker: bar_count} for all tickers in the CSV."""
    counts: Dict[str, int] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = (row.get("ticker") or "").strip().upper()
            if t:
                counts[t] = counts.get(t, 0) + 1
    return counts


def _fetch_prices(ticker: str, start_date: str, end_date: str) -> List[Dict[str, Any]]:
    """Fetch daily prices via yfinance. Returns list of row dicts."""
    import yfinance as yf

    tk = yf.Ticker(ticker)
    hist = tk.history(start=start_date, end=end_date, auto_adjust=False)
    if hist.empty:
        return []

    rows = []
    for idx, row in hist.iterrows():
        dt = idx.strftime("%Y-%m-%d")
        close = row.get("Close")
        if close is None or close != close:  # NaN check
            continue
        rows.append({
            "date": dt,
            "ticker": ticker,
            "close": str(close),
            "open": str(row["Open"]) if row.get("Open") == row.get("Open") else "",
            "high": str(row["High"]) if row.get("High") == row.get("High") else "",
            "low": str(row["Low"]) if row.get("Low") == row.get("Low") else "",
            "volume": str(int(row["Volume"])) if row.get("Volume") == row.get("Volume") else "",
        })
    return rows


def _remove_tickers_from_csv(price_path: Path, tickers_to_remove: Set[str]) -> int:
    """Remove all rows for given tickers from the CSV. Returns rows removed."""
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".csv", dir=price_path.parent)
    removed = 0
    with open(price_path, "r", encoding="utf-8") as fin, \
         open(tmp_fd, "w", newline="", encoding="utf-8") as fout:
        reader = csv.DictReader(fin)
        fieldnames = reader.fieldnames or []
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for row in reader:
            t = (row.get("ticker") or "").strip().upper()
            if t in tickers_to_remove:
                removed += 1
            else:
                writer.writerow(row)
    Path(tmp_path).replace(price_path)
    return removed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Backfill missing price history for universe tickers."
    )
    parser.add_argument(
        "--universe",
        default="production_data/universe.json",
        help="Path to universe.json",
    )
    parser.add_argument(
        "--price-history",
        default="production_data/price_history.csv",
        help="Path to price_history.csv",
    )
    parser.add_argument(
        "--start-date",
        default="2020-01-01",
        help="Earliest date to fetch (default: 2020-01-01)",
    )
    parser.add_argument(
        "--refill-short",
        action="store_true",
        help="Also re-fetch tickers with < min-bars bars",
    )
    parser.add_argument(
        "--min-bars",
        type=int,
        default=126,
        help="Minimum bars threshold for --refill-short (default: 126)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List targets without fetching",
    )
    args = parser.parse_args()

    universe_path = Path(args.universe)
    price_path = Path(args.price_history)

    if not universe_path.exists():
        print(f"ERROR: universe not found: {universe_path}", file=sys.stderr)
        return 1
    if not price_path.exists():
        print(f"ERROR: price_history not found: {price_path}", file=sys.stderr)
        return 1

    universe_tickers = _load_universe_tickers(universe_path)
    bar_counts = _load_csv_bar_counts(price_path)
    csv_tickers = set(bar_counts.keys())

    # --- Identify targets ---
    missing = [t for t in universe_tickers if t not in csv_tickers]
    short: List[Tuple[str, int]] = []
    if args.refill_short:
        for t in universe_tickers:
            bars = bar_counts.get(t, 0)
            if 0 < bars < args.min_bars:
                short.append((t, bars))

    print(f"Universe: {len(universe_tickers)} tickers")
    print(f"Price CSV: {len(csv_tickers)} tickers")
    print(f"Missing (no rows): {len(missing)}")
    if args.refill_short:
        print(f"Short series (< {args.min_bars} bars): {len(short)}")

    if missing:
        print(f"\nMissing tickers:")
        for t in sorted(missing):
            print(f"  {t}")

    if short:
        print(f"\nShort-series tickers:")
        for t, bars in sorted(short):
            print(f"  {t}: {bars} bars")

    all_targets = sorted(set(missing) | {t for t, _ in short})
    if not all_targets:
        print("\nNothing to backfill.")
        return 0

    print(f"\nTotal targets: {len(all_targets)}")

    if args.dry_run:
        print("\n(dry run — no data fetched)")
        return 0

    # --- Remove truncated rows for short-series tickers ---
    short_set = {t for t, _ in short}
    if short_set:
        removed = _remove_tickers_from_csv(price_path, short_set)
        print(f"\nRemoved {removed} truncated rows for {len(short_set)} short-series tickers")

    # --- Fetch ---
    end_date = date.today().isoformat()
    total_rows = 0
    fetched: List[Dict[str, Any]] = []
    failed: List[str] = []

    for ticker in all_targets:
        was_short = ticker in short_set
        label = f"(was {bar_counts.get(ticker, 0)} bars)" if was_short else "(new)"
        print(f"\nFetching {ticker} {label} ...", end=" ", flush=True)
        try:
            rows = _fetch_prices(ticker, args.start_date, end_date)
        except Exception as e:
            print(f"FAILED ({e})")
            failed.append(ticker)
            continue

        if not rows:
            print(f"no data returned")
            failed.append(ticker)
            continue

        print(f"{len(rows)} bars ({rows[0]['date']} to {rows[-1]['date']})")
        fetched.extend(rows)
        total_rows += len(rows)

    if fetched:
        fieldnames = ["date", "ticker", "close", "open", "high", "low", "volume"]
        with open(price_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            for row in fetched:
                writer.writerow(row)
        print(f"\nAppended {total_rows} rows to {price_path}")
    else:
        print("\nNo data fetched.")

    if failed:
        print(f"Failed/empty: {', '.join(failed)}")

    # --- Summary ---
    new_counts = _load_csv_bar_counts(price_path)
    improved = 0
    for t in all_targets:
        old = bar_counts.get(t, 0)
        new = new_counts.get(t, 0)
        if new > old:
            improved += 1
    print(f"\nImproved: {improved}/{len(all_targets)} tickers")

    return 0


if __name__ == "__main__":
    sys.exit(main())
