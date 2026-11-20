#!/usr/bin/env python3
"""
build_ipo_dates.py - Build IPO date lookup from price_history.csv

For each ticker, finds the earliest date with a price entry (proxy for IPO/listing date)
and the latest date (proxy for delist date). Saves to production_data/ipo_dates.json.

Used by run_screen.py to enforce PIT survivorship filtering:
- Exclude tickers that hadn't IPO'd yet as of the snapshot date
- Exclude tickers that delisted before the snapshot date

Safeguards (added 2026-11-12 after the PR #606 regeneration):
- The result is merged with the existing file. A ticker's first_price_date
  never moves later and its last_price_date never moves earlier, and a ticker
  missing from the price file is kept. A short or partial price history can
  therefore no longer erase listing dates.
- A price history where most tickers share one first date later than the
  research history floor (2020-01-31, the first historical snapshot) is
  treated as truncated and refused (exit 2) unless --allow-truncated is
  passed. A shared first date at or before the floor is expected: the price
  file itself starts on 2020-01-02 for most names.
- The output is written atomically.

Usage:
    python3 tools/build_ipo_dates.py
    python3 tools/build_ipo_dates.py --price-history PATH --output PATH
    python3 tools/build_ipo_dates.py --no-merge --allow-truncated   # deliberate rebuild
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# When more than this share of tickers have the same first date, that date is
# a fetch-window boundary rather than each ticker's listing. That is harmless
# when the boundary is at or before HISTORY_FLOOR, the first historical
# snapshot the research benchmarks use, and destructive when it is later:
# every survivorship filter then drops those tickers before the boundary.
TRUNCATION_SHARE = 0.5
TRUNCATION_MIN_TICKERS = 20
HISTORY_FLOOR = "2020-01-31"


def build_ipo_dates(price_history_path: Path) -> dict:
    """Read price_history.csv and return {ticker: {first_date, last_date}}."""
    first_dates: dict[str, str] = {}
    last_dates: dict[str, str] = {}

    with open(price_history_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ticker = row.get("ticker", "").strip()
            date_str = row.get("date", "").strip()
            if not ticker or not date_str:
                continue
            # Track first (earliest) date per ticker
            if ticker not in first_dates or date_str < first_dates[ticker]:
                first_dates[ticker] = date_str
            # Track last (latest) date per ticker
            if ticker not in last_dates or date_str > last_dates[ticker]:
                last_dates[ticker] = date_str

    tickers = {}
    for ticker in sorted(first_dates.keys()):
        tickers[ticker] = {
            "first_price_date": first_dates[ticker],
            "last_price_date": last_dates[ticker],
        }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "price_history.csv first-price-date proxy",
        "ticker_count": len(tickers),
        "tickers": tickers,
    }


def first_date_concentration(result: dict) -> Tuple[float, str]:
    """Share of tickers on the single most common first_price_date, and that date."""
    firsts = [v.get("first_price_date", "") for v in result.get("tickers", {}).values()]
    firsts = [d for d in firsts if d]
    if not firsts:
        return 0.0, ""
    day, count = Counter(firsts).most_common(1)[0]
    return count / len(firsts), day


def looks_truncated(result: dict, history_floor: str = HISTORY_FLOOR) -> bool:
    """True when most tickers share a first date later than ``history_floor``."""
    if len(result.get("tickers", {})) < TRUNCATION_MIN_TICKERS:
        return False
    share, day = first_date_concentration(result)
    return share > TRUNCATION_SHARE and day > history_floor


def merge_with_existing(built: dict, existing: Optional[dict]) -> Tuple[dict, Dict[str, int]]:
    """Merge a fresh build with the file on disk.

    First dates only move earlier, last dates only move later, and a ticker
    absent from the price file keeps its existing entry.
    """
    moved = {"first_kept_earlier": 0, "first_moved_earlier": 0, "last_kept_later": 0, "dropped": 0}
    old = (existing or {}).get("tickers", {}) or {}
    merged: Dict[str, Dict[str, str]] = {}

    for ticker in sorted(set(built.get("tickers", {})) | set(old)):
        new_entry = built.get("tickers", {}).get(ticker)
        old_entry = old.get(ticker)
        if new_entry is None:
            merged[ticker] = dict(old_entry)
            continue
        first, last = new_entry["first_price_date"], new_entry["last_price_date"]
        if old_entry:
            old_first = old_entry.get("first_price_date", "")
            old_last = old_entry.get("last_price_date", "")
            if old_first and old_first < first:
                first = old_first
                moved["first_kept_earlier"] += 1
            elif old_first and first < old_first:
                moved["first_moved_earlier"] += 1
            if old_last and old_last > last:
                last = old_last
                moved["last_kept_later"] += 1
        merged[ticker] = {"first_price_date": first, "last_price_date": last}

    out = dict(built)
    out["tickers"] = merged
    out["ticker_count"] = len(merged)
    return out, moved


def _write_atomic(path: Path, payload: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def main(argv: Optional[List[str]] = None) -> int:
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--price-history", type=Path, default=project_root / "production_data" / "price_history.csv")
    parser.add_argument("--output", type=Path, default=project_root / "production_data" / "ipo_dates.json")
    parser.add_argument("--no-merge", action="store_true", help="Ignore the existing file (deliberate rebuild)")
    parser.add_argument(
        "--allow-truncated",
        action="store_true",
        help=f"Write even when more than {TRUNCATION_SHARE:.0%} of tickers share one first date after {HISTORY_FLOOR}",
    )
    args = parser.parse_args(argv)

    if not args.price_history.exists():
        print(f"ERROR: {args.price_history} not found", file=sys.stderr)
        return 1

    print(f"Reading {args.price_history}...")
    result = build_ipo_dates(args.price_history)
    print(f"Found {result['ticker_count']} tickers in the price history")

    if not args.no_merge and args.output.exists():
        try:
            existing = json.loads(args.output.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            print(f"ERROR: cannot read existing {args.output}: {exc}", file=sys.stderr)
            return 1
        result, moved = merge_with_existing(result, existing)
        print(
            f"Merged with existing file: {moved['first_moved_earlier']} first dates moved earlier, "
            f"{moved['first_kept_earlier']} kept earlier than the price history, "
            f"{moved['last_kept_later']} last dates kept later"
        )

    share, day = first_date_concentration(result)
    print(f"Most common first_price_date: {day} ({share:.0%} of {result['ticker_count']} tickers)")
    if looks_truncated(result) and not args.allow_truncated:
        print(
            f"ERROR: {share:.0%} of tickers have first_price_date {day}, after the {HISTORY_FLOOR} history floor. "
            f"The price history looks truncated, "
            f"so {args.output} was NOT written. Rebuild from a full history, or pass --allow-truncated.",
            file=sys.stderr,
        )
        return 2

    for ticker, dates in list(result["tickers"].items())[:5]:
        print(f"  {ticker}: first={dates['first_price_date']}, last={dates['last_price_date']}")

    _write_atomic(args.output, result)
    print(f"Saved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
