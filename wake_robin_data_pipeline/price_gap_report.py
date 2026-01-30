#!/usr/bin/env python3
"""Offline Price Gap Report CLI (PIT-safe)."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

DEFAULT_UNIVERSE_PATHS = [
    "production_data/universe.json",
    "wake_robin_data_pipeline/outputs/universe_snapshot_latest.json",
]


def load_universe_tickers(path: Optional[str]) -> List[str]:
    """Load ticker list from universe file."""
    search_paths = [path] if path else DEFAULT_UNIVERSE_PATHS
    for p in search_paths:
        fp = Path(p)
        if fp.exists():
            with open(fp) as f:
                data = json.load(f)
            if isinstance(data, dict):
                secs = data.get("active_securities", [])
            else:
                secs = data
            return sorted(set(s.get("ticker") for s in secs if s.get("ticker")))
    return []


def load_price_coverage(price_file: str, as_of: str) -> Dict[str, Dict]:
    """Load price data and compute coverage stats per ticker."""
    coverage = defaultdict(lambda: {"dates": []})
    with open(price_file, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            date, ticker = row.get("date", ""), row.get("ticker", "")
            if not ticker or not date or date > as_of:
                continue
            if row.get("close"):
                coverage[ticker]["dates"].append(date)
    # Compute stats
    result = {}
    for ticker, data in coverage.items():
        dates = sorted(data["dates"])
        result[ticker] = {
            "rows_total": len(dates),
            "first_date": dates[0] if dates else None,
            "last_date": dates[-1] if dates else None,
        }
    return result


def compute_gap_report(
    universe_tickers: List[str],
    price_coverage: Dict[str, Dict],
    min_20: int = 20,
    min_60: int = 60,
    min_120: int = 120,
) -> Dict:
    """Compute gap report for universe tickers."""
    by_ticker = {}
    missing = []
    blocking_120 = []
    blocking_60 = []
    blocking_20 = []

    for ticker in sorted(universe_tickers):
        cov = price_coverage.get(ticker)
        if not cov:
            by_ticker[ticker] = {
                "rows_total": 0,
                "first_date": None,
                "last_date": None,
                "ok_20": False,
                "ok_60": False,
                "ok_120": False,
                "blocking_reason": "missing_ticker",
            }
            missing.append(ticker)
            continue

        rows = cov["rows_total"]
        ok_20 = rows >= min_20
        ok_60 = rows >= min_60
        ok_120 = rows >= min_120

        if not ok_120:
            reason = "insufficient_rows_120"
            blocking_120.append(ticker)
        elif not ok_60:
            reason = "insufficient_rows_60"
            blocking_60.append(ticker)
        elif not ok_20:
            reason = "insufficient_rows_20"
            blocking_20.append(ticker)
        else:
            reason = None

        by_ticker[ticker] = {
            "rows_total": rows,
            "first_date": cov["first_date"],
            "last_date": cov["last_date"],
            "ok_20": ok_20,
            "ok_60": ok_60,
            "ok_120": ok_120,
            "blocking_reason": reason,
        }

    present = len(universe_tickers) - len(missing)
    summary = {
        "universe_tickers": len(universe_tickers),
        "present_in_prices": present,
        "ok_20": sum(1 for t in by_ticker.values() if t["ok_20"]),
        "ok_60": sum(1 for t in by_ticker.values() if t["ok_60"]),
        "ok_120": sum(1 for t in by_ticker.values() if t["ok_120"]),
        "blocking_120": len(blocking_120),
        "blocking_60": len(blocking_60),
        "blocking_20": len(blocking_20),
        "missing": len(missing),
    }
    blocking_tickers = sorted(missing) + sorted(blocking_120) + sorted(blocking_60) + sorted(blocking_20)
    return {"summary": summary, "by_ticker": by_ticker, "blocking_tickers": blocking_tickers}


def build_report(
    as_of: str,
    price_file: str,
    universe_file: Optional[str],
    min_20: int = 20,
    min_60: int = 60,
    min_120: int = 120,
) -> Dict:
    """Build complete gap report."""
    tickers = load_universe_tickers(universe_file)
    coverage = load_price_coverage(price_file, as_of)
    report = compute_gap_report(tickers, coverage, min_20, min_60, min_120)
    return {
        "as_of_date": as_of,
        "price_file": price_file,
        "universe_file": universe_file or "default",
        "thresholds": {"min_rows_20": min_20, "min_rows_60": min_60, "min_rows_120": min_120},
        **report,
    }


def main():
    parser = argparse.ArgumentParser(description="Price history gap report")
    parser.add_argument("--as-of", required=True, help="As-of date (YYYY-MM-DD)")
    parser.add_argument("--price-file", required=True, help="Price history CSV")
    parser.add_argument("--universe", default=None, help="Universe JSON (optional)")
    parser.add_argument("--min-rows-20", type=int, default=20)
    parser.add_argument("--min-rows-60", type=int, default=60)
    parser.add_argument("--min-rows-120", type=int, default=120)
    parser.add_argument("--out", required=True, help="Output JSON path")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    report = build_report(
        args.as_of, args.price_file, args.universe,
        args.min_rows_20, args.min_rows_60, args.min_rows_120,
    )

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    if not args.quiet:
        s = report["summary"]
        print(f"Gap report for as_of={args.as_of}")
        print(f"  Universe: {s['universe_tickers']} tickers")
        print(f"  Present in prices: {s['present_in_prices']}")
        print(f"  OK for 120d: {s['ok_120']} | blocking: {s['blocking_120']}")
        print(f"  OK for 60d: {s['ok_60']} | blocking: {s['blocking_60']}")
        print(f"  OK for 20d: {s['ok_20']} | blocking: {s['blocking_20']}")
        print(f"  Missing tickers: {s['missing']}")
        if report["blocking_tickers"]:
            print(f"  Blocking: {report['blocking_tickers'][:10]}{'...' if len(report['blocking_tickers']) > 10 else ''}")
        print(f"  Output: {args.out}")


if __name__ == "__main__":
    main()
