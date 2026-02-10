#!/usr/bin/env python3
"""Price history provenance — inspect and fingerprint price_history.csv.

Computes SHA256 hash and summary statistics, optionally writes a manifest
JSON for artifact archival.  Can also verify a known hash prefix (--check).

Usage:
    python scripts/price_history_provenance.py [options]

    --price-csv PATH      (default: production_data/price_history.csv)
    --output PATH         (default: production_data/price_history_manifest.json)
    --check SHA256_PREFIX  Verify hash matches prefix, exit 1 if not
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from governance.hashing import hash_file

DEFAULT_PRICE_CSV = PROJECT_ROOT / "production_data" / "price_history.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "production_data" / "price_history_manifest.json"

SCHEMA_VERSION = 1


def compute_price_history_stats(csv_path: Path) -> Dict[str, Any]:
    """Compute SHA256 hash and summary statistics for price_history.csv.

    Returns:
        Dict with: sha256, row_count, ticker_count, date_min, date_max,
        generated_at.
    """
    sha256 = hash_file(csv_path)

    tickers: set = set()
    dates: list = []
    row_count = 0

    with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_count += 1
            ticker = row.get("ticker") or row.get("symbol") or ""
            if ticker:
                tickers.add(ticker)
            date = row.get("date") or row.get("trade_date") or ""
            if date:
                dates.append(date)

    date_min = min(dates) if dates else None
    date_max = max(dates) if dates else None

    return {
        "schema_version": SCHEMA_VERSION,
        "price_history_sha256": sha256,
        "row_count": row_count,
        "ticker_count": len(tickers),
        "date_min": date_min,
        "date_max": date_max,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Price history provenance — fingerprint and manifest."
    )
    parser.add_argument(
        "--price-csv", type=str, default=str(DEFAULT_PRICE_CSV),
        help=f"Path to price_history.csv (default: {DEFAULT_PRICE_CSV})",
    )
    parser.add_argument(
        "--output", type=str, default=str(DEFAULT_OUTPUT),
        help=f"Output manifest JSON path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--check", type=str, default=None, metavar="SHA256_PREFIX",
        help="Verify hash matches this prefix, exit 1 if not",
    )
    args = parser.parse_args()

    csv_path = Path(args.price_csv)
    if not csv_path.exists():
        print(f"ERROR: Price CSV not found: {csv_path}", file=sys.stderr)
        return 1

    stats = compute_price_history_stats(csv_path)

    # Check mode
    if args.check:
        actual = stats["price_history_sha256"]
        expected_prefix = args.check.lower()
        if actual.startswith(expected_prefix):
            print(f"OK: hash {actual[:16]}... matches prefix {expected_prefix}")
            return 0
        else:
            print(
                f"FAIL: hash {actual[:16]}... does not match prefix {expected_prefix}",
                file=sys.stderr,
            )
            return 1

    # Write manifest
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
        f.write("\n")

    # Print summary
    print(f"Price history provenance")
    print(f"  File:     {csv_path}")
    print(f"  SHA256:   {stats['price_history_sha256'][:16]}...")
    print(f"  Rows:     {stats['row_count']:,}")
    print(f"  Tickers:  {stats['ticker_count']}")
    print(f"  Dates:    {stats['date_min']} to {stats['date_max']}")
    print(f"  Manifest: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
