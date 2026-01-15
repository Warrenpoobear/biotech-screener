#!/usr/bin/env python3
"""
Build Returns Database from Morningstar Direct (Updated for fixed API)

Fetches historical returns data for the biotech universe and caches it locally.
Uses Morningstar SecIds for accurate data retrieval.

Usage:
    $env:MD_AUTH_TOKEN="your-token-here"
    python build_returns_db.py --universe universe.csv --start-date 2020-01-01
"""

import argparse
import csv
import json
import os
import sys
from datetime import date
from pathlib import Path
from typing import List, Dict, Optional

sys.path.insert(0, str(Path(__file__).parent))

import morningstar_returns as mr

# Try to import morningstar_data for ticker lookup
try:
    import morningstar_data as md
    MD_AVAILABLE = True
except ImportError:
    MD_AVAILABLE = False


def load_universe(universe_path: Path) -> List[Dict[str, str]]:
    """
    Load universe from CSV. Expects columns: ticker, [secid]
    If secid not provided, will attempt to look up from Morningstar.
    """
    securities = []

    with open(universe_path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        headers = {h.lower(): h for h in reader.fieldnames or []}

        ticker_col = headers.get("ticker") or headers.get("symbol")
        secid_col = headers.get("secid") or headers.get("sec_id") or headers.get("morningstar_id")

        if not ticker_col:
            raise ValueError(f"No 'ticker' column found. Available: {list(reader.fieldnames or [])}")

        for row in reader:
            ticker = row[ticker_col].strip().upper()
            if not ticker or ticker.startswith("#"):
                continue

            sec_id = row.get(secid_col, "").strip() if secid_col else ""
            securities.append({"ticker": ticker, "secid": sec_id})

    return securities


def lookup_secids(tickers: List[str]) -> Dict[str, str]:
    """
    Look up Morningstar SecIds for tickers using investments() search.
    Returns {ticker: secid} mapping.
    """
    if not MD_AVAILABLE:
        print("  Warning: morningstar_data not available for SecId lookup")
        return {}

    mapping = {}

    # Try ticker:US format first (works for US equities)
    sec_ids_to_try = [f"{t}:US" for t in tickers]

    try:
        # Use investments() to search
        results = md.direct.investments(sec_ids_to_try)

        if results is not None and not results.empty:
            for idx, row in results.iterrows():
                ticker = str(row.get('Ticker', '')).upper()
                sec_id = str(row.get('SecId', ''))
                if ticker and sec_id:
                    mapping[ticker] = sec_id

    except Exception as e:
        print(f"  Warning: SecId lookup failed: {e}")

    return mapping


def main():
    parser = argparse.ArgumentParser(description="Build returns database from Morningstar Direct")
    parser.add_argument("--universe", type=Path, required=True, help="CSV with ticker,[secid] columns")
    parser.add_argument("--start-date", type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=None, help="End date (YYYY-MM-DD, default: today)")
    parser.add_argument("--output-dir", type=Path, default=Path("data/returns"), help="Output directory")
    parser.add_argument("--batch-size", type=int, default=20, help="Securities per batch")

    args = parser.parse_args()

    # Validate
    if not args.universe.exists():
        print(f"Error: File not found: {args.universe}")
        sys.exit(1)

    if not os.environ.get("MD_AUTH_TOKEN"):
        print("Error: MD_AUTH_TOKEN not set")
        print("  $env:MD_AUTH_TOKEN='your-token'")
        sys.exit(1)

    # Parse dates
    start_date = date.fromisoformat(args.start_date)
    end_date = date.fromisoformat(args.end_date) if args.end_date else date.today()

    # Load universe
    print(f"Loading universe from: {args.universe}")
    securities = load_universe(args.universe)
    print(f"  Found {len(securities)} securities")

    # Check which have SecIds
    with_secid = [s for s in securities if s['secid']]
    without_secid = [s for s in securities if not s['secid']]

    print(f"  With SecId: {len(with_secid)}")
    print(f"  Need lookup: {len(without_secid)}")

    # Look up missing SecIds
    if without_secid:
        print("\nLooking up Morningstar SecIds...")
        tickers_to_lookup = [s['ticker'] for s in without_secid]
        found_mapping = lookup_secids(tickers_to_lookup)

        for s in without_secid:
            if s['ticker'] in found_mapping:
                s['secid'] = found_mapping[s['ticker']]

        still_missing = [s['ticker'] for s in securities if not s['secid']]
        if still_missing:
            print(f"  Warning: Could not find SecIds for: {still_missing[:10]}{'...' if len(still_missing) > 10 else ''}")

    # Filter to securities with SecIds
    valid_securities = [s for s in securities if s['secid']]
    print(f"\nWill fetch returns for {len(valid_securities)} securities")

    if not valid_securities:
        print("Error: No valid securities to fetch")
        sys.exit(1)

    # Fetch returns in batches
    print("\n" + "="*60)
    print("FETCHING RETURNS DATA")
    print("="*60)
    print(f"  Date range: {start_date} to {end_date}")

    all_records = []
    sec_ids = [s['secid'] for s in valid_securities]
    ticker_map = {s['secid']: s['ticker'] for s in valid_securities}

    for i in range(0, len(sec_ids), args.batch_size):
        batch = sec_ids[i:i+args.batch_size]
        batch_num = i // args.batch_size + 1
        total_batches = (len(sec_ids) + args.batch_size - 1) // args.batch_size

        print(f"\n  Batch {batch_num}/{total_batches} ({len(batch)} securities)...")

        try:
            data = mr.fetch_returns(
                sec_ids=batch,
                start_date=start_date.isoformat(),
                end_date=end_date.isoformat(),
                frequency='monthly'
            )

            records = data.get('absolute', [])
            print(f"    Retrieved {len(records)} records")

            # Add ticker mapping
            for rec in records:
                rec['ticker'] = ticker_map.get(rec['sec_id'], rec['sec_id'])

            all_records.extend(records)

        except Exception as e:
            print(f"    Error: {e}")
            continue

    # Build database structure
    print("\n" + "="*60)
    print("BUILDING DATABASE")
    print("="*60)

    # Group by security
    returns_by_ticker = {}
    for rec in all_records:
        ticker = rec['ticker']
        if ticker not in returns_by_ticker:
            returns_by_ticker[ticker] = []
        returns_by_ticker[ticker].append({
            'date': rec['date'],
            'return': float(rec['return_pct'])
        })

    database = {
        'metadata': {
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(),
            'source': 'morningstar_direct',
            'created_at': date.today().isoformat(),
            'ticker_count': len(returns_by_ticker),
        },
        'returns': returns_by_ticker,
        'ticker_to_secid': ticker_map,
    }

    # Save
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_file = args.output_dir / f"returns_db_{start_date}_{end_date}.json"

    with open(output_file, 'w') as f:
        json.dump(database, f, indent=2)

    print(f"\n  Saved to: {output_file}")
    print(f"  Tickers: {len(returns_by_ticker)}")
    print(f"  Total records: {len(all_records)}")

    print("\n" + "="*60)
    print("BUILD COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
