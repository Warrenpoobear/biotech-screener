#!/usr/bin/env python3
"""
Build Returns Database from Morningstar Direct

Fetches historical returns data for the biotech universe and caches it locally.
This script REQUIRES a valid MD_AUTH_TOKEN.

Once the database is built, validation can be run unlimited times WITHOUT a token.

Usage:
    # Set token first
    $env:MD_AUTH_TOKEN="your-token-here"  # Windows PowerShell
    export MD_AUTH_TOKEN="your-token-here" # Linux/Mac

    # Build database
    python build_returns_database.py --universe universe.csv --start-date 2020-01-01

    # Or with specific end date
    python build_returns_database.py --universe universe.csv --start-date 2020-01-01 --end-date 2024-12-31

Output:
    data/returns/returns_db_2020-01-01_2024-12-31.json

Architecture:
    - Batch processing (20 tickers at a time to avoid timeouts)
    - Automatic resume (previously fetched data is cached)
    - Complete audit trail for regulatory compliance
    - Includes XBI benchmark automatically
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import date, datetime
from pathlib import Path
from typing import List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from morningstar_returns import MorningstarReturnsFetcher


def load_universe(universe_path: Path) -> List[str]:
    """
    Load ticker universe from CSV.

    Expects CSV with 'ticker' column (case-insensitive header).
    """
    tickers = []

    with open(universe_path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        # Find ticker column (case-insensitive)
        headers = {h.lower(): h for h in reader.fieldnames or []}
        ticker_col = headers.get("ticker") or headers.get("symbol") or headers.get("tickers")

        if not ticker_col:
            raise ValueError(
                f"No 'ticker' or 'symbol' column found in {universe_path}. "
                f"Available columns: {list(reader.fieldnames or [])}"
            )

        for row in reader:
            ticker = row[ticker_col].strip().upper()
            if ticker and not ticker.startswith("#"):
                tickers.append(ticker)

    return tickers


def main():
    parser = argparse.ArgumentParser(
        description="Build returns database from Morningstar Direct",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Build full historical database
    python build_returns_database.py --universe example_universe.csv --start-date 2020-01-01

    # Build database for specific period
    python build_returns_database.py --universe universe.csv --start-date 2023-01-01 --end-date 2023-12-31

    # Use custom output directory
    python build_returns_database.py --universe universe.csv --start-date 2020-01-01 --output-dir ./my_data

Required Environment Variable:
    MD_AUTH_TOKEN - Your Morningstar Direct authentication token
    Get this from Morningstar Direct web interface.
        """,
    )

    parser.add_argument(
        "--universe",
        type=Path,
        required=True,
        help="Path to CSV file with ticker universe (must have 'ticker' column)",
    )

    parser.add_argument(
        "--start-date",
        type=str,
        required=True,
        help="Start date for returns (YYYY-MM-DD)",
    )

    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date for returns (YYYY-MM-DD, default: today)",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/returns"),
        help="Output directory for database (default: data/returns)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Tickers per API batch (default: 20)",
    )

    parser.add_argument(
        "--no-benchmark",
        action="store_true",
        help="Skip XBI benchmark (not recommended)",
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.universe.exists():
        print(f"Error: Universe file not found: {args.universe}")
        sys.exit(1)

    # Check for token
    if not os.environ.get("MD_AUTH_TOKEN"):
        print("Error: MD_AUTH_TOKEN environment variable not set.")
        print()
        print("Set your Morningstar Direct token:")
        print("  Windows PowerShell: $env:MD_AUTH_TOKEN='your-token-here'")
        print("  Linux/Mac bash:     export MD_AUTH_TOKEN='your-token-here'")
        print()
        print("Get token from Morningstar Direct web interface.")
        sys.exit(1)

    # Parse dates
    try:
        start_date = date.fromisoformat(args.start_date)
    except ValueError:
        print(f"Error: Invalid start date format: {args.start_date}")
        print("Expected format: YYYY-MM-DD (e.g., 2020-01-01)")
        sys.exit(1)

    if args.end_date:
        try:
            end_date = date.fromisoformat(args.end_date)
        except ValueError:
            print(f"Error: Invalid end date format: {args.end_date}")
            print("Expected format: YYYY-MM-DD (e.g., 2024-12-31)")
            sys.exit(1)
    else:
        end_date = date.today()

    if start_date >= end_date:
        print(f"Error: start_date ({start_date}) must be before end_date ({end_date})")
        sys.exit(1)

    # Load universe
    print(f"Loading universe from: {args.universe}")
    try:
        tickers = load_universe(args.universe)
    except Exception as e:
        print(f"Error loading universe: {e}")
        sys.exit(1)

    if not tickers:
        print("Error: No tickers found in universe file")
        sys.exit(1)

    print(f"  Found {len(tickers)} tickers")
    print()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Build database
    print("=" * 60)
    print("BUILDING RETURNS DATABASE")
    print("=" * 60)
    print(f"  Tickers:    {len(tickers)}")
    print(f"  Start:      {start_date}")
    print(f"  End:        {end_date}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Output:     {args.output_dir}")
    print()

    try:
        fetcher = MorningstarReturnsFetcher(
            cache_dir=args.output_dir,
            batch_size=args.batch_size,
        )

        data = fetcher.fetch_returns(
            tickers=tickers,
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
            include_benchmark=not args.no_benchmark,
        )

        output_path = fetcher.save_database(data)

    except EnvironmentError as e:
        print(f"\nToken Error: {e}")
        sys.exit(1)
    except ConnectionError as e:
        print(f"\nConnection Error: {e}")
        print("\nPossible causes:")
        print("  - Token expired (tokens typically last 24 hours)")
        print("  - Network connectivity issues")
        print("  - Morningstar Direct service unavailable")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Summary
    print()
    print("=" * 60)
    print("DATABASE BUILD COMPLETE")
    print("=" * 60)
    print(f"  Output file: {output_path}")
    print(f"  Tickers:     {data['coverage']['received']}/{data['coverage']['requested']}")
    print()
    print("Next steps:")
    print("  1. Run validation (NO TOKEN NEEDED):")
    print(f"     python validate_signals.py --database {output_path} --ranked-list your_screen.csv")
    print()
    print("  2. To update database later (TOKEN NEEDED):")
    print("     python build_returns_database.py --universe universe.csv --start-date <new_start>")


if __name__ == "__main__":
    main()
