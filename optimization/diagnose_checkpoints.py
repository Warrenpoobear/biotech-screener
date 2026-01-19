#!/usr/bin/env python3
"""
Diagnostic tool for checkpoint files.

Examines the structure of checkpoint JSON files to verify that tickers
can be extracted correctly for price fetching.

Run with: python optimization/diagnose_checkpoints.py
"""

import json
from pathlib import Path


def diagnose_checkpoints(checkpoint_dir='checkpoints'):
    """Examine checkpoint file structure and extract tickers."""
    checkpoint_path = Path(checkpoint_dir)

    print("=" * 70)
    print("CHECKPOINT FILE DIAGNOSTIC")
    print("=" * 70)
    print()

    # Check if directory exists
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint directory not found: {checkpoint_dir}")
        print()
        print("Expected location: checkpoints/")
        print("Files should be named: module_5_YYYY-MM-DD.json")
        return False

    # Find checkpoint files
    checkpoint_files = sorted(checkpoint_path.glob('module_5_*.json'))

    if not checkpoint_files:
        print(f"ERROR: No checkpoint files found in '{checkpoint_dir}'")
        print()
        print("Looking for files matching pattern: module_5_YYYY-MM-DD.json")
        return False

    print(f"Found {len(checkpoint_files)} checkpoint files")
    print()

    # Examine first file in detail
    sample_file = checkpoint_files[0]
    print(f"Examining: {sample_file.name}")
    print("-" * 70)

    try:
        with open(sample_file) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in {sample_file.name}: {e}")
        return False
    except Exception as e:
        print(f"ERROR: Could not read {sample_file.name}: {e}")
        return False

    # Show structure
    print("Top-level keys in JSON:")
    for key in data.keys():
        value = data[key]
        if isinstance(value, list):
            print(f"  - {key}: list with {len(value)} items")
            if value and isinstance(value[0], dict):
                print(f"    First item keys: {list(value[0].keys())[:8]}")
        elif isinstance(value, dict):
            print(f"  - {key}: dict with {len(value)} keys")
        else:
            print(f"  - {key}: {type(value).__name__}")
    print()

    # Try to extract tickers
    print("Checking for ticker locations...")
    print("-" * 70)

    all_tickers = set()
    extraction_method = None

    # Method 1: data.ranked_securities
    if 'data' in data and isinstance(data['data'], dict):
        securities = data['data'].get('ranked_securities', [])
        if securities:
            print(f"Found 'data.ranked_securities' with {len(securities)} items")
            if securities and isinstance(securities[0], dict):
                print(f"  Sample keys: {list(securities[0].keys())[:5]}")
            for sec in securities:
                ticker = sec.get('ticker')
                if ticker:
                    all_tickers.add(ticker)
            if all_tickers:
                extraction_method = "data.ranked_securities"
                print(f"  Found 'ticker' field")

    # Method 2: ranked_securities (top level)
    if not all_tickers and 'ranked_securities' in data:
        securities = data['ranked_securities']
        if securities:
            print(f"Found 'ranked_securities' with {len(securities)} items")
            if securities and isinstance(securities[0], dict):
                print(f"  Sample keys: {list(securities[0].keys())[:5]}")
            for sec in securities:
                ticker = sec.get('ticker')
                if ticker:
                    all_tickers.add(ticker)
            if all_tickers:
                extraction_method = "ranked_securities"
                print(f"  Found 'ticker' field")

    # Method 3: results
    if not all_tickers and 'results' in data:
        securities = data['results']
        if securities:
            print(f"Found 'results' with {len(securities)} items")
            if securities and isinstance(securities[0], dict):
                print(f"  Sample keys: {list(securities[0].keys())[:5]}")
            for sec in securities:
                ticker = sec.get('ticker')
                if ticker:
                    all_tickers.add(ticker)
            if all_tickers:
                extraction_method = "results"
                print(f"  Found 'ticker' field")

    # Method 4: securities
    if not all_tickers and 'securities' in data:
        securities = data['securities']
        if securities:
            print(f"Found 'securities' with {len(securities)} items")
            if securities and isinstance(securities[0], dict):
                print(f"  Sample keys: {list(securities[0].keys())[:5]}")
            for sec in securities:
                ticker = sec.get('ticker')
                if ticker:
                    all_tickers.add(ticker)
            if all_tickers:
                extraction_method = "securities"
                print(f"  Found 'ticker' field")

    print()

    if all_tickers:
        sample_tickers = sorted(all_tickers)[:5]
        print(f"  Sample tickers: {sample_tickers}")
        print()

        # Now scan all files
        print("Scanning all checkpoint files...")
        print("-" * 70)

        all_tickers_full = set()
        dates = []

        for filepath in checkpoint_files:
            try:
                with open(filepath) as f:
                    file_data = json.load(f)

                # Extract using the method that worked
                if extraction_method == "data.ranked_securities":
                    securities = file_data.get('data', {}).get('ranked_securities', [])
                elif extraction_method == "ranked_securities":
                    securities = file_data.get('ranked_securities', [])
                elif extraction_method == "results":
                    securities = file_data.get('results', [])
                elif extraction_method == "securities":
                    securities = file_data.get('securities', [])
                else:
                    securities = []

                for sec in securities:
                    ticker = sec.get('ticker')
                    if ticker:
                        all_tickers_full.add(ticker)

                # Extract date from filename
                date_str = filepath.stem.replace('module_5_', '')
                dates.append(date_str)

            except Exception:
                continue

        print(f"Total unique tickers found: {len(all_tickers_full)}")
        print(f"Date range: {min(dates)} to {max(dates)}")
        print()

        # Success
        print("=" * 70)
        print("RECOMMENDATION")
        print("=" * 70)
        print()
        print("Tickers successfully extracted!")
        print()
        print("Your checkpoint files are formatted correctly.")
        print("The price fetcher should work now.")
        print()
        print("Run the fixed price fetcher with:")
        print("  python optimization/fetch_prices_interactive.py")
        print()
        return True

    else:
        # Could not find tickers
        print("Could not automatically find tickers in checkpoint files.")
        print()
        print("=" * 70)
        print("RECOMMENDATION")
        print("=" * 70)
        print()
        print("The checkpoint file structure is different than expected.")
        print()
        print("Here's what the first checkpoint file looks like:")
        print()

        # Show a preview of the JSON
        preview = json.dumps(data, indent=2)[:2000]
        print(preview)
        if len(json.dumps(data, indent=2)) > 2000:
            print("... (truncated)")

        print()
        print("Please share this output so I can update the extraction logic.")
        return False


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Diagnose checkpoint file structure for price fetching'
    )
    parser.add_argument(
        '--checkpoint-dir',
        default='checkpoints',
        help='Directory containing checkpoint files (default: checkpoints)'
    )

    args = parser.parse_args()

    success = diagnose_checkpoints(args.checkpoint_dir)
    exit(0 if success else 1)
