#!/usr/bin/env python3
"""
Clean biotech universe files and remove ineligible securities.
Implements fail-loud validation to prevent data contamination.
"""
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Set, List, Dict, Any
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from validators.ticker_validator import validate_ticker_list, is_valid_ticker


def extract_tickers_from_universe(data: Any) -> List[str]:
    """Extract tickers from various universe file formats."""
    tickers = []

    if isinstance(data, list):
        for item in data:
            if isinstance(item, str):
                tickers.append(item)
            elif isinstance(item, dict):
                # Handle {"ticker": "MRNA", ...} format
                if 'ticker' in item:
                    tickers.append(item['ticker'])
                elif 'symbol' in item:
                    tickers.append(item['symbol'])

    elif isinstance(data, dict):
        # Handle {"tickers": [...]} format
        if 'tickers' in data:
            tickers.extend(extract_tickers_from_universe(data['tickers']))
        # Handle {"active_securities": [...]} format
        if 'active_securities' in data:
            tickers.extend(extract_tickers_from_universe(data['active_securities']))
        # Handle direct ticker keys
        for key in data:
            if isinstance(data[key], dict) and 'ticker' in data[key]:
                tickers.append(data[key]['ticker'])

    return tickers


def load_universe_file(filepath: Path) -> tuple:
    """Load tickers from universe file and return (data, tickers)."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    tickers = extract_tickers_from_universe(data)
    return data, tickers


def clean_universe_data(data: Any, valid_tickers: Set[str]) -> Any:
    """
    Clean universe data by removing invalid tickers.
    Returns cleaned data structure.
    """
    if isinstance(data, list):
        cleaned = []
        for item in data:
            if isinstance(item, str):
                if item in valid_tickers:
                    cleaned.append(item)
            elif isinstance(item, dict):
                ticker = item.get('ticker') or item.get('symbol')
                if ticker and ticker in valid_tickers:
                    cleaned.append(item)
                elif not ticker:
                    # Keep items without ticker field
                    cleaned.append(item)
        return cleaned

    elif isinstance(data, dict):
        cleaned = {}
        for key, value in data.items():
            if key in ['tickers', 'active_securities']:
                cleaned[key] = clean_universe_data(value, valid_tickers)
            elif isinstance(value, dict) and 'ticker' in value:
                if value['ticker'] in valid_tickers:
                    cleaned[key] = value
            else:
                cleaned[key] = value
        return cleaned

    return data


def save_cleaned_universe(filepath: Path, data: Any, metadata: dict):
    """Save cleaned universe with metadata."""
    # Add metadata if it's a dict
    if isinstance(data, dict):
        data['_cleaning_metadata'] = {
            'cleaned_at': datetime.utcnow().isoformat() + 'Z',
            **metadata
        }

    # Atomic write
    temp_path = filepath.with_suffix('.tmp')
    with open(temp_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    temp_path.replace(filepath)


def save_audit_log(output_dir: Path, removed_tickers: dict, source_file: str):
    """Save audit log of removed tickers."""
    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    audit_file = output_dir / f'universe_cleaning_audit_{timestamp}.json'

    audit_data = {
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'source_file': source_file,
        'removed_count': len(removed_tickers),
        'removed_tickers': removed_tickers
    }

    with open(audit_file, 'w', encoding='utf-8') as f:
        json.dump(audit_data, f, indent=2)

    print(f"  📋 Audit log saved: {audit_file.name}")
    return audit_file


def clean_universe_files(check_only: bool = False, verbose: bool = True):
    """Main function to clean all universe files."""

    # Locate project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent

    # Find all universe files
    universe_patterns = [
        '**/universe*.json',
        '**/biotech_universe*.json',
        '**/pilot_universe*.json',
    ]

    universe_files = []
    for pattern in universe_patterns:
        universe_files.extend(project_root.glob(pattern))

    # Deduplicate
    universe_files = list(set(universe_files))

    # Skip backup files
    universe_files = [f for f in universe_files if '.backup' not in f.name]

    print(f"Found {len(universe_files)} universe files to {'check' if check_only else 'clean'}")
    print()

    total_removed = {}
    all_valid = True

    for filepath in sorted(universe_files):
        rel_path = filepath.relative_to(project_root)
        print(f"Processing: {rel_path}")

        try:
            # Load tickers
            data, tickers = load_universe_file(filepath)
            print(f"  Loaded {len(tickers)} tickers")

            # Validate
            validation_result = validate_ticker_list(tickers)

            # Report results
            if validation_result['invalid']:
                all_valid = False
                print(f"  ❌ Found {len(validation_result['invalid'])} invalid tickers:")

                # Show first few invalid tickers
                shown = 0
                for ticker, reason in validation_result['invalid'].items():
                    if shown < 5:
                        display_ticker = ticker[:40] + "..." if len(ticker) > 40 else ticker
                        print(f"     - '{display_ticker}': {reason}")
                        shown += 1
                if len(validation_result['invalid']) > 5:
                    print(f"     ... and {len(validation_result['invalid']) - 5} more")

                if not check_only:
                    # Create backup
                    backup_path = filepath.with_suffix('.backup.json')
                    if not backup_path.exists():
                        import shutil
                        shutil.copy(filepath, backup_path)
                        print(f"  💾 Backup saved: {backup_path.name}")

                    # Clean the data
                    valid_set = set(validation_result['valid'])
                    cleaned_data = clean_universe_data(data, valid_set)

                    # Save cleaned version
                    save_cleaned_universe(
                        filepath,
                        cleaned_data,
                        {
                            'original_count': len(tickers),
                            'removed_count': len(validation_result['invalid']),
                        }
                    )
                    print(f"  ✅ Cleaned universe saved: {len(validation_result['valid'])} tickers")

                    # Save audit log
                    audit_dir = project_root / 'audit_logs'
                    audit_dir.mkdir(exist_ok=True)
                    save_audit_log(audit_dir, validation_result['invalid'], str(rel_path))

                total_removed.update(validation_result['invalid'])
            else:
                print(f"  ✅ Already clean ({len(tickers)} valid tickers)")

        except Exception as e:
            print(f"  ⚠️  Error processing: {e}")
            all_valid = False

        print()

    # Summary
    print("=" * 60)
    if total_removed:
        print(f"SUMMARY: {'Would remove' if check_only else 'Removed'} {len(total_removed)} invalid tickers")
        if check_only:
            print("\nRun without --check-only to clean the files.")
    else:
        print("✅ All universe files are clean!")
    print("=" * 60)

    return all_valid


def main():
    parser = argparse.ArgumentParser(
        description='Clean biotech universe files and remove ineligible securities.'
    )
    parser.add_argument(
        '--check-only',
        action='store_true',
        help='Check validity without modifying files'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output'
    )
    args = parser.parse_args()

    all_valid = clean_universe_files(
        check_only=args.check_only,
        verbose=args.verbose
    )

    # Exit with error code if invalid (for pre-commit hooks)
    sys.exit(0 if all_valid else 1)


if __name__ == '__main__':
    main()
