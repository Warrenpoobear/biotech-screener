#!/usr/bin/env python3
"""
expand_universe_to_etfs.py - Expand Universe to Include All XBI, IBB, NBI Constituents

Adds missing ETF constituents to your universe.json.

Usage:
    python expand_universe_to_etfs.py --universe production_data/universe.json --output production_data/universe_expanded.json
"""

import json
from pathlib import Path
from datetime import date
import argparse
from check_etf_coverage import get_xbi_constituents, get_ibb_constituents, get_nbi_constituents


def expand_universe(
    universe_path: Path,
    output_path: Path,
    include_xbi: bool = True,
    include_ibb: bool = True,
    include_nbi: bool = True
) -> dict:
    """
    Expand universe to include all ETF constituents.
    
    Args:
        universe_path: Current universe.json
        output_path: Where to save expanded universe
        include_xbi: Include XBI constituents
        include_ibb: Include IBB constituents
        include_nbi: Include NBI constituents
    
    Returns:
        {
            "original_size": 98,
            "expanded_size": 281,
            "added_tickers": [...],
            "output_path": "..."
        }
    """
    
    # Load existing universe
    with open(universe_path) as f:
        universe_data = json.load(f)
    
    # Extract existing tickers
    existing_tickers = set()
    for security in universe_data:
        ticker = security.get('ticker')
        if ticker and ticker != '_XBI_BENCHMARK_':
            existing_tickers.add(ticker)
    
    # Get ETF constituents
    all_etf_constituents = set()
    
    if include_xbi:
        all_etf_constituents |= get_xbi_constituents()
    if include_ibb:
        all_etf_constituents |= get_ibb_constituents()
    if include_nbi:
        all_etf_constituents |= get_nbi_constituents()
    
    # Find missing tickers
    missing_tickers = all_etf_constituents - existing_tickers
    
    # Create new universe with missing tickers added
    expanded_universe = universe_data.copy()
    
    for ticker in sorted(missing_tickers):
        # Add minimal stub entry
        # You'll need to populate these with real data later
        new_security = {
            "ticker": ticker,
            "name": f"{ticker} (Added from ETF)",
            "exchange": "NASDAQ",  # Placeholder
            "sector": "Biotechnology",
            "market_cap": None,  # Populate later
            "status": "active",
            "added_from_etf": True,
            "added_date": date.today().isoformat(),
            "sources": []
        }
        
        # Add source information
        if include_xbi and ticker in get_xbi_constituents():
            new_security['sources'].append('XBI')
        if include_ibb and ticker in get_ibb_constituents():
            new_security['sources'].append('IBB')
        if include_nbi and ticker in get_nbi_constituents():
            new_security['sources'].append('NBI')
        
        expanded_universe.append(new_security)
    
    # Save expanded universe
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(expanded_universe, f, indent=2, sort_keys=False)
        f.write('\n')
    
    result = {
        "original_size": len(existing_tickers),
        "expanded_size": len(existing_tickers) + len(missing_tickers),
        "added_count": len(missing_tickers),
        "added_tickers": sorted(missing_tickers),
        "output_path": str(output_path)
    }
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Expand universe to include all XBI, IBB, NBI constituents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Expand to all three ETFs
  python expand_universe_to_etfs.py --universe production_data/universe.json --output production_data/universe_expanded.json
  
  # Expand to XBI only
  python expand_universe_to_etfs.py --universe production_data/universe.json --output production_data/universe_expanded.json --xbi-only
  
  # Preview what would be added (dry run)
  python expand_universe_to_etfs.py --universe production_data/universe.json --dry-run

Note:
  New securities will have placeholder data. You'll need to populate:
  - Market cap
  - Company name (currently uses ticker)
  - Exchange (currently defaults to NASDAQ)
  
  Run collect_universe_data.py afterward to fetch full data.
        """
    )
    
    parser.add_argument(
        '--universe',
        type=Path,
        required=True,
        help='Path to current universe.json'
    )
    
    parser.add_argument(
        '--output',
        type=Path,
        help='Path for expanded universe (default: universe_expanded.json)'
    )
    
    parser.add_argument(
        '--xbi-only',
        action='store_true',
        help='Only add XBI constituents'
    )
    
    parser.add_argument(
        '--ibb-only',
        action='store_true',
        help='Only add IBB constituents'
    )
    
    parser.add_argument(
        '--nbi-only',
        action='store_true',
        help='Only add NBI constituents'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview changes without writing file'
    )
    
    args = parser.parse_args()
    
    # Determine which ETFs to include
    if args.xbi_only:
        include_xbi, include_ibb, include_nbi = True, False, False
    elif args.ibb_only:
        include_xbi, include_ibb, include_nbi = False, True, False
    elif args.nbi_only:
        include_xbi, include_ibb, include_nbi = False, False, True
    else:
        include_xbi, include_ibb, include_nbi = True, True, True
    
    # Set output path
    if args.output:
        output_path = args.output
    else:
        output_path = args.universe.parent / 'universe_expanded.json'
    
    # Dry run?
    if args.dry_run:
        output_path = Path('/tmp/universe_expanded_dry_run.json')
    
    print("="*80)
    print("UNIVERSE EXPANSION")
    print("="*80)
    
    # Expand universe
    result = expand_universe(
        universe_path=args.universe,
        output_path=output_path,
        include_xbi=include_xbi,
        include_ibb=include_ibb,
        include_nbi=include_nbi
    )
    
    # Report
    print(f"\n📊 EXPANSION SUMMARY")
    print("-"*80)
    print(f"Original universe: {result['original_size']} tickers")
    print(f"Expanded universe: {result['expanded_size']} tickers")
    print(f"Added: {result['added_count']} tickers")
    
    print(f"\n✅ ETFs INCLUDED:")
    if include_xbi:
        print(f"   • XBI (SPDR S&P Biotech)")
    if include_ibb:
        print(f"   • IBB (iShares Biotechnology)")
    if include_nbi:
        print(f"   • NBI (Nasdaq Biotechnology)")
    
    if result['added_count'] > 0:
        print(f"\n📝 ADDED TICKERS ({result['added_count']}):")
        print("-"*80)
        for i, ticker in enumerate(result['added_tickers'], 1):
            print(f"  {i:3d}. {ticker}")
    
    if args.dry_run:
        print(f"\n⚠️  DRY RUN: No files modified")
        print(f"   Preview saved to: {output_path}")
    else:
        print(f"\n✅ Expanded universe saved to: {output_path}")
        print(f"\n⚠️  IMPORTANT: New securities have placeholder data!")
        print(f"   Run these commands to populate full data:")
        print(f"   1. Fetch market cap / company info")
        print(f"   2. Fetch financial data")
        print(f"   3. Fetch clinical trials")
        print(f"\n   Or run your data collection pipeline with the new universe.")
    
    print(f"\n{'='*80}\n")
    
    return 0


if __name__ == "__main__":
    exit(main())
