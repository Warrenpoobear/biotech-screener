#!/usr/bin/env python3
"""
add_etf_tickers_to_universe.py - Add Complete ETF Holdings to Universe

Takes the etf_holdings_complete.json and adds missing tickers to universe.json.

Usage:
    python add_etf_tickers_to_universe.py
"""

import json
from pathlib import Path
from datetime import date


def add_etf_tickers_to_universe():
    """Add complete ETF holdings to universe"""
    
    print("="*80)
    print("ADDING ETF TICKERS TO UNIVERSE")
    print("="*80)
    
    # Load ETF holdings
    etf_file = Path('etf_holdings_complete.json')
    if not etf_file.exists():
        print(f"\n❌ ETF holdings file not found: {etf_file}")
        print(f"\nRun this first:")
        print(f"  python import_etf_csvs.py")
        return 1
    
    with open(etf_file) as f:
        holdings = json.load(f)
    
    # Get all unique ETF tickers
    all_etf_tickers = set()
    all_etf_tickers.update(holdings.get('xbi', []))
    all_etf_tickers.update(holdings.get('ibb', []))
    all_etf_tickers.update(holdings.get('nbi', []))
    
    print(f"\n📊 Complete ETF universe: {len(all_etf_tickers)} tickers")
    
    # Load current universe
    universe_file = Path('production_data/universe.json')
    if not universe_file.exists():
        print(f"\n❌ Universe file not found: {universe_file}")
        print(f"\nCreate production_data/ directory and add universe.json")
        return 1
    
    with open(universe_file) as f:
        universe = json.load(f)
    
    # Extract current tickers
    current_tickers = set()
    for security in universe:
        ticker = security.get('ticker')
        if ticker and ticker != '_XBI_BENCHMARK_':
            current_tickers.add(ticker)
    
    print(f"📊 Current universe: {len(current_tickers)} tickers")
    
    # Find missing tickers
    missing_tickers = all_etf_tickers - current_tickers
    
    if not missing_tickers:
        print(f"\n✅ No new tickers to add - universe is already complete!")
        return 0
    
    print(f"📊 Missing from universe: {len(missing_tickers)} tickers")
    print(f"\n📝 Tickers to add (first 20):")
    for i, ticker in enumerate(sorted(missing_tickers)[:20], 1):
        # Determine which ETFs contain this ticker
        sources = []
        if ticker in holdings.get('xbi', []):
            sources.append('XBI')
        if ticker in holdings.get('ibb', []):
            sources.append('IBB')
        if ticker in holdings.get('nbi', []):
            sources.append('NBI')
        
        print(f"   {i:2d}. {ticker:6s}  ({', '.join(sources)})")
    
    if len(missing_tickers) > 20:
        print(f"   ... and {len(missing_tickers) - 20} more")
    
    # Ask for confirmation
    print(f"\n" + "="*80)
    response = input(f"Add {len(missing_tickers)} tickers to universe? (yes/no): ")
    
    if response.lower() not in ['yes', 'y']:
        print(f"❌ Cancelled - no changes made")
        return 0
    
    # Add missing tickers
    added_count = 0
    today = date.today().isoformat()
    
    for ticker in sorted(missing_tickers):
        # Determine which ETFs contain this ticker
        sources = []
        if ticker in holdings.get('xbi', []):
            sources.append('XBI')
        if ticker in holdings.get('ibb', []):
            sources.append('IBB')
        if ticker in holdings.get('nbi', []):
            sources.append('NBI')
        
        # Create new security entry
        new_security = {
            'ticker': ticker,
            'name': f'{ticker}',  # Will be populated by data collection
            'exchange': 'NASDAQ',  # Most biotech are NASDAQ (update later if needed)
            'sector': 'Biotechnology',
            'status': 'active',
            'added_from_etf': True,
            'added_date': today,
            'etf_sources': sources,
            'market_cap': None,  # Populate later
            'description': f'Added from ETF holdings ({", ".join(sources)})'
        }
        
        universe.append(new_security)
        added_count += 1
    
    # Backup original
    backup_file = universe_file.parent / f'universe_backup_{today}.json'
    with open(backup_file, 'w') as f:
        json.dump(json.load(open(universe_file)), f, indent=2)
    
    print(f"\n✅ Backup saved to: {backup_file}")
    
    # Save expanded universe
    with open(universe_file, 'w') as f:
        json.dump(universe, f, indent=2)
    
    print(f"✅ Updated universe saved to: {universe_file}")
    
    # Summary
    print(f"\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Original universe: {len(current_tickers)} tickers")
    print(f"Added: {added_count} tickers")
    print(f"New universe: {len(universe)} tickers")
    print(f"ETF coverage: 100% ✅")
    print("="*80)
    
    # Next steps
    print(f"\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print(f"1. Populate data for new tickers:")
    print(f"   python collect_financial_data.py")
    print(f"   python collect_ctgov_data.py --output production_data/trial_records.json")
    print(f"\n2. Re-run screening:")
    print(f"   python run_screen.py --as-of-date {today} --data-dir production_data --output screening_complete.json")
    print(f"\n3. Expect Module 1 to filter many new tickers:")
    print(f"   - Recent IPOs (no financial data yet)")
    print(f"   - Platform companies (no clinical trials)")
    print(f"   - Pre-clinical (no Phase 1+ trials)")
    print(f"   - Illiquid names")
    print(f"\n4. Final active universe will be ~150-200 tickers (correct!)")
    print("="*80 + "\n")
    
    return 0


if __name__ == "__main__":
    exit(add_etf_tickers_to_universe())
