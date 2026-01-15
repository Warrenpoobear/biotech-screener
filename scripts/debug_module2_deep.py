#!/usr/bin/env python3
"""
debug_module2_deep.py - Deep dive into why Module 2 scores 0
"""

import json
from pathlib import Path


def debug_module2():
    """Debug Module 2 scoring issue."""
    
    print("="*80)
    print("MODULE 2 DEEP DIAGNOSTIC")
    print("="*80)
    print()
    
    # Load data
    print("Step 1: Loading data...")
    print()
    
    universe_path = Path("production_data/universe.json")
    financial_path = Path("production_data/financial_records.json")
    
    with open(universe_path) as f:
        universe = json.load(f)
    
    with open(financial_path) as f:
        financial_records = json.load(f)
    
    print(f"✅ Universe: {len(universe)} records")
    print(f"✅ Financial: {len(financial_records)} records")
    print()
    
    # Get active tickers from universe
    active_tickers = {s['ticker'] for s in universe if s.get('ticker') != '_XBI_BENCHMARK_'}
    print(f"✅ Active tickers: {len(active_tickers)}")
    print(f"   Sample: {list(active_tickers)[:5]}")
    print()
    
    # Check ticker overlap
    print("Step 2: Checking ticker overlap...")
    print()
    
    financial_tickers = {r['ticker'] for r in financial_records}
    
    in_both = active_tickers & financial_tickers
    only_universe = active_tickers - financial_tickers
    only_financial = financial_tickers - active_tickers
    
    print(f"✅ In both: {len(in_both)}")
    print(f"⚠️  Only in universe: {len(only_universe)}")
    if only_universe:
        print(f"   {list(only_universe)[:5]}")
    print(f"⚠️  Only in financial: {len(only_financial)}")
    if only_financial:
        print(f"   {list(only_financial)[:5]}")
    print()
    
    # Check field requirements
    print("Step 3: Checking what fields records have...")
    print()
    
    sample_records = financial_records[:5]
    for rec in sample_records:
        ticker = rec.get('ticker')
        has_market_cap = rec.get('market_cap') is not None
        has_cash = rec.get('cash') is not None
        has_debt = rec.get('debt') is not None
        has_revenue = rec.get('revenue_ttm') is not None
        
        fields = []
        if has_market_cap: fields.append('market_cap')
        if has_cash: fields.append('cash')
        if has_debt: fields.append('debt')
        if has_revenue: fields.append('revenue')
        
        print(f"  {ticker}: {', '.join(fields) if fields else 'NO FIELDS!'}")
    print()
    
    # Check data coverage
    print("Step 4: Checking field coverage...")
    print()
    
    stats = {
        'market_cap': 0,
        'cash': 0,
        'debt': 0,
        'revenue': 0,
        'any_field': 0,
    }
    
    for rec in financial_records:
        if rec.get('market_cap'): stats['market_cap'] += 1
        if rec.get('cash'): stats['cash'] += 1
        if rec.get('debt'): stats['debt'] += 1
        if rec.get('revenue_ttm'): stats['revenue'] += 1
        if any([rec.get('market_cap'), rec.get('cash'), rec.get('debt'), rec.get('revenue_ttm')]):
            stats['any_field'] += 1
    
    total = len(financial_records)
    print(f"Market cap:  {stats['market_cap']}/{total} ({stats['market_cap']/total*100:.0f}%)")
    print(f"Cash:        {stats['cash']}/{total} ({stats['cash']/total*100:.0f}%)")
    print(f"Debt:        {stats['debt']}/{total} ({stats['debt']/total*100:.0f}%)")
    print(f"Revenue:     {stats['revenue']}/{total} ({stats['revenue']/total*100:.0f}%)")
    print(f"Any field:   {stats['any_field']}/{total} ({stats['any_field']/total*100:.0f}%)")
    print()
    
    # Try calling Module 2 directly
    print("Step 5: Calling Module 2 directly...")
    print()
    
    try:
        from module_2_financial import compute_module_2_financial
        
        result = compute_module_2_financial(
            financial_records=financial_records,
            active_tickers=active_tickers,
            as_of_date='2026-01-06'
        )
        
        scored = result.get('diagnostic_counts', {}).get('scored', 0)
        print(f"Module 2 result: Scored {scored}/{len(active_tickers)}")
        print()
        
        if scored == 0:
            print("❌ Module 2 still scores 0!")
            print()
            print("Checking Module 2 source code for requirements...")
            print()
            
            # Look at what Module 2 actually does
            import inspect
            source = inspect.getsource(compute_module_2_financial)
            
            # Look for common filtering patterns
            if 'market_cap' in source:
                print("✓ Module 2 checks for market_cap")
            if 'cash' in source:
                print("✓ Module 2 checks for cash")
            if 'continue' in source:
                print("⚠️  Module 2 has 'continue' statements (skipping records)")
            if 'None' in source:
                print("⚠️  Module 2 checks for None values")
            
            print()
            print("LIKELY ISSUE:")
            print("Module 2 probably requires specific fields to be non-None")
            print("and is silently skipping all records that don't meet requirements.")
            print()
            print("Next step: Check what Module 2 actually requires!")
            
        else:
            print(f"✅ Module 2 works when called directly! Scored {scored}")
            print()
            print("The issue must be in how run_screen.py calls it.")
            
    except Exception as e:
        print(f"❌ Error calling Module 2: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    print("="*80)


if __name__ == "__main__":
    debug_module2()
