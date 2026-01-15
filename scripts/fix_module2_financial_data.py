#!/usr/bin/env python3
"""
fix_module2_financial_data.py

Transform universe.json → financial_records.json with correct field names.

Issue: Module 2 expects 'cash', 'debt' but data has 'cash_usd', 'debt_usd'
       AND financial_data in universe is mostly NULL
       BUT market_data has the real data!
"""

import json
from pathlib import Path


def transform_universe_to_financial_records(
    universe_path="production_data/universe.json",
    output_path="production_data/financial_records.json"
):
    """
    Transform universe.json to financial_records.json with Module 2 expected format.
    
    Module 2 expects:
        ticker, cash, debt, revenue_ttm, market_cap, etc.
    
    Universe has:
        market_data: {market_cap, price, shares_outstanding, ...}
        financial_data: {cash, debt, revenue_ttm, ...} ← Mostly NULL!
    
    We need to merge both and fix field names.
    """
    
    print("="*80)
    print("CREATING FINANCIAL_RECORDS.JSON FOR MODULE 2")
    print("="*80)
    print()
    
    with open(universe_path) as f:
        universe = json.load(f)
    
    print(f"Loaded {len(universe)} securities from universe.json")
    print()
    
    financial_records = []
    stats = {
        'has_market_cap': 0,
        'has_cash': 0,
        'has_debt': 0,
        'has_revenue': 0,
        'completely_missing': 0,
    }
    
    for security in universe:
        ticker = security.get('ticker', 'UNKNOWN')
        
        # Skip benchmark
        if ticker == '_XBI_BENCHMARK_':
            continue
        
        # Get both sources of financial data
        fin_data = security.get('financial_data') or {}
        market_data = security.get('market_data') or {}
        
        # Create record with Module 2 expected field names
        record = {
            # Required fields
            'ticker': ticker,
            'as_of_date': security.get('as_of_date'),
            
            # Financial fields (prefer financial_data, fallback to market_data)
            'cash': fin_data.get('cash') or market_data.get('cash'),
            'debt': fin_data.get('debt') or market_data.get('debt'),
            'net_debt': fin_data.get('net_debt'),
            'revenue_ttm': fin_data.get('revenue_ttm') or market_data.get('revenue_ttm'),
            'assets': fin_data.get('assets'),
            'liabilities': fin_data.get('liabilities'),
            'equity': fin_data.get('equity'),
            
            # Market fields
            'market_cap': market_data.get('market_cap'),
            'price': market_data.get('price'),
            'shares_outstanding': market_data.get('shares_outstanding'),
            
            # Additional fields Module 2 might use
            'currency': fin_data.get('currency', 'USD'),
            'cik': fin_data.get('cik'),
        }
        
        # Track coverage
        if record['market_cap']: stats['has_market_cap'] += 1
        if record['cash']: stats['has_cash'] += 1
        if record['debt']: stats['has_debt'] += 1
        if record['revenue_ttm']: stats['has_revenue'] += 1
        
        if not any([record['market_cap'], record['cash'], record['debt'], record['revenue_ttm']]):
            stats['completely_missing'] += 1
            print(f"  ⚠️  {ticker}: No financial data at all")
        else:
            has_fields = []
            if record['market_cap']: has_fields.append('market_cap')
            if record['cash']: has_fields.append('cash')
            if record['debt']: has_fields.append('debt')
            if record['revenue_ttm']: has_fields.append('revenue')
            print(f"  ✓ {ticker}: {', '.join(has_fields)}")
        
        financial_records.append(record)
    
    # Save
    with open(output_path, 'w') as f:
        json.dump(financial_records, f, indent=2)
    
    print()
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print()
    print(f"Created: {len(financial_records)} financial records")
    print(f"Saved to: {output_path}")
    print()
    
    total = len(financial_records)
    print("Field Coverage:")
    print(f"  Market cap:  {stats['has_market_cap']}/{total} ({stats['has_market_cap']/total*100:.0f}%)")
    print(f"  Cash:        {stats['has_cash']}/{total} ({stats['has_cash']/total*100:.0f}%)")
    print(f"  Debt:        {stats['has_debt']}/{total} ({stats['has_debt']/total*100:.0f}%)")
    print(f"  Revenue:     {stats['has_revenue']}/{total} ({stats['has_revenue']/total*100:.0f}%)")
    print(f"  Missing all: {stats['completely_missing']}/{total}")
    print()
    
    if stats['has_market_cap'] > total * 0.8:
        print("✅ GOOD: >80% have market cap")
    else:
        print("⚠️  WARNING: Low market cap coverage")
    
    print()
    print("="*80)
    print("NEXT STEP")
    print("="*80)
    print()
    print("Module 2 should now work!")
    print()
    print("Verify in run_screen.py line 162:")
    print("  Current: financial_records = load_json_data(data_dir / 'financial.json', ...)")
    print("  Should be: financial_records = load_json_data(data_dir / 'financial_records.json', ...)")
    print()
    print("Then re-run:")
    print("  python run_screen.py --as-of-date 2026-01-06 --data-dir production_data --output screening_M2_FIXED.json")
    print()


if __name__ == "__main__":
    transform_universe_to_financial_records()
