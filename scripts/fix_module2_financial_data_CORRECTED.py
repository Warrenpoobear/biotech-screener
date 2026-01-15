#!/usr/bin/env python3
"""
fix_module2_financial_data_CORRECTED.py

Transform universe.json → financial_records.json with CORRECT field names.

CRITICAL FIX: Module 2 expects _usd suffix fields!
- cash_usd (not cash)
- debt_usd (not debt)
- ttm_revenue_usd (not revenue_ttm)
- market_cap_usd (not market_cap)
"""

import json
from pathlib import Path


def transform_universe_to_financial_records(
    universe_path="production_data/universe.json",
    output_path="production_data/financial_records.json"
):
    """
    Transform universe.json to financial_records.json with Module 2 expected format.
    
    Module 2 expects (see line 43-46 of module_2_financial.py):
        cash_usd, debt_usd, ttm_revenue_usd, market_cap_usd
    """
    
    print("="*80)
    print("CREATING FINANCIAL_RECORDS.JSON WITH CORRECT FIELD NAMES")
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
        
        # Create record with Module 2 EXPECTED field names (with _usd suffix)
        record = {
            # Required fields
            'ticker': ticker,
            'as_of_date': security.get('as_of_date'),
            
            # Financial fields with _usd suffix (what Module 2 expects!)
            'cash_usd': fin_data.get('cash') or market_data.get('cash'),
            'debt_usd': fin_data.get('debt') or market_data.get('debt'),
            'ttm_revenue_usd': fin_data.get('revenue_ttm') or market_data.get('revenue_ttm'),
            'market_cap_usd': market_data.get('market_cap'),
            
            # Additional context fields
            'net_debt': fin_data.get('net_debt'),
            'assets': fin_data.get('assets'),
            'liabilities': fin_data.get('liabilities'),
            'equity': fin_data.get('equity'),
            
            # Metadata
            'currency': fin_data.get('currency', 'USD'),
            'cik': fin_data.get('cik'),
            'data_source': 'universe.json',
        }
        
        # Track coverage
        if record['market_cap_usd']: stats['has_market_cap'] += 1
        if record['cash_usd']: stats['has_cash'] += 1
        if record['debt_usd']: stats['has_debt'] += 1
        if record['ttm_revenue_usd']: stats['has_revenue'] += 1
        
        if not any([record['market_cap_usd'], record['cash_usd'], record['debt_usd'], record['ttm_revenue_usd']]):
            stats['completely_missing'] += 1
            print(f"  ⚠️  {ticker}: No financial data at all")
        else:
            has_fields = []
            if record['market_cap_usd']: has_fields.append('market_cap_usd')
            if record['cash_usd']: has_fields.append('cash_usd')
            if record['debt_usd']: has_fields.append('debt_usd')
            if record['ttm_revenue_usd']: has_fields.append('ttm_revenue_usd')
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
    print("Field Coverage (with correct _usd names):")
    print(f"  market_cap_usd:  {stats['has_market_cap']}/{total} ({stats['has_market_cap']/total*100:.0f}%)")
    print(f"  cash_usd:        {stats['has_cash']}/{total} ({stats['has_cash']/total*100:.0f}%)")
    print(f"  debt_usd:        {stats['has_debt']}/{total} ({stats['has_debt']/total*100:.0f}%)")
    print(f"  ttm_revenue_usd: {stats['has_revenue']}/{total} ({stats['has_revenue']/total*100:.0f}%)")
    print(f"  Missing all:     {stats['completely_missing']}/{total}")
    print()
    
    if stats['has_market_cap'] > total * 0.8:
        print("✅ GOOD: >80% have market_cap_usd")
    else:
        print("⚠️  WARNING: Low market_cap_usd coverage")
    
    print()
    print("="*80)
    print("NEXT STEP")
    print("="*80)
    print()
    print("Module 2 should now work!")
    print()
    print("Re-run:")
    print("  python run_screen.py --as-of-date 2026-01-06 --data-dir production_data --output screening_M2_FIXED.json")
    print()
    print("Expected: Module 2 scored 95/98 (was 0)")
    print()


if __name__ == "__main__":
    transform_universe_to_financial_records()
