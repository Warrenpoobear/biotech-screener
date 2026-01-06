#!/usr/bin/env python3
"""
Check if data exists in alternate field names in universe.json
"""

import json
import sys

def check_alternate_fields(universe_path="production_data/universe.json"):
    """Check if data exists but in different field names."""
    
    with open(universe_path) as f:
        universe = json.load(f)
    
    print("="*80)
    print("CHECKING FOR DATA IN ALTERNATE FIELDS")
    print("="*80)
    print()
    
    # Check first stock in detail
    sample = universe[0]
    ticker = sample.get('ticker', 'UNKNOWN')
    
    print(f"Examining {ticker} in detail:")
    print(f"Fields present: {list(sample.keys())}")
    print()
    
    # Check each potential field
    fields_to_check = {
        'financials': 'Financial data (expected: financial_data)',
        'financial_data': 'Financial data (correct name)',
        'market_data': 'Market data',
        'clinical': 'Clinical data (expected: clinical_data)',
        'clinical_data': 'Clinical data (correct name)',
        'catalyst_data': 'Catalyst data',
        'time_series': 'Time series data',
    }
    
    for field, description in fields_to_check.items():
        value = sample.get(field)
        if value:
            print(f"✅ {field} EXISTS:")
            print(f"   Description: {description}")
            print(f"   Type: {type(value)}")
            if isinstance(value, dict):
                print(f"   Keys: {list(value.keys())[:10]}")  # First 10 keys
            elif isinstance(value, list):
                print(f"   Length: {len(value)}")
                if len(value) > 0:
                    print(f"   First item type: {type(value[0])}")
            print()
        else:
            print(f"❌ {field} MISSING")
            print()
    
    # Count how many stocks have data in each field
    print("="*80)
    print("DATA COVERAGE BY FIELD")
    print("="*80)
    print()
    
    for field, description in fields_to_check.items():
        count = sum(1 for s in universe if s.get(field))
        pct = count / len(universe) * 100
        status = "✅" if count > len(universe) * 0.5 else ("⚠️" if count > 0 else "❌")
        print(f"{status} {field}: {count}/{len(universe)} ({pct:.1f}%)")
    
    # Show sample of financial data if it exists
    print()
    print("="*80)
    print("SAMPLE DATA STRUCTURES")
    print("="*80)
    print()
    
    if sample.get('financials'):
        print("FINANCIALS field structure:")
        print(json.dumps(sample['financials'], indent=2)[:500])
        print("..." if len(json.dumps(sample['financials'])) > 500 else "")
        print()
    
    if sample.get('market_data'):
        print("MARKET_DATA field structure:")
        print(json.dumps(sample['market_data'], indent=2)[:500])
        print("..." if len(json.dumps(sample['market_data'])) > 500 else "")
        print()
    
    if sample.get('clinical'):
        print("CLINICAL field structure:")
        print(json.dumps(sample['clinical'], indent=2)[:500])
        print("..." if len(json.dumps(sample['clinical'])) > 500 else "")
        print()

if __name__ == "__main__":
    check_alternate_fields()
