#!/usr/bin/env python3
"""
diagnose_module2_coverage.py

Diagnose why Module 2 shows "Scored: 0"
Same pattern as our Module 4 fix - check for field mismatches.
"""

import json
import sys
from pathlib import Path
from collections import Counter

def diagnose_module2(data_dir="production_data"):
    """Diagnose Module 2 financial data coverage."""
    
    print("="*80)
    print("MODULE 2 (FINANCIAL) DIAGNOSTIC")
    print("="*80)
    print()
    
    # Check what Module 2 expects
    print("Step 1: Check what Module 2 expects...")
    print()
    
    try:
        from module_2_financial import compute_module_2_financial
        import inspect
        sig = inspect.signature(compute_module_2_financial)
        print(f"Module 2 signature: {sig}")
        print()
        
        # Get docstring hints
        doc = compute_module_2_financial.__doc__
        if doc:
            print("Expected input format:")
            print(doc[:500])
            print()
    except Exception as e:
        print(f"⚠️  Could not inspect Module 2: {e}")
        print()
    
    # Check what files exist
    print("Step 2: Check what data files exist...")
    print()
    
    data_path = Path(data_dir)
    files = list(data_path.glob("*.json"))
    
    for f in sorted(files):
        size = f.stat().st_size
        print(f"  {'✅' if size > 100 else '⚠️'} {f.name:<30} ({size:>10,} bytes)")
    print()
    
    # Check if financial.json exists
    financial_file = data_path / "financial.json"
    if not financial_file.exists():
        print("❌ financial.json NOT FOUND")
        print()
        print("Module 2 probably expects 'financial.json' but you don't have it.")
        print()
        print("Options:")
        print("  A. Transform universe.json financial data → financial_records.json")
        print("  B. Update run_screen.py to use different source")
        print("  C. Create placeholder financial.json")
        print()
    else:
        print("✅ financial.json exists")
        print()
        
        # Analyze its structure
        with open(financial_file) as f:
            financial_data = json.load(f)
        
        print(f"Records: {len(financial_data)}")
        
        if financial_data:
            sample = financial_data[0]
            print(f"Sample record keys: {list(sample.keys())}")
            print()
            
            # Check key fields
            required_fields = ['ticker', 'cash', 'debt', 'revenue_ttm', 'market_cap']
            
            field_coverage = {}
            for field in required_fields:
                count = sum(1 for r in financial_data if r.get(field) is not None)
                field_coverage[field] = count
                pct = count / len(financial_data) * 100
                status = "✅" if pct > 80 else ("⚠️" if pct > 20 else "❌")
                print(f"  {status} {field:<20} {count}/{len(financial_data)} ({pct:.0f}%)")
            print()
    
    # Check universe.json for financial data
    print("Step 3: Check universe.json for embedded financial data...")
    print()
    
    universe_file = data_path / "universe.json"
    if universe_file.exists():
        with open(universe_file) as f:
            universe = json.load(f)
        
        print(f"Universe records: {len(universe)}")
        
        # Check for financial data in universe
        has_financials = sum(1 for s in universe if s.get('financials'))
        has_financial_data = sum(1 for s in universe if s.get('financial_data'))
        has_market_data = sum(1 for s in universe if s.get('market_data'))
        
        print(f"  With 'financials' field: {has_financials}/{len(universe)} ({has_financials/len(universe)*100:.0f}%)")
        print(f"  With 'financial_data' field: {has_financial_data}/{len(universe)} ({has_financial_data/len(universe)*100:.0f}%)")
        print(f"  With 'market_data' field: {has_market_data}/{len(universe)} ({has_market_data/len(universe)*100:.0f}%)")
        print()
        
        if has_financial_data > 0 or has_market_data > 0:
            print("✅ Financial data exists in universe.json!")
            print()
            print("SOLUTION: Transform universe.json → financial_records.json")
            print()
            
            # Show sample
            sample = next((s for s in universe if s.get('financial_data') or s.get('market_data')), None)
            if sample:
                print("Sample financial data structure:")
                fin = sample.get('financial_data') or sample.get('financials')
                if fin:
                    print(json.dumps(fin, indent=2)[:300])
                print()
        elif has_financials > 0:
            print("⚠️  Financial data exists as 'financials' not 'financial_data'")
            print()
            print("SOLUTION: Map 'financials' → 'financial_data' (same as Module 4 fix)")
            print()
    
    # Generate fix recommendations
    print("="*80)
    print("RECOMMENDED FIX")
    print("="*80)
    print()
    
    if not financial_file.exists():
        if has_financial_data > 0 or has_market_data > 0:
            print("Create financial_records.json from universe.json:")
            print()
            print("  python transform_financial_for_module2.py")
            print()
            print("Then update run_screen.py line 162:")
            print("  FROM: financial_records = load_json_data(data_dir / 'financial.json', ...)")
            print("  TO:   financial_records = load_json_data(data_dir / 'financial_records.json', ...)")
        else:
            print("No financial data found anywhere!")
            print()
            print("You need to collect financial data first:")
            print("  python collect_all_universe_data.py")
            print()
            print("(This will populate universe.json with financial data)")
    else:
        print("financial.json exists but Module 2 still scores 0.")
        print()
        print("Check Module 2 code for field name mismatches:")
        print("  Get-Content module_2_financial.py | Select-String 'cash|debt|revenue'")
        print()
        print("Common issues:")
        print("  • Module expects 'totalCash' but data has 'cash'")
        print("  • Module expects nested structure but data is flat")
        print("  • Module silently skips records with missing required fields")
    
    print()


def create_financial_records_from_universe(
    universe_path="production_data/universe.json",
    output_path="production_data/financial_records.json"
):
    """Transform universe.json financial data into Module 2 format."""
    
    print("="*80)
    print("CREATING FINANCIAL_RECORDS.JSON FROM UNIVERSE")
    print("="*80)
    print()
    
    with open(universe_path) as f:
        universe = json.load(f)
    
    print(f"Loaded {len(universe)} securities")
    print()
    
    financial_records = []
    
    for security in universe:
        ticker = security.get('ticker', 'UNKNOWN')
        
        # Skip benchmark
        if ticker == '_XBI_BENCHMARK_':
            continue
        
        # Get financial data from various possible locations
        fin_data = (
            security.get('financial_data') or
            security.get('financials') or
            {}
        )
        
        market_data = security.get('market_data') or {}
        
        # Create record
        record = {
            'ticker': ticker,
            'as_of_date': security.get('as_of_date'),
            
            # Financial fields
            'cash': fin_data.get('cash'),
            'debt': fin_data.get('debt'),
            'net_debt': fin_data.get('net_debt'),
            'revenue_ttm': fin_data.get('revenue_ttm'),
            'assets': fin_data.get('assets'),
            'liabilities': fin_data.get('liabilities'),
            'equity': fin_data.get('equity'),
            
            # Market fields
            'market_cap': market_data.get('market_cap'),
            'price': market_data.get('price'),
            'shares_outstanding': market_data.get('shares_outstanding'),
            
            # Meta
            'currency': fin_data.get('currency', 'USD'),
            'cik': fin_data.get('cik'),
        }
        
        # Only add if has some data
        if any([record['cash'], record['debt'], record['revenue_ttm'], record['market_cap']]):
            financial_records.append(record)
            print(f"  ✓ {ticker}: Has financial data")
        else:
            print(f"  ⚠️ {ticker}: No financial data")
    
    # Save
    with open(output_path, 'w') as f:
        json.dump(financial_records, f, indent=2)
    
    print()
    print(f"✅ Created {len(financial_records)} financial records")
    print(f"   Saved to: {output_path}")
    print()
    
    # Show coverage
    has_cash = sum(1 for r in financial_records if r['cash'])
    has_debt = sum(1 for r in financial_records if r['debt'])
    has_revenue = sum(1 for r in financial_records if r['revenue_ttm'])
    has_market_cap = sum(1 for r in financial_records if r['market_cap'])
    
    print("Field coverage:")
    print(f"  Cash: {has_cash}/{len(financial_records)} ({has_cash/len(financial_records)*100:.0f}%)")
    print(f"  Debt: {has_debt}/{len(financial_records)} ({has_debt/len(financial_records)*100:.0f}%)")
    print(f"  Revenue: {has_revenue}/{len(financial_records)} ({has_revenue/len(financial_records)*100:.0f}%)")
    print(f"  Market cap: {has_market_cap}/{len(financial_records)} ({has_market_cap/len(financial_records)*100:.0f}%)")
    print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Diagnose Module 2 coverage")
    parser.add_argument("--fix", action="store_true",
                       help="Create financial_records.json from universe.json")
    parser.add_argument("--data-dir", default="production_data",
                       help="Data directory")
    args = parser.parse_args()
    
    if args.fix:
        create_financial_records_from_universe()
    else:
        diagnose_module2(args.data_dir)
