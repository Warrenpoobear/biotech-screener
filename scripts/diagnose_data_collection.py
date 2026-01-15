#!/usr/bin/env python3
"""
Data Collection Diagnostic Script
Analyzes what data exists and identifies gaps in the collection pipeline.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

def analyze_data_coverage(universe_path: str = "production_data/universe.json"):
    """Analyze data coverage across all modules."""
    
    print("="*100)
    print("DATA COLLECTION DIAGNOSTIC")
    print("="*100)
    print()
    
    # Load universe
    try:
        with open(universe_path, 'r') as f:
            universe = json.load(f)
    except FileNotFoundError:
        print(f"❌ ERROR: {universe_path} not found!")
        sys.exit(1)
    
    total = len(universe)
    print(f"Total securities in universe: {total}")
    print()
    
    # Check each data type
    print("-"*100)
    print("DATA COVERAGE BY MODULE")
    print("-"*100)
    
    # Financial data (Module 2)
    fin_complete = []
    fin_missing = []
    for sec in universe:
        ticker = sec.get('ticker', 'UNKNOWN')
        if sec.get('financial_data'):
            fin_complete.append(ticker)
        else:
            fin_missing.append(ticker)
    
    print(f"\n📊 FINANCIAL DATA (Module 2):")
    print(f"  Complete: {len(fin_complete)}/{total} ({len(fin_complete)/total*100:.1f}%)")
    print(f"  Missing:  {len(fin_missing)}/{total} ({len(fin_missing)/total*100:.1f}%)")
    if len(fin_complete) > 0 and len(fin_complete) <= 10:
        print(f"  Stocks WITH data: {', '.join(fin_complete)}")
    if len(fin_missing) > 0 and len(fin_missing) <= 10:
        print(f"  Stocks MISSING data: {', '.join(fin_missing)}")
    
    # Catalyst data (Module 3)
    cat_complete = []
    cat_missing = []
    for sec in universe:
        ticker = sec.get('ticker', 'UNKNOWN')
        if sec.get('catalyst_data'):
            cat_complete.append(ticker)
        else:
            cat_missing.append(ticker)
    
    print(f"\n📅 CATALYST DATA (Module 3):")
    print(f"  Complete: {len(cat_complete)}/{total} ({len(cat_complete)/total*100:.1f}%)")
    print(f"  Missing:  {len(cat_missing)}/{total} ({len(cat_missing)/total*100:.1f}%)")
    if len(cat_complete) > 0 and len(cat_complete) <= 10:
        print(f"  Stocks WITH data: {', '.join(cat_complete)}")
    
    # Clinical data (Module 4)
    clin_complete = []
    clin_missing = []
    for sec in universe:
        ticker = sec.get('ticker', 'UNKNOWN')
        if sec.get('clinical_data'):
            clin_complete.append(ticker)
        else:
            clin_missing.append(ticker)
    
    print(f"\n🔬 CLINICAL DATA (Module 4):")
    print(f"  Complete: {len(clin_complete)}/{total} ({len(clin_complete)/total*100:.1f}%)")
    print(f"  Missing:  {len(clin_missing)}/{total} ({len(clin_missing)/total*100:.1f}%)")
    if len(clin_complete) > 0 and len(clin_complete) <= 10:
        print(f"  Stocks WITH data: {', '.join(clin_complete)}")
    
    # Defensive features
    def_complete = []
    def_missing = []
    for sec in universe:
        ticker = sec.get('ticker', 'UNKNOWN')
        if sec.get('defensive_features'):
            def_complete.append(ticker)
        else:
            def_missing.append(ticker)
    
    print(f"\n🛡️  DEFENSIVE FEATURES:")
    print(f"  Complete: {len(def_complete)}/{total} ({len(def_complete)/total*100:.1f}%)")
    print(f"  Missing:  {len(def_missing)}/{total} ({len(def_missing)/total*100:.1f}%)")
    if len(def_missing) > 0 and len(def_missing) <= 10:
        print(f"  Stocks MISSING data: {', '.join(def_missing)}")
    
    # Identify complete vs incomplete stocks
    print()
    print("-"*100)
    print("STOCKS BY COMPLETENESS")
    print("-"*100)
    
    complete_all = []
    partial = []
    defensive_only = []
    
    for sec in universe:
        ticker = sec.get('ticker', 'UNKNOWN')
        has_fin = bool(sec.get('financial_data'))
        has_cat = bool(sec.get('catalyst_data'))
        has_clin = bool(sec.get('clinical_data'))
        has_def = bool(sec.get('defensive_features'))
        
        if has_fin and has_cat and has_clin and has_def:
            complete_all.append(ticker)
        elif has_def and not (has_fin or has_cat or has_clin):
            defensive_only.append(ticker)
        elif has_def:
            partial.append(ticker)
    
    print(f"\n✅ COMPLETE (All 4 modules): {len(complete_all)} stocks")
    if len(complete_all) > 0:
        print(f"   {', '.join(complete_all)}")
    
    print(f"\n⚠️  PARTIAL (Defensive + some modules): {len(partial)} stocks")
    if len(partial) > 0 and len(partial) <= 20:
        print(f"   {', '.join(partial)}")
    
    print(f"\n❌ DEFENSIVE ONLY (No fundamental data): {len(defensive_only)} stocks")
    if len(defensive_only) > 0 and len(defensive_only) <= 20:
        print(f"   {', '.join(defensive_only[:20])}")
        if len(defensive_only) > 20:
            print(f"   ... and {len(defensive_only) - 20} more")
    
    # Examine a sample stock with no data
    print()
    print("-"*100)
    print("SAMPLE STOCK EXAMINATION")
    print("-"*100)
    
    if len(defensive_only) > 0:
        sample = defensive_only[0]
        sample_data = next(s for s in universe if s.get('ticker') == sample)
        
        print(f"\nExamining {sample} (defensive-only stock):")
        print(f"  Fields present: {list(sample_data.keys())}")
        print(f"  Ticker: {sample_data.get('ticker')}")
        print(f"  Name: {sample_data.get('name', 'N/A')}")
        print(f"  CUSIP: {sample_data.get('cusip', 'N/A')}")
        print(f"  Has defensive_features: {bool(sample_data.get('defensive_features'))}")
        print(f"  Has financial_data: {bool(sample_data.get('financial_data'))}")
        print(f"  Has catalyst_data: {bool(sample_data.get('catalyst_data'))}")
        print(f"  Has clinical_data: {bool(sample_data.get('clinical_data'))}")
    
    if len(complete_all) > 0:
        complete = complete_all[0]
        complete_data = next(s for s in universe if s.get('ticker') == complete)
        
        print(f"\nExamining {complete} (complete stock):")
        print(f"  Fields present: {list(complete_data.keys())}")
        
        if complete_data.get('financial_data'):
            fin = complete_data['financial_data']
            print(f"  Financial data keys: {list(fin.keys())}")
        
        if complete_data.get('catalyst_data'):
            cat = complete_data['catalyst_data']
            print(f"  Catalyst data keys: {list(cat.keys())}")
        
        if complete_data.get('clinical_data'):
            clin = complete_data['clinical_data']
            print(f"  Clinical data keys: {list(clin.keys())}")
    
    # Summary
    print()
    print("="*100)
    print("SUMMARY & RECOMMENDATIONS")
    print("="*100)
    print()
    
    if len(complete_all) < 10:
        print("⚠️  CRITICAL: Only {} stocks have complete data (<10%)".format(len(complete_all)))
        print()
        print("LIKELY CAUSES:")
        print("  1. Data collection scripts haven't been run for most stocks")
        print("  2. API rate limits preventing data collection")
        print("  3. Data collection scripts have bugs/errors")
        print("  4. Wrong API keys or credentials")
        print()
        print("NEXT STEPS:")
        print("  1. Check if data collection scripts exist")
        print("  2. Review logs for errors during collection")
        print("  3. Test collection on a single stock manually")
        print("  4. Fix issues and re-run full collection")
    elif len(complete_all) < total * 0.5:
        print("⚠️  WARNING: Only {}/{} stocks have complete data (<50%)".format(len(complete_all), total))
        print()
        print("This will significantly impact screening quality.")
        print("Recommend fixing data collection before proceeding.")
    else:
        print("✅ GOOD: {}/{} stocks have complete data ({}%)".format(
            len(complete_all), total, len(complete_all)/total*100))
        print()
        print("Data coverage is sufficient for screening.")
    
    print()
    print("="*100)
    
    return {
        'total': total,
        'complete': complete_all,
        'partial': partial,
        'defensive_only': defensive_only,
        'financial_complete': fin_complete,
        'catalyst_complete': cat_complete,
        'clinical_complete': clin_complete,
        'defensive_complete': def_complete,
    }


if __name__ == "__main__":
    import sys
    
    universe_path = "production_data/universe.json"
    if len(sys.argv) > 1:
        universe_path = sys.argv[1]
    
    analyze_data_coverage(universe_path)
