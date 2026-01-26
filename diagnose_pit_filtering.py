#!/usr/bin/env python3
"""
diagnose_pit_filtering.py

Diagnose PIT (Point-in-Time) filtering issues.
Checks if trial records have proper date fields for temporal discipline.
"""

import json
import sys
from datetime import datetime
from collections import Counter


def diagnose_pit_coverage(trial_records_path="production_data/trial_records.json"):
    """Analyze PIT date field coverage in trial records."""
    
    print("="*80)
    print("PIT FILTERING DIAGNOSTIC")
    print("="*80)
    print()
    
    # Load trials
    try:
        with open(trial_records_path) as f:
            trials = json.load(f)
    except FileNotFoundError:
        print(f"❌ File not found: {trial_records_path}")
        print()
        print("Run this first:")
        print("  python transform_clinical_for_module4.py")
        return
    
    print(f"Loaded {len(trials)} trial records")
    print()
    
    # Check date field coverage
    print("="*80)
    print("DATE FIELD COVERAGE")
    print("="*80)
    print()
    
    date_fields = [
        'last_update_posted',
        'study_first_posted',
        'results_first_posted',
        'source_date',
        'first_posted',
        'last_posted',
    ]
    
    coverage = {}
    for field in date_fields:
        count = sum(1 for t in trials if t.get(field) is not None and t.get(field) != '')
        coverage[field] = count
        pct = count / len(trials) * 100 if trials else 0
        
        status = "✅" if pct > 80 else ("⚠️" if pct > 0 else "❌")
        print(f"{status} {field:<25} {count}/{len(trials)} ({pct:.1f}%)")
    
    total_with_any_date = sum(1 for t in trials if any(t.get(field) for field in date_fields))
    pct_any = total_with_any_date / len(trials) * 100 if trials else 0
    
    print()
    print(f"{'✅' if pct_any > 80 else '❌'} ANY date field: {total_with_any_date}/{len(trials)} ({pct_any:.1f}%)")
    print()
    
    # Sample trials
    print("="*80)
    print("SAMPLE TRIALS")
    print("="*80)
    print()
    
    # Show one with date
    with_date = next((t for t in trials if any(t.get(f) for f in date_fields)), None)
    if with_date:
        print("✅ Trial WITH date:")
        print(f"  Ticker: {with_date.get('ticker')}")
        print(f"  NCT ID: {with_date.get('nct_id')}")
        for field in date_fields:
            if with_date.get(field):
                print(f"  {field}: {with_date.get(field)}")
        print()
    
    # Show one without date
    without_date = next((t for t in trials if not any(t.get(f) for f in date_fields)), None)
    if without_date:
        print("❌ Trial WITHOUT date:")
        print(f"  Ticker: {without_date.get('ticker')}")
        print(f"  NCT ID: {without_date.get('nct_id')}")
        for field in date_fields:
            val = without_date.get(field)
            print(f"  {field}: {val if val else 'None'}")
        print()
    
    # Parse date values
    print("="*80)
    print("DATE VALUE ANALYSIS")
    print("="*80)
    print()
    
    # Find most common date field
    best_field = max(coverage.items(), key=lambda x: x[1])[0] if coverage else None
    
    if best_field and coverage[best_field] > 0:
        print(f"Using '{best_field}' (most coverage)")
        print()
        
        date_values = [t.get(best_field) for t in trials if t.get(best_field)]
        
        # Try to parse
        parsed = []
        unparsed = []
        
        for val in date_values[:100]:  # Sample first 100
            try:
                # Try ISO format
                dt = datetime.fromisoformat(val.replace('Z', '+00:00'))
                parsed.append(dt)
            except (ValueError, TypeError):
                try:
                    # Try common formats
                    for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%Y/%m/%d']:
                        try:
                            dt = datetime.strptime(val, fmt)
                            parsed.append(dt)
                            break
                        except ValueError:
                            continue
                    else:
                        unparsed.append(val)
                except (ValueError, TypeError):
                    unparsed.append(val)
        
        parse_rate = len(parsed) / (len(parsed) + len(unparsed)) * 100 if (parsed or unparsed) else 0
        
        print(f"Parsed: {len(parsed)}/{len(parsed) + len(unparsed)} ({parse_rate:.0f}%)")
        
        if parsed:
            print(f"Date range: {min(parsed).date()} to {max(parsed).date()}")
            print()
        
        if unparsed:
            print(f"⚠️  Unparseable date formats found:")
            for val in unparsed[:5]:
                print(f"  '{val}'")
            print()
    
    # Diagnosis
    print("="*80)
    print("DIAGNOSIS")
    print("="*80)
    print()
    
    if pct_any < 10:
        print("❌ CRITICAL: <10% of trials have date fields")
        print()
        print("Impact:")
        print("  • PIT filtering cannot work")
        print("  • Potential lookahead bias (using future data)")
        print("  • Historical backtesting not possible")
        print()
        print("Fix:")
        print("  1. Update collect_all_universe_data.py to fetch date fields")
        print("  2. Re-run collection: python collect_all_universe_data.py")
        print("  3. Re-transform: python transform_clinical_for_module4.py")
        print("  4. Re-screen: python run_screen.py ...")
        print()
        
        print("Add to collect_clinical_data():")
        print("""
    status_module = protocol.get('statusModule', {})
    trial = {
        'nct_id': ...,
        'phase': ...,
        'status': ...,
        
        # ADD THESE:
        'last_update_posted': status_module.get('lastUpdatePostDate'),
        'study_first_posted': status_module.get('studyFirstPostDate'),
        'results_first_posted': status_module.get('resultsFirstPostDate'),
    }
""")
        
    elif pct_any < 80:
        print("⚠️  PARTIAL: Some trials have dates but coverage is incomplete")
        print()
        print("This means:")
        print("  • PIT filtering works for some trials")
        print("  • But ~20-80% of trials lack dates")
        print("  • May miss some future data filtering")
        print()
        print("Consider: Improve date collection coverage")
        
    else:
        print("✅ GOOD: >80% of trials have date fields")
        print()
        
        if parse_rate < 80:
            print("⚠️  But date parsing has issues!")
            print()
            print("Check date formats and add parsing logic to Module 4")
        else:
            print("✅ Date parsing works well")
            print()
            print("PIT filtering should be working correctly.")
            print()
            print("If Module 4 still shows 'PIT filtered: 0', check:")
            print("  1. Are all trials before as_of_date? (nothing to filter)")
            print("  2. Is Module 4 actually using the date field?")
            print("  3. Is the date comparison logic correct?")
    
    print()


def add_date_collection_example():
    """Show how to add date collection to data collector."""
    
    example = """
# In collect_all_universe_data.py, update collect_clinical_data():

def collect_clinical_data(ticker, company_name=None):
    try:
        import requests
        # ... existing search code ...
        
        for study in studies:
            protocol = study.get('protocolSection', {})
            id_module = protocol.get('identificationModule', {})
            status_module = protocol.get('statusModule', {})     # ← ADD THIS
            design_module = protocol.get('designModule', {})
            
            trial = {
                'nct_id': id_module.get('nctId'),
                'title': id_module.get('briefTitle'),
                'status': status_module.get('overallStatus'),
                'phase': (design_module.get('phases', ['N/A']) or ['N/A'])[0],
                
                # ADD THESE DATE FIELDS:
                'last_update_posted': status_module.get('lastUpdatePostDate'),
                'study_first_posted': status_module.get('studyFirstPostDate'),
                'results_first_posted': status_module.get('resultsFirstPostDate'),
            }
            
            # ... rest of code ...
"""
    
    print("="*80)
    print("HOW TO ADD DATE COLLECTION")
    print("="*80)
    print()
    print(example)
    print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Diagnose PIT filtering")
    parser.add_argument("--trials", default="production_data/trial_records.json",
                       help="Path to trial_records.json")
    parser.add_argument("--show-fix", action="store_true",
                       help="Show how to add date collection")
    args = parser.parse_args()
    
    if args.show_fix:
        add_date_collection_example()
    else:
        diagnose_pit_coverage(args.trials)
