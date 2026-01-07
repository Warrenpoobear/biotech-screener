#!/usr/bin/env python3
"""
production_validation.py

Add this to the end of every screening run.
Catches silent failures before they reach production.
"""

from decimal import Decimal
from typing import Dict, Any, List
import json


def validate_screening_output(
    result: Dict[str, Any],
    as_of_date: str,
    config: Dict[str, Any]
) -> bool:
    """
    Validate screening output against production invariants.
    
    Returns:
        True if all checks pass, False otherwise
    
    Invariants checked:
        1. Weight sum == 1 - cash_target
        2. Excluded weights == 0
        3. Top-N count == N (if enabled)
        4. Module coverage rates reasonable
        5. PIT filtering sanity
        6. Data freshness consistency
    """
    
    print("\n" + "="*80)
    print("PRODUCTION VALIDATION")
    print("="*80)
    print()
    
    checks = []
    m5 = result.get('module_5_composite', {})
    ranked = m5.get('ranked_securities', [])
    
    if not ranked:
        print("❌ CRITICAL: No securities ranked!")
        return False
    
    # ========================================================================
    # INVARIANT 1: Weight Sum
    # ========================================================================
    total_weight = sum(Decimal(str(s['position_weight'])) for s in ranked)
    cash_target = Decimal(str(config.get('cash_target', '0.10')))
    expected_weight = Decimal('1.0') - cash_target
    weight_diff = abs(total_weight - expected_weight)
    weight_tolerance = Decimal('0.0015')  # 15 bps tolerance
    
    weight_check = weight_diff < weight_tolerance
    checks.append(('Weight sum', weight_check))
    
    status = "✅" if weight_check else "❌"
    print(f"{status} Weight Sum: {float(total_weight):.4f} (expected {float(expected_weight):.4f}, diff {float(weight_diff):.4f})")
    
    if not weight_check:
        print(f"   FAIL: Weight difference {float(weight_diff):.4f} exceeds tolerance {float(weight_tolerance):.4f}")
    
    # ========================================================================
    # INVARIANT 2: Excluded Weights
    # ========================================================================
    excluded = [s for s in ranked if not s.get('rankable', True)]
    excluded_weight = sum(Decimal(str(s['position_weight'])) for s in excluded)
    
    excluded_check = excluded_weight == 0
    checks.append(('Excluded weights zero', excluded_check))
    
    status = "✅" if excluded_check else "❌"
    print(f"{status} Excluded Weights: {float(excluded_weight):.4f} (expected 0.0000)")
    
    if not excluded_check:
        print(f"   FAIL: {len(excluded)} excluded securities have non-zero weights")
        for s in excluded[:3]:
            print(f"     {s['ticker']}: {s['position_weight']}")
    
    # ========================================================================
    # INVARIANT 3: Top-N Count
    # ========================================================================
    top_n = config.get('top_n')
    if top_n:
        invested = sum(1 for s in ranked if Decimal(str(s['position_weight'])) > 0)
        excluded_by_topn = sum(1 for s in ranked if 'NOT_IN_TOP_N' in s.get('position_flags', []))
        
        topn_check = invested == top_n
        checks.append(('Top-N count', topn_check))
        
        status = "✅" if topn_check else "❌"
        print(f"{status} Top-N Count: {invested} invested (expected {top_n})")
        
        if not topn_check:
            print(f"   FAIL: Expected {top_n} positions, got {invested}")
            print(f"   Excluded by top-N: {excluded_by_topn}")
        
        # Additional check: Excluded by top-N should have zero weight
        topn_excluded_with_weight = sum(1 for s in ranked 
                                       if 'NOT_IN_TOP_N' in s.get('position_flags', [])
                                       and Decimal(str(s['position_weight'])) > 0)
        if topn_excluded_with_weight > 0:
            print(f"   ⚠️  {topn_excluded_with_weight} securities marked NOT_IN_TOP_N but have weight!")
    
    # ========================================================================
    # INVARIANT 4: Module Coverage
    # ========================================================================
    print()
    print("📊 Module Coverage:")
    
    m1_diag = result.get('module_1_universe', {}).get('diagnostic_counts', {})
    m2_diag = result.get('module_2_financial', {}).get('diagnostic_counts', {})
    m3_diag = result.get('module_3_catalyst', {}).get('diagnostic_counts', {})
    m4_diag = result.get('module_4_clinical_dev', {}).get('diagnostic_counts', {})
    
    universe_count = len(ranked)
    
    # Module 1
    m1_active = m1_diag.get('active', 0)
    print(f"  Module 1 (Universe):  {m1_active} active")
    
    # Module 2
    m2_scored = m2_diag.get('scored', 0)
    m2_pct = m2_scored / universe_count * 100 if universe_count > 0 else 0
    m2_check = m2_pct > 80  # Expect >80% coverage
    checks.append(('Module 2 coverage >80%', m2_check))
    
    status = "✅" if m2_check else "⚠️"
    print(f"  {status} Module 2 (Financial): {m2_scored}/{universe_count} ({m2_pct:.0f}%)")
    
    if not m2_check and m2_scored == 0:
        print(f"     ❌ ZERO coverage! Check field mapping & data source")
    
    # Module 3
    m3_with_catalyst = m3_diag.get('with_catalyst', 0)
    m3_pct = m3_with_catalyst / universe_count * 100 if universe_count > 0 else 0
    # Don't require 80% for catalysts (they're rarer)
    m3_check = True  # Informational only
    
    status = "✅" if m3_pct > 20 else "⚠️"
    print(f"  {status} Module 3 (Catalyst):  {m3_with_catalyst}/{universe_count} ({m3_pct:.0f}%)")
    
    # Module 4
    m4_scored = m4_diag.get('scored', 0)
    m4_trials = m4_diag.get('total_trials', 0)
    m4_pct = m4_scored / universe_count * 100 if universe_count > 0 else 0
    m4_check = m4_pct > 80  # Expect >80% coverage
    checks.append(('Module 4 coverage >80%', m4_check))
    
    status = "✅" if m4_check else "⚠️"
    print(f"  {status} Module 4 (Clinical):  {m4_scored}/{universe_count} ({m4_pct:.0f}%), {m4_trials} trials")
    
    if not m4_check:
        print(f"     ⚠️  Low coverage! Many stocks missing clinical data")
    
    # ========================================================================
    # INVARIANT 5: PIT Filtering Sanity
    # ========================================================================
    print()
    print("🗓️  Point-in-Time Discipline:")
    
    pit_filtered = m4_diag.get('pit_filtered', 0)
    trials_evaluated = m4_diag.get('total_trials', 0)
    
    # If we have lots of trials but filtered exactly 0, that's suspicious
    pit_check = not (trials_evaluated > 100 and pit_filtered == 0)
    checks.append(('PIT filtering plausible', pit_check))
    
    status = "✅" if pit_check else "⚠️"
    print(f"  {status} Trials evaluated: {trials_evaluated}")
    print(f"  {status} PIT filtered: {pit_filtered}")
    
    if not pit_check:
        print(f"     ⚠️  WARNING: {trials_evaluated} trials but 0 filtered suggests missing date fields")
        print(f"     This may allow lookahead bias. Add date collection to trials.")
    
    # ========================================================================
    # INVARIANT 6: Date Consistency
    # ========================================================================
    print()
    print("📅 Date Consistency:")
    
    m1_date = result.get('module_1_universe', {}).get('as_of_date')
    m2_date = result.get('module_2_financial', {}).get('as_of_date')
    m3_date = result.get('module_3_catalyst', {}).get('as_of_date')
    m4_date = result.get('module_4_clinical_dev', {}).get('as_of_date')
    m5_date = result.get('module_5_composite', {}).get('as_of_date')
    
    all_dates = [m1_date, m2_date, m3_date, m4_date, m5_date]
    dates_match = all(d == as_of_date for d in all_dates if d)
    checks.append(('All module dates match', dates_match))
    
    status = "✅" if dates_match else "❌"
    print(f"  {status} Analysis date: {as_of_date}")
    print(f"  {status} All modules match: {dates_match}")
    
    if not dates_match:
        print(f"     ❌ FAIL: Modules have mismatched dates!")
        for i, (name, date) in enumerate([
            ('Module 1', m1_date),
            ('Module 2', m2_date),
            ('Module 3', m3_date),
            ('Module 4', m4_date),
            ('Module 5', m5_date),
        ], 1):
            if date != as_of_date:
                print(f"       {name}: {date} (expected {as_of_date})")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print()
    print("="*80)
    
    passed = sum(1 for _, check in checks if check)
    total = len(checks)
    pass_rate = passed / total * 100 if total > 0 else 0
    
    if passed == total:
        print(f"✅ PASS: All {total} validation checks passed")
        print("="*80)
        return True
    else:
        print(f"⚠️  PARTIAL: {passed}/{total} checks passed ({pass_rate:.0f}%)")
        print()
        print("Failed checks:")
        for name, check in checks:
            if not check:
                print(f"  ❌ {name}")
        print("="*80)
        return False


def add_validation_to_pipeline():
    """
    Example of how to integrate into run_screen.py
    """
    example_code = """
# Add to end of run_screening_pipeline() in run_screen.py:

from production_validation import validate_screening_output

def run_screening_pipeline(...):
    # ... existing pipeline code ...
    
    output = {
        'as_of_date': as_of_date,
        'module_1_universe': m1_result,
        'module_2_financial': m2_result,
        'module_3_catalyst': m3_result,
        'module_4_clinical_dev': m4_result,
        'module_5_composite': m5_result,
        # ... other fields ...
    }
    
    # VALIDATE BEFORE RETURNING
    config = {
        'cash_target': '0.10',
        'top_n': 60,  # Or None if not using top-N
    }
    
    validate_screening_output(output, as_of_date, config)
    
    return output
"""
    print(example_code)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python production_validation.py <screening_output.json>")
        print()
        print("Or call from within your pipeline:")
        add_validation_to_pipeline()
        sys.exit(1)
    
    # Load screening output
    with open(sys.argv[1]) as f:
        result = json.load(f)
    
    # Run validation
    as_of_date = result.get('as_of_date', result.get('module_5_composite', {}).get('as_of_date'))
    
    config = {
        'cash_target': '0.10',
        'top_n': None,  # Set if using top-N
    }
    
    success = validate_screening_output(result, as_of_date, config)
    
    sys.exit(0 if success else 1)
