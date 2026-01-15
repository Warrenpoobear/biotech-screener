#!/usr/bin/env python3
"""
test_pit_discipline.py - Regression Tests for Point-in-Time Discipline

Verifies that Module 4 correctly rejects future trial data to prevent lookahead bias.
"""
from __future__ import annotations

import sys
from decimal import Decimal
from typing import Any, Dict, List

# Import module to test
from module_4_clinical_dev import compute_module_4_clinical_dev


def test_future_trial_rejected():
    """Test that future trial data is rejected by PIT filter."""
    print("=" * 70)
    print("TEST: Future Trial Rejected (PIT Discipline)")
    print("=" * 70)
    
    as_of_date = "2024-12-15"
    active_tickers = ["TESTCO"]
    
    trial_records = [
        {
            "ticker": "TESTCO",
            "nct_id": "NCT006",
            "phase": "phase 2",
            "status": "recruiting",
            "last_update_posted": "2025-01-01",  # FUTURE DATE
            "randomized": True,
            "blinded": "double",
            "primary_endpoint": "overall survival",
        }
    ]
    
    print(f"\nSetup:")
    print(f"  as_of_date:         {as_of_date}")
    print(f"  Trial update date:  {trial_records[0]['last_update_posted']}")
    print(f"  Expected:           Trial REJECTED (future data)")
    
    result = compute_module_4_clinical_dev(
        trial_records=trial_records,
        active_tickers=active_tickers,
        as_of_date=as_of_date,
    )
    
    print(f"\nResults:")
    print(f"  PIT filtered:       {result['diagnostic_counts']['pit_filtered']}")
    print(f"  Total trials:       {result['diagnostic_counts']['total_trials']}")
    print(f"  Ticker trial count: {result['scores'][0]['trial_count']}")
    
    assert result["diagnostic_counts"]["pit_filtered"] == 1
    assert result["diagnostic_counts"]["total_trials"] == 0
    assert result["scores"][0]["trial_count"] == 0
    
    print("\n✅ PASS: Future trial correctly rejected by PIT filter")
    print("=" * 70)


def test_past_trial_accepted():
    """Test that past trial data is accepted by PIT filter."""
    print("\n" + "=" * 70)
    print("TEST: Past Trial Accepted (PIT Discipline)")
    print("=" * 70)
    
    as_of_date = "2024-12-15"
    active_tickers = ["TESTCO"]
    
    trial_records = [
        {
            "ticker": "TESTCO",
            "nct_id": "NCT007",
            "phase": "phase 2",
            "status": "recruiting",
            "last_update_posted": "2024-12-01",  # PAST DATE
            "randomized": True,
            "blinded": "double",
            "primary_endpoint": "overall survival",
        }
    ]
    
    print(f"\nSetup:")
    print(f"  as_of_date:         {as_of_date}")
    print(f"  Trial update date:  {trial_records[0]['last_update_posted']}")
    print(f"  Expected:           Trial ACCEPTED (past data)")
    
    result = compute_module_4_clinical_dev(
        trial_records=trial_records,
        active_tickers=active_tickers,
        as_of_date=as_of_date,
    )
    
    print(f"\nResults:")
    print(f"  PIT filtered:       {result['diagnostic_counts']['pit_filtered']}")
    print(f"  Total trials:       {result['diagnostic_counts']['total_trials']}")
    
    assert result["diagnostic_counts"]["pit_filtered"] == 0
    assert result["diagnostic_counts"]["total_trials"] == 1
    assert result["scores"][0]["trial_count"] == 1
    
    print("\n✅ PASS: Past trial correctly accepted by PIT filter")
    print("=" * 70)


def test_mixed_pit_boundary():
    """Test PIT filter with multiple trials at boundary."""
    print("\n" + "=" * 70)
    print("TEST: Mixed Trials at PIT Boundary")
    print("=" * 70)
    
    as_of_date = "2024-12-15"
    active_tickers = ["TESTCO"]
    
    trial_records = [
        {
            "ticker": "TESTCO",
            "nct_id": "NCT008",
            "phase": "phase 3",
            "status": "recruiting",
            "last_update_posted": "2024-12-10",  # PAST
            "randomized": True,
            "blinded": "double",
            "primary_endpoint": "overall survival",
        },
        {
            "ticker": "TESTCO",
            "nct_id": "NCT009",
            "phase": "phase 2",
            "status": "recruiting",
            "last_update_posted": "2024-12-15",  # BOUNDARY
            "randomized": False,
            "blinded": "none",
            "primary_endpoint": "safety",
        },
        {
            "ticker": "TESTCO",
            "nct_id": "NCT010",
            "phase": "phase 1",
            "status": "recruiting",
            "last_update_posted": "2024-12-16",  # FUTURE
            "randomized": False,
            "blinded": "none",
            "primary_endpoint": "pharmacokinetic",
        },
        {
            "ticker": "TESTCO",
            "nct_id": "NCT011",
            "phase": "phase 2",
            "status": "recruiting",
            "last_update_posted": "2025-01-01",  # FUTURE
            "randomized": True,
            "blinded": "single",
            "primary_endpoint": "progression-free survival",
        },
    ]
    
    print(f"\nSetup:")
    print(f"  as_of_date:   {as_of_date}")
    print(f"  Total trials: {len(trial_records)}")
    print(f"  Expected:     2 accepted, 2 rejected")
    
    result = compute_module_4_clinical_dev(
        trial_records=trial_records,
        active_tickers=active_tickers,
        as_of_date=as_of_date,
    )
    
    print(f"\nResults:")
    print(f"  PIT filtered:       {result['diagnostic_counts']['pit_filtered']}")
    print(f"  Total trials:       {result['diagnostic_counts']['total_trials']}")
    
    assert result["diagnostic_counts"]["pit_filtered"] == 2
    assert result["diagnostic_counts"]["total_trials"] == 2
    assert result["scores"][0]["trial_count"] == 2
    
    print("\n✅ PASS: Mixed trials correctly filtered at PIT boundary")
    print("=" * 70)


def test_no_source_date_handling():
    """Test handling of trials with missing source dates."""
    print("\n" + "=" * 70)
    print("TEST: Missing Source Date Handling")
    print("=" * 70)
    
    as_of_date = "2024-12-15"
    active_tickers = ["TESTCO"]
    
    trial_records = [
        {
            "ticker": "TESTCO",
            "nct_id": "NCT012",
            "phase": "phase 2",
            "status": "recruiting",
            # NO last_update_posted field
            "randomized": True,
            "blinded": "double",
            "primary_endpoint": "overall survival",
        }
    ]
    
    print(f"\nSetup:")
    print(f"  as_of_date:         {as_of_date}")
    print(f"  Trial update date:  <MISSING>")
    print(f"  Expected:           Trial ACCEPTED (no date check)")
    
    result = compute_module_4_clinical_dev(
        trial_records=trial_records,
        active_tickers=active_tickers,
        as_of_date=as_of_date,
    )
    
    print(f"\nResults:")
    print(f"  PIT filtered:       {result['diagnostic_counts']['pit_filtered']}")
    print(f"  Total trials:       {result['diagnostic_counts']['total_trials']}")
    
    assert result["diagnostic_counts"]["pit_filtered"] == 0
    assert result["diagnostic_counts"]["total_trials"] == 1
    
    print("\n✅ PASS: Missing source date handled correctly")
    print("=" * 70)


def run_all_tests() -> int:
    """Run all PIT discipline tests."""
    print("\n" + "=" * 70)
    print("PIT DISCIPLINE REGRESSION TEST SUITE")
    print("=" * 70)
    
    tests = [
        ("Future Trial Rejected", test_future_trial_rejected),
        ("Past Trial Accepted", test_past_trial_accepted),
        ("Mixed PIT Boundary", test_mixed_pit_boundary),
        ("Missing Source Date", test_no_source_date_handling),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            failed += 1
            print(f"\n❌ FAIL: {name}")
            print(f"   {e}")
        except Exception as e:
            failed += 1
            print(f"\n❌ ERROR: {name}")
            print(f"   {e}")
    
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run:     {passed + failed}")
    print(f"Passed:        {passed} ✅")
    print(f"Failed:        {failed}")
    print("=" * 70)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(run_all_tests())