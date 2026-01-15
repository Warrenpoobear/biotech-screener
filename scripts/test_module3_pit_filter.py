#!/usr/bin/env python3
"""
test_module3_pit_filter.py - Verify Module 3 PIT Filter

Tests that Module 3 correctly filters trials by last_update_posted date.
"""
import sys
sys.path.insert(0, '/home/claude')  # Add working directory to path

from module_3_catalyst import compute_module_3_catalyst


def test_module3_future_trial_filtered():
    """Test that future trial announcements are filtered."""
    print("=" * 70)
    print("TEST: Module 3 PIT Filter - Future Trial Filtered")
    print("=" * 70)
    
    as_of_date = "2024-12-15"
    active_tickers = ["TEST"]
    
    trial_records = [
        {
            "ticker": "TEST",
            "nct_id": "NCT999",
            "phase": "phase 3",
            "status": "recruiting",
            "primary_completion_date": "2025-06-30",  # Future readout (OK - this is the catalyst)
            "last_update_posted": "2025-01-01",  # FUTURE DATA (should be filtered)
        }
    ]
    
    print(f"\nSetup:")
    print(f"  as_of_date:             {as_of_date}")
    print(f"  Trial completion date:  {trial_records[0]['primary_completion_date']} (future)")
    print(f"  Trial update date:      {trial_records[0]['last_update_posted']} (future)")
    print(f"  Expected:               Trial FILTERED (data not yet available)")
    
    result = compute_module_3_catalyst(
        trial_records=trial_records,
        active_tickers=active_tickers,
        as_of_date=as_of_date,
    )
    
    print(f"\nResults:")
    print(f"  PIT filtered:           {result['diagnostic_counts']['pit_filtered']}")
    print(f"  Trials evaluated:       {result['diagnostic_counts']['total_trials_evaluated']}")
    print(f"  With catalyst:          {result['diagnostic_counts']['with_catalyst']}")
    
    # Assertions
    assert result["diagnostic_counts"]["pit_filtered"] == 1, \
        f"Expected 1 PIT filtered trial, got {result['diagnostic_counts']['pit_filtered']}"
    
    assert result["diagnostic_counts"]["total_trials_evaluated"] == 0, \
        f"Expected 0 evaluated trials, got {result['diagnostic_counts']['total_trials_evaluated']}"
    
    assert result["diagnostic_counts"]["with_catalyst"] == 0, \
        f"Expected 0 with catalyst, got {result['diagnostic_counts']['with_catalyst']}"
    
    print("\n✅ PASS: Future trial data correctly filtered by Module 3 PIT filter")
    print("=" * 70)


def test_module3_past_trial_accepted():
    """Test that past trial announcements are accepted."""
    print("\n" + "=" * 70)
    print("TEST: Module 3 PIT Filter - Past Trial Accepted")
    print("=" * 70)
    
    as_of_date = "2024-12-15"
    active_tickers = ["TEST"]
    
    trial_records = [
        {
            "ticker": "TEST",
            "nct_id": "NCT888",
            "phase": "phase 3",
            "status": "recruiting",
            "primary_completion_date": "2025-06-30",  # Future readout (OK - this is the catalyst)
            "last_update_posted": "2024-11-01",  # PAST DATA (should be accepted)
        }
    ]
    
    print(f"\nSetup:")
    print(f"  as_of_date:             {as_of_date}")
    print(f"  Trial completion date:  {trial_records[0]['primary_completion_date']} (future)")
    print(f"  Trial update date:      {trial_records[0]['last_update_posted']} (past)")
    print(f"  Expected:               Trial ACCEPTED (data available)")
    
    result = compute_module_3_catalyst(
        trial_records=trial_records,
        active_tickers=active_tickers,
        as_of_date=as_of_date,
    )
    
    print(f"\nResults:")
    print(f"  PIT filtered:           {result['diagnostic_counts']['pit_filtered']}")
    print(f"  Trials evaluated:       {result['diagnostic_counts']['total_trials_evaluated']}")
    print(f"  With catalyst:          {result['diagnostic_counts']['with_catalyst']}")
    
    # Assertions
    assert result["diagnostic_counts"]["pit_filtered"] == 0, \
        f"Expected 0 PIT filtered trials, got {result['diagnostic_counts']['pit_filtered']}"
    
    assert result["diagnostic_counts"]["total_trials_evaluated"] == 1, \
        f"Expected 1 evaluated trial, got {result['diagnostic_counts']['total_trials_evaluated']}"
    
    assert result["diagnostic_counts"]["with_catalyst"] == 1, \
        f"Expected 1 with catalyst, got {result['diagnostic_counts']['with_catalyst']}"
    
    print("\n✅ PASS: Past trial data correctly accepted by Module 3 PIT filter")
    print("=" * 70)


def run_all_tests():
    """Run all Module 3 PIT filter tests."""
    print("\n" + "=" * 70)
    print("MODULE 3 PIT FILTER TEST SUITE")
    print("=" * 70)
    print("Verifies Module 3 correctly filters trials by data availability date.\n")
    
    tests = [
        ("Future Trial Filtered", test_module3_future_trial_filtered),
        ("Past Trial Accepted", test_module3_past_trial_accepted),
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
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run:     {passed + failed}")
    print(f"Passed:        {passed} ✅")
    print(f"Failed:        {failed} {'❌' if failed > 0 else ''}")
    print("=" * 70)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
