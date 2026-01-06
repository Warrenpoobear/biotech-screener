"""
test_defensive_integration.py

Quick test to verify defensive overlays integration works with your Module 5.

USAGE:
------
1. Place this file alongside:
   - module_5_composite.py (your existing file)
   - defensive_overlay_adapter.py (downloaded)
   - module_5_composite_with_defensive.py (downloaded)

2. Run: python test_defensive_integration.py

This will create sample data and test the integration without needing your full pipeline.
"""

import json
from decimal import Decimal
from datetime import datetime

def create_sample_data():
    """Create sample scores_by_ticker with defensive features."""
    
    sample_tickers = ["VRTX", "ALNY", "INCY", "GOSS", "JAZZ"]
    
    scores_by_ticker = {}
    
    for i, ticker in enumerate(sample_tickers):
        scores_by_ticker[ticker] = {
            "clinical_dev": {
                "score": Decimal(str(85 - i*5)),
                "lead_phase": "phase_3" if i < 3 else "phase_2",
            },
            "financial": {
                "score": Decimal(str(80 - i*5)),
                "market_cap_mm": 15000 - i*3000,
            },
            "catalyst": {
                "score": Decimal(str(75 - i*5)),
            },
            "defensive_features": {
                "vol_60d": str(0.25 + i*0.05),
                "vol_20d": str(0.28 + i*0.05),
                "vol_ratio": str(1.12 - i*0.02),
                "corr_xbi_120d": str(0.35 + i*0.10),
                "beta_xbi_60d": str(1.0 + i*0.05),
                "drawdown_current": str(-0.10 - i*0.08),
                "rsi_14d": str(65 - i*5),
                "ret_21d": str(0.05 - i*0.02),
                "skew_60d": str(-0.15 + i*0.05),
            }
        }
    
    return scores_by_ticker, set(sample_tickers)


def test_adapter_functions():
    """Test the defensive overlay adapter functions."""
    print("\n" + "="*60)
    print("TEST 1: Adapter Functions")
    print("="*60)
    
    try:
        from defensive_overlay_adapter import (
            defensive_multiplier,
            raw_inv_vol_weight,
            apply_caps_and_renormalize,
        )
        print("✓ Successfully imported defensive_overlay_adapter")
    except ImportError as e:
        print(f"❌ Failed to import defensive_overlay_adapter: {e}")
        print("   Make sure defensive_overlay_adapter.py is in the same directory")
        return False
    
    # Test defensive_multiplier
    test_features = {
        "corr_xbi_120d": "0.35",  # Low correlation → bonus
        "drawdown_current": "-0.10",
        "vol_60d": "0.25",
    }
    
    mult, notes = defensive_multiplier(test_features)
    print(f"\n✓ Defensive multiplier test:")
    print(f"  - Multiplier: {mult}")
    print(f"  - Notes: {notes}")
    
    if mult == Decimal("1.05"):
        print("  ✓ Correct! Low correlation gives 1.05x bonus")
    else:
        print(f"  ⚠️  Expected 1.05, got {mult}")
    
    # Test raw_inv_vol_weight
    weight = raw_inv_vol_weight(test_features)
    print(f"\n✓ Position weight test:")
    print(f"  - Raw weight: {weight}")
    expected = Decimal("1") / Decimal("0.25")
    if weight == expected:
        print(f"  ✓ Correct! 1 / 0.25 = {expected}")
    else:
        print(f"  ⚠️  Expected {expected}, got {weight}")
    
    # Test caps and renormalization
    test_records = [
        {"ticker": "A", "rankable": True, "_position_weight_raw": Decimal("4")},
        {"ticker": "B", "rankable": True, "_position_weight_raw": Decimal("3")},
        {"ticker": "C", "rankable": True, "_position_weight_raw": Decimal("2")},
        {"ticker": "D", "rankable": False, "_position_weight_raw": Decimal("1")},
    ]
    
    apply_caps_and_renormalize(test_records)
    
    total = sum(Decimal(r["position_weight"]) for r in test_records)
    print(f"\n✓ Position sizing test:")
    print(f"  - Total weight: {total}")
    print(f"  - A: {test_records[0]['position_weight']}")
    print(f"  - B: {test_records[1]['position_weight']}")
    print(f"  - C: {test_records[2]['position_weight']}")
    print(f"  - D (excluded): {test_records[3]['position_weight']}")
    
    if abs(total - Decimal("0.9000")) < Decimal("0.0001"):
        print("  ✓ Correct! Weights sum to 0.9000 (90% invested)")
    else:
        print(f"  ⚠️  Expected 0.9000, got {total}")
    
    return True


def test_wrapper():
    """Test the wrapper function."""
    print("\n" + "="*60)
    print("TEST 2: Wrapper Integration")
    print("="*60)
    
    try:
        from module_5_composite_with_defensive import rank_securities_with_defensive
        print("✓ Successfully imported module_5_composite_with_defensive")
    except ImportError as e:
        print(f"❌ Failed to import wrapper: {e}")
        print("   Make sure module_5_composite_with_defensive.py is in the same directory")
        return False
    
    # Create sample data
    scores_by_ticker, active_tickers = create_sample_data()
    
    print(f"\n✓ Created sample data:")
    print(f"  - {len(active_tickers)} tickers: {', '.join(sorted(active_tickers))}")
    print(f"  - All have defensive_features")
    
    # Call the wrapper
    print("\n✓ Calling rank_securities_with_defensive...")
    try:
        output = rank_securities_with_defensive(
            scores_by_ticker=scores_by_ticker,
            active_tickers=active_tickers,
            as_of_date="2026-01-06",
            normalization="cohort",
            cohort_mode="stage_only",
            validate=True,  # This will print validation diagnostics
        )
        print("\n✓ Successfully generated output!")
    except Exception as e:
        print(f"❌ Error during ranking: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Check output structure
    print("\n" + "="*60)
    print("OUTPUT VALIDATION")
    print("="*60)
    
    ranked = output.get("ranked_securities", [])
    print(f"\n✓ Ranked {len(ranked)} securities")
    
    # Check required fields
    required_fields = [
        "ticker",
        "composite_score",
        "composite_rank",
        "position_weight",
        "defensive_notes",
        "defensive_multiplier",
    ]
    
    if ranked:
        missing_fields = [f for f in required_fields if f not in ranked[0]]
        if missing_fields:
            print(f"⚠️  Missing fields: {missing_fields}")
        else:
            print("✓ All required fields present")
        
        # Show top 3
        print("\n✓ Top 3 rankings:")
        print(f"{'Rank':<6}{'Ticker':<8}{'Score':<10}{'Weight':<10}{'Multiplier':<12}{'Notes'}")
        print("-" * 70)
        for r in ranked[:3]:
            notes_str = ", ".join(r.get("defensive_notes", [])) if r.get("defensive_notes") else "-"
            print(f"{r['composite_rank']:<6}{r['ticker']:<8}{r['composite_score']:<10}{r['position_weight']:<10}{r.get('defensive_multiplier', '-'):<12}{notes_str}")
        
        # Check weight sum
        total_weight = sum(Decimal(r["position_weight"]) for r in ranked)
        print(f"\n✓ Total allocated weight: {total_weight}")
        if abs(total_weight - Decimal("0.9000")) < Decimal("0.001"):
            print("  ✓ Correct! Very close to 0.9000")
        else:
            print(f"  ⚠️  Expected ~0.9000, got {total_weight}")
    
    # Save output for inspection
    output_file = "test_defensive_output.json"
    with open(output_file, "w") as f:
        # Convert Decimals to strings for JSON
        json_output = json.loads(json.dumps(output, default=str))
        json.dump(json_output, f, indent=2)
    
    print(f"\n✓ Full output saved to: {output_file}")
    
    return True


def test_existing_module():
    """Test that existing module_5_composite still works."""
    print("\n" + "="*60)
    print("TEST 3: Original Module 5 (Unchanged)")
    print("="*60)
    
    try:
        from module_5_composite import rank_securities
        print("✓ Successfully imported original module_5_composite")
    except ImportError as e:
        print(f"❌ Failed to import module_5_composite: {e}")
        print("   Make sure module_5_composite.py is in the same directory")
        return False
    
    # Create sample data
    scores_by_ticker, active_tickers = create_sample_data()
    
    print("\n✓ Testing original function (without defensive overlays)...")
    try:
        output = rank_securities(
            scores_by_ticker=scores_by_ticker,
            active_tickers=active_tickers,
            as_of_date="2026-01-06",
            normalization="cohort",
            cohort_mode="stage_only",
        )
        print("✓ Original function still works!")
        print(f"  - Ranked {len(output.get('ranked_securities', []))} securities")
        
        # Check that defensive fields are NOT present
        ranked = output.get("ranked_securities", [])
        if ranked:
            has_defensive = "position_weight" in ranked[0] or "defensive_notes" in ranked[0]
            if has_defensive:
                print("  ⚠️  Defensive fields present (unexpected)")
            else:
                print("  ✓ No defensive fields (correct - using original function)")
    except Exception as e:
        print(f"❌ Error in original function: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("DEFENSIVE OVERLAYS INTEGRATION TEST")
    print("="*60)
    print("\nThis will test your defensive overlays integration.")
    print("Make sure these files are in the same directory:")
    print("  - module_5_composite.py (your existing file)")
    print("  - defensive_overlay_adapter.py (downloaded)")
    print("  - module_5_composite_with_defensive.py (downloaded)")
    print("  - test_defensive_integration.py (this file)")
    
    input("\nPress Enter to start tests...")
    
    # Run tests
    results = {
        "adapter": test_adapter_functions(),
        "wrapper": test_wrapper(),
        "original": test_existing_module(),
    }
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✓ PASS" if passed_test else "❌ FAIL"
        print(f"{status:8} - {test_name.title()} test")
    
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your integration is ready to use.")
        print("\nNext steps:")
        print("1. In your pipeline, change:")
        print("   FROM: from module_5_composite import rank_securities")
        print("   TO:   from module_5_composite_with_defensive import rank_securities_with_defensive")
        print("\n2. Run your full pipeline")
        print("3. Check that position_weight appears in your output")
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")
        print("Make sure all required files are in the same directory.")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
