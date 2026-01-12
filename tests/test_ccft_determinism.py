#!/usr/bin/env python3
"""
CCFT Determinism Validation Tests

Cross-Context Float Tolerance (CCFT) compliance tests ensure that
all scoring modules produce byte-identical outputs across multiple runs.

Tests:
1. Catalyst impact scorer produces identical hashes
2. Short interest signal scorer handles missing data correctly
3. Canonical JSON serialization is deterministic

Run with: python -m pytest tests/test_ccft_determinism.py -v
Or directly: python tests/test_ccft_determinism.py

Author: Wake Robin Capital Management
Version: 1.0.0
"""

import hashlib
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from decimal import Decimal
from catalyst_impact_weighting import CatalystImpactScorer
from short_interest_engine import ShortInterestSignalEngine
from utils.json_canonical import to_canonical_json, hash_canonical_json


def test_catalyst_scorer_determinism():
    """Run same inputs 3x, verify identical hashes."""
    results = []

    for _ in range(3):
        scorer = CatalystImpactScorer()
        result = scorer.calculate_catalyst_impact(
            ticker="XYZ",
            catalyst_type="phase_3_topline",
            catalyst_date="2025-06-15",
            as_of_date="2025-01-11",
            mechanism="antibody",
            current_phase="phase_3"
        )
        json_str = to_canonical_json(result)
        hash_val = hashlib.sha256(json_str.encode()).hexdigest()
        results.append(hash_val)

    assert len(set(results)) == 1, f"Non-deterministic: {results}"
    print(f"  Catalyst scorer determinism verified: {results[0][:16]}")


def test_catalyst_scorer_config_strings():
    """Verify config values are strings, not floats."""
    scorer = CatalystImpactScorer()

    # Check that config values are strings (CCFT compliance)
    assert isinstance(scorer.config["max_multiplier"], str), \
        "max_multiplier should be string for Decimal init"
    assert isinstance(scorer.config["min_multiplier"], str), \
        "min_multiplier should be string for Decimal init"

    # Verify they can be converted to Decimal
    max_mult = Decimal(scorer.config["max_multiplier"])
    min_mult = Decimal(scorer.config["min_multiplier"])

    assert max_mult == Decimal("1.5"), f"Expected 1.5, got {max_mult}"
    assert min_mult == Decimal("0.5"), f"Expected 0.5, got {min_mult}"

    print("  Config strings verified: max=1.5, min=0.5")


def test_short_interest_missing_data():
    """Verify UNKNOWN returned for missing inputs."""
    engine = ShortInterestSignalEngine()

    # Test with None si_pct
    result = engine._assess_squeeze_potential(
        si_pct=None,
        dtc=Decimal("5.0")
    )
    assert result == "UNKNOWN", f"Expected UNKNOWN for missing si_pct, got {result}"

    # Test with None dtc
    result = engine._assess_squeeze_potential(
        si_pct=Decimal("25.0"),
        dtc=None
    )
    assert result == "UNKNOWN", f"Expected UNKNOWN for missing dtc, got {result}"

    # Test with both None
    result = engine._assess_squeeze_potential(
        si_pct=None,
        dtc=None
    )
    assert result == "UNKNOWN", f"Expected UNKNOWN for both missing, got {result}"

    print("  Missing data semantics validated")


def test_short_interest_valid_data():
    """Verify correct categories for valid data."""
    engine = ShortInterestSignalEngine()

    # Test EXTREME
    result = engine._assess_squeeze_potential(
        si_pct=Decimal("45.0"),
        dtc=Decimal("12.0")
    )
    assert result == "EXTREME", f"Expected EXTREME, got {result}"

    # Test HIGH
    result = engine._assess_squeeze_potential(
        si_pct=Decimal("25.0"),
        dtc=Decimal("8.0")
    )
    assert result == "HIGH", f"Expected HIGH, got {result}"

    # Test MODERATE
    result = engine._assess_squeeze_potential(
        si_pct=Decimal("12.0"),
        dtc=Decimal("6.0")
    )
    assert result == "MODERATE", f"Expected MODERATE, got {result}"

    # Test LOW
    result = engine._assess_squeeze_potential(
        si_pct=Decimal("5.0"),
        dtc=Decimal("2.0")
    )
    assert result == "LOW", f"Expected LOW, got {result}"

    print("  Valid data categories verified")


def test_canonical_json_determinism():
    """Verify canonical JSON produces identical output."""
    test_data = {
        "ticker": "XYZ",
        "score": Decimal("84.32"),
        "nested": {"b": 2, "a": 1},
        "items": [3, 1, 2],
    }

    results = []
    for _ in range(3):
        json_str = to_canonical_json(test_data)
        results.append(json_str)

    assert len(set(results)) == 1, f"Non-deterministic JSON: {results}"

    # Verify keys are sorted
    assert results[0].index('"a"') < results[0].index('"b"'), \
        "Keys should be sorted"

    print(f"  Canonical JSON determinism verified")


def test_canonical_json_hash():
    """Verify hash function produces consistent results."""
    test_data = {"ticker": "ABC", "value": 123}

    hashes = [hash_canonical_json(test_data) for _ in range(3)]

    assert len(set(hashes)) == 1, f"Non-deterministic hash: {hashes}"
    assert len(hashes[0]) == 16, f"Expected 16-char hash, got {len(hashes[0])}"

    print(f"  Hash function verified: {hashes[0]}")


def test_decimal_handling():
    """Verify Decimal values are serialized consistently."""
    test_data = {
        "integer_decimal": Decimal("100"),
        "precise_decimal": Decimal("123.456789"),
        "zero": Decimal("0"),
        "negative": Decimal("-42.5"),
    }

    json_str = to_canonical_json(test_data)

    # Verify Decimals are converted to strings in JSON
    assert '"123.456789"' in json_str or '123.456789' in json_str, \
        "Decimal should be preserved"

    print("  Decimal handling verified")


def test_catalyst_impact_bounds():
    """Verify impact multiplier stays within bounds."""
    scorer = CatalystImpactScorer()

    # Test case that might exceed bounds
    result = scorer.calculate_catalyst_impact(
        ticker="XYZ",
        catalyst_type="fda_approval",  # 1.40 base
        catalyst_date="2025-02-01",    # Imminent: 1.25
        as_of_date="2025-01-11",
        mechanism="cell_therapy",       # 1.20
        current_phase="nda_bla"         # High success rate
    )

    multiplier = Decimal(result["impact_multiplier"])
    max_allowed = Decimal("1.5")
    min_allowed = Decimal("0.5")

    assert multiplier <= max_allowed, \
        f"Multiplier {multiplier} exceeds max {max_allowed}"
    assert multiplier >= min_allowed, \
        f"Multiplier {multiplier} below min {min_allowed}"

    print(f"  Bounds verified: {multiplier} in [{min_allowed}, {max_allowed}]")


def run_all_tests():
    """Run all CCFT determinism tests."""
    print("=" * 60)
    print("CCFT DETERMINISM VALIDATION TESTS")
    print("=" * 60)

    tests = [
        ("Catalyst Scorer Determinism", test_catalyst_scorer_determinism),
        ("Catalyst Config Strings", test_catalyst_scorer_config_strings),
        ("Short Interest Missing Data", test_short_interest_missing_data),
        ("Short Interest Valid Data", test_short_interest_valid_data),
        ("Canonical JSON Determinism", test_canonical_json_determinism),
        ("Canonical JSON Hash", test_canonical_json_hash),
        ("Decimal Handling", test_decimal_handling),
        ("Catalyst Impact Bounds", test_catalyst_impact_bounds),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        print(f"\n[TEST] {name}")
        try:
            test_func()
            passed += 1
            print(f"  PASSED")
        except AssertionError as e:
            failed += 1
            print(f"  FAILED: {e}")
        except Exception as e:
            failed += 1
            print(f"  ERROR: {e}")

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed == 0:
        print("\n  All CCFT determinism tests passed")
        return 0
    else:
        print(f"\n  {failed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit(run_all_tests())
