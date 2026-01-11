#!/usr/bin/env python3
"""
Golden fixture tests for Clinical Development Module v2.

Tests:
1. PIT cutoff filtering
2. Endpoint parsing with regex word boundaries
3. Dedup stability by nct_id
4. Decimal arithmetic precision
5. Lead program identification
6. Status quality scoring

Run: python tests/test_clinical_v2_golden.py
"""

import sys
from datetime import date
from decimal import Decimal
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from module_4_clinical_dev_v2 import (
    compute_module_4_clinical_dev_v2,
    _classify_endpoint,
    _parse_phase,
    _parse_status,
    _is_pit_admissible,
    _dedup_trials_by_nct_id,
    _identify_lead_program,
    _compute_lead_program_key,
    _score_execution,
    TrialPITRecord,
    TrialStatus,
    STRONG_ENDPOINT_PATTERNS,
    WEAK_ENDPOINT_PATTERNS,
)


# ============================================================================
# TEST FIXTURES
# ============================================================================

def create_trial_record(
    ticker: str,
    nct_id: str,
    phase: str = "PHASE2",
    status: str = "RECRUITING",
    conditions: list = None,
    primary_endpoint: str = "",
    last_update_posted: str = "2026-01-05",
    first_posted: str = None,
) -> dict:
    """Create a trial record fixture."""
    return {
        "ticker": ticker,
        "nct_id": nct_id,
        "phase": phase,
        "status": status,
        "conditions": conditions or ["cancer"],
        "primary_endpoint": primary_endpoint,
        "last_update_posted": last_update_posted,
        "first_posted": first_posted,
        "randomized": True,
        "blinded": "double-blind",
    }


def create_pit_record(
    nct_id: str,
    ticker: str = "TEST",
    phase: str = "phase 2",
    status: TrialStatus = TrialStatus.RECRUITING,
    pit_admissible: bool = True,
    last_update_posted: str = "2026-01-05",
) -> TrialPITRecord:
    """Create a TrialPITRecord fixture."""
    return TrialPITRecord(
        nct_id=nct_id,
        ticker=ticker,
        phase=phase,
        status=status,
        conditions=["cancer"],
        primary_endpoint="overall survival",
        randomized=True,
        blinded="double-blind",
        last_update_posted=last_update_posted,
        pit_date_field_used="last_update_posted",
        pit_reference_date=last_update_posted,
        pit_admissible=pit_admissible,
        pit_reason="admissible" if pit_admissible else "future_date",
        endpoint_classification="strong",
        endpoint_matched_pattern="overall_survival",
    )


# ============================================================================
# PIT CUTOFF TESTS
# ============================================================================

def test_pit_cutoff_filtering():
    """Test PIT cutoff correctly filters future-dated records."""
    print("  Running test_pit_cutoff_filtering...")

    as_of_date = "2026-01-11"
    pit_cutoff = "2026-01-10"  # as_of_date - 1

    # Record with date BEFORE cutoff (admissible)
    admissible, reason = _is_pit_admissible("2026-01-09", pit_cutoff)
    assert admissible is True, f"Expected admissible for 2026-01-09, got {admissible}"
    assert reason == "admissible"

    # Record with date ON cutoff (admissible)
    admissible, reason = _is_pit_admissible("2026-01-10", pit_cutoff)
    assert admissible is True, f"Expected admissible for 2026-01-10, got {admissible}"

    # Record with date AFTER cutoff (not admissible)
    admissible, reason = _is_pit_admissible("2026-01-11", pit_cutoff)
    assert admissible is False, f"Expected not admissible for 2026-01-11, got {admissible}"
    assert "future_date" in reason

    # Record with missing date (not admissible)
    admissible, reason = _is_pit_admissible(None, pit_cutoff)
    assert admissible is False
    assert reason == "missing_date"

    print("    PASSED")


def test_pit_filtering_in_scoring():
    """Test PIT filtering is applied in full scoring."""
    print("  Running test_pit_filtering_in_scoring...")

    as_of_date = "2026-01-11"
    active_tickers = ["TEST"]

    trials = [
        # This should be included (before cutoff)
        create_trial_record("TEST", "NCT001", last_update_posted="2026-01-05"),
        # This should be filtered (after cutoff)
        create_trial_record("TEST", "NCT002", last_update_posted="2026-01-15"),
        # This should be included (on cutoff)
        create_trial_record("TEST", "NCT003", last_update_posted="2026-01-10"),
    ]

    result = compute_module_4_clinical_dev_v2(trials, active_tickers, as_of_date)

    diag = result["diagnostic_counts"]
    assert diag["total_trials_raw"] == 3
    assert diag["total_pit_filtered"] == 1  # NCT002 should be filtered
    assert diag["total_trials_pit_admissible"] == 2

    print("    PASSED")


# ============================================================================
# ENDPOINT PARSING TESTS
# ============================================================================

def test_endpoint_word_boundaries():
    """Test endpoint parsing uses word boundaries correctly."""
    print("  Running test_endpoint_word_boundaries...")

    # "os" should match as abbreviation with word boundary
    cls, pattern = _classify_endpoint("Primary endpoint: OS at 12 months")
    assert cls == "strong", f"Expected 'strong' for 'OS at 12 months', got {cls}"

    # "os" in the middle of a word should NOT match
    cls, pattern = _classify_endpoint("Diagnosis time measurement")
    assert cls == "neutral", f"Expected 'neutral' for 'Diagnosis', got {cls}"

    # "pk" should match as abbreviation
    cls, pattern = _classify_endpoint("PK parameters after dosing")
    assert cls == "weak", f"Expected 'weak' for 'PK parameters', got {cls}"

    # "pk" in "spokesperson" should NOT match
    cls, pattern = _classify_endpoint("Spokesperson interview count")
    assert cls == "neutral", f"Expected 'neutral' for 'spokesperson', got {cls}"

    print("    PASSED")


def test_endpoint_priority_ladder():
    """Test strong endpoints take priority over weak."""
    print("  Running test_endpoint_priority_ladder...")

    # Mixed endpoint - strong should win
    cls, pattern = _classify_endpoint("Overall survival and safety assessment")
    assert cls == "strong", f"Expected 'strong' for mixed endpoint, got {cls}"

    # Pure strong
    cls, pattern = _classify_endpoint("Progression-free survival at 6 months")
    assert cls == "strong"
    assert pattern == "pfs"

    # Pure weak
    cls, pattern = _classify_endpoint("Safety and tolerability")
    assert cls == "weak"

    # Neutral
    cls, pattern = _classify_endpoint("Change in tumor size")
    assert cls == "neutral"

    print("    PASSED")


def test_endpoint_no_double_counting():
    """Test endpoints are classified once, not double-counted."""
    print("  Running test_endpoint_no_double_counting...")

    as_of_date = "2026-01-11"
    active_tickers = ["TEST"]

    trials = [
        create_trial_record("TEST", "NCT001", primary_endpoint="Overall survival and safety"),
    ]

    result = compute_module_4_clinical_dev_v2(trials, active_tickers, as_of_date)

    score = result["scores"][0]
    # Should be classified as strong (1), not both strong and weak
    assert score["n_strong_endpoints"] == 1
    assert score["n_weak_endpoints"] == 0

    print("    PASSED")


# ============================================================================
# DEDUP STABILITY TESTS
# ============================================================================

def test_dedup_by_nct_id():
    """Test deduplication by nct_id is deterministic."""
    print("  Running test_dedup_by_nct_id...")

    # Create duplicate records with same nct_id
    trials = [
        create_pit_record("NCT001", last_update_posted="2026-01-03"),
        create_pit_record("NCT001", last_update_posted="2026-01-05"),  # More recent
        create_pit_record("NCT001", last_update_posted="2026-01-01"),
        create_pit_record("NCT002", last_update_posted="2026-01-05"),
    ]

    deduped = _dedup_trials_by_nct_id(trials)

    assert len(deduped) == 2, f"Expected 2 unique trials, got {len(deduped)}"

    # NCT001 should keep the most recent update
    nct001 = next(t for t in deduped if t.nct_id == "NCT001")
    assert nct001.last_update_posted == "2026-01-05"

    print("    PASSED")


def test_dedup_prefers_pit_admissible():
    """Test dedup prefers PIT-admissible records."""
    print("  Running test_dedup_prefers_pit_admissible...")

    trials = [
        # Older but PIT-admissible
        create_pit_record("NCT001", pit_admissible=True, last_update_posted="2026-01-03"),
        # Newer but NOT PIT-admissible
        create_pit_record("NCT001", pit_admissible=False, last_update_posted="2026-01-10"),
    ]

    deduped = _dedup_trials_by_nct_id(trials)

    assert len(deduped) == 1
    assert deduped[0].pit_admissible is True
    assert deduped[0].last_update_posted == "2026-01-03"

    print("    PASSED")


def test_dedup_deterministic_across_runs():
    """Test dedup produces same result across multiple runs."""
    print("  Running test_dedup_deterministic_across_runs...")

    # Run multiple times with shuffled input
    for _ in range(5):
        trials = [
            create_pit_record("NCT001", last_update_posted="2026-01-05"),
            create_pit_record("NCT002", last_update_posted="2026-01-03"),
            create_pit_record("NCT001", last_update_posted="2026-01-03"),
            create_pit_record("NCT003", last_update_posted="2026-01-05"),
        ]

        deduped = _dedup_trials_by_nct_id(trials)

        assert len(deduped) == 3
        nct_ids = sorted(t.nct_id for t in deduped)
        assert nct_ids == ["NCT001", "NCT002", "NCT003"]

    print("    PASSED")


# ============================================================================
# LEAD PROGRAM IDENTIFICATION TESTS
# ============================================================================

def test_lead_program_highest_phase():
    """Test lead program is highest phase."""
    print("  Running test_lead_program_highest_phase...")

    trials = [
        create_pit_record("NCT001", phase="phase 1"),
        create_pit_record("NCT002", phase="phase 3"),
        create_pit_record("NCT003", phase="phase 2"),
    ]

    lead_phase, lead_nct, lead_key = _identify_lead_program(trials)

    assert lead_phase == "phase 3"
    assert lead_nct == "NCT002"

    print("    PASSED")


def test_lead_program_deterministic_tiebreak():
    """Test lead program uses deterministic tie-break for same phase."""
    print("  Running test_lead_program_deterministic_tiebreak...")

    trials = [
        create_pit_record("NCT003", phase="phase 2"),
        create_pit_record("NCT001", phase="phase 2"),
        create_pit_record("NCT002", phase="phase 2"),
    ]

    lead_phase, lead_nct, lead_key = _identify_lead_program(trials)

    assert lead_phase == "phase 2"
    # Should be deterministic - run multiple times
    results = set()
    for _ in range(10):
        _, nct, _ = _identify_lead_program(trials)
        results.add(nct)

    # Should always be the same
    assert len(results) == 1, f"Non-deterministic results: {results}"

    print("    PASSED")


def test_lead_program_key_stability():
    """Test lead program key is stable (SHA256-based)."""
    print("  Running test_lead_program_key_stability...")

    key1 = _compute_lead_program_key("NCT001", "phase 2")
    key2 = _compute_lead_program_key("NCT001", "phase 2")
    key3 = _compute_lead_program_key("NCT001", "phase 3")

    assert key1 == key2, "Same inputs should produce same key"
    assert key1 != key3, "Different inputs should produce different key"
    assert len(key1) == 16, "Key should be 16 hex chars"

    print("    PASSED")


# ============================================================================
# STATUS QUALITY SCORING TESTS
# ============================================================================

def test_termination_rate_calculation():
    """Test termination rate uses Decimal arithmetic."""
    print("  Running test_termination_rate_calculation...")

    trials = [
        create_pit_record("NCT001", status=TrialStatus.COMPLETED),
        create_pit_record("NCT002", status=TrialStatus.TERMINATED),
        create_pit_record("NCT003", status=TrialStatus.WITHDRAWN),
        create_pit_record("NCT004", status=TrialStatus.RECRUITING),
    ]

    exec_score, completion_rate, termination_rate, status_quality = _score_execution(trials)

    # 2 out of 4 are terminated/withdrawn
    assert termination_rate == Decimal("0.5"), f"Expected 0.5, got {termination_rate}"
    assert completion_rate == Decimal("0.25"), f"Expected 0.25, got {completion_rate}"

    # Check it's Decimal, not float
    assert isinstance(termination_rate, Decimal)
    assert isinstance(completion_rate, Decimal)
    assert isinstance(status_quality, Decimal)

    print("    PASSED")


def test_status_quality_score():
    """Test status quality scoring uses correct weights."""
    print("  Running test_status_quality_score...")

    # All completed = 1.0 quality
    trials = [
        create_pit_record("NCT001", status=TrialStatus.COMPLETED),
        create_pit_record("NCT002", status=TrialStatus.COMPLETED),
    ]
    _, _, _, status_quality = _score_execution(trials)
    assert status_quality == Decimal("1.0"), f"Expected 1.0 for all completed, got {status_quality}"

    # All terminated = 0.0 quality
    trials = [
        create_pit_record("NCT001", status=TrialStatus.TERMINATED),
        create_pit_record("NCT002", status=TrialStatus.WITHDRAWN),
    ]
    _, _, _, status_quality = _score_execution(trials)
    assert status_quality == Decimal("0.0"), f"Expected 0.0 for all terminated, got {status_quality}"

    print("    PASSED")


# ============================================================================
# DECIMAL PRECISION TESTS
# ============================================================================

def test_score_precision():
    """Test all scores are properly quantized."""
    print("  Running test_score_precision...")

    as_of_date = "2026-01-11"
    active_tickers = ["TEST"]

    trials = [
        create_trial_record("TEST", "NCT001", phase="PHASE2", primary_endpoint="overall survival"),
        create_trial_record("TEST", "NCT002", phase="PHASE2", primary_endpoint="safety"),
        create_trial_record("TEST", "NCT003", phase="PHASE1", primary_endpoint="pk parameters"),
    ]

    result = compute_module_4_clinical_dev_v2(trials, active_tickers, as_of_date)
    score = result["scores"][0]

    # Check score strings have max 2 decimal places
    for key in ["clinical_score", "phase_score", "design_score", "execution_score", "endpoint_score"]:
        val = score[key]
        parts = val.split(".")
        if len(parts) == 2:
            assert len(parts[1]) <= 2, f"{key} has too many decimal places: {val}"

    print("    PASSED")


def test_rate_precision():
    """Test rates are properly quantized to 4 decimal places."""
    print("  Running test_rate_precision...")

    as_of_date = "2026-01-11"
    active_tickers = ["TEST"]

    trials = [
        create_trial_record("TEST", "NCT001", status="COMPLETED"),
        create_trial_record("TEST", "NCT002", status="TERMINATED"),
        create_trial_record("TEST", "NCT003", status="RECRUITING"),
    ]

    result = compute_module_4_clinical_dev_v2(trials, active_tickers, as_of_date)
    score = result["scores"][0]

    # Check rate strings have max 4 decimal places
    for key in ["completion_rate", "termination_rate", "status_quality_score"]:
        val = score[key]
        parts = val.split(".")
        if len(parts) == 2:
            assert len(parts[1]) <= 4, f"{key} has too many decimal places: {val}"

    print("    PASSED")


# ============================================================================
# PHASE PARSING TESTS
# ============================================================================

def test_phase_parsing():
    """Test phase parsing handles CT.gov formats."""
    print("  Running test_phase_parsing...")

    # CT.gov format
    assert _parse_phase("PHASE1") == "phase 1"
    assert _parse_phase("PHASE2") == "phase 2"
    assert _parse_phase("PHASE3") == "phase 3"
    assert _parse_phase("PHASE1_PHASE2") == "phase 1/2"
    assert _parse_phase("PHASE2_PHASE3") == "phase 2/3"

    # Natural format
    assert _parse_phase("Phase 1") == "phase 1"
    assert _parse_phase("phase 2/3") == "phase 2/3"
    assert _parse_phase("Preclinical") == "preclinical"

    # Unknown
    assert _parse_phase("") == "unknown"
    assert _parse_phase(None) == "unknown"

    print("    PASSED")


def test_status_parsing():
    """Test status parsing handles various formats."""
    print("  Running test_status_parsing...")

    assert _parse_status("COMPLETED") == TrialStatus.COMPLETED
    assert _parse_status("completed") == TrialStatus.COMPLETED
    assert _parse_status("ACTIVE_NOT_RECRUITING") == TrialStatus.ACTIVE
    assert _parse_status("RECRUITING") == TrialStatus.RECRUITING
    assert _parse_status("TERMINATED") == TrialStatus.TERMINATED
    assert _parse_status("WITHDRAWN") == TrialStatus.WITHDRAWN
    assert _parse_status("SUSPENDED") == TrialStatus.SUSPENDED
    assert _parse_status("") == TrialStatus.UNKNOWN
    assert _parse_status(None) == TrialStatus.UNKNOWN

    print("    PASSED")


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

def test_full_scoring_output():
    """Test full scoring produces expected output structure."""
    print("  Running test_full_scoring_output...")

    as_of_date = "2026-01-11"
    active_tickers = ["ACME", "BETA"]

    trials = [
        create_trial_record("ACME", "NCT001", phase="PHASE3", primary_endpoint="overall survival"),
        create_trial_record("ACME", "NCT002", phase="PHASE2", primary_endpoint="safety"),
        create_trial_record("BETA", "NCT003", phase="PHASE1", primary_endpoint="pk"),
    ]

    result = compute_module_4_clinical_dev_v2(trials, active_tickers, as_of_date)

    # Check structure
    assert "as_of_date" in result
    assert "scores" in result
    assert "diagnostic_counts" in result
    assert "provenance" in result

    # Check scores are sorted by ticker
    assert result["scores"][0]["ticker"] == "ACME"
    assert result["scores"][1]["ticker"] == "BETA"

    # Check required fields
    for score in result["scores"]:
        assert "clinical_score" in score
        assert "lead_phase" in score
        assert "lead_trial_nct_id" in score
        assert "lead_program_key" in score
        assert "termination_rate" in score
        assert "completion_rate" in score
        assert "pit_filtered_count" in score
        assert "n_trials_pit_admissible" in score

    print("    PASSED")


def test_empty_trials():
    """Test handling of tickers with no trials."""
    print("  Running test_empty_trials...")

    as_of_date = "2026-01-11"
    active_tickers = ["EMPTY"]

    result = compute_module_4_clinical_dev_v2([], active_tickers, as_of_date)

    assert len(result["scores"]) == 1
    score = result["scores"][0]
    assert score["ticker"] == "EMPTY"
    assert score["n_trials_unique"] == 0
    assert "no_trials" in score["flags"]

    print("    PASSED")


# ============================================================================
# RUN ALL TESTS
# ============================================================================

def run_all_tests():
    """Run all golden fixture tests."""
    print("=" * 60)
    print("CLINICAL MODULE v2 GOLDEN FIXTURE TESTS")
    print("=" * 60)
    print()

    test_functions = [
        # PIT tests
        ("PIT Cutoff Tests:", [
            test_pit_cutoff_filtering,
            test_pit_filtering_in_scoring,
        ]),
        ("Endpoint Parsing Tests:", [
            test_endpoint_word_boundaries,
            test_endpoint_priority_ladder,
            test_endpoint_no_double_counting,
        ]),
        ("Dedup Stability Tests:", [
            test_dedup_by_nct_id,
            test_dedup_prefers_pit_admissible,
            test_dedup_deterministic_across_runs,
        ]),
        ("Lead Program Tests:", [
            test_lead_program_highest_phase,
            test_lead_program_deterministic_tiebreak,
            test_lead_program_key_stability,
        ]),
        ("Status Quality Tests:", [
            test_termination_rate_calculation,
            test_status_quality_score,
        ]),
        ("Decimal Precision Tests:", [
            test_score_precision,
            test_rate_precision,
        ]),
        ("Parsing Tests:", [
            test_phase_parsing,
            test_status_parsing,
        ]),
        ("Integration Tests:", [
            test_full_scoring_output,
            test_empty_trials,
        ]),
    ]

    passed = 0
    failed = 0

    for section_name, tests in test_functions:
        print(section_name)
        for test_fn in tests:
            try:
                test_fn()
                passed += 1
            except AssertionError as e:
                print(f"    FAILED: {e}")
                failed += 1
            except Exception as e:
                print(f"    ERROR: {e}")
                failed += 1

    print()
    print("=" * 60)
    print(f"RESULTS: {passed}/{passed + failed} tests passed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
