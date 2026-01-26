#!/usr/bin/env python3
"""
Unit tests for module_4_clinical_dev_v2.py

Tests clinical development quality scoring:
- Input validation (as_of_date, active_tickers, trial_records)
- Date parsing and PIT cutoff computation
- PIT enforcement (date field selection, admissibility)
- Phase parsing (CT.gov formats)
- Status parsing (all trial statuses)
- Endpoint classification (strong/weak/neutral)
- Conditions normalization and tokenization
- Component scoring functions
- Lead program identification with tie-breaking
- Trial deduplication
- Main scoring function
- Determinism verification
"""

import pytest
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
import hashlib
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from module_4_clinical_dev_v2 import (
    # Input validation
    _validate_as_of_date,
    _validate_active_tickers,
    _validate_trial_records,
    _safe_string_field,
    ValidationError,
    # Date parsing
    _parse_date_safe,
    _compute_pit_cutoff,
    # PIT enforcement
    _select_pit_date,
    _is_pit_admissible,
    # Parsing functions
    _parse_phase,
    _parse_status,
    _classify_endpoint,
    _normalize_conditions,
    _tokenize_conditions,
    # Scoring functions
    _score_trial_count,
    _score_indication_diversity,
    _score_recency,
    _score_design,
    _score_execution,
    _score_endpoints,
    # Lead program
    _compute_lead_program_key,
    _identify_lead_program,
    # Deduplication
    _dedup_trials_by_nct_id,
    # Main functions
    compute_module_4_clinical_dev_v2,
    compute_module_4_clinical_dev,
    # Dataclasses
    TrialPITRecord,
    TickerClinicalSummaryV2,
    TrialStatus,
    # Constants
    PHASE_SCORES,
    RECENCY_STALE_THRESHOLD,
    MAX_ENDPOINT_LENGTH,
    MAX_CONDITION_STRING_LENGTH,
)


# ============================================================================
# TEST FIXTURES
# ============================================================================

@pytest.fixture
def as_of_date():
    """Standard as_of_date for tests."""
    return "2026-01-15"


@pytest.fixture
def sample_trial_record():
    """Basic trial record for testing."""
    return {
        "ticker": "ACME",
        "nct_id": "NCT12345678",
        "phase": "PHASE2",
        "status": "ACTIVE",
        "conditions": "Breast Cancer",
        "primary_endpoint": "Overall Survival",
        "randomized": True,
        "blinded": "double-blind",
        "first_posted": "2025-06-01",
        "last_update_posted": "2025-12-01",
    }


@pytest.fixture
def sample_pit_record():
    """Sample TrialPITRecord for testing."""
    return TrialPITRecord(
        nct_id="NCT12345678",
        ticker="ACME",
        phase="phase 2",
        status=TrialStatus.ACTIVE,
        conditions=["breast cancer"],
        primary_endpoint="Overall Survival",
        randomized=True,
        blinded="double-blind",
        last_update_posted="2025-12-01",
        pit_date_field_used="first_posted",
        pit_reference_date="2025-06-01",
        pit_admissible=True,
        pit_reason="admissible",
        endpoint_classification="strong",
        endpoint_matched_pattern="overall_survival",
    )


# ============================================================================
# INPUT VALIDATION TESTS
# ============================================================================

class TestValidateAsOfDate:
    """Tests for _validate_as_of_date function."""

    def test_valid_iso_string(self):
        """Valid ISO date string should pass."""
        result = _validate_as_of_date("2026-01-15")
        assert result == "2026-01-15"

    def test_valid_date_object(self):
        """date object should be converted to ISO string."""
        result = _validate_as_of_date(date(2026, 1, 15))
        assert result == "2026-01-15"

    def test_datetime_with_time(self):
        """Datetime string with time should extract date part."""
        result = _validate_as_of_date("2026-01-15T10:30:00")
        assert result == "2026-01-15"

    def test_whitespace_trimmed(self):
        """Whitespace should be trimmed."""
        result = _validate_as_of_date("  2026-01-15  ")
        assert result == "2026-01-15"

    def test_none_raises(self):
        """None should raise ValidationError."""
        with pytest.raises(ValidationError, match="cannot be None"):
            _validate_as_of_date(None)

    def test_empty_string_raises(self):
        """Empty string should raise ValidationError."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            _validate_as_of_date("")

    def test_invalid_format_raises(self):
        """Invalid date format should raise ValidationError."""
        with pytest.raises(ValidationError, match="valid ISO date"):
            _validate_as_of_date("not-a-date")

    def test_integer_raises(self):
        """Integer should raise ValidationError."""
        with pytest.raises(ValidationError, match="must be string or date"):
            _validate_as_of_date(20260115)


class TestValidateActiveTickers:
    """Tests for _validate_active_tickers function."""

    def test_set_input(self):
        """Set of tickers should be converted to sorted list."""
        result = _validate_active_tickers({"ACME", "BETA", "ALPHA"})
        assert result == ["ACME", "ALPHA", "BETA"]

    def test_list_input(self):
        """List of tickers should be normalized and sorted."""
        result = _validate_active_tickers(["beta", "ACME", "alpha"])
        assert result == ["ACME", "ALPHA", "BETA"]

    def test_tuple_input(self):
        """Tuple should be accepted."""
        result = _validate_active_tickers(("ACME", "BETA"))
        assert result == ["ACME", "BETA"]

    def test_frozenset_input(self):
        """Frozenset should be accepted."""
        result = _validate_active_tickers(frozenset(["ACME", "BETA"]))
        assert sorted(result) == ["ACME", "BETA"]

    def test_none_returns_empty(self):
        """None should return empty list."""
        result = _validate_active_tickers(None)
        assert result == []

    def test_lowercase_converted(self):
        """Lowercase tickers should be uppercased."""
        result = _validate_active_tickers(["acme", "beta"])
        assert result == ["ACME", "BETA"]

    def test_whitespace_trimmed(self):
        """Whitespace in tickers should be trimmed."""
        result = _validate_active_tickers(["  ACME  ", " BETA "])
        assert result == ["ACME", "BETA"]

    def test_empty_strings_filtered(self):
        """Empty strings should be filtered out."""
        result = _validate_active_tickers(["ACME", "", "  ", "BETA"])
        assert result == ["ACME", "BETA"]

    def test_duplicates_removed(self):
        """Duplicate tickers should be removed."""
        result = _validate_active_tickers(["ACME", "ACME", "acme"])
        assert result == ["ACME"]

    def test_invalid_type_raises(self):
        """Invalid type should raise ValidationError."""
        with pytest.raises(ValidationError, match="must be set, list"):
            _validate_active_tickers("ACME")

    def test_non_string_ticker_raises(self):
        """Non-string ticker should raise ValidationError."""
        with pytest.raises(ValidationError, match="must be string"):
            _validate_active_tickers([123, "ACME"])


class TestValidateTrialRecords:
    """Tests for _validate_trial_records function."""

    def test_valid_list(self):
        """Valid list of dicts should pass."""
        records = [{"ticker": "ACME"}, {"ticker": "BETA"}]
        result = _validate_trial_records(records)
        assert result == records

    def test_none_returns_empty(self):
        """None should return empty list."""
        result = _validate_trial_records(None)
        assert result == []

    def test_empty_list(self):
        """Empty list should pass."""
        result = _validate_trial_records([])
        assert result == []

    def test_non_list_raises(self):
        """Non-list should raise ValidationError."""
        with pytest.raises(ValidationError, match="must be list"):
            _validate_trial_records({"ticker": "ACME"})

    def test_non_dict_element_raises(self):
        """Non-dict element should raise ValidationError."""
        with pytest.raises(ValidationError, match="must be dict"):
            _validate_trial_records([{"ticker": "ACME"}, "invalid"])


class TestSafeStringField:
    """Tests for _safe_string_field function."""

    def test_normal_string(self):
        """Normal string should pass through."""
        assert _safe_string_field("test") == "test"

    def test_none_returns_empty(self):
        """None should return empty string."""
        assert _safe_string_field(None) == ""

    def test_whitespace_trimmed(self):
        """Whitespace should be trimmed."""
        assert _safe_string_field("  test  ") == "test"

    def test_integer_converted(self):
        """Integer should be converted to string."""
        assert _safe_string_field(123) == "123"

    def test_truncation(self):
        """Long strings should be truncated."""
        long_string = "x" * 1000
        result = _safe_string_field(long_string, max_length=100)
        assert len(result) == 100


# ============================================================================
# DATE PARSING TESTS
# ============================================================================

class TestParseDateSafe:
    """Tests for _parse_date_safe function."""

    def test_full_iso_date(self):
        """Full ISO date should parse correctly."""
        result = _parse_date_safe("2026-01-15")
        assert result == date(2026, 1, 15)

    def test_datetime_string(self):
        """Datetime string should extract date part."""
        result = _parse_date_safe("2026-01-15T10:30:00")
        assert result == date(2026, 1, 15)

    def test_datetime_with_timezone(self):
        """Datetime with timezone should parse."""
        result = _parse_date_safe("2026-01-15T10:30:00+05:00")
        assert result == date(2026, 1, 15)

    def test_datetime_with_z(self):
        """Datetime with Z suffix should parse."""
        result = _parse_date_safe("2026-01-15T10:30:00Z")
        assert result == date(2026, 1, 15)

    def test_partial_date_year_month(self):
        """YYYY-MM should default to first of month."""
        result = _parse_date_safe("2026-01")
        assert result == date(2026, 1, 1)

    def test_partial_date_year_only(self):
        """YYYY should default to Jan 1."""
        result = _parse_date_safe("2026")
        assert result == date(2026, 1, 1)

    def test_none_returns_none(self):
        """None should return None."""
        assert _parse_date_safe(None) is None

    def test_empty_string_returns_none(self):
        """Empty string should return None."""
        assert _parse_date_safe("") is None

    def test_invalid_returns_none(self):
        """Invalid date should return None."""
        assert _parse_date_safe("not-a-date") is None


class TestComputePitCutoff:
    """Tests for _compute_pit_cutoff function."""

    def test_basic_cutoff(self):
        """PIT cutoff should be as_of_date - 1."""
        result = _compute_pit_cutoff("2026-01-15")
        assert result == "2026-01-14"

    def test_year_boundary(self):
        """Should handle year boundary."""
        result = _compute_pit_cutoff("2026-01-01")
        assert result == "2025-12-31"

    def test_month_boundary(self):
        """Should handle month boundary."""
        result = _compute_pit_cutoff("2026-03-01")
        assert result == "2026-02-28"


# ============================================================================
# PIT ENFORCEMENT TESTS
# ============================================================================

class TestSelectPitDate:
    """Tests for _select_pit_date function."""

    def test_first_posted_priority(self):
        """first_posted should be highest priority."""
        trial = {
            "first_posted": "2025-01-01",
            "last_update_posted": "2025-06-01",
            "source_date": "2025-03-01",
        }
        value, field = _select_pit_date(trial)
        assert value == "2025-01-01"
        assert field == "first_posted"

    def test_last_update_fallback(self):
        """Should fall back to last_update_posted."""
        trial = {
            "last_update_posted": "2025-06-01",
            "source_date": "2025-03-01",
        }
        value, field = _select_pit_date(trial)
        assert value == "2025-06-01"
        assert field == "last_update_posted"

    def test_source_date_fallback(self):
        """Should fall back to source_date."""
        trial = {"source_date": "2025-03-01"}
        value, field = _select_pit_date(trial)
        assert value == "2025-03-01"
        assert field == "source_date"

    def test_collected_at_fallback(self):
        """Should fall back to collected_at."""
        trial = {"collected_at": "2025-02-01"}
        value, field = _select_pit_date(trial)
        assert value == "2025-02-01"
        assert field == "collected_at"

    def test_no_date_fields(self):
        """Should return None and 'none' when no date fields."""
        value, field = _select_pit_date({})
        assert value is None
        assert field == "none"


class TestIsPitAdmissible:
    """Tests for _is_pit_admissible function."""

    def test_admissible_date(self):
        """Date before cutoff should be admissible."""
        result, reason = _is_pit_admissible("2025-01-01", "2026-01-14")
        assert result is True
        assert reason == "admissible"

    def test_equal_to_cutoff(self):
        """Date equal to cutoff should be admissible."""
        result, reason = _is_pit_admissible("2026-01-14", "2026-01-14")
        assert result is True
        assert reason == "admissible"

    def test_future_date(self):
        """Date after cutoff should not be admissible."""
        result, reason = _is_pit_admissible("2026-01-20", "2026-01-14")
        assert result is False
        assert "future_date" in reason

    def test_missing_date(self):
        """None date should not be admissible."""
        result, reason = _is_pit_admissible(None, "2026-01-14")
        assert result is False
        assert reason == "missing_date"

    def test_unparseable_date(self):
        """Unparseable date should not be admissible."""
        result, reason = _is_pit_admissible("invalid", "2026-01-14")
        assert result is False
        assert reason == "unparseable_date"


# ============================================================================
# PHASE PARSING TESTS
# ============================================================================

class TestParsePhase:
    """Tests for _parse_phase function."""

    def test_ctgov_phase1(self):
        """PHASE1 CT.gov format."""
        assert _parse_phase("PHASE1") == "phase 1"

    def test_ctgov_phase2(self):
        """PHASE2 CT.gov format."""
        assert _parse_phase("PHASE2") == "phase 2"

    def test_ctgov_phase3(self):
        """PHASE3 CT.gov format."""
        assert _parse_phase("PHASE3") == "phase 3"

    def test_phase_combination_2_3(self):
        """Phase 2/3 combination."""
        assert _parse_phase("PHASE2_PHASE3") == "phase 2/3"
        assert _parse_phase("Phase 2/3") == "phase 2/3"

    def test_phase_combination_1_2(self):
        """Phase 1/2 combination."""
        assert _parse_phase("PHASE1_PHASE2") == "phase 1/2"
        assert _parse_phase("Phase 1/2") == "phase 1/2"

    def test_approved(self):
        """Approved/Phase 4."""
        assert _parse_phase("APPROVED") == "approved"
        assert _parse_phase("PHASE4") == "approved"
        assert _parse_phase("4") == "approved"

    def test_preclinical(self):
        """Preclinical phase."""
        assert _parse_phase("PRECLINICAL") == "preclinical"
        assert _parse_phase("pre-clinical") == "preclinical"

    def test_unknown(self):
        """Unknown phases."""
        assert _parse_phase("NA") == "unknown"
        assert _parse_phase("") == "unknown"
        assert _parse_phase(None) == "unknown"

    def test_case_insensitive(self):
        """Parsing should be case insensitive."""
        assert _parse_phase("phase2") == "phase 2"
        assert _parse_phase("PHASE2") == "phase 2"
        assert _parse_phase("Phase2") == "phase 2"


# ============================================================================
# STATUS PARSING TESTS
# ============================================================================

class TestParseStatus:
    """Tests for _parse_status function."""

    def test_completed(self):
        """Completed status."""
        assert _parse_status("COMPLETED") == TrialStatus.COMPLETED

    def test_active(self):
        """Active status."""
        assert _parse_status("ACTIVE") == TrialStatus.ACTIVE

    def test_active_not_recruiting(self):
        """Active not recruiting should map to ACTIVE."""
        assert _parse_status("ACTIVE_NOT_RECRUITING") == TrialStatus.ACTIVE
        assert _parse_status("Active, not recruiting") == TrialStatus.ACTIVE

    def test_recruiting(self):
        """Recruiting status."""
        assert _parse_status("RECRUITING") == TrialStatus.RECRUITING

    def test_not_yet_recruiting(self):
        """Not yet recruiting status."""
        assert _parse_status("NOT_YET_RECRUITING") == TrialStatus.NOT_YET_RECRUITING

    def test_enrolling_by_invitation(self):
        """Enrolling by invitation status."""
        assert _parse_status("ENROLLING_BY_INVITATION") == TrialStatus.ENROLLING_BY_INVITATION

    def test_suspended(self):
        """Suspended status."""
        assert _parse_status("SUSPENDED") == TrialStatus.SUSPENDED

    def test_terminated(self):
        """Terminated status."""
        assert _parse_status("TERMINATED") == TrialStatus.TERMINATED

    def test_withdrawn(self):
        """Withdrawn status."""
        assert _parse_status("WITHDRAWN") == TrialStatus.WITHDRAWN

    def test_unknown(self):
        """Unknown status."""
        assert _parse_status("SOMETHING_ELSE") == TrialStatus.UNKNOWN
        assert _parse_status("") == TrialStatus.UNKNOWN
        assert _parse_status(None) == TrialStatus.UNKNOWN


# ============================================================================
# ENDPOINT CLASSIFICATION TESTS
# ============================================================================

class TestClassifyEndpoint:
    """Tests for _classify_endpoint function."""

    def test_overall_survival(self):
        """Overall survival is strong endpoint."""
        classification, pattern = _classify_endpoint("Primary: Overall Survival at 2 years")
        assert classification == "strong"
        assert pattern == "overall_survival"

    def test_progression_free_survival(self):
        """PFS is strong endpoint."""
        classification, pattern = _classify_endpoint("Progression-Free Survival")
        assert classification == "strong"
        assert pattern == "pfs"

    def test_objective_response_rate(self):
        """ORR is strong endpoint."""
        classification, pattern = _classify_endpoint("Objective Response Rate (ORR)")
        assert classification == "strong"

    def test_complete_response(self):
        """Complete response is strong endpoint."""
        classification, pattern = _classify_endpoint("Complete Response rate")
        assert classification == "strong"
        assert pattern == "complete_response"

    def test_safety_is_weak(self):
        """Safety is weak endpoint."""
        classification, pattern = _classify_endpoint("Safety and tolerability")
        assert classification == "weak"
        assert pattern == "safety"

    def test_biomarker_is_weak(self):
        """Biomarker is weak endpoint."""
        classification, pattern = _classify_endpoint("Change in biomarker levels")
        assert classification == "weak"
        assert pattern == "biomarker"

    def test_pharmacokinetics_is_weak(self):
        """Pharmacokinetics is weak endpoint."""
        classification, pattern = _classify_endpoint("Pharmacokinetic profile")
        assert classification == "weak"
        assert pattern == "pk"

    def test_dose_finding_is_weak(self):
        """Dose finding is weak endpoint."""
        classification, pattern = _classify_endpoint("Maximum tolerated dose")
        assert classification == "weak"

    def test_neutral_endpoint(self):
        """Generic endpoint is neutral."""
        classification, pattern = _classify_endpoint("Change from baseline in symptom score")
        assert classification == "neutral"
        assert pattern is None

    def test_empty_string(self):
        """Empty string is neutral."""
        classification, pattern = _classify_endpoint("")
        assert classification == "neutral"


# ============================================================================
# CONDITIONS PARSING TESTS
# ============================================================================

class TestNormalizeConditions:
    """Tests for _normalize_conditions function."""

    def test_simple_string(self):
        """Simple string condition."""
        result = _normalize_conditions("Breast Cancer")
        assert result == ["breast cancer"]

    def test_comma_separated(self):
        """Comma-separated conditions."""
        result = _normalize_conditions("Breast Cancer, Lung Cancer")
        assert result == ["breast cancer", "lung cancer"]

    def test_semicolon_separated(self):
        """Semicolon-separated conditions."""
        result = _normalize_conditions("Breast Cancer; Lung Cancer")
        assert result == ["breast cancer", "lung cancer"]

    def test_list_input(self):
        """List of conditions."""
        result = _normalize_conditions(["Breast Cancer", "Lung Cancer"])
        assert result == ["breast cancer", "lung cancer"]

    def test_nested_list(self):
        """Nested list of conditions."""
        result = _normalize_conditions([["Breast Cancer"], ["Lung Cancer"]])
        assert result == ["breast cancer", "lung cancer"]

    def test_dict_with_name(self):
        """Dict with 'name' key."""
        result = _normalize_conditions({"name": "Breast Cancer"})
        assert "breast cancer" in result

    def test_none_returns_empty(self):
        """None should return empty list."""
        result = _normalize_conditions(None)
        assert result == []

    def test_deduplication(self):
        """Duplicate conditions should be deduplicated."""
        result = _normalize_conditions("Cancer, CANCER, cancer")
        assert result == ["cancer"]

    def test_sorted_output(self):
        """Output should be sorted."""
        result = _normalize_conditions("Zebra disease, Apple syndrome")
        assert result == ["apple syndrome", "zebra disease"]


class TestTokenizeConditions:
    """Tests for _tokenize_conditions function."""

    def test_basic_tokenization(self):
        """Basic word tokenization."""
        result = _tokenize_conditions(["breast cancer", "lung cancer"])
        assert "breast" in result
        assert "cancer" in result
        assert "lung" in result

    def test_stopwords_removed(self):
        """Stopwords should be removed."""
        result = _tokenize_conditions(["cancer of the lung"])
        assert "the" not in result
        assert "of" not in result

    def test_short_words_removed(self):
        """Words <= 2 chars should be removed."""
        result = _tokenize_conditions(["a b c cancer"])
        assert "a" not in result
        assert "b" not in result
        assert "c" not in result
        assert "cancer" in result

    def test_empty_list(self):
        """Empty list should return empty set."""
        result = _tokenize_conditions([])
        assert result == set()


# ============================================================================
# SCORING FUNCTION TESTS
# ============================================================================

class TestScoreTrialCount:
    """Tests for _score_trial_count function."""

    def test_zero_trials(self):
        """Zero trials should score 0."""
        assert _score_trial_count(0) == Decimal("0")

    def test_one_trial(self):
        """One trial scores 0.5."""
        assert _score_trial_count(1) == Decimal("0.5")

    def test_two_trials(self):
        """Two trials scores 1.0."""
        assert _score_trial_count(2) == Decimal("1.0")

    def test_five_trials(self):
        """Five trials scores 2.0."""
        assert _score_trial_count(5) == Decimal("2.0")

    def test_ten_trials(self):
        """Ten trials scores 3.5."""
        assert _score_trial_count(10) == Decimal("3.5")

    def test_twenty_trials(self):
        """Twenty trials scores 4.5."""
        assert _score_trial_count(20) == Decimal("4.5")

    def test_many_trials(self):
        """Many trials caps at 5.0."""
        assert _score_trial_count(100) == Decimal("5.0")
        assert _score_trial_count(1000) == Decimal("5.0")

    def test_negative_handled(self):
        """Negative input should return 0."""
        assert _score_trial_count(-5) == Decimal("0")

    def test_invalid_type(self):
        """Invalid type should return 0."""
        assert _score_trial_count("invalid") == Decimal("0")


class TestScoreIndicationDiversity:
    """Tests for _score_indication_diversity function."""

    def test_empty_conditions(self):
        """Empty conditions should score 0."""
        assert _score_indication_diversity([]) == Decimal("0")

    def test_single_condition(self):
        """Single condition with few tokens."""
        result = _score_indication_diversity([["breast cancer"]])
        assert result > Decimal("0")

    def test_diverse_conditions(self):
        """Diverse conditions should score higher."""
        diverse = [
            ["breast cancer", "ovarian cancer"],
            ["lung cancer", "melanoma"],
            ["lymphoma", "leukemia"],
        ]
        result = _score_indication_diversity(diverse)
        assert result >= Decimal("3.0")


class TestScoreRecency:
    """Tests for _score_recency function."""

    def test_recent_update(self):
        """Recent update (< 30 days) should score 5.0."""
        score, days, unknown, stale = _score_recency("2026-01-10", "2026-01-15")
        assert score == Decimal("5.0")
        assert days == 5
        assert unknown is False
        assert stale is False

    def test_30_90_days(self):
        """30-90 days should score 4.5."""
        score, days, unknown, stale = _score_recency("2025-11-15", "2026-01-15")
        assert score == Decimal("4.5")
        assert 30 <= days < 90

    def test_stale_update(self):
        """Stale update (> 2 years) should score 1.0 and be marked stale."""
        old_date = (date(2026, 1, 15) - timedelta(days=RECENCY_STALE_THRESHOLD + 10)).isoformat()
        score, days, unknown, stale = _score_recency(old_date, "2026-01-15")
        assert score == Decimal("1.0")
        assert stale is True

    def test_unknown_recency(self):
        """None date should return unknown penalty."""
        score, days, unknown, stale = _score_recency(None, "2026-01-15")
        assert unknown is True
        assert days is None

    def test_future_date(self):
        """Future date should be marked unknown."""
        score, days, unknown, stale = _score_recency("2026-02-01", "2026-01-15")
        assert unknown is True
        assert days < 0


class TestScoreDesign:
    """Tests for _score_design function."""

    def test_base_score(self, sample_pit_record):
        """Base score should be 12."""
        # Modify to minimal features
        record = TrialPITRecord(
            nct_id="NCT00000000",
            ticker="TEST",
            phase="phase 2",
            status=TrialStatus.ACTIVE,
            conditions=[],
            primary_endpoint="",
            randomized=False,
            blinded="",
            last_update_posted=None,
            pit_date_field_used="none",
            pit_reference_date=None,
            pit_admissible=True,
            pit_reason="admissible",
            endpoint_classification="neutral",
            endpoint_matched_pattern=None,
        )
        score = _score_design(record)
        assert score == Decimal("12")

    def test_randomized_bonus(self, sample_pit_record):
        """Randomized trial should get bonus."""
        sample_pit_record.randomized = True
        sample_pit_record.blinded = ""
        sample_pit_record.endpoint_classification = "neutral"
        score = _score_design(sample_pit_record)
        assert score >= Decimal("17")  # 12 base + 5 randomized

    def test_double_blind_bonus(self, sample_pit_record):
        """Double-blind should get bonus."""
        sample_pit_record.randomized = False
        sample_pit_record.blinded = "double-blind"
        sample_pit_record.endpoint_classification = "neutral"
        score = _score_design(sample_pit_record)
        assert score >= Decimal("16")  # 12 base + 4 double-blind

    def test_strong_endpoint_bonus(self, sample_pit_record):
        """Strong endpoint should get bonus."""
        sample_pit_record.randomized = False
        sample_pit_record.blinded = ""
        sample_pit_record.endpoint_classification = "strong"
        score = _score_design(sample_pit_record)
        assert score >= Decimal("16")  # 12 base + 4 strong

    def test_weak_endpoint_penalty(self, sample_pit_record):
        """Weak endpoint should get penalty."""
        sample_pit_record.randomized = False
        sample_pit_record.blinded = ""
        sample_pit_record.endpoint_classification = "weak"
        score = _score_design(sample_pit_record)
        assert score == Decimal("9")  # 12 base - 3 weak

    def test_score_capped_at_25(self, sample_pit_record):
        """Score should be capped at 25."""
        sample_pit_record.randomized = True
        sample_pit_record.blinded = "double-blind"
        sample_pit_record.endpoint_classification = "strong"
        score = _score_design(sample_pit_record)
        assert score <= Decimal("25")


class TestScoreExecution:
    """Tests for _score_execution function."""

    def test_empty_trials(self):
        """Empty trials should return base scores."""
        score, comp_rate, term_rate, quality = _score_execution([])
        assert score == Decimal("12")
        assert comp_rate == Decimal("0")
        assert term_rate == Decimal("0")

    def test_all_completed(self, sample_pit_record):
        """All completed trials should have high score."""
        trials = [sample_pit_record] * 3
        for t in trials:
            t.status = TrialStatus.COMPLETED
        score, comp_rate, term_rate, quality = _score_execution(trials)
        assert comp_rate == Decimal("1.0")
        assert term_rate == Decimal("0")
        assert score > Decimal("12")

    def test_all_terminated(self, sample_pit_record):
        """All terminated trials should have low score."""
        trials = [sample_pit_record] * 3
        for t in trials:
            t.status = TrialStatus.TERMINATED
        score, comp_rate, term_rate, quality = _score_execution(trials)
        assert term_rate == Decimal("1.0")
        assert score < Decimal("12")

    def test_mixed_statuses(self, sample_pit_record):
        """Mixed statuses should be weighted."""
        t1 = sample_pit_record
        t1.status = TrialStatus.COMPLETED
        t2 = TrialPITRecord(
            nct_id="NCT00000002",
            ticker="ACME",
            phase="phase 2",
            status=TrialStatus.ACTIVE,
            conditions=[],
            primary_endpoint="",
            randomized=False,
            blinded="",
            last_update_posted=None,
            pit_date_field_used="none",
            pit_reference_date=None,
            pit_admissible=True,
            pit_reason="admissible",
            endpoint_classification="neutral",
            endpoint_matched_pattern=None,
        )
        score, comp_rate, term_rate, quality = _score_execution([t1, t2])
        assert Decimal("0") < comp_rate < Decimal("1")


class TestScoreEndpoints:
    """Tests for _score_endpoints function."""

    def test_empty_trials(self):
        """Empty trials should return base score."""
        score, strong, weak, neutral = _score_endpoints([])
        assert score == Decimal("10")
        assert strong == 0
        assert weak == 0
        assert neutral == 0

    def test_strong_endpoints_bonus(self, sample_pit_record):
        """Strong endpoints should increase score."""
        sample_pit_record.endpoint_classification = "strong"
        score, strong, weak, neutral = _score_endpoints([sample_pit_record])
        assert strong == 1
        assert score > Decimal("10")

    def test_weak_endpoints_penalty(self, sample_pit_record):
        """Weak endpoints should decrease score."""
        sample_pit_record.endpoint_classification = "weak"
        score, strong, weak, neutral = _score_endpoints([sample_pit_record])
        assert weak == 1
        assert score < Decimal("10")

    def test_score_capped(self, sample_pit_record):
        """Score should be capped at 0-20."""
        trials = [sample_pit_record] * 10
        for t in trials:
            t.endpoint_classification = "strong"
        score, _, _, _ = _score_endpoints(trials)
        assert score <= Decimal("20")


# ============================================================================
# LEAD PROGRAM TESTS
# ============================================================================

class TestComputeLeadProgramKey:
    """Tests for _compute_lead_program_key function."""

    def test_deterministic(self):
        """Same inputs should produce same key."""
        key1 = _compute_lead_program_key("NCT12345678", "phase 2")
        key2 = _compute_lead_program_key("NCT12345678", "phase 2")
        assert key1 == key2

    def test_different_inputs_different_keys(self):
        """Different inputs should produce different keys."""
        key1 = _compute_lead_program_key("NCT12345678", "phase 2")
        key2 = _compute_lead_program_key("NCT12345679", "phase 2")
        assert key1 != key2

    def test_key_length(self):
        """Key should be 16 characters."""
        key = _compute_lead_program_key("NCT12345678", "phase 2")
        assert len(key) == 16


class TestIdentifyLeadProgram:
    """Tests for _identify_lead_program function."""

    def test_empty_trials(self):
        """Empty trials should return unknown."""
        phase, nct_id, key = _identify_lead_program([])
        assert phase == "unknown"
        assert nct_id is None
        assert key == ""

    def test_single_trial(self, sample_pit_record):
        """Single trial should be the lead."""
        phase, nct_id, key = _identify_lead_program([sample_pit_record])
        assert phase == "phase 2"
        assert nct_id == "NCT12345678"

    def test_highest_phase_wins(self, sample_pit_record):
        """Highest phase should win."""
        t1 = sample_pit_record
        t1.phase = "phase 2"
        t2 = TrialPITRecord(
            nct_id="NCT00000002",
            ticker="ACME",
            phase="phase 3",
            status=TrialStatus.ACTIVE,
            conditions=[],
            primary_endpoint="",
            randomized=False,
            blinded="",
            last_update_posted=None,
            pit_date_field_used="none",
            pit_reference_date=None,
            pit_admissible=True,
            pit_reason="admissible",
            endpoint_classification="neutral",
            endpoint_matched_pattern=None,
        )
        phase, nct_id, key = _identify_lead_program([t1, t2])
        assert phase == "phase 3"
        assert nct_id == "NCT00000002"

    def test_tie_break_by_key(self, sample_pit_record):
        """Ties should be broken by deterministic key."""
        t1 = sample_pit_record
        t1.nct_id = "NCT00000001"
        t1.phase = "phase 2"
        t2 = TrialPITRecord(
            nct_id="NCT00000002",
            ticker="ACME",
            phase="phase 2",
            status=TrialStatus.ACTIVE,
            conditions=[],
            primary_endpoint="",
            randomized=False,
            blinded="",
            last_update_posted=None,
            pit_date_field_used="none",
            pit_reference_date=None,
            pit_admissible=True,
            pit_reason="admissible",
            endpoint_classification="neutral",
            endpoint_matched_pattern=None,
        )
        phase, nct_id, key = _identify_lead_program([t1, t2])
        assert phase == "phase 2"
        # Result should be deterministic
        phase2, nct_id2, key2 = _identify_lead_program([t2, t1])
        assert nct_id == nct_id2


# ============================================================================
# DEDUPLICATION TESTS
# ============================================================================

class TestDedupTrialsByNctId:
    """Tests for _dedup_trials_by_nct_id function."""

    def test_no_duplicates(self, sample_pit_record):
        """No duplicates should return same list."""
        result = _dedup_trials_by_nct_id([sample_pit_record])
        assert len(result) == 1

    def test_duplicates_removed(self, sample_pit_record):
        """Duplicates should be removed."""
        t1 = sample_pit_record
        t2 = TrialPITRecord(
            nct_id="NCT12345678",  # Same NCT ID
            ticker="ACME",
            phase="phase 2",
            status=TrialStatus.ACTIVE,
            conditions=[],
            primary_endpoint="",
            randomized=False,
            blinded="",
            last_update_posted="2025-11-01",  # Older
            pit_date_field_used="none",
            pit_reference_date=None,
            pit_admissible=True,
            pit_reason="admissible",
            endpoint_classification="neutral",
            endpoint_matched_pattern=None,
        )
        result = _dedup_trials_by_nct_id([t1, t2])
        assert len(result) == 1

    def test_pit_admissible_preferred(self, sample_pit_record):
        """PIT-admissible should be preferred over non-admissible."""
        t1 = sample_pit_record
        t1.pit_admissible = False
        t2 = TrialPITRecord(
            nct_id="NCT12345678",  # Same NCT ID
            ticker="ACME",
            phase="phase 2",
            status=TrialStatus.ACTIVE,
            conditions=[],
            primary_endpoint="",
            randomized=False,
            blinded="",
            last_update_posted="2025-11-01",
            pit_date_field_used="none",
            pit_reference_date=None,
            pit_admissible=True,  # Admissible
            pit_reason="admissible",
            endpoint_classification="neutral",
            endpoint_matched_pattern=None,
        )
        result = _dedup_trials_by_nct_id([t1, t2])
        assert len(result) == 1
        assert result[0].pit_admissible is True


# ============================================================================
# MAIN FUNCTION TESTS
# ============================================================================

class TestComputeModule4ClinicalDevV2:
    """Tests for compute_module_4_clinical_dev_v2 function."""

    def test_basic_scoring(self, as_of_date, sample_trial_record):
        """Basic scoring should work."""
        result = compute_module_4_clinical_dev_v2(
            trial_records=[sample_trial_record],
            active_tickers=["ACME"],
            as_of_date=as_of_date,
        )
        assert "scores" in result
        assert "diagnostic_counts" in result
        assert "provenance" in result
        assert result["as_of_date"] == as_of_date

    def test_empty_tickers(self, as_of_date, sample_trial_record):
        """Empty tickers should return empty scores."""
        result = compute_module_4_clinical_dev_v2(
            trial_records=[sample_trial_record],
            active_tickers=[],
            as_of_date=as_of_date,
        )
        assert result["scores"] == []
        assert result["diagnostic_counts"]["tickers_scored"] == 0

    def test_empty_trials(self, as_of_date):
        """Empty trials should still score tickers with no_trials flag."""
        result = compute_module_4_clinical_dev_v2(
            trial_records=[],
            active_tickers=["ACME"],
            as_of_date=as_of_date,
        )
        assert len(result["scores"]) == 1
        assert "no_trials" in result["scores"][0]["flags"]

    def test_pit_filtering(self, as_of_date):
        """Future-dated trials should be filtered."""
        future_trial = {
            "ticker": "ACME",
            "nct_id": "NCT12345678",
            "phase": "PHASE2",
            "status": "ACTIVE",
            "first_posted": "2026-02-01",  # After as_of_date
        }
        result = compute_module_4_clinical_dev_v2(
            trial_records=[future_trial],
            active_tickers=["ACME"],
            as_of_date=as_of_date,
        )
        # Trial should be filtered, ticker should have no_trials
        assert "no_trials" in result["scores"][0]["flags"]

    def test_multiple_tickers(self, as_of_date):
        """Multiple tickers should all be scored."""
        trials = [
            {
                "ticker": "ACME",
                "nct_id": "NCT00000001",
                "phase": "PHASE2",
                "status": "ACTIVE",
                "first_posted": "2025-01-01",
            },
            {
                "ticker": "BETA",
                "nct_id": "NCT00000002",
                "phase": "PHASE3",
                "status": "COMPLETED",
                "first_posted": "2025-01-01",
            },
        ]
        result = compute_module_4_clinical_dev_v2(
            trial_records=trials,
            active_tickers=["ACME", "BETA"],
            as_of_date=as_of_date,
        )
        assert result["diagnostic_counts"]["tickers_scored"] == 2
        tickers = [s["ticker"] for s in result["scores"]]
        assert "ACME" in tickers
        assert "BETA" in tickers

    def test_accepts_set_tickers(self, as_of_date, sample_trial_record):
        """Should accept Set[str] for active_tickers."""
        result = compute_module_4_clinical_dev_v2(
            trial_records=[sample_trial_record],
            active_tickers={"ACME"},  # Set
            as_of_date=as_of_date,
        )
        assert len(result["scores"]) == 1

    def test_accepts_date_object(self, sample_trial_record):
        """Should accept date object for as_of_date."""
        result = compute_module_4_clinical_dev_v2(
            trial_records=[sample_trial_record],
            active_tickers=["ACME"],
            as_of_date=date(2026, 1, 15),
        )
        assert result["as_of_date"] == "2026-01-15"

    def test_deterministic_output(self, as_of_date, sample_trial_record):
        """Same inputs should produce identical output."""
        result1 = compute_module_4_clinical_dev_v2(
            trial_records=[sample_trial_record],
            active_tickers=["ACME"],
            as_of_date=as_of_date,
        )
        result2 = compute_module_4_clinical_dev_v2(
            trial_records=[sample_trial_record],
            active_tickers=["ACME"],
            as_of_date=as_of_date,
        )
        assert result1["scores"] == result2["scores"]

    def test_diagnostic_counts(self, as_of_date, sample_trial_record):
        """Diagnostic counts should be accurate."""
        result = compute_module_4_clinical_dev_v2(
            trial_records=[sample_trial_record],
            active_tickers=["ACME"],
            as_of_date=as_of_date,
        )
        diag = result["diagnostic_counts"]
        assert diag["total_trials_raw"] == 1
        assert "pit_fields_used" in diag
        assert "status_distribution" in diag
        assert "endpoint_distribution" in diag


class TestComputeModule4ClinicalDev:
    """Tests for compute_module_4_clinical_dev wrapper function."""

    def test_backwards_compatible_output(self, as_of_date, sample_trial_record):
        """Wrapper should produce backwards-compatible output."""
        result = compute_module_4_clinical_dev(
            trial_records=[sample_trial_record],
            active_tickers=["ACME"],
            as_of_date=as_of_date,
        )
        # Check for v1 keys
        score = result["scores"][0]
        assert "ticker" in score
        assert "clinical_score" in score
        assert "lead_phase" in score
        assert "trial_count" in score  # v1 field name
        assert "n_trials_unique" in score  # Also present

    def test_diagnostic_counts_format(self, as_of_date, sample_trial_record):
        """Diagnostic counts should have v1 format."""
        result = compute_module_4_clinical_dev(
            trial_records=[sample_trial_record],
            active_tickers=["ACME"],
            as_of_date=as_of_date,
        )
        diag = result["diagnostic_counts"]
        assert "scored" in diag  # v1 key name
        assert "total_trials" in diag  # v1 key name


# ============================================================================
# DETERMINISM TESTS
# ============================================================================

class TestDeterminism:
    """Tests verifying deterministic behavior."""

    def test_repeated_scoring(self, as_of_date):
        """Multiple runs should produce identical results."""
        trials = [
            {
                "ticker": "ACME",
                "nct_id": f"NCT0000000{i}",
                "phase": "PHASE2",
                "status": "ACTIVE",
                "first_posted": "2025-01-01",
                "conditions": "Cancer",
            }
            for i in range(10)
        ]
        results = [
            compute_module_4_clinical_dev_v2(
                trial_records=trials,
                active_tickers=["ACME"],
                as_of_date=as_of_date,
            )
            for _ in range(5)
        ]
        # All results should be identical
        first_score = results[0]["scores"][0]["clinical_score"]
        for r in results[1:]:
            assert r["scores"][0]["clinical_score"] == first_score

    def test_ticker_order_independence(self, as_of_date):
        """Ticker order should not affect results."""
        trials = [
            {"ticker": "ACME", "nct_id": "NCT00000001", "phase": "PHASE2",
             "status": "ACTIVE", "first_posted": "2025-01-01"},
            {"ticker": "BETA", "nct_id": "NCT00000002", "phase": "PHASE3",
             "status": "ACTIVE", "first_posted": "2025-01-01"},
        ]
        result1 = compute_module_4_clinical_dev_v2(
            trial_records=trials,
            active_tickers=["ACME", "BETA"],
            as_of_date=as_of_date,
        )
        result2 = compute_module_4_clinical_dev_v2(
            trial_records=trials,
            active_tickers=["BETA", "ACME"],  # Different order
            as_of_date=as_of_date,
        )
        # Scores should be the same (output is sorted by ticker)
        assert result1["scores"] == result2["scores"]

    def test_trial_order_independence(self, as_of_date):
        """Trial record order should not affect results."""
        trial1 = {"ticker": "ACME", "nct_id": "NCT00000001", "phase": "PHASE2",
                  "status": "ACTIVE", "first_posted": "2025-01-01"}
        trial2 = {"ticker": "ACME", "nct_id": "NCT00000002", "phase": "PHASE3",
                  "status": "COMPLETED", "first_posted": "2025-01-01"}

        result1 = compute_module_4_clinical_dev_v2(
            trial_records=[trial1, trial2],
            active_tickers=["ACME"],
            as_of_date=as_of_date,
        )
        result2 = compute_module_4_clinical_dev_v2(
            trial_records=[trial2, trial1],  # Different order
            active_tickers=["ACME"],
            as_of_date=as_of_date,
        )
        assert result1["scores"][0]["clinical_score"] == result2["scores"][0]["clinical_score"]


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

class TestEdgeCases:
    """Edge case tests for clinical development scoring."""

    def test_ticker_not_in_active_list(self, as_of_date):
        """Trials for non-active tickers should be ignored."""
        trial = {
            "ticker": "UNKNOWN",
            "nct_id": "NCT00000001",
            "phase": "PHASE2",
            "status": "ACTIVE",
            "first_posted": "2025-01-01",
        }
        result = compute_module_4_clinical_dev_v2(
            trial_records=[trial],
            active_tickers=["ACME"],  # UNKNOWN not in list
            as_of_date=as_of_date,
        )
        # ACME should have no_trials flag
        assert "no_trials" in result["scores"][0]["flags"]

    def test_missing_nct_id(self, as_of_date):
        """Trials without nct_id should be skipped."""
        trial = {
            "ticker": "ACME",
            "phase": "PHASE2",
            "status": "ACTIVE",
            "first_posted": "2025-01-01",
        }
        result = compute_module_4_clinical_dev_v2(
            trial_records=[trial],
            active_tickers=["ACME"],
            as_of_date=as_of_date,
        )
        # Should have validation issue counted
        assert result["diagnostic_counts"]["validation_issues"]["empty_nct_id"] == 1

    def test_very_long_endpoint(self, as_of_date):
        """Very long endpoint should be truncated."""
        trial = {
            "ticker": "ACME",
            "nct_id": "NCT00000001",
            "phase": "PHASE2",
            "status": "ACTIVE",
            "first_posted": "2025-01-01",
            "primary_endpoint": "x" * (MAX_ENDPOINT_LENGTH + 100),
        }
        result = compute_module_4_clinical_dev_v2(
            trial_records=[trial],
            active_tickers=["ACME"],
            as_of_date=as_of_date,
        )
        assert result["diagnostic_counts"]["validation_issues"]["endpoint_truncated"] == 1

    def test_all_statuses_handled(self, as_of_date):
        """All trial statuses should be handled without error."""
        statuses = [
            "COMPLETED", "ACTIVE", "RECRUITING", "NOT_YET_RECRUITING",
            "ENROLLING_BY_INVITATION", "SUSPENDED", "TERMINATED", "WITHDRAWN",
        ]
        trials = [
            {
                "ticker": "ACME",
                "nct_id": f"NCT0000000{i}",
                "phase": "PHASE2",
                "status": status,
                "first_posted": "2025-01-01",
            }
            for i, status in enumerate(statuses)
        ]
        result = compute_module_4_clinical_dev_v2(
            trial_records=trials,
            active_tickers=["ACME"],
            as_of_date=as_of_date,
        )
        assert len(result["scores"]) == 1

    def test_all_phases_handled(self, as_of_date):
        """All phase types should be handled."""
        phases = ["PHASE1", "PHASE2", "PHASE3", "PHASE1_PHASE2", "PHASE2_PHASE3", "APPROVED"]
        trials = [
            {
                "ticker": "ACME",
                "nct_id": f"NCT0000000{i}",
                "phase": phase,
                "status": "ACTIVE",
                "first_posted": "2025-01-01",
            }
            for i, phase in enumerate(phases)
        ]
        result = compute_module_4_clinical_dev_v2(
            trial_records=trials,
            active_tickers=["ACME"],
            as_of_date=as_of_date,
        )
        # Lead phase should be approved (highest)
        assert result["scores"][0]["lead_phase"] == "approved"


# ============================================================================
# DATACLASS TESTS
# ============================================================================

class TestTrialPITRecord:
    """Tests for TrialPITRecord dataclass."""

    def test_to_dict(self, sample_pit_record):
        """to_dict should produce serializable dict."""
        d = sample_pit_record.to_dict()
        assert d["nct_id"] == "NCT12345678"
        assert d["ticker"] == "ACME"
        assert d["status"] == "active"  # Enum value
        assert d["pit_admissible"] is True


class TestTickerClinicalSummaryV2:
    """Tests for TickerClinicalSummaryV2 dataclass."""

    def test_to_dict(self, as_of_date):
        """to_dict should produce serializable dict."""
        summary = TickerClinicalSummaryV2(
            ticker="ACME",
            as_of_date=as_of_date,
            clinical_score=Decimal("75.50"),
            phase_score=Decimal("18"),
            phase_progress=Decimal("3"),
            trial_count_bonus=Decimal("2"),
            diversity_bonus=Decimal("1.5"),
            recency_bonus=Decimal("5"),
            design_score=Decimal("21"),
            execution_score=Decimal("15"),
            endpoint_score=Decimal("12"),
            lead_phase="phase 2",
            lead_trial_nct_id="NCT12345678",
            lead_program_key="abc123",
            n_trials_raw=5,
            n_trials_unique=5,
            n_trials_pit_admissible=5,
            pit_filtered_count=0,
            completion_rate=Decimal("0.6"),
            termination_rate=Decimal("0.1"),
            status_quality_score=Decimal("0.75"),
            n_strong_endpoints=2,
            n_weak_endpoints=1,
            n_neutral_endpoints=2,
            recency_days=30,
            recency_unknown=False,
            recency_stale=False,
            flags=["early_stage"],
            severity="none",
        )
        d = summary.to_dict()
        assert d["ticker"] == "ACME"
        assert d["clinical_score"] == "75.50"  # Decimal as string
        assert d["_schema"]["schema_version"] is not None
