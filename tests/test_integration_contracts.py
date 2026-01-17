"""
Tests for Integration Contracts.

Validates:
1. Schema validation functions work correctly
2. Score extraction helpers handle all formats
3. Version compatibility checks
4. Normalization helpers
5. Cross-module handoff validation
"""
import pytest
from datetime import date
from decimal import Decimal

from common.integration_contracts import (
    # Validation functions
    validate_module_1_output,
    validate_module_2_output,
    validate_module_3_output,
    validate_module_4_output,
    validate_module_5_output,
    validate_pipeline_handoff,
    SchemaValidationError,
    # Score extraction
    extract_financial_score,
    extract_catalyst_score,
    extract_clinical_score,
    extract_market_cap_mm,
    # Normalization
    normalize_date_input,
    normalize_date_string,
    normalize_ticker_set,
    # Version checking
    check_schema_version,
    SUPPORTED_SCHEMA_VERSIONS,
    # Type re-exports
    EventType,
    EventSeverity,
    TickerCatalystSummaryV2,
)


class TestModule1Validation:
    """Test Module 1 output schema validation."""

    def test_valid_output(self):
        """Valid Module 1 output passes validation."""
        output = {
            "as_of_date": "2024-01-15",
            "active_securities": [{"ticker": "AAPL", "status": "active"}],
            "excluded_securities": [],
            "diagnostic_counts": {"active": 1, "excluded": 0},
        }
        # Should not raise
        validate_module_1_output(output)

    def test_missing_active_securities(self):
        """Missing active_securities raises error."""
        output = {
            "excluded_securities": [],
            "diagnostic_counts": {},
        }
        with pytest.raises(SchemaValidationError, match="missing keys"):
            validate_module_1_output(output)

    def test_invalid_active_securities_type(self):
        """Non-list active_securities raises error."""
        output = {
            "active_securities": "not a list",
            "excluded_securities": [],
            "diagnostic_counts": {},
        }
        with pytest.raises(SchemaValidationError, match="must be a list"):
            validate_module_1_output(output)


class TestModule2Validation:
    """Test Module 2 output schema validation."""

    def test_valid_output(self):
        """Valid Module 2 output passes validation."""
        output = {
            "scores": [
                {"ticker": "AAPL", "financial_score": 85.5, "severity": "none"}
            ],
            "diagnostic_counts": {"scored": 1},
        }
        validate_module_2_output(output)

    def test_missing_scores(self):
        """Missing scores raises error."""
        output = {"diagnostic_counts": {}}
        with pytest.raises(SchemaValidationError, match="missing keys"):
            validate_module_2_output(output)

    def test_score_record_missing_financial_score(self):
        """Score record without financial_score or financial_normalized raises."""
        output = {
            "scores": [{"ticker": "AAPL", "severity": "none"}],
            "diagnostic_counts": {},
        }
        with pytest.raises(SchemaValidationError, match="missing.*financial"):
            validate_module_2_output(output)

    def test_legacy_field_name_accepted(self):
        """Legacy financial_normalized field is accepted."""
        output = {
            "scores": [
                {"ticker": "AAPL", "financial_normalized": 85.5, "severity": "none"}
            ],
            "diagnostic_counts": {},
        }
        validate_module_2_output(output)


class TestModule3Validation:
    """Test Module 3 output schema validation."""

    def test_valid_output(self):
        """Valid Module 3 output passes validation."""
        output = {
            "summaries": {"AAPL": {"score_blended": 50}},
            "diagnostic_counts": {"events_detected": 0},
            "as_of_date": "2024-01-15",
        }
        validate_module_3_output(output)

    def test_missing_summaries(self):
        """Missing summaries raises error."""
        output = {
            "diagnostic_counts": {},
            "as_of_date": "2024-01-15",
        }
        with pytest.raises(SchemaValidationError, match="missing keys"):
            validate_module_3_output(output)

    def test_summaries_must_be_dict(self):
        """summaries must be a dict."""
        output = {
            "summaries": [],
            "diagnostic_counts": {},
            "as_of_date": "2024-01-15",
        }
        with pytest.raises(SchemaValidationError, match="must be a dict"):
            validate_module_3_output(output)


class TestModule4Validation:
    """Test Module 4 output schema validation."""

    def test_valid_output(self):
        """Valid Module 4 output passes validation."""
        output = {
            "scores": [
                {"ticker": "AAPL", "clinical_score": "75.5", "lead_phase": "Phase 3"}
            ],
            "diagnostic_counts": {"scored": 1},
            "as_of_date": "2024-01-15",
        }
        validate_module_4_output(output)

    def test_missing_as_of_date(self):
        """Missing as_of_date raises error."""
        output = {
            "scores": [],
            "diagnostic_counts": {},
        }
        with pytest.raises(SchemaValidationError, match="missing keys"):
            validate_module_4_output(output)


class TestModule5Validation:
    """Test Module 5 output schema validation."""

    def test_valid_output(self):
        """Valid Module 5 output passes validation."""
        output = {
            "ranked_securities": [
                {"ticker": "AAPL", "composite_score": "85.5", "composite_rank": 1}
            ],
            "excluded_securities": [],
            "diagnostic_counts": {"rankable": 1},
        }
        validate_module_5_output(output)

    def test_missing_ranked_securities(self):
        """Missing ranked_securities raises error."""
        output = {
            "excluded_securities": [],
            "diagnostic_counts": {},
        }
        with pytest.raises(SchemaValidationError, match="missing keys"):
            validate_module_5_output(output)


class TestPipelineHandoff:
    """Test cross-module handoff validation."""

    def test_module_1_to_module_2_handoff(self):
        """Validate Module 1 -> Module 2 handoff."""
        m1_output = {
            "active_securities": [{"ticker": "AAPL"}],
            "excluded_securities": [],
            "diagnostic_counts": {},
        }
        validate_pipeline_handoff("module_1", "module_2", m1_output)

    def test_unknown_source_module_passes_without_validation(self):
        """Unknown source module passes validation (no validator defined)."""
        # When source module is unknown, validate_pipeline_handoff uses the source module validator
        # which doesn't exist, so it passes through
        validate_pipeline_handoff("module_99", "module_2", {"any": "data"})

    def test_valid_module_2_to_module_5_handoff(self):
        """Validate Module 2 -> Module 5 handoff."""
        m2_output = {
            "scores": [{"ticker": "AAPL", "financial_score": 85.0}],
            "diagnostic_counts": {"scored": 1},
        }
        validate_pipeline_handoff("module_2", "module_5", m2_output)


class TestScoreExtraction:
    """Test score extraction helpers."""

    def test_extract_financial_score_standard(self):
        """Extract financial_score from standard field."""
        record = {"financial_score": 85.5}
        assert extract_financial_score(record) == 85.5

    def test_extract_financial_score_legacy(self):
        """Extract financial_score from legacy field."""
        record = {"financial_normalized": 90.0}
        assert extract_financial_score(record) == 90.0

    def test_extract_financial_score_string(self):
        """Extract financial_score from string value."""
        record = {"financial_score": "75.5"}
        assert extract_financial_score(record) == 75.5

    def test_extract_financial_score_missing(self):
        """Return None when no financial score present."""
        record = {"ticker": "AAPL"}
        assert extract_financial_score(record) is None

    def test_extract_clinical_score_decimal_string(self):
        """Extract clinical_score from Decimal string."""
        record = {"clinical_score": "82.50"}
        assert extract_clinical_score(record) == 82.50

    def test_extract_clinical_score_missing(self):
        """Return None when no clinical score present."""
        record = {}
        assert extract_clinical_score(record) is None

    def test_extract_catalyst_score_from_dataclass(self):
        """Extract catalyst score from dataclass with score_blended."""
        # Create a mock object with score_blended attribute
        class MockSummary:
            score_blended = 65.5

        assert extract_catalyst_score(MockSummary()) == 65.5

    def test_extract_catalyst_score_from_dict(self):
        """Extract catalyst score from dict."""
        record = {"catalyst_score": 70.0}
        assert extract_catalyst_score(record) == 70.0

    def test_extract_catalyst_score_legacy(self):
        """Extract catalyst score from legacy field."""
        record = {"catalyst_score_net": 55.0}
        assert extract_catalyst_score(record) == 55.0

    def test_extract_market_cap_mm(self):
        """Extract market_cap_mm from record."""
        record = {"market_cap_mm": 5000.0}
        assert extract_market_cap_mm(record) == 5000.0

    def test_extract_market_cap_mm_string(self):
        """Extract market_cap_mm from string value."""
        record = {"market_cap_mm": "2500.5"}
        assert extract_market_cap_mm(record) == 2500.5


class TestNormalization:
    """Test normalization helpers."""

    def test_normalize_date_input_from_date(self):
        """Normalize date object returns date."""
        d = date(2024, 1, 15)
        assert normalize_date_input(d) == d

    def test_normalize_date_input_from_string(self):
        """Normalize ISO string returns date."""
        result = normalize_date_input("2024-01-15")
        assert result == date(2024, 1, 15)

    def test_normalize_date_input_invalid(self):
        """Invalid date string raises ValueError."""
        with pytest.raises(ValueError):
            normalize_date_input("not-a-date")

    def test_normalize_date_string_from_date(self):
        """Normalize date object to ISO string."""
        result = normalize_date_string(date(2024, 1, 15))
        assert result == "2024-01-15"

    def test_normalize_date_string_from_string(self):
        """Normalize ISO string returns same string."""
        result = normalize_date_string("2024-01-15")
        assert result == "2024-01-15"

    def test_normalize_ticker_set_from_set(self):
        """Normalize set returns set."""
        tickers = {"AAPL", "GOOG"}
        result = normalize_ticker_set(tickers)
        assert result == tickers

    def test_normalize_ticker_set_from_list(self):
        """Normalize list returns set."""
        tickers = ["AAPL", "GOOG", "AAPL"]  # Duplicate
        result = normalize_ticker_set(tickers)
        assert result == {"AAPL", "GOOG"}

    def test_normalize_ticker_set_preserves_case(self):
        """Normalize preserves ticker case (uppercasing is caller's responsibility)."""
        tickers = ["aapl", "Goog"]
        result = normalize_ticker_set(tickers)
        # normalize_ticker_set just converts to set, doesn't uppercase
        assert result == {"aapl", "Goog"}


class TestVersionChecking:
    """Test schema version checking."""

    def test_supported_version_returns_true(self):
        """Supported version returns (True, version)."""
        result = {"schema_version": "m3catalyst_vnext_20260111"}
        is_supported, version = check_schema_version("module_3", result)
        assert is_supported is True
        assert version == "m3catalyst_vnext_20260111"

    def test_unsupported_version_returns_false(self):
        """Unsupported version returns (False, version) and warns."""
        result = {"schema_version": "unknown_version_999"}
        with pytest.warns(DeprecationWarning):
            is_supported, version = check_schema_version("module_3", result)
        assert is_supported is False
        assert version == "unknown_version_999"

    def test_missing_version_allowed(self):
        """Missing version returns (True, None) for backwards compat."""
        result = {}
        is_supported, version = check_schema_version("module_3", result)
        assert is_supported is True
        assert version is None

    def test_nested_schema_version(self):
        """Version in _schema dict is found."""
        result = {"_schema": {"schema_version": "m3catalyst_vnext_20260111"}}
        is_supported, version = check_schema_version("module_3", result)
        assert is_supported is True

    def test_unknown_module_always_supported(self):
        """Unknown module returns (True, None)."""
        result = {"schema_version": "anything"}
        is_supported, version = check_schema_version("module_99", result)
        assert is_supported is True

    def test_supported_versions_dict_exists(self):
        """SUPPORTED_SCHEMA_VERSIONS has expected modules."""
        assert "module_1" in SUPPORTED_SCHEMA_VERSIONS
        assert "module_3" in SUPPORTED_SCHEMA_VERSIONS
        assert "module_5" in SUPPORTED_SCHEMA_VERSIONS


class TestTypeReexports:
    """Test that Module 3 types are properly re-exported."""

    def test_event_type_enum(self):
        """EventType enum is accessible."""
        assert EventType.CT_STATUS_SEVERE_NEG.value == "CT_STATUS_SEVERE_NEG"
        assert EventType.CT_TIMELINE_PUSHOUT.value == "CT_TIMELINE_PUSHOUT"

    def test_event_severity_enum(self):
        """EventSeverity enum is accessible."""
        assert EventSeverity.CRITICAL_POSITIVE.value == "CRITICAL_POSITIVE"
        assert EventSeverity.SEVERE_NEGATIVE.value == "SEVERE_NEGATIVE"

    def test_ticker_catalyst_summary_v2_exists(self):
        """TickerCatalystSummaryV2 is importable."""
        assert TickerCatalystSummaryV2 is not None


class TestDeterminism:
    """Test that validation is deterministic."""

    def test_same_input_same_result(self):
        """Same input always produces same validation result."""
        output = {
            "active_securities": [{"ticker": "A"}, {"ticker": "B"}],
            "excluded_securities": [],
            "diagnostic_counts": {"active": 2},
        }

        # Run validation multiple times
        for _ in range(10):
            validate_module_1_output(output)  # Should not raise

    def test_validation_order_independent(self):
        """Validation works regardless of dict key order."""
        output1 = {
            "active_securities": [],
            "excluded_securities": [],
            "diagnostic_counts": {},
        }
        output2 = {
            "diagnostic_counts": {},
            "excluded_securities": [],
            "active_securities": [],
        }

        # Both should pass
        validate_module_1_output(output1)
        validate_module_1_output(output2)
