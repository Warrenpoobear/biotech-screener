#!/usr/bin/env python3
"""
Unit tests for common/data_consistency.py

Tests data consistency validation utilities:
- ConsistencyIssue and ConsistencyReport dataclasses
- Ticker extraction from records
- Duplicate detection
- Coverage checking
- Data consistency validation across files
- Record completeness validation
"""

import pytest
from datetime import date
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from common.data_consistency import (
    ConsistencyIssue,
    ConsistencyReport,
    extract_tickers,
    find_duplicates,
    check_coverage,
    validate_data_consistency,
    validate_record_completeness,
    log_consistency_report,
)


# ============================================================================
# TEST FIXTURES
# ============================================================================

@pytest.fixture
def sample_universe():
    """Sample universe records."""
    return [
        {"ticker": "ACME", "status": "active", "market_cap_mm": 500},
        {"ticker": "BETA", "status": "active", "market_cap_mm": 1000},
        {"ticker": "GAMMA", "status": "active", "market_cap_mm": 1500},
    ]


@pytest.fixture
def sample_financial_records():
    """Sample financial records."""
    return [
        {"ticker": "ACME", "cash": 100000000, "burn_rate": -5000000},
        {"ticker": "BETA", "cash": 200000000, "burn_rate": -10000000},
    ]


@pytest.fixture
def sample_trial_records():
    """Sample trial records."""
    return [
        {"nct_id": "NCT11111111", "lead_sponsor_ticker": "ACME", "phase": "PHASE3"},
        {"nct_id": "NCT22222222", "lead_sponsor_ticker": "BETA", "phase": "PHASE2"},
        {"nct_id": "NCT33333333", "lead_sponsor_ticker": "ACME", "phase": "PHASE2"},
    ]


@pytest.fixture
def sample_market_data():
    """Sample market data records."""
    return [
        {"ticker": "ACME", "price": 25.50, "volume": 1000000},
        {"ticker": "BETA", "price": 45.00, "volume": 500000},
        {"ticker": "GAMMA", "price": 32.75, "volume": 750000},
    ]


# ============================================================================
# CONSISTENCY ISSUE TESTS
# ============================================================================

class TestConsistencyIssue:
    """Tests for ConsistencyIssue dataclass."""

    def test_basic_creation(self):
        """Should create issue with required fields."""
        issue = ConsistencyIssue(
            severity="error",
            category="missing",
            message="Test issue",
        )

        assert issue.severity == "error"
        assert issue.category == "missing"
        assert issue.message == "Test issue"

    def test_default_lists(self):
        """Should have empty default lists."""
        issue = ConsistencyIssue(
            severity="warning",
            category="orphan",
            message="Test",
        )

        assert issue.affected_tickers == []
        assert issue.details == {}

    def test_str_representation(self):
        """String should include key info."""
        issue = ConsistencyIssue(
            severity="error",
            category="missing",
            message="Missing data for 5 tickers",
            affected_tickers=["ACME", "BETA", "GAMMA"],
        )

        s = str(issue)

        assert "[ERROR]" in s
        assert "missing" in s
        assert "ACME" in s

    def test_str_truncates_long_list(self):
        """String should truncate long ticker lists."""
        issue = ConsistencyIssue(
            severity="error",
            category="missing",
            message="Missing data",
            affected_tickers=[f"TICK{i}" for i in range(10)],
        )

        s = str(issue)

        assert "...and 5 more" in s


# ============================================================================
# CONSISTENCY REPORT TESTS
# ============================================================================

class TestConsistencyReport:
    """Tests for ConsistencyReport dataclass."""

    def test_valid_report(self):
        """Valid report should indicate success."""
        report = ConsistencyReport(valid=True, issues=[], stats={"count": 100})

        assert report.valid is True
        assert len(report.issues) == 0

    def test_invalid_report(self):
        """Invalid report should contain errors."""
        issue = ConsistencyIssue(
            severity="error",
            category="missing",
            message="Critical error",
        )
        report = ConsistencyReport(valid=False, issues=[issue])

        assert report.valid is False
        assert report.has_errors() is True

    def test_errors_method(self):
        """errors() should return only errors."""
        issues = [
            ConsistencyIssue(severity="error", category="missing", message="Error 1"),
            ConsistencyIssue(severity="warning", category="orphan", message="Warning 1"),
            ConsistencyIssue(severity="error", category="duplicate", message="Error 2"),
        ]
        report = ConsistencyReport(valid=False, issues=issues)

        errors = report.errors()

        assert len(errors) == 2
        assert all(e.severity == "error" for e in errors)

    def test_warnings_method(self):
        """warnings() should return only warnings."""
        issues = [
            ConsistencyIssue(severity="error", category="missing", message="Error 1"),
            ConsistencyIssue(severity="warning", category="orphan", message="Warning 1"),
            ConsistencyIssue(severity="info", category="missing", message="Info 1"),
        ]
        report = ConsistencyReport(valid=False, issues=issues)

        warnings = report.warnings()

        assert len(warnings) == 1
        assert warnings[0].severity == "warning"

    def test_summary(self):
        """summary() should include error and warning counts."""
        issues = [
            ConsistencyIssue(severity="error", category="missing", message="Error"),
            ConsistencyIssue(severity="warning", category="orphan", message="Warning 1"),
            ConsistencyIssue(severity="warning", category="duplicate", message="Warning 2"),
        ]
        report = ConsistencyReport(valid=False, issues=issues)

        summary = report.summary()

        assert "1 errors" in summary
        assert "2 warnings" in summary


# ============================================================================
# EXTRACT TICKERS TESTS
# ============================================================================

class TestExtractTickers:
    """Tests for extract_tickers function."""

    def test_basic_extraction(self, sample_universe):
        """Should extract tickers from records."""
        tickers = extract_tickers(sample_universe)

        assert tickers == {"ACME", "BETA", "GAMMA"}

    def test_custom_field(self):
        """Should use custom ticker field."""
        records = [
            {"symbol": "ACME", "name": "Acme Corp"},
            {"symbol": "BETA", "name": "Beta Inc"},
        ]

        tickers = extract_tickers(records, ticker_field="symbol")

        assert tickers == {"ACME", "BETA"}

    def test_uppercase_normalization(self):
        """Should normalize tickers to uppercase."""
        records = [
            {"ticker": "acme"},
            {"ticker": "Beta"},
            {"ticker": "GAMMA"},
        ]

        tickers = extract_tickers(records)

        assert tickers == {"ACME", "BETA", "GAMMA"}

    def test_handles_missing_field(self):
        """Should skip records without ticker field."""
        records = [
            {"ticker": "ACME"},
            {"name": "No Ticker"},  # Missing ticker
            {"ticker": "BETA"},
        ]

        tickers = extract_tickers(records)

        assert tickers == {"ACME", "BETA"}

    def test_handles_none_values(self):
        """Should skip records with None ticker."""
        records = [
            {"ticker": "ACME"},
            {"ticker": None},
            {"ticker": "BETA"},
        ]

        tickers = extract_tickers(records)

        assert tickers == {"ACME", "BETA"}

    def test_empty_records(self):
        """Should handle empty record list."""
        tickers = extract_tickers([])

        assert tickers == set()


# ============================================================================
# FIND DUPLICATES TESTS
# ============================================================================

class TestFindDuplicates:
    """Tests for find_duplicates function."""

    def test_no_duplicates(self):
        """Should return empty for unique records."""
        records = [
            {"ticker": "ACME"},
            {"ticker": "BETA"},
            {"ticker": "GAMMA"},
        ]

        dups = find_duplicates(records, ["ticker"])

        assert len(dups) == 0

    def test_finds_duplicates(self):
        """Should find duplicate records."""
        records = [
            {"ticker": "ACME"},
            {"ticker": "BETA"},
            {"ticker": "ACME"},  # Duplicate
        ]

        dups = find_duplicates(records, ["ticker"])

        assert len(dups) == 1
        assert "ACME" in list(dups.keys())[0]
        assert dups[list(dups.keys())[0]] == [0, 2]  # Indices

    def test_compound_key(self):
        """Should use compound key for duplicates."""
        records = [
            {"ticker": "ACME", "date": "2026-01-01"},
            {"ticker": "ACME", "date": "2026-01-02"},  # Different date
            {"ticker": "ACME", "date": "2026-01-01"},  # Duplicate
        ]

        dups = find_duplicates(records, ["ticker", "date"])

        assert len(dups) == 1

    def test_handles_missing_key_fields(self):
        """Should handle records missing key fields."""
        records = [
            {"ticker": "ACME", "date": "2026-01-01"},
            {"ticker": "BETA"},  # Missing date
            {"ticker": "ACME", "date": "2026-01-01"},  # Duplicate
        ]

        dups = find_duplicates(records, ["ticker", "date"])

        assert len(dups) == 1


# ============================================================================
# CHECK COVERAGE TESTS
# ============================================================================

class TestCheckCoverage:
    """Tests for check_coverage function."""

    def test_full_coverage(self):
        """Should report 100% coverage when all tickers present."""
        universe = {"ACME", "BETA", "GAMMA"}
        data = {"ACME", "BETA", "GAMMA"}

        missing, orphans, coverage = check_coverage(universe, data, "test")

        assert len(missing) == 0
        assert len(orphans) == 0
        assert coverage == 100.0

    def test_partial_coverage(self):
        """Should report partial coverage."""
        universe = {"ACME", "BETA", "GAMMA", "DELTA"}
        data = {"ACME", "BETA"}

        missing, orphans, coverage = check_coverage(universe, data, "test")

        assert missing == {"GAMMA", "DELTA"}
        assert len(orphans) == 0
        assert coverage == 50.0

    def test_identifies_orphans(self):
        """Should identify orphan tickers."""
        universe = {"ACME", "BETA"}
        data = {"ACME", "BETA", "GAMMA"}  # GAMMA is orphan

        missing, orphans, coverage = check_coverage(universe, data, "test")

        assert len(missing) == 0
        assert orphans == {"GAMMA"}
        assert coverage == 100.0

    def test_empty_universe(self):
        """Should handle empty universe."""
        universe = set()
        data = {"ACME", "BETA"}

        missing, orphans, coverage = check_coverage(universe, data, "test")

        assert len(missing) == 0
        assert coverage == 0


# ============================================================================
# VALIDATE DATA CONSISTENCY TESTS
# ============================================================================

class TestValidateDataConsistency:
    """Tests for validate_data_consistency function."""

    def test_all_consistent(self, sample_universe, sample_market_data):
        """Should pass when all data is consistent."""
        report = validate_data_consistency(
            universe=sample_universe,
            market_data=sample_market_data,
        )

        assert report.valid is True
        assert len(report.errors()) == 0

    def test_detects_missing_financial_data(self, sample_universe, sample_financial_records):
        """Should detect missing financial data."""
        # sample_financial_records missing GAMMA
        report = validate_data_consistency(
            universe=sample_universe,
            financial_records=sample_financial_records,
            min_coverage_pct=80.0,
        )

        # Should have a warning (66.7% coverage < 80%)
        assert any(i.category == "missing" and "Financial" in i.message
                   for i in report.issues)

    def test_detects_orphan_records(self, sample_universe):
        """Should detect orphan records."""
        financial_with_orphan = [
            {"ticker": "ACME", "cash": 100},
            {"ticker": "UNKNOWN", "cash": 50},  # Orphan
        ]

        report = validate_data_consistency(
            universe=sample_universe,
            financial_records=financial_with_orphan,
        )

        orphan_issues = [i for i in report.issues if i.category == "orphan"]
        assert len(orphan_issues) > 0

    def test_detects_duplicate_universe_tickers(self):
        """Should detect duplicate tickers in universe."""
        universe_with_dup = [
            {"ticker": "ACME", "status": "active"},
            {"ticker": "BETA", "status": "active"},
            {"ticker": "ACME", "status": "active"},  # Duplicate
        ]

        report = validate_data_consistency(universe=universe_with_dup)

        dup_issues = [i for i in report.issues if i.category == "duplicate"]
        assert len(dup_issues) > 0
        assert report.valid is False

    def test_detects_trial_duplicates(self, sample_universe):
        """Should detect duplicate trial NCT IDs."""
        trials_with_dup = [
            {"nct_id": "NCT11111111", "lead_sponsor_ticker": "ACME"},
            {"nct_id": "NCT22222222", "lead_sponsor_ticker": "BETA"},
            {"nct_id": "NCT11111111", "lead_sponsor_ticker": "ACME"},  # Duplicate
        ]

        report = validate_data_consistency(
            universe=sample_universe,
            trial_records=trials_with_dup,
        )

        dup_issues = [i for i in report.issues
                      if i.category == "duplicate" and "Trial" in i.message]
        assert len(dup_issues) > 0

    def test_coverage_below_threshold_is_error(self, sample_universe):
        """Coverage below threshold should be error."""
        partial_financial = [{"ticker": "ACME", "cash": 100}]  # Only 1 of 3

        report = validate_data_consistency(
            universe=sample_universe,
            financial_records=partial_financial,
            min_coverage_pct=50.0,  # 33% < 50%
        )

        error_issues = [i for i in report.issues
                        if i.severity == "error" and i.category == "missing"]
        assert len(error_issues) > 0

    def test_stats_populated(self, sample_universe, sample_financial_records):
        """Should populate stats dict."""
        report = validate_data_consistency(
            universe=sample_universe,
            financial_records=sample_financial_records,
        )

        assert "universe_count" in report.stats
        assert report.stats["universe_count"] == 3
        assert "financial_coverage_pct" in report.stats

    def test_handles_none_data_sources(self, sample_universe):
        """Should handle when optional data sources are None."""
        report = validate_data_consistency(
            universe=sample_universe,
            financial_records=None,
            trial_records=None,
            market_data=None,
            short_interest=None,
        )

        # Should still complete without error
        assert report.stats["universe_count"] == 3


# ============================================================================
# VALIDATE RECORD COMPLETENESS TESTS
# ============================================================================

class TestValidateRecordCompleteness:
    """Tests for validate_record_completeness function."""

    def test_all_fields_present(self):
        """Should pass when all required fields present."""
        records = [
            {"ticker": "ACME", "cash": 100, "burn": -10},
            {"ticker": "BETA", "cash": 200, "burn": -20},
        ]

        report = validate_record_completeness(
            records=records,
            required_fields=["ticker", "cash", "burn"],
            data_name="Financial",
        )

        assert report.valid is True
        assert len(report.issues) == 0

    def test_detects_missing_fields(self):
        """Should detect missing required fields."""
        records = [
            {"ticker": "ACME", "cash": 100},  # Missing burn
            {"ticker": "BETA", "cash": 200, "burn": -20},
        ]

        report = validate_record_completeness(
            records=records,
            required_fields=["ticker", "cash", "burn"],
            data_name="Financial",
        )

        missing_issues = [i for i in report.issues if "burn" in i.message]
        assert len(missing_issues) > 0

    def test_none_values_treated_as_missing(self):
        """Should treat None values as missing."""
        records = [
            {"ticker": "ACME", "cash": None},  # None cash
            {"ticker": "BETA", "cash": 200},
        ]

        report = validate_record_completeness(
            records=records,
            required_fields=["ticker", "cash"],
            data_name="Financial",
        )

        missing_issues = [i for i in report.issues if "cash" in i.message]
        assert len(missing_issues) > 0

    def test_severity_based_on_percentage(self):
        """Severity should scale with missing percentage."""
        # All records missing field = error
        records = [
            {"ticker": "ACME"},
            {"ticker": "BETA"},
        ]

        report = validate_record_completeness(
            records=records,
            required_fields=["ticker", "cash"],
            data_name="Financial",
        )

        # 100% missing = error
        assert any(i.severity == "error" for i in report.issues)

    def test_stats_populated(self):
        """Should populate stats."""
        records = [
            {"ticker": "ACME", "cash": 100},
            {"ticker": "BETA", "cash": 200},
        ]

        report = validate_record_completeness(
            records=records,
            required_fields=["ticker", "cash"],
            data_name="Financial",
        )

        assert report.stats["total_records"] == 2
        assert report.stats["fields_checked"] == ["ticker", "cash"]


# ============================================================================
# LOG CONSISTENCY REPORT TESTS
# ============================================================================

class TestLogConsistencyReport:
    """Tests for log_consistency_report function."""

    def test_logs_without_error(self, sample_universe, sample_market_data):
        """Should log report without raising errors."""
        report = validate_data_consistency(
            universe=sample_universe,
            market_data=sample_market_data,
        )

        # Should not raise
        log_consistency_report(report)


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

class TestEdgeCases:
    """Edge case tests for data consistency validation."""

    def test_empty_universe(self):
        """Should handle empty universe."""
        report = validate_data_consistency(universe=[])

        assert report.stats["universe_count"] == 0

    def test_single_record_universe(self):
        """Should handle single record universe."""
        universe = [{"ticker": "ACME", "status": "active"}]
        financial = [{"ticker": "ACME", "cash": 100}]

        report = validate_data_consistency(
            universe=universe,
            financial_records=financial,
        )

        assert report.valid is True
        assert report.stats["financial_coverage_pct"] == 100.0

    def test_mixed_case_tickers_normalized(self):
        """Should normalize mixed case tickers."""
        universe = [{"ticker": "acme"}, {"ticker": "BETA"}]
        financial = [{"ticker": "ACME"}, {"ticker": "beta"}]

        report = validate_data_consistency(
            universe=universe,
            financial_records=financial,
        )

        assert report.stats["financial_coverage_pct"] == 100.0

    def test_trial_ticker_field_fallback(self):
        """Should fall back to 'ticker' field for trials."""
        universe = [{"ticker": "ACME"}]
        trials = [
            {"ticker": "ACME", "nct_id": "NCT11111111"},  # Uses ticker, not lead_sponsor_ticker
        ]

        report = validate_data_consistency(
            universe=universe,
            trial_records=trials,
        )

        assert report.stats["trials_tickers_with_data"] == 1


# ============================================================================
# DETERMINISM TESTS
# ============================================================================

class TestDeterminism:
    """Tests verifying deterministic behavior."""

    def test_validate_consistency_deterministic(self, sample_universe, sample_financial_records):
        """validate_data_consistency should be deterministic."""
        reports = [
            validate_data_consistency(
                universe=sample_universe,
                financial_records=sample_financial_records,
            )
            for _ in range(5)
        ]

        # All reports should have same validity and issue count
        for i in range(1, len(reports)):
            assert reports[0].valid == reports[i].valid
            assert len(reports[0].issues) == len(reports[i].issues)

    def test_extract_tickers_deterministic(self, sample_universe):
        """extract_tickers should be deterministic."""
        results = [extract_tickers(sample_universe) for _ in range(5)]

        for i in range(1, len(results)):
            assert results[0] == results[i]
