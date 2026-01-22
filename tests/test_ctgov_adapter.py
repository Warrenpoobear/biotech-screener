#!/usr/bin/env python3
"""
Tests for ctgov_adapter.py

The CT.gov adapter converts trial records to canonical format for Module 3.
These tests cover:
- Status normalization
- Date extraction and parsing
- Field extraction from multiple input formats
- PIT validation
- Batch processing and validation gates
"""

import pytest
from datetime import date
from decimal import Decimal

from ctgov_adapter import (
    CTGovAdapter,
    CTGovStatus,
    CompletionType,
    CanonicalTrialRecord,
    AdapterConfig,
    AdapterStats,
    AdapterError,
    MissingRequiredFieldError,
    FutureDataError,
    process_trial_records_batch,
)


class TestCTGovStatus:
    """Tests for CTGovStatus enum."""

    def test_standard_status_parsing(self):
        """Standard status strings should be parsed correctly."""
        assert CTGovStatus.from_string("RECRUITING") == CTGovStatus.RECRUITING
        assert CTGovStatus.from_string("COMPLETED") == CTGovStatus.COMPLETED
        assert CTGovStatus.from_string("TERMINATED") == CTGovStatus.TERMINATED
        assert CTGovStatus.from_string("WITHDRAWN") == CTGovStatus.WITHDRAWN
        assert CTGovStatus.from_string("SUSPENDED") == CTGovStatus.SUSPENDED

    def test_case_insensitive_parsing(self):
        """Status parsing should be case-insensitive."""
        assert CTGovStatus.from_string("recruiting") == CTGovStatus.RECRUITING
        assert CTGovStatus.from_string("Recruiting") == CTGovStatus.RECRUITING
        assert CTGovStatus.from_string("RECRUITING") == CTGovStatus.RECRUITING

    def test_alias_parsing(self):
        """Status aliases should be recognized."""
        assert CTGovStatus.from_string("ACTIVE") == CTGovStatus.ACTIVE_NOT_RECRUITING
        assert CTGovStatus.from_string("APPROVED") == CTGovStatus.APPROVED_FOR_MARKETING

    def test_space_and_dash_normalization(self):
        """Spaces and dashes should be normalized to underscores."""
        assert CTGovStatus.from_string("ACTIVE NOT RECRUITING") == CTGovStatus.ACTIVE_NOT_RECRUITING
        assert CTGovStatus.from_string("ACTIVE-NOT-RECRUITING") == CTGovStatus.ACTIVE_NOT_RECRUITING
        assert CTGovStatus.from_string("NOT YET RECRUITING") == CTGovStatus.NOT_YET_RECRUITING

    def test_unknown_status(self):
        """Unknown status strings should return UNKNOWN."""
        assert CTGovStatus.from_string("INVALID_STATUS") == CTGovStatus.UNKNOWN
        assert CTGovStatus.from_string("xyz") == CTGovStatus.UNKNOWN

    def test_empty_status(self):
        """Empty/None status should return UNKNOWN."""
        assert CTGovStatus.from_string("") == CTGovStatus.UNKNOWN
        assert CTGovStatus.from_string(None) == CTGovStatus.UNKNOWN

    def test_terminal_negative_property(self):
        """Terminal negative statuses should be identified."""
        assert CTGovStatus.WITHDRAWN.is_terminal_negative is True
        assert CTGovStatus.TERMINATED.is_terminal_negative is True
        assert CTGovStatus.SUSPENDED.is_terminal_negative is True
        assert CTGovStatus.RECRUITING.is_terminal_negative is False

    def test_terminal_positive_property(self):
        """Terminal positive statuses should be identified."""
        assert CTGovStatus.COMPLETED.is_terminal_positive is True
        assert CTGovStatus.APPROVED_FOR_MARKETING.is_terminal_positive is True
        assert CTGovStatus.RECRUITING.is_terminal_positive is False

    def test_is_active_property(self):
        """Active statuses should be identified."""
        assert CTGovStatus.RECRUITING.is_active is True
        assert CTGovStatus.ACTIVE_NOT_RECRUITING.is_active is True
        assert CTGovStatus.NOT_YET_RECRUITING.is_active is True
        assert CTGovStatus.COMPLETED.is_active is False
        assert CTGovStatus.TERMINATED.is_active is False


class TestCompletionType:
    """Tests for CompletionType enum."""

    def test_standard_types(self):
        """Standard completion types should be parsed."""
        assert CompletionType.from_string("ACTUAL") == CompletionType.ACTUAL
        assert CompletionType.from_string("ANTICIPATED") == CompletionType.ANTICIPATED
        assert CompletionType.from_string("ESTIMATED") == CompletionType.ESTIMATED

    def test_case_insensitive(self):
        """Parsing should be case-insensitive."""
        assert CompletionType.from_string("actual") == CompletionType.ACTUAL
        assert CompletionType.from_string("Actual") == CompletionType.ACTUAL

    def test_none_returns_none(self):
        """None/empty should return None."""
        assert CompletionType.from_string(None) is None
        assert CompletionType.from_string("") is None


class TestCanonicalTrialRecord:
    """Tests for CanonicalTrialRecord dataclass."""

    def test_to_dict(self):
        """Record should serialize to dict."""
        record = CanonicalTrialRecord(
            ticker="ACME",
            nct_id="NCT12345678",
            overall_status=CTGovStatus.RECRUITING,
            last_update_posted=date(2026, 1, 10),
            primary_completion_date=date(2026, 6, 1),
            primary_completion_type=CompletionType.ANTICIPATED,
            completion_date=date(2026, 12, 1),
            completion_type=CompletionType.ESTIMATED,
            results_first_posted=None,
        )
        d = record.to_dict()

        assert d["ticker"] == "ACME"
        assert d["nct_id"] == "NCT12345678"
        assert d["overall_status"] == "RECRUITING"
        assert d["last_update_posted"] == "2026-01-10"
        assert d["primary_completion_date"] == "2026-06-01"
        assert d["primary_completion_type"] == "ANTICIPATED"
        assert d["results_first_posted"] is None

    def test_from_dict(self):
        """Record should deserialize from dict."""
        d = {
            "ticker": "ACME",
            "nct_id": "NCT12345678",
            "overall_status": "RECRUITING",
            "last_update_posted": "2026-01-10",
            "primary_completion_date": "2026-06-01",
            "primary_completion_type": "ANTICIPATED",
            "completion_date": None,
            "completion_type": None,
            "results_first_posted": None,
        }
        record = CanonicalTrialRecord.from_dict(d)

        assert record.ticker == "ACME"
        assert record.nct_id == "NCT12345678"
        assert record.overall_status == CTGovStatus.RECRUITING
        assert record.last_update_posted == date(2026, 1, 10)

    def test_compute_hash(self):
        """Hash should be deterministic."""
        record = CanonicalTrialRecord(
            ticker="ACME",
            nct_id="NCT12345678",
            overall_status=CTGovStatus.RECRUITING,
            last_update_posted=date(2026, 1, 10),
            primary_completion_date=None,
            primary_completion_type=None,
            completion_date=None,
            completion_type=None,
            results_first_posted=None,
        )
        hash1 = record.compute_hash()
        hash2 = record.compute_hash()

        assert hash1 == hash2
        assert len(hash1) == 16


class TestCTGovAdapterExtraction:
    """Tests for CTGovAdapter field extraction."""

    def test_extract_flat_record(self):
        """Should extract from flat record format."""
        adapter = CTGovAdapter()
        record = {
            "ticker": "ACME",
            "nct_id": "NCT12345678",
            "overall_status": "RECRUITING",
            "last_update_posted": "2026-01-10",
            "primary_completion_date": "2026-06-01",
        }

        canonical = adapter.extract_canonical_record(record, as_of_date=date(2026, 1, 15))

        assert canonical.ticker == "ACME"
        assert canonical.nct_id == "NCT12345678"
        assert canonical.overall_status == CTGovStatus.RECRUITING
        assert canonical.last_update_posted == date(2026, 1, 10)

    def test_extract_nested_ctgov_record(self):
        """Should extract from nested ctgov_record format."""
        adapter = CTGovAdapter()
        record = {
            "ticker": "ACME",
            "ctgov_record": {
                "protocolSection": {
                    "identificationModule": {
                        "nctId": "NCT12345678"
                    },
                    "statusModule": {
                        "overallStatus": "RECRUITING",
                        "lastUpdatePostDateStruct": {
                            "date": "2026-01-10"
                        }
                    }
                }
            }
        }

        canonical = adapter.extract_canonical_record(record, as_of_date=date(2026, 1, 15))

        assert canonical.ticker == "ACME"
        assert canonical.nct_id == "NCT12345678"
        assert canonical.overall_status == CTGovStatus.RECRUITING

    def test_ticker_uppercase(self):
        """Ticker should be normalized to uppercase."""
        adapter = CTGovAdapter()
        record = {
            "ticker": "acme",
            "nct_id": "NCT12345678",
            "last_update_posted": "2026-01-10",
        }

        canonical = adapter.extract_canonical_record(record, as_of_date=date(2026, 1, 15))

        assert canonical.ticker == "ACME"

    def test_symbol_fallback(self):
        """Should fall back to 'symbol' field for ticker."""
        adapter = CTGovAdapter()
        record = {
            "symbol": "ACME",
            "nct_id": "NCT12345678",
            "last_update_posted": "2026-01-10",
        }

        canonical = adapter.extract_canonical_record(record, as_of_date=date(2026, 1, 15))

        assert canonical.ticker == "ACME"

    def test_status_fallback(self):
        """Should fall back to 'status' field for overall_status."""
        adapter = CTGovAdapter()
        record = {
            "ticker": "ACME",
            "nct_id": "NCT12345678",
            "status": "RECRUITING",  # Not overall_status
            "last_update_posted": "2026-01-10",
        }

        canonical = adapter.extract_canonical_record(record, as_of_date=date(2026, 1, 15))

        assert canonical.overall_status == CTGovStatus.RECRUITING

    def test_missing_ticker_raises(self):
        """Missing ticker should raise MissingRequiredFieldError."""
        adapter = CTGovAdapter()
        record = {
            "nct_id": "NCT12345678",
            "last_update_posted": "2026-01-10",
        }

        with pytest.raises(MissingRequiredFieldError) as exc_info:
            adapter.extract_canonical_record(record, as_of_date=date(2026, 1, 15))

        assert "ticker" in str(exc_info.value)

    def test_missing_nct_id_raises(self):
        """Missing NCT ID should raise MissingRequiredFieldError."""
        adapter = CTGovAdapter()
        record = {
            "ticker": "ACME",
            "last_update_posted": "2026-01-10",
        }

        with pytest.raises(MissingRequiredFieldError) as exc_info:
            adapter.extract_canonical_record(record, as_of_date=date(2026, 1, 15))

        assert "nct_id" in str(exc_info.value)

    def test_missing_last_update_posted_raises(self):
        """Missing last_update_posted should raise MissingRequiredFieldError."""
        adapter = CTGovAdapter()
        record = {
            "ticker": "ACME",
            "nct_id": "NCT12345678",
        }

        with pytest.raises(MissingRequiredFieldError) as exc_info:
            adapter.extract_canonical_record(record, as_of_date=date(2026, 1, 15))

        assert "last_update_posted" in str(exc_info.value)


class TestPITValidation:
    """Tests for point-in-time validation."""

    def test_future_data_raises(self):
        """Data with last_update_posted > as_of_date should raise."""
        adapter = CTGovAdapter()
        record = {
            "ticker": "ACME",
            "nct_id": "NCT12345678",
            "last_update_posted": "2026-01-20",  # Future
        }

        with pytest.raises(FutureDataError):
            adapter.extract_canonical_record(record, as_of_date=date(2026, 1, 15))

    def test_same_day_data_ok(self):
        """Data with last_update_posted == as_of_date should be OK."""
        adapter = CTGovAdapter()
        record = {
            "ticker": "ACME",
            "nct_id": "NCT12345678",
            "last_update_posted": "2026-01-15",
        }

        canonical = adapter.extract_canonical_record(record, as_of_date=date(2026, 1, 15))

        assert canonical.last_update_posted == date(2026, 1, 15)

    def test_past_data_ok(self):
        """Data with last_update_posted < as_of_date should be OK."""
        adapter = CTGovAdapter()
        record = {
            "ticker": "ACME",
            "nct_id": "NCT12345678",
            "last_update_posted": "2026-01-10",
        }

        canonical = adapter.extract_canonical_record(record, as_of_date=date(2026, 1, 15))

        assert canonical.last_update_posted == date(2026, 1, 10)


class TestAdapterStats:
    """Tests for AdapterStats tracking."""

    def test_stats_tracked(self):
        """Stats should be tracked during extraction."""
        adapter = CTGovAdapter()

        # Successful extraction
        adapter.extract_canonical_record(
            {"ticker": "ACME", "nct_id": "NCT001", "last_update_posted": "2026-01-10"},
            as_of_date=date(2026, 1, 15)
        )

        assert adapter.stats.total_records == 1
        assert adapter.stats.successful_extractions == 1

    def test_missing_status_tracked(self):
        """Missing overall_status should be tracked."""
        adapter = CTGovAdapter()
        adapter.extract_canonical_record(
            {"ticker": "ACME", "nct_id": "NCT001", "last_update_posted": "2026-01-10"},
            as_of_date=date(2026, 1, 15)
        )

        assert adapter.stats.missing_overall_status == 1

    def test_success_rate(self):
        """Success rate should be calculated correctly."""
        stats = AdapterStats(total_records=10, successful_extractions=8)
        assert stats.success_rate == 0.8


class TestBatchProcessing:
    """Tests for batch processing."""

    def test_batch_processing(self):
        """Should process batch of records."""
        records = [
            {"ticker": "ACME", "nct_id": "NCT001", "overall_status": "RECRUITING", "last_update_posted": "2026-01-10"},
            {"ticker": "BETA", "nct_id": "NCT002", "overall_status": "COMPLETED", "last_update_posted": "2026-01-08"},
        ]

        canonical, stats = process_trial_records_batch(records, as_of_date=date(2026, 1, 15))

        assert len(canonical) == 2
        assert stats.total_records == 2
        assert stats.successful_extractions == 2

    def test_batch_skips_missing_required(self):
        """Batch should skip records missing required fields."""
        records = [
            {"ticker": "ACME", "nct_id": "NCT001", "overall_status": "RECRUITING", "last_update_posted": "2026-01-10"},
            {"ticker": "BETA", "overall_status": "RECRUITING", "last_update_posted": "2026-01-08"},  # Missing nct_id
            {"ticker": "GAMMA", "nct_id": "NCT003", "overall_status": "COMPLETED", "last_update_posted": "2026-01-09"},
        ]

        canonical, stats = process_trial_records_batch(records, as_of_date=date(2026, 1, 15))

        assert len(canonical) == 2  # ACME and GAMMA succeed, BETA skipped
        tickers = [c.ticker for c in canonical]
        assert "ACME" in tickers
        assert "GAMMA" in tickers
        assert "BETA" not in tickers

    def test_batch_filters_future_data(self):
        """Batch should filter future-dated records by default."""
        records = [
            {"ticker": "ACME", "nct_id": "NCT001", "overall_status": "RECRUITING", "last_update_posted": "2026-01-10"},
            {"ticker": "BETA", "nct_id": "NCT002", "overall_status": "COMPLETED", "last_update_posted": "2026-01-20"},  # Future
            {"ticker": "GAMMA", "nct_id": "NCT003", "overall_status": "RECRUITING", "last_update_posted": "2026-01-12"},
        ]

        config = AdapterConfig(fail_on_future_data=False)
        canonical, stats = process_trial_records_batch(
            records, as_of_date=date(2026, 1, 15), config=config
        )

        assert len(canonical) == 2  # ACME and GAMMA, BETA filtered
        assert stats.future_data_violations == 1

    def test_batch_strict_mode_raises(self):
        """Strict mode should raise on future data."""
        records = [
            {"ticker": "ACME", "nct_id": "NCT001", "last_update_posted": "2026-01-20"},  # Future
        ]

        config = AdapterConfig(fail_on_future_data=True)
        with pytest.raises(FutureDataError):
            process_trial_records_batch(records, as_of_date=date(2026, 1, 15), config=config)


class TestAdapterConfig:
    """Tests for AdapterConfig."""

    def test_default_values(self):
        """Default configuration values should be set."""
        config = AdapterConfig()
        assert config.max_missing_overall_status == 0.05
        assert config.allow_partial_dates is False
        assert config.fail_on_future_data is False
        assert config.max_future_data_ratio == 0.50

    def test_custom_config(self):
        """Custom configuration should be accepted."""
        config = AdapterConfig(
            fail_on_future_data=True,
            max_missing_overall_status=0.10,
        )
        assert config.fail_on_future_data is True
        assert config.max_missing_overall_status == 0.10
