#!/usr/bin/env python3
"""
Tests for module_1_universe.py

Module 1 handles universe filtering and status classification.
These tests cover:
- Market cap filtering (MIN_MARKET_CAP_MM = 50M)
- Shell company detection
- SPAC/blank check exclusion
- Status gate classification (ACTIVE, EXCLUDED_*)
- Various data structure formats
"""

import pytest
from decimal import Decimal
from datetime import date

from module_1_universe import (
    compute_module_1_universe,
    _extract_market_cap_mm,
    _classify_status,
    MIN_MARKET_CAP_MM,
    SHELL_KEYWORDS,
)
from common.types import StatusGate


class TestExtractMarketCapMm:
    """Tests for _extract_market_cap_mm helper function."""

    def test_direct_market_cap_mm_field(self):
        """Extract from direct market_cap_mm field."""
        record = {"market_cap_mm": 100}
        result = _extract_market_cap_mm(record)
        assert result == Decimal("100")

    def test_direct_market_cap_mm_string(self):
        """Handle string market_cap_mm."""
        record = {"market_cap_mm": "250.5"}
        result = _extract_market_cap_mm(record)
        assert result == Decimal("250.5")

    def test_direct_market_cap_mm_decimal(self):
        """Handle Decimal market_cap_mm."""
        record = {"market_cap_mm": Decimal("500")}
        result = _extract_market_cap_mm(record)
        assert result == Decimal("500")

    def test_nested_market_data_market_cap(self):
        """Extract from nested market_data.market_cap (raw $ to millions)."""
        record = {"market_data": {"market_cap": 5_000_000_000}}
        result = _extract_market_cap_mm(record)
        assert result == Decimal("5000")

    def test_flat_market_cap_raw(self):
        """Extract from flat market_cap (raw $ to millions)."""
        record = {"market_cap": 2_500_000_000}
        result = _extract_market_cap_mm(record)
        assert result == Decimal("2500")

    def test_priority_market_cap_mm_over_raw(self):
        """market_cap_mm should take priority over raw market_cap."""
        record = {
            "market_cap_mm": 100,  # Should use this
            "market_cap": 5_000_000_000,  # Not this
        }
        result = _extract_market_cap_mm(record)
        assert result == Decimal("100")

    def test_missing_market_cap_returns_none(self):
        """Missing market cap should return None."""
        record = {"ticker": "ACME"}
        result = _extract_market_cap_mm(record)
        assert result is None

    def test_invalid_market_cap_returns_none(self):
        """Invalid market cap value should return None."""
        record = {"market_cap_mm": "not-a-number"}
        result = _extract_market_cap_mm(record)
        assert result is None

    def test_empty_market_data_dict(self):
        """Empty market_data dict should fall through to other options."""
        record = {"market_data": {}, "market_cap": 1_000_000_000}
        result = _extract_market_cap_mm(record)
        assert result == Decimal("1000")

    def test_market_data_not_dict(self):
        """Non-dict market_data should be handled gracefully."""
        record = {"market_data": "invalid", "market_cap": 1_000_000_000}
        result = _extract_market_cap_mm(record)
        assert result == Decimal("1000")


class TestClassifyStatus:
    """Tests for _classify_status helper function."""

    def test_active_with_valid_market_cap(self):
        """Valid record should be classified as ACTIVE."""
        record = {"market_cap_mm": 100}
        status, reason = _classify_status(record)
        assert status == StatusGate.ACTIVE
        assert reason is None

    def test_delisted_status(self):
        """Delisted status should be EXCLUDED_DELISTED."""
        record = {"status": "delisted", "market_cap_mm": 100}
        status, reason = _classify_status(record)
        assert status == StatusGate.EXCLUDED_DELISTED
        assert "delisted" in reason

    def test_delisted_status_short(self):
        """Short delisted status 'd' should be EXCLUDED_DELISTED."""
        record = {"status": "d", "market_cap_mm": 100}
        status, reason = _classify_status(record)
        assert status == StatusGate.EXCLUDED_DELISTED

    def test_acquired_status(self):
        """Acquired status should be EXCLUDED_ACQUIRED."""
        record = {"status": "acquired", "market_cap_mm": 100}
        status, reason = _classify_status(record)
        assert status == StatusGate.EXCLUDED_ACQUIRED

    def test_ma_status(self):
        """M&A status should be EXCLUDED_ACQUIRED."""
        record = {"status": "m&a", "market_cap_mm": 100}
        status, reason = _classify_status(record)
        assert status == StatusGate.EXCLUDED_ACQUIRED

    def test_shell_company_acquisition_corp(self):
        """Shell company with 'acquisition corp' should be EXCLUDED_SHELL."""
        record = {
            "company_name": "XYZ Acquisition Corp",
            "market_cap_mm": 500,
        }
        status, reason = _classify_status(record)
        assert status == StatusGate.EXCLUDED_SHELL
        assert "acquisition corp" in reason

    def test_shell_company_spac(self):
        """Shell company with 'SPAC' should be EXCLUDED_SHELL."""
        record = {
            "company_name": "Some SPAC Holdings",
            "market_cap_mm": 300,
        }
        status, reason = _classify_status(record)
        assert status == StatusGate.EXCLUDED_SHELL
        assert "spac" in reason

    def test_shell_company_blank_check(self):
        """Shell company with 'blank check' should be EXCLUDED_SHELL."""
        record = {
            "company_name": "Blank Check Company Inc",
            "market_cap_mm": 200,
        }
        status, reason = _classify_status(record)
        assert status == StatusGate.EXCLUDED_SHELL
        assert "blank check" in reason

    def test_shell_company_nested_name(self):
        """Shell company name in nested market_data should be detected."""
        record = {
            "market_data": {
                "company_name": "Special Purpose Acquisition Corp",
                "market_cap": 500_000_000,
            },
        }
        status, reason = _classify_status(record)
        assert status == StatusGate.EXCLUDED_SHELL

    def test_missing_market_cap(self):
        """Missing market cap should be EXCLUDED_MISSING_DATA."""
        record = {"company_name": "Acme Pharma"}
        status, reason = _classify_status(record)
        assert status == StatusGate.EXCLUDED_MISSING_DATA
        assert "missing_market_cap" in reason

    def test_small_cap_below_threshold(self):
        """Market cap below threshold should be EXCLUDED_SMALL_CAP."""
        record = {"market_cap_mm": 40}  # Below MIN_MARKET_CAP_MM of 50
        status, reason = _classify_status(record)
        assert status == StatusGate.EXCLUDED_SMALL_CAP
        assert "40" in reason
        assert "50" in reason

    def test_small_cap_at_boundary(self):
        """Market cap exactly at threshold should be ACTIVE."""
        record = {"market_cap_mm": 50}
        status, reason = _classify_status(record)
        assert status == StatusGate.ACTIVE

    def test_small_cap_just_below(self):
        """Market cap just below threshold should be EXCLUDED."""
        record = {"market_cap_mm": Decimal("49.99")}
        status, reason = _classify_status(record)
        assert status == StatusGate.EXCLUDED_SMALL_CAP

    def test_case_insensitive_status(self):
        """Status check should be case-insensitive."""
        record = {"status": "DELISTED", "market_cap_mm": 100}
        status, reason = _classify_status(record)
        assert status == StatusGate.EXCLUDED_DELISTED

    def test_case_insensitive_shell_keywords(self):
        """Shell keyword check should be case-insensitive."""
        record = {"company_name": "ABC SPAC HOLDINGS", "market_cap_mm": 100}
        status, reason = _classify_status(record)
        assert status == StatusGate.EXCLUDED_SHELL


class TestComputeModule1Universe:
    """Tests for compute_module_1_universe main function."""

    def test_basic_active_securities(self):
        """Basic active securities should be returned."""
        raw_records = [
            {"ticker": "ACME", "company_name": "Acme Pharma", "market_cap_mm": 100},
            {"ticker": "BETA", "company_name": "Beta Bio", "market_cap_mm": 200},
        ]
        result = compute_module_1_universe(raw_records, "2026-01-15")

        assert len(result["active_securities"]) == 2
        assert len(result["excluded_securities"]) == 0
        assert result["diagnostic_counts"]["active"] == 2
        assert result["diagnostic_counts"]["excluded"] == 0

    def test_filters_small_cap(self):
        """Small cap securities should be excluded."""
        raw_records = [
            {"ticker": "LARGE", "market_cap_mm": 100},
            {"ticker": "SMALL", "market_cap_mm": 30},
        ]
        result = compute_module_1_universe(raw_records, "2026-01-15")

        assert len(result["active_securities"]) == 1
        assert result["active_securities"][0]["ticker"] == "LARGE"
        assert len(result["excluded_securities"]) == 1
        assert result["excluded_securities"][0]["ticker"] == "SMALL"
        assert result["excluded_securities"][0]["reason"] == "excluded_small_cap"

    def test_filters_shell_companies(self):
        """Shell companies should be excluded."""
        raw_records = [
            {"ticker": "GOOD", "company_name": "Good Pharma", "market_cap_mm": 100},
            {"ticker": "SPAC", "company_name": "XYZ SPAC Corp", "market_cap_mm": 300},
        ]
        result = compute_module_1_universe(raw_records, "2026-01-15")

        assert len(result["active_securities"]) == 1
        assert result["active_securities"][0]["ticker"] == "GOOD"
        assert len(result["excluded_securities"]) == 1
        assert result["excluded_securities"][0]["reason"] == "excluded_shell"

    def test_filters_delisted(self):
        """Delisted securities should be excluded."""
        raw_records = [
            {"ticker": "ACTIVE", "market_cap_mm": 100},
            {"ticker": "GONE", "market_cap_mm": 100, "status": "delisted"},
        ]
        result = compute_module_1_universe(raw_records, "2026-01-15")

        assert len(result["active_securities"]) == 1
        assert len(result["excluded_securities"]) == 1
        assert result["excluded_securities"][0]["reason"] == "excluded_delisted"

    def test_filters_acquired(self):
        """Acquired securities should be excluded."""
        raw_records = [
            {"ticker": "ACTIVE", "market_cap_mm": 100},
            {"ticker": "BOUGHT", "market_cap_mm": 100, "status": "acquired"},
        ]
        result = compute_module_1_universe(raw_records, "2026-01-15")

        assert len(result["active_securities"]) == 1
        assert len(result["excluded_securities"]) == 1
        assert result["excluded_securities"][0]["reason"] == "excluded_acquired"

    def test_filters_missing_market_cap(self):
        """Securities with missing market cap should be excluded."""
        raw_records = [
            {"ticker": "GOOD", "market_cap_mm": 100},
            {"ticker": "NOMCAP", "company_name": "No Market Cap Inc"},
        ]
        result = compute_module_1_universe(raw_records, "2026-01-15")

        assert len(result["active_securities"]) == 1
        assert len(result["excluded_securities"]) == 1
        assert result["excluded_securities"][0]["reason"] == "excluded_missing_data"

    def test_universe_whitelist(self):
        """Universe whitelist should filter to specified tickers."""
        raw_records = [
            {"ticker": "ACME", "market_cap_mm": 100},
            {"ticker": "BETA", "market_cap_mm": 200},
            {"ticker": "GAMMA", "market_cap_mm": 150},
        ]
        result = compute_module_1_universe(
            raw_records,
            "2026-01-15",
            universe_tickers=["ACME", "GAMMA"],
        )

        assert len(result["active_securities"]) == 2
        tickers = [s["ticker"] for s in result["active_securities"]]
        assert "ACME" in tickers
        assert "GAMMA" in tickers
        assert "BETA" not in tickers

    def test_empty_input(self):
        """Empty input should return empty results."""
        result = compute_module_1_universe([], "2026-01-15")

        assert result["active_securities"] == []
        assert result["excluded_securities"] == []
        assert result["diagnostic_counts"]["total_input"] == 0

    def test_records_without_ticker_skipped(self):
        """Records without ticker should be skipped."""
        raw_records = [
            {"ticker": "ACME", "market_cap_mm": 100},
            {"company_name": "No Ticker Inc", "market_cap_mm": 200},  # No ticker
            {"ticker": "", "market_cap_mm": 200},  # Empty ticker
        ]
        result = compute_module_1_universe(raw_records, "2026-01-15")

        assert len(result["active_securities"]) == 1
        assert result["active_securities"][0]["ticker"] == "ACME"

    def test_ticker_normalized_to_uppercase(self):
        """Tickers should be normalized to uppercase."""
        raw_records = [
            {"ticker": "acme", "market_cap_mm": 100},
            {"ticker": "Beta", "market_cap_mm": 200},
        ]
        result = compute_module_1_universe(raw_records, "2026-01-15")

        assert result["active_securities"][0]["ticker"] == "ACME"
        assert result["active_securities"][1]["ticker"] == "BETA"

    def test_output_sorted_by_ticker(self):
        """Output should be sorted by ticker."""
        raw_records = [
            {"ticker": "ZETA", "market_cap_mm": 100},
            {"ticker": "ALPHA", "market_cap_mm": 200},
            {"ticker": "BETA", "market_cap_mm": 150},
        ]
        result = compute_module_1_universe(raw_records, "2026-01-15")

        tickers = [s["ticker"] for s in result["active_securities"]]
        assert tickers == sorted(tickers)

    def test_diagnostic_counts_by_reason(self):
        """Diagnostic counts should track exclusions by reason."""
        raw_records = [
            {"ticker": "GOOD", "market_cap_mm": 100},
            {"ticker": "SMALL1", "market_cap_mm": 10},
            {"ticker": "SMALL2", "market_cap_mm": 20},
            {"ticker": "DEAD", "market_cap_mm": 100, "status": "delisted"},
            {"ticker": "SPAC1", "company_name": "SPAC Holdings", "market_cap_mm": 200},
        ]
        result = compute_module_1_universe(raw_records, "2026-01-15")

        counts = result["diagnostic_counts"]["excluded_by_reason"]
        assert counts.get("excluded_small_cap", 0) == 2
        assert counts.get("excluded_delisted", 0) == 1
        assert counts.get("excluded_shell", 0) == 1

    def test_provenance_included(self):
        """Provenance should be included in output."""
        raw_records = [{"ticker": "ACME", "market_cap_mm": 100}]
        result = compute_module_1_universe(raw_records, "2026-01-15")

        assert "provenance" in result
        assert result["provenance"] is not None

    def test_as_of_date_in_output(self):
        """as_of_date should be preserved in output."""
        raw_records = [{"ticker": "ACME", "market_cap_mm": 100}]
        result = compute_module_1_universe(raw_records, "2026-01-15")

        assert result["as_of_date"] == "2026-01-15"

    def test_active_security_fields(self):
        """Active securities should have expected fields."""
        raw_records = [
            {
                "ticker": "ACME",
                "company_name": "Acme Pharma",
                "market_cap_mm": 100,
            }
        ]
        result = compute_module_1_universe(raw_records, "2026-01-15")

        sec = result["active_securities"][0]
        assert sec["ticker"] == "ACME"
        assert sec["status"] == "active"
        assert sec["market_cap_mm"] == "100"
        assert sec["company_name"] == "Acme Pharma"

    def test_excluded_security_fields(self):
        """Excluded securities should have expected fields."""
        raw_records = [{"ticker": "SMALL", "market_cap_mm": 10}]
        result = compute_module_1_universe(raw_records, "2026-01-15")

        sec = result["excluded_securities"][0]
        assert sec["ticker"] == "SMALL"
        assert sec["reason"] == "excluded_small_cap"
        assert "reason_detail" in sec

    def test_determinism(self):
        """Same inputs should produce same outputs."""
        raw_records = [
            {"ticker": "ACME", "market_cap_mm": 100},
            {"ticker": "BETA", "market_cap_mm": 200},
        ]

        result1 = compute_module_1_universe(raw_records, "2026-01-15")
        result2 = compute_module_1_universe(raw_records, "2026-01-15")

        # Compare key outputs (excluding provenance timestamps)
        assert result1["active_securities"] == result2["active_securities"]
        assert result1["excluded_securities"] == result2["excluded_securities"]
        assert result1["diagnostic_counts"] == result2["diagnostic_counts"]


class TestMinMarketCapThreshold:
    """Tests specifically for market cap threshold constant."""

    def test_min_market_cap_is_50(self):
        """MIN_MARKET_CAP_MM should be 50."""
        assert MIN_MARKET_CAP_MM == Decimal("50")

    def test_boundary_at_49_99(self):
        """49.99 should be excluded."""
        raw_records = [{"ticker": "TEST", "market_cap_mm": Decimal("49.99")}]
        result = compute_module_1_universe(raw_records, "2026-01-15")
        assert len(result["excluded_securities"]) == 1

    def test_boundary_at_50_00(self):
        """50.00 should be active."""
        raw_records = [{"ticker": "TEST", "market_cap_mm": Decimal("50.00")}]
        result = compute_module_1_universe(raw_records, "2026-01-15")
        assert len(result["active_securities"]) == 1

    def test_boundary_at_50_01(self):
        """50.01 should be active."""
        raw_records = [{"ticker": "TEST", "market_cap_mm": Decimal("50.01")}]
        result = compute_module_1_universe(raw_records, "2026-01-15")
        assert len(result["active_securities"]) == 1


class TestShellKeywords:
    """Tests for shell company keyword detection."""

    def test_shell_keywords_frozen(self):
        """SHELL_KEYWORDS should be a frozenset."""
        assert isinstance(SHELL_KEYWORDS, frozenset)

    def test_expected_keywords_present(self):
        """Expected keywords should be in SHELL_KEYWORDS."""
        assert "acquisition corp" in SHELL_KEYWORDS
        assert "spac" in SHELL_KEYWORDS
        assert "blank check" in SHELL_KEYWORDS
        assert "shell company" in SHELL_KEYWORDS

    @pytest.mark.parametrize("keyword", list(SHELL_KEYWORDS))
    def test_each_keyword_triggers_exclusion(self, keyword):
        """Each shell keyword should trigger exclusion."""
        raw_records = [
            {
                "ticker": "TEST",
                "company_name": f"ABC {keyword.upper()} XYZ",
                "market_cap_mm": 100,
            }
        ]
        result = compute_module_1_universe(raw_records, "2026-01-15")
        assert len(result["excluded_securities"]) == 1
        assert result["excluded_securities"][0]["reason"] == "excluded_shell"
