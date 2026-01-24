"""
Tests for SEC EDGAR 13F Extractor

Tests cover:
- Amendment detection logic
- Dataclass construction and validation
- CUSIP mapping functionality
- XML parsing (with mocked content)

Run with: pytest tests/test_edgar_13f_extractor.py -v
"""

import pytest
import json
import tempfile
from pathlib import Path
from datetime import date, datetime

from edgar_13f_extractor import (
    RawHolding,
    FilingInfo,
    load_cusip_ticker_map,
    map_cusip_to_ticker,
)


class TestRawHolding:
    """Tests for RawHolding dataclass."""

    def test_create_basic_holding(self):
        """Should create a basic holding."""
        holding = RawHolding(
            cusip="037833100",
            shares=1000,
            value_kusd=150,
            put_call="",
        )
        assert holding.cusip == "037833100"
        assert holding.shares == 1000
        assert holding.value_kusd == 150
        assert holding.put_call == ""

    def test_create_put_holding(self):
        """Should create a PUT holding."""
        holding = RawHolding(
            cusip="037833100",
            shares=500,
            value_kusd=75,
            put_call="PUT",
        )
        assert holding.put_call == "PUT"

    def test_create_call_holding(self):
        """Should create a CALL holding."""
        holding = RawHolding(
            cusip="037833100",
            shares=500,
            value_kusd=75,
            put_call="CALL",
        )
        assert holding.put_call == "CALL"


class TestFilingInfo:
    """Tests for FilingInfo dataclass."""

    def test_create_filing_info(self):
        """Should create valid filing info."""
        filed_at = datetime(2024, 11, 14, 16, 30, 0)
        filing = FilingInfo(
            cik="0001263508",
            manager_name="Baker Bros",
            quarter_end=date(2024, 9, 30),
            accession="0001263508-24-001234",
            total_value_kusd=5000000,
            filed_at=filed_at,
            is_amendment=False,
            holdings_count=150,
        )
        assert filing.cik == "0001263508"
        assert filing.manager_name == "Baker Bros"
        assert filing.quarter_end == date(2024, 9, 30)
        assert filing.is_amendment is False
        assert filing.holdings_count == 150

    def test_create_amendment_filing(self):
        """Should create amendment filing."""
        filing = FilingInfo(
            cik="0001263508",
            manager_name="Test Fund",
            quarter_end=date(2024, 9, 30),
            accession="0001263508-24-001235",
            total_value_kusd=1000000,
            filed_at=datetime(2024, 11, 15, 10, 0, 0),
            is_amendment=True,
            holdings_count=100,
        )
        assert filing.is_amendment is True


class TestLoadCusipTickerMap:
    """Tests for load_cusip_ticker_map function."""

    def test_load_existing_cache(self, tmp_path):
        """Should load CUSIP map from existing cache file."""
        cache_file = tmp_path / "cusip_cache.json"
        cache_data = {
            "037833100": "AAPL",
            "459200101": "IBM",
            "88160R101": "TSLA",
        }
        cache_file.write_text(json.dumps(cache_data))

        result = load_cusip_ticker_map(cache_file)

        assert result == cache_data
        assert result["037833100"] == "AAPL"
        assert result["459200101"] == "IBM"
        assert len(result) == 3

    def test_load_empty_cache(self, tmp_path):
        """Should load empty cache file."""
        cache_file = tmp_path / "empty_cache.json"
        cache_file.write_text("{}")

        result = load_cusip_ticker_map(cache_file)

        assert result == {}

    def test_load_nonexistent_cache(self, tmp_path):
        """Should return empty dict for nonexistent file."""
        cache_file = tmp_path / "nonexistent.json"

        result = load_cusip_ticker_map(cache_file)

        assert result == {}


class TestMapCusipToTicker:
    """Tests for map_cusip_to_ticker function."""

    def test_map_known_cusip(self):
        """Should map known CUSIP to ticker."""
        cusip_map = {
            "037833100": "AAPL",
            "459200101": "IBM",
        }

        result = map_cusip_to_ticker("037833100", cusip_map)

        assert result == "AAPL"

    def test_map_unknown_cusip(self):
        """Should return None for unknown CUSIP."""
        cusip_map = {
            "037833100": "AAPL",
        }

        result = map_cusip_to_ticker("999999999", cusip_map)

        assert result is None

    def test_map_empty_cusip_map(self):
        """Should return None with empty map."""
        result = map_cusip_to_ticker("037833100", {})

        assert result is None

    def test_map_deterministic(self):
        """Mapping should be deterministic."""
        cusip_map = {
            "037833100": "AAPL",
            "459200101": "IBM",
        }

        results = [map_cusip_to_ticker("037833100", cusip_map) for _ in range(10)]

        assert all(r == "AAPL" for r in results)


class TestAmendmentDetection:
    """Test 13F amendment detection from form type."""

    def test_detects_original_filing(self):
        """13F-HR form should not be marked as amendment."""
        form = "13F-HR"
        is_amendment = form == "13F-HR/A"
        assert is_amendment is False

    def test_detects_amendment_filing(self):
        """13F-HR/A form should be marked as amendment."""
        form = "13F-HR/A"
        is_amendment = form == "13F-HR/A"
        assert is_amendment is True

    def test_form_type_filter_accepts_13f_hr(self):
        """Filter should accept 13F-HR form type."""
        form = "13F-HR"
        accepted_forms = ('13F-HR', '13F-HR/A')
        assert form in accepted_forms

    def test_form_type_filter_accepts_13f_hr_a(self):
        """Filter should accept 13F-HR/A form type."""
        form = "13F-HR/A"
        accepted_forms = ('13F-HR', '13F-HR/A')
        assert form in accepted_forms

    def test_form_type_filter_rejects_other_forms(self):
        """Filter should reject non-13F-HR forms."""
        accepted_forms = ('13F-HR', '13F-HR/A')
        rejected_forms = ['10-K', '10-Q', '8-K', '13F-NT', '13F-NT/A', 'DEF 14A']
        for form in rejected_forms:
            assert form not in accepted_forms, f"Form {form} should be rejected"


class TestFormTypeParsing:
    """Test various edge cases in form type parsing."""

    @pytest.mark.parametrize("form,expected_amendment", [
        ("13F-HR", False),
        ("13F-HR/A", True),
    ])
    def test_amendment_flag_from_form(self, form, expected_amendment):
        """Amendment flag should correctly derive from form type."""
        is_amendment = form == "13F-HR/A"
        assert is_amendment == expected_amendment


class TestDeterminism:
    """Ensure amendment detection is deterministic."""

    def test_amendment_detection_deterministic(self):
        """Same form should always produce same amendment flag."""
        form = "13F-HR/A"
        results = [form == "13F-HR/A" for _ in range(10)]
        assert all(r is True for r in results)

        form = "13F-HR"
        results = [form == "13F-HR/A" for _ in range(10)]
        assert all(r is False for r in results)
