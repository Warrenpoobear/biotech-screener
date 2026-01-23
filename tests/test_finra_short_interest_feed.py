#!/usr/bin/env python3
"""
Tests for FINRA Short Interest Feed

Covers:
- Business day calculations
- PIT safety validation
- Date availability checks
- Settlement date computation
- File parsing
"""

import pytest
from datetime import date, timedelta
from pathlib import Path
from decimal import Decimal
import tempfile
import os

# Import module under test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from finra_short_interest_feed import (
    business_days_between,
    add_business_days,
    estimate_dissemination_date,
    is_data_available,
    get_latest_available_settlement_date,
    parse_finra_si_file,
    compute_sha256,
    DISSEMINATION_LAG_BUSINESS_DAYS,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def as_of_date():
    """Standard test date."""
    return date(2026, 1, 15)


@pytest.fixture
def sample_si_file_content():
    """Sample FINRA SI file content (pipe-delimited)."""
    return """DATE|SYMBOL|SECURITY_NAME|MARKET|CURRENT_SHORT_POSITION|PREVIOUS_SHORT_POSITION|CHANGE|AVG_DAILY_VOLUME|DAYS_TO_COVER
20260101|ACME|Acme Corp|NASDAQ|1500000|1200000|300000|500000|3.0
20260101|BETA|Beta Inc|NYSE|2500000|2000000|500000|750000|3.33
20260101|GAMA|Gamma Ltd|NASDAQ|800000|900000|-100000|200000|4.0
"""


@pytest.fixture
def sample_si_file_csv():
    """Sample FINRA SI file content (comma-delimited)."""
    return """DATE,TICKER,NAME,MARKET,SHORTINTEREST,PREVSI,ADV,DTC
20260101,ACME,Acme Corp,NASDAQ,1500000,1200000,500000,3.0
20260101,BETA,Beta Inc,NYSE,2500000,2000000,750000,3.33
"""


@pytest.fixture
def sample_si_file(tmp_path, sample_si_file_content):
    """Create a temporary SI file for testing."""
    file_path = tmp_path / "test_si.txt"
    file_path.write_text(sample_si_file_content)
    return file_path


# ============================================================================
# BUSINESS DAY CALCULATIONS
# ============================================================================

class TestBusinessDays:
    """Tests for business day calculations."""

    def test_business_days_weekday_to_weekday(self):
        """Monday to Friday is 4 business days."""
        monday = date(2026, 1, 12)  # Monday
        friday = date(2026, 1, 16)  # Friday
        assert business_days_between(monday, friday) == 4

    def test_business_days_includes_weekends(self):
        """Monday to next Monday crosses weekend."""
        monday1 = date(2026, 1, 12)  # Monday
        monday2 = date(2026, 1, 19)  # Next Monday
        # Tue, Wed, Thu, Fri, Mon = 5 business days
        assert business_days_between(monday1, monday2) == 5

    def test_business_days_same_day(self):
        """Same day is 0 business days."""
        day = date(2026, 1, 15)
        assert business_days_between(day, day) == 0

    def test_business_days_weekend_only(self):
        """Friday to Monday is 1 business day (just Monday)."""
        friday = date(2026, 1, 16)  # Friday
        monday = date(2026, 1, 19)  # Monday
        assert business_days_between(friday, monday) == 1

    def test_add_business_days_basic(self):
        """Add 5 business days from Monday."""
        monday = date(2026, 1, 12)
        result = add_business_days(monday, 5)
        # Mon + 5 business days = Mon (Tue, Wed, Thu, Fri, Mon)
        assert result == date(2026, 1, 19)

    def test_add_business_days_crosses_weekend(self):
        """Add business days that cross a weekend."""
        thursday = date(2026, 1, 15)
        result = add_business_days(thursday, 3)
        # Thu + 3 business days = Tue (Fri, Mon, Tue)
        assert result == date(2026, 1, 20)

    def test_add_business_days_zero(self):
        """Adding 0 business days returns same date."""
        day = date(2026, 1, 15)
        result = add_business_days(day, 0)
        assert result == day

    def test_add_business_days_from_friday(self):
        """Add 1 business day from Friday gives Monday."""
        friday = date(2026, 1, 16)
        result = add_business_days(friday, 1)
        assert result == date(2026, 1, 19)  # Monday


# ============================================================================
# DISSEMINATION DATE ESTIMATION
# ============================================================================

class TestDisseminationDate:
    """Tests for dissemination date estimation."""

    def test_estimate_dissemination_date_basic(self):
        """Dissemination is ~11 business days after settlement."""
        settlement = date(2026, 1, 2)
        dissemination = estimate_dissemination_date(settlement)

        # Should be 11 business days later
        expected = add_business_days(settlement, DISSEMINATION_LAG_BUSINESS_DAYS)
        assert dissemination == expected

    def test_dissemination_date_crosses_weekend(self):
        """Dissemination calculation handles weekends."""
        settlement = date(2026, 1, 15)  # Wednesday
        dissemination = estimate_dissemination_date(settlement)

        # 11 business days from Wed crosses at least 2 weekends
        assert dissemination.weekday() < 5  # Must be a weekday
        assert (dissemination - settlement).days >= 11  # At least 11 calendar days


# ============================================================================
# PIT SAFETY
# ============================================================================

class TestPITSafety:
    """Tests for point-in-time safety enforcement."""

    def test_data_not_available_before_dissemination(self, as_of_date):
        """Data is not available before dissemination date."""
        # Settlement date whose dissemination is after as_of_date
        recent_settlement = as_of_date - timedelta(days=5)
        assert not is_data_available(recent_settlement, as_of_date)

    def test_data_available_after_dissemination(self, as_of_date):
        """Data is available after dissemination date."""
        # Settlement date far enough in the past
        old_settlement = as_of_date - timedelta(days=30)
        assert is_data_available(old_settlement, as_of_date)

    def test_data_available_on_dissemination_date(self):
        """Data is available exactly on dissemination date."""
        settlement = date(2026, 1, 1)
        dissemination = estimate_dissemination_date(settlement)
        assert is_data_available(settlement, dissemination)

    def test_pit_violation_edge_case(self):
        """Test edge case where as_of is one day before dissemination."""
        settlement = date(2026, 1, 1)
        dissemination = estimate_dissemination_date(settlement)
        day_before = dissemination - timedelta(days=1)
        assert not is_data_available(settlement, day_before)


# ============================================================================
# SETTLEMENT DATE COMPUTATION
# ============================================================================

class TestSettlementDate:
    """Tests for latest available settlement date computation."""

    def test_get_latest_settlement_returns_past_date(self, as_of_date):
        """Latest available settlement is in the past."""
        settlement = get_latest_available_settlement_date(as_of_date)
        assert settlement < as_of_date

    def test_latest_settlement_is_pit_safe(self, as_of_date):
        """Latest settlement's data is available as of as_of_date."""
        settlement = get_latest_available_settlement_date(as_of_date)
        assert is_data_available(settlement, as_of_date)

    def test_settlement_is_business_day(self, as_of_date):
        """Settlement dates should be business days."""
        settlement = get_latest_available_settlement_date(as_of_date)
        assert settlement.weekday() < 5  # Mon-Fri

    def test_settlement_for_different_dates(self):
        """Settlement dates change with as_of_date."""
        date1 = date(2026, 1, 15)
        date2 = date(2026, 2, 15)

        settlement1 = get_latest_available_settlement_date(date1)
        settlement2 = get_latest_available_settlement_date(date2)

        # Later as_of_date should give later or equal settlement
        assert settlement2 >= settlement1


# ============================================================================
# FILE PARSING
# ============================================================================

class TestFileParsing:
    """Tests for FINRA SI file parsing."""

    def test_parse_pipe_delimited_file(self, sample_si_file):
        """Parse pipe-delimited FINRA file."""
        settlement = date(2026, 1, 1)
        records = parse_finra_si_file(sample_si_file, settlement)

        assert len(records) == 3
        assert records[0]["symbol"] == "ACME"
        assert records[0]["short_interest_shares"] == 1500000
        assert records[0]["previous_si_shares"] == 1200000

    def test_parse_comma_delimited_file(self, tmp_path, sample_si_file_csv):
        """Parse comma-delimited FINRA file."""
        file_path = tmp_path / "test_si.csv"
        file_path.write_text(sample_si_file_csv)

        settlement = date(2026, 1, 1)
        records = parse_finra_si_file(file_path, settlement)

        assert len(records) == 2
        assert records[0]["symbol"] == "ACME"

    def test_parse_calculates_change_pct(self, sample_si_file):
        """Parser calculates SI change percentage."""
        settlement = date(2026, 1, 1)
        records = parse_finra_si_file(sample_si_file, settlement)

        # ACME: 1500000 from 1200000 = 25% increase
        acme = next(r for r in records if r["symbol"] == "ACME")
        assert acme["si_change_pct"] == pytest.approx(25.0, rel=0.01)

        # GAMA: 800000 from 900000 = -11.11% decrease
        gama = next(r for r in records if r["symbol"] == "GAMA")
        assert gama["si_change_pct"] == pytest.approx(-11.11, rel=0.01)

    def test_parse_includes_settlement_date(self, sample_si_file):
        """Parsed records include settlement date."""
        settlement = date(2026, 1, 1)
        records = parse_finra_si_file(sample_si_file, settlement)

        for record in records:
            assert record["settlement_date"] == settlement.isoformat()

    def test_parse_includes_source(self, sample_si_file):
        """Parsed records include source identifier."""
        settlement = date(2026, 1, 1)
        records = parse_finra_si_file(sample_si_file, settlement)

        for record in records:
            assert record["source"] == "FINRA"

    def test_parse_skips_invalid_rows(self, tmp_path):
        """Parser skips rows with invalid data."""
        content = """SYMBOL|CURRENT_SHORT_POSITION|PREVIOUS_SHORT_POSITION
VALID|1000000|800000
|500000|400000
TOOLONGTICKERXXX|300000|200000
BADNUM|notanumber|100000
"""
        file_path = tmp_path / "test_si.txt"
        file_path.write_text(content)

        settlement = date(2026, 1, 1)
        records = parse_finra_si_file(file_path, settlement)

        # Only VALID should be parsed
        assert len(records) == 1
        assert records[0]["symbol"] == "VALID"

    def test_parse_empty_file(self, tmp_path):
        """Parser handles empty file gracefully."""
        file_path = tmp_path / "empty.txt"
        file_path.write_text("")

        settlement = date(2026, 1, 1)
        records = parse_finra_si_file(file_path, settlement)

        assert records == []

    def test_parse_header_only_file(self, tmp_path):
        """Parser handles file with only header."""
        content = "SYMBOL|CURRENT_SHORT_POSITION|PREVIOUS_SHORT_POSITION"
        file_path = tmp_path / "header_only.txt"
        file_path.write_text(content)

        settlement = date(2026, 1, 1)
        records = parse_finra_si_file(file_path, settlement)

        assert records == []


# ============================================================================
# HASHING
# ============================================================================

class TestHashing:
    """Tests for deterministic hashing."""

    def test_sha256_basic(self):
        """Basic SHA256 hash computation."""
        data = b"test data"
        hash_result = compute_sha256(data)

        assert len(hash_result) == 64  # SHA256 hex is 64 chars
        assert hash_result.isalnum()

    def test_sha256_deterministic(self):
        """Same data produces same hash."""
        data = b"deterministic test data"
        hash1 = compute_sha256(data)
        hash2 = compute_sha256(data)

        assert hash1 == hash2

    def test_sha256_different_for_different_data(self):
        """Different data produces different hash."""
        data1 = b"data one"
        data2 = b"data two"

        assert compute_sha256(data1) != compute_sha256(data2)

    def test_sha256_empty_data(self):
        """Hash of empty data is deterministic."""
        hash_result = compute_sha256(b"")
        # Known SHA256 of empty string
        assert hash_result == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


# ============================================================================
# EDGE CASES
# ============================================================================

class TestEdgeCases:
    """Edge case tests."""

    def test_year_boundary_settlement(self):
        """Settlement date computation across year boundary."""
        as_of = date(2026, 1, 5)
        settlement = get_latest_available_settlement_date(as_of)

        # Settlement could be in previous year
        assert settlement.year in (2025, 2026)

    def test_february_handling(self):
        """Business day calculations handle February correctly."""
        # Non-leap year February
        feb_end = date(2026, 2, 28)
        result = add_business_days(feb_end, 1)
        assert result == date(2026, 3, 2)  # March 1 is Sunday, so March 2

    def test_large_business_day_addition(self):
        """Adding many business days works correctly."""
        start = date(2026, 1, 1)
        result = add_business_days(start, 252)  # ~1 trading year

        # Should be about 1 calendar year later
        assert 350 <= (result - start).days <= 370


# ============================================================================
# DETERMINISM
# ============================================================================

class TestDeterminism:
    """Tests for deterministic behavior."""

    def test_settlement_date_deterministic(self, as_of_date):
        """Same as_of_date always produces same settlement date."""
        settlement1 = get_latest_available_settlement_date(as_of_date)
        settlement2 = get_latest_available_settlement_date(as_of_date)

        assert settlement1 == settlement2

    def test_parsing_deterministic(self, sample_si_file):
        """Parsing same file produces identical results."""
        settlement = date(2026, 1, 1)

        records1 = parse_finra_si_file(sample_si_file, settlement)
        records2 = parse_finra_si_file(sample_si_file, settlement)

        assert records1 == records2
