#!/usr/bin/env python3
"""
Tests for Data Readiness Preflight

Covers:
- Schema validation
- Column name normalization
- Trading calendar inference
- Ticker coverage computation
- Preflight gate logic
"""

import pytest
import tempfile
from datetime import date
from pathlib import Path
from decimal import Decimal

# Import module under test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from backtest.data_readiness import (
    normalize_column_name,
    validate_and_load_csv,
    infer_trading_calendar,
    compute_ticker_coverage,
    run_data_readiness_preflight,
    compute_next_trading_day,
    SchemaValidationError,
    REQUIRED_COLUMNS,
    COLUMN_MAPPINGS,
    MIN_COVERAGE_GATE,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def valid_csv_content():
    """Valid CSV content with standard columns."""
    return """ticker,date,closeadj
ACME,2026-01-02,100.50
ACME,2026-01-03,101.25
ACME,2026-01-06,102.00
BETA,2026-01-02,50.00
BETA,2026-01-03,50.50
BETA,2026-01-06,51.00
"""


@pytest.fixture
def csv_with_alt_columns():
    """CSV with alternative column names."""
    return """symbol,trade_date,adj_close
ACME,2026-01-02,100.50
ACME,2026-01-03,101.25
BETA,2026-01-02,50.00
"""


@pytest.fixture
def csv_with_invalid_data():
    """CSV with some invalid data."""
    return """ticker,date,closeadj
ACME,2026-01-02,100.50
ACME,invalid_date,101.25
BETA,2026-01-02,-10.00
GAMA,2026-01-02,not_a_number
DELT,2026-01-02,50.00
"""


@pytest.fixture
def csv_file(tmp_path, valid_csv_content):
    """Create a temporary CSV file."""
    path = tmp_path / "prices.csv"
    path.write_text(valid_csv_content)
    return path


@pytest.fixture
def sample_data():
    """Sample loaded data for testing."""
    return {
        "ACME": {
            "2026-01-02": Decimal("100.50"),
            "2026-01-03": Decimal("101.25"),
            "2026-01-06": Decimal("102.00"),
        },
        "BETA": {
            "2026-01-02": Decimal("50.00"),
            "2026-01-03": Decimal("50.50"),
            "2026-01-06": Decimal("51.00"),
        },
    }


# ============================================================================
# COLUMN NORMALIZATION
# ============================================================================

class TestColumnNormalization:
    """Tests for column name normalization."""

    def test_standard_columns_unchanged(self):
        """Standard column names are unchanged."""
        assert normalize_column_name("ticker") == "ticker"
        assert normalize_column_name("date") == "date"
        assert normalize_column_name("closeadj") == "closeadj"

    def test_alternate_ticker_names(self):
        """Alternate ticker column names are normalized."""
        assert normalize_column_name("symbol") == "ticker"
        assert normalize_column_name("sym") == "ticker"

    def test_alternate_date_names(self):
        """Alternate date column names are normalized."""
        assert normalize_column_name("trade_date") == "date"
        assert normalize_column_name("tradedate") == "date"

    def test_alternate_price_names(self):
        """Alternate price column names are normalized."""
        assert normalize_column_name("adj_close") == "closeadj"
        assert normalize_column_name("close_adj") == "closeadj"
        assert normalize_column_name("adjclose") == "closeadj"
        assert normalize_column_name("adjusted_close") == "closeadj"

    def test_case_insensitive(self):
        """Normalization is case-insensitive."""
        assert normalize_column_name("TICKER") == "ticker"
        assert normalize_column_name("Date") == "date"
        assert normalize_column_name("CloseAdj") == "closeadj"

    def test_strips_whitespace(self):
        """Normalization strips whitespace."""
        assert normalize_column_name("  ticker  ") == "ticker"

    def test_unknown_columns_passed_through(self):
        """Unknown columns are passed through lowercased."""
        assert normalize_column_name("volume") == "volume"
        assert normalize_column_name("OPEN") == "open"


# ============================================================================
# VALIDATE AND LOAD CSV
# ============================================================================

class TestValidateAndLoadCSV:
    """Tests for validate_and_load_csv function."""

    def test_loads_valid_csv(self, csv_file):
        """Loads valid CSV successfully."""
        data, diag = validate_and_load_csv(str(csv_file))

        assert len(data) == 2  # ACME and BETA
        assert "ACME" in data
        assert "BETA" in data
        assert diag["rows_valid"] == 6

    def test_returns_decimal_prices(self, csv_file):
        """Prices are returned as Decimal."""
        data, _ = validate_and_load_csv(str(csv_file))

        assert isinstance(data["ACME"]["2026-01-02"], Decimal)
        assert data["ACME"]["2026-01-02"] == Decimal("100.50")

    def test_handles_alternate_columns(self, tmp_path, csv_with_alt_columns):
        """Handles alternate column names."""
        path = tmp_path / "prices.csv"
        path.write_text(csv_with_alt_columns)

        data, diag = validate_and_load_csv(str(path))

        assert "ACME" in data
        assert "2026-01-02" in data["ACME"]

    def test_tracks_column_mapping(self, csv_file):
        """Tracks column name mapping."""
        _, diag = validate_and_load_csv(str(csv_file))

        assert "column_mapping" in diag

    def test_ticker_filter(self, csv_file):
        """Filters tickers when specified."""
        data, diag = validate_and_load_csv(str(csv_file), ticker_filter=["ACME"])

        assert "ACME" in data
        assert "BETA" not in data

    def test_rejects_invalid_prices(self, tmp_path, csv_with_invalid_data):
        """Rejects rows with invalid prices."""
        path = tmp_path / "prices.csv"
        path.write_text(csv_with_invalid_data)

        data, diag = validate_and_load_csv(str(path))

        assert diag["invalid_prices"] >= 2  # Negative and non-number

    def test_rejects_invalid_dates(self, tmp_path, csv_with_invalid_data):
        """Rejects rows with invalid dates."""
        path = tmp_path / "prices.csv"
        path.write_text(csv_with_invalid_data)

        data, diag = validate_and_load_csv(str(path))

        assert diag["invalid_dates"] >= 1

    def test_tracks_duplicates(self, tmp_path):
        """Tracks duplicate rows."""
        content = """ticker,date,closeadj
ACME,2026-01-02,100.00
ACME,2026-01-02,100.50
"""
        path = tmp_path / "prices.csv"
        path.write_text(content)

        data, diag = validate_and_load_csv(str(path))

        assert diag["duplicates_found"] == 1
        # Should keep last value
        assert data["ACME"]["2026-01-02"] == Decimal("100.50")

    def test_missing_file_raises(self):
        """Raises SchemaValidationError for missing file."""
        with pytest.raises(SchemaValidationError, match="not found"):
            validate_and_load_csv("/nonexistent/file.csv")

    def test_missing_columns_raises(self, tmp_path):
        """Raises SchemaValidationError for missing required columns."""
        content = """ticker,date
ACME,2026-01-02
"""
        path = tmp_path / "prices.csv"
        path.write_text(content)

        with pytest.raises(SchemaValidationError, match="Missing required columns"):
            validate_and_load_csv(str(path))


# ============================================================================
# TRADING CALENDAR
# ============================================================================

class TestTradingCalendar:
    """Tests for trading calendar functions."""

    def test_infer_calendar(self, sample_data):
        """Infers trading calendar from data."""
        calendar = infer_trading_calendar(sample_data)

        assert calendar["min_date"] == "2026-01-02"
        assert calendar["max_date"] == "2026-01-06"
        assert len(calendar["trading_days"]) == 3

    def test_empty_data_calendar(self):
        """Handles empty data."""
        calendar = infer_trading_calendar({})

        assert calendar["min_date"] is None
        assert calendar["max_date"] is None
        assert calendar["trading_days"] == []

    def test_compute_next_trading_day(self, sample_data):
        """Computes next trading day."""
        calendar = infer_trading_calendar(sample_data)

        # From 2026-01-02, next trading day is 2026-01-03
        next_day = compute_next_trading_day(calendar, "2026-01-02", offset_days=1)
        assert next_day == "2026-01-03"

    def test_next_trading_day_skips_weekend(self, sample_data):
        """Skips non-trading days."""
        calendar = infer_trading_calendar(sample_data)

        # From 2026-01-03 (Fri), next is 2026-01-06 (Mon)
        next_day = compute_next_trading_day(calendar, "2026-01-03", offset_days=1)
        assert next_day == "2026-01-06"

    def test_next_trading_day_beyond_range(self, sample_data):
        """Returns None when beyond data range."""
        calendar = infer_trading_calendar(sample_data)

        next_day = compute_next_trading_day(calendar, "2026-01-06", offset_days=1)
        assert next_day is None


# ============================================================================
# TICKER COVERAGE
# ============================================================================

class TestTickerCoverage:
    """Tests for ticker coverage computation."""

    def test_compute_coverage(self, sample_data):
        """Computes per-ticker coverage."""
        calendar = infer_trading_calendar(sample_data)
        coverage = compute_ticker_coverage(
            sample_data, calendar, "2026-01-02", "2026-01-06"
        )

        assert "ACME" in coverage
        assert "BETA" in coverage
        assert coverage["ACME"]["coverage_pct"] == 100.0
        assert coverage["BETA"]["coverage_pct"] == 100.0

    def test_partial_coverage(self):
        """Computes partial coverage correctly."""
        data = {
            "ACME": {
                "2026-01-02": Decimal("100"),
                "2026-01-03": Decimal("101"),
            },
            "BETA": {
                "2026-01-02": Decimal("50"),
            },
        }
        calendar = infer_trading_calendar(data)
        coverage = compute_ticker_coverage(data, calendar, "2026-01-02", "2026-01-03")

        assert coverage["ACME"]["coverage_pct"] == 100.0
        assert coverage["BETA"]["coverage_pct"] == 50.0


# ============================================================================
# PREFLIGHT
# ============================================================================

class TestPreflightReport:
    """Tests for run_data_readiness_preflight function."""

    def test_preflight_passes(self, csv_file):
        """Preflight passes for valid data."""
        result = run_data_readiness_preflight(str(csv_file))

        assert result["gate_passed"] is True
        assert "passed" in result["gate_reason"].lower()

    def test_preflight_structure(self, csv_file):
        """Preflight returns expected structure."""
        result = run_data_readiness_preflight(str(csv_file))

        assert "gate_passed" in result
        assert "gate_reason" in result
        assert "schema_validation" in result
        assert "date_coverage" in result
        assert "ticker_coverage" in result
        assert "missingness_breakdown" in result
        assert "trading_calendar" in result

    def test_preflight_with_filter(self, csv_file):
        """Preflight respects ticker filter."""
        result = run_data_readiness_preflight(str(csv_file), ticker_filter=["ACME"])

        assert result["ticker_coverage"]["n_tickers"] == 1

    def test_preflight_fails_missing_file(self):
        """Preflight fails for missing file."""
        result = run_data_readiness_preflight("/nonexistent.csv")

        assert result["gate_passed"] is False
        assert "Schema validation failed" in result["gate_reason"]

    def test_preflight_fails_no_data(self, tmp_path):
        """Preflight fails when no valid data."""
        content = """ticker,date,closeadj
ACME,invalid,100.00
"""
        path = tmp_path / "prices.csv"
        path.write_text(content)

        result = run_data_readiness_preflight(str(path))

        assert result["gate_passed"] is False


# ============================================================================
# EDGE CASES
# ============================================================================

class TestEdgeCases:
    """Edge case tests."""

    def test_bom_handling(self, tmp_path):
        """Handles UTF-8 BOM in CSV."""
        # Write content with BOM using utf-8-sig encoding (adds BOM automatically)
        content = "ticker,date,closeadj\nACME,2026-01-02,100.00\n"
        path = tmp_path / "prices.csv"
        path.write_bytes(content.encode("utf-8-sig"))

        data, diag = validate_and_load_csv(str(path))

        assert "ACME" in data

    def test_empty_csv(self, tmp_path):
        """Handles empty CSV (header only)."""
        content = "ticker,date,closeadj\n"
        path = tmp_path / "prices.csv"
        path.write_text(content)

        data, diag = validate_and_load_csv(str(path))

        assert data == {}
        assert diag["rows_valid"] == 0

    def test_ticker_case_normalization(self, tmp_path):
        """Ticker names are uppercased."""
        content = """ticker,date,closeadj
acme,2026-01-02,100.00
Acme,2026-01-03,101.00
"""
        path = tmp_path / "prices.csv"
        path.write_text(content)

        data, diag = validate_and_load_csv(str(path))

        assert "ACME" in data
        assert len(data) == 1  # Both rows for same ticker

    def test_whitespace_handling(self, tmp_path):
        """Handles whitespace in values."""
        content = """ticker,date,closeadj
 ACME ,2026-01-02, 100.50
"""
        path = tmp_path / "prices.csv"
        path.write_text(content)

        data, diag = validate_and_load_csv(str(path))

        assert "ACME" in data
        assert data["ACME"]["2026-01-02"] == Decimal("100.50")


# ============================================================================
# DETERMINISM
# ============================================================================

class TestDeterminism:
    """Tests for deterministic behavior."""

    def test_validate_load_deterministic(self, csv_file):
        """Loading CSV is deterministic."""
        data1, diag1 = validate_and_load_csv(str(csv_file))
        data2, diag2 = validate_and_load_csv(str(csv_file))

        assert data1 == data2
        assert diag1["rows_valid"] == diag2["rows_valid"]

    def test_calendar_deterministic(self, sample_data):
        """Calendar inference is deterministic."""
        cal1 = infer_trading_calendar(sample_data)
        cal2 = infer_trading_calendar(sample_data)

        assert cal1 == cal2

    def test_preflight_deterministic(self, csv_file):
        """Preflight is deterministic."""
        result1 = run_data_readiness_preflight(str(csv_file))
        result2 = run_data_readiness_preflight(str(csv_file))

        assert result1["gate_passed"] == result2["gate_passed"]
        assert result1["ticker_coverage"]["avg_coverage_pct"] == result2["ticker_coverage"]["avg_coverage_pct"]

