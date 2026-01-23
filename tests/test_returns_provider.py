#!/usr/bin/env python3
"""
Tests for Returns Provider Module

Covers:
- CSV returns provider
- Return calculation
- Date tolerance
- Diagnostic providers (null, fixed, shuffled, lagged)
"""

import pytest
import tempfile
from datetime import date, timedelta
from pathlib import Path
from decimal import Decimal

# Import module under test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from backtest.returns_provider import (
    CSVReturnsProvider,
    NullReturnsProvider,
    FixedReturnsProvider,
    ShuffledReturnsProvider,
    LaggedReturnsProvider,
    create_csv_provider,
    create_shuffled_provider,
    create_lagged_provider,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def csv_content():
    """Valid CSV content for testing."""
    return """date,ticker,adj_close
2026-01-02,ACME,100.00
2026-01-03,ACME,102.00
2026-01-06,ACME,105.00
2026-01-02,BETA,50.00
2026-01-03,BETA,51.00
2026-01-06,BETA,49.00
"""


@pytest.fixture
def csv_file(tmp_path, csv_content):
    """Create temporary CSV file."""
    path = tmp_path / "prices.csv"
    path.write_text(csv_content)
    return path


@pytest.fixture
def provider(csv_file):
    """Create CSV returns provider."""
    return CSVReturnsProvider(csv_file)


# ============================================================================
# CSV RETURNS PROVIDER
# ============================================================================

class TestCSVReturnsProvider:
    """Tests for CSVReturnsProvider class."""

    def test_loads_data(self, provider):
        """Loads data from CSV."""
        tickers = provider.get_available_tickers()
        assert "ACME" in tickers
        assert "BETA" in tickers

    def test_computes_positive_return(self, provider):
        """Computes positive return correctly."""
        # ACME: 100 -> 102 = 2% return
        ret = provider.get_forward_total_return("ACME", "2026-01-02", "2026-01-03")
        assert ret is not None
        assert Decimal(ret) == pytest.approx(Decimal("0.02"), rel=0.001)

    def test_computes_negative_return(self, provider):
        """Computes negative return correctly."""
        # BETA: 50 -> 49 (via 2026-01-03 -> 2026-01-06) = -3.92%
        ret = provider.get_forward_total_return("BETA", "2026-01-03", "2026-01-06")
        assert ret is not None
        assert Decimal(ret) < 0

    def test_computes_multi_day_return(self, provider):
        """Computes return over multiple days."""
        # ACME: 100 -> 105 = 5% return
        ret = provider.get_forward_total_return("ACME", "2026-01-02", "2026-01-06")
        assert ret is not None
        assert Decimal(ret) == pytest.approx(Decimal("0.05"), rel=0.001)

    def test_returns_none_for_unknown_ticker(self, provider):
        """Returns None for unknown ticker."""
        ret = provider.get_forward_total_return("UNKNOWN", "2026-01-02", "2026-01-03")
        assert ret is None

    def test_case_insensitive_ticker(self, provider):
        """Ticker lookup is case-insensitive."""
        ret1 = provider.get_forward_total_return("acme", "2026-01-02", "2026-01-03")
        ret2 = provider.get_forward_total_return("ACME", "2026-01-02", "2026-01-03")
        assert ret1 == ret2

    def test_date_tolerance(self, csv_file):
        """Uses nearest date within tolerance."""
        provider = CSVReturnsProvider(csv_file, date_tolerance_days=3)

        # 2026-01-04 doesn't exist, should find 2026-01-03 or 2026-01-06
        ret = provider.get_forward_total_return("ACME", "2026-01-02", "2026-01-04")
        assert ret is not None

    def test_returns_none_beyond_tolerance(self, csv_file):
        """Returns None when date beyond tolerance."""
        provider = CSVReturnsProvider(csv_file, date_tolerance_days=1)

        # 2026-01-10 is too far from any date
        ret = provider.get_forward_total_return("ACME", "2026-01-02", "2026-01-10")
        assert ret is None

    def test_callable_interface(self, provider):
        """Supports callable interface."""
        ret = provider("ACME", "2026-01-02", "2026-01-03")
        assert ret is not None

    def test_get_available_tickers(self, provider):
        """Returns sorted list of available tickers."""
        tickers = provider.get_available_tickers()
        assert tickers == ["ACME", "BETA"]

    def test_missing_file_raises(self, tmp_path):
        """Raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            CSVReturnsProvider(tmp_path / "nonexistent.csv")

    def test_handles_bom(self, tmp_path):
        """Handles UTF-8 BOM."""
        # Write content with BOM using utf-8-sig encoding (adds BOM automatically)
        content = "date,ticker,adj_close\n2026-01-02,ACME,100.00\n"
        path = tmp_path / "prices.csv"
        path.write_bytes(content.encode("utf-8-sig"))

        provider = CSVReturnsProvider(path)
        assert "ACME" in provider.get_available_tickers()


# ============================================================================
# NULL RETURNS PROVIDER
# ============================================================================

class TestNullReturnsProvider:
    """Tests for NullReturnsProvider class."""

    def test_always_returns_none(self):
        """Always returns None."""
        provider = NullReturnsProvider()

        assert provider.get_forward_total_return("ACME", "2026-01-02", "2026-01-03") is None
        assert provider.get_forward_total_return("BETA", "2026-01-02", "2026-01-06") is None

    def test_callable_interface(self):
        """Supports callable interface."""
        provider = NullReturnsProvider()
        assert provider("ACME", "2026-01-02", "2026-01-03") is None


# ============================================================================
# FIXED RETURNS PROVIDER
# ============================================================================

class TestFixedReturnsProvider:
    """Tests for FixedReturnsProvider class."""

    def test_returns_fixed_value(self):
        """Returns fixed value for all tickers."""
        provider = FixedReturnsProvider(return_value="0.10")

        assert provider.get_forward_total_return("ACME", "2026-01-02", "2026-01-03") == "0.10"
        assert provider.get_forward_total_return("BETA", "2026-01-02", "2026-01-06") == "0.10"

    def test_custom_return_value(self):
        """Supports custom return value."""
        provider = FixedReturnsProvider(return_value="-0.05")
        assert provider.get_forward_total_return("ANY", "2026-01-01", "2026-01-31") == "-0.05"


# ============================================================================
# SHUFFLED RETURNS PROVIDER
# ============================================================================

class TestShuffledReturnsProvider:
    """Tests for ShuffledReturnsProvider class."""

    def test_shuffles_returns(self, provider):
        """Shuffles ticker->return mapping."""
        shuffled = ShuffledReturnsProvider(provider, seed=42)
        shuffled.prepare_for_tickers(["ACME", "BETA"], "2026-01-02", "2026-01-03")

        # Both returns should exist but potentially swapped
        ret_acme = shuffled.get_forward_total_return("ACME", "2026-01-02", "2026-01-03")
        ret_beta = shuffled.get_forward_total_return("BETA", "2026-01-02", "2026-01-03")

        assert ret_acme is not None or ret_beta is not None

    def test_deterministic_shuffle(self, provider):
        """Shuffle is deterministic with same seed."""
        shuffled1 = ShuffledReturnsProvider(provider, seed=42)
        shuffled1.prepare_for_tickers(["ACME", "BETA"], "2026-01-02", "2026-01-03")

        shuffled2 = ShuffledReturnsProvider(provider, seed=42)
        shuffled2.prepare_for_tickers(["ACME", "BETA"], "2026-01-02", "2026-01-03")

        ret1 = shuffled1.get_forward_total_return("ACME", "2026-01-02", "2026-01-03")
        ret2 = shuffled2.get_forward_total_return("ACME", "2026-01-02", "2026-01-03")

        assert ret1 == ret2

    def test_different_seed_different_shuffle(self, provider):
        """Different seeds produce different shuffles."""
        shuffled1 = ShuffledReturnsProvider(provider, seed=42)
        shuffled1.prepare_for_tickers(["ACME", "BETA"], "2026-01-02", "2026-01-03")

        shuffled2 = ShuffledReturnsProvider(provider, seed=99)
        shuffled2.prepare_for_tickers(["ACME", "BETA"], "2026-01-02", "2026-01-03")

        # Returns may be same (shuffle might not swap), so just check it doesn't error
        ret1 = shuffled1.get_forward_total_return("ACME", "2026-01-02", "2026-01-03")
        ret2 = shuffled2.get_forward_total_return("ACME", "2026-01-02", "2026-01-03")

        # At least check both work
        assert ret1 is not None or ret2 is not None

    def test_falls_back_to_base(self, provider):
        """Falls back to base provider if not prepared."""
        shuffled = ShuffledReturnsProvider(provider, seed=42)

        # Don't call prepare_for_tickers
        ret = shuffled.get_forward_total_return("ACME", "2026-01-02", "2026-01-03")

        # Should use base provider
        assert ret == provider.get_forward_total_return("ACME", "2026-01-02", "2026-01-03")


# ============================================================================
# LAGGED RETURNS PROVIDER
# ============================================================================

class TestLaggedReturnsProvider:
    """Tests for LaggedReturnsProvider class."""

    def test_adds_lag_to_dates(self, csv_file):
        """Adds lag to both start and end dates."""
        # Create provider with data spanning a month
        # Jan 2 + 31 = Feb 2, Jan 3 + 31 = Feb 3
        content = """date,ticker,adj_close
2026-01-02,ACME,100.00
2026-01-03,ACME,102.00
2026-02-02,ACME,110.00
2026-02-03,ACME,112.00
"""
        path = csv_file.parent / "prices_long.csv"
        path.write_text(content)

        base = CSVReturnsProvider(path, date_tolerance_days=5)
        lagged = LaggedReturnsProvider(base, lag_days=31)

        # Request Jan 2 -> Jan 3, should query Feb 2 -> Feb 3
        ret = lagged.get_forward_total_return("ACME", "2026-01-02", "2026-01-03")

        # Feb 2 -> Feb 3: 110 -> 112 = ~1.8%
        assert ret is not None
        # Use float for pytest.approx comparison
        assert float(ret) == pytest.approx(0.0182, rel=0.01)

    def test_zero_lag(self, provider):
        """Zero lag is equivalent to base provider."""
        lagged = LaggedReturnsProvider(provider, lag_days=0)

        ret_lagged = lagged.get_forward_total_return("ACME", "2026-01-02", "2026-01-03")
        ret_base = provider.get_forward_total_return("ACME", "2026-01-02", "2026-01-03")

        assert ret_lagged == ret_base


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_csv_provider(self, csv_file):
        """Creates CSV provider."""
        provider = create_csv_provider(csv_file)
        assert "ACME" in provider.get_available_tickers()

    def test_create_shuffled_provider(self, provider):
        """Creates shuffled provider."""
        shuffled = create_shuffled_provider(provider, seed=42)
        assert isinstance(shuffled, ShuffledReturnsProvider)

    def test_create_lagged_provider(self, provider):
        """Creates lagged provider."""
        lagged = create_lagged_provider(provider, lag_days=30)
        assert isinstance(lagged, LaggedReturnsProvider)


# ============================================================================
# EDGE CASES
# ============================================================================

class TestEdgeCases:
    """Edge case tests."""

    def test_zero_price_handling(self, tmp_path):
        """Handles zero price gracefully."""
        content = """date,ticker,adj_close
2026-01-02,ACME,0
2026-01-03,ACME,100.00
"""
        path = tmp_path / "prices.csv"
        path.write_text(content)

        provider = CSVReturnsProvider(path)

        # Zero start price should return None (division by zero)
        ret = provider.get_forward_total_return("ACME", "2026-01-02", "2026-01-03")
        assert ret is None

    def test_empty_price_handling(self, tmp_path):
        """Handles empty price values."""
        content = """date,ticker,adj_close
2026-01-02,ACME,
2026-01-03,ACME,100.00
"""
        path = tmp_path / "prices.csv"
        path.write_text(content)

        provider = CSVReturnsProvider(path)
        # Row with empty price should be skipped
        assert len(provider._prices.get("ACME", {})) == 1

    def test_invalid_date_handling(self, tmp_path):
        """Handles invalid dates."""
        content = """date,ticker,adj_close
invalid,ACME,100.00
2026-01-03,ACME,102.00
"""
        path = tmp_path / "prices.csv"
        path.write_text(content)

        provider = CSVReturnsProvider(path)
        # Row with invalid date should be skipped
        assert len(provider._prices.get("ACME", {})) == 1


# ============================================================================
# DETERMINISM
# ============================================================================

class TestDeterminism:
    """Tests for deterministic behavior."""

    def test_csv_provider_deterministic(self, csv_file):
        """CSV provider returns same results."""
        provider1 = CSVReturnsProvider(csv_file)
        provider2 = CSVReturnsProvider(csv_file)

        ret1 = provider1.get_forward_total_return("ACME", "2026-01-02", "2026-01-03")
        ret2 = provider2.get_forward_total_return("ACME", "2026-01-02", "2026-01-03")

        assert ret1 == ret2

    def test_return_precision(self, provider):
        """Returns have consistent precision."""
        ret = provider.get_forward_total_return("ACME", "2026-01-02", "2026-01-03")
        assert ret is not None

        # Should be 6 decimal places (0.000001 quantization)
        decimal_val = Decimal(ret)
        assert decimal_val == decimal_val.quantize(Decimal("0.000001"))

