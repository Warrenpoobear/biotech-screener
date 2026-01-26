#!/usr/bin/env python3
"""
Tests for backtest/sharadar_provider.py

Sharadar returns provider for production-grade equity returns.
Tests cover:
- SharadarReturnsProvider initialization and from_csv
- Price lookups with tolerance
- Forward return calculation
- Delisting policy handling
- Diagnostics tracking
- PolygonReturnsProvider (basic tests)
- create_returns_provider factory
"""

import csv
import pytest
import tempfile
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

from backtest.sharadar_provider import (
    SharadarReturnsProvider,
    PolygonReturnsProvider,
    create_returns_provider,
    # Constants
    PRICE_QUANTIZE,
    RETURN_QUANTIZE,
    DATE_TOLERANCE_DAYS,
    DELISTING_POLICY_CONSERVATIVE,
    DELISTING_POLICY_PENALTY,
    DELISTING_POLICY_LAST_PRICE,
)


class TestSharadarReturnsProviderInit:
    """Tests for SharadarReturnsProvider initialization."""

    def test_stores_prices(self):
        """Should store prices dict."""
        prices = {
            "AMGN": {"2024-01-01": Decimal("100.00")},
        }
        provider = SharadarReturnsProvider(prices)

        assert "AMGN" in provider._tickers

    def test_builds_date_index(self):
        """Should build sorted date index."""
        prices = {
            "AMGN": {
                "2024-01-03": Decimal("102.00"),
                "2024-01-01": Decimal("100.00"),
                "2024-01-02": Decimal("101.00"),
            },
        }
        provider = SharadarReturnsProvider(prices)

        assert provider._date_index["AMGN"] == ["2024-01-01", "2024-01-02", "2024-01-03"]

    def test_initializes_diagnostics(self):
        """Should initialize diagnostics dict."""
        provider = SharadarReturnsProvider({})
        diagnostics = provider.get_diagnostics()

        assert "n_returns_calculated" in diagnostics
        assert "n_missing_ticker_not_in_data" in diagnostics


class TestSharadarFromCsv:
    """Tests for SharadarReturnsProvider.from_csv."""

    @pytest.fixture
    def temp_csv(self):
        """Create a temporary CSV file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["ticker", "date", "closeadj"])
            writer.writeheader()
            writer.writerow({"ticker": "AMGN", "date": "2024-01-01", "closeadj": "100.00"})
            writer.writerow({"ticker": "AMGN", "date": "2024-01-02", "closeadj": "105.00"})
            writer.writerow({"ticker": "GILD", "date": "2024-01-01", "closeadj": "80.00"})
            f.flush()
            yield f.name
        Path(f.name).unlink()

    def test_loads_prices(self, temp_csv):
        """Should load prices from CSV."""
        provider = SharadarReturnsProvider.from_csv(temp_csv)

        assert "AMGN" in provider._tickers
        assert "GILD" in provider._tickers

    def test_applies_ticker_filter(self, temp_csv):
        """Should filter to specified tickers."""
        provider = SharadarReturnsProvider.from_csv(
            temp_csv, ticker_filter=["AMGN"]
        )

        assert "AMGN" in provider._tickers
        assert "GILD" not in provider._tickers

    def test_quantizes_prices(self, temp_csv):
        """Should quantize prices to 6 decimal places."""
        provider = SharadarReturnsProvider.from_csv(temp_csv)
        price = provider.get_price("AMGN", "2024-01-01")

        # Check it's quantized (no more than 6 decimal places)
        assert str(price) == str(price.quantize(PRICE_QUANTIZE))


class TestGetAvailableTickers:
    """Tests for get_available_tickers method."""

    def test_returns_sorted_list(self):
        """Should return sorted list of tickers."""
        prices = {
            "GILD": {"2024-01-01": Decimal("80.00")},
            "AMGN": {"2024-01-01": Decimal("100.00")},
            "VRTX": {"2024-01-01": Decimal("350.00")},
        }
        provider = SharadarReturnsProvider(prices)

        tickers = provider.get_available_tickers()
        assert tickers == ["AMGN", "GILD", "VRTX"]


class TestGetDateRange:
    """Tests for get_date_range method."""

    def test_returns_range(self):
        """Should return (first_date, last_date) tuple."""
        prices = {
            "AMGN": {
                "2024-01-01": Decimal("100.00"),
                "2024-01-15": Decimal("110.00"),
                "2024-01-31": Decimal("120.00"),
            },
        }
        provider = SharadarReturnsProvider(prices)

        result = provider.get_date_range("AMGN")
        assert result == ("2024-01-01", "2024-01-31")

    def test_returns_none_for_missing_ticker(self):
        """Should return None for missing ticker."""
        provider = SharadarReturnsProvider({})
        assert provider.get_date_range("UNKNOWN") is None


class TestFindNearestPrice:
    """Tests for _find_nearest_price method."""

    def test_exact_date(self):
        """Should find exact date."""
        prices = {
            "AMGN": {"2024-01-01": Decimal("100.00")},
        }
        provider = SharadarReturnsProvider(prices)

        result = provider._find_nearest_price("AMGN", "2024-01-01")
        assert result == ("2024-01-01", Decimal("100.00"))

    def test_within_tolerance(self):
        """Should find price within tolerance."""
        prices = {
            "AMGN": {"2024-01-01": Decimal("100.00")},
        }
        provider = SharadarReturnsProvider(prices)

        # Look for Jan 3, should find Jan 1 (2 days tolerance)
        result = provider._find_nearest_price("AMGN", "2024-01-03")
        assert result == ("2024-01-01", Decimal("100.00"))

    def test_beyond_tolerance(self):
        """Should return None beyond tolerance."""
        prices = {
            "AMGN": {"2024-01-01": Decimal("100.00")},
        }
        provider = SharadarReturnsProvider(prices)

        # Look for Jan 15, should not find Jan 1 (14 days > 5 tolerance)
        result = provider._find_nearest_price("AMGN", "2024-01-15")
        assert result is None

    def test_unknown_ticker(self):
        """Should return None for unknown ticker."""
        provider = SharadarReturnsProvider({})
        assert provider._find_nearest_price("UNKNOWN", "2024-01-01") is None


class TestGetForwardTotalReturn:
    """Tests for get_forward_total_return method."""

    def test_calculates_return(self):
        """Should calculate return correctly."""
        prices = {
            "AMGN": {
                "2024-01-01": Decimal("100.00"),
                "2024-03-01": Decimal("110.00"),
            },
        }
        provider = SharadarReturnsProvider(prices)

        result = provider.get_forward_total_return("AMGN", "2024-01-01", "2024-03-01")

        # Return = (110 / 100) - 1 = 0.10
        assert result == Decimal("0.100000")

    def test_quantizes_return(self):
        """Should quantize return to 6 decimal places."""
        prices = {
            "AMGN": {
                "2024-01-01": Decimal("100.00"),
                "2024-03-01": Decimal("133.33"),
            },
        }
        provider = SharadarReturnsProvider(prices)

        result = provider.get_forward_total_return("AMGN", "2024-01-01", "2024-03-01")
        assert str(result) == str(result.quantize(RETURN_QUANTIZE))

    def test_tracks_diagnostics(self):
        """Should track diagnostics."""
        prices = {
            "AMGN": {
                "2024-01-01": Decimal("100.00"),
                "2024-03-01": Decimal("110.00"),
            },
        }
        provider = SharadarReturnsProvider(prices)

        provider.get_forward_total_return("AMGN", "2024-01-01", "2024-03-01")
        provider.get_forward_total_return("UNKNOWN", "2024-01-01", "2024-03-01")

        diagnostics = provider.get_diagnostics()
        assert diagnostics["n_returns_calculated"] == 1
        assert diagnostics["n_missing_ticker_not_in_data"] == 1

    def test_returns_none_for_missing_start(self):
        """Should return None for missing start price."""
        prices = {
            "AMGN": {"2024-03-01": Decimal("110.00")},
        }
        provider = SharadarReturnsProvider(prices)

        result = provider.get_forward_total_return("AMGN", "2024-01-01", "2024-03-01")
        assert result is None

    def test_returns_none_for_zero_start(self):
        """Should return None for zero start price."""
        prices = {
            "AMGN": {
                "2024-01-01": Decimal("0.00"),
                "2024-03-01": Decimal("110.00"),
            },
        }
        provider = SharadarReturnsProvider(prices)

        result = provider.get_forward_total_return("AMGN", "2024-01-01", "2024-03-01")
        assert result is None


class TestDelistingPolicies:
    """Tests for delisting policy handling."""

    def test_conservative_returns_none(self):
        """Conservative policy should return None for delisted."""
        prices = {
            "AMGN": {
                "2024-01-01": Decimal("100.00"),
                # No price after this - appears delisted
            },
        }
        provider = SharadarReturnsProvider(
            prices, delisting_policy=DELISTING_POLICY_CONSERVATIVE
        )

        # Looking for return through 2024-03-01, but data ends Jan 1
        result = provider.get_forward_total_return("AMGN", "2024-01-01", "2024-03-01")
        assert result is None


class TestCallable:
    """Tests for callable interface."""

    def test_provider_is_callable(self):
        """Provider should be callable."""
        prices = {
            "AMGN": {
                "2024-01-01": Decimal("100.00"),
                "2024-03-01": Decimal("110.00"),
            },
        }
        provider = SharadarReturnsProvider(prices)

        result = provider("AMGN", "2024-01-01", "2024-03-01")
        assert result == Decimal("0.100000")


class TestHasTicker:
    """Tests for has_ticker method."""

    def test_returns_true_for_existing(self):
        """Should return True for existing ticker."""
        prices = {"AMGN": {"2024-01-01": Decimal("100.00")}}
        provider = SharadarReturnsProvider(prices)

        assert provider.has_ticker("AMGN") is True

    def test_returns_false_for_missing(self):
        """Should return False for missing ticker."""
        prices = {"AMGN": {"2024-01-01": Decimal("100.00")}}
        provider = SharadarReturnsProvider(prices)

        assert provider.has_ticker("UNKNOWN") is False


class TestGetPrice:
    """Tests for get_price method."""

    def test_gets_price(self):
        """Should get price for date."""
        prices = {"AMGN": {"2024-01-01": Decimal("100.00")}}
        provider = SharadarReturnsProvider(prices)

        result = provider.get_price("AMGN", "2024-01-01")
        assert result == Decimal("100.00")

    def test_returns_none_for_missing(self):
        """Should return None for missing price."""
        prices = {"AMGN": {"2024-01-01": Decimal("100.00")}}
        provider = SharadarReturnsProvider(prices)

        assert provider.get_price("AMGN", "2024-06-01") is None


class TestGetDelistedDate:
    """Tests for get_delisted_date method."""

    def test_returns_none_without_as_of_date(self):
        """Should return None without as_of_date for safety."""
        prices = {
            "AMGN": {"2024-01-01": Decimal("100.00")},
        }
        provider = SharadarReturnsProvider(prices)

        result = provider.get_delisted_date("AMGN")
        assert result is None

    def test_returns_last_date_when_delisted(self):
        """Should return last date when delisted before as_of."""
        prices = {
            "AMGN": {"2024-01-01": Decimal("100.00")},
        }
        provider = SharadarReturnsProvider(prices)

        # Check as of Feb 1 - AMGN last traded Jan 1 (>10 days ago)
        as_of = date(2024, 2, 1)
        result = provider.get_delisted_date("AMGN", as_of_date=as_of)
        assert result == "2024-01-01"

    def test_returns_none_when_active(self):
        """Should return None when ticker appears active."""
        prices = {
            "AMGN": {"2024-01-25": Decimal("100.00")},
        }
        provider = SharadarReturnsProvider(prices)

        # Check as of Jan 30 - AMGN traded 5 days ago (< 10 days)
        as_of = date(2024, 1, 30)
        result = provider.get_delisted_date("AMGN", as_of_date=as_of)
        assert result is None


class TestPolygonReturnsProvider:
    """Tests for PolygonReturnsProvider class."""

    def test_stores_prices(self):
        """Should store prices."""
        prices = {"AMGN": {"2024-01-01": Decimal("100.00")}}
        provider = PolygonReturnsProvider(prices)

        assert "AMGN" in provider._tickers

    def test_get_available_tickers(self):
        """Should return available tickers."""
        prices = {
            "GILD": {"2024-01-01": Decimal("80.00")},
            "AMGN": {"2024-01-01": Decimal("100.00")},
        }
        provider = PolygonReturnsProvider(prices)

        assert provider.get_available_tickers() == ["AMGN", "GILD"]

    def test_get_forward_total_return(self):
        """Should calculate return."""
        prices = {
            "AMGN": {
                "2024-01-01": Decimal("100.00"),
                "2024-03-01": Decimal("110.00"),
            },
        }
        provider = PolygonReturnsProvider(prices)

        result = provider.get_forward_total_return("AMGN", "2024-01-01", "2024-03-01")
        assert result == Decimal("0.100000")


class TestCreateReturnsProvider:
    """Tests for create_returns_provider factory."""

    @pytest.fixture
    def temp_csv(self):
        """Create a temporary CSV file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["ticker", "date", "closeadj"])
            writer.writeheader()
            writer.writerow({"ticker": "AMGN", "date": "2024-01-01", "closeadj": "100.00"})
            f.flush()
            yield f.name
        Path(f.name).unlink()

    def test_creates_sharadar_csv_provider(self, temp_csv):
        """Should create SharadarReturnsProvider from CSV."""
        provider = create_returns_provider("sharadar_csv", filepath=temp_csv)

        assert isinstance(provider, SharadarReturnsProvider)
        assert "AMGN" in provider._tickers

    def test_raises_for_unknown_source(self):
        """Should raise for unknown source."""
        with pytest.raises(ValueError) as exc_info:
            create_returns_provider("unknown_source")

        assert "Unknown source" in str(exc_info.value)


class TestConstants:
    """Tests for module constants."""

    def test_price_quantize(self):
        """Price quantization should be 6 decimal places."""
        assert PRICE_QUANTIZE == Decimal("0.000001")

    def test_return_quantize(self):
        """Return quantization should be 6 decimal places."""
        assert RETURN_QUANTIZE == Decimal("0.000001")

    def test_date_tolerance(self):
        """Date tolerance should be 5 days."""
        assert DATE_TOLERANCE_DAYS == 5

    def test_delisting_policies(self):
        """Delisting policies should be defined."""
        assert DELISTING_POLICY_CONSERVATIVE == "conservative"
        assert DELISTING_POLICY_PENALTY == "penalty"
        assert DELISTING_POLICY_LAST_PRICE == "last_price"
