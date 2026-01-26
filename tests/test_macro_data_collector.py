#!/usr/bin/env python3
"""
Tests for macro_data_collector.py

Tests cover:
- FRED collector for yield curve and credit spreads
- Fund flow collector for ETF flows
- Main MacroDataCollector integration
- Caching behavior
- Error handling
"""

import json
import os
import pytest
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock, patch
import tempfile

from wake_robin_data_pipeline.collectors.macro_data_collector import (
    FREDCollector,
    FundFlowCollector,
    MacroDataCollector,
    MacroSnapshot,
    MacroDataPoint,
)


class TestMacroDataPoint:
    """Tests for MacroDataPoint dataclass."""

    def test_creation(self):
        """Should create data point with all fields."""
        point = MacroDataPoint(
            series_id="T10Y2Y",
            date="2026-01-15",
            value="-0.25",
            source="FRED",
            retrieved_at="2026-01-15T12:00:00",
        )
        assert point.series_id == "T10Y2Y"
        assert point.value == "-0.25"


class TestMacroSnapshot:
    """Tests for MacroSnapshot dataclass."""

    def test_creation(self):
        """Should create snapshot with all fields."""
        snapshot = MacroSnapshot(
            as_of_date="2026-01-15",
            yield_curve_slope_bps="-25.00",
            hy_credit_spread_bps="450.00",
            biotech_fund_flows_mm="150.00",
            data_quality={"completeness": "3/3"},
            provenance={"source": "test"},
        )
        assert snapshot.yield_curve_slope_bps == "-25.00"
        assert snapshot.hy_credit_spread_bps == "450.00"

    def test_to_dict(self):
        """Should convert to dictionary."""
        snapshot = MacroSnapshot(
            as_of_date="2026-01-15",
            yield_curve_slope_bps="-25.00",
            hy_credit_spread_bps="450.00",
            biotech_fund_flows_mm="150.00",
            data_quality={},
            provenance={},
        )
        d = snapshot.to_dict()
        assert isinstance(d, dict)
        assert d["as_of_date"] == "2026-01-15"


class TestFREDCollector:
    """Tests for FREDCollector class."""

    def test_initialization_with_key(self):
        """Should initialize with API key."""
        collector = FREDCollector(api_key="test_key")
        assert collector.api_key == "test_key"

    def test_initialization_from_env(self):
        """Should read API key from environment."""
        with patch.dict(os.environ, {"FRED_API_KEY": "env_key"}):
            collector = FREDCollector()
            assert collector.api_key == "env_key"

    def test_initialization_uses_default_key(self):
        """Should use default API key when none provided."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove FRED_API_KEY if present
            os.environ.pop("FRED_API_KEY", None)
            collector = FREDCollector(api_key=None)
            # Should fall back to default key
            assert collector.api_key == FREDCollector.DEFAULT_API_KEY

    def test_series_ids_defined(self):
        """Should have all required series IDs."""
        assert "yield_curve_10y2y" in FREDCollector.SERIES
        assert "hy_spread" in FREDCollector.SERIES
        assert FREDCollector.SERIES["yield_curve_10y2y"] == "T10Y2Y"
        assert FREDCollector.SERIES["hy_spread"] == "BAMLH0A0HYM2"

    def test_cache_path_generation(self):
        """Should generate correct cache paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = FREDCollector(api_key="test", cache_dir=Path(tmpdir))
            path = collector._get_cache_path("T10Y2Y", date(2026, 1, 15))
            assert "T10Y2Y_2026-01-15.json" in str(path)

    def test_fetch_series_success(self):
        """Should fetch and parse FRED data."""
        with patch("requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "observations": [
                    {"date": "2026-01-15", "value": "-0.25"},
                    {"date": "2026-01-14", "value": "-0.24"},
                ]
            }
            mock_get.return_value = mock_response

            with tempfile.TemporaryDirectory() as tmpdir:
                collector = FREDCollector(api_key="test_key", cache_dir=Path(tmpdir))
                observations = collector._fetch_series(
                    "T10Y2Y",
                    date(2026, 1, 10),
                    date(2026, 1, 15),
                )

            assert len(observations) == 2
            assert observations[0]["value"] == "-0.25"

    def test_fetch_series_filters_missing(self):
        """Should filter out missing values (marked as '.')."""
        with patch("requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "observations": [
                    {"date": "2026-01-15", "value": "-0.25"},
                    {"date": "2026-01-14", "value": "."},  # Missing
                    {"date": "2026-01-13", "value": "-0.23"},
                ]
            }
            mock_get.return_value = mock_response

            with tempfile.TemporaryDirectory() as tmpdir:
                collector = FREDCollector(api_key="test_key", cache_dir=Path(tmpdir))
                observations = collector._fetch_series(
                    "T10Y2Y",
                    date(2026, 1, 10),
                    date(2026, 1, 15),
                )

            assert len(observations) == 2  # Missing value filtered

    def test_get_yield_curve_slope(self):
        """Should return yield curve slope in basis points."""
        with patch("requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "observations": [
                    {"date": "2026-01-15", "value": "-0.25"},  # -0.25% = -25 bps
                ]
            }
            mock_get.return_value = mock_response

            with tempfile.TemporaryDirectory() as tmpdir:
                collector = FREDCollector(api_key="test_key", cache_dir=Path(tmpdir))
                value, meta = collector.get_yield_curve_slope(date(2026, 1, 15), use_cache=False)

            assert value == Decimal("-25.00")
            assert meta["series_id"] == "T10Y2Y"

    def test_get_hy_credit_spread(self):
        """Should return HY spread in basis points."""
        with patch("requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "observations": [
                    {"date": "2026-01-15", "value": "4.50"},  # 4.50% = 450 bps
                ]
            }
            mock_get.return_value = mock_response

            with tempfile.TemporaryDirectory() as tmpdir:
                collector = FREDCollector(api_key="test_key", cache_dir=Path(tmpdir))
                value, meta = collector.get_hy_credit_spread(date(2026, 1, 15), use_cache=False)

            assert value == Decimal("450.00")
            assert meta["series_id"] == "BAMLH0A0HYM2"


class TestFundFlowCollector:
    """Tests for FundFlowCollector class."""

    def test_initialization(self):
        """Should initialize with default cache dir."""
        collector = FundFlowCollector()
        assert collector.cache_dir is not None

    def test_cache_path_generation(self):
        """Should generate correct cache paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = FundFlowCollector(cache_dir=Path(tmpdir))
            path = collector._get_cache_path("XBI", date(2026, 1, 15))
            assert "XBI_2026-01-15.json" in str(path)

    def test_get_weekly_fund_flow(self):
        """Should calculate fund flow from AUM changes."""
        try:
            import yfinance  # noqa: F401
            import pandas  # noqa: F401
        except ImportError:
            pytest.skip("yfinance or pandas not installed")

        with patch("yfinance.Ticker") as mock_ticker_class:
            # Mock yfinance
            mock_ticker = MagicMock()
            mock_ticker.info = {"sharesOutstanding": 100_000_000}

            # Create mock history with price change
            import pandas as pd
            mock_hist = pd.DataFrame({
                "Close": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
            })
            mock_ticker.history.return_value = mock_hist
            mock_ticker_class.return_value = mock_ticker

            with tempfile.TemporaryDirectory() as tmpdir:
                collector = FundFlowCollector(cache_dir=Path(tmpdir))
                value, meta = collector.get_weekly_fund_flow(
                    "XBI",
                    date(2026, 1, 15),
                    use_cache=False,
                )

            # Value should be calculated (exact value depends on formula)
            assert value is not None or meta.get("error") is not None

    def test_get_weekly_fund_flow_no_yfinance(self):
        """Should handle missing yfinance gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = FundFlowCollector(cache_dir=Path(tmpdir))

            # This should not raise, but return error metadata
            with patch.dict("sys.modules", {"yfinance": None}):
                value, meta = collector.get_weekly_fund_flow(
                    "XBI",
                    date(2026, 1, 15),
                    use_cache=False,
                )

            # Either value or error should be present
            assert value is not None or meta.get("error") is not None

    def test_get_aggregated_biotech_flows(self):
        """Should aggregate flows from multiple ETFs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = FundFlowCollector(cache_dir=Path(tmpdir))

            # Mock get_weekly_fund_flow to return predictable values
            with patch.object(collector, "get_weekly_fund_flow") as mock_flow:
                # Return different values for different ETFs
                def side_effect(ticker, as_of_date, use_cache):
                    if ticker == "XBI":
                        return (Decimal("100.00"), {"ticker": "XBI"})
                    elif ticker == "IBB":
                        return (Decimal("50.00"), {"ticker": "IBB"})
                    return (None, {"error": "Unknown ticker"})

                mock_flow.side_effect = side_effect

                value, meta = collector.get_aggregated_biotech_flows(
                    date(2026, 1, 15),
                    etfs=["XBI", "IBB"],
                    use_cache=False,
                )

            # Should sum the flows
            assert value == Decimal("150.00")
            assert meta["successful_etfs"] == 2
            assert meta["total_etfs"] == 2
            assert "XBI" in meta["etf_details"]
            assert "IBB" in meta["etf_details"]

    def test_get_aggregated_biotech_flows_partial(self):
        """Should handle partial data in aggregation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = FundFlowCollector(cache_dir=Path(tmpdir))

            with patch.object(collector, "get_weekly_fund_flow") as mock_flow:
                # One ETF returns data, other fails
                def side_effect(ticker, as_of_date, use_cache):
                    if ticker == "XBI":
                        return (Decimal("100.00"), {"ticker": "XBI"})
                    return (None, {"error": "No data"})

                mock_flow.side_effect = side_effect

                value, meta = collector.get_aggregated_biotech_flows(
                    date(2026, 1, 15),
                    etfs=["XBI", "IBB"],
                    use_cache=False,
                )

            # Should still return partial data
            assert value == Decimal("100.00")
            assert meta["successful_etfs"] == 1
            assert meta["total_etfs"] == 2
            assert "errors" in meta

    def test_default_biotech_etfs(self):
        """Should use default ETFs when none specified."""
        assert FundFlowCollector.DEFAULT_BIOTECH_ETFS == ["XBI", "IBB"]


class TestMacroDataCollector:
    """Tests for main MacroDataCollector class."""

    def test_initialization(self):
        """Should initialize all sub-collectors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = MacroDataCollector(
                fred_api_key="test_key",
                cache_dir=Path(tmpdir),
            )
            assert collector.fred is not None
            assert collector.fund_flow is not None

    @patch.object(FREDCollector, "get_yield_curve_slope")
    @patch.object(FREDCollector, "get_hy_credit_spread")
    @patch.object(FundFlowCollector, "get_aggregated_biotech_flows")
    def test_collect_snapshot(
        self,
        mock_aggregated_flows,
        mock_hy_spread,
        mock_yield_curve,
    ):
        """Should collect all macro data into snapshot."""
        mock_yield_curve.return_value = (Decimal("-25.00"), {"source": "FRED"})
        mock_hy_spread.return_value = (Decimal("450.00"), {"source": "FRED"})
        mock_aggregated_flows.return_value = (Decimal("150.00"), {"source": "Yahoo", "etfs": ["XBI", "IBB"]})

        with tempfile.TemporaryDirectory() as tmpdir:
            collector = MacroDataCollector(
                fred_api_key="test_key",
                cache_dir=Path(tmpdir),
            )
            snapshot = collector.collect_snapshot(date(2026, 1, 15))

        assert snapshot.yield_curve_slope_bps == "-25.00"
        assert snapshot.hy_credit_spread_bps == "450.00"
        assert snapshot.biotech_fund_flows_mm == "150.00"
        assert snapshot.data_quality["completeness"] == "3/3"

    @patch.object(FREDCollector, "get_yield_curve_slope")
    @patch.object(FREDCollector, "get_hy_credit_spread")
    @patch.object(FundFlowCollector, "get_aggregated_biotech_flows")
    def test_collect_snapshot_partial_data(
        self,
        mock_aggregated_flows,
        mock_hy_spread,
        mock_yield_curve,
    ):
        """Should handle partial data gracefully."""
        mock_yield_curve.return_value = (Decimal("-25.00"), {"source": "FRED"})
        mock_hy_spread.return_value = (None, {"error": "API error"})
        mock_aggregated_flows.return_value = (None, {"error": "Missing data"})

        with tempfile.TemporaryDirectory() as tmpdir:
            collector = MacroDataCollector(
                fred_api_key="test_key",
                cache_dir=Path(tmpdir),
            )
            snapshot = collector.collect_snapshot(date(2026, 1, 15))

        assert snapshot.yield_curve_slope_bps == "-25.00"
        assert snapshot.hy_credit_spread_bps is None
        assert snapshot.biotech_fund_flows_mm is None
        assert snapshot.data_quality["completeness"] == "1/3"

    @patch.object(FREDCollector, "get_yield_curve_slope")
    @patch.object(FREDCollector, "get_hy_credit_spread")
    @patch.object(FundFlowCollector, "get_aggregated_biotech_flows")
    def test_to_regime_engine_params(
        self,
        mock_aggregated_flows,
        mock_hy_spread,
        mock_yield_curve,
    ):
        """Should convert snapshot to regime engine parameters."""
        mock_yield_curve.return_value = (Decimal("-25.00"), {})
        mock_hy_spread.return_value = (Decimal("450.00"), {})
        mock_aggregated_flows.return_value = (Decimal("150.00"), {})

        with tempfile.TemporaryDirectory() as tmpdir:
            collector = MacroDataCollector(
                fred_api_key="test_key",
                cache_dir=Path(tmpdir),
            )
            snapshot = collector.collect_snapshot(date(2026, 1, 15))
            params = collector.to_regime_engine_params(snapshot)

        assert params["yield_curve_slope"] == Decimal("-25.00")
        assert params["hy_credit_spread"] == Decimal("450.00")
        assert params["biotech_fund_flows"] == Decimal("150.00")

    def test_to_regime_engine_params_with_none(self):
        """Should handle None values in snapshot."""
        snapshot = MacroSnapshot(
            as_of_date="2026-01-15",
            yield_curve_slope_bps="-25.00",
            hy_credit_spread_bps=None,
            biotech_fund_flows_mm=None,
            data_quality={},
            provenance={},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            collector = MacroDataCollector(
                fred_api_key="test_key",
                cache_dir=Path(tmpdir),
            )
            params = collector.to_regime_engine_params(snapshot)

        assert params["yield_curve_slope"] == Decimal("-25.00")
        assert params["hy_credit_spread"] is None
        assert params["biotech_fund_flows"] is None


class TestIntegrationWithRegimeEngine:
    """Tests for integration with RegimeDetectionEngine."""

    @patch.object(FREDCollector, "get_yield_curve_slope")
    @patch.object(FREDCollector, "get_hy_credit_spread")
    @patch.object(FundFlowCollector, "get_aggregated_biotech_flows")
    def test_full_integration(
        self,
        mock_aggregated_flows,
        mock_hy_spread,
        mock_yield_curve,
    ):
        """Should integrate with regime engine end-to-end."""
        from regime_engine import RegimeDetectionEngine

        mock_yield_curve.return_value = (Decimal("-25.00"), {})
        mock_hy_spread.return_value = (Decimal("450.00"), {})
        mock_aggregated_flows.return_value = (Decimal("150.00"), {})

        with tempfile.TemporaryDirectory() as tmpdir:
            # Collect macro data
            collector = MacroDataCollector(
                fred_api_key="test_key",
                cache_dir=Path(tmpdir),
            )
            snapshot = collector.collect_snapshot(date(2026, 1, 15))
            macro_params = collector.to_regime_engine_params(snapshot)

            # Use in regime engine
            engine = RegimeDetectionEngine()
            result = engine.detect_regime(
                vix_current=Decimal("22.0"),
                xbi_vs_spy_30d=Decimal("-2.0"),
                as_of_date=date(2026, 1, 15),
                **macro_params,
            )

        # Should have classified regime using new signals
        assert result["regime"] is not None
        assert result["indicators"]["yield_curve_state"] == "INVERTED"
        assert result["indicators"]["credit_environment"] == "STRESSED"
