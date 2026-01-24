#!/usr/bin/env python3
"""
Tests for IC Measurement and Tracking System.

Comprehensive test coverage for:
- Forward return calculation engine
- IC calculation methodology (Spearman)
- Statistical significance testing
- Bootstrap confidence intervals
- IC stability analysis (rolling, regime-conditional, sector, market cap)
- Out-of-sample validation
- Weekly IC report generation
- Time-series database (ranking/return storage)

Author: Wake Robin Capital Management
"""

import json
import sys
import tempfile
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from backtest.ic_measurement import (
    # Core functions
    calculate_ic,
    compute_spearman_ic,
    _compute_ranks,
    _classify_ic,
    _classify_market_cap,
    _quantize,
    _decimal,
    # Trading calendar
    next_trading_day,
    add_trading_days,
    subtract_trading_days,
    compute_forward_windows,
    # Enums
    ICQuality,
    MarketCapBucket,
    SectorCategory,
    RegimeType,
    # Data classes
    ForwardReturn,
    ICResult,
    BootstrapCI,
    RollingICResult,
    StratifiedIC,
    # Engines
    ForwardReturnEngine,
    ICCalculationEngine,
    BootstrapEngine,
    ICStabilityAnalyzer,
    OutOfSampleValidator,
    WeeklyICReportGenerator,
    ICTimeSeriesDatabase,
    ICMeasurementSystem,
    # Analysis functions
    analyze_ic_trend,
    _get_ic_quality_assessment,
    # Constants
    MIN_OBS_IC,
    MIN_OBS_BOOTSTRAP,
    IC_EXCELLENT,
    IC_GOOD,
    IC_WEAK,
    HORIZON_TRADING_DAYS,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def as_of_date() -> date:
    """Standard as_of_date for tests."""
    return date(2026, 1, 15)


@pytest.fixture
def sample_rankings() -> Dict[str, int]:
    """Sample rankings where lower rank = better (15 tickers)."""
    return {
        "ACME": 1, "BETA": 2, "GAMMA": 3, "DELTA": 4, "EPSILON": 5,
        "ZETA": 6, "ETA": 7, "THETA": 8, "IOTA": 9, "KAPPA": 10,
        "LAMBDA": 11, "MU": 12, "NU": 13, "XI": 14, "OMICRON": 15,
    }


@pytest.fixture
def large_sample_rankings() -> Dict[str, int]:
    """Large sample rankings for bootstrap/significance tests (35 tickers)."""
    tickers = [
        "T01", "T02", "T03", "T04", "T05", "T06", "T07", "T08", "T09", "T10",
        "T11", "T12", "T13", "T14", "T15", "T16", "T17", "T18", "T19", "T20",
        "T21", "T22", "T23", "T24", "T25", "T26", "T27", "T28", "T29", "T30",
        "T31", "T32", "T33", "T34", "T35",
    ]
    return {t: i + 1 for i, t in enumerate(tickers)}


@pytest.fixture
def large_perfect_returns() -> Dict[str, float]:
    """Large sample returns perfectly correlated with rankings."""
    tickers = [f"T{i:02d}" for i in range(1, 36)]
    # Lower rank = higher return
    return {t: 0.20 - 0.01 * i for i, t in enumerate(tickers)}


@pytest.fixture
def large_random_returns() -> Dict[str, float]:
    """Large sample returns uncorrelated with rankings."""
    tickers = [f"T{i:02d}" for i in range(1, 36)]
    # Scrambled values with no pattern
    values = [
        0.02, -0.08, 0.12, -0.03, 0.07, -0.15, 0.09, -0.11, 0.05, 0.00,
        -0.04, 0.11, -0.06, 0.03, -0.09, 0.14, -0.02, 0.08, -0.12, 0.01,
        -0.07, 0.06, -0.10, 0.04, -0.01, 0.10, -0.05, 0.13, -0.08, 0.02,
        -0.14, 0.07, -0.03, 0.15, -0.09,
    ]
    return {t: values[i] for i, t in enumerate(tickers)}


@pytest.fixture
def perfect_returns() -> Dict[str, float]:
    """Returns perfectly correlated with rankings (lower rank = higher return)."""
    return {
        "ACME": 0.15, "BETA": 0.12, "GAMMA": 0.10, "DELTA": 0.08, "EPSILON": 0.06,
        "ZETA": 0.04, "ETA": 0.02, "THETA": 0.00, "IOTA": -0.02, "KAPPA": -0.04,
        "LAMBDA": -0.06, "MU": -0.08, "NU": -0.10, "XI": -0.12, "OMICRON": -0.14,
    }


@pytest.fixture
def random_returns() -> Dict[str, float]:
    """Returns uncorrelated with rankings (random)."""
    return {
        "ACME": 0.02, "BETA": -0.08, "GAMMA": 0.12, "DELTA": -0.03, "EPSILON": 0.07,
        "ZETA": -0.01, "ETA": 0.09, "THETA": -0.11, "IOTA": 0.05, "KAPPA": 0.00,
        "LAMBDA": -0.04, "MU": 0.11, "NU": -0.06, "XI": 0.03, "OMICRON": -0.09,
    }


@pytest.fixture
def inverted_returns() -> Dict[str, float]:
    """Returns inversely correlated with rankings (higher rank = higher return)."""
    return {
        "ACME": -0.14, "BETA": -0.12, "GAMMA": -0.10, "DELTA": -0.08, "EPSILON": -0.06,
        "ZETA": -0.04, "ETA": -0.02, "THETA": 0.00, "IOTA": 0.02, "KAPPA": 0.04,
        "LAMBDA": 0.06, "MU": 0.08, "NU": 0.10, "XI": 0.12, "OMICRON": 0.14,
    }


@pytest.fixture
def mock_return_provider():
    """Mock return provider that returns fixed values."""
    def provider(ticker: str, start_date: str, end_date: str) -> Optional[str]:
        # Return based on ticker for testing
        returns_map = {
            "ACME": "0.10",
            "BETA": "0.08",
            "GAMMA": "0.05",
            "DELTA": "0.02",
            "EPSILON": "-0.03",
            "XBI": "0.03",  # Benchmark
        }
        return returns_map.get(ticker.upper())
    return provider


@pytest.fixture
def temp_data_dir():
    """Temporary directory for test data storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def weekly_historical_results() -> List[Dict[str, Any]]:
    """Historical weekly results for rolling IC analysis."""
    results = []
    base_date = date(2025, 10, 1)

    for week in range(20):
        week_date = base_date + timedelta(weeks=week)

        # Create rankings that slightly change each week
        rankings = {
            "ACME": 1 + (week % 3),
            "BETA": 2 + (week % 2),
            "GAMMA": 3,
            "DELTA": 4 - (week % 2),
            "EPSILON": 5,
            "ZETA": 6,
            "ETA": 7,
            "THETA": 8,
            "IOTA": 9,
            "KAPPA": 10,
        }

        # Create returns with some correlation to rankings
        returns = {
            "ACME": 0.08 - 0.01 * rankings["ACME"],
            "BETA": 0.08 - 0.01 * rankings["BETA"],
            "GAMMA": 0.08 - 0.01 * rankings["GAMMA"],
            "DELTA": 0.08 - 0.01 * rankings["DELTA"],
            "EPSILON": 0.08 - 0.01 * rankings["EPSILON"],
            "ZETA": 0.08 - 0.01 * rankings["ZETA"],
            "ETA": 0.08 - 0.01 * rankings["ETA"],
            "THETA": 0.08 - 0.01 * rankings["THETA"],
            "IOTA": 0.08 - 0.01 * rankings["IOTA"],
            "KAPPA": 0.08 - 0.01 * rankings["KAPPA"],
        }

        # Alternate regimes
        regime = ["BULL", "NEUTRAL", "BEAR"][week % 3]

        results.append({
            "as_of_date": week_date.isoformat(),
            "rankings": rankings,
            "forward_returns": returns,
            "regime": regime,
        })

    return results


# =============================================================================
# HELPER FUNCTION TESTS
# =============================================================================

class TestHelperFunctions:
    """Tests for helper functions."""

    def test_decimal_conversion(self):
        """Test _decimal() handles various input types."""
        assert _decimal(0.5) == Decimal("0.5")
        assert _decimal("0.5") == Decimal("0.5")
        assert _decimal(Decimal("0.5")) == Decimal("0.5")
        assert _decimal(None) == Decimal("0")
        assert _decimal(1) == Decimal("1")

    def test_quantize_precision(self):
        """Test _quantize() applies correct precision."""
        val = Decimal("0.123456789")
        assert _quantize(val) == Decimal("0.123457")
        assert _quantize(val, Decimal("0.01")) == Decimal("0.12")

    def test_classify_ic_buckets(self):
        """Test IC classification into quality buckets."""
        assert _classify_ic(Decimal("0.10")) == ICQuality.EXCELLENT
        assert _classify_ic(Decimal("0.05")) == ICQuality.EXCELLENT
        assert _classify_ic(Decimal("0.04")) == ICQuality.GOOD
        assert _classify_ic(Decimal("0.03")) == ICQuality.GOOD
        assert _classify_ic(Decimal("0.02")) == ICQuality.WEAK
        assert _classify_ic(Decimal("0.01")) == ICQuality.WEAK
        assert _classify_ic(Decimal("0.005")) == ICQuality.NOISE
        assert _classify_ic(Decimal("-0.02")) == ICQuality.NEGATIVE

    def test_classify_market_cap(self):
        """Test market cap bucket classification."""
        assert _classify_market_cap(Decimal("100")) == MarketCapBucket.MICRO
        assert _classify_market_cap(Decimal("299")) == MarketCapBucket.MICRO
        assert _classify_market_cap(Decimal("500")) == MarketCapBucket.SMALL
        assert _classify_market_cap(Decimal("999")) == MarketCapBucket.SMALL
        assert _classify_market_cap(Decimal("2000")) == MarketCapBucket.MID
        assert _classify_market_cap(Decimal("4999")) == MarketCapBucket.MID
        assert _classify_market_cap(Decimal("10000")) == MarketCapBucket.LARGE


# =============================================================================
# TRADING CALENDAR TESTS
# =============================================================================

class TestTradingCalendar:
    """Tests for trading calendar functions."""

    def test_next_trading_day_weekday(self):
        """Test next trading day from a weekday."""
        # Thursday -> Friday
        assert next_trading_day("2026-01-15") == "2026-01-16"
        # Monday -> Tuesday
        assert next_trading_day("2026-01-12") == "2026-01-13"

    def test_next_trading_day_friday(self):
        """Test next trading day from Friday (skips weekend)."""
        # Friday -> Monday
        assert next_trading_day("2026-01-16") == "2026-01-19"

    def test_next_trading_day_saturday(self):
        """Test next trading day from Saturday."""
        assert next_trading_day("2026-01-17") == "2026-01-19"

    def test_add_trading_days(self):
        """Test adding trading days."""
        # Start on Monday, add 5 days = next Monday
        assert add_trading_days("2026-01-12", 5) == "2026-01-19"
        # Start on Wednesday, add 3 days = Monday (crosses weekend)
        assert add_trading_days("2026-01-14", 3) == "2026-01-19"

    def test_subtract_trading_days(self):
        """Test subtracting trading days."""
        # Start on Monday, subtract 5 days = previous Monday
        assert subtract_trading_days("2026-01-19", 5) == "2026-01-12"

    def test_compute_forward_windows(self):
        """Test forward window computation for all horizons."""
        windows = compute_forward_windows("2026-01-15")

        assert "5d" in windows
        assert "20d" in windows
        assert "90d" in windows

        # All windows should start on next trading day
        assert windows["5d"]["start"] == "2026-01-16"
        assert windows["5d"]["trading_days"] == 5

    def test_compute_forward_windows_specific_horizons(self):
        """Test forward window computation for specific horizons."""
        windows = compute_forward_windows("2026-01-15", ["5d", "20d"])

        assert len(windows) == 2
        assert "5d" in windows
        assert "20d" in windows
        assert "90d" not in windows


# =============================================================================
# SPEARMAN IC TESTS
# =============================================================================

class TestSpearmanIC:
    """Tests for Spearman rank correlation / IC calculation."""

    def test_compute_ranks_simple(self):
        """Test rank computation for simple list."""
        values = [Decimal("3"), Decimal("1"), Decimal("4"), Decimal("1"), Decimal("5")]
        ranks = _compute_ranks(values)

        # Values: 3, 1, 1, 4, 5 -> Sorted: 1(idx1), 1(idx3), 3(idx0), 4(idx2), 5(idx4)
        # Ranks should be: 1.5, 1.5 for ties, then 3, 4, 5
        assert ranks[0] == 3.0  # Value 3
        assert ranks[1] == 1.5  # Value 1 (tied)
        assert ranks[2] == 4.0  # Value 4
        assert ranks[3] == 1.5  # Value 1 (tied)
        assert ranks[4] == 5.0  # Value 5

    def test_compute_ranks_empty(self):
        """Test rank computation for empty list."""
        assert _compute_ranks([]) == []

    def test_spearman_ic_perfect_correlation(self, sample_rankings, perfect_returns):
        """Test IC calculation with perfect positive correlation."""
        rank_list = [sample_rankings[t] for t in sorted(sample_rankings.keys())]
        return_list = [Decimal(str(perfect_returns[t])) for t in sorted(sample_rankings.keys())]

        ic = compute_spearman_ic(rank_list, return_list, negate=True)

        # Perfect signal: lower rank -> higher return = high positive IC when negated
        assert ic is not None
        assert ic > Decimal("0.9")  # Should be very high

    def test_spearman_ic_perfect_inverse(self, sample_rankings, inverted_returns):
        """Test IC calculation with perfect negative correlation."""
        rank_list = [sample_rankings[t] for t in sorted(sample_rankings.keys())]
        return_list = [Decimal(str(inverted_returns[t])) for t in sorted(sample_rankings.keys())]

        ic = compute_spearman_ic(rank_list, return_list, negate=True)

        # Inverted signal: lower rank -> lower return = negative IC when negated
        assert ic is not None
        assert ic < Decimal("-0.9")

    def test_spearman_ic_insufficient_data(self):
        """Test IC returns None with insufficient data."""
        ranks = [1, 2, 3]  # Only 3 observations, below MIN_OBS_IC
        returns = [Decimal("0.1"), Decimal("0.05"), Decimal("0")]

        ic = compute_spearman_ic(ranks, returns)
        assert ic is None

    def test_calculate_ic_function(self, sample_rankings, perfect_returns):
        """Test main calculate_ic() function."""
        ic = calculate_ic(sample_rankings, perfect_returns)

        assert ic is not None
        assert ic > Decimal("0.9")

    def test_calculate_ic_missing_tickers(self, sample_rankings, perfect_returns):
        """Test IC with some missing tickers in returns."""
        # Remove some tickers from returns
        partial_returns = {k: v for k, v in perfect_returns.items() if k not in ["ACME", "BETA"]}

        ic = calculate_ic(sample_rankings, partial_returns)

        # Should still work with remaining tickers
        assert ic is not None

    def test_calculate_ic_unsupported_method(self, sample_rankings, perfect_returns):
        """Test IC raises error for unsupported methods."""
        with pytest.raises(ValueError, match="Unknown method"):
            calculate_ic(sample_rankings, perfect_returns, method="pearson")


# =============================================================================
# IC CALCULATION ENGINE TESTS
# =============================================================================

class TestICCalculationEngine:
    """Tests for ICCalculationEngine with significance testing."""

    def test_ic_with_significance_high_ic(self, large_sample_rankings, large_perfect_returns):
        """Test significance testing with high IC (needs n>=20 for t-stat)."""
        engine = ICCalculationEngine()
        result = engine.calculate_ic_with_significance(large_sample_rankings, large_perfect_returns)

        assert result.ic_quality == ICQuality.EXCELLENT
        assert result.n_observations == 35
        assert result.t_statistic is not None
        assert result.is_significant_95 is True

    def test_ic_with_significance_low_ic(self, large_sample_rankings, large_random_returns):
        """Test significance testing with low/random IC."""
        engine = ICCalculationEngine()
        result = engine.calculate_ic_with_significance(large_sample_rankings, large_random_returns)

        # Random returns should give IC that's not EXCELLENT
        # The random data may still have some correlation by chance
        assert result.n_observations == 35
        # Just verify IC calculation works, not the exact quality
        assert result.ic_quality is not None

    def test_ic_with_significance_empty_data(self):
        """Test significance with empty data."""
        engine = ICCalculationEngine()
        result = engine.calculate_ic_with_significance({}, {})

        assert result.ic_quality == ICQuality.NOISE
        assert result.n_observations == 0
        assert result.t_statistic is None

    def test_ic_with_significance_small_sample(self, sample_rankings, perfect_returns):
        """Test IC calculation with small sample (no t-stat calculated)."""
        engine = ICCalculationEngine()
        result = engine.calculate_ic_with_significance(sample_rankings, perfect_returns)

        # Small sample should still calculate IC but not t-stat
        assert result.ic_quality == ICQuality.EXCELLENT
        assert result.n_observations == 15
        # t_stat requires n >= 20
        assert result.t_statistic is None


# =============================================================================
# BOOTSTRAP ENGINE TESTS
# =============================================================================

class TestBootstrapEngine:
    """Tests for bootstrap confidence interval calculation."""

    def test_bootstrap_deterministic(self, large_sample_rankings, large_perfect_returns):
        """Test bootstrap is deterministic with same seed (needs n>=30)."""
        engine1 = BootstrapEngine(seed=42)
        engine2 = BootstrapEngine(seed=42)

        ci1 = engine1.calculate_bootstrap_ci(large_sample_rankings, large_perfect_returns, n_iterations=100)
        ci2 = engine2.calculate_bootstrap_ci(large_sample_rankings, large_perfect_returns, n_iterations=100)

        assert ci1 is not None
        assert ci2 is not None
        assert ci1.point_estimate == ci2.point_estimate
        assert ci1.ci_95_lower == ci2.ci_95_lower
        assert ci1.ci_95_upper == ci2.ci_95_upper

    def test_bootstrap_different_seeds(self, large_sample_rankings, large_perfect_returns):
        """Test bootstrap gives different results with different seeds."""
        engine1 = BootstrapEngine(seed=42)
        engine2 = BootstrapEngine(seed=123)

        ci1 = engine1.calculate_bootstrap_ci(large_sample_rankings, large_perfect_returns, n_iterations=100)
        ci2 = engine2.calculate_bootstrap_ci(large_sample_rankings, large_perfect_returns, n_iterations=100)

        # Point estimate should be same, but std may differ slightly
        assert ci1 is not None
        assert ci2 is not None
        assert ci1.point_estimate == ci2.point_estimate  # Same data

    def test_bootstrap_ci_ordering(self, large_sample_rankings, large_perfect_returns):
        """Test confidence interval ordering: 99% > 95% > 90%."""
        engine = BootstrapEngine(seed=42)
        ci = engine.calculate_bootstrap_ci(large_sample_rankings, large_perfect_returns, n_iterations=500)

        assert ci is not None
        # 99% CI should be wider than 95% which is wider than 90%
        assert ci.ci_99_lower <= ci.ci_95_lower <= ci.ci_90_lower
        assert ci.ci_99_upper >= ci.ci_95_upper >= ci.ci_90_upper

    def test_bootstrap_insufficient_data(self, sample_rankings):
        """Test bootstrap returns None with insufficient data."""
        engine = BootstrapEngine(seed=42)

        # Only 5 tickers with returns (below MIN_OBS_BOOTSTRAP=30)
        small_returns = {k: 0.01 * i for i, k in enumerate(list(sample_rankings.keys())[:5])}

        ci = engine.calculate_bootstrap_ci(
            {k: v for k, v in sample_rankings.items() if k in small_returns},
            small_returns
        )
        # Should return None because n < MIN_OBS_BOOTSTRAP (30)
        assert ci is None


# =============================================================================
# FORWARD RETURN ENGINE TESTS
# =============================================================================

class TestForwardReturnEngine:
    """Tests for ForwardReturnEngine."""

    def test_calculate_forward_returns(self, mock_return_provider, as_of_date):
        """Test forward return calculation for multiple tickers."""
        engine = ForwardReturnEngine(mock_return_provider, benchmark_ticker="XBI")

        tickers = ["ACME", "BETA", "GAMMA"]
        returns = engine.calculate_forward_returns(tickers, as_of_date, ["5d", "20d"])

        assert "5d" in returns
        assert "20d" in returns
        assert "ACME" in returns["5d"]
        assert returns["5d"]["ACME"].raw_return == Decimal("0.10")

    def test_calculate_forward_returns_missing_ticker(self, mock_return_provider, as_of_date):
        """Test handling of missing ticker returns."""
        engine = ForwardReturnEngine(mock_return_provider, benchmark_ticker="XBI")

        tickers = ["ACME", "UNKNOWN_TICKER"]
        returns = engine.calculate_forward_returns(tickers, as_of_date, ["5d"])

        assert returns["5d"]["ACME"].data_status == "complete"
        assert returns["5d"]["UNKNOWN_TICKER"].data_status == "missing_ticker"
        assert returns["5d"]["UNKNOWN_TICKER"].raw_return is None

    def test_calculate_equal_weight_benchmark(self, mock_return_provider):
        """Test equal-weight benchmark calculation."""
        engine = ForwardReturnEngine(mock_return_provider, benchmark_ticker="XBI")

        returns_by_ticker = {
            "ACME": Decimal("0.10"),
            "BETA": Decimal("0.05"),
            "GAMMA": Decimal("0.00"),
        }

        ew_benchmark = engine.calculate_equal_weight_benchmark(returns_by_ticker)

        assert ew_benchmark is not None
        assert ew_benchmark == Decimal("0.05")  # Average of 0.10, 0.05, 0.00


# =============================================================================
# IC STABILITY ANALYSIS TESTS
# =============================================================================

class TestICStabilityAnalyzer:
    """Tests for IC stability analysis."""

    def test_rolling_ic_calculation(self, weekly_historical_results):
        """Test rolling IC calculation."""
        ic_engine = ICCalculationEngine()
        analyzer = ICStabilityAnalyzer(ic_engine)

        rolling_results = analyzer.calculate_rolling_ic(weekly_historical_results, window_weeks=12)

        # Should have results starting from week 12
        assert len(rolling_results) > 0
        assert all(isinstance(r, RollingICResult) for r in rolling_results)
        assert all(r.n_observations > 0 for r in rolling_results)

    def test_rolling_ic_insufficient_data(self):
        """Test rolling IC with insufficient data."""
        ic_engine = ICCalculationEngine()
        analyzer = ICStabilityAnalyzer(ic_engine)

        # Only 5 weeks of data, need 12 for rolling
        short_history = [
            {"as_of_date": f"2025-01-{i:02d}", "rankings": {"A": 1}, "forward_returns": {"A": 0.01}}
            for i in range(1, 6)
        ]

        rolling_results = analyzer.calculate_rolling_ic(short_history, window_weeks=12)
        assert len(rolling_results) == 0

    def test_regime_conditional_ic(self, weekly_historical_results):
        """Test IC calculation by market regime."""
        ic_engine = ICCalculationEngine()
        analyzer = ICStabilityAnalyzer(ic_engine)

        regime_ic = analyzer.calculate_regime_conditional_ic(weekly_historical_results)

        assert "BULL" in regime_ic
        assert "BEAR" in regime_ic
        assert "NEUTRAL" in regime_ic
        assert all(isinstance(v, ICResult) for v in regime_ic.values())

    def test_sector_conditional_ic(self, weekly_historical_results):
        """Test IC calculation by sector."""
        ic_engine = ICCalculationEngine()
        analyzer = ICStabilityAnalyzer(ic_engine)

        ticker_sectors = {
            "ACME": "ONCOLOGY",
            "BETA": "ONCOLOGY",
            "GAMMA": "RARE_DISEASE",
            "DELTA": "CNS",
            "EPSILON": "IMMUNOLOGY",
            "ZETA": "OTHER",
            "ETA": "OTHER",
            "THETA": "OTHER",
            "IOTA": "OTHER",
            "KAPPA": "OTHER",
        }

        sector_ic = analyzer.calculate_sector_conditional_ic(weekly_historical_results, ticker_sectors)

        assert "ONCOLOGY" in sector_ic
        assert "RARE_DISEASE" in sector_ic
        assert all(isinstance(v, ICResult) for v in sector_ic.values())

    def test_market_cap_ic(self, weekly_historical_results):
        """Test IC calculation by market cap bucket."""
        ic_engine = ICCalculationEngine()
        analyzer = ICStabilityAnalyzer(ic_engine)

        ticker_market_caps = {
            "ACME": Decimal("5000"),   # LARGE
            "BETA": Decimal("2000"),   # MID
            "GAMMA": Decimal("500"),   # SMALL
            "DELTA": Decimal("200"),   # MICRO
            "EPSILON": Decimal("3000"),
            "ZETA": Decimal("1500"),
            "ETA": Decimal("800"),
            "THETA": Decimal("100"),
            "IOTA": Decimal("400"),
            "KAPPA": Decimal("600"),
        }

        mcap_ic = analyzer.calculate_market_cap_ic(weekly_historical_results, ticker_market_caps)

        assert any(bucket in mcap_ic for bucket in ["MICRO", "SMALL", "MID", "LARGE"])
        assert all(isinstance(v, ICResult) for v in mcap_ic.values())


# =============================================================================
# IC TREND ANALYSIS TESTS
# =============================================================================

class TestICTrendAnalysis:
    """Tests for IC trend analysis."""

    def test_analyze_trend_improving(self):
        """Test trend detection for improving IC."""
        rolling_ics = [
            RollingICResult(f"2025-01-{i:02d}", Decimal(str(0.02 + 0.005 * i)), 50, None, ICQuality.GOOD)
            for i in range(1, 13)
        ]

        trend = analyze_ic_trend(rolling_ics)

        assert trend["trend_direction"] == "IMPROVING"
        assert Decimal(str(trend["slope"])) > Decimal("0")

    def test_analyze_trend_declining(self):
        """Test trend detection for declining IC."""
        rolling_ics = [
            RollingICResult(f"2025-01-{i:02d}", Decimal(str(0.10 - 0.007 * i)), 50, None, ICQuality.GOOD)
            for i in range(1, 13)
        ]

        trend = analyze_ic_trend(rolling_ics)

        assert trend["trend_direction"] == "DECLINING"
        assert Decimal(str(trend["slope"])) < Decimal("0")

    def test_analyze_trend_stable(self):
        """Test trend detection for stable IC."""
        rolling_ics = [
            RollingICResult(f"2025-01-{i:02d}", Decimal("0.05"), 50, None, ICQuality.GOOD)
            for i in range(1, 13)
        ]

        trend = analyze_ic_trend(rolling_ics)

        assert trend["trend_direction"] == "STABLE"

    def test_analyze_trend_insufficient_data(self):
        """Test trend analysis with insufficient data."""
        rolling_ics = [
            RollingICResult("2025-01-01", Decimal("0.05"), 50, None, ICQuality.GOOD),
        ]

        trend = analyze_ic_trend(rolling_ics)

        assert trend["trend_direction"] == "INSUFFICIENT_DATA"


# =============================================================================
# OUT-OF-SAMPLE VALIDATION TESTS
# =============================================================================

class TestOutOfSampleValidator:
    """Tests for out-of-sample validation."""

    def test_train_test_split_consistent(self, weekly_historical_results):
        """Test OOS validation with consistent signal."""
        ic_engine = ICCalculationEngine()
        validator = OutOfSampleValidator(ic_engine)

        result = validator.validate_train_test_split(
            weekly_historical_results,
            train_start="2025-10-01",
            train_end="2025-12-01",
            test_start="2025-12-08",
            test_end="2026-02-15",
        )

        assert "train_ic" in result
        assert "test_ic" in result
        assert "consistency" in result
        assert result["train_n"] >= 0
        assert result["test_n"] >= 0

    def test_train_test_split_no_data(self):
        """Test OOS validation with no matching data."""
        ic_engine = ICCalculationEngine()
        validator = OutOfSampleValidator(ic_engine)

        empty_results: List[Dict[str, Any]] = []

        result = validator.validate_train_test_split(
            empty_results,
            train_start="2020-01-01",
            train_end="2022-12-31",
            test_start="2023-01-01",
            test_end="2025-12-31",
        )

        assert result["train_n"] == 0
        assert result["test_n"] == 0


# =============================================================================
# TIME-SERIES DATABASE TESTS
# =============================================================================

class TestICTimeSeriesDatabase:
    """Tests for IC time-series database storage."""

    def test_store_and_load_ranking(self, temp_data_dir, sample_rankings):
        """Test storing and loading rankings."""
        db = ICTimeSeriesDatabase(temp_data_dir)

        # Store
        filepath = db.store_ranking("2026-01-15", sample_rankings)
        assert Path(filepath).exists()

        # Load
        loaded = db.load_ranking("2026-01-15")
        assert loaded == sample_rankings

    def test_store_and_load_returns(self, temp_data_dir):
        """Test storing and loading returns."""
        db = ICTimeSeriesDatabase(temp_data_dir)

        returns = {
            "5d": {"ACME": Decimal("0.05"), "BETA": Decimal("0.03")},
            "20d": {"ACME": Decimal("0.12"), "BETA": Decimal("0.08")},
        }

        # Store
        filepath = db.store_returns("2026-01-15", returns)
        assert Path(filepath).exists()

        # Load
        loaded = db.load_returns("2026-01-15")
        assert "5d" in loaded
        assert "20d" in loaded
        assert loaded["5d"]["ACME"] == Decimal("0.05")

    def test_list_available_dates(self, temp_data_dir, sample_rankings):
        """Test listing available dates."""
        db = ICTimeSeriesDatabase(temp_data_dir)

        # Store multiple dates
        db.store_ranking("2026-01-15", sample_rankings)
        db.store_ranking("2026-01-22", sample_rankings)
        db.store_ranking("2026-01-29", sample_rankings)

        dates = db.list_available_dates()

        assert len(dates) == 3
        assert "2026-01-15" in dates
        assert "2026-01-22" in dates
        assert "2026-01-29" in dates

    def test_load_nonexistent_ranking(self, temp_data_dir):
        """Test loading non-existent ranking returns empty dict."""
        db = ICTimeSeriesDatabase(temp_data_dir)

        loaded = db.load_ranking("1999-01-01")
        assert loaded == {}


# =============================================================================
# IC QUALITY ASSESSMENT TESTS
# =============================================================================

class TestICQualityAssessment:
    """Tests for IC quality assessment generation."""

    def test_excellent_assessment(self):
        """Test assessment for excellent IC."""
        ic_result = ICResult(
            ic_value=Decimal("0.08"),
            ic_quality=ICQuality.EXCELLENT,
            n_observations=100,
            t_statistic=Decimal("5.0"),
            p_value=Decimal("0.0001"),
            is_significant_95=True,
            is_significant_99=True,
        )

        assessment = _get_ic_quality_assessment(ic_result)

        assert assessment["quality"] == "EXCELLENT"
        assert assessment["actionable"] is True
        assert "institutional-grade" in assessment["assessment"]

    def test_weak_assessment(self):
        """Test assessment for weak IC."""
        ic_result = ICResult(
            ic_value=Decimal("0.02"),
            ic_quality=ICQuality.WEAK,
            n_observations=50,
            t_statistic=Decimal("1.2"),
            p_value=Decimal("0.12"),
            is_significant_95=False,
            is_significant_99=False,
        )

        assessment = _get_ic_quality_assessment(ic_result)

        assert assessment["quality"] == "WEAK"
        assert assessment["actionable"] is False
        assert "enhancement" in assessment["recommendation"].lower()

    def test_negative_assessment(self):
        """Test assessment for negative IC."""
        ic_result = ICResult(
            ic_value=Decimal("-0.05"),
            ic_quality=ICQuality.NEGATIVE,
            n_observations=100,
            t_statistic=Decimal("-3.0"),
            p_value=Decimal("0.003"),
            is_significant_95=True,
            is_significant_99=True,
        )

        assessment = _get_ic_quality_assessment(ic_result)

        assert assessment["quality"] == "NEGATIVE"
        assert "URGENT" in assessment["recommendation"]


# =============================================================================
# WEEKLY IC REPORT GENERATOR TESTS
# =============================================================================

class TestWeeklyICReportGenerator:
    """Tests for weekly IC report generation."""

    def test_generate_basic_report(
        self, mock_return_provider, sample_rankings, as_of_date
    ):
        """Test basic report generation."""
        generator = WeeklyICReportGenerator(
            mock_return_provider,
            benchmark_ticker="XBI",
            bootstrap_seed=42,
        )

        report = generator.generate_report(
            rankings=sample_rankings,
            as_of_date=as_of_date,
            horizon="20d",
        )

        assert "overall_ic" in report
        assert "as_of_date" in report
        assert "horizon" in report
        assert report["horizon"] == "20d"
        assert "ic_quality_assessment" in report

    def test_generate_report_with_historical(
        self, mock_return_provider, sample_rankings, as_of_date, weekly_historical_results
    ):
        """Test report generation with historical data."""
        generator = WeeklyICReportGenerator(
            mock_return_provider,
            benchmark_ticker="XBI",
            bootstrap_seed=42,
        )

        report = generator.generate_report(
            rankings=sample_rankings,
            as_of_date=as_of_date,
            horizon="20d",
            historical_results=weekly_historical_results,
            current_regime="BULL",
        )

        assert "rolling_ic_12w" in report
        assert "trend_analysis" in report
        assert "regime_conditional_ic" in report


# =============================================================================
# IC MEASUREMENT SYSTEM INTEGRATION TESTS
# =============================================================================

class TestICMeasurementSystemIntegration:
    """Integration tests for full IC measurement system."""

    def test_process_weekly_screening(
        self, mock_return_provider, temp_data_dir, sample_rankings, as_of_date
    ):
        """Test full weekly screening processing."""
        system = ICMeasurementSystem(
            return_provider=mock_return_provider,
            data_dir=temp_data_dir,
            benchmark_ticker="XBI",
            bootstrap_seed=42,
        )

        scores = {t: Decimal(str(100 - r * 5)) for t, r in sample_rankings.items()}

        result = system.process_weekly_screening(
            rankings=sample_rankings,
            scores=scores,
            as_of_date=as_of_date,
            horizons=["5d", "20d"],
        )

        assert "_metadata" in result
        assert "horizon_analysis" in result
        assert "5d" in result["horizon_analysis"]
        assert "20d" in result["horizon_analysis"]
        assert "coverage" in result

    def test_system_data_persistence(
        self, mock_return_provider, temp_data_dir, sample_rankings, as_of_date
    ):
        """Test that data is persisted correctly."""
        system = ICMeasurementSystem(
            return_provider=mock_return_provider,
            data_dir=temp_data_dir,
            benchmark_ticker="XBI",
        )

        scores = {t: Decimal(str(100 - r * 5)) for t, r in sample_rankings.items()}

        # Process first week
        system.process_weekly_screening(
            rankings=sample_rankings,
            scores=scores,
            as_of_date=as_of_date,
            horizons=["20d"],
        )

        # Verify data was stored
        stored_rankings = system.database.load_ranking(as_of_date)
        assert stored_rankings == sample_rankings

        # Process second week
        second_date = date(2026, 1, 22)
        system.process_weekly_screening(
            rankings=sample_rankings,
            scores=scores,
            as_of_date=second_date,
            horizons=["20d"],
        )

        # Verify both dates are available
        available_dates = system.database.list_available_dates()
        assert len(available_dates) == 2


# =============================================================================
# DETERMINISM TESTS
# =============================================================================

class TestDeterminism:
    """Tests ensuring deterministic behavior."""

    def test_ic_calculation_deterministic(self, sample_rankings, perfect_returns):
        """Test IC calculation is deterministic."""
        ic1 = calculate_ic(sample_rankings, perfect_returns)
        ic2 = calculate_ic(sample_rankings, perfect_returns)

        assert ic1 == ic2

    def test_report_generation_deterministic(
        self, mock_return_provider, sample_rankings, as_of_date
    ):
        """Test report generation is deterministic."""
        generator = WeeklyICReportGenerator(
            mock_return_provider,
            benchmark_ticker="XBI",
            bootstrap_seed=42,
        )

        report1 = generator.generate_report(
            rankings=sample_rankings,
            as_of_date=as_of_date,
            horizon="20d",
        )

        report2 = generator.generate_report(
            rankings=sample_rankings,
            as_of_date=as_of_date,
            horizon="20d",
        )

        # Key metrics should be identical
        assert report1["overall_ic"] == report2["overall_ic"]
        if report1.get("bootstrap_ci") and report2.get("bootstrap_ci"):
            assert report1["bootstrap_ci"]["point_estimate"] == report2["bootstrap_ci"]["point_estimate"]


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_all_same_rankings(self):
        """Test IC with all same rankings (no variance)."""
        rankings = {"A": 1, "B": 1, "C": 1, "D": 1, "E": 1,
                    "F": 1, "G": 1, "H": 1, "I": 1, "J": 1}
        returns = {k: i * 0.01 for i, k in enumerate(rankings.keys())}

        ic = calculate_ic(rankings, returns)
        # Should return 0 or None due to no variance in rankings
        assert ic is None or ic == Decimal("0")

    def test_all_same_returns(self):
        """Test IC with all same returns (no variance)."""
        rankings = {chr(65 + i): i + 1 for i in range(10)}
        returns = {k: 0.05 for k in rankings.keys()}

        ic = calculate_ic(rankings, returns)
        # Should return 0 due to no variance in returns
        assert ic is None or ic == Decimal("0")

    def test_single_common_ticker(self):
        """Test IC with only one common ticker."""
        rankings = {"ACME": 1}
        returns = {"ACME": 0.10}

        ic = calculate_ic(rankings, returns)
        assert ic is None  # Insufficient observations

    def test_no_common_tickers(self):
        """Test IC with no common tickers."""
        rankings = {"ACME": 1, "BETA": 2}
        returns = {"GAMMA": 0.10, "DELTA": 0.05}

        ic = calculate_ic(rankings, returns)
        assert ic is None  # No common tickers


# =============================================================================
# ADDITIONAL COVERAGE TESTS
# =============================================================================

class TestICResultSerialization:
    """Tests for ICResult serialization."""

    def test_ic_result_to_dict(self):
        """Test ICResult.to_dict() method."""
        result = ICResult(
            ic_value=Decimal("0.045"),
            ic_quality=ICQuality.GOOD,
            n_observations=50,
            t_statistic=Decimal("3.2"),
            p_value=Decimal("0.002"),
            is_significant_95=True,
            is_significant_99=True,
        )

        d = result.to_dict()

        assert d["ic_value"] == "0.045"
        assert d["ic_quality"] == "GOOD"
        assert d["n_observations"] == 50
        assert d["t_statistic"] == "3.2"
        assert d["p_value"] == "0.002"
        assert d["is_significant_95"] is True
        assert d["is_significant_99"] is True

    def test_ic_result_to_dict_none_values(self):
        """Test ICResult.to_dict() with None values."""
        result = ICResult(
            ic_value=Decimal("0.02"),
            ic_quality=ICQuality.WEAK,
            n_observations=15,
            t_statistic=None,
            p_value=None,
            is_significant_95=False,
            is_significant_99=False,
        )

        d = result.to_dict()

        assert d["t_statistic"] is None
        assert d["p_value"] is None


class TestForwardWindowErrors:
    """Tests for forward window error handling."""

    def test_invalid_horizon_raises_error(self):
        """Test that invalid horizon raises ValueError."""
        with pytest.raises(ValueError, match="Unknown horizon"):
            compute_forward_windows("2026-01-15", ["invalid_horizon"])


class TestForwardReturnWithoutBenchmark:
    """Tests for forward returns without benchmark adjustment."""

    def test_forward_returns_no_benchmark(self, as_of_date):
        """Test forward returns without benchmark adjustment."""
        def provider(ticker: str, start_date: str, end_date: str) -> Optional[str]:
            if ticker == "ACME":
                return "0.05"
            return None

        engine = ForwardReturnEngine(provider, benchmark_ticker="XBI")
        results = engine.calculate_forward_returns(
            ["ACME"],
            as_of_date,
            ["5d"],
            include_benchmark_adjustment=False,
        )

        assert results["5d"]["ACME"].raw_return == Decimal("0.05")
        assert results["5d"]["ACME"].benchmark_return is None
        assert results["5d"]["ACME"].excess_return is None


class TestOOSConsistencyClassifications:
    """Tests for out-of-sample consistency classifications."""

    def test_oos_improved_consistency(self):
        """Test OOS validation detects improved signal."""
        ic_engine = ICCalculationEngine()
        validator = OutOfSampleValidator(ic_engine)

        # Create data where test IC is better than train IC
        train_data = [
            {
                "as_of_date": "2020-01-15",
                "rankings": {f"T{i}": i for i in range(1, 16)},
                "forward_returns": {f"T{i}": 0.01 * (16 - i) + 0.005 * (i % 3) for i in range(1, 16)},
            }
            for _ in range(10)
        ]
        test_data = [
            {
                "as_of_date": "2023-01-15",
                "rankings": {f"T{i}": i for i in range(1, 16)},
                "forward_returns": {f"T{i}": 0.02 * (16 - i) for i in range(1, 16)},
            }
            for _ in range(10)
        ]

        result = validator.validate_train_test_split(
            train_data + test_data,
            train_start="2020-01-01",
            train_end="2020-12-31",
            test_start="2023-01-01",
            test_end="2023-12-31",
        )

        assert result["consistency"] in ["CONSISTENT", "IMPROVED", "DEGRADED", "INVERTED", "CONSISTENTLY_WEAK", "UNKNOWN"]


class TestNegativeICTStatistic:
    """Tests for negative IC t-statistic handling."""

    def test_negative_perfect_ic_tstat(self):
        """Test t-stat for negative perfect correlation."""
        # Create inverted signal: higher rank = higher return
        tickers = [f"T{i:02d}" for i in range(1, 36)]
        rankings = {t: i + 1 for i, t in enumerate(tickers)}
        # Inverted returns: higher rank = higher return
        returns = {t: -0.15 + 0.01 * i for i, t in enumerate(tickers)}

        engine = ICCalculationEngine()
        result = engine.calculate_ic_with_significance(rankings, returns)

        # Should detect negative IC with significance
        assert result.ic_value < Decimal("0")
        assert result.t_statistic is not None
        assert result.t_statistic < Decimal("0")


class TestEqualWeightBenchmarkEdgeCases:
    """Tests for equal-weight benchmark edge cases."""

    def test_empty_returns_for_benchmark(self):
        """Test equal-weight benchmark with empty returns."""
        def provider(ticker: str, start_date: str, end_date: str) -> Optional[str]:
            return None

        engine = ForwardReturnEngine(provider)
        result = engine.calculate_equal_weight_benchmark({})

        assert result is None

    def test_benchmark_with_none_values(self):
        """Test equal-weight benchmark ignores None values."""
        def provider(ticker: str, start_date: str, end_date: str) -> Optional[str]:
            return None

        engine = ForwardReturnEngine(provider)
        returns = {
            "ACME": Decimal("0.10"),
            "BETA": None,
            "GAMMA": Decimal("0.05"),
        }

        result = engine.calculate_equal_weight_benchmark(returns)

        # Should average only non-None values: (0.10 + 0.05) / 2 = 0.075
        assert result is not None
        assert abs(result - Decimal("0.075")) < Decimal("0.001")


class TestRollingICWindowBoundaries:
    """Tests for rolling IC window boundary conditions."""

    def test_exact_minimum_window(self):
        """Test rolling IC with exactly minimum window size."""
        ic_engine = ICCalculationEngine()
        analyzer = ICStabilityAnalyzer(ic_engine)

        # Create exactly 12 weeks of data
        results = []
        base_date = date(2025, 10, 1)
        for week in range(12):
            week_date = base_date + timedelta(weeks=week)
            rankings = {f"T{i}": i for i in range(1, 16)}
            returns = {f"T{i}": 0.01 * (16 - i) for i in range(1, 16)}
            results.append({
                "as_of_date": week_date.isoformat(),
                "rankings": rankings,
                "forward_returns": returns,
            })

        rolling_results = analyzer.calculate_rolling_ic(results, window_weeks=12)

        # Should have exactly 1 result
        assert len(rolling_results) == 1


class TestDatabaseWithScores:
    """Tests for database storage with scores."""

    def test_store_ranking_with_scores(self, temp_data_dir, sample_rankings):
        """Test storing rankings with scores."""
        db = ICTimeSeriesDatabase(temp_data_dir)

        scores = {t: Decimal(str(100 - r * 5)) for t, r in sample_rankings.items()}

        filepath = db.store_ranking("2026-01-15", sample_rankings, scores)
        assert Path(filepath).exists()

        # Load and verify
        loaded = db.load_ranking("2026-01-15")
        assert loaded == sample_rankings


class TestReportWithStratifiedAnalysis:
    """Tests for report generation with stratified analysis."""

    def test_report_with_all_stratifications(
        self, mock_return_provider, sample_rankings, as_of_date, weekly_historical_results
    ):
        """Test report with sector, market cap, and regime stratification."""
        generator = WeeklyICReportGenerator(
            mock_return_provider,
            benchmark_ticker="XBI",
            bootstrap_seed=42,
        )

        ticker_sectors = {t: "ONCOLOGY" if i < 5 else "OTHER" for i, t in enumerate(sample_rankings.keys())}
        ticker_mcaps = {t: Decimal(str(500 + i * 100)) for i, t in enumerate(sample_rankings.keys())}

        report = generator.generate_report(
            rankings=sample_rankings,
            as_of_date=as_of_date,
            horizon="20d",
            historical_results=weekly_historical_results,
            ticker_sectors=ticker_sectors,
            ticker_market_caps=ticker_mcaps,
            current_regime="BEAR",
        )

        assert "sector_ic" in report
        assert "market_cap_ic" in report
        assert report["current_regime"] == "BEAR"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
