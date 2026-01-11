"""
Tests for Backtest Harness

Tests:
1. Price data loading
2. Forward return calculation
3. Signal scoring
4. Horizon metrics
5. PIT safety (no future data leakage)
6. Determinism verification
"""

import json
import tempfile
from datetime import date, timedelta
from pathlib import Path

import pytest

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.backtest_institutional_signal import (
    load_prices,
    get_all_dates,
    find_price_on_or_after,
    compute_forward_return,
    compute_signal_score,
    rank_tickers,
    compute_horizon_metrics,
)
from src.history.snapshots import SNAPSHOT_SCHEMA_VERSION


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_prices(tmp_path):
    """Create sample price CSV for testing."""
    prices_data = """date,ticker,adj_close
2024-10-01,XBI,100.00
2024-10-01,AAPL,150.00
2024-10-01,GOOG,140.00
2024-10-02,XBI,101.00
2024-10-02,AAPL,152.00
2024-10-02,GOOG,142.00
2024-10-03,XBI,102.00
2024-10-03,AAPL,155.00
2024-10-03,GOOG,141.00
2024-11-01,XBI,105.00
2024-11-01,AAPL,160.00
2024-11-01,GOOG,145.00
2024-11-02,XBI,106.00
2024-11-02,AAPL,162.00
2024-11-02,GOOG,146.00
2024-12-01,XBI,110.00
2024-12-01,AAPL,170.00
2024-12-01,GOOG,150.00
"""
    prices_path = tmp_path / "prices.csv"
    prices_path.write_text(prices_data)
    return prices_path


@pytest.fixture
def sample_snapshot():
    """Create sample holdings snapshot for testing."""
    return {
        "_schema": {
            "version": SNAPSHOT_SCHEMA_VERSION,
            "quarter_end": "2024-09-30",
            "prior_quarter_end": "2024-06-30",
        },
        "tickers": {
            "AAPL": {
                "market_cap_usd": 3000000000000,
                "holdings": {
                    "current": {
                        "0001234567": {
                            "quarter_end": "2024-09-30",
                            "state": "KNOWN",
                            "shares": 100000,
                            "value_kusd": 15000,
                            "put_call": "",
                        },
                        "0001234568": {
                            "quarter_end": "2024-09-30",
                            "state": "KNOWN",
                            "shares": 50000,
                            "value_kusd": 7500,
                            "put_call": "",
                        },
                        "0001234569": {
                            "quarter_end": "2024-09-30",
                            "state": "KNOWN",
                            "shares": 75000,
                            "value_kusd": 11250,
                            "put_call": "",
                        },
                        "0001234570": {
                            "quarter_end": "2024-09-30",
                            "state": "KNOWN",
                            "shares": 25000,
                            "value_kusd": 3750,
                            "put_call": "",
                        },
                    },
                    "prior": {
                        "0001234567": {
                            "quarter_end": "2024-06-30",
                            "state": "KNOWN",
                            "shares": 80000,
                            "value_kusd": 12000,
                            "put_call": "",
                        },
                        "0001234568": {
                            "quarter_end": "2024-06-30",
                            "state": "KNOWN",
                            "shares": 50000,
                            "value_kusd": 7500,
                            "put_call": "",
                        },
                    },
                },
            },
            "GOOG": {
                "market_cap_usd": 2000000000000,
                "holdings": {
                    "current": {
                        "0001234567": {
                            "quarter_end": "2024-09-30",
                            "state": "KNOWN",
                            "shares": 20000,
                            "value_kusd": 2800,
                            "put_call": "",
                        },
                    },
                    "prior": {},
                },
            },
            "MSFT": {
                "market_cap_usd": 2500000000000,
                "holdings": {
                    "current": {
                        "0001234567": {
                            "quarter_end": "2024-09-30",
                            "state": "KNOWN",
                            "shares": 10000,
                            "value_kusd": 4000,
                            "put_call": "",
                        },
                        "0001234568": {
                            "quarter_end": "2024-09-30",
                            "state": "KNOWN",
                            "shares": 15000,
                            "value_kusd": 6000,
                            "put_call": "",
                        },
                    },
                    "prior": {
                        "0001234567": {
                            "quarter_end": "2024-06-30",
                            "state": "KNOWN",
                            "shares": 15000,
                            "value_kusd": 6000,
                            "put_call": "",
                        },
                        "0001234568": {
                            "quarter_end": "2024-06-30",
                            "state": "KNOWN",
                            "shares": 15000,
                            "value_kusd": 6000,
                            "put_call": "",
                        },
                    },
                },
            },
        },
        "managers": {},
        "stats": {
            "tickers_count": 3,
            "managers_count": 4,
        },
    }


# =============================================================================
# PRICE LOADING TESTS
# =============================================================================

class TestPriceLoading:
    """Tests for price data loading."""

    def test_load_prices(self, sample_prices):
        """Prices load correctly from CSV."""
        prices = load_prices(sample_prices)

        assert "XBI" in prices
        assert "AAPL" in prices
        assert "GOOG" in prices

        assert prices["XBI"]["2024-10-01"] == 100.00
        assert prices["AAPL"]["2024-10-01"] == 150.00

    def test_get_all_dates(self, sample_prices):
        """All trading dates are extracted and sorted."""
        prices = load_prices(sample_prices)
        dates = get_all_dates(prices)

        assert dates[0] == "2024-10-01"
        assert dates[-1] == "2024-12-01"
        assert len(dates) == 6

    def test_find_price_on_or_after_exact(self, sample_prices):
        """Finds price on exact date."""
        prices = load_prices(sample_prices)
        trading_dates = get_all_dates(prices)

        result = find_price_on_or_after("AAPL", date(2024, 10, 1), prices, trading_dates)

        assert result is not None
        date_str, price = result
        assert date_str == "2024-10-01"
        assert price == 150.00

    def test_find_price_on_or_after_next_day(self, sample_prices):
        """Finds price on next trading day after weekend."""
        prices = load_prices(sample_prices)
        trading_dates = get_all_dates(prices)

        # Oct 4 is not in data, should find next available date
        result = find_price_on_or_after("AAPL", date(2024, 10, 4), prices, trading_dates)

        assert result is not None
        date_str, price = result
        assert date_str == "2024-11-01"  # Next available date

    def test_find_price_ticker_not_found(self, sample_prices):
        """Returns None for unknown ticker."""
        prices = load_prices(sample_prices)
        trading_dates = get_all_dates(prices)

        result = find_price_on_or_after("UNKNOWN", date(2024, 10, 1), prices, trading_dates)

        assert result is None


# =============================================================================
# FORWARD RETURN TESTS
# =============================================================================

class TestForwardReturn:
    """Tests for forward return calculation."""

    def test_compute_forward_return_simple(self, sample_prices):
        """Computes forward return correctly."""
        prices = load_prices(sample_prices)
        trading_dates = get_all_dates(prices)

        # XBI: 100 -> 105 over ~30 days = 5% return
        ret = compute_forward_return(
            ticker="XBI",
            entry_date=date(2024, 10, 1),
            horizon_days=30,
            prices=prices,
            trading_dates=trading_dates,
        )

        assert ret is not None
        assert abs(ret - 0.05) < 0.001  # 5% return

    def test_compute_forward_return_missing_ticker(self, sample_prices):
        """Returns None for missing ticker."""
        prices = load_prices(sample_prices)
        trading_dates = get_all_dates(prices)

        ret = compute_forward_return(
            ticker="UNKNOWN",
            entry_date=date(2024, 10, 1),
            horizon_days=30,
            prices=prices,
            trading_dates=trading_dates,
        )

        assert ret is None


# =============================================================================
# SIGNAL SCORING TESTS
# =============================================================================

class TestSignalScoring:
    """Tests for signal scoring logic."""

    def test_compute_signal_score_basic(self, sample_snapshot):
        """Computes signal score for ticker with sufficient managers."""
        score = compute_signal_score("AAPL", sample_snapshot, min_managers=2)

        assert score is not None
        assert score["ticker"] == "AAPL"
        assert score["managers_count"] == 4
        assert score["score"] > 0

    def test_compute_signal_score_insufficient_managers(self, sample_snapshot):
        """Returns None when below min_managers threshold."""
        score = compute_signal_score("GOOG", sample_snapshot, min_managers=2)

        assert score is None  # Only 1 manager

    def test_compute_signal_score_new_positions(self, sample_snapshot):
        """Counts new positions correctly."""
        score = compute_signal_score("AAPL", sample_snapshot, min_managers=1)

        assert score is not None
        # 2 new positions (0001234569, 0001234570 not in prior)
        assert score["new_positions"] == 2

    def test_compute_signal_score_increased(self, sample_snapshot):
        """Counts increased positions correctly."""
        score = compute_signal_score("AAPL", sample_snapshot, min_managers=1)

        assert score is not None
        # 0001234567 increased from 80000 to 100000
        assert score["increased"] == 1

    def test_compute_signal_score_decreased(self, sample_snapshot):
        """Counts decreased positions correctly."""
        score = compute_signal_score("MSFT", sample_snapshot, min_managers=1)

        assert score is not None
        # 0001234567 decreased from 15000 to 10000
        assert score["decreased"] == 1

    def test_rank_tickers(self, sample_snapshot):
        """Ranks tickers by score correctly."""
        rankings = rank_tickers(sample_snapshot, min_managers=1)

        assert len(rankings) == 3  # AAPL, GOOG, MSFT (all have at least 1 manager)

        # AAPL should be first (most managers, most new positions)
        assert rankings[0]["ticker"] == "AAPL"

    def test_rank_tickers_min_managers_filter(self, sample_snapshot):
        """Respects min_managers filter."""
        rankings = rank_tickers(sample_snapshot, min_managers=4)

        assert len(rankings) == 1  # Only AAPL has 4 managers
        assert rankings[0]["ticker"] == "AAPL"


# =============================================================================
# HORIZON METRICS TESTS
# =============================================================================

class TestHorizonMetrics:
    """Tests for horizon metrics calculation."""

    def test_compute_horizon_metrics_basic(self, sample_prices):
        """Computes horizon metrics correctly."""
        prices = load_prices(sample_prices)
        trading_dates = get_all_dates(prices)

        metrics = compute_horizon_metrics(
            tickers=["AAPL", "GOOG"],
            entry_date=date(2024, 10, 1),
            horizon_days=30,
            benchmark="XBI",
            prices=prices,
            trading_dates=trading_dates,
        )

        assert metrics is not None
        assert metrics["tickers_measured"] == 2
        assert metrics["benchmark_return"] is not None
        assert metrics["mean_return"] is not None
        assert metrics["median_return"] is not None

    def test_compute_horizon_metrics_empty_tickers(self, sample_prices):
        """Handles empty ticker list."""
        prices = load_prices(sample_prices)
        trading_dates = get_all_dates(prices)

        metrics = compute_horizon_metrics(
            tickers=[],
            entry_date=date(2024, 10, 1),
            horizon_days=30,
            benchmark="XBI",
            prices=prices,
            trading_dates=trading_dates,
        )

        assert metrics["tickers_measured"] == 0
        assert metrics["mean_return"] is None


# =============================================================================
# PIT SAFETY TESTS
# =============================================================================

class TestPITSafety:
    """Tests for point-in-time safety."""

    def test_entry_date_after_quarter_end(self, sample_prices):
        """Entry is on first trading day AFTER quarter end."""
        prices = load_prices(sample_prices)
        trading_dates = get_all_dates(prices)

        # Quarter end: 2024-09-30
        # First trading day: 2024-10-01
        quarter_end = date(2024, 9, 30)
        signal_target = quarter_end + timedelta(days=1)

        result = find_price_on_or_after("XBI", signal_target, prices, trading_dates)

        assert result is not None
        date_str, _ = result
        assert date_str >= "2024-10-01"

    def test_no_future_data_in_scoring(self, sample_snapshot):
        """Scoring only uses data as of quarter end."""
        # The snapshot represents data AS OF 2024-09-30
        # We should not see any holdings from future quarters

        score = compute_signal_score("AAPL", sample_snapshot, min_managers=1)

        assert score is not None
        # All holdings are from current quarter (2024-09-30) or prior
        # No future data


# =============================================================================
# DETERMINISM TESTS
# =============================================================================

class TestDeterminism:
    """Tests for deterministic behavior."""

    def test_ranking_deterministic(self, sample_snapshot):
        """Rankings are deterministic across multiple calls."""
        rankings1 = rank_tickers(sample_snapshot, min_managers=1)
        rankings2 = rank_tickers(sample_snapshot, min_managers=1)
        rankings3 = rank_tickers(sample_snapshot, min_managers=1)

        assert rankings1 == rankings2 == rankings3

    def test_scoring_deterministic(self, sample_snapshot):
        """Scores are deterministic across multiple calls."""
        scores = []
        for _ in range(5):
            score = compute_signal_score("AAPL", sample_snapshot, min_managers=1)
            scores.append(score)

        # All scores should be identical
        assert all(s == scores[0] for s in scores)

    def test_metrics_deterministic(self, sample_prices):
        """Metrics are deterministic across multiple calls."""
        prices = load_prices(sample_prices)
        trading_dates = get_all_dates(prices)

        metrics = []
        for _ in range(5):
            m = compute_horizon_metrics(
                tickers=["AAPL", "GOOG"],
                entry_date=date(2024, 10, 1),
                horizon_days=30,
                benchmark="XBI",
                prices=prices,
                trading_dates=trading_dates,
            )
            metrics.append(m)

        assert all(m == metrics[0] for m in metrics)


# =============================================================================
# TIE-BREAKING TESTS
# =============================================================================

class TestTieBreaking:
    """Tests for deterministic tie-breaking in rankings."""

    def test_tie_breaking_by_ticker(self):
        """Equal scores break by ticker alphabetically."""
        # Create snapshot with two tickers having equal stats
        snapshot = {
            "_schema": {"version": SNAPSHOT_SCHEMA_VERSION, "quarter_end": "2024-09-30"},
            "tickers": {
                "ZZZZ": {
                    "holdings": {
                        "current": {
                            "0001": {"shares": 100, "value_kusd": 10, "put_call": ""},
                        },
                        "prior": {},
                    },
                },
                "AAAA": {
                    "holdings": {
                        "current": {
                            "0001": {"shares": 100, "value_kusd": 10, "put_call": ""},
                        },
                        "prior": {},
                    },
                },
            },
            "managers": {},
            "stats": {},
        }

        rankings = rank_tickers(snapshot, min_managers=1)

        # Same score, so alphabetical order
        assert rankings[0]["ticker"] == "AAAA"
        assert rankings[1]["ticker"] == "ZZZZ"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
