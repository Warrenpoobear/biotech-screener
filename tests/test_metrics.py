"""Tests for Backtest Metrics Module."""
import pytest
from decimal import Decimal
from typing import Optional

from backtest.metrics import (
    next_trading_day,
    add_trading_days,
    compute_forward_windows,
    compute_spearman_ic,
    assign_quintiles,
    assign_buckets,
    compute_bucket_returns,
    compute_quintile_returns,
    compute_period_metrics,
    aggregate_metrics,
    run_metrics_suite,
    generate_attribution_frame,
    _normalize_horizon,
    _get_bucket_config,
    _rank_data,
    BUCKET_THRESHOLDS,
)


class TestTradingCalendar:
    def test_next_trading_day_weekday(self):
        assert next_trading_day("2024-07-01") == "2024-07-02"
    
    def test_next_trading_day_friday(self):
        assert next_trading_day("2024-07-05") == "2024-07-08"
    
    def test_add_trading_days(self):
        result = add_trading_days("2024-07-01", 5)
        assert result == "2024-07-08"
    
    def test_compute_forward_windows(self):
        windows = compute_forward_windows("2024-07-01")
        assert "63d" in windows
        assert windows["63d"]["start"] == "2024-07-02"
        assert windows["63d"]["display"] == "3m"


class TestHorizonNormalization:
    def test_normalize_internal_format(self):
        assert _normalize_horizon("63d") == "63d"
        assert _normalize_horizon("126d") == "126d"
        assert _normalize_horizon("252d") == "252d"
    
    def test_normalize_display_format(self):
        assert _normalize_horizon("3m") == "63d"
        assert _normalize_horizon("6m") == "126d"
        assert _normalize_horizon("12m") == "252d"
    
    def test_normalize_invalid(self):
        with pytest.raises(ValueError):
            _normalize_horizon("invalid")


class TestAdaptiveBuckets:
    def test_tercile_for_small_n(self):
        config = _get_bucket_config(15)
        assert config["n_buckets"] == 3
        assert config["bucket_type"] == "tercile"
    
    def test_relaxed_quintile_for_medium_n(self):
        config = _get_bucket_config(35)
        assert config["n_buckets"] == 5
        assert config["bucket_type"] == "quintile_relaxed"
        assert config["min_obs_per_bucket"] == 2
    
    def test_standard_quintile_for_large_n(self):
        config = _get_bucket_config(60)
        assert config["n_buckets"] == 5
        assert config["bucket_type"] == "quintile_standard"
        assert config["min_obs_per_bucket"] == 5
    
    def test_assign_buckets_tercile(self):
        ranked_secs = [{"ticker": f"T{i}", "composite_rank": i} for i in range(1, 16)]
        buckets, config = assign_buckets(ranked_secs)
        assert config["n_buckets"] == 3
        assert buckets["T1"] == 1  # Bottom tercile
        assert buckets["T15"] == 3  # Top tercile
    
    def test_bucket_returns_tercile(self):
        ranked_secs = [{"ticker": f"T{i}", "composite_rank": i} for i in range(1, 16)]
        returns = {f"T{i}": Decimal(str(i * 0.01)) for i in range(1, 16)}
        result = compute_bucket_returns(ranked_secs, returns)
        assert result["bucket_type"] == "tercile"
        assert result["n_buckets"] == 3
        assert result["top_minus_bottom"] is not None


class TestSpearmanIC:
    def test_perfect_positive(self):
        scores = [Decimal(str(i * 10)) for i in range(1, 15)]
        returns = [Decimal(str(i * 0.01)) for i in range(1, 15)]
        ic = compute_spearman_ic(scores, returns)
        assert ic is not None
        assert abs(ic - Decimal("1.0")) < Decimal("0.001")
    
    def test_perfect_negative(self):
        scores = [Decimal(str(i * 10)) for i in range(1, 15)]
        returns = [Decimal(str((15 - i) * 0.01)) for i in range(1, 15)]
        ic = compute_spearman_ic(scores, returns)
        assert ic is not None
        assert abs(ic + Decimal("1.0")) < Decimal("0.001")
    
    def test_insufficient_observations(self):
        scores = [Decimal("10"), Decimal("20")]
        returns = [Decimal("0.01"), Decimal("0.02")]
        ic = compute_spearman_ic(scores, returns)
        assert ic is None


class TestQuintileReturns:
    def test_with_sufficient_obs(self):
        ranked_secs = [{"ticker": f"T{i:02d}", "composite_rank": i} for i in range(1, 51)]
        returns = {f"T{i:02d}": Decimal(str(i * 0.01)) for i in range(1, 51)}
        result = compute_quintile_returns(ranked_secs, returns)
        assert result["q1_mean_return"] is not None
        assert result["q5_mean_return"] is not None
        assert Decimal(result["q5_mean_return"]) > Decimal(result["q1_mean_return"])


class TestPeriodMetrics:
    def test_structure(self):
        def mock_provider(ticker, start, end):
            return "0.05"
        
        snapshot = {
            "as_of_date": "2024-07-01",
            "ranked_securities": [
                {"ticker": f"T{i}", "composite_score": str(50 + i), "composite_rank": i, 
                 "stage_bucket": "late", "market_cap_bucket": "small"}
                for i in range(1, 21)
            ],
        }
        
        result = compute_period_metrics(snapshot, mock_provider, horizons=["63d"])
        assert "63d" in result["horizons"]
        assert result["horizons"]["63d"]["display_name"] == "3m"
        assert "bucket_metrics" in result["horizons"]["63d"]
    
    def test_pit_safe(self):
        def mock_provider(ticker, start, end):
            return "0.05"
        
        snapshot = {
            "as_of_date": "2024-07-01",
            "ranked_securities": [{"ticker": "T1", "composite_score": "50", "composite_rank": 1,
                                   "stage_bucket": "mid", "market_cap_bucket": "small"}],
        }
        
        result = compute_period_metrics(snapshot, mock_provider, horizons=["63d"])
        window = result["horizons"]["63d"]["window"]
        assert window["start"] == "2024-07-02"  # Next trading day


class TestAggregation:
    def test_aggregate_ic(self):
        all_period = {
            "2024-01-01": {"horizons": {"63d": {"ic_spearman": "0.10", "q5_minus_q1": "0.05", "monotonic": True, "bucket_metrics": {"top_minus_bottom": "0.04", "monotonic": True}}}},
            "2024-02-01": {"horizons": {"63d": {"ic_spearman": "0.20", "q5_minus_q1": "0.10", "monotonic": True, "bucket_metrics": {"top_minus_bottom": "0.08", "monotonic": True}}}},
        }
        result = aggregate_metrics(all_period, horizons=["63d"])
        assert "63d" in result
        assert result["63d"]["display_name"] == "3m"
        assert Decimal(result["63d"]["ic_mean"]) == Decimal("0.15")


class TestFullSuite:
    def test_run_suite(self):
        def mock_provider(ticker, start, end):
            return "0.05"
        
        snapshots = [{
            "as_of_date": "2024-01-01",
            "ranked_securities": [
                {"ticker": f"T{i}", "composite_score": str(50 + i), "composite_rank": i,
                 "stage_bucket": "mid", "market_cap_bucket": "small"}
                for i in range(1, 21)
            ],
            "provenance": {"ruleset_version": "1.1.0"},
        }]
        
        result = run_metrics_suite(snapshots, mock_provider, "test_run", horizons=["63d"])
        assert result["run_id"] == "test_run"
        assert "63d" in result["horizons"]
        assert "3m" in result["horizons_display"]


class TestAttributionFrame:
    def test_structure(self):
        def mock_provider(ticker, start, end):
            return "0.10"
        
        snapshot = {
            "as_of_date": "2024-07-01",
            "ranked_securities": [
                {"ticker": "TEST", "composite_score": "75", "composite_rank": 1,
                 "stage_bucket": "late", "market_cap_bucket": "mid", "severity": "none",
                 "uncertainty_penalty": "0.1", "flags": ["flag_a"]}
            ],
        }
        
        frame = generate_attribution_frame(snapshot, mock_provider, "3m")  # Using display format
        assert len(frame) == 1
        assert frame[0]["ticker"] == "TEST"
        assert frame[0]["forward_return"] == "0.10"
        assert frame[0]["bucket_type"] == "tercile"  # N=1 → tercile


class TestDeterminism:
    def test_same_inputs_same_ic(self):
        scores = [Decimal(str(i * 10)) for i in range(1, 15)]
        returns = [Decimal(str(i * 0.01)) for i in range(1, 15)]
        ic1 = compute_spearman_ic(scores, returns)
        ic2 = compute_spearman_ic(scores, returns)
        assert ic1 == ic2
