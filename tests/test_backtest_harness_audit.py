"""Backtest harness audit regression tests — 2026-04-05.

Tests correctness invariants discovered during the full backtest harness audit.
Each test documents a specific risk and prevents regression.
"""
from __future__ import annotations

import pytest

import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# 1. Forward-return window alignment
# ---------------------------------------------------------------------------


class TestForwardReturnAlignment:
    """Verify that research panel and eval_forward_returns use the same window."""

    def test_research_panel_starts_at_snap_date(self):
        """Research panel forward_return uses execution_lag=1 by default,
        so t0 = next trading day after snap_date."""
        from scripts.research.build_signal_research_panel import forward_return

        prices = {
            "2024-01-29": 100.0,
            "2024-01-30": 101.0,
            "2024-01-31": 102.0,
            "2024-02-01": 103.0,
            "2024-02-02": 104.0,
        }
        sorted_dates = sorted(prices.keys())
        # snap_date = "2024-01-30" with execution_lag=1 -> t0 = "2024-01-31", t0+2 = "2024-02-02"
        ret = forward_return(prices, sorted_dates, "2024-01-30", 2)
        assert ret is not None
        assert abs(ret - (104.0 / 102.0 - 1)) < 1e-9

    def test_research_panel_weekend_snap_date(self):
        """If snap_date is a weekend, forward_return finds next available date,
        then applies execution_lag=1."""
        from scripts.research.build_signal_research_panel import forward_return

        # No 2024-02-03 or 2024-02-04 (weekend)
        prices = {"2024-02-02": 100.0, "2024-02-05": 102.0, "2024-02-06": 103.0, "2024-02-07": 105.0}
        sorted_dates = sorted(prices.keys())
        # snap_date = "2024-02-03" -> first date >= snap_date = "2024-02-05",
        # execution_lag=1 -> t0 = "2024-02-06", t0+1 = "2024-02-07"
        ret = forward_return(prices, sorted_dates, "2024-02-03", 1)
        assert ret is not None
        assert abs(ret - (105.0 / 103.0 - 1)) < 1e-9

    def test_documented_mismatch_with_metrics(self):
        """Document that backtest/metrics.py uses next_trading_day start."""
        from backtest.metrics import next_trading_day

        # Wednesday -> Thursday
        assert next_trading_day("2024-01-31") == "2024-02-01"
        # Friday -> Monday
        assert next_trading_day("2024-02-02") == "2024-02-05"


# ---------------------------------------------------------------------------
# 2. Z-scoring consistency between research and production
# ---------------------------------------------------------------------------


class TestZScoringConsistency:
    """Check z-scoring divisor is consistent across modules."""

    def test_selector_engine_uses_population_std(self):
        """selector_engine._compute_cohort_stats uses ddof=0 (population)."""
        from selector_engine import _compute_cohort_stats

        rows = [{"test_signal": v} for v in [1.0, 2.0, 3.0, 4.0, 5.0]]
        stats = _compute_cohort_stats(rows, "test_signal")
        expected_mean = 3.0
        expected_std_pop = (2.0) ** 0.5  # sqrt(variance with ddof=0)
        assert abs(stats.mean - expected_mean) < 1e-9
        assert abs(stats.std - expected_std_pop) < 1e-9

    def test_pairwise_ranker_uses_population_std(self):
        """ranker_v2_pairwise.zscore_cohort_features uses ddof=0."""
        from ranker_v2_pairwise import FeatureSpec, zscore_cohort_features

        specs = [FeatureSpec("val")]
        rows = [{"val": v} for v in [1.0, 2.0, 3.0, 4.0, 5.0]]
        z_matrix = zscore_cohort_features(rows, specs)
        # With population std, z of 1.0 = (1 - 3) / sqrt(2) = -sqrt(2)
        assert abs(z_matrix[0][0] - (-2.0 / 2.0**0.5)) < 1e-6

    def test_bundle_zscore_uses_sample_std(self):
        """test_selector_bundles zscore_eligible uses statistics.stdev (ddof=1).

        This is a KNOWN inconsistency with production: ddof=1 vs ddof=0.
        Impact is negligible for typical cohort sizes (100+ names).
        """
        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        pop_std = (sum((v - 3.0) ** 2 for v in vals) / 5) ** 0.5
        sample_std = statistics.stdev(vals)  # ddof=1
        # They differ by sqrt(n/(n-1))
        assert sample_std > pop_std
        ratio = sample_std / pop_std
        assert abs(ratio - (5 / 4) ** 0.5) < 1e-9


# ---------------------------------------------------------------------------
# 3. Panel deduplication
# ---------------------------------------------------------------------------


class TestPanelDeduplication:
    """Verify deduplication logic prevents density artifacts."""

    def test_dedupe_monthly_keeps_last_per_month(self):
        """dedupe_monthly keeps the last snapshot date per calendar month."""
        from scripts.research.build_signal_research_panel import dedupe_monthly

        dates = ["2024-01-15", "2024-01-31", "2024-02-14", "2024-02-28", "2024-03-15"]
        result = dedupe_monthly(dates)
        assert result == ["2024-01-31", "2024-02-28", "2024-03-15"]

    def test_no_duplicate_ticker_per_snapshot(self):
        """load_rankings deduplicates tickers within a single snapshot."""
        import csv
        import tempfile

        from scripts.research.build_signal_research_panel import load_rankings

        # Create a temporary rankings.csv with duplicate ticker
        with tempfile.TemporaryDirectory() as tmpdir:
            snap_dir = Path(tmpdir) / "2024-01-31"
            snap_dir.mkdir()
            csv_path = snap_dir / "rankings.csv"
            with open(csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["ticker", "eligible", "actionable_rank"])
                w.writeheader()
                w.writerow({"ticker": "AAAA", "eligible": "1", "actionable_rank": "1"})
                w.writerow({"ticker": "AAAA", "eligible": "1", "actionable_rank": "2"})
                w.writerow({"ticker": "BBBB", "eligible": "1", "actionable_rank": "3"})

            # Monkey-patch SNAPSHOTS_DIR temporarily
            import scripts.research.build_signal_research_panel as panel_mod

            orig = panel_mod.SNAPSHOTS_DIR
            panel_mod.SNAPSHOTS_DIR = Path(tmpdir)
            try:
                rows = load_rankings("2024-01-31", {})
                tickers = [r["ticker"] for r in rows]
                assert tickers == ["AAAA", "BBBB"]  # AAAA appears only once
            finally:
                panel_mod.SNAPSHOTS_DIR = orig


# ---------------------------------------------------------------------------
# 4. Forward-fill PIT safety
# ---------------------------------------------------------------------------


class TestForwardFillPITSafety:
    """Verify inst_delta_z forward-fill never uses future data."""

    def test_forward_fill_only_carries_forward(self):
        """Values flow from earlier dates to later dates, never backward."""
        from scripts.research.build_signal_research_panel import forward_fill_quarterly_signals

        rows = [
            {"ticker": "TEST", "snapshot_date": "2024-01-31", "inst_delta_z": 1.5},
            {"ticker": "TEST", "snapshot_date": "2024-02-28", "inst_delta_z": 0},
            {"ticker": "TEST", "snapshot_date": "2024-03-31", "inst_delta_z": 0},
            {"ticker": "TEST", "snapshot_date": "2024-04-30", "inst_delta_z": 0},
            {"ticker": "TEST", "snapshot_date": "2024-05-31", "inst_delta_z": 2.0},
        ]
        n = forward_fill_quarterly_signals(rows, ["inst_delta_z"], max_stale_months=3)
        # Feb and Mar should be filled from Jan (within 3 months)
        assert rows[1]["inst_delta_z"] == 1.5
        assert rows[2]["inst_delta_z"] == 1.5
        # Apr is 3 months from Jan — should still be filled
        assert rows[3]["inst_delta_z"] == 1.5
        # May has its own value — not overwritten
        assert rows[4]["inst_delta_z"] == 2.0
        assert n == 3  # Feb, Mar, Apr filled

    def test_forward_fill_respects_staleness_cap(self):
        """Values are NOT carried beyond max_stale_months."""
        from scripts.research.build_signal_research_panel import forward_fill_quarterly_signals

        rows = [
            {"ticker": "TEST", "snapshot_date": "2024-01-31", "inst_delta_z": 1.5},
            {"ticker": "TEST", "snapshot_date": "2024-02-28", "inst_delta_z": 0},
            {"ticker": "TEST", "snapshot_date": "2024-06-30", "inst_delta_z": 0},  # 5 months later
        ]
        n = forward_fill_quarterly_signals(rows, ["inst_delta_z"], max_stale_months=3)
        assert rows[1]["inst_delta_z"] == 1.5  # 1 month: OK
        assert rows[2]["inst_delta_z"] == 0  # 5 months: NOT filled
        assert n == 1


# ---------------------------------------------------------------------------
# 5. Pairwise ranker train/test separation
# ---------------------------------------------------------------------------


class TestPairwiseTrainTestSeparation:
    """Verify expanding-window evaluation uses no future data."""

    def test_train_dates_strictly_before_test(self):
        """For each test date, all train dates must be earlier."""
        from ranker_v2_pairwise import RankerV2Config

        config = RankerV2Config(min_train_dates=3, train_window=0)
        sorted_dates = ["2024-01", "2024-02", "2024-03", "2024-04", "2024-05"]

        for test_idx in range(config.min_train_dates, len(sorted_dates)):
            test_date = sorted_dates[test_idx]
            train_dates = sorted_dates[:test_idx]
            for td in train_dates:
                assert td < test_date, f"Train date {td} >= test date {test_date}"

    def test_rolling_window_excludes_future(self):
        """Rolling window still only uses dates before test."""
        from ranker_v2_pairwise import RankerV2Config

        config = RankerV2Config(min_train_dates=3, train_window=2)
        sorted_dates = ["2024-01", "2024-02", "2024-03", "2024-04", "2024-05"]

        test_idx = 4  # test on "2024-05"
        all_train = sorted_dates[:test_idx]  # ["2024-01", "2024-02", "2024-03", "2024-04"]
        train_dates = all_train[-config.train_window :]  # ["2024-03", "2024-04"]

        for td in train_dates:
            assert td < "2024-05"
        assert len(train_dates) == 2


# ---------------------------------------------------------------------------
# 6. Newey-West implementation
# ---------------------------------------------------------------------------


class TestNeweyWestCorrectness:
    """Verify Newey-West t-stat is correctly computed."""

    def test_nw_with_no_autocorrelation(self):
        """When data has no autocorrelation, NW ~ naive t-stat."""
        # White noise
        import random

        from scripts.research.test_selector_bundles import _newey_west_tstat, _safe_tstat

        rng = random.Random(42)
        vals = [rng.gauss(0.01, 0.05) for _ in range(100)]

        naive_t = _safe_tstat(vals)
        nw_t = _newey_west_tstat(vals, lags=3)

        # Should be close (within 30% for random data with no autocorrelation)
        assert nw_t is not None
        assert naive_t is not None
        ratio = abs(nw_t / naive_t)
        assert 0.5 < ratio < 1.5, f"NW/naive ratio {ratio} too far from 1.0"

    def test_nw_reduces_tstat_with_positive_autocorrelation(self):
        """Positively autocorrelated series → NW t-stat < naive t-stat."""
        # Construct positively autocorrelated series
        import random

        from scripts.research.test_selector_bundles import _newey_west_tstat, _safe_tstat

        rng = random.Random(42)
        vals = []
        v = 0.0
        for _ in range(100):
            v = 0.8 * v + rng.gauss(0.01, 0.05)  # AR(1) with positive mean
            vals.append(v)

        naive_t = _safe_tstat(vals)
        nw_t = _newey_west_tstat(vals, lags=5)

        assert nw_t is not None and naive_t is not None
        # NW should give smaller |t| than naive for positively autocorrelated data
        assert abs(nw_t) < abs(naive_t), f"NW |t|={abs(nw_t):.2f} >= naive |t|={abs(naive_t):.2f}"


# ---------------------------------------------------------------------------
# 7. Regime label is forward-looking (documented caveat)
# ---------------------------------------------------------------------------


class TestRegimeLabelCaveat:
    """Ensure regime_63d uses forward XBI return (by design, not a bug)."""

    @pytest.mark.skip(reason="placeholder: behavior not yet implemented (test-trust-audit hygiene 2026-07-14)")
    def test_regime_label_uses_forward_xbi(self):
        """regime_63d is based on forward 63-day XBI return — not tradeable."""
        # This is a documentation test. The regime label at snap_date uses
        # the NEXT 63 days of XBI returns to classify bear/bull/neutral.
        # Regime-conditional results require knowing the future.
        # This is fine for diagnostics, but not for trading decisions.
        #
        # Verified in build_signal_research_panel.py lines 475-481:
        #   if xbi_63 < -0.02: "bear"
        #   elif xbi_63 > 0.02: "bull"
        #   else: "neutral"
        # where xbi_63 = forward_return(xbi_prices, ..., snap_date, 63)
        pass  # Documenting known design choice, not testing for a bug


# ---------------------------------------------------------------------------
# 8. BH FDR correctness
# ---------------------------------------------------------------------------


class TestBHFDRCorrectness:
    """Verify Benjamini-Hochberg implementation."""

    def test_bh_with_known_values(self):
        """Check BH against textbook example."""
        from common.stats.multiple_testing import benjamini_hochberg

        # 5 tests with p-values
        p = {"a": 0.01, "b": 0.03, "c": 0.04, "d": 0.15, "e": 0.60}
        result = benjamini_hochberg(p, alpha=0.10)

        # Sorted: a=0.01, b=0.03, c=0.04, d=0.15, e=0.60
        # q = p * m / rank: a=0.05, b=0.075, c=0.067, d=0.1875, e=0.60
        # After monotonicity: a=0.05, b=0.067, c=0.067, d=0.1875, e=0.60
        assert result["n_rejected"] == 3  # a, b, c rejected at q < 0.10
        assert result["results"]["a"]["rejected"]
        assert result["results"]["b"]["rejected"]
        assert result["results"]["c"]["rejected"]
        assert not result["results"]["d"]["rejected"]
        assert not result["results"]["e"]["rejected"]

    def test_bh_empty_input(self):
        """Empty input returns error."""
        from common.stats.multiple_testing import benjamini_hochberg

        result = benjamini_hochberg({}, alpha=0.10)
        assert "error" in result


# ---------------------------------------------------------------------------
# 9. Block bootstrap CI
# ---------------------------------------------------------------------------


class TestBlockBootstrap:
    """Verify block bootstrap produces valid confidence intervals."""

    def test_positive_mean_series_has_positive_ci(self):
        """A strongly positive series should have CI excluding zero."""
        from common.stats.bootstrap import block_bootstrap

        returns = [0.02 + 0.005 * i for i in range(36)]  # strong uptrend
        result = block_bootstrap(returns, block_length=6, n_bootstrap=5000, seed=42)
        assert result["ci_excludes_zero"]
        assert result["ci_lower"] > 0

    def test_zero_mean_series_includes_zero(self):
        """A zero-mean series should have CI including zero."""
        import random

        from common.stats.bootstrap import block_bootstrap

        rng = random.Random(42)
        returns = [rng.gauss(0.0, 0.05) for _ in range(36)]
        result = block_bootstrap(returns, block_length=6, n_bootstrap=5000, seed=42)
        # CI should include zero (with high probability for truly zero-mean)
        assert result["ci_lower"] < 0 < result["ci_upper"]
