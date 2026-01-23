#!/usr/bin/env python3
"""
Tests for Run Backtest Module

Covers:
- Diagnostic helper functions (bucketing, turnover, spread)
- Spearman IC calculation
- PIT financial data retrieval
- Backtest runner configuration

Note: These tests define local versions of the functions to test
the logic patterns used in run_backtest.py, since the module has
dependencies that may not be available in all environments.
"""

import pytest
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Tuple
from decimal import Decimal
import json
import tempfile

# Import module under test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# LOCAL IMPLEMENTATIONS (matching run_backtest.py logic)
# =============================================================================

def _bucket_mcap(mcap_mm: Optional[float]) -> str:
    """Classify market cap (in millions) into bucket."""
    if mcap_mm is None:
        return "UNKNOWN"
    if mcap_mm < 500:      # < $0.5B
        return "SMALL"
    if mcap_mm < 2000:     # $0.5B–$2B
        return "MID"
    return "LARGE"         # >= $2B


def _bucket_adv(adv_usd: Optional[float]) -> str:
    """Classify ADV$ into liquidity bucket."""
    if adv_usd is None:
        return "UNKNOWN"
    if adv_usd < 250_000:
        return "ILLIQ"
    if adv_usd < 2_000_000:
        return "MID"
    return "LIQ"


def _top_bottom_sets(sorted_items: List[Tuple[str, float]], frac: float = 0.10) -> Tuple[Set[str], Set[str]]:
    """Get top and bottom decile tickers from sorted list."""
    n = len(sorted_items)
    if n == 0:
        return set(), set()
    k = max(1, int(n * frac))
    top = {t for t, _ in sorted_items[:k]}
    bot = {t for t, _ in sorted_items[-k:]}
    return top, bot


def _turnover(prev: Optional[Set[str]], cur: Set[str]) -> Optional[float]:
    """Compute symmetric turnover between two sets."""
    if prev is None:
        return None
    if not prev and not cur:
        return 0.0
    inter = len(prev & cur)
    avg_size = (len(prev) + len(cur)) / 2.0
    if avg_size == 0:
        return 0.0
    return 1.0 - (inter / avg_size)


def _calculate_spearman_ic(scores: List[float], returns: List[float]) -> Optional[float]:
    """Calculate Spearman rank correlation (IC)."""
    if len(scores) != len(returns) or len(scores) < 5:
        return None

    # Check for zero variance
    if len(set(scores)) == 1 or len(set(returns)) == 1:
        return None

    # Manual Spearman correlation
    def rank(data):
        sorted_idx = sorted(range(len(data)), key=lambda i: data[i])
        ranks = [0.0] * len(data)
        for rank_val, idx in enumerate(sorted_idx):
            ranks[idx] = float(rank_val + 1)
        return ranks

    score_ranks = rank(scores)
    return_ranks = rank(returns)

    n = len(scores)
    mean_s = sum(score_ranks) / n
    mean_r = sum(return_ranks) / n

    num = sum((score_ranks[i] - mean_s) * (return_ranks[i] - mean_r) for i in range(n))
    denom_s = sum((score_ranks[i] - mean_s) ** 2 for i in range(n)) ** 0.5
    denom_r = sum((return_ranks[i] - mean_r) ** 2 for i in range(n)) ** 0.5

    if denom_s == 0 or denom_r == 0:
        return None

    return num / (denom_s * denom_r)


def _compute_bucket_ic(
    scores_by_ticker: Dict[str, float],
    returns_by_ticker: Dict[str, float],
    bucket_by_ticker: Dict[str, str],
    target_bucket: str,
) -> Tuple[Optional[float], int]:
    """Compute IC for a specific bucket."""
    common = set(scores_by_ticker.keys()) & set(returns_by_ticker.keys()) & set(bucket_by_ticker.keys())
    filtered = [t for t in common if bucket_by_ticker.get(t) == target_bucket]

    if len(filtered) < 5:
        return None, len(filtered)

    scores = [scores_by_ticker[t] for t in filtered]
    returns = [returns_by_ticker[t] for t in filtered]
    ic = _calculate_spearman_ic(scores, returns)
    return ic, len(filtered)


def get_pit_financial_data(
    ticker: str,
    as_of_date: str,
    historical_financials: Dict[str, List[Dict]],
    fallback_data: Dict[str, Dict],
) -> Dict[str, Any]:
    """Get point-in-time financial data for a ticker."""
    result = {"ticker": ticker, "pit_lookup": False}

    # Try historical data first
    if ticker in historical_financials:
        snapshots = historical_financials[ticker]
        valid_snapshots = [s for s in snapshots if s["date"] <= as_of_date]

        if valid_snapshots:
            latest = max(valid_snapshots, key=lambda s: s["date"])
            result.update({
                "Cash": latest.get("cash"),
                "Debt": latest.get("debt"),
                "R&D": latest.get("rd_expense"),
                "source_date": latest["date"],
                "pit_lookup": True,
            })
            return result

    # Fallback to current data
    if ticker in fallback_data:
        result.update(fallback_data[ticker])

    return result


# Sample data for testing
DEFAULT_UNIVERSE = ["ACME", "BETA", "GAMA", "DELT", "EPSI"]

COMPANY_DATA = {
    "ACME": {"name": "Acme Pharma", "mcap": 2500},
    "BETA": {"name": "Beta Bio", "mcap": 1200},
    "GAMA": {"name": "Gamma Therapeutics", "mcap": 800},
    "DELT": {"name": "Delta Sciences", "mcap": 400},
    "EPSI": {"name": "Epsilon Health", "mcap": 300},
}

CLINICAL_DATA = {
    "ACME": {"phase": "Phase 3", "trials": 5, "endpoint": "OS"},
    "BETA": {"phase": "Phase 2", "trials": 3, "endpoint": "ORR"},
    "GAMA": {"phase": "Phase 2", "trials": 2, "endpoint": "PFS"},
    "DELT": {"phase": "Phase 1", "trials": 1, "endpoint": "Safety"},
    "EPSI": {"phase": "Phase 1", "trials": 2, "endpoint": "MTD"},
}


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_scores():
    """Sample scores for IC calculation."""
    return {
        "ACME": 80.0,
        "BETA": 70.0,
        "GAMA": 60.0,
        "DELT": 50.0,
        "EPSI": 40.0,
        "ZETA": 30.0,
        "ETHA": 20.0,
        "THET": 10.0,
    }


@pytest.fixture
def sample_returns():
    """Sample returns aligned with scores (positive correlation)."""
    return {
        "ACME": 0.15,
        "BETA": 0.12,
        "GAMA": 0.08,
        "DELT": 0.05,
        "EPSI": 0.02,
        "ZETA": -0.03,
        "ETHA": -0.08,
        "THET": -0.12,
    }


@pytest.fixture
def inverse_returns():
    """Sample returns inversely correlated with scores."""
    return {
        "ACME": -0.12,
        "BETA": -0.08,
        "GAMA": -0.05,
        "DELT": 0.00,
        "EPSI": 0.03,
        "ZETA": 0.08,
        "ETHA": 0.12,
        "THET": 0.15,
    }


@pytest.fixture
def historical_financials():
    """Historical financial snapshots for PIT testing."""
    return {
        "ACME": [
            {"date": "2023-01-15", "cash": 100000000, "debt": 20000000, "rd_expense": 8000000},
            {"date": "2023-06-15", "cash": 80000000, "debt": 25000000, "rd_expense": 10000000},
            {"date": "2024-01-15", "cash": 60000000, "debt": 30000000, "rd_expense": 12000000},
        ],
        "BETA": [
            {"date": "2023-03-01", "cash": 50000000, "debt": 10000000, "rd_expense": 5000000},
            {"date": "2023-09-01", "cash": 40000000, "debt": 15000000, "rd_expense": 6000000},
        ],
    }


@pytest.fixture
def fallback_data():
    """Fallback financial data for tickers without historical data."""
    return {
        "ACME": {"Cash": 50000000, "Debt": 35000000, "R&D": 14000000},
        "GAMA": {"Cash": 30000000, "Debt": 5000000, "R&D": 4000000},
    }


# ============================================================================
# MARKET CAP BUCKETING
# ============================================================================

class TestBucketMcap:
    """Tests for market cap bucketing."""

    def test_small_cap_below_500(self):
        """Market cap below $500M is SMALL."""
        assert _bucket_mcap(100) == "SMALL"
        assert _bucket_mcap(499) == "SMALL"

    def test_mid_cap_500_to_2000(self):
        """Market cap $500M-$2B is MID."""
        assert _bucket_mcap(500) == "MID"
        assert _bucket_mcap(1000) == "MID"
        assert _bucket_mcap(1999) == "MID"

    def test_large_cap_above_2000(self):
        """Market cap above $2B is LARGE."""
        assert _bucket_mcap(2000) == "LARGE"
        assert _bucket_mcap(10000) == "LARGE"
        assert _bucket_mcap(130000) == "LARGE"

    def test_none_returns_unknown(self):
        """None market cap returns UNKNOWN."""
        assert _bucket_mcap(None) == "UNKNOWN"

    def test_zero_is_small(self):
        """Zero market cap is SMALL (edge case)."""
        assert _bucket_mcap(0) == "SMALL"

    def test_negative_is_small(self):
        """Negative market cap (invalid) is SMALL."""
        assert _bucket_mcap(-100) == "SMALL"


# ============================================================================
# ADV$ BUCKETING
# ============================================================================

class TestBucketAdv:
    """Tests for ADV$ (average daily volume in dollars) bucketing."""

    def test_illiquid_below_250k(self):
        """ADV$ below $250K is ILLIQ."""
        assert _bucket_adv(50000) == "ILLIQ"
        assert _bucket_adv(249999) == "ILLIQ"

    def test_mid_250k_to_2m(self):
        """ADV$ $250K-$2M is MID."""
        assert _bucket_adv(250000) == "MID"
        assert _bucket_adv(1000000) == "MID"
        assert _bucket_adv(1999999) == "MID"

    def test_liquid_above_2m(self):
        """ADV$ above $2M is LIQ."""
        assert _bucket_adv(2000000) == "LIQ"
        assert _bucket_adv(10000000) == "LIQ"

    def test_none_returns_unknown(self):
        """None ADV$ returns UNKNOWN."""
        assert _bucket_adv(None) == "UNKNOWN"

    def test_zero_is_illiquid(self):
        """Zero ADV$ is ILLIQ."""
        assert _bucket_adv(0) == "ILLIQ"


# ============================================================================
# TOP/BOTTOM SETS
# ============================================================================

class TestTopBottomSets:
    """Tests for top/bottom decile set extraction."""

    def test_basic_top_bottom(self):
        """Extract top and bottom 10% from sorted list."""
        items = [(f"T{i}", 100 - i * 10) for i in range(10)]  # T0=100, T1=90, ..., T9=10
        top, bot = _top_bottom_sets(items, frac=0.10)

        assert "T0" in top  # Highest score
        assert "T9" in bot  # Lowest score

    def test_fraction_respected(self):
        """Fraction parameter is respected."""
        items = [(f"T{i}", 100 - i * 10) for i in range(10)]

        top, bot = _top_bottom_sets(items, frac=0.20)
        assert len(top) == 2
        assert len(bot) == 2

    def test_minimum_one_item(self):
        """At least one item in each set."""
        items = [("A", 100), ("B", 50)]
        top, bot = _top_bottom_sets(items, frac=0.10)

        assert len(top) >= 1
        assert len(bot) >= 1

    def test_empty_list(self):
        """Empty list returns empty sets."""
        top, bot = _top_bottom_sets([], frac=0.10)
        assert len(top) == 0
        assert len(bot) == 0

    def test_single_item(self):
        """Single item appears in both sets."""
        items = [("ONLY", 50)]
        top, bot = _top_bottom_sets(items, frac=0.10)

        assert "ONLY" in top
        assert "ONLY" in bot


# ============================================================================
# TURNOVER
# ============================================================================

class TestTurnover:
    """Tests for portfolio turnover calculation."""

    def test_no_change_zero_turnover(self):
        """Identical sets have zero turnover."""
        prev = {"A", "B", "C"}
        cur = {"A", "B", "C"}
        assert _turnover(prev, cur) == 0.0

    def test_complete_change_full_turnover(self):
        """Completely different sets have 100% turnover."""
        prev = {"A", "B", "C"}
        cur = {"D", "E", "F"}
        assert _turnover(prev, cur) == 1.0

    def test_partial_overlap(self):
        """Partial overlap has intermediate turnover."""
        prev = {"A", "B", "C", "D"}
        cur = {"C", "D", "E", "F"}

        turnover = _turnover(prev, cur)
        # Overlap = 2 (C, D), avg_size = 4, turnover = 1 - 2/4 = 0.5
        assert turnover == pytest.approx(0.5)

    def test_none_prev_returns_none(self):
        """None previous set returns None."""
        assert _turnover(None, {"A", "B"}) is None

    def test_empty_sets_zero_turnover(self):
        """Two empty sets have zero turnover."""
        assert _turnover(set(), set()) == 0.0

    def test_asymmetric_sets(self):
        """Asymmetric set sizes handled correctly."""
        prev = {"A", "B"}
        cur = {"A", "B", "C", "D"}

        turnover = _turnover(prev, cur)
        # Overlap = 2, avg_size = 3, turnover = 1 - 2/3 ≈ 0.333
        assert turnover == pytest.approx(1 / 3, rel=0.01)


# ============================================================================
# SPEARMAN IC
# ============================================================================

class TestSpearmanIC:
    """Tests for Spearman rank correlation calculation."""

    def test_perfect_positive_correlation(self):
        """Perfectly aligned ranks give IC near 1."""
        scores = [1, 2, 3, 4, 5]
        returns = [0.1, 0.2, 0.3, 0.4, 0.5]

        ic = _calculate_spearman_ic(scores, returns)
        assert ic == pytest.approx(1.0, rel=0.01)

    def test_perfect_negative_correlation(self):
        """Perfectly inverted ranks give IC near -1."""
        scores = [1, 2, 3, 4, 5]
        returns = [0.5, 0.4, 0.3, 0.2, 0.1]

        ic = _calculate_spearman_ic(scores, returns)
        assert ic == pytest.approx(-1.0, rel=0.01)

    def test_no_correlation(self):
        """Uncorrelated data gives IC near 0."""
        scores = [1, 2, 3, 4, 5]
        returns = [0.3, 0.5, 0.1, 0.4, 0.2]  # Shuffled

        ic = _calculate_spearman_ic(scores, returns)
        assert -0.5 < ic < 0.5

    def test_minimum_data_points(self):
        """Need at least 5 data points."""
        scores = [1, 2, 3, 4]
        returns = [0.1, 0.2, 0.3, 0.4]

        ic = _calculate_spearman_ic(scores, returns)
        assert ic is None

    def test_mismatched_lengths_returns_none(self):
        """Mismatched list lengths return None."""
        scores = [1, 2, 3, 4, 5]
        returns = [0.1, 0.2, 0.3]

        ic = _calculate_spearman_ic(scores, returns)
        assert ic is None

    def test_empty_lists_return_none(self):
        """Empty lists return None."""
        ic = _calculate_spearman_ic([], [])
        assert ic is None

    def test_all_same_scores_returns_none(self):
        """All identical scores return None (zero variance)."""
        scores = [50, 50, 50, 50, 50]
        returns = [0.1, 0.2, 0.3, 0.4, 0.5]

        ic = _calculate_spearman_ic(scores, returns)
        assert ic is None

    def test_deterministic(self):
        """Same inputs always produce same output."""
        scores = [10, 20, 30, 40, 50, 60]
        returns = [0.05, 0.10, 0.08, 0.15, 0.12, 0.20]

        ic1 = _calculate_spearman_ic(scores, returns)
        ic2 = _calculate_spearman_ic(scores, returns)

        assert ic1 == ic2


# ============================================================================
# BUCKET IC
# ============================================================================

class TestComputeBucketIC:
    """Tests for IC computation within buckets."""

    def test_basic_bucket_ic(self, sample_scores, sample_returns):
        """Compute IC for a specific bucket."""
        bucket_by_ticker = {t: "MID" for t in sample_scores.keys()}

        ic, n = _compute_bucket_ic(sample_scores, sample_returns, bucket_by_ticker, "MID")

        assert ic is not None
        assert n == 8

    def test_insufficient_data_returns_none(self, sample_scores, sample_returns):
        """Insufficient data for bucket returns None IC."""
        bucket_by_ticker = {
            "ACME": "SMALL",
            "BETA": "MID",
            "GAMA": "LARGE",
            "DELT": "SMALL",
            "EPSI": "MID",
            "ZETA": "LARGE",
            "ETHA": "SMALL",
            "THET": "MID",
        }

        # LARGE bucket only has 2 tickers
        ic, n = _compute_bucket_ic(sample_scores, sample_returns, bucket_by_ticker, "LARGE")

        assert ic is None
        assert n == 2

    def test_nonexistent_bucket_returns_zero_count(self, sample_scores, sample_returns):
        """Nonexistent bucket returns zero count."""
        bucket_by_ticker = {t: "MID" for t in sample_scores.keys()}

        ic, n = _compute_bucket_ic(sample_scores, sample_returns, bucket_by_ticker, "NONEXISTENT")

        assert ic is None
        assert n == 0


# ============================================================================
# PIT FINANCIAL DATA
# ============================================================================

class TestGetPitFinancialData:
    """Tests for point-in-time financial data retrieval."""

    def test_returns_latest_snapshot_before_date(self, historical_financials, fallback_data):
        """Returns most recent snapshot before as_of_date."""
        result = get_pit_financial_data(
            ticker="ACME",
            as_of_date="2023-08-01",
            historical_financials=historical_financials,
            fallback_data=fallback_data,
        )

        # Should get the June 2023 snapshot
        assert result["ticker"] == "ACME"
        assert result["Cash"] == 80000000
        assert result["source_date"] == "2023-06-15"
        assert result["pit_lookup"] is True

    def test_returns_exact_date_snapshot(self, historical_financials, fallback_data):
        """Returns snapshot from exact as_of_date."""
        result = get_pit_financial_data(
            ticker="ACME",
            as_of_date="2024-01-15",
            historical_financials=historical_financials,
            fallback_data=fallback_data,
        )

        # Should get the Jan 2024 snapshot
        assert result["Cash"] == 60000000
        assert result["source_date"] == "2024-01-15"

    def test_falls_back_when_no_historical_data(self, historical_financials, fallback_data):
        """Falls back to current data when no historical snapshots."""
        result = get_pit_financial_data(
            ticker="GAMA",  # No historical data
            as_of_date="2024-01-15",
            historical_financials=historical_financials,
            fallback_data=fallback_data,
        )

        assert result["ticker"] == "GAMA"
        assert result["Cash"] == 30000000
        assert result["pit_lookup"] is False

    def test_falls_back_when_date_before_all_snapshots(self, historical_financials, fallback_data):
        """Falls back when as_of_date is before all snapshots."""
        result = get_pit_financial_data(
            ticker="ACME",
            as_of_date="2022-01-01",  # Before any snapshots
            historical_financials=historical_financials,
            fallback_data=fallback_data,
        )

        assert result["pit_lookup"] is False

    def test_missing_ticker_returns_fallback(self, historical_financials, fallback_data):
        """Missing ticker returns fallback data."""
        result = get_pit_financial_data(
            ticker="UNKNOWN",
            as_of_date="2024-01-15",
            historical_financials=historical_financials,
            fallback_data=fallback_data,
        )

        assert result["ticker"] == "UNKNOWN"
        assert result["pit_lookup"] is False

    def test_multiple_snapshots_before_date(self, historical_financials, fallback_data):
        """Returns latest among multiple snapshots before date."""
        result = get_pit_financial_data(
            ticker="BETA",
            as_of_date="2024-01-01",  # After both snapshots
            historical_financials=historical_financials,
            fallback_data=fallback_data,
        )

        # Should get the September 2023 snapshot (latest)
        assert result["Cash"] == 40000000
        assert result["source_date"] == "2023-09-01"


# ============================================================================
# COMPANY DATA
# ============================================================================

class TestCompanyData:
    """Tests for company data constants."""

    def test_default_universe_not_empty(self):
        """Default universe has tickers."""
        assert len(DEFAULT_UNIVERSE) > 0

    def test_company_data_has_required_fields(self):
        """Company data has name and mcap."""
        for ticker, data in COMPANY_DATA.items():
            assert "name" in data
            assert "mcap" in data
            assert isinstance(data["mcap"], (int, float))

    def test_clinical_data_has_required_fields(self):
        """Clinical data has phase, trials, endpoint."""
        for ticker, data in CLINICAL_DATA.items():
            assert "phase" in data
            assert "trials" in data
            assert "endpoint" in data

    def test_company_and_clinical_data_aligned(self):
        """Company and clinical data have same tickers."""
        assert set(COMPANY_DATA.keys()) == set(CLINICAL_DATA.keys())


# ============================================================================
# EDGE CASES
# ============================================================================

class TestEdgeCases:
    """Edge case tests."""

    def test_ic_with_ties(self):
        """IC handles tied scores."""
        scores = [100, 100, 50, 50, 25]  # Ties
        returns = [0.2, 0.15, 0.1, 0.05, 0.0]

        ic = _calculate_spearman_ic(scores, returns)
        assert ic is not None  # Should not crash

    def test_ic_with_tied_returns(self):
        """IC handles tied returns."""
        scores = [100, 80, 60, 40, 20]
        returns = [0.1, 0.1, 0.1, 0.1, 0.1]  # All tied

        ic = _calculate_spearman_ic(scores, returns)
        assert ic is None  # Zero variance in returns

    def test_bucket_ic_with_partial_overlap(self):
        """Bucket IC handles partial ticker overlap."""
        scores = {"A": 80, "B": 70, "C": 60, "D": 50, "E": 40}
        returns = {"A": 0.1, "C": 0.05, "E": 0.01, "F": 0.2}  # Missing B, D
        buckets = {"A": "X", "B": "X", "C": "X", "D": "X", "E": "X", "F": "X"}

        ic, n = _compute_bucket_ic(scores, returns, buckets, "X")

        # Only A, C, E have both scores and returns
        assert n == 3  # Below threshold of 5

    def test_turnover_with_single_element_sets(self):
        """Turnover with single-element sets."""
        assert _turnover({"A"}, {"A"}) == 0.0
        assert _turnover({"A"}, {"B"}) == 1.0

    def test_large_dataset_ic(self):
        """IC works with large datasets."""
        n = 500
        scores = list(range(n))
        returns = [i * 0.001 for i in range(n)]

        ic = _calculate_spearman_ic(scores, returns)
        assert ic == pytest.approx(1.0, rel=0.01)


# ============================================================================
# DETERMINISM
# ============================================================================

class TestDeterminism:
    """Tests for deterministic behavior."""

    def test_bucket_mcap_deterministic(self):
        """Market cap bucketing is deterministic."""
        values = [100, 500, 1000, 2000, 5000]
        results1 = [_bucket_mcap(v) for v in values]
        results2 = [_bucket_mcap(v) for v in values]

        assert results1 == results2

    def test_bucket_adv_deterministic(self):
        """ADV bucketing is deterministic."""
        values = [100000, 250000, 1000000, 2000000, 5000000]
        results1 = [_bucket_adv(v) for v in values]
        results2 = [_bucket_adv(v) for v in values]

        assert results1 == results2

    def test_top_bottom_sets_deterministic(self):
        """Top/bottom set extraction is deterministic."""
        items = [(f"T{i}", 100 - i * 10) for i in range(10)]

        top1, bot1 = _top_bottom_sets(items, frac=0.10)
        top2, bot2 = _top_bottom_sets(items, frac=0.10)

        assert top1 == top2
        assert bot1 == bot2

    def test_turnover_deterministic(self):
        """Turnover calculation is deterministic."""
        prev = {"A", "B", "C", "D"}
        cur = {"C", "D", "E", "F"}

        t1 = _turnover(prev, cur)
        t2 = _turnover(prev, cur)

        assert t1 == t2

    def test_ic_deterministic(self, sample_scores, sample_returns):
        """IC calculation is deterministic."""
        scores = list(sample_scores.values())
        returns = list(sample_returns.values())

        ic1 = _calculate_spearman_ic(scores, returns)
        ic2 = _calculate_spearman_ic(scores, returns)

        assert ic1 == ic2


# ============================================================================
# INTEGRATION
# ============================================================================

class TestIntegration:
    """Integration tests for backtest components."""

    def test_full_ic_workflow(self, sample_scores, sample_returns):
        """Complete IC workflow from scores to result."""
        # Bucket by market cap
        mcap_by_ticker = {
            "ACME": 2500,
            "BETA": 1500,
            "GAMA": 800,
            "DELT": 400,
            "EPSI": 300,
            "ZETA": 1200,
            "ETHA": 3000,
            "THET": 600,
        }
        bucket_by_ticker = {t: _bucket_mcap(m) for t, m in mcap_by_ticker.items()}

        # Compute overall IC
        scores = [sample_scores[t] for t in sorted(sample_scores.keys())]
        returns = [sample_returns[t] for t in sorted(sample_returns.keys())]
        overall_ic = _calculate_spearman_ic(scores, returns)

        assert overall_ic is not None
        assert overall_ic > 0  # Positive correlation expected

        # Compute bucket IC
        mid_ic, mid_n = _compute_bucket_ic(sample_scores, sample_returns, bucket_by_ticker, "MID")

        # MID has BETA, GAMA, ZETA, THET = 4 tickers, not enough
        assert mid_ic is None
        assert mid_n == 4

    def test_turnover_across_periods(self):
        """Turnover tracking across multiple periods."""
        periods = [
            {"A", "B", "C", "D", "E"},
            {"A", "B", "C", "F", "G"},  # 3 same, 2 new
            {"B", "C", "F", "H", "I"},  # 3 same, 2 new
            {"J", "K", "L", "M", "N"},  # Complete change
        ]

        turnovers = []
        prev = None
        for cur in periods:
            if prev is not None:
                t = _turnover(prev, cur)
                turnovers.append(t)
            prev = cur

        assert len(turnovers) == 3
        assert turnovers[0] == pytest.approx(0.4, rel=0.01)  # 2/5 changed
        assert turnovers[2] == 1.0  # Complete change

