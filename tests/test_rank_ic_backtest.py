"""Tests for archive-mode helpers in run_rank_ic_backtest.py."""
from __future__ import annotations

import pytest
import sys
from pathlib import Path

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from run_rank_ic_backtest import assert_rank_unique, signal_to_rankings


# ── assert_rank_unique ──────────────────────────────────────────────────────

class TestAssertRankUnique:
    def test_unique_ranks_pass(self):
        rankings = {"A": 1, "B": 2, "C": 3}
        result = assert_rank_unique(rankings, "2026-01-01", {})
        assert result is True

    def test_duplicate_ranks_fail(self):
        rankings = {"A": 1, "B": 1, "C": 3}
        with pytest.raises(ValueError, match="Rank ties"):
            assert_rank_unique(rankings, "2026-01-01", {})

    def test_duplicate_ranks_allowed_with_flag(self):
        rankings = {"A": 1, "B": 1, "C": 3}
        result = assert_rank_unique(rankings, "2026-01-01", {}, allow_ties=True)
        assert result is True

    def test_empty_rankings(self):
        result = assert_rank_unique({}, "2026-01-01", {})
        assert result is True

    def test_single_entry(self):
        result = assert_rank_unique({"A": 1}, "2026-01-01", {})
        assert result is True


# ── signal_to_rankings ──────────────────────────────────────────────────────

class TestSignalToRankings:
    def test_higher_is_better(self):
        signals = {"A": 0.9, "B": 0.5, "C": 0.1}
        result = signal_to_rankings(signals, higher_is_better=True)
        assert result == {"A": 1, "B": 2, "C": 3}

    def test_lower_is_better(self):
        signals = {"A": 0.9, "B": 0.5, "C": 0.1}
        result = signal_to_rankings(signals, higher_is_better=False)
        assert result == {"C": 1, "B": 2, "A": 3}

    def test_dense_ranking(self):
        """Ranks should be dense 1..N with no gaps."""
        signals = {"X": 10.0, "Y": 20.0, "Z": 15.0}
        result = signal_to_rankings(signals, higher_is_better=True)
        assert sorted(result.values()) == [1, 2, 3]
        assert result["Y"] == 1
        assert result["Z"] == 2
        assert result["X"] == 3

    def test_single_ticker(self):
        result = signal_to_rankings({"A": 0.5}, higher_is_better=True)
        assert result == {"A": 1}

    def test_identical_signals_deterministic(self):
        """With identical values, output should still assign unique dense ranks."""
        signals = {"A": 0.5, "B": 0.5, "C": 0.5}
        result = signal_to_rankings(signals, higher_is_better=True)
        assert sorted(result.values()) == [1, 2, 3]

    def test_score_rank_pct_lower_is_better(self):
        """score_rank_pct: lower = better rank. Use higher_is_better=False."""
        signals = {"A": 0.01, "B": 0.50, "C": 0.99}
        result = signal_to_rankings(signals, higher_is_better=False)
        assert result["A"] == 1  # best
        assert result["C"] == 3  # worst
