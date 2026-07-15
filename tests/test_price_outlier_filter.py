#!/usr/bin/env python3
"""Tests for price outlier / split-artifact filtering in run_screen.py."""

from datetime import date

import pytest

from run_screen import SPLIT_DROP_THRESHOLD, SPLIT_JUMP_THRESHOLD, _filter_price_outliers, _validate_price_splits

# ---------------------------------------------------------------------------
# _filter_price_outliers
# ---------------------------------------------------------------------------


class TestFilterPriceOutliers:
    def test_empty_series(self):
        filtered, warns = _filter_price_outliers([])
        assert filtered == []
        assert warns == []

    def test_single_row(self):
        series = [(date(2026, 1, 1), 10.0)]
        filtered, warns = _filter_price_outliers(series)
        assert filtered == series
        assert warns == []

    def test_clean_series_no_outliers(self):
        series = [(date(2026, 1, i), 10.0 + i * 0.1) for i in range(1, 11)]
        filtered, warns = _filter_price_outliers(series)
        assert len(filtered) == 10
        assert warns == []

    def test_forward_split_detected(self):
        """A 4:1 forward split: all pre-split rows truncated."""
        series = [
            (date(2026, 1, 1), 100.0),
            (date(2026, 1, 2), 100.5),
            (date(2026, 1, 3), 20.0),  # 4:1 split → -80%, exceeds -75%
            (date(2026, 1, 4), 20.5),
        ]
        filtered, warns = _filter_price_outliers(series)
        assert len(warns) == 1
        assert warns[0]["flag"] == "forward_split"
        assert warns[0]["date"] == "2026-01-03"
        # ALL pre-split rows truncated, only post-split regime kept
        assert len(filtered) == 2
        assert filtered[0][0] == date(2026, 1, 3)
        assert filtered[1][0] == date(2026, 1, 4)

    def test_reverse_split_detected(self):
        """Reverse split: price jumps >300%; pre-split regime truncated."""
        series = [
            (date(2026, 1, 1), 2.0),
            (date(2026, 1, 2), 2.1),
            (date(2026, 1, 3), 10.5),  # 5x jump → +400%, exceeds +300%
            (date(2026, 1, 4), 10.8),
        ]
        filtered, warns = _filter_price_outliers(series)
        assert len(warns) == 1
        assert warns[0]["flag"] == "reverse_split"
        # All pre-split rows truncated
        assert len(filtered) == 2
        assert filtered[0][0] == date(2026, 1, 3)

    def test_multiple_splits(self):
        """Two splits: truncate to after the LATEST split."""
        series = [
            (date(2026, 1, 1), 100.0),
            (date(2026, 1, 2), 20.0),  # forward split 5:1 → -80%
            (date(2026, 1, 3), 20.5),
            (date(2026, 1, 4), 105.0),  # reverse split → +412%
            (date(2026, 1, 5), 106.0),
        ]
        filtered, warns = _filter_price_outliers(series)
        assert len(warns) == 2
        # Truncated to after latest split (index 3)
        assert len(filtered) == 2
        assert filtered[0][0] == date(2026, 1, 4)

    def test_threshold_boundary_not_triggered(self):
        """Moves just under thresholds should not trigger."""
        series = [
            (date(2026, 1, 1), 100.0),
            (date(2026, 1, 2), 390.0),  # +290% < +300% threshold
            (date(2026, 1, 3), 100.0),  # -74.4% > -75% threshold
        ]
        filtered, warns = _filter_price_outliers(series)
        assert len(warns) == 0
        assert len(filtered) == 3

    def test_zero_prev_close_skipped(self):
        series = [
            (date(2026, 1, 1), 0.0),
            (date(2026, 1, 2), 10.0),
        ]
        filtered, warns = _filter_price_outliers(series)
        assert len(warns) == 0

    def test_custom_thresholds(self):
        series = [
            (date(2026, 1, 1), 10.0),
            (date(2026, 1, 2), 15.0),  # +50%
        ]
        # With tight threshold, this triggers — truncates to post-split only
        filtered, warns = _filter_price_outliers(series, jump_threshold=0.4)
        assert len(warns) == 1
        assert len(filtered) == 1
        assert filtered[0][0] == date(2026, 1, 2)
        # With default threshold, it doesn't
        filtered2, warns2 = _filter_price_outliers(series)
        assert len(warns2) == 0

    def test_long_presplit_regime_truncated(self):
        """RNA-like scenario: 60 rows at old price, split, 2 rows at new price.
        All 60 pre-split rows must be truncated, not just the immediate predecessor.
        """
        # 60 days at ~$72, then 5:1 split to ~$14
        from datetime import timedelta

        base = date(2025, 11, 1)
        series = [(base + timedelta(days=i), 72.0 + i * 0.1) for i in range(60)]
        split_day = base + timedelta(days=60)
        series.append((split_day, 14.50))  # split day
        series.append((split_day + timedelta(days=1), 14.75))
        filtered, warns = _filter_price_outliers(series)
        assert len(warns) == 1
        assert warns[0]["flag"] == "forward_split"
        # Only post-split rows survive
        assert len(filtered) == 2
        assert filtered[0][1] == 14.50
        assert filtered[1][1] == 14.75


# ---------------------------------------------------------------------------
# _validate_price_splits
# ---------------------------------------------------------------------------


class TestValidatePriceSplits:
    def test_empty_dict(self):
        result = _validate_price_splits({})
        assert result == {}

    def test_clean_tickers(self):
        prices = {
            "AAPL": [(date(2026, 1, i), 150.0 + i) for i in range(1, 6)],
            "XBI": [(date(2026, 1, i), 90.0 + i * 0.5) for i in range(1, 6)],
        }
        result = _validate_price_splits(prices)
        assert result == {}

    def test_detects_split_in_one_ticker(self):
        prices = {
            "GOOD": [(date(2026, 1, i), 50.0 + i) for i in range(1, 4)],
            "BAD": [
                (date(2026, 1, 1), 100.0),
                (date(2026, 1, 2), 10.0),  # -90% → forward split
                (date(2026, 1, 3), 10.5),
            ],
        }
        result = _validate_price_splits(prices)
        assert "BAD" in result
        assert "GOOD" not in result
        assert len(result["BAD"]) == 1

    def test_multiple_tickers_with_splits(self):
        prices = {
            "A": [
                (date(2026, 1, 1), 5.0),
                (date(2026, 1, 2), 50.0),  # +900%
            ],
            "B": [
                (date(2026, 1, 1), 200.0),
                (date(2026, 1, 2), 20.0),  # -90%
            ],
        }
        result = _validate_price_splits(prices)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Integration: verify thresholds match existing constants
# ---------------------------------------------------------------------------


class TestThresholdsConsistent:
    def test_jump_threshold(self):
        assert SPLIT_JUMP_THRESHOLD == 3.0

    def test_drop_threshold(self):
        assert SPLIT_DROP_THRESHOLD == -0.75


# ---------------------------------------------------------------------------
# Defensive: ensure all yfinance callers use explicit auto_adjust=True
# ---------------------------------------------------------------------------


class TestAutoAdjustExplicit:
    """Guard against yfinance callers relying on implicit auto_adjust default."""

    @staticmethod
    def _assert_auto_adjust_explicit(func, label):
        import ast
        import inspect
        import textwrap

        src = textwrap.dedent(inspect.getsource(func))
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            callee = node.func
            name = getattr(callee, "attr", "") or getattr(callee, "id", "")
            if name not in ("download", "history"):
                continue
            kw_names = {kw.arg for kw in node.keywords}
            assert "auto_adjust" in kw_names, f"{label}: yfinance .{name}() must pass auto_adjust=True explicitly"
            return
        pytest.skip(f"{label}: no .download()/.history() call found")

    def test_fetch_prices_interactive(self):
        """Parse source file directly — it's a script with top-level side effects."""
        import ast
        from pathlib import Path

        src_path = Path(__file__).resolve().parent.parent / "fetch_prices_interactive.py"
        tree = ast.parse(src_path.read_text(encoding="utf-8"))
        found = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = getattr(node.func, "attr", "") or getattr(node.func, "id", "")
                if name == "download":
                    kw_names = {kw.arg for kw in node.keywords}
                    assert (
                        "auto_adjust" in kw_names
                    ), "fetch_prices_interactive: yf.download() must pass auto_adjust=True"
                    found = True
        assert found, "No yf.download() call found in fetch_prices_interactive"

    def test_price_history_backfill(self):
        from wake_robin_data_pipeline.price_history_backfill import fetch_prices_yfinance

        # Helper asserts the yfinance call passes auto_adjust=True; returns None once the check completes
        result = self._assert_auto_adjust_explicit(fetch_prices_yfinance, "price_history_backfill")
        assert result is None
