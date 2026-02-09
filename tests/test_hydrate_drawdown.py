"""Tests for _hydrate_drawdown() in run_screen.py."""
from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from run_screen import _hydrate_drawdown


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rec(
    ticker: str,
    drawdown: float | None = None,
    drawdown_current: float | None = None,
    drawdown_60d: float | None = None,
) -> Dict[str, Any]:
    """Build a minimal rec with defensive_features."""
    df: Dict[str, Any] = {}
    if drawdown is not None:
        df["drawdown"] = drawdown
    if drawdown_current is not None:
        df["drawdown_current"] = drawdown_current
    if drawdown_60d is not None:
        df["drawdown_60d"] = drawdown_60d
    return {"ticker": ticker, "defensive_features": df}


def _write_price_csv(path: Path, rows: list[dict]) -> None:
    """Write a price_history.csv."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ticker", "date", "close"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Step A: key normalization
# ---------------------------------------------------------------------------

class TestKeyNormalization:

    def test_drawdown_current_copied(self):
        """drawdown_current is normalized to drawdown."""
        recs = {"A": _make_rec("A", drawdown_current=-0.20)}
        n = _hydrate_drawdown(recs, None, "2025-06-01")
        assert n == 1
        assert recs["A"]["defensive_features"]["drawdown"] == -0.20

    def test_drawdown_60d_copied(self):
        """drawdown_60d is normalized to drawdown."""
        recs = {"B": _make_rec("B", drawdown_60d=-0.10)}
        n = _hydrate_drawdown(recs, None, "2025-06-01")
        assert n == 1
        assert recs["B"]["defensive_features"]["drawdown"] == -0.10

    def test_drawdown_current_preferred_over_60d(self):
        """drawdown_current takes precedence over drawdown_60d."""
        recs = {"C": _make_rec("C", drawdown_current=-0.30, drawdown_60d=-0.10)}
        n = _hydrate_drawdown(recs, None, "2025-06-01")
        assert n == 1
        assert recs["C"]["defensive_features"]["drawdown"] == -0.30

    def test_existing_drawdown_not_overwritten(self):
        """If drawdown is already present, normalization is skipped."""
        recs = {"D": _make_rec("D", drawdown=-0.05, drawdown_current=-0.20)}
        n = _hydrate_drawdown(recs, None, "2025-06-01")
        assert n == 0
        assert recs["D"]["defensive_features"]["drawdown"] == -0.05

    def test_no_defensive_features(self):
        """Rec with no defensive_features dict gets one created."""
        recs = {"E": {"ticker": "E"}}
        n = _hydrate_drawdown(recs, None, "2025-06-01")
        assert n == 0
        assert "defensive_features" in recs["E"]


# ---------------------------------------------------------------------------
# Step B: compute from price_history.csv
# ---------------------------------------------------------------------------

class TestComputeFromPriceHistory:

    def test_basic_computation(self):
        """Drawdown computed as (current / peak) - 1.0."""
        recs = {"XYZ": _make_rec("XYZ")}
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "price_history.csv"
            # Peak at 100, current at 80 → dd = -0.20
            prices = [
                {"ticker": "XYZ", "date": "2025-01-01", "close": "90"},
                {"ticker": "XYZ", "date": "2025-03-01", "close": "100"},
                {"ticker": "XYZ", "date": "2025-06-01", "close": "80"},
            ]
            _write_price_csv(csv_path, prices)
            n = _hydrate_drawdown(recs, csv_path, "2025-06-01")
        assert n == 1
        assert abs(recs["XYZ"]["defensive_features"]["drawdown"] - (-0.20)) < 1e-4

    def test_pit_safety(self):
        """Future dates (after as_of_date) are excluded."""
        recs = {"PIT": _make_rec("PIT")}
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "price_history.csv"
            prices = [
                {"ticker": "PIT", "date": "2025-01-01", "close": "100"},
                {"ticker": "PIT", "date": "2025-06-01", "close": "80"},
                # Future — should be excluded
                {"ticker": "PIT", "date": "2025-12-01", "close": "50"},
            ]
            _write_price_csv(csv_path, prices)
            n = _hydrate_drawdown(recs, csv_path, "2025-06-01")
        # dd = (80 / 100) - 1 = -0.20 (not -0.50 if future were included)
        assert abs(recs["PIT"]["defensive_features"]["drawdown"] - (-0.20)) < 1e-4

    def test_no_price_data_leaves_none(self):
        """Tickers with no price rows remain without drawdown."""
        recs = {"MISS": _make_rec("MISS")}
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "price_history.csv"
            _write_price_csv(csv_path, [])
            n = _hydrate_drawdown(recs, csv_path, "2025-06-01")
        assert n == 0
        assert recs["MISS"]["defensive_features"].get("drawdown") is None

    def test_already_normalized_skipped(self):
        """Tickers already hydrated by Step A are not recomputed."""
        recs = {"NORM": _make_rec("NORM", drawdown_current=-0.10)}
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "price_history.csv"
            prices = [
                {"ticker": "NORM", "date": "2025-01-01", "close": "100"},
                {"ticker": "NORM", "date": "2025-06-01", "close": "50"},
            ]
            _write_price_csv(csv_path, prices)
            n = _hydrate_drawdown(recs, csv_path, "2025-06-01")
        # Should have used normalized value, not computed -0.50
        assert n == 1
        assert recs["NORM"]["defensive_features"]["drawdown"] == -0.10

    def test_missing_csv_returns_zero(self):
        """Non-existent csv path returns 0 hydrated."""
        recs = {"X": _make_rec("X")}
        n = _hydrate_drawdown(recs, Path("/nonexistent.csv"), "2025-06-01")
        assert n == 0

    def test_none_path_returns_zero(self):
        """None price_history_path returns 0 hydrated for step B."""
        recs = {"X": _make_rec("X")}
        n = _hydrate_drawdown(recs, None, "2025-06-01")
        assert n == 0
