"""Tests for scripts/repair_price_history_splits.py."""
from __future__ import annotations

import csv
import sys
import tempfile
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Dict, List

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.repair_price_history_splits import (
    compute_adjusted_prices,
    detect_splits,
    write_adjusted_csv,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prices(ticker: str, date_prices: List[tuple]) -> Dict[str, Dict[date, Decimal]]:
    """Build prices dict for one ticker: [(date_str, price_float), ...]."""
    return {
        ticker: {date.fromisoformat(d): Decimal(str(p)) for d, p in date_prices}
    }


def _merge(*maps) -> Dict[str, Dict[date, Decimal]]:
    out: Dict[str, Dict[date, Decimal]] = {}
    for m in maps:
        out.update(m)
    return out


def _write_price_csv(path: Path, rows: list[tuple[str, str, str]]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ticker", "date", "close"])
        for ticker, dt, price in rows:
            w.writerow([ticker, dt, price])


# ---------------------------------------------------------------------------
# detect_splits
# ---------------------------------------------------------------------------

class TestDetectSplits:

    def test_reverse_split_detected(self):
        """10x jump → reverse_split with correct factor."""
        prices = _prices("DRUG", [
            ("2024-10-14", 1.0),
            ("2024-10-15", 35.0),
            ("2024-10-16", 34.0),
        ])
        splits = detect_splits(prices)
        assert "DRUG" in splits
        assert len(splits["DRUG"]) == 1
        e = splits["DRUG"][0]
        assert e["flag_type"] == "reverse_split"
        assert e["date"] == date(2024, 10, 15)
        assert float(e["factor"]) == pytest.approx(35.0, rel=1e-4)

    def test_forward_split_detected(self):
        """80% drop → forward_split with correct factor."""
        prices = _prices("SPLT", [
            ("2025-03-01", 100.0),
            ("2025-03-02", 100.0),
            ("2025-03-03", 20.0),
            ("2025-03-04", 21.0),
        ])
        splits = detect_splits(prices)
        assert "SPLT" in splits
        e = splits["SPLT"][0]
        assert e["flag_type"] == "forward_split"
        assert float(e["factor"]) == pytest.approx(0.2, rel=1e-4)

    def test_normal_volatility_not_flagged(self):
        """±50% single day is NOT flagged."""
        prices = _prices("BIOT", [
            ("2025-01-10", 10.0),
            ("2025-01-13", 15.0),   # +50%
            ("2025-01-14", 7.5),    # -50%
            ("2025-01-15", 8.0),
        ])
        splits = detect_splits(prices)
        assert "BIOT" not in splits

    def test_multiple_splits_same_ticker(self):
        """Two splits on same ticker are both detected, sorted by date."""
        prices = _prices("JBIO", [
            ("2024-06-14", 800.0),
            ("2024-06-17", 50.0),    # forward split (-93.75%)
            ("2024-06-18", 52.0),
            ("2025-04-28", 90.0),
            ("2025-04-29", 10.0),    # another forward split (-88.9%)
            ("2025-04-30", 11.0),
        ])
        splits = detect_splits(prices)
        assert len(splits["JBIO"]) == 2
        assert splits["JBIO"][0]["date"] < splits["JBIO"][1]["date"]

    def test_zero_price_skipped(self):
        """Zero previous price doesn't cause division error."""
        prices = _prices("BAD", [
            ("2025-01-01", 0.0),
            ("2025-01-02", 10.0),
        ])
        splits = detect_splits(prices)
        assert "BAD" not in splits


# ---------------------------------------------------------------------------
# compute_adjusted_prices
# ---------------------------------------------------------------------------

class TestComputeAdjustedPrices:

    def test_single_reverse_split(self):
        """Pre-split prices multiplied by factor; post-split unchanged."""
        prices = _prices("DRUG", [
            ("2024-10-13", Decimal("1.00")),
            ("2024-10-14", Decimal("1.00")),
            ("2024-10-15", Decimal("35.00")),  # 35x reverse split
            ("2024-10-16", Decimal("34.00")),
        ])
        splits = detect_splits(prices)
        adj = compute_adjusted_prices(prices, splits)
        # Pre-split prices should be multiplied by 35
        assert adj["DRUG"][date(2024, 10, 13)] == pytest.approx(Decimal("35.00"), rel=1e-4)
        assert adj["DRUG"][date(2024, 10, 14)] == pytest.approx(Decimal("35.00"), rel=1e-4)
        # Post-split prices unchanged
        assert adj["DRUG"][date(2024, 10, 15)] == Decimal("35.00")
        assert adj["DRUG"][date(2024, 10, 16)] == Decimal("34.00")

    def test_continuity_after_adjustment(self):
        """After adjustment, the return across the split date should be modest."""
        prices = _prices("DRUG", [
            ("2024-10-14", Decimal("1.08")),
            ("2024-10-15", Decimal("38.49")),
            ("2024-10-16", Decimal("37.50")),
        ])
        splits = detect_splits(prices)
        adj = compute_adjusted_prices(prices, splits)
        # Return from 10/14 to 10/15 should now be ~0% (both ~38.49 level)
        ret = float(adj["DRUG"][date(2024, 10, 15)] / adj["DRUG"][date(2024, 10, 14)]) - 1.0
        assert abs(ret) < 0.01  # < 1% — continuous

    def test_forward_split_adjustment(self):
        """Forward split: pre-split prices multiplied by factor < 1."""
        prices = _prices("SPLT", [
            ("2025-03-01", Decimal("100.0")),
            ("2025-03-02", Decimal("100.0")),
            ("2025-03-03", Decimal("20.0")),   # 5:1 forward split
            ("2025-03-04", Decimal("21.0")),
        ])
        splits = detect_splits(prices)
        adj = compute_adjusted_prices(prices, splits)
        # Pre-split prices multiplied by 0.2
        assert adj["SPLT"][date(2025, 3, 1)] == pytest.approx(Decimal("20.0"), rel=1e-4)
        # Post-split unchanged
        assert adj["SPLT"][date(2025, 3, 3)] == Decimal("20.0")

    def test_two_splits_cumulative(self):
        """Two splits: all pre-first-split prices get both factors."""
        prices = _prices("MULTI", [
            ("2024-01-01", Decimal("2.0")),
            ("2024-01-02", Decimal("2.0")),
            # First split: 10x reverse split (2→20)
            ("2024-06-01", Decimal("20.0")),
            ("2024-06-02", Decimal("19.0")),
            ("2024-06-03", Decimal("5.0")),
            # Second split: 4x reverse split (5→20, +300%)
            ("2024-12-01", Decimal("20.0")),
            ("2024-12-02", Decimal("19.0")),
        ])
        splits = detect_splits(prices)
        assert len(splits["MULTI"]) == 2
        f1 = splits["MULTI"][0]["factor"]  # 10
        f2 = splits["MULTI"][1]["factor"]  # 4
        adj = compute_adjusted_prices(prices, splits)
        # Pre-first-split: both factors applied (10 * 4 = 40x)
        assert adj["MULTI"][date(2024, 1, 1)] == pytest.approx(Decimal("2.0") * f1 * f2, rel=1e-4)
        # Between splits: only factor2 (4x)
        assert adj["MULTI"][date(2024, 6, 1)] == pytest.approx(Decimal("20.0") * f2, rel=1e-4)
        assert adj["MULTI"][date(2024, 6, 3)] == pytest.approx(Decimal("5.0") * f2, rel=1e-4)
        # Post-second-split: unchanged
        assert adj["MULTI"][date(2024, 12, 1)] == Decimal("20.0")

    def test_unaffected_ticker_unchanged(self):
        """Tickers without splits pass through unchanged."""
        prices = _merge(
            _prices("GOOD", [("2025-01-01", 50.0), ("2025-01-02", 51.0)]),
            _prices("SPLIT", [("2025-01-01", 10.0), ("2025-01-02", 100.0)]),
        )
        splits = detect_splits(prices)
        adj = compute_adjusted_prices(prices, splits)
        assert adj["GOOD"][date(2025, 1, 1)] == Decimal("50.0")
        assert adj["GOOD"][date(2025, 1, 2)] == Decimal("51.0")


# ---------------------------------------------------------------------------
# write_adjusted_csv
# ---------------------------------------------------------------------------

class TestWriteAdjustedCsv:

    def test_roundtrip(self, tmp_path):
        """Written CSV can be re-read with correct values."""
        adjusted = {
            "AAA": {date(2025, 1, 1): Decimal("10.123456"), date(2025, 1, 2): Decimal("10.5")},
            "BBB": {date(2025, 1, 1): Decimal("20.0")},
        }
        out_path = tmp_path / "out.csv"
        n = write_adjusted_csv(adjusted, out_path)
        assert n == 3

        # Re-read
        with open(out_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 3
        assert rows[0]["ticker"] == "AAA"
        assert rows[0]["date"] == "2025-01-01"
        assert Decimal(rows[0]["close"]) == Decimal("10.123456")


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------

class TestEndToEnd:

    def test_full_pipeline_synthetic(self, tmp_path):
        """Full detect → adjust → write pipeline on synthetic data."""
        prices = _merge(
            _prices("CLEAN", [
                ("2025-01-01", 50.0), ("2025-01-02", 51.0), ("2025-01-03", 49.0),
            ]),
            _prices("SPLIT", [
                ("2025-01-01", 2.0), ("2025-01-02", 2.0),
                ("2025-01-03", 40.0),  # 20x reverse split
                ("2025-01-04", 39.0),
            ]),
        )
        splits = detect_splits(prices)
        adj = compute_adjusted_prices(prices, splits)

        out_path = tmp_path / "adjusted.csv"
        write_adjusted_csv(adj, out_path)

        # Verify SPLIT is continuous
        ret = float(adj["SPLIT"][date(2025, 1, 3)] / adj["SPLIT"][date(2025, 1, 2)]) - 1.0
        assert abs(ret) < 0.01

        # Verify CLEAN unchanged
        assert adj["CLEAN"][date(2025, 1, 1)] == Decimal("50.0")

    def test_empty_prices(self):
        """Empty prices dict produces no splits and no adjusted prices."""
        splits = detect_splits({})
        adj = compute_adjusted_prices({}, splits)
        assert splits == {}
        assert adj == {}
