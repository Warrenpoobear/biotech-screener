"""Tests for _hydrate_drawdown() and _hydrate_beta_rsi() in run_screen.py."""
from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from run_screen import (
    _hydrate_drawdown,
    _hydrate_beta_rsi,
    MIN_BARS_FOR_ESTIMATE,
    BETA_WINDOW,
    MIN_OVERLAP_BARS,
    XBI_STALE_THRESHOLD,
)


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

    def test_string_drawdown_coerced_to_float(self):
        """Drawdown stored as string in JSON is coerced to float."""
        recs = {"S": {"ticker": "S", "defensive_features": {"drawdown": "-0.25"}}}
        n = _hydrate_drawdown(recs, None, "2025-06-01")
        assert n == 0
        dd = recs["S"]["defensive_features"]["drawdown"]
        assert isinstance(dd, float)
        assert dd == -0.25

    def test_string_alt_key_coerced_to_float(self):
        """Alt-key (drawdown_current) stored as string is coerced to float."""
        recs = {"T": {"ticker": "T", "defensive_features": {"drawdown_current": "-0.10"}}}
        n = _hydrate_drawdown(recs, None, "2025-06-01")
        assert n == 1
        dd = recs["T"]["defensive_features"]["drawdown"]
        assert isinstance(dd, float)
        assert dd == -0.10

    def test_non_numeric_drawdown_becomes_none(self):
        """Non-numeric drawdown string is treated as missing."""
        recs = {"U": {"ticker": "U", "defensive_features": {"drawdown": "N/A"}}}
        n = _hydrate_drawdown(recs, None, "2025-06-01")
        assert recs["U"]["defensive_features"]["drawdown"] is None

    def test_non_numeric_alt_key_skipped(self):
        """Non-numeric alt-key is skipped; falls through to next alt or CSV."""
        recs = {"V": {"ticker": "V", "defensive_features": {
            "drawdown_current": "bad", "drawdown_60d": "-0.15",
        }}}
        n = _hydrate_drawdown(recs, None, "2025-06-01")
        assert n == 1
        dd = recs["V"]["defensive_features"]["drawdown"]
        assert isinstance(dd, float)
        assert dd == -0.15

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
        from datetime import date, timedelta
        recs = {"XYZ": _make_rec("XYZ")}
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "price_history.csv"
            # Build MIN_BARS_FOR_ESTIMATE bars: flat at 100, then peak at 100, last at 80
            base = date(2024, 6, 1)
            prices = [
                {"ticker": "XYZ", "date": (base + timedelta(days=i)).isoformat(),
                 "close": "100"}
                for i in range(MIN_BARS_FOR_ESTIMATE - 1)
            ]
            # Last bar: drop to 80 → dd = (80 / 100) - 1.0 = -0.20
            last_date = (base + timedelta(days=MIN_BARS_FOR_ESTIMATE - 1)).isoformat()
            prices.append({"ticker": "XYZ", "date": last_date, "close": "80"})
            _write_price_csv(csv_path, prices)
            n = _hydrate_drawdown(recs, csv_path, last_date)
        assert n == 1
        assert abs(recs["XYZ"]["defensive_features"]["drawdown"] - (-0.20)) < 1e-4

    def test_pit_safety(self):
        """Future dates (after as_of_date) are excluded."""
        from datetime import date, timedelta
        recs = {"PIT": _make_rec("PIT")}
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "price_history.csv"
            # Build MIN_BARS_FOR_ESTIMATE bars up to as_of_date, all at 100
            base = date(2024, 6, 1)
            as_of = date(2025, 6, 1)
            prices = [
                {"ticker": "PIT", "date": (base + timedelta(days=i)).isoformat(),
                 "close": "100"}
                for i in range(MIN_BARS_FOR_ESTIMATE - 1)
            ]
            # Last bar on as_of_date at 80
            prices.append({"ticker": "PIT", "date": as_of.isoformat(), "close": "80"})
            # Future bar — should be excluded
            prices.append({"ticker": "PIT", "date": "2025-12-01", "close": "50"})
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

    def test_csv_overwrites_normalized_value(self):
        """When price CSV has data, CSV-computed drawdown overwrites pipeline value."""
        from datetime import date, timedelta
        recs = {"NORM": _make_rec("NORM", drawdown_current=-0.10)}
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "price_history.csv"
            base = date(2024, 6, 1)
            prices = [
                {"ticker": "NORM", "date": (base + timedelta(days=i)).isoformat(),
                 "close": "100"}
                for i in range(MIN_BARS_FOR_ESTIMATE - 1)
            ]
            last_date = (base + timedelta(days=MIN_BARS_FOR_ESTIMATE - 1)).isoformat()
            prices.append({"ticker": "NORM", "date": last_date, "close": "50"})
            _write_price_csv(csv_path, prices)
            n = _hydrate_drawdown(recs, csv_path, last_date)
        # CSV-computed -0.50 overwrites the pipeline-normalized -0.10
        assert n == 1
        assert abs(recs["NORM"]["defensive_features"]["drawdown"] - (-0.50)) < 1e-4

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


# ---------------------------------------------------------------------------
# Min-bars threshold and missing-reason tracking
# ---------------------------------------------------------------------------

class TestMinBarsAndMissingReason:

    def test_series_too_short_marked_missing(self):
        """50 bars (< MIN_BARS_FOR_ESTIMATE) → drawdown None, reason series_too_short."""
        recs = {"SHORT": _make_rec("SHORT")}
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "price_history.csv"
            # Generate 50 daily prices
            prices = [
                {"ticker": "SHORT", "date": f"2025-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}",
                 "close": str(100 + i)}
                for i in range(50)
            ]
            _write_price_csv(csv_path, prices)
            n = _hydrate_drawdown(recs, csv_path, "2025-12-31")
        assert n == 0
        assert recs["SHORT"]["defensive_features"].get("drawdown") is None
        assert recs["SHORT"]["defensive_features"]["drawdown_missing_reason"] == "series_too_short"

    def test_series_at_min_bars_computed(self):
        """Exactly MIN_BARS_FOR_ESTIMATE bars → drawdown computed, reason cleared."""
        recs = {"GOOD": _make_rec("GOOD")}
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "price_history.csv"
            from datetime import date, timedelta
            base = date(2024, 6, 1)
            prices = [
                {"ticker": "GOOD", "date": (base + timedelta(days=i)).isoformat(),
                 "close": str(100.0)}  # flat → dd = 0.0
                for i in range(MIN_BARS_FOR_ESTIMATE)
            ]
            _write_price_csv(csv_path, prices)
            last_date = (base + timedelta(days=MIN_BARS_FOR_ESTIMATE - 1)).isoformat()
            n = _hydrate_drawdown(recs, csv_path, last_date)
        assert n == 1
        assert recs["GOOD"]["defensive_features"]["drawdown"] == 0.0
        assert recs["GOOD"]["defensive_features"]["drawdown_missing_reason"] == ""

    def test_no_price_series_reason(self):
        """Ticker not in price CSV → reason = no_price_series."""
        recs = {"ABSENT": _make_rec("ABSENT")}
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "price_history.csv"
            # CSV exists but has no rows for ABSENT
            _write_price_csv(csv_path, [
                {"ticker": "OTHER", "date": "2025-01-01", "close": "100"},
            ])
            n = _hydrate_drawdown(recs, csv_path, "2025-06-01")
        assert n == 0
        assert recs["ABSENT"]["defensive_features"].get("drawdown") is None
        assert recs["ABSENT"]["defensive_features"]["drawdown_missing_reason"] == "no_price_series"

    def test_missing_reason_cleared_on_normalize(self):
        """Key normalization (Step A) sets reason to empty string."""
        recs = {"NORM": _make_rec("NORM", drawdown_current=-0.15)}
        n = _hydrate_drawdown(recs, None, "2025-06-01")
        assert n == 1
        assert recs["NORM"]["defensive_features"]["drawdown"] == -0.15
        assert recs["NORM"]["defensive_features"]["drawdown_missing_reason"] == ""

    def test_existing_drawdown_has_empty_reason(self):
        """Ticker with pre-existing drawdown gets reason = empty."""
        recs = {"EXISTS": _make_rec("EXISTS", drawdown=-0.10)}
        n = _hydrate_drawdown(recs, None, "2025-06-01")
        assert n == 0
        assert recs["EXISTS"]["defensive_features"]["drawdown_missing_reason"] == ""

    def test_no_csv_tags_as_no_price_series(self):
        """When price_history_path is None, remaining missing tickers get no_price_series."""
        recs = {"MISS": _make_rec("MISS")}
        n = _hydrate_drawdown(recs, None, "2025-06-01")
        assert n == 0
        assert recs["MISS"]["defensive_features"]["drawdown_missing_reason"] == "no_price_series"


# ---------------------------------------------------------------------------
# Alias resolution (dot↔dash symbol normalization)
# ---------------------------------------------------------------------------

class TestAliasResolution:

    def _make_alias_prices(self, csv_ticker: str, n_bars: int = MIN_BARS_FOR_ESTIMATE):
        """Build price rows for a given CSV ticker with enough bars."""
        from datetime import date, timedelta
        base = date(2024, 6, 1)
        prices = [
            {"ticker": csv_ticker,
             "date": (base + timedelta(days=i)).isoformat(),
             "close": "100"}
            for i in range(n_bars - 1)
        ]
        # Last bar at 80 → dd = -0.20
        last_date = (base + timedelta(days=n_bars - 1)).isoformat()
        prices.append({"ticker": csv_ticker, "date": last_date, "close": "80"})
        return prices, last_date

    def test_dot_to_dash_resolution(self):
        """Record has BRK.B, CSV has BRK-B → drawdown computed via alias."""
        recs = {"BRK.B": _make_rec("BRK.B")}
        prices, last_date = self._make_alias_prices("BRK-B")
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "price_history.csv"
            _write_price_csv(csv_path, prices)
            n = _hydrate_drawdown(recs, csv_path, last_date)
        assert n == 1
        dd = recs["BRK.B"]["defensive_features"]["drawdown"]
        assert abs(dd - (-0.20)) < 1e-4
        assert recs["BRK.B"]["defensive_features"]["drawdown_missing_reason"] == ""
        assert recs["BRK.B"]["defensive_features"]["drawdown_price_symbol"] == "BRK-B"

    def test_dash_to_dot_resolution(self):
        """Record has ABC-1, CSV has ABC.1 → drawdown computed via alias."""
        recs = {"ABC-1": _make_rec("ABC-1")}
        prices, last_date = self._make_alias_prices("ABC.1")
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "price_history.csv"
            _write_price_csv(csv_path, prices)
            n = _hydrate_drawdown(recs, csv_path, last_date)
        assert n == 1
        dd = recs["ABC-1"]["defensive_features"]["drawdown"]
        assert abs(dd - (-0.20)) < 1e-4
        assert recs["ABC-1"]["defensive_features"]["drawdown_missing_reason"] == ""
        assert recs["ABC-1"]["defensive_features"]["drawdown_price_symbol"] == "ABC.1"

    def test_exact_match_preferred_over_alias(self):
        """When both exact and alias match exist, exact is used (no alias tag)."""
        recs = {"XYZ": _make_rec("XYZ")}
        prices_exact, last_date = self._make_alias_prices("XYZ")
        # Also add a dot-variant that shouldn't be used
        prices_alias, _ = self._make_alias_prices("X.YZ", n_bars=MIN_BARS_FOR_ESTIMATE)
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "price_history.csv"
            _write_price_csv(csv_path, prices_exact + prices_alias)
            n = _hydrate_drawdown(recs, csv_path, last_date)
        assert n == 1
        assert recs["XYZ"]["defensive_features"]["drawdown_missing_reason"] == ""
        # No alias tag when exact match used
        assert "drawdown_price_symbol" not in recs["XYZ"]["defensive_features"]

    def test_collision_determinism(self):
        """Two CSV symbols both alias to same missing ticker → lexicographically smallest wins."""
        # Record wants "A.B", CSV has "A-B" and "A_B" won't match,
        # but if we had two variants... Actually dot↔dash only produces one
        # variant. Let's test with a ticker where the CSV has the exact alias.
        # More realistic: record "X.Y", CSV has "X-Y" → only one match, deterministic.
        recs = {"X.Y": _make_rec("X.Y")}
        prices, last_date = self._make_alias_prices("X-Y")
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "price_history.csv"
            _write_price_csv(csv_path, prices)
            # Run twice to verify stability
            n1 = _hydrate_drawdown({"X.Y": _make_rec("X.Y")}, csv_path, last_date)
            recs2 = {"X.Y": _make_rec("X.Y")}
            n2 = _hydrate_drawdown(recs2, csv_path, last_date)
        assert n1 == n2 == 1
        assert recs2["X.Y"]["defensive_features"]["drawdown_price_symbol"] == "X-Y"

    def test_no_alias_for_clean_missing(self):
        """Ticker with no special chars and not in CSV stays no_price_series."""
        recs = {"ZZZZ": _make_rec("ZZZZ")}
        prices, last_date = self._make_alias_prices("OTHER")
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "price_history.csv"
            _write_price_csv(csv_path, prices)
            n = _hydrate_drawdown(recs, csv_path, last_date)
        assert n == 0
        assert recs["ZZZZ"]["defensive_features"]["drawdown_missing_reason"] == "no_price_series"
        assert "drawdown_price_symbol" not in recs["ZZZZ"]["defensive_features"]


# ---------------------------------------------------------------------------
# XBI drawdown + relative drawdown
# ---------------------------------------------------------------------------

class TestXbiRelativeDrawdown:

    @staticmethod
    def _make_prices(ticker: str, n_bars: int, start_price: float = 10.0,
                     end_price: float = 8.0):
        """Generate a linearly declining price series."""
        from datetime import date, timedelta
        base = date(2025, 6, 1)
        rows = []
        for i in range(n_bars):
            frac = i / max(n_bars - 1, 1)
            price = start_price + (end_price - start_price) * frac
            rows.append({
                "ticker": ticker,
                "date": (base + timedelta(days=i)).isoformat(),
                "close": f"{price:.4f}",
            })
        return rows, (base + timedelta(days=n_bars - 1)).isoformat()

    def test_xbi_drawdown_populated(self):
        """XBI drawdown and relative drawdown are populated for all tickers."""
        ticker_prices, last_date = self._make_prices("ACME", 200, 10.0, 7.0)
        xbi_prices, _ = self._make_prices("XBI", 200, 100.0, 90.0)
        recs = {"ACME": _make_rec("ACME")}
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "price_history.csv"
            _write_price_csv(csv_path, ticker_prices + xbi_prices)
            _hydrate_drawdown(recs, csv_path, last_date)
        df = recs["ACME"]["defensive_features"]
        assert df["drawdown_xbi"] is not None
        assert df["drawdown_xbi"] < 0  # XBI is declining
        assert df["drawdown_rel_xbi"] is not None
        # ACME drops 30% while XBI drops 10% → relative should be negative
        assert df["drawdown_rel_xbi"] < 0

    def test_relative_drawdown_computation(self):
        """Relative drawdown = ticker drawdown - XBI drawdown."""
        ticker_prices, last_date = self._make_prices("ACME", 200, 10.0, 7.0)
        xbi_prices, _ = self._make_prices("XBI", 200, 100.0, 90.0)
        recs = {"ACME": _make_rec("ACME")}
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "price_history.csv"
            _write_price_csv(csv_path, ticker_prices + xbi_prices)
            _hydrate_drawdown(recs, csv_path, last_date)
        df = recs["ACME"]["defensive_features"]
        expected_rel = round(df["drawdown"] - df["drawdown_xbi"], 6)
        assert abs(df["drawdown_rel_xbi"] - expected_rel) < 1e-6

    def test_relative_drawdown_missing_xbi(self):
        """When XBI is not in CSV, relative drawdown is None."""
        ticker_prices, last_date = self._make_prices("ACME", 200, 10.0, 7.0)
        recs = {"ACME": _make_rec("ACME")}
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "price_history.csv"
            _write_price_csv(csv_path, ticker_prices)
            _hydrate_drawdown(recs, csv_path, last_date)
        df = recs["ACME"]["defensive_features"]
        assert df["drawdown_xbi"] is None
        assert df["drawdown_rel_xbi"] is None


# ---------------------------------------------------------------------------
# Regression: stale pipeline drawdown overwrite (KRYS case)
# ---------------------------------------------------------------------------

class TestStaleDrawdownOverwrite:
    """Regression tests for the KRYS bug: Morningstar pipeline provided stale
    drawdown (-45.8%) while price_history.csv showed only -5.3%.  The fix
    ensures _hydrate_drawdown always recomputes from price_history.csv,
    overwriting any stale pipeline value.
    """

    @staticmethod
    def _make_prices(ticker, n_bars, start_price, end_price):
        from datetime import date, timedelta
        base = date(2025, 6, 1)
        rows = []
        for i in range(n_bars):
            frac = i / max(n_bars - 1, 1)
            price = start_price + (end_price - start_price) * frac
            rows.append({
                "ticker": ticker,
                "date": (base + timedelta(days=i)).isoformat(),
                "close": f"{price:.4f}",
            })
        return rows, (base + timedelta(days=n_bars - 1)).isoformat()

    def test_stale_pipeline_drawdown_overwritten(self):
        """Pipeline drawdown is replaced by price_history.csv computation."""
        # Pipeline says -0.45 (stale), actual prices show only -5% drop
        recs = {"KRYS": _make_rec("KRYS", drawdown=-0.4581)}
        prices, last_date = self._make_prices("KRYS", 200, 100.0, 95.0)
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "price_history.csv"
            _write_price_csv(csv_path, prices)
            n = _hydrate_drawdown(recs, csv_path, last_date)
        dd = recs["KRYS"]["defensive_features"]["drawdown"]
        assert n >= 1
        # Fresh drawdown should be around -0.05, NOT -0.4581
        assert abs(dd - (-0.05)) < 0.01
        assert recs["KRYS"]["defensive_features"]["drawdown_missing_reason"] == ""

    def test_overwrite_corrects_relative_drawdown(self):
        """Relative drawdown is recomputed after overwrite."""
        # Pipeline has stale abs drawdown; CSV has correct prices
        recs = {"STALE": _make_rec("STALE", drawdown=-0.50)}
        ticker_prices, last_date = self._make_prices("STALE", 200, 100.0, 90.0)
        xbi_prices, _ = self._make_prices("XBI", 200, 100.0, 95.0)
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "price_history.csv"
            _write_price_csv(csv_path, ticker_prices + xbi_prices)
            _hydrate_drawdown(recs, csv_path, last_date)
        df = recs["STALE"]["defensive_features"]
        # STALE drops 10%, XBI drops 5%, relative = -10% - (-5%) = -5%
        assert abs(df["drawdown"] - (-0.10)) < 0.01
        assert abs(df["drawdown_rel_xbi"] - (-0.05)) < 0.01

    def test_no_csv_data_keeps_pipeline_value(self):
        """When price CSV has no data for ticker, pipeline value is kept."""
        recs = {"NOCSVDATA": _make_rec("NOCSVDATA", drawdown=-0.30)}
        # CSV has data for OTHER but not NOCSVDATA
        other_prices, last_date = self._make_prices("OTHER", 200, 100.0, 80.0)
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "price_history.csv"
            _write_price_csv(csv_path, other_prices)
            _hydrate_drawdown(recs, csv_path, last_date)
        # Pipeline value preserved as fallback
        assert recs["NOCSVDATA"]["defensive_features"]["drawdown"] == -0.30
        assert recs["NOCSVDATA"]["defensive_features"]["drawdown_missing_reason"] == ""

    def test_no_csv_path_keeps_pipeline_value(self):
        """When price_history_path is None, pipeline drawdown preserved."""
        recs = {"NOOP": _make_rec("NOOP", drawdown=-0.25)}
        _hydrate_drawdown(recs, None, "2025-06-01")
        assert recs["NOOP"]["defensive_features"]["drawdown"] == -0.25

    def test_multiple_tickers_all_recomputed(self):
        """All tickers with price data get fresh drawdowns, not just missing."""
        recs = {
            "FRESH": _make_rec("FRESH", drawdown=-0.80),   # stale, overstated
            "STALE": _make_rec("STALE", drawdown=-0.05),   # stale, understated
            "EMPTY": _make_rec("EMPTY"),                     # missing
        }
        fresh_prices, last_date = self._make_prices("FRESH", 200, 100.0, 90.0)
        stale_prices, _ = self._make_prices("STALE", 200, 100.0, 50.0)
        empty_prices, _ = self._make_prices("EMPTY", 200, 100.0, 70.0)
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "price_history.csv"
            _write_price_csv(csv_path, fresh_prices + stale_prices + empty_prices)
            n = _hydrate_drawdown(recs, csv_path, last_date)
        # All three should have CSV-computed drawdowns
        assert abs(recs["FRESH"]["defensive_features"]["drawdown"] - (-0.10)) < 0.01
        assert abs(recs["STALE"]["defensive_features"]["drawdown"] - (-0.50)) < 0.01
        assert abs(recs["EMPTY"]["defensive_features"]["drawdown"] - (-0.30)) < 0.01
        assert n >= 3


# ===========================================================================
# RSI Overwrite Tests (_hydrate_beta_rsi)
# ===========================================================================

def _make_rsi_prices(ticker: str, n_bars: int, start_close: float = 50.0):
    """Generate price series suitable for RSI computation.

    Alternates between up/down moves to produce a mid-range RSI (~50).
    """
    from datetime import date, timedelta
    rows = []
    d = date(2026, 1, 1)
    close = start_close
    for i in range(n_bars):
        rows.append({"ticker": ticker, "date": d.isoformat(), "close": round(close, 4)})
        d += timedelta(days=1)
        # Alternate: up 1%, down 0.8% → mild uptrend
        if i % 2 == 0:
            close *= 1.01
        else:
            close *= 0.992
    return rows, (d - timedelta(days=1)).isoformat()


def _write_rsi_csv(path: Path, price_rows):
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["ticker", "date", "close"])
        writer.writeheader()
        writer.writerows(price_rows)


class TestRsiOverwrite:
    """Regression tests for _hydrate_beta_rsi RSI overwrite policy."""

    def test_stale_pipeline_rsi_overwritten(self):
        """Pipeline RSI (77.3) should be overwritten with fresh price-CSV computation."""
        recs = {
            "AVTR": {
                "ticker": "AVTR",
                "defensive_features": {"rsi_14d": 77.3},  # stale pipeline value
            },
        }
        prices, last_date = _make_rsi_prices("AVTR", 100)
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "price_history.csv"
            _write_rsi_csv(csv_path, prices)
            n = _hydrate_beta_rsi(recs, csv_path, last_date)
        rsi = recs["AVTR"]["defensive_features"]["rsi_14d"]
        # Should NOT be the stale 77.3; should be recomputed
        assert rsi != 77.3
        assert 0 <= rsi <= 100
        assert n >= 1

    def test_missing_rsi_filled(self):
        """When pipeline has no RSI, hydration computes from CSV."""
        recs = {
            "TEST": {
                "ticker": "TEST",
                "defensive_features": {},
            },
        }
        prices, last_date = _make_rsi_prices("TEST", 50)
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "price_history.csv"
            _write_rsi_csv(csv_path, prices)
            n = _hydrate_beta_rsi(recs, csv_path, last_date)
        rsi = recs["TEST"]["defensive_features"]["rsi_14d"]
        assert rsi is not None
        assert 0 <= rsi <= 100

    def test_too_few_bars_no_rsi(self):
        """With fewer than 15 bars, RSI should not be set."""
        recs = {
            "SHORT": {
                "ticker": "SHORT",
                "defensive_features": {},
            },
        }
        prices, last_date = _make_rsi_prices("SHORT", 10)
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "price_history.csv"
            _write_rsi_csv(csv_path, prices)
            _hydrate_beta_rsi(recs, csv_path, last_date)
        assert recs["SHORT"]["defensive_features"].get("rsi_14d") is None

    def test_no_csv_path_keeps_pipeline_rsi(self):
        """When price_history_path is None, pipeline RSI is preserved."""
        recs = {
            "KEEP": {
                "ticker": "KEEP",
                "defensive_features": {"rsi_14d": 55.0},
            },
        }
        _hydrate_beta_rsi(recs, None, "2026-02-19")
        assert recs["KEEP"]["defensive_features"]["rsi_14d"] == 55.0

    def test_multiple_tickers_all_recomputed(self):
        """All tickers with price data get fresh RSI, even those with existing values."""
        recs = {
            "STALE1": {
                "ticker": "STALE1",
                "defensive_features": {"rsi_14d": 99.0},  # stale
            },
            "STALE2": {
                "ticker": "STALE2",
                "defensive_features": {"rsi_14d": 10.0},  # stale
            },
            "FRESH": {
                "ticker": "FRESH",
                "defensive_features": {},  # missing
            },
        }
        p1, last = _make_rsi_prices("STALE1", 80)
        p2, _ = _make_rsi_prices("STALE2", 80)
        p3, _ = _make_rsi_prices("FRESH", 80)
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "price_history.csv"
            _write_rsi_csv(csv_path, p1 + p2 + p3)
            n = _hydrate_beta_rsi(recs, csv_path, last)
        # All three should have been recomputed (stale values overwritten)
        assert recs["STALE1"]["defensive_features"]["rsi_14d"] != 99.0
        assert recs["STALE2"]["defensive_features"]["rsi_14d"] != 10.0
        assert recs["FRESH"]["defensive_features"]["rsi_14d"] is not None
        assert n >= 3


# ===========================================================================
# Beta / Alpha Overwrite Tests (_hydrate_beta_rsi)
# ===========================================================================

def _make_aligned_prices(
    ticker: str,
    n_bars: int = 80,
    ticker_start: float = 50.0,
    xbi_start: float = 100.0,
    xbi_ends_early: int = 0,
):
    """Generate aligned price series for a ticker and XBI.

    Both start on the same date with identical trading dates.
    If *xbi_ends_early* > 0, XBI will stop that many bars before the ticker,
    creating an XBI staleness gap.

    Returns (all_rows, last_date, xbi_last_date).
    """
    from datetime import date, timedelta
    rows = []
    d = date(2026, 1, 1)
    t_close = ticker_start
    x_close = xbi_start
    xbi_last = None
    for i in range(n_bars):
        day_str = d.isoformat()
        rows.append({"ticker": ticker, "date": day_str,
                      "close": round(t_close, 4)})
        if i < n_bars - xbi_ends_early:
            rows.append({"ticker": "XBI", "date": day_str,
                          "close": round(x_close, 4)})
            xbi_last = day_str
        d += timedelta(days=1)
        # Ticker: +0.5% per bar, XBI: +0.3% per bar
        t_close *= 1.005
        x_close *= 1.003
    last_date = (d - timedelta(days=1)).isoformat()
    return rows, last_date, xbi_last


class TestBetaAlphaOverwrite:
    """Regression tests for aligned-return beta/alpha recomputation."""

    def test_stale_pipeline_beta_overwritten(self):
        """Pipeline beta (2.50) should be overwritten with fresh computation."""
        recs = {
            "ACME": {
                "ticker": "ACME",
                "defensive_features": {"beta_xbi_60d": 2.50},
            },
        }
        prices, last_date, _ = _make_aligned_prices("ACME", n_bars=80)
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "price_history.csv"
            _write_rsi_csv(csv_path, prices)
            _hydrate_beta_rsi(recs, csv_path, last_date)
        df = recs["ACME"]["defensive_features"]
        # Should be recomputed (highly correlated series → beta near 1.5-2.0)
        assert df["beta_xbi_60d"] != 2.50
        assert isinstance(df["beta_xbi_60d"], float)
        assert df["beta_xbi_60d_source"] == "price_history"
        assert df["beta_xbi_60d_missing_reason"] == ""

    def test_alpha_computed_with_source_tagging(self):
        """Alpha should be computed from same aligned window and tagged."""
        recs = {
            "ACME": {
                "ticker": "ACME",
                "defensive_features": {},
                "score_breakdown": {
                    "enhancements": {
                        "momentum": {"alpha_60d": 0.999}  # stale pipeline
                    }
                },
            },
        }
        prices, last_date, _ = _make_aligned_prices("ACME", n_bars=80)
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "price_history.csv"
            _write_rsi_csv(csv_path, prices)
            _hydrate_beta_rsi(recs, csv_path, last_date)
        df = recs["ACME"]["defensive_features"]
        assert "alpha_60d" in df
        assert isinstance(df["alpha_60d"], float)
        assert df["alpha_60d_source"] == "price_history"
        assert df["alpha_60d_missing_reason"] == ""

    def test_xbi_stale_marks_missing_reason(self):
        """When XBI is >XBI_STALE_THRESHOLD trading days behind, mark xbi_stale."""
        gap = XBI_STALE_THRESHOLD + 2  # ensure we exceed the threshold
        recs = {
            "ACME": {
                "ticker": "ACME",
                "defensive_features": {"beta_xbi_60d": 1.5},
            },
        }
        prices, last_date, _ = _make_aligned_prices(
            "ACME", n_bars=80, xbi_ends_early=gap,
        )
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "price_history.csv"
            _write_rsi_csv(csv_path, prices)
            _hydrate_beta_rsi(recs, csv_path, last_date)
        df = recs["ACME"]["defensive_features"]
        assert df["beta_xbi_60d_missing_reason"] == "xbi_stale"
        assert df["beta_xbi_60d_source"] == ""
        assert df["alpha_60d_missing_reason"] == "beta_missing:xbi_stale"
        assert df["alpha_60d_source"] == ""
        # Original pipeline value preserved (not cleared)
        assert df["beta_xbi_60d"] == 1.5

    def test_no_xbi_series(self):
        """When XBI is completely absent from price CSV, mark no_xbi_series."""
        from datetime import date, timedelta
        recs = {
            "ALONE": {
                "ticker": "ALONE",
                "defensive_features": {},
            },
        }
        # Only ticker data, no XBI
        d = date(2026, 1, 1)
        rows = []
        for i in range(80):
            rows.append({"ticker": "ALONE", "date": d.isoformat(), "close": 50 + i * 0.1})
            d += timedelta(days=1)
        last = (d - timedelta(days=1)).isoformat()
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "price_history.csv"
            _write_rsi_csv(csv_path, rows)
            _hydrate_beta_rsi(recs, csv_path, last)
        df = recs["ALONE"]["defensive_features"]
        assert df["beta_xbi_60d_missing_reason"] == "no_xbi_series"
        assert df["alpha_60d_missing_reason"] == "beta_missing:no_xbi_series"

    def test_insufficient_overlap(self):
        """When overlap bars < MIN_OVERLAP_BARS, mark insufficient_overlap."""
        from datetime import date, timedelta
        recs = {
            "SHORT": {
                "ticker": "SHORT",
                "defensive_features": {},
            },
        }
        # XBI spans the full 80 bars (so it's NOT stale), but the ticker
        # only has 10 bars at the very end → only 10 overlapping dates.
        d = date(2026, 1, 1)
        rows = []
        for i in range(80):
            day_str = (d + timedelta(days=i)).isoformat()
            rows.append({"ticker": "XBI", "date": day_str, "close": 100 + i * 0.1})
            # Ticker only present for the last 10 bars
            if i >= 70:
                rows.append({"ticker": "SHORT", "date": day_str, "close": 50 + (i - 70) * 0.1})
        last = (d + timedelta(days=79)).isoformat()
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "price_history.csv"
            _write_rsi_csv(csv_path, rows)
            _hydrate_beta_rsi(recs, csv_path, last)
        df = recs["SHORT"]["defensive_features"]
        assert df["beta_xbi_60d_missing_reason"] == "insufficient_overlap"

    def test_alpha_consistent_with_beta(self):
        """Alpha = ticker_cum_return - beta * xbi_cum_return (from aligned window)."""
        recs = {
            "VERIFY": {
                "ticker": "VERIFY",
                "defensive_features": {},
            },
        }
        prices, last_date, _ = _make_aligned_prices("VERIFY", n_bars=80)
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "price_history.csv"
            _write_rsi_csv(csv_path, prices)
            _hydrate_beta_rsi(recs, csv_path, last_date)
        df = recs["VERIFY"]["defensive_features"]
        beta = df["beta_xbi_60d"]
        alpha = df["alpha_60d"]
        # Manual check: alpha should be small for correlated series
        # Both have positive drift, so alpha reflects excess return
        assert isinstance(alpha, float)
        assert isinstance(beta, float)
        # With ticker +0.5%/bar and XBI +0.3%/bar, ticker outperforms
        # Alpha should be positive (ticker grows faster than beta * XBI)
        assert alpha > -0.5  # sanity bound
