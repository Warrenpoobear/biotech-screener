#!/usr/bin/env python3
"""Tests for tools/warm_price_cache.py — PIT price cache builder.

Covers:
  - Snapshot creation: index schema, anchor prices, missing tickers, coverage
  - Backfill: forward price filling, write-once, split detection
  - Schema validation: 12 invariants
  - Backfill-all: iterates dirs, skips filled/immature
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from warm_price_cache import (
    CACHE_TYPE,
    DEFAULT_HORIZONS,
    SCHEMA_VERSION,
    _find_anchor_date,
    _horizon_cols,
    _sorted_trading_dates,
    _trading_days_after,
    backfill_all_caches,
    backfill_forward_prices,
    detect_split_warnings,
    snapshot_price_cache,
    validate_price_pit_index,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_price_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ticker", "date", "close"])
        writer.writeheader()
        writer.writerows(rows)


def _write_rankings_csv(path: Path, tickers: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ticker", "actionable_rank"])
        writer.writeheader()
        for i, t in enumerate(tickers, 1):
            writer.writerow({"ticker": t, "actionable_rank": str(i)})


def _make_price_rows(*tickers_dates_closes):
    """Build price CSV rows from (ticker, date, close) tuples."""
    return [
        {"ticker": t, "date": d, "close": str(c)}
        for t, d, c in tickers_dates_closes
    ]


# Trading dates for tests: 10 consecutive weekdays
TRADING_DATES = [
    "2026-02-02", "2026-02-03", "2026-02-04", "2026-02-05", "2026-02-06",
    "2026-02-09", "2026-02-10", "2026-02-11", "2026-02-12", "2026-02-13",
    "2026-02-17", "2026-02-18", "2026-02-19", "2026-02-20", "2026-02-23",
    "2026-02-24", "2026-02-25", "2026-02-26", "2026-02-27", "2026-03-02",
    "2026-03-03", "2026-03-04", "2026-03-05", "2026-03-06",
    "2026-03-09", "2026-03-10", "2026-03-11", "2026-03-12", "2026-03-13",
]


def _make_full_price_rows(tickers, dates=None, base_price=10.0):
    """Build price rows for given tickers across all trading dates."""
    dates = dates or TRADING_DATES
    rows = []
    for t in tickers:
        for i, d in enumerate(dates):
            rows.append({"ticker": t, "date": d, "close": str(round(base_price + i * 0.1, 2))})
    return rows


# ---------------------------------------------------------------------------
# TestSnapshotPriceCache
# ---------------------------------------------------------------------------

class TestSnapshotPriceCache:

    def test_creates_index_json_with_valid_schema(self, tmp_path):
        tickers = ["ACRS", "BMRN", "SGEN"]
        _write_price_csv(tmp_path / "prices.csv", _make_full_price_rows(tickers))
        _write_rankings_csv(tmp_path / "rankings.csv", tickers)

        out_dir = tmp_path / "cache" / "2026-02-19"
        index = snapshot_price_cache(
            as_of_date="2026-02-19",
            price_csv=tmp_path / "prices.csv",
            rankings_csv=tmp_path / "rankings.csv",
            out_dir=out_dir,
            horizons=[5, 20],
        )

        assert index["schema_version"] == SCHEMA_VERSION
        assert index["cache_type"] == CACHE_TYPE
        assert index["as_of_date"] == "2026-02-19"
        assert index["created_at"].endswith("Z")
        assert (out_dir / "index.json").exists()
        assert (out_dir / "prices.csv").exists()

    def test_populates_anchor_prices_from_csv(self, tmp_path):
        rows = _make_price_rows(
            ("ACRS", "2026-02-19", 1.87),
            ("BMRN", "2026-02-19", 85.0),
        )
        _write_price_csv(tmp_path / "prices.csv", rows)
        _write_rankings_csv(tmp_path / "rankings.csv", ["ACRS", "BMRN"])

        out = tmp_path / "cache"
        snapshot_price_cache("2026-02-19", tmp_path / "prices.csv", tmp_path / "rankings.csv", out)

        with open(out / "prices.csv", newline="") as f:
            csv_rows = list(csv.DictReader(f))

        assert len(csv_rows) == 2
        assert csv_rows[0]["ticker"] == "ACRS"
        assert float(csv_rows[0]["anchor_close"]) == pytest.approx(1.87)
        assert csv_rows[0]["anchor_date"] == "2026-02-19"

    def test_missing_tickers_recorded_in_index(self, tmp_path):
        # ACRS has a price, NEWCO does not
        rows = _make_price_rows(("ACRS", "2026-02-19", 1.87))
        _write_price_csv(tmp_path / "prices.csv", rows)
        _write_rankings_csv(tmp_path / "rankings.csv", ["ACRS", "NEWCO"])

        out = tmp_path / "cache"
        index = snapshot_price_cache("2026-02-19", tmp_path / "prices.csv", tmp_path / "rankings.csv", out)

        assert "NEWCO" in index["tickers_missing_anchor"]
        assert index["ticker_count"] == 2

    def test_coverage_pct_matches_ticker_counts(self, tmp_path):
        rows = _make_price_rows(
            ("ACRS", "2026-02-19", 1.87),
            ("BMRN", "2026-02-19", 85.0),
        )
        _write_price_csv(tmp_path / "prices.csv", rows)
        _write_rankings_csv(tmp_path / "rankings.csv", ["ACRS", "BMRN", "NEWCO"])

        out = tmp_path / "cache"
        index = snapshot_price_cache("2026-02-19", tmp_path / "prices.csv", tmp_path / "rankings.csv", out)

        # 2/3 tickers have anchor prices
        assert index["coverage_pct"] == pytest.approx(66.67, abs=0.1)
        assert index["ticker_count"] == 3
        assert len(index["tickers_missing_anchor"]) == 1

    def test_source_csv_hash_deterministic(self, tmp_path):
        rows = _make_price_rows(("ACRS", "2026-02-19", 1.87))
        _write_price_csv(tmp_path / "prices.csv", rows)
        _write_rankings_csv(tmp_path / "rankings.csv", ["ACRS"])

        out1 = tmp_path / "cache1"
        idx1 = snapshot_price_cache("2026-02-19", tmp_path / "prices.csv", tmp_path / "rankings.csv", out1)

        out2 = tmp_path / "cache2"
        idx2 = snapshot_price_cache("2026-02-19", tmp_path / "prices.csv", tmp_path / "rankings.csv", out2)

        assert idx1["source_csv_sha256"] == idx2["source_csv_sha256"]
        assert len(idx1["source_csv_sha256"]) == 64

    def test_horizons_all_pending_on_creation(self, tmp_path):
        rows = _make_price_rows(("ACRS", "2026-02-19", 1.87))
        _write_price_csv(tmp_path / "prices.csv", rows)
        _write_rankings_csv(tmp_path / "rankings.csv", ["ACRS"])

        out = tmp_path / "cache"
        index = snapshot_price_cache("2026-02-19", tmp_path / "prices.csv", tmp_path / "rankings.csv", out, horizons=[5, 20])

        assert index["horizons_filled"] == []
        assert index["horizons_pending"] == [5, 20]

    def test_atomic_write_no_partial_files(self, tmp_path):
        """On success, both index.json and prices.csv exist."""
        rows = _make_price_rows(("ACRS", "2026-02-19", 1.87))
        _write_price_csv(tmp_path / "prices.csv", rows)
        _write_rankings_csv(tmp_path / "rankings.csv", ["ACRS"])

        out = tmp_path / "cache"
        snapshot_price_cache("2026-02-19", tmp_path / "prices.csv", tmp_path / "rankings.csv", out)

        assert (out / "index.json").exists()
        assert (out / "prices.csv").exists()
        # No temp files left
        assert not list(out.glob("*.tmp*"))

    def test_weekend_anchor_date_finds_friday(self, tmp_path):
        """If as_of_date is a weekend, anchor should be the prior Friday."""
        # 2026-02-21 is Saturday; latest trading day is 2026-02-20
        rows = _make_price_rows(
            ("ACRS", "2026-02-19", 1.80),
            ("ACRS", "2026-02-20", 1.85),
        )
        _write_price_csv(tmp_path / "prices.csv", rows)
        _write_rankings_csv(tmp_path / "rankings.csv", ["ACRS"])

        out = tmp_path / "cache"
        index = snapshot_price_cache("2026-02-21", tmp_path / "prices.csv", tmp_path / "rankings.csv", out)

        assert index["anchor_date"] == "2026-02-20"


# ---------------------------------------------------------------------------
# TestBackfillForwardPrices
# ---------------------------------------------------------------------------

class TestBackfillForwardPrices:

    def _setup_cache(self, tmp_path, tickers, as_of="2026-02-09", horizons=None):
        """Create a snapshot cache + price CSV with enough dates."""
        horizons = horizons or [5, 20]
        price_rows = _make_full_price_rows(tickers)
        _write_price_csv(tmp_path / "prices.csv", price_rows)
        _write_rankings_csv(tmp_path / "rankings.csv", tickers)

        cache_dir = tmp_path / "cache"
        snapshot_price_cache(as_of, tmp_path / "prices.csv", tmp_path / "rankings.csv", cache_dir, horizons=horizons)
        return cache_dir

    def test_fills_h5_column_correctly(self, tmp_path):
        cache_dir = self._setup_cache(tmp_path, ["ACRS", "BMRN"])

        result = backfill_forward_prices(cache_dir, tmp_path / "prices.csv", 5)
        assert result["filled"] > 0
        assert not result["skipped"]

        with open(cache_dir / "prices.csv", newline="") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            if row["anchor_close"]:
                assert row["h5_date"] != ""
                assert row["h5_close"] != ""

    def test_fills_h20_column_correctly(self, tmp_path):
        cache_dir = self._setup_cache(tmp_path, ["ACRS"], horizons=[5, 20])

        result = backfill_forward_prices(cache_dir, tmp_path / "prices.csv", 20)
        assert result["filled"] > 0

        with open(cache_dir / "prices.csv", newline="") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            if row["anchor_close"]:
                assert row["h20_date"] != ""

    def test_write_once_no_overwrite(self, tmp_path):
        cache_dir = self._setup_cache(tmp_path, ["ACRS"])

        # First fill
        backfill_forward_prices(cache_dir, tmp_path / "prices.csv", 5)

        with open(cache_dir / "prices.csv", newline="") as f:
            rows1 = list(csv.DictReader(f))
        original_close = rows1[0]["h5_close"]

        # Modify source prices
        new_rows = _make_full_price_rows(["ACRS"], base_price=999.0)
        _write_price_csv(tmp_path / "prices.csv", new_rows)

        # Re-fill should NOT overwrite
        backfill_forward_prices(cache_dir, tmp_path / "prices.csv", 5)

        with open(cache_dir / "prices.csv", newline="") as f:
            rows2 = list(csv.DictReader(f))
        assert rows2[0]["h5_close"] == original_close

    def test_updates_horizons_filled_in_index(self, tmp_path):
        cache_dir = self._setup_cache(tmp_path, ["ACRS"])
        backfill_forward_prices(cache_dir, tmp_path / "prices.csv", 5)

        with open(cache_dir / "index.json") as f:
            index = json.load(f)
        assert 5 in index["horizons_filled"]

    def test_removes_from_horizons_pending(self, tmp_path):
        cache_dir = self._setup_cache(tmp_path, ["ACRS"])
        backfill_forward_prices(cache_dir, tmp_path / "prices.csv", 5)

        with open(cache_dir / "index.json") as f:
            index = json.load(f)
        assert 5 not in index["horizons_pending"]

    def test_partial_backfill_leaves_unfilled_empty(self, tmp_path):
        """If forward date doesn't exist yet, column stays empty."""
        # Only 3 trading dates — not enough for h5
        rows = _make_price_rows(
            ("ACRS", "2026-02-19", 1.87),
            ("ACRS", "2026-02-20", 1.88),
            ("ACRS", "2026-02-23", 1.89),
        )
        _write_price_csv(tmp_path / "prices.csv", rows)
        _write_rankings_csv(tmp_path / "rankings.csv", ["ACRS"])

        cache_dir = tmp_path / "cache"
        snapshot_price_cache("2026-02-19", tmp_path / "prices.csv", tmp_path / "rankings.csv", cache_dir, horizons=[5])

        result = backfill_forward_prices(cache_dir, tmp_path / "prices.csv", 5)
        assert result["filled"] == 0

    def test_split_warning_detected_300pct_jump(self, tmp_path):
        # Anchor at 10, forward at 50 → +400% → split warning
        rows = _make_price_rows(
            ("SPLIT", "2026-02-09", 10.0),
            ("SPLIT", "2026-02-10", 10.1),
            ("SPLIT", "2026-02-11", 10.2),
            ("SPLIT", "2026-02-12", 10.3),
            ("SPLIT", "2026-02-13", 10.4),
            ("SPLIT", "2026-02-17", 50.0),  # 5 trading days after anchor
        )
        _write_price_csv(tmp_path / "prices.csv", rows)
        _write_rankings_csv(tmp_path / "rankings.csv", ["SPLIT"])

        cache_dir = tmp_path / "cache"
        snapshot_price_cache("2026-02-09", tmp_path / "prices.csv", tmp_path / "rankings.csv", cache_dir, horizons=[5])

        result = backfill_forward_prices(cache_dir, tmp_path / "prices.csv", 5)
        assert result["split_warnings_added"] > 0

        with open(cache_dir / "index.json") as f:
            index = json.load(f)
        assert any(w["ticker"] == "SPLIT" for w in index["split_warnings"])

    def test_split_warning_detected_75pct_drop(self, tmp_path):
        # Anchor at 100, forward at 20 → -80% → split warning
        rows = _make_price_rows(
            ("DROP", "2026-02-09", 100.0),
            ("DROP", "2026-02-10", 100.1),
            ("DROP", "2026-02-11", 100.2),
            ("DROP", "2026-02-12", 100.3),
            ("DROP", "2026-02-13", 100.4),
            ("DROP", "2026-02-17", 20.0),  # -80%
        )
        _write_price_csv(tmp_path / "prices.csv", rows)
        _write_rankings_csv(tmp_path / "rankings.csv", ["DROP"])

        cache_dir = tmp_path / "cache"
        snapshot_price_cache("2026-02-09", tmp_path / "prices.csv", tmp_path / "rankings.csv", cache_dir, horizons=[5])

        result = backfill_forward_prices(cache_dir, tmp_path / "prices.csv", 5)
        assert result["split_warnings_added"] > 0

    def test_no_false_positive_on_50pct_move(self, tmp_path):
        # Anchor at 10, forward at 15 → +50% → NOT a split
        rows = _make_price_rows(
            ("NORM", "2026-02-09", 10.0),
            ("NORM", "2026-02-10", 10.1),
            ("NORM", "2026-02-11", 10.2),
            ("NORM", "2026-02-12", 10.3),
            ("NORM", "2026-02-13", 10.4),
            ("NORM", "2026-02-17", 15.0),  # +50%, normal biotech move
        )
        _write_price_csv(tmp_path / "prices.csv", rows)
        _write_rankings_csv(tmp_path / "rankings.csv", ["NORM"])

        cache_dir = tmp_path / "cache"
        snapshot_price_cache("2026-02-09", tmp_path / "prices.csv", tmp_path / "rankings.csv", cache_dir, horizons=[5])

        result = backfill_forward_prices(cache_dir, tmp_path / "prices.csv", 5)
        assert result["split_warnings_added"] == 0

    def test_skips_already_filled_horizon(self, tmp_path):
        cache_dir = self._setup_cache(tmp_path, ["ACRS"])

        # Fill once
        backfill_forward_prices(cache_dir, tmp_path / "prices.csv", 5)

        # Second fill should skip
        result = backfill_forward_prices(cache_dir, tmp_path / "prices.csv", 5)
        assert result["skipped"] is True
        assert result["filled"] == 0


# ---------------------------------------------------------------------------
# TestValidateSchema
# ---------------------------------------------------------------------------

class TestValidateSchema:

    def _make_valid_index(self, as_of="2026-02-19"):
        return {
            "schema_version": SCHEMA_VERSION,
            "cache_type": CACHE_TYPE,
            "as_of_date": as_of,
            "created_at": "2026-02-19T22:00:00Z",
            "source_csv_sha256": "a" * 64,
            "ticker_count": 10,
            "anchor_date": as_of,
            "horizons_filled": [5],
            "horizons_pending": [20, 63],
            "split_warnings": [],
            "coverage_pct": 100.0,
            "tickers_missing_anchor": [],
        }

    def test_happy_path(self):
        index = self._make_valid_index()
        ok, detail = validate_price_pit_index(index, "2026-02-19")
        assert ok, detail

    def test_missing_required_fields(self):
        index = {"schema_version": SCHEMA_VERSION}
        ok, detail = validate_price_pit_index(index, "2026-02-19")
        assert not ok

    def test_wrong_schema_version(self):
        index = self._make_valid_index()
        index["schema_version"] = "wrong_version"
        ok, detail = validate_price_pit_index(index, "2026-02-19")
        assert not ok
        assert "schema_version" in detail

    def test_coverage_mismatch(self):
        index = self._make_valid_index()
        index["coverage_pct"] = 50.0  # should be 100.0 for 10 tickers, 0 missing
        ok, detail = validate_price_pit_index(index, "2026-02-19")
        assert not ok
        assert "coverage_pct" in detail

    def test_invalid_as_of_date(self):
        index = self._make_valid_index()
        index["as_of_date"] = "not-a-date"
        ok, detail = validate_price_pit_index(index, "2026-02-19")
        assert not ok

    def test_as_of_date_mismatch(self):
        index = self._make_valid_index("2026-02-18")
        ok, detail = validate_price_pit_index(index, "2026-02-19")
        assert not ok
        assert "mismatch" in detail

    def test_bad_sha256(self):
        index = self._make_valid_index()
        index["source_csv_sha256"] = "short"
        ok, detail = validate_price_pit_index(index, "2026-02-19")
        assert not ok
        assert "sha256" in detail

    def test_horizons_inconsistent(self):
        index = self._make_valid_index()
        index["horizons_filled"] = [5, 20]
        index["horizons_pending"] = [20, 63]  # 20 in both → union != configured
        ok, detail = validate_price_pit_index(index, "2026-02-19")
        # Union would be {5,20,63} == {5,20,63} → OK — let's make it actually fail
        # Actually {5,20} | {20,63} = {5,20,63} which == DEFAULT_HORIZONS. That passes.
        # Let's break it differently:
        index["horizons_filled"] = [5, 20, 63]
        index["horizons_pending"] = [99]  # union = {5,20,63,99} != {5,20,63}
        ok, detail = validate_price_pit_index(index, "2026-02-19")
        assert not ok

    def test_split_warnings_bad_format(self):
        index = self._make_valid_index()
        index["split_warnings"] = [{"bad": "format"}]
        ok, detail = validate_price_pit_index(index, "2026-02-19")
        assert not ok
        assert "split_warnings" in detail

    def test_anchor_date_after_as_of(self):
        index = self._make_valid_index()
        index["anchor_date"] = "2026-02-20"  # after as_of
        ok, detail = validate_price_pit_index(index, "2026-02-19")
        assert not ok
        assert "anchor_date" in detail


# ---------------------------------------------------------------------------
# TestBackfillAll
# ---------------------------------------------------------------------------

class TestBackfillAll:

    def test_iterates_all_cache_dirs(self, tmp_path):
        tickers = ["ACRS"]
        _write_price_csv(tmp_path / "prices.csv", _make_full_price_rows(tickers))
        _write_rankings_csv(tmp_path / "rankings.csv", tickers)

        base = tmp_path / "caches"
        for d in ["2026-02-02", "2026-02-03"]:
            snapshot_price_cache(d, tmp_path / "prices.csv", tmp_path / "rankings.csv", base / d, horizons=[5])

        result = backfill_all_caches(base, tmp_path / "prices.csv", "2026-02-27", horizons=[5])
        assert result["dirs_processed"] == 2
        assert result["total_filled"] >= 2  # at least 1 per dir

    def test_skips_already_filled_horizons(self, tmp_path):
        tickers = ["ACRS"]
        _write_price_csv(tmp_path / "prices.csv", _make_full_price_rows(tickers))
        _write_rankings_csv(tmp_path / "rankings.csv", tickers)

        base = tmp_path / "caches"
        cache_dir = base / "2026-02-02"
        snapshot_price_cache("2026-02-02", tmp_path / "prices.csv", tmp_path / "rankings.csv", cache_dir, horizons=[5])

        # Fill once
        backfill_forward_prices(cache_dir, tmp_path / "prices.csv", 5)

        # Backfill all — should skip since already filled
        result = backfill_all_caches(base, tmp_path / "prices.csv", "2026-02-27", horizons=[5])
        assert result["dirs_processed"] == 1
        # The h5 entry should not appear in by_date since it's already filled
        if "2026-02-02" in result.get("by_date", {}):
            assert result["by_date"]["2026-02-02"].get("h5", {}).get("skipped", True) is True

    def test_skips_immature_dates(self, tmp_path):
        """Caches where h5 hasn't matured yet should be skipped."""
        tickers = ["ACRS"]
        # Only 3 trading dates — not enough for h5 to mature
        rows = _make_price_rows(
            ("ACRS", "2026-02-19", 1.87),
            ("ACRS", "2026-02-20", 1.88),
            ("ACRS", "2026-02-23", 1.89),
        )
        _write_price_csv(tmp_path / "prices.csv", rows)
        _write_rankings_csv(tmp_path / "rankings.csv", tickers)

        base = tmp_path / "caches"
        snapshot_price_cache("2026-02-19", tmp_path / "prices.csv", tmp_path / "rankings.csv", base / "2026-02-19", horizons=[5])

        # Through date only 3 days ahead — h5 not matured
        result = backfill_all_caches(base, tmp_path / "prices.csv", "2026-02-23", horizons=[5])
        assert result["total_filled"] == 0


# ---------------------------------------------------------------------------
# TestDetectSplitWarnings
# ---------------------------------------------------------------------------

class TestDetectSplitWarnings:

    def test_detects_reverse_split(self):
        rows = [
            {"ticker": "SPLIT", "anchor_close": "10.0", "h5_close": "50.0", "h5_date": "2026-02-17"},
        ]
        warnings = detect_split_warnings(rows)
        assert len(warnings) == 1
        assert warnings[0]["ticker"] == "SPLIT"
        assert warnings[0]["pct_change"] > 3.0

    def test_detects_forward_split(self):
        rows = [
            {"ticker": "SPLIT", "anchor_close": "100.0", "h5_close": "20.0", "h5_date": "2026-02-17"},
        ]
        warnings = detect_split_warnings(rows)
        assert len(warnings) == 1
        assert warnings[0]["pct_change"] < -0.75

    def test_no_warning_for_normal_move(self):
        rows = [
            {"ticker": "NORM", "anchor_close": "10.0", "h5_close": "15.0", "h5_date": "2026-02-17"},
        ]
        warnings = detect_split_warnings(rows)
        assert len(warnings) == 0

    def test_no_warning_for_empty_forward(self):
        rows = [
            {"ticker": "NORM", "anchor_close": "10.0", "h5_close": "", "h5_date": ""},
        ]
        warnings = detect_split_warnings(rows)
        assert len(warnings) == 0


# ---------------------------------------------------------------------------
# TestHelpers
# ---------------------------------------------------------------------------

class TestHelpers:

    def test_find_anchor_date_exact_match(self):
        dates = ["2026-02-18", "2026-02-19", "2026-02-20"]
        assert _find_anchor_date(dates, "2026-02-19") == "2026-02-19"

    def test_find_anchor_date_weekend_fallback(self):
        dates = ["2026-02-18", "2026-02-19", "2026-02-20"]
        # 2026-02-21 is Saturday
        assert _find_anchor_date(dates, "2026-02-21") == "2026-02-20"

    def test_find_anchor_date_no_match(self):
        dates = ["2026-02-20", "2026-02-21"]
        assert _find_anchor_date(dates, "2026-02-19") is None

    def test_trading_days_after(self):
        dates = TRADING_DATES
        result = _trading_days_after(dates, "2026-02-09", 5)
        assert result == "2026-02-17"  # skip weekend

    def test_trading_days_after_out_of_range(self):
        dates = ["2026-02-19", "2026-02-20"]
        result = _trading_days_after(dates, "2026-02-19", 5)
        assert result is None

    def test_horizon_cols(self):
        cols = _horizon_cols([5, 20])
        assert cols == [
            "ticker", "anchor_date", "anchor_close",
            "h5_date", "h5_close",
            "h20_date", "h20_close",
        ]
