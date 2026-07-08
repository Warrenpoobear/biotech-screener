#!/usr/bin/env python3
"""Tests for tools/forward_eval_gate.py — forward-return rolling IC gate.

Covers:
  - Rolling IC: positive IC → PASS, negative IC → WARN, cold-start → PASS
  - Split exclusion
  - Gate integration with run_daily_production
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import date
from pathlib import Path
from typing import Dict, List

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from forward_eval_gate import (
    _discover_pit_dates,
    _get_split_warning_tickers,
    _load_pit_prices,
    _load_rankings_signal,
    evaluate_rolling_ic,
)
from run_daily_production import GATE_ALLOWLIST, GateConfig, check_forward_eval

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_rankings_csv(
    path: Path, tickers_ranks: List[tuple], eligible=None, weights=None, with_scope_cols=True
) -> None:
    """Write rankings.csv with (ticker, rank) pairs.

    ``eligible``: None → all rows eligible; else an iterable of eligible tickers.
    ``weights``: optional {ticker: target_weight_pct}; default 0.
    ``with_scope_cols``: when False, omit eligible/target_weight_pct columns
    entirely (used to test the scope-fallback path).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    elig_set = None if eligible is None else set(eligible)
    weights = weights or {}
    fieldnames = ["ticker", "actionable_rank"]
    if with_scope_cols:
        fieldnames += ["eligible", "target_weight_pct"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t, r in tickers_ranks:
            row = {"ticker": t, "actionable_rank": str(r)}
            if with_scope_cols:
                row["eligible"] = "True" if (elig_set is None or t in elig_set) else "False"
                row["target_weight_pct"] = str(weights.get(t, 0))
            writer.writerow(row)


def _write_pit_cache(cache_dir: Path, as_of: str, rows: List[Dict], horizons_filled=None, split_warnings=None):
    """Write a complete PIT cache (index.json + prices.csv)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    horizons_filled = horizons_filled or [5]

    index = {
        "schema_version": "price_pit_index.v1",
        "cache_type": "price_pit",
        "as_of_date": as_of,
        "created_at": f"{as_of}T22:00:00Z",
        "source_csv_sha256": "a" * 64,
        "ticker_count": len(rows),
        "anchor_date": as_of,
        "horizons_filled": horizons_filled,
        "horizons_pending": [h for h in [5, 20, 63] if h not in horizons_filled],
        "split_warnings": split_warnings or [],
        "coverage_pct": 100.0,
        "tickers_missing_anchor": [],
    }
    with open(cache_dir / "index.json", "w") as f:
        json.dump(index, f, indent=2)

    fieldnames = ["ticker", "anchor_date", "anchor_close", "h5_date", "h5_close"]
    with open(cache_dir / "prices.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _make_positive_ic_data(n_tickers=20):
    """Create data where better-ranked tickers have higher returns (positive IC)."""
    tickers = [f"T{i:03d}" for i in range(n_tickers)]
    tickers_ranks = [(t, i + 1) for i, t in enumerate(tickers)]

    # Better rank → higher forward return (monotonic)
    pit_rows = []
    for i, t in enumerate(tickers):
        anchor = 10.0
        # Higher rank (lower number) → higher return
        fwd = anchor * (1.0 + 0.10 - i * 0.01)
        pit_rows.append(
            {
                "ticker": t,
                "anchor_date": "2026-02-09",
                "anchor_close": str(anchor),
                "h5_date": "2026-02-17",
                "h5_close": str(round(fwd, 4)),
            }
        )
    return tickers_ranks, pit_rows


def _make_negative_ic_data(n_tickers=20):
    """Create data where better-ranked tickers have LOWER returns (negative IC)."""
    tickers = [f"T{i:03d}" for i in range(n_tickers)]
    tickers_ranks = [(t, i + 1) for i, t in enumerate(tickers)]

    pit_rows = []
    for i, t in enumerate(tickers):
        anchor = 10.0
        # Higher rank (lower number) → LOWER return (inverted)
        fwd = anchor * (1.0 - 0.10 + i * 0.01)
        pit_rows.append(
            {
                "ticker": t,
                "anchor_date": "2026-02-09",
                "anchor_close": str(anchor),
                "h5_date": "2026-02-17",
                "h5_close": str(round(fwd, 4)),
            }
        )
    return tickers_ranks, pit_rows


# ---------------------------------------------------------------------------
# TestEvaluateRollingIC
# ---------------------------------------------------------------------------


class TestEvaluateRollingIC:

    def test_positive_ic_returns_pass(self, tmp_path):
        """When ranking predicts returns, IC > 0 → PASS."""
        snapshot_dir = tmp_path / "snapshots"
        cache_base = tmp_path / "caches"

        tickers_ranks, pit_rows = _make_positive_ic_data()

        # Create 5 snapshot dates with positive IC data
        # Spaced >= 7 days apart so the default de-overlap (horizon 5 → 7d gap) keeps all 5.
        dates = ["2026-01-05", "2026-01-12", "2026-01-19", "2026-01-26", "2026-02-02"]
        for d in dates:
            _write_rankings_csv(snapshot_dir / d / "rankings.csv", tickers_ranks)
            # Adjust pit_rows anchor_date for each
            rows = [{**r, "anchor_date": d} for r in pit_rows]
            _write_pit_cache(cache_base / d, d, rows, horizons_filled=[5])

        status, detail, value, threshold = evaluate_rolling_ic(
            snapshot_dir=snapshot_dir,
            price_cache_base=cache_base,
            current_date="2026-02-09",
            horizon=5,
            lookback_n=5,
            min_dates=3,
        )
        assert status == "PASS"
        assert value["mean_ic"] > 0
        assert value["n_evaluated"] >= 3

    def test_negative_ic_returns_warn(self, tmp_path):
        """When ranking inversely predicts returns, IC < 0 → WARN."""
        snapshot_dir = tmp_path / "snapshots"
        cache_base = tmp_path / "caches"

        tickers_ranks, pit_rows = _make_negative_ic_data()

        # Spaced >= 7 days apart so the default de-overlap (horizon 5 → 7d gap) keeps all 5.
        dates = ["2026-01-05", "2026-01-12", "2026-01-19", "2026-01-26", "2026-02-02"]
        for d in dates:
            _write_rankings_csv(snapshot_dir / d / "rankings.csv", tickers_ranks)
            rows = [{**r, "anchor_date": d} for r in pit_rows]
            _write_pit_cache(cache_base / d, d, rows, horizons_filled=[5])

        status, detail, value, threshold = evaluate_rolling_ic(
            snapshot_dir=snapshot_dir,
            price_cache_base=cache_base,
            current_date="2026-02-09",
            horizon=5,
            lookback_n=5,
            min_dates=3,
            ic_warn_floor=0.00,
        )
        assert status == "WARN"
        assert value["mean_ic"] < 0

    def test_cold_start_returns_pass(self, tmp_path):
        """No PIT caches at all → PASS with cold-start."""
        snapshot_dir = tmp_path / "snapshots"
        cache_base = tmp_path / "caches"
        cache_base.mkdir(parents=True)

        status, detail, value, threshold = evaluate_rolling_ic(
            snapshot_dir=snapshot_dir,
            price_cache_base=cache_base,
            current_date="2026-02-09",
            horizon=5,
        )
        assert status == "PASS"
        assert "cold-start" in detail

    def test_insufficient_dates_returns_warn(self, tmp_path):
        """Fewer than min_dates evaluated → WARN."""
        snapshot_dir = tmp_path / "snapshots"
        cache_base = tmp_path / "caches"

        tickers_ranks, pit_rows = _make_positive_ic_data()

        # Only 1 date, but min_dates=3
        d = "2026-02-02"
        _write_rankings_csv(snapshot_dir / d / "rankings.csv", tickers_ranks)
        rows = [{**r, "anchor_date": d} for r in pit_rows]
        _write_pit_cache(cache_base / d, d, rows, horizons_filled=[5])

        status, detail, value, threshold = evaluate_rolling_ic(
            snapshot_dir=snapshot_dir,
            price_cache_base=cache_base,
            current_date="2026-02-09",
            horizon=5,
            min_dates=3,
        )
        assert status == "WARN"
        assert "insufficient dates" in detail

    def test_split_warning_tickers_excluded(self, tmp_path):
        """Tickers with split warnings should be excluded from IC."""
        snapshot_dir = tmp_path / "snapshots"
        cache_base = tmp_path / "caches"

        tickers_ranks, pit_rows = _make_positive_ic_data(n_tickers=10)

        d = "2026-02-02"
        _write_rankings_csv(snapshot_dir / d / "rankings.csv", tickers_ranks)
        # Mark first ticker as split warning
        _write_pit_cache(
            cache_base / d,
            d,
            pit_rows,
            horizons_filled=[5],
            split_warnings=[{"ticker": "T000", "horizon": 5}],
        )

        status, detail, value, threshold = evaluate_rolling_ic(
            snapshot_dir=snapshot_dir,
            price_cache_base=cache_base,
            current_date="2026-02-09",
            horizon=5,
            min_dates=1,
        )
        # T000 should be excluded
        for dd in value.get("date_details", []):
            if dd.get("n_split_excluded") is not None:
                assert dd["n_split_excluded"] >= 1

    def test_reuses_spearman_ic_from_eval(self):
        """Verify we import spearman_ic from eval_forward_returns, not reimplement."""
        from scripts.eval_forward_returns import spearman_ic as eval_ic
        from tools.forward_eval_gate import spearman_ic as gate_ic

        assert eval_ic is gate_ic


# ---------------------------------------------------------------------------
# TestCheckForwardEval
# ---------------------------------------------------------------------------


class TestCheckForwardEval:

    def test_wired_to_gate_allowlist(self):
        assert "forward_eval" in GATE_ALLOWLIST

    def test_price_pit_cache_in_allowlist(self):
        assert "price_pit_cache" in GATE_ALLOWLIST

    def test_never_returns_fail(self, tmp_path):
        """check_forward_eval should never return FAIL."""
        snapshot_dir = tmp_path / "snapshots"
        cache_base = tmp_path / "caches"
        cache_base.mkdir(parents=True)

        config = GateConfig()
        result = check_forward_eval(snapshot_dir, cache_base, "2026-02-09", config)
        assert result.status in ("PASS", "WARN")
        assert result.name == "forward_eval"

    def test_gate_config_thresholds_respected(self, tmp_path):
        """Custom GateConfig thresholds should be passed through."""
        snapshot_dir = tmp_path / "snapshots"
        cache_base = tmp_path / "caches"
        cache_base.mkdir(parents=True)

        config = GateConfig(
            forward_eval_ic_warn_floor=0.10,
            forward_eval_lookback_n=5,
            forward_eval_horizon=5,
        )
        result = check_forward_eval(snapshot_dir, cache_base, "2026-02-09", config)
        assert result.threshold["ic_warn_floor"] == 0.10
        assert result.threshold["horizon"] == 5

    def test_cold_start_is_pass(self, tmp_path):
        """Empty cache base → PASS (cold start)."""
        snapshot_dir = tmp_path / "snapshots"
        cache_base = tmp_path / "caches"
        cache_base.mkdir(parents=True)

        config = GateConfig()
        result = check_forward_eval(snapshot_dir, cache_base, "2026-02-09", config)
        assert result.status == "PASS"
        assert "cold-start" in result.detail


# ---------------------------------------------------------------------------
# TestDiscoverPitDates
# ---------------------------------------------------------------------------


class TestDiscoverPitDates:

    def test_returns_dates_before_current(self, tmp_path):
        for d in ["2026-02-01", "2026-02-02", "2026-02-03"]:
            cache = tmp_path / d
            cache.mkdir(parents=True)
            with open(cache / "index.json", "w") as f:
                json.dump({"horizons_filled": [5]}, f)

        dates = _discover_pit_dates(tmp_path, "2026-02-03", 5, 10)
        assert dates == ["2026-02-02", "2026-02-01"]
        assert "2026-02-03" not in dates

    def test_respects_lookback_limit(self, tmp_path):
        for d in ["2026-02-01", "2026-02-02", "2026-02-03"]:
            cache = tmp_path / d
            cache.mkdir(parents=True)
            with open(cache / "index.json", "w") as f:
                json.dump({"horizons_filled": [5]}, f)

        dates = _discover_pit_dates(tmp_path, "2026-02-04", 5, 2)
        assert len(dates) == 2

    def test_skips_unfilled_horizons(self, tmp_path):
        cache = tmp_path / "2026-02-01"
        cache.mkdir(parents=True)
        with open(cache / "index.json", "w") as f:
            json.dump({"horizons_filled": []}, f)  # h5 not filled

        dates = _discover_pit_dates(tmp_path, "2026-02-02", 5, 10)
        assert len(dates) == 0


# ---------------------------------------------------------------------------
# TestDeoverlap — issue #482: rolling window must not double-count episodes
# ---------------------------------------------------------------------------


class TestDeoverlap:

    def _mk(self, tmp_path):
        for d in ["2026-02-02", "2026-02-03", "2026-02-04", "2026-02-05", "2026-02-06"]:
            cache = tmp_path / d
            cache.mkdir(parents=True)
            with open(cache / "index.json", "w") as f:
                json.dump({"horizons_filled": [5]}, f)

    def test_no_gap_returns_consecutive(self, tmp_path):
        self._mk(tmp_path)
        dates = _discover_pit_dates(tmp_path, "2026-02-10", 5, 10, min_gap_days=0)
        assert dates == ["2026-02-06", "2026-02-05", "2026-02-04", "2026-02-03", "2026-02-02"]

    def test_gap_deoverlaps_adjacent_dates(self, tmp_path):
        self._mk(tmp_path)
        # From newest 02-06: keep; skip 02-05/04 (<3d); keep 02-03; skip 02-02.
        dates = _discover_pit_dates(tmp_path, "2026-02-10", 5, 10, min_gap_days=3)
        assert dates == ["2026-02-06", "2026-02-03"]
        for a, b in zip(dates, dates[1:]):
            gap = (date.fromisoformat(a) - date.fromisoformat(b)).days
            assert gap >= 3

    def test_default_gap_derived_from_horizon(self, tmp_path):
        # Consecutive daily caches; default de-overlap (horizon 5 → 7d gap) collapses
        # them to a single episode, so evaluate reports insufficient dates rather than
        # a spurious high-n window.
        snapshot_dir = tmp_path / "snapshots"
        cache_base = tmp_path / "caches"
        tickers_ranks, pit_rows = _make_positive_ic_data()
        for d in ["2026-02-02", "2026-02-03", "2026-02-04", "2026-02-05", "2026-02-06"]:
            _write_rankings_csv(snapshot_dir / d / "rankings.csv", tickers_ranks)
            rows = [{**r, "anchor_date": d} for r in pit_rows]
            _write_pit_cache(cache_base / d, d, rows, horizons_filled=[5])
        status, detail, value, threshold = evaluate_rolling_ic(
            snapshot_dir=snapshot_dir,
            price_cache_base=cache_base,
            current_date="2026-02-10",
            horizon=5,
            lookback_n=10,
            min_dates=3,
        )
        assert value["min_gap_days"] == 7  # round(5 * 7 / 5)
        assert value["n_evaluated"] == 1  # 5 adjacent days → 1 independent window


# ---------------------------------------------------------------------------
# TestScope — issue #482: IC must measure the tradeable set, not full universe
# ---------------------------------------------------------------------------


class TestScope:

    def test_eligible_scope_excludes_ineligible(self, tmp_path):
        ranks = [("A", 1), ("B", 2), ("C", 3), ("D", 4)]
        _write_rankings_csv(tmp_path / "rankings.csv", ranks, eligible={"A", "B"})
        universe = _load_rankings_signal(tmp_path, scope="universe")
        eligible = _load_rankings_signal(tmp_path, scope="eligible")
        assert set(universe) == {"A", "B", "C", "D"}
        assert set(eligible) == {"A", "B"}

    def test_portfolio_scope_only_weighted(self, tmp_path):
        ranks = [("A", 1), ("B", 2), ("C", 3)]
        _write_rankings_csv(tmp_path / "rankings.csv", ranks, weights={"A": 0.5, "B": 0.0})
        port = _load_rankings_signal(tmp_path, scope="portfolio")
        assert set(port) == {"A"}

    def test_top_k_scope(self, tmp_path):
        ranks = [("A", 1), ("B", 2), ("C", 3), ("D", 4)]
        _write_rankings_csv(tmp_path / "rankings.csv", ranks)
        top2 = _load_rankings_signal(tmp_path, scope="top_k", top_k=2)
        assert set(top2) == {"A", "B"}

    def test_missing_column_falls_back_to_universe(self, tmp_path):
        ranks = [("A", 1), ("B", 2), ("C", 3)]
        _write_rankings_csv(tmp_path / "rankings.csv", ranks, with_scope_cols=False)
        # scope='eligible' but no 'eligible' column → must not blank the gate
        sig = _load_rankings_signal(tmp_path, scope="eligible")
        assert set(sig) == {"A", "B", "C"}

    def test_evaluate_records_scope(self, tmp_path):
        snapshot_dir = tmp_path / "snapshots"
        cache_base = tmp_path / "caches"
        tickers_ranks, pit_rows = _make_positive_ic_data()
        d = "2026-02-02"
        _write_rankings_csv(snapshot_dir / d / "rankings.csv", tickers_ranks)
        rows = [{**r, "anchor_date": d} for r in pit_rows]
        _write_pit_cache(cache_base / d, d, rows, horizons_filled=[5])
        _, _, value, threshold = evaluate_rolling_ic(
            snapshot_dir=snapshot_dir,
            price_cache_base=cache_base,
            current_date="2026-02-09",
            horizon=5,
            min_dates=1,
        )
        assert value["scope"] == "eligible"
        assert threshold["scope"] == "eligible"
