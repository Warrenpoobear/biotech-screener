#!/usr/bin/env python3
"""Tests for tools/snapshot_preflight.py — snapshot eligibility preflight.

Covers:
  - Check 1: rankings exist (missing, empty, missing cols, low cols, pass)
  - Check 2: eligible count (too few, sufficient, missing file)
  - Check 3: hydration coverage (no source cols, low coverage, all hydrated, no eligible, mixed)
  - Check 4: PIT price cache (missing, corrupt, invalid schema, valid)
  - Check 5: split warnings (no cache, warnings present, clean)
  - Orchestrator: all pass, one warn, short-circuit, pit optional
  - Batch: multi-date summary, date range filter, empty root
  - Skip reason format
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from snapshot_preflight import (
    CheckResult,
    PreflightReport,
    SnapshotPreflightResult,
    check_eligible_count,
    check_hydration_coverage,
    check_pit_price_cache,
    check_rankings_exist,
    check_split_warnings,
    run_preflight,
    run_preflight_batch,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_rankings(
    snap_dir: Path,
    rows: List[Dict[str, str]],
    fieldnames: Optional[List[str]] = None,
) -> None:
    """Write a rankings.csv with the given rows."""
    snap_dir.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else ["ticker"]
    with open(snap_dir / "rankings.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _make_full_row(
    ticker: str,
    eligible: str = "1",
    rank: str = "1",
    beta_src: str = "price_history",
    alpha_src: str = "price_history",
) -> Dict[str, str]:
    """Build a row with all required + hydration columns (25+ cols total)."""
    base = {
        "ticker": ticker,
        "eligible": eligible,
        "actionable_rank": rank,
        "de_beta_xbi_60d_source": beta_src,
        "de_alpha_60d_source": alpha_src,
    }
    # Pad to 25+ columns
    for i in range(20):
        base[f"col_{i}"] = ""
    return base


def _write_pit_index(
    cache_dir: Path,
    as_of: str,
    *,
    valid: bool = True,
    split_warnings: Optional[List[Dict]] = None,
) -> None:
    """Write a PIT price cache index.json."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    if valid:
        index = {
            "schema_version": "price_pit_index.v1",
            "cache_type": "price_pit",
            "as_of_date": as_of,
            "created_at": f"{as_of}T22:00:00Z",
            "source_csv_sha256": "a" * 64,
            "ticker_count": 10,
            "anchor_date": as_of,
            "horizons_filled": [5, 20, 63],
            "horizons_pending": [],
            "split_warnings": split_warnings or [],
            "coverage_pct": 100.0,
            "tickers_missing_anchor": [],
        }
    else:
        index = {"bad": "data"}
    with open(cache_dir / "index.json", "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)


# ---------------------------------------------------------------------------
# Check 1: rankings exist
# ---------------------------------------------------------------------------

class TestCheckRankingsExist:
    def test_missing_file(self, tmp_path: Path) -> None:
        r = check_rankings_exist(tmp_path)
        assert r.status == "FAIL"
        assert "missing" in r.detail

    def test_empty_file(self, tmp_path: Path) -> None:
        (tmp_path / "rankings.csv").write_text("")
        r = check_rankings_exist(tmp_path)
        assert r.status == "FAIL"
        assert "empty" in r.detail

    def test_missing_required_cols(self, tmp_path: Path) -> None:
        (tmp_path / "rankings.csv").write_text("ticker,some_col\n")
        r = check_rankings_exist(tmp_path)
        assert r.status == "FAIL"
        assert "missing required columns" in r.detail

    def test_low_col_count(self, tmp_path: Path) -> None:
        cols = ["ticker", "eligible", "actionable_rank", "a", "b"]
        (tmp_path / "rankings.csv").write_text(",".join(cols) + "\n")
        r = check_rankings_exist(tmp_path, min_cols=10)
        assert r.status == "WARN"
        assert "5 columns" in r.detail

    def test_pass(self, tmp_path: Path) -> None:
        rows = [_make_full_row("ACME")]
        _write_rankings(tmp_path, rows)
        r = check_rankings_exist(tmp_path, min_cols=5)
        assert r.status == "PASS"

    def test_header_only(self, tmp_path: Path) -> None:
        """Header with required cols but no data rows → PASS for this check."""
        cols = ["ticker", "eligible", "actionable_rank"] + [f"c{i}" for i in range(22)]
        (tmp_path / "rankings.csv").write_text(",".join(cols) + "\n")
        r = check_rankings_exist(tmp_path)
        assert r.status == "PASS"


# ---------------------------------------------------------------------------
# Check 2: eligible count
# ---------------------------------------------------------------------------

class TestCheckEligibleCount:
    def test_too_few(self, tmp_path: Path) -> None:
        rows = [_make_full_row(f"T{i}", eligible="1") for i in range(3)]
        _write_rankings(tmp_path, rows)
        r = check_eligible_count(tmp_path, min_eligible=10)
        assert r.status == "FAIL"
        assert "3 eligible" in r.detail

    def test_sufficient(self, tmp_path: Path) -> None:
        rows = [_make_full_row(f"T{i}", eligible="1") for i in range(15)]
        _write_rankings(tmp_path, rows)
        r = check_eligible_count(tmp_path)
        assert r.status == "PASS"

    def test_missing_file(self, tmp_path: Path) -> None:
        r = check_eligible_count(tmp_path)
        assert r.status == "FAIL"


# ---------------------------------------------------------------------------
# Check 3: hydration coverage
# ---------------------------------------------------------------------------

class TestCheckHydrationCoverage:
    def test_no_source_cols(self, tmp_path: Path) -> None:
        rows = [{"ticker": "A", "eligible": "1", "actionable_rank": "1"}]
        _write_rankings(tmp_path, rows)
        r = check_hydration_coverage(tmp_path)
        assert r.status == "WARN"
        assert "pre-hydration" in r.detail

    def test_low_coverage(self, tmp_path: Path) -> None:
        rows = [
            _make_full_row("A", beta_src="price_history", alpha_src="price_history"),
            _make_full_row("B", beta_src="missing", alpha_src="missing"),
            _make_full_row("C", beta_src="missing", alpha_src="missing"),
            _make_full_row("D", beta_src="missing", alpha_src="missing"),
        ]
        _write_rankings(tmp_path, rows)
        r = check_hydration_coverage(tmp_path, warn_threshold=0.50)
        assert r.status == "WARN"
        assert "25%" in r.detail

    def test_all_hydrated(self, tmp_path: Path) -> None:
        rows = [_make_full_row(f"T{i}") for i in range(10)]
        _write_rankings(tmp_path, rows)
        r = check_hydration_coverage(tmp_path)
        assert r.status == "PASS"
        assert "100%" in r.detail

    def test_no_eligible(self, tmp_path: Path) -> None:
        rows = [_make_full_row("A", eligible="0")]
        _write_rankings(tmp_path, rows)
        r = check_hydration_coverage(tmp_path)
        assert r.status == "WARN"
        assert "no eligible" in r.detail

    def test_mixed_coverage_above_threshold(self, tmp_path: Path) -> None:
        rows = [_make_full_row(f"T{i}") for i in range(9)]
        rows.append(_make_full_row("BAD", beta_src="missing", alpha_src="missing"))
        _write_rankings(tmp_path, rows)
        r = check_hydration_coverage(tmp_path, warn_threshold=0.80)
        assert r.status == "PASS"


# ---------------------------------------------------------------------------
# Check 4: PIT price cache
# ---------------------------------------------------------------------------

class TestCheckPitPriceCache:
    def test_missing_cache(self, tmp_path: Path) -> None:
        r = check_pit_price_cache("2025-01-15", tmp_path)
        assert r.status == "WARN"
        assert "no PIT cache" in r.detail

    def test_corrupt_index(self, tmp_path: Path) -> None:
        d = tmp_path / "2025-01-15"
        d.mkdir()
        (d / "index.json").write_text("not json{{{")
        r = check_pit_price_cache("2025-01-15", tmp_path)
        assert r.status == "FAIL"
        assert "unreadable" in r.detail

    def test_invalid_schema(self, tmp_path: Path) -> None:
        _write_pit_index(tmp_path / "2025-01-15", "2025-01-15", valid=False)
        r = check_pit_price_cache("2025-01-15", tmp_path)
        assert r.status == "FAIL"
        assert "schema" in r.detail

    def test_valid(self, tmp_path: Path) -> None:
        _write_pit_index(tmp_path / "2025-01-15", "2025-01-15")
        r = check_pit_price_cache("2025-01-15", tmp_path)
        assert r.status == "PASS"


# ---------------------------------------------------------------------------
# Check 5: split warnings
# ---------------------------------------------------------------------------

class TestCheckSplitWarnings:
    def test_no_cache(self, tmp_path: Path) -> None:
        r = check_split_warnings("2025-01-15", tmp_path)
        assert r.status == "PASS"
        assert "vacuous" in r.detail

    def test_warnings_present(self, tmp_path: Path) -> None:
        sw = [{"ticker": "ACME", "horizon": 20}, {"ticker": "BETA", "horizon": 63}]
        _write_pit_index(tmp_path / "2025-01-15", "2025-01-15", split_warnings=sw)
        r = check_split_warnings("2025-01-15", tmp_path, eval_horizons=[20])
        assert r.status == "WARN"
        assert "ACME" in r.detail
        # Only horizon-20 warning should be relevant
        assert "1 split warning" in r.detail

    def test_clean(self, tmp_path: Path) -> None:
        _write_pit_index(tmp_path / "2025-01-15", "2025-01-15", split_warnings=[])
        r = check_split_warnings("2025-01-15", tmp_path)
        assert r.status == "PASS"


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class TestRunPreflight:
    def test_all_pass(self, tmp_path: Path) -> None:
        snap = tmp_path / "2025-01-15"
        rows = [_make_full_row(f"T{i}") for i in range(15)]
        _write_rankings(snap, rows)
        pf = run_preflight(snap, "2025-01-15", min_cols=5)
        assert pf.status == "PASS"
        assert len(pf.checks) == 3  # rankings, eligible, hydration

    def test_one_warn(self, tmp_path: Path) -> None:
        snap = tmp_path / "2025-01-15"
        rows = [_make_full_row(f"T{i}", beta_src="missing", alpha_src="missing")
                for i in range(15)]
        _write_rankings(snap, rows)
        pf = run_preflight(snap, "2025-01-15", min_cols=5, warn_threshold=0.80)
        assert pf.status == "WARN"

    def test_short_circuit_on_missing_rankings(self, tmp_path: Path) -> None:
        snap = tmp_path / "2025-01-15"
        snap.mkdir(parents=True)
        pf = run_preflight(snap, "2025-01-15")
        assert pf.status == "FAIL"
        assert len(pf.checks) == 1  # only rankings check

    def test_pit_checks_optional(self, tmp_path: Path) -> None:
        snap = tmp_path / "2025-01-15"
        rows = [_make_full_row(f"T{i}") for i in range(15)]
        _write_rankings(snap, rows)
        pit_base = tmp_path / "pit"
        _write_pit_index(pit_base / "2025-01-15", "2025-01-15")

        pf_no_pit = run_preflight(snap, "2025-01-15", min_cols=5, check_pit=False)
        assert len(pf_no_pit.checks) == 3

        pf_pit = run_preflight(
            snap, "2025-01-15", min_cols=5,
            check_pit=True, pit_cache_base=pit_base,
        )
        assert len(pf_pit.checks) == 5


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------

class TestRunPreflightBatch:
    def test_multi_date_summary(self, tmp_path: Path) -> None:
        for d in ("2025-01-10", "2025-01-11", "2025-01-12"):
            snap = tmp_path / d
            rows = [_make_full_row(f"T{i}") for i in range(15)]
            _write_rankings(snap, rows)
        # Make one fail (delete the CSV)
        (tmp_path / "2025-01-11" / "rankings.csv").unlink()

        report = run_preflight_batch(tmp_path, min_cols=5)
        assert report.n_total == 3
        assert report.n_pass == 2
        assert report.n_fail == 1

    def test_date_range_filter(self, tmp_path: Path) -> None:
        for d in ("2025-01-10", "2025-01-15", "2025-01-20"):
            rows = [_make_full_row(f"T{i}") for i in range(15)]
            _write_rankings(tmp_path / d, rows)
        report = run_preflight_batch(
            tmp_path, date_from="2025-01-12", date_to="2025-01-18", min_cols=5,
        )
        assert report.n_total == 1
        assert report.results[0].date == "2025-01-15"

    def test_empty_root(self, tmp_path: Path) -> None:
        report = run_preflight_batch(tmp_path)
        assert report.n_total == 0


# ---------------------------------------------------------------------------
# Skip reason format
# ---------------------------------------------------------------------------

class TestSkipReasonFormat:
    def test_format_matches_eval_pattern(self, tmp_path: Path) -> None:
        """Verify skip reason can be formatted like eval_forward_returns PIT skip."""
        snap = tmp_path / "2025-01-15"
        snap.mkdir(parents=True)
        pf = run_preflight(snap, "2025-01-15")
        assert pf.status == "FAIL"

        # Build a skip reason the same way eval_forward_returns does
        details = "; ".join(c.detail for c in pf.checks if c.status == "FAIL")
        reason = f"preflight_FAIL: {details}"
        assert reason.startswith("preflight_FAIL:")
        assert "rankings.csv missing" in reason
