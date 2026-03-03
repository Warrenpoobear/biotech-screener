#!/usr/bin/env python3
"""Tests for tools/snapshot_preflight.py — snapshot eligibility preflight.

Covers:
  - Check 1: rankings exist (missing, empty, missing cols, low cols, pass)
  - Check 2: eligible count (too few, sufficient, missing file)
  - Check 3: hydration coverage (no source cols, low coverage, all hydrated, no eligible, mixed)
  - Check 4: PIT price cache (missing, corrupt, invalid schema, valid)
  - Check 5: split warnings (no cache, warnings present, clean)
  - Orchestrator: all pass, one warn, short-circuit, pit optional
  - Orchestrator metrics: eligible_n, rankings_cols_n, hydration_coverage_pct, pit_cache_status
  - Batch: multi-date summary, date range filter, empty root
  - Artifacts: summary CSV schema, manifest fields
  - Skip reason format
  - Audited backtest runner: smoke test
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
    SUMMARY_CSV_COLUMNS,
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
    write_preflight_artifacts,
    write_preflight_manifest,
    write_preflight_summary_csv,
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


# ---------------------------------------------------------------------------
# Metrics extraction
# ---------------------------------------------------------------------------

class TestPreflightMetrics:
    def test_eligible_n_extracted(self, tmp_path: Path) -> None:
        snap = tmp_path / "2025-01-15"
        rows = [_make_full_row(f"T{i}") for i in range(12)]
        _write_rankings(snap, rows)
        pf = run_preflight(snap, "2025-01-15", min_cols=5)
        assert pf.metrics["eligible_n"] == 12

    def test_rankings_cols_n_extracted(self, tmp_path: Path) -> None:
        snap = tmp_path / "2025-01-15"
        rows = [_make_full_row(f"T{i}") for i in range(12)]
        _write_rankings(snap, rows)
        pf = run_preflight(snap, "2025-01-15", min_cols=5)
        assert pf.metrics["rankings_cols_n"] == 25  # 5 named + 20 padding

    def test_hydration_coverage_pct(self, tmp_path: Path) -> None:
        snap = tmp_path / "2025-01-15"
        rows = [_make_full_row(f"T{i}") for i in range(10)]
        _write_rankings(snap, rows)
        pf = run_preflight(snap, "2025-01-15", min_cols=5)
        assert pf.metrics["hydration_coverage_pct"] == 100

    def test_pit_cache_status(self, tmp_path: Path) -> None:
        snap = tmp_path / "2025-01-15"
        rows = [_make_full_row(f"T{i}") for i in range(12)]
        _write_rankings(snap, rows)
        pit_base = tmp_path / "pit"
        _write_pit_index(pit_base / "2025-01-15", "2025-01-15")
        pf = run_preflight(
            snap, "2025-01-15", min_cols=5,
            check_pit=True, pit_cache_base=pit_base,
        )
        assert pf.metrics["pit_cache_status"] == "OK"
        assert pf.metrics["split_warning_count"] == 0

    def test_summary_row_format(self, tmp_path: Path) -> None:
        snap = tmp_path / "2025-01-15"
        rows = [_make_full_row(f"T{i}") for i in range(12)]
        _write_rankings(snap, rows)
        pf = run_preflight(snap, "2025-01-15", min_cols=5)
        row = pf.to_summary_row()
        assert row["date"] == "2025-01-15"
        assert row["status"] == "PASS"
        assert row["eligible_n"] == "12"


# ---------------------------------------------------------------------------
# Artifact writing
# ---------------------------------------------------------------------------

class TestPreflightArtifacts:
    def _make_report(self, tmp_path: Path) -> PreflightReport:
        """Build a report with 3 dates: 2 PASS, 1 FAIL."""
        for d in ("2025-01-10", "2025-01-11", "2025-01-12"):
            rows = [_make_full_row(f"T{i}") for i in range(15)]
            _write_rankings(tmp_path / d, rows)
        (tmp_path / "2025-01-11" / "rankings.csv").unlink()
        return run_preflight_batch(tmp_path, min_cols=5)

    def test_summary_csv_schema(self, tmp_path: Path) -> None:
        report = self._make_report(tmp_path / "snaps")
        out_dir = tmp_path / "artifacts"
        csv_path = write_preflight_summary_csv(report, out_dir)
        assert csv_path.exists()

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == SUMMARY_CSV_COLUMNS
            rows = list(reader)
        assert len(rows) == 3
        statuses = [r["status"] for r in rows]
        assert "PASS" in statuses
        assert "FAIL" in statuses

    def test_summary_csv_fail_reasons_populated(self, tmp_path: Path) -> None:
        report = self._make_report(tmp_path / "snaps")
        out_dir = tmp_path / "artifacts"
        write_preflight_summary_csv(report, out_dir)
        with open(out_dir / "preflight_summary.csv", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        fail_row = [r for r in rows if r["status"] == "FAIL"][0]
        assert "rankings.csv missing" in fail_row["fail_reasons"]

    def test_manifest_fields(self, tmp_path: Path) -> None:
        report = self._make_report(tmp_path / "snaps")
        out_dir = tmp_path / "artifacts"
        manifest_path = write_preflight_manifest(
            report, out_dir,
            run_id="test_run_001",
            snapshot_root=tmp_path / "snaps",
            date_from="2025-01-10",
            date_to="2025-01-12",
            horizons=[63],
            strict=False,
            file_hashes={"price_history_csv": "abc123"},
        )
        assert manifest_path.exists()
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        assert manifest["run_id"] == "test_run_001"
        assert manifest["counts"]["total"] == 3
        assert manifest["counts"]["fail"] == 1
        assert manifest["horizons"] == [63]
        assert manifest["file_hashes"]["price_history_csv"] == "abc123"
        assert "created_at" in manifest
        assert "git_sha" in manifest

    def test_write_preflight_artifacts_both_files(self, tmp_path: Path) -> None:
        report = self._make_report(tmp_path / "snaps")
        out_dir = tmp_path / "artifacts"
        csv_p, manifest_p = write_preflight_artifacts(
            report, out_dir,
            run_id="test_both",
            snapshot_root=tmp_path / "snaps",
        )
        assert csv_p.exists()
        assert manifest_p.exists()
        assert csv_p.name == "preflight_summary.csv"
        assert manifest_p.name == "preflight_manifest.json"

    def test_manifest_date_range_empty(self, tmp_path: Path) -> None:
        """Manifest with no date_from/date_to stores empty strings."""
        report = PreflightReport()
        out_dir = tmp_path / "artifacts"
        manifest_path = write_preflight_manifest(
            report, out_dir, run_id="empty_test",
        )
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        assert manifest["date_from"] == ""
        assert manifest["date_to"] == ""


# ---------------------------------------------------------------------------
# Audited backtest runner (smoke test)
# ---------------------------------------------------------------------------

class TestAuditedBacktestRunner:
    def test_smoke_tiny_snapshot_root(self, tmp_path: Path) -> None:
        """End-to-end smoke: preflight + eval on 2 synthetic dates."""
        # Create tiny snapshot root
        snap_root = tmp_path / "snapshots"
        price_csv = tmp_path / "prices.csv"

        # Write 2 dates with minimal data
        for d in ("2025-06-10", "2025-06-11"):
            snap = snap_root / d
            rows = [_make_full_row(f"T{i}", rank=str(i + 1)) for i in range(15)]
            _write_rankings(snap, rows)
            # Write metadata.json for PIT check
            meta = {"as_of_date": d}
            with open(snap / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(meta, f)

        # Write minimal price CSV (just enough for eval to not crash)
        with open(price_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Date"] + [f"T{i}" for i in range(15)])
            # Write a few price rows around our dates
            for day in range(5, 15):
                writer.writerow([f"2025-06-{day:02d}"] + [str(100 + i) for i in range(15)])

        out_root = tmp_path / "output"

        # Import and run
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "research"))
        from run_audited_backtest import run_audited_backtest

        rc = run_audited_backtest(
            snapshot_root=snap_root,
            price_csv=price_csv,
            horizons=[5],
            out_root=out_root,
            run_id="smoke_test",
            top_k=5,
            cost_bps=0,
            anchor_mode="prev_trading_day",
            benchmark="none",
        )

        # Check that artifacts were created
        run_dir = out_root / "smoke_test"
        assert run_dir.is_dir()
        assert (run_dir / "AUDIT.md").exists()
        assert (run_dir / "preflight" / "preflight_summary.csv").exists()
        assert (run_dir / "preflight" / "preflight_manifest.json").exists()
        assert (run_dir / "eval").is_dir()

        # Verify AUDIT.md content
        audit_text = (run_dir / "AUDIT.md").read_text(encoding="utf-8")
        assert "smoke_test" in audit_text
        assert "Preflight Summary" in audit_text
        assert "2" in audit_text  # 2 total snapshots

        # Verify manifest
        with open(run_dir / "preflight" / "preflight_manifest.json", encoding="utf-8") as f:
            manifest = json.load(f)
        assert manifest["run_id"] == "smoke_test"
        assert manifest["counts"]["total"] == 2
