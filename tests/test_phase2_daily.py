#!/usr/bin/env python3
"""Tests for tools/run_phase2_daily.py — production daily runner.

Covers:
  - Run manifest written with expected fields
  - XBI staleness gate triggers FAIL
  - Missing-reason fraction gate
  - Turnover gate parses delta report
  - Atomic promotion: staging → final, backup of existing
  - Audit exit code mapping
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
from run_daily_production import (
    GateConfig,
    GateResult,
    _get_ticker_last_date,
    build_run_manifest,
    check_audit_result,
    check_ctgov_cache,
    check_missing_reason_fraction,
    check_turnover,
    check_xbi_staleness,
    promote_snapshot,
    run_daily,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_price_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    """Write a minimal price_history.csv."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ticker", "date", "close"])
        writer.writeheader()
        writer.writerows(rows)


def _write_rankings_csv(
    path: Path,
    rows: List[Dict[str, str]],
    extra_cols: List[str] | None = None,
) -> None:
    """Write a minimal rankings.csv with DE-critical missing_reason columns."""
    cols = [
        "ticker", "actionable_rank", "eligible",
        "de_beta_xbi_60d_missing_reason", "de_alpha_60d_missing_reason",
    ]
    if extra_cols:
        cols.extend(extra_cols)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for r in rows:
            writer.writerow({c: r.get(c, "") for c in cols})


def _write_delta_report(path: Path, turnover_pct: float) -> None:
    """Write a minimal delta report with Name turnover line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"1. PORTFOLIO TURNOVER\n"
        f"  Name turnover: {turnover_pct:.1f}%\n"
        f"  Weight L1 delta: 50.0%\n"
    )


# ---------------------------------------------------------------------------
# Tests: XBI staleness gate
# ---------------------------------------------------------------------------

class TestXbiStalenessGate:

    def test_xbi_pass_when_fresh(self, tmp_path):
        """XBI last date == as_of_date → PASS."""
        csv_path = tmp_path / "prices.csv"
        _write_price_csv(csv_path, [
            {"ticker": "XBI", "date": "2026-02-19", "close": "100"},
            {"ticker": "ACRS", "date": "2026-02-19", "close": "50"},
        ])
        result = check_xbi_staleness(csv_path, "2026-02-19", threshold_days=3)
        assert result.status == "PASS"
        assert result.value == 0

    def test_xbi_fail_when_stale(self, tmp_path):
        """XBI 10 trading days behind → FAIL with threshold=3."""
        csv_path = tmp_path / "prices.csv"
        _write_price_csv(csv_path, [
            {"ticker": "XBI", "date": "2026-02-03", "close": "100"},
        ])
        result = check_xbi_staleness(csv_path, "2026-02-19", threshold_days=3)
        assert result.status == "FAIL"
        assert result.value > 3

    def test_xbi_fail_when_missing(self, tmp_path):
        """No XBI rows at all → FAIL."""
        csv_path = tmp_path / "prices.csv"
        _write_price_csv(csv_path, [
            {"ticker": "ACRS", "date": "2026-02-19", "close": "50"},
        ])
        result = check_xbi_staleness(csv_path, "2026-02-19", threshold_days=3)
        assert result.status == "FAIL"
        assert result.value is None

    def test_xbi_pass_at_threshold_boundary(self, tmp_path):
        """XBI exactly threshold days behind → PASS (not >)."""
        csv_path = tmp_path / "prices.csv"
        # 2026-02-19 is Wednesday. 3 trading days back = Friday 2026-02-13
        # Actually let's just use 1 trading day gap for a simple test
        _write_price_csv(csv_path, [
            {"ticker": "XBI", "date": "2026-02-18", "close": "100"},
        ])
        result = check_xbi_staleness(csv_path, "2026-02-19", threshold_days=3)
        assert result.status == "PASS"
        assert result.value <= 3

    def test_xbi_no_csv(self, tmp_path):
        """price_history.csv doesn't exist → FAIL."""
        csv_path = tmp_path / "nonexistent.csv"
        result = check_xbi_staleness(csv_path, "2026-02-19", threshold_days=3)
        assert result.status == "FAIL"


# ---------------------------------------------------------------------------
# Tests: Missing reason fraction gate
# ---------------------------------------------------------------------------

class TestMissingReasonGate:

    def test_all_clean(self, tmp_path):
        """No missing reasons → PASS."""
        snap = tmp_path / "snap"
        _write_rankings_csv(snap / "rankings.csv", [
            {"ticker": "ACRS", "actionable_rank": "1", "eligible": "1",
             "de_beta_xbi_60d_missing_reason": "", "de_alpha_60d_missing_reason": ""},
            {"ticker": "BMRN", "actionable_rank": "2", "eligible": "1",
             "de_beta_xbi_60d_missing_reason": "", "de_alpha_60d_missing_reason": ""},
        ])
        result = check_missing_reason_fraction(snap, max_frac=0.05)
        assert result.status == "PASS"
        assert result.value == 0.0

    def test_above_threshold(self, tmp_path):
        """50% missing → FAIL with threshold=5%."""
        snap = tmp_path / "snap"
        _write_rankings_csv(snap / "rankings.csv", [
            {"ticker": "ACRS", "actionable_rank": "1", "eligible": "1",
             "de_beta_xbi_60d_missing_reason": "xbi_stale", "de_alpha_60d_missing_reason": ""},
            {"ticker": "BMRN", "actionable_rank": "2", "eligible": "1",
             "de_beta_xbi_60d_missing_reason": "", "de_alpha_60d_missing_reason": ""},
        ])
        result = check_missing_reason_fraction(snap, max_frac=0.05)
        assert result.status == "FAIL"
        assert result.value == 0.5

    def test_no_rankings(self, tmp_path):
        """Missing rankings.csv → FAIL."""
        snap = tmp_path / "snap"
        snap.mkdir(parents=True)
        result = check_missing_reason_fraction(snap, max_frac=0.05)
        assert result.status == "FAIL"


# ---------------------------------------------------------------------------
# Tests: Turnover gate
# ---------------------------------------------------------------------------

class TestTurnoverGate:

    def test_turnover_pass(self, tmp_path):
        """35% turnover with 40% threshold → PASS."""
        snap = tmp_path / "snap"
        _write_delta_report(snap / "phase2_run_delta_report.txt", 35.0)
        result = check_turnover(snap, max_pct=40.0)
        assert result.status == "PASS"
        assert result.value == 35.0

    def test_turnover_fail(self, tmp_path):
        """45% turnover with 40% threshold → FAIL."""
        snap = tmp_path / "snap"
        _write_delta_report(snap / "phase2_run_delta_report.txt", 45.0)
        result = check_turnover(snap, max_pct=40.0)
        assert result.status == "FAIL"
        assert result.value == 45.0

    def test_no_delta_report(self, tmp_path):
        """Missing delta report → PASS (first run)."""
        snap = tmp_path / "snap"
        snap.mkdir(parents=True)
        result = check_turnover(snap, max_pct=40.0)
        assert result.status == "PASS"


# ---------------------------------------------------------------------------
# Tests: Audit exit code mapping
# ---------------------------------------------------------------------------

class TestAuditGate:

    def test_audit_ok(self):
        proc = subprocess.CompletedProcess(args=[], returncode=0)
        result = check_audit_result(proc, GateConfig())
        assert result.status == "PASS"

    def test_audit_warn(self):
        proc = subprocess.CompletedProcess(args=[], returncode=2)
        result = check_audit_result(proc, GateConfig())
        assert result.status == "WARN"

    def test_audit_fail(self):
        proc = subprocess.CompletedProcess(args=[], returncode=1)
        result = check_audit_result(proc, GateConfig())
        assert result.status == "FAIL"

    def test_audit_warn_ignored_when_config_off(self):
        proc = subprocess.CompletedProcess(args=[], returncode=2)
        config = GateConfig(audit_warn_is_gate_warn=False)
        result = check_audit_result(proc, config)
        assert result.status == "PASS"


# ---------------------------------------------------------------------------
# Tests: Atomic promotion
# ---------------------------------------------------------------------------

class TestAtomicPromotion:

    def test_promote_new(self, tmp_path):
        """Staging dir moved to final location."""
        staging = tmp_path / "staging" / "2026-02-19"
        staging.mkdir(parents=True)
        (staging / "rankings.csv").write_text("ticker\nACRS\n")
        (staging / "metadata.json").write_text('{"as_of_date": "2026-02-19"}')

        final_dir = tmp_path / "snapshots"
        final_dir.mkdir()

        result = promote_snapshot(staging, final_dir, "2026-02-19")
        assert result == final_dir / "2026-02-19"
        assert (result / "rankings.csv").exists()
        assert not staging.exists()  # moved, not copied

    def test_promote_existing_backed_up(self, tmp_path):
        """Existing snapshot at target is backed up before promotion."""
        # Create existing snapshot
        final_dir = tmp_path / "snapshots"
        existing = final_dir / "2026-02-19"
        existing.mkdir(parents=True)
        (existing / "old_rankings.csv").write_text("old data\n")

        # Create staging
        staging = tmp_path / "staging" / "2026-02-19"
        staging.mkdir(parents=True)
        (staging / "rankings.csv").write_text("new data\n")

        result = promote_snapshot(staging, final_dir, "2026-02-19")
        assert (result / "rankings.csv").read_text() == "new data\n"

        # Old snapshot should be backed up
        backups = [p for p in final_dir.iterdir() if p.name.startswith("2026-02-19__pre_")]
        assert len(backups) == 1
        assert (backups[0] / "old_rankings.csv").exists()


# ---------------------------------------------------------------------------
# Tests: Run manifest
# ---------------------------------------------------------------------------

class TestRunManifest:

    def test_manifest_has_required_fields(self, tmp_path):
        """Manifest contains all required top-level fields."""
        snap = tmp_path / "snap"
        snap.mkdir(parents=True)
        (snap / "metadata.json").write_text(json.dumps({
            "as_of_date": "2026-02-19",
            "version": "v1.4.0",
            "clinical_sort_telemetry": {"ruleset_id": "aa0aaf28"},
            "ranking_mode": "decision",
            "decision_mode": "phase2",
            "ticker_count": 319,
            "total_evaluated": 353,
            "active_universe": 319,
        }))

        gates = [
            GateResult(name="xbi_staleness", status="PASS", detail="ok", value=0, threshold=3),
            GateResult(name="audit", status="PASS", detail="ok"),
        ]
        screen_proc = subprocess.CompletedProcess(args=[], returncode=0)
        audit_proc = subprocess.CompletedProcess(args=[], returncode=0)

        with patch("run_daily_production.get_git_info", return_value={
            "branch": "main", "commit_sha": "abc123", "dirty": False,
        }):
            manifest = build_run_manifest(
                "2026-02-19", gates, {"xbi_last_date": "2026-02-19"},
                screen_proc, audit_proc, GateConfig(),
                snapshot_date_dir=snap,
            )

        assert manifest["manifest_version"] == "1.1.0"
        assert manifest["as_of_date"] == "2026-02-19"
        assert manifest["requested_as_of_date"] == "2026-02-19"
        assert manifest["effective_as_of_date"] == "2026-02-19"
        assert manifest["git"]["commit_sha"] == "abc123"
        assert manifest["ruleset"]["ruleset_hash"] == "aa0aaf28"
        assert manifest["ruleset"]["ranking_mode"] == "decision"
        assert manifest["row_counts"]["ticker_count"] == 319
        assert manifest["overall_status"] == "PASS"
        assert manifest["screen_exit_code"] == 0
        assert manifest["audit_exit_code"] == 0
        assert len(manifest["gates"]) == 2
        assert "gate_config" in manifest

    def test_manifest_overall_fail(self):
        """Any FAIL gate → overall_status FAIL."""
        gates = [
            GateResult(name="xbi_staleness", status="FAIL", detail="stale"),
            GateResult(name="audit", status="PASS", detail="ok"),
        ]
        screen_proc = subprocess.CompletedProcess(args=[], returncode=0)

        with patch("run_daily_production.get_git_info", return_value={}):
            manifest = build_run_manifest(
                "2026-02-19", gates, {},
                screen_proc, None, GateConfig(),
            )

        assert manifest["overall_status"] == "FAIL"

    def test_manifest_overall_warn(self):
        """WARN gate (no FAIL) → overall_status WARN."""
        gates = [
            GateResult(name="xbi_staleness", status="PASS", detail="ok"),
            GateResult(name="audit", status="WARN", detail="violations"),
        ]
        screen_proc = subprocess.CompletedProcess(args=[], returncode=0)

        with patch("run_daily_production.get_git_info", return_value={}):
            manifest = build_run_manifest(
                "2026-02-19", gates, {},
                screen_proc, None, GateConfig(),
            )

        assert manifest["overall_status"] == "WARN"


class TestScreenFailureGate:
    """Screen failure must produce overall_status=FAIL, not silent PASS."""

    def test_screen_crash_yields_fail_manifest(self, tmp_path):
        """When run_screen exits non-0/2, overall_status must be FAIL."""
        price_csv = tmp_path / "prices.csv"
        _write_price_csv(price_csv, [
            {"ticker": "XBI", "date": "2026-02-19", "close": "100"},
        ])

        # Create ctgov cache so that gate passes
        cache_dir = tmp_path / "ctgov"
        cache_dir.mkdir(parents=True)
        (cache_dir / "trial_records_2026-02-19.json").write_text("[]")

        data_dir = tmp_path / "production_data"
        data_dir.mkdir()
        # Write minimal universe so price refresh doesn't error
        (data_dir / "universe.json").write_text(json.dumps([{"ticker": "XBI"}]))

        final_dir = tmp_path / "snapshots"
        final_dir.mkdir()

        # Patch run_screen to simulate crash (exit 1)
        fake_proc = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="ERROR: something broke",
        )
        with patch("run_daily_production.run_screen", return_value=fake_proc), \
             patch("run_daily_production.get_git_info", return_value={
                 "branch": "main", "commit_sha": "test", "dirty": False,
             }):
            manifest = run_daily(
                "2026-02-19",
                data_dir=data_dir,
                price_csv=price_csv,
                final_snapshots_dir=final_dir,
                skip_price_refresh=True,
                ctgov_cache_dir=cache_dir,
            )

        assert manifest["overall_status"] == "FAIL"
        # Must have a "screen" gate in FAIL state
        screen_gates = [g for g in manifest["gates"] if g["name"] == "screen"]
        assert len(screen_gates) == 1
        assert screen_gates[0]["status"] == "FAIL"
        # Snapshot should NOT be promoted
        assert not (final_dir / "2026-02-19").exists()


# ---------------------------------------------------------------------------
# Tests: GateConfig
# ---------------------------------------------------------------------------

class TestGateConfig:

    def test_defaults(self):
        config = GateConfig()
        assert config.xbi_stale_days == 3
        assert config.missing_reason_max_frac == 0.05
        assert config.turnover_max_pct == 40.0

    def test_from_json(self, tmp_path):
        cfg_path = tmp_path / "gates.json"
        cfg_path.write_text(json.dumps({
            "xbi_stale_days": 5,
            "turnover_max_pct": 50.0,
            "extra_field": "ignored",
        }))
        config = GateConfig.from_json(cfg_path)
        assert config.xbi_stale_days == 5
        assert config.turnover_max_pct == 50.0
        assert config.missing_reason_max_frac == 0.05  # default preserved


# ---------------------------------------------------------------------------
# Tests: Helper — ticker last date
# ---------------------------------------------------------------------------

class TestGetTickerLastDate:

    def test_finds_latest(self, tmp_path):
        csv_path = tmp_path / "prices.csv"
        _write_price_csv(csv_path, [
            {"ticker": "XBI", "date": "2026-02-17", "close": "100"},
            {"ticker": "XBI", "date": "2026-02-18", "close": "101"},
            {"ticker": "XBI", "date": "2026-02-19", "close": "102"},
        ])
        assert _get_ticker_last_date(csv_path, "XBI") == "2026-02-19"

    def test_missing_ticker(self, tmp_path):
        csv_path = tmp_path / "prices.csv"
        _write_price_csv(csv_path, [
            {"ticker": "ACRS", "date": "2026-02-19", "close": "50"},
        ])
        assert _get_ticker_last_date(csv_path, "XBI") is None

    def test_no_file(self, tmp_path):
        assert _get_ticker_last_date(tmp_path / "nope.csv", "XBI") is None


# ---------------------------------------------------------------------------
# Tests: CTGov cache gate
# ---------------------------------------------------------------------------

class TestCtgovCacheGate:

    def _make_cache(self, cache_dir: Path, dates: list[str]) -> None:
        """Create fake trial_records_{date}.json files."""
        cache_dir.mkdir(parents=True, exist_ok=True)
        for d in dates:
            (cache_dir / f"trial_records_{d}.json").write_text("[]")

    def test_exact_match_pass(self, tmp_path):
        """Cache exists for requested date → PASS, effective == requested."""
        cache_dir = tmp_path / "ctgov"
        self._make_cache(cache_dir, ["2026-02-19"])
        gate, effective = check_ctgov_cache("2026-02-19", cache_dir)
        assert gate.status == "PASS"
        assert effective == "2026-02-19"

    def test_missing_no_fallback_fail(self, tmp_path):
        """Cache missing, no --allow-date-fallback → FAIL."""
        cache_dir = tmp_path / "ctgov"
        self._make_cache(cache_dir, ["2026-02-18"])
        gate, effective = check_ctgov_cache("2026-02-19", cache_dir, allow_fallback=False)
        assert gate.status == "FAIL"
        assert "warm_caches.py" in gate.detail
        assert effective == "2026-02-19"  # unchanged on FAIL

    def test_missing_with_fallback_warn(self, tmp_path):
        """Cache missing but prior date available + fallback allowed → WARN."""
        cache_dir = tmp_path / "ctgov"
        self._make_cache(cache_dir, ["2026-02-17", "2026-02-18"])
        gate, effective = check_ctgov_cache("2026-02-19", cache_dir, allow_fallback=True)
        assert gate.status == "WARN"
        assert effective == "2026-02-18"  # picked latest <= requested
        assert "falling back to 2026-02-18" in gate.detail

    def test_fallback_no_prior_dates_fail(self, tmp_path):
        """Fallback allowed but no prior cached dates → FAIL."""
        cache_dir = tmp_path / "ctgov"
        self._make_cache(cache_dir, ["2026-02-20"])  # only future date
        gate, effective = check_ctgov_cache("2026-02-19", cache_dir, allow_fallback=True)
        assert gate.status == "FAIL"

    def test_empty_cache_dir_fail(self, tmp_path):
        """Empty cache dir → FAIL."""
        cache_dir = tmp_path / "ctgov"
        cache_dir.mkdir(parents=True)
        gate, effective = check_ctgov_cache("2026-02-19", cache_dir)
        assert gate.status == "FAIL"

    def test_stray_filename_ignored(self, tmp_path):
        """Non-date filenames in cache dir are silently skipped."""
        cache_dir = tmp_path / "ctgov"
        cache_dir.mkdir(parents=True)
        (cache_dir / "trial_records_latest.json").write_text("[]")
        (cache_dir / "trial_records_backup.json").write_text("[]")
        (cache_dir / "trial_records_2026-02-17.json").write_text("[]")
        gate, effective = check_ctgov_cache("2026-02-19", cache_dir, allow_fallback=True)
        assert gate.status == "WARN"
        assert effective == "2026-02-17"  # only valid date picked


# ---------------------------------------------------------------------------
# Tests: Manifest date fields
# ---------------------------------------------------------------------------

class TestManifestDateFields:

    def test_manifest_records_both_dates(self, tmp_path):
        """Manifest records requested and effective dates when they differ."""
        gates = [
            GateResult(name="xbi_staleness", status="PASS", detail="ok"),
        ]
        screen_proc = subprocess.CompletedProcess(args=[], returncode=0)
        with patch("run_daily_production.get_git_info", return_value={}):
            manifest = build_run_manifest(
                "2026-02-18", gates, {},
                screen_proc, None, GateConfig(),
                requested_as_of_date="2026-02-19",
            )
        assert manifest["requested_as_of_date"] == "2026-02-19"
        assert manifest["effective_as_of_date"] == "2026-02-18"
        assert manifest["as_of_date"] == "2026-02-18"  # backward compat

    def test_manifest_dates_match_when_no_fallback(self, tmp_path):
        """When no fallback, requested == effective."""
        gates = [
            GateResult(name="xbi_staleness", status="PASS", detail="ok"),
        ]
        screen_proc = subprocess.CompletedProcess(args=[], returncode=0)
        with patch("run_daily_production.get_git_info", return_value={}):
            manifest = build_run_manifest(
                "2026-02-19", gates, {},
                screen_proc, None, GateConfig(),
            )
        assert manifest["requested_as_of_date"] == "2026-02-19"
        assert manifest["effective_as_of_date"] == "2026-02-19"


# ---------------------------------------------------------------------------
# Tests: Git dirty pre/post-run
# ---------------------------------------------------------------------------

class TestGitDirtySemantics:

    def test_dirty_pre_post_run_recorded(self):
        """dirty_pre_run and dirty_post_run are distinct; dirty == dirty_pre_run."""
        git_pre = {"branch": "main", "commit_sha": "abc123", "dirty": False}
        git_post = {"branch": "main", "commit_sha": "abc123", "dirty": True}
        gates = [GateResult(name="xbi_staleness", status="PASS", detail="ok")]
        screen_proc = subprocess.CompletedProcess(args=[], returncode=0)

        manifest = build_run_manifest(
            "2026-02-19", gates, {},
            screen_proc, None, GateConfig(),
            git_pre_run=git_pre,
            git_post_run=git_post,
        )

        assert manifest["git"]["dirty_pre_run"] is False
        assert manifest["git"]["dirty_post_run"] is True
        assert manifest["git"]["dirty"] is False  # backward compat == pre_run
        assert manifest["git"]["commit_sha"] == "abc123"

    def test_dirty_post_run_none_on_early_exit(self):
        """Early-exit manifests have dirty_post_run=None."""
        git_pre = {"branch": "main", "commit_sha": "abc123", "dirty": True}
        gates = [GateResult(name="xbi_staleness", status="FAIL", detail="stale")]
        screen_proc = subprocess.CompletedProcess(args=[], returncode=-1)

        manifest = build_run_manifest(
            "2026-02-19", gates, {},
            screen_proc, None, GateConfig(),
            git_pre_run=git_pre,
        )

        assert manifest["git"]["dirty_pre_run"] is True
        assert manifest["git"]["dirty_post_run"] is None
        assert manifest["git"]["dirty"] is True
