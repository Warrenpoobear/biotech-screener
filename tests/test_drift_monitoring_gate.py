"""Tests for drift monitoring gate in run_daily_production.py."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from run_daily_production import (
    DriftThresholds,
    _compute_drift_metrics,
    _find_prior_snapshot,
    check_drift_monitoring,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

HEADER = [
    "ticker", "actionable_rank", "tier_dev", "eligible",
    "composite_rank", "composite_score",
]


def _write_rankings(path: Path, rows: list[dict]) -> None:
    """Write a minimal rankings.csv with the given rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER)
        writer.writeheader()
        writer.writerows(rows)


def _make_tickers(n: int, start: int = 1) -> list[dict]:
    """Generate n ticker rows ranked 1..n."""
    return [
        {
            "ticker": f"TICK{i}",
            "actionable_rank": str(i),
            "tier_dev": "A" if i <= n // 4 else "B",
            "eligible": "True",
            "composite_rank": str(i),
            "composite_score": str(round(1.0 - i / (n + 1), 4)),
        }
        for i in range(start, start + n)
    ]


# ---------------------------------------------------------------------------
# DriftThresholds dataclass tests
# ---------------------------------------------------------------------------


class TestDriftThresholds:
    def test_thresholds_id_stable(self):
        """Default thresholds always produce the same hash."""
        t1 = DriftThresholds()
        t2 = DriftThresholds()
        assert t1.thresholds_id == t2.thresholds_id
        assert len(t1.thresholds_id) == 8

    def test_thresholds_id_changes_with_values(self):
        t1 = DriftThresholds()
        t2 = DriftThresholds(warn_top20_overlap_pct=50.0)
        assert t1.thresholds_id != t2.thresholds_id

    def test_thresholds_from_json(self, tmp_path):
        """Load + round-trip preserves values."""
        original = DriftThresholds(
            warn_top20_overlap_pct=65.0,
            warn_tier_migration_count=5,
        )
        path = tmp_path / "thresholds.json"
        with open(path, "w") as f:
            json.dump(original.to_json(), f)

        loaded = DriftThresholds.from_json(path)
        assert loaded.warn_top20_overlap_pct == 65.0
        assert loaded.warn_tier_migration_count == 5
        assert loaded.thresholds_id == original.thresholds_id

    def test_to_json_includes_id(self):
        t = DriftThresholds()
        d = t.to_json()
        assert "thresholds_id" in d
        assert d["warn_top20_overlap_pct"] == 70.0

    def test_from_json_ignores_extra_keys(self, tmp_path):
        path = tmp_path / "thresholds.json"
        with open(path, "w") as f:
            json.dump({"warn_top20_overlap_pct": 60.0, "unknown_field": 999}, f)
        loaded = DriftThresholds.from_json(path)
        assert loaded.warn_top20_overlap_pct == 60.0


# ---------------------------------------------------------------------------
# _find_prior_snapshot tests
# ---------------------------------------------------------------------------


class TestFindPriorSnapshot:
    def test_finds_correct_prior(self, tmp_path):
        """Discovers the most recent prior date with valid rankings.csv."""
        for d in ["2026-02-17", "2026-02-18", "2026-02-19"]:
            _write_rankings(tmp_path / d / "rankings.csv", _make_tickers(10))

        result = _find_prior_snapshot(tmp_path, "2026-02-19")
        assert result is not None
        assert result.name == "2026-02-18"

    def test_skips_dirs_without_tier_dev(self, tmp_path):
        """Skips snapshots with rankings.csv lacking tier_dev column."""
        # Good prior
        _write_rankings(tmp_path / "2026-02-17" / "rankings.csv", _make_tickers(10))

        # Bad prior — no tier_dev column
        bad_dir = tmp_path / "2026-02-18"
        bad_dir.mkdir(parents=True)
        with open(bad_dir / "rankings.csv", "w") as f:
            f.write("ticker,actionable_rank\n")
            f.write("TICK1,1\n")

        # Current
        _write_rankings(tmp_path / "2026-02-19" / "rankings.csv", _make_tickers(10))

        result = _find_prior_snapshot(tmp_path, "2026-02-19")
        assert result is not None
        assert result.name == "2026-02-17"

    def test_no_prior_returns_none(self, tmp_path):
        _write_rankings(tmp_path / "2026-02-19" / "rankings.csv", _make_tickers(10))
        result = _find_prior_snapshot(tmp_path, "2026-02-19")
        assert result is None

    def test_skips_non_date_dirs(self, tmp_path):
        """Ignores directories that don't match YYYY-MM-DD pattern."""
        _write_rankings(tmp_path / "2026-02-18" / "rankings.csv", _make_tickers(10))
        _write_rankings(tmp_path / "2026-02-19" / "rankings.csv", _make_tickers(10))
        (tmp_path / "not_a_date").mkdir()
        (tmp_path / "2026-02-19__pre_backup").mkdir()

        result = _find_prior_snapshot(tmp_path, "2026-02-19")
        assert result is not None
        assert result.name == "2026-02-18"

    def test_current_date_not_in_list(self, tmp_path):
        """Handles case where current_date dir doesn't exist yet."""
        _write_rankings(tmp_path / "2026-02-17" / "rankings.csv", _make_tickers(10))
        _write_rankings(tmp_path / "2026-02-18" / "rankings.csv", _make_tickers(10))

        result = _find_prior_snapshot(tmp_path, "2026-02-19")
        assert result is not None
        assert result.name == "2026-02-18"


# ---------------------------------------------------------------------------
# check_drift_monitoring tests
# ---------------------------------------------------------------------------


class TestCheckDriftMonitoring:
    def test_no_prior_snapshot_pass(self, tmp_path):
        """No prior → PASS, detail says skipped."""
        staging = tmp_path / "staging" / "2026-02-19"
        _write_rankings(staging / "rankings.csv", _make_tickers(30))
        snapshot_dir = tmp_path / "snapshots"
        snapshot_dir.mkdir()

        gate = check_drift_monitoring(staging, snapshot_dir, "2026-02-19", DriftThresholds())
        assert gate.status == "PASS"
        assert "skipped" in gate.detail.lower()

    def test_identical_snapshots_pass(self, tmp_path):
        """Identical data → PASS, 100% overlap, rho=1.0."""
        rows = _make_tickers(30)
        snapshot_dir = tmp_path / "snapshots"
        _write_rankings(snapshot_dir / "2026-02-18" / "rankings.csv", rows)

        staging = tmp_path / "staging" / "2026-02-19"
        _write_rankings(staging / "rankings.csv", rows)

        gate = check_drift_monitoring(staging, snapshot_dir, "2026-02-19", DriftThresholds())
        assert gate.status == "PASS"
        assert gate.value["top20_overlap_pct"] == 100.0
        assert gate.value["rank_spearman_rho"] == 1.0

    def test_moderate_drift_pass(self, tmp_path):
        """Moderate drift within thresholds → PASS."""
        rows_prior = _make_tickers(30)
        # Swap positions 1 and 2 — small change
        rows_current = [dict(r) for r in rows_prior]
        rows_current[0]["actionable_rank"] = "2"
        rows_current[0]["composite_rank"] = "2"
        rows_current[1]["actionable_rank"] = "1"
        rows_current[1]["composite_rank"] = "1"

        snapshot_dir = tmp_path / "snapshots"
        _write_rankings(snapshot_dir / "2026-02-18" / "rankings.csv", rows_prior)

        staging = tmp_path / "staging" / "2026-02-19"
        _write_rankings(staging / "rankings.csv", rows_current)

        gate = check_drift_monitoring(staging, snapshot_dir, "2026-02-19", DriftThresholds())
        assert gate.status == "PASS"

    def test_high_drift_warn(self, tmp_path):
        """50% top-20 overlap → WARN."""
        # Prior: TICK1..TICK30
        rows_prior = _make_tickers(30)
        # Current: completely different top-20 (shift by 15)
        rows_current = _make_tickers(30, start=16)  # TICK16..TICK45

        snapshot_dir = tmp_path / "snapshots"
        _write_rankings(snapshot_dir / "2026-02-18" / "rankings.csv", rows_prior)

        staging = tmp_path / "staging" / "2026-02-19"
        _write_rankings(staging / "rankings.csv", rows_current)

        gate = check_drift_monitoring(staging, snapshot_dir, "2026-02-19", DriftThresholds())
        assert gate.status == "WARN"
        assert any("warn_top20_overlap_pct" in r for r in gate.detail.split(";"))

    def test_low_spearman_warn(self, tmp_path):
        """Reversed ranks → rho drops below threshold → WARN."""
        rows_prior = _make_tickers(30)
        # Reverse all ranks
        rows_current = []
        for i, r in enumerate(rows_prior):
            row = dict(r)
            row["actionable_rank"] = str(30 - i)
            row["composite_rank"] = str(30 - i)
            rows_current.append(row)

        snapshot_dir = tmp_path / "snapshots"
        _write_rankings(snapshot_dir / "2026-02-18" / "rankings.csv", rows_prior)

        staging = tmp_path / "staging" / "2026-02-19"
        _write_rankings(staging / "rankings.csv", rows_current)

        gate = check_drift_monitoring(staging, snapshot_dir, "2026-02-19", DriftThresholds())
        assert gate.status == "WARN"
        assert any("warn_rank_spearman_rho" in r for r in gate.detail.split(";"))

    def test_tier_migration_warn(self, tmp_path):
        """15 tier changes → WARN (threshold is 10)."""
        rows_prior = _make_tickers(30)
        rows_current = [dict(r) for r in rows_prior]
        # Change 15 tickers from A to B or vice versa
        for i in range(15):
            old = rows_current[i]["tier_dev"]
            rows_current[i]["tier_dev"] = "B" if old == "A" else "A"

        snapshot_dir = tmp_path / "snapshots"
        _write_rankings(snapshot_dir / "2026-02-18" / "rankings.csv", rows_prior)

        staging = tmp_path / "staging" / "2026-02-19"
        _write_rankings(staging / "rankings.csv", rows_current)

        gate = check_drift_monitoring(staging, snapshot_dir, "2026-02-19", DriftThresholds())
        assert gate.status == "WARN"
        assert "warn_tier_migration_count" in gate.detail

    def test_eligibility_change_warn(self, tmp_path):
        """12 eligibility changes → WARN (threshold is 10)."""
        rows_prior = _make_tickers(30)
        rows_current = [dict(r) for r in rows_prior]
        for i in range(12):
            rows_current[i]["eligible"] = "False"

        snapshot_dir = tmp_path / "snapshots"
        _write_rankings(snapshot_dir / "2026-02-18" / "rankings.csv", rows_prior)

        staging = tmp_path / "staging" / "2026-02-19"
        _write_rankings(staging / "rankings.csv", rows_current)

        gate = check_drift_monitoring(staging, snapshot_dir, "2026-02-19", DriftThresholds())
        assert gate.status == "WARN"
        assert "warn_eligibility_change_count" in gate.detail

    def test_multiple_warn_reasons(self, tmp_path):
        """Several thresholds breached → all reasons listed."""
        rows_prior = _make_tickers(30)
        # Completely different universe — triggers overlap + migration + eligibility
        rows_current = _make_tickers(30, start=16)

        snapshot_dir = tmp_path / "snapshots"
        _write_rankings(snapshot_dir / "2026-02-18" / "rankings.csv", rows_prior)

        staging = tmp_path / "staging" / "2026-02-19"
        _write_rankings(staging / "rankings.csv", rows_current)

        gate = check_drift_monitoring(staging, snapshot_dir, "2026-02-19", DriftThresholds())
        assert gate.status == "WARN"
        # At least overlap should be triggered
        assert "warn_top20_overlap_pct" in gate.detail

    def test_never_fails(self, tmp_path):
        """Even extreme drift → status is WARN, never FAIL."""
        rows_prior = _make_tickers(30)
        # Totally different universe
        rows_current = _make_tickers(30, start=100)

        snapshot_dir = tmp_path / "snapshots"
        _write_rankings(snapshot_dir / "2026-02-18" / "rankings.csv", rows_prior)

        staging = tmp_path / "staging" / "2026-02-19"
        _write_rankings(staging / "rankings.csv", rows_current)

        gate = check_drift_monitoring(staging, snapshot_dir, "2026-02-19", DriftThresholds())
        assert gate.status in ("PASS", "WARN")
        assert gate.status != "FAIL"

    def test_drift_report_json_written(self, tmp_path):
        """drift_report.json exists in snapshot dir."""
        rows = _make_tickers(30)
        snapshot_dir = tmp_path / "snapshots"
        _write_rankings(snapshot_dir / "2026-02-18" / "rankings.csv", rows)

        staging = tmp_path / "staging" / "2026-02-19"
        _write_rankings(staging / "rankings.csv", rows)

        check_drift_monitoring(staging, snapshot_dir, "2026-02-19", DriftThresholds())
        assert (staging / "drift_report.json").exists()

    def test_drift_report_md_written(self, tmp_path):
        """drift_report.md exists in snapshot dir."""
        rows = _make_tickers(30)
        snapshot_dir = tmp_path / "snapshots"
        _write_rankings(snapshot_dir / "2026-02-18" / "rankings.csv", rows)

        staging = tmp_path / "staging" / "2026-02-19"
        _write_rankings(staging / "rankings.csv", rows)

        check_drift_monitoring(staging, snapshot_dir, "2026-02-19", DriftThresholds())
        assert (staging / "drift_report.md").exists()

    def test_drift_report_schema(self, tmp_path):
        """JSON report has required keys: version, thresholds_id, metrics, status."""
        rows = _make_tickers(30)
        snapshot_dir = tmp_path / "snapshots"
        _write_rankings(snapshot_dir / "2026-02-18" / "rankings.csv", rows)

        staging = tmp_path / "staging" / "2026-02-19"
        _write_rankings(staging / "rankings.csv", rows)

        check_drift_monitoring(staging, snapshot_dir, "2026-02-19", DriftThresholds())

        with open(staging / "drift_report.json") as f:
            report = json.load(f)

        assert report["version"] == "1.0.0"
        assert "thresholds_id" in report
        assert "metrics" in report
        assert "status" in report
        assert "current_date" in report
        assert "prior_date" in report
        assert "warn_reasons" in report

        metrics = report["metrics"]
        assert "top20_overlap_pct" in metrics
        assert "top60_overlap_pct" in metrics
        assert "rank_spearman_rho" in metrics
        assert "tier_migration_count" in metrics
        assert "eligibility_change_count" in metrics

    def test_actionable_rank_fallback(self, tmp_path):
        """Falls back to composite_rank for old-format CSV."""
        # Write prior with only composite_rank (no actionable_rank)
        prior_dir = tmp_path / "snapshots" / "2026-02-18"
        prior_dir.mkdir(parents=True)
        with open(prior_dir / "rankings.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["ticker", "composite_rank", "tier_dev", "eligible"])
            writer.writeheader()
            for i in range(1, 31):
                writer.writerow({
                    "ticker": f"TICK{i}",
                    "composite_rank": str(i),
                    "tier_dev": "A" if i <= 7 else "B",
                    "eligible": "True",
                })

        staging = tmp_path / "staging" / "2026-02-19"
        _write_rankings(staging / "rankings.csv", _make_tickers(30))

        gate = check_drift_monitoring(
            staging, tmp_path / "snapshots", "2026-02-19", DriftThresholds(),
        )
        assert gate.status in ("PASS", "WARN")
        assert gate.value["rank_column_prior"] == "composite_rank"
        assert gate.value["rank_column_current"] == "actionable_rank"

    def test_mean_abs_rank_delta_warn(self, tmp_path):
        """Large rank shuffles in top-60 → WARN on mean_abs_rank_delta_top60."""
        rows_prior = _make_tickers(60)
        # Shuffle: shift every rank by 20
        rows_current = []
        for r in rows_prior:
            row = dict(r)
            old_rank = int(row["actionable_rank"])
            new_rank = ((old_rank + 19) % 60) + 1
            row["actionable_rank"] = str(new_rank)
            row["composite_rank"] = str(new_rank)
            rows_current.append(row)

        snapshot_dir = tmp_path / "snapshots"
        _write_rankings(snapshot_dir / "2026-02-18" / "rankings.csv", rows_prior)

        staging = tmp_path / "staging" / "2026-02-19"
        _write_rankings(staging / "rankings.csv", rows_current)

        gate = check_drift_monitoring(staging, snapshot_dir, "2026-02-19", DriftThresholds())
        assert gate.status == "WARN"
        assert "warn_mean_abs_rank_delta_top60" in gate.detail
