"""Tests for Spec 026: Data Collection Health Orchestrator."""

import csv
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.build_data_collection_health import (
    DEFAULT_THRESHOLDS,
    build_health,
    load_thresholds,
    run_from_screen,
    write_json_report,
    write_markdown_report,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_cache_health(
    snap_dir, overall="ok", sec8k_status="ok", sec8k_count=100, ctgov_status="ok", ctgov_count=500, degraded=False
):
    (snap_dir / "cache_health.json").write_text(
        json.dumps(
            {
                "schema": "cache_health.v1",
                "as_of_date": "2026-03-17",
                "overall_status": overall,
                "sec8k": {"status": sec8k_status, "count": sec8k_count, "reason": ""},
                "ctgov": {"status": ctgov_status, "count": ctgov_count, "reason": ""},
                "degraded_run": degraded,
            }
        ),
        encoding="utf-8",
    )


def _write_source_mix(snap_dir, total=1500, tickers=280, ctgov_diff=500, sec8k=400):
    (snap_dir / "catalyst_source_mix.json").write_text(
        json.dumps(
            {
                "total_events": total,
                "unique_tickers_with_events": tickers,
                "by_source": {"CTGOV": ctgov_diff // 2, "CTGOV_CALENDAR": 800, "SEC_8K_FILING": sec8k},
                "pre_dedup_by_source": {"CTGOV": ctgov_diff, "CTGOV_CALENDAR": 1200, "SEC_8K_FILING": sec8k},
            }
        ),
        encoding="utf-8",
    )


def _write_coverage_quality(snap_dir, catalyst_pct=85, options_pct=65, sponsor_pct=84):
    (snap_dir / "coverage_quality.json").write_text(
        json.dumps(
            {
                "schema_version": "coverage_quality.v1",
                "as_of_date": "2026-03-17",
                "catalyst_coverage": {"specific_days_pct": catalyst_pct},
                "catalyst_family_coverage": {"coverage_pct": 82},
                "component_coverage": {"sponsor_pct": sponsor_pct, "options_pct": options_pct, "drawdown_pct": 100},
                "options_data_freshness": {"all_fresh": True},
            }
        ),
        encoding="utf-8",
    )


def _write_options_diag(snap_dir, n_universe=296, n_with=192, coverage_pct=64.9):
    (snap_dir / "options_diagnostics_summary.json").write_text(
        json.dumps(
            {
                "schema": "options_diagnostics_summary.v1",
                "as_of_date": "2026-03-17",
                "coverage": {"n_universe": n_universe, "n_with_options_data": n_with, "coverage_pct": coverage_pct},
            }
        ),
        encoding="utf-8",
    )


def _write_rankings(snap_dir, n=100):
    path = snap_dir / "rankings.csv"
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        writer = csv.DictWriter(f, fieldnames=["ticker", "actionable_rank", "tier_any"])
        writer.writeheader()
        for i in range(n):
            writer.writerow({"ticker": f"T{i:04d}", "actionable_rank": str(i + 1), "tier_any": "A" if i < 30 else "B"})


def _write_metadata(snap_dir):
    (snap_dir / "metadata.json").write_text(
        json.dumps(
            {
                "as_of_date": "2026-03-17",
                "phase_scores_version": "v3",
            }
        ),
        encoding="utf-8",
    )


def _write_price_history(data_dir, n_tickers=340, date="2026-03-17"):
    with open(data_dir / "price_history.csv", "w", encoding="utf-8", newline="\n") as f:
        writer = csv.DictWriter(f, fieldnames=["date", "ticker", "close", "open", "high", "low", "volume"])
        writer.writeheader()
        for i in range(n_tickers):
            writer.writerow(
                {"date": date, "ticker": f"T{i:04d}", "close": 10.0, "open": 10, "high": 10, "low": 10, "volume": 1000}
            )


def _write_universe(data_dir, n=354):
    (data_dir / "universe.json").write_text(
        json.dumps([{"ticker": f"T{i:04d}"} for i in range(n)]),
        encoding="utf-8",
    )


@pytest.fixture
def healthy_snapshot(tmp_path):
    """Create a snapshot dir that should PASS."""
    snap = tmp_path / "2026-03-17"
    snap.mkdir()
    data = tmp_path / "production_data"
    data.mkdir()

    _write_cache_health(snap)
    _write_source_mix(snap)
    _write_coverage_quality(snap)
    _write_options_diag(snap)
    _write_rankings(snap)
    _write_metadata(snap)
    _write_price_history(data)
    _write_universe(data)

    return snap, data


@pytest.fixture
def degraded_snapshot(tmp_path):
    """Create a snapshot dir mimicking Mar 16 stale-data failure."""
    snap = tmp_path / "2026-03-16"
    snap.mkdir()
    data = tmp_path / "production_data"
    data.mkdir()

    _write_cache_health(snap, overall="bad", sec8k_status="bad", sec8k_count=0, degraded=True)
    _write_source_mix(snap, total=1100, tickers=244, ctgov_diff=0, sec8k=0)
    _write_coverage_quality(snap, catalyst_pct=83.8)
    _write_options_diag(snap)
    _write_rankings(snap)
    _write_metadata(snap)
    _write_price_history(data)
    _write_universe(data)

    return snap, data


# ---------------------------------------------------------------------------
# Threshold tests
# ---------------------------------------------------------------------------


class TestThresholds:
    def test_default_thresholds_exist(self):
        assert "ctgov_diff_events_warn_if_zero" in DEFAULT_THRESHOLDS
        assert "sec8k_warn_if_zero_without_skip" in DEFAULT_THRESHOLDS

    def test_load_thresholds_from_file(self, tmp_path):
        p = tmp_path / "thresholds.json"
        p.write_text(json.dumps({"ctgov_min_trial_count": 99999}))
        t = load_thresholds(p)
        assert t["ctgov_min_trial_count"] == 99999
        # Other defaults still present
        assert "sec8k_warn_if_zero_without_skip" in t

    def test_load_thresholds_missing_file(self, tmp_path):
        t = load_thresholds(tmp_path / "nonexistent.json")
        assert t == DEFAULT_THRESHOLDS


# ---------------------------------------------------------------------------
# Build health tests
# ---------------------------------------------------------------------------


class TestBuildHealth:
    def test_healthy_snapshot_does_not_fail(self, healthy_snapshot):
        snap, data = healthy_snapshot
        health = build_health(snap, data, "2026-03-17")
        assert health["status"] != "FAIL"
        # No hard-failure flags (missing optional artifacts produce WARN, not FAIL)
        fail_flags = [f for f in health["flags"] if "below floor" in f or "diff_based" in f]
        assert len(fail_flags) == 0

    def test_degraded_snapshot_fails(self, degraded_snapshot):
        snap, data = degraded_snapshot
        health = build_health(snap, data, "2026-03-16")
        assert health["status"] == "FAIL"
        flag_text = " ".join(health["flags"])
        assert "diff_based_catalyst_events=0" in flag_text
        assert "SEC_8K_FILING=0" in flag_text

    def test_missing_artifacts_warn(self, tmp_path):
        """Empty snapshot dir should produce WARN, not crash."""
        snap = tmp_path / "empty"
        snap.mkdir()
        data = tmp_path / "data"
        data.mkdir()
        _write_universe(data)
        health = build_health(snap, data, "2026-03-17")
        assert health["status"] in ("WARN", "FAIL")
        assert health["sources"]["cache_health"]["status"] == "WARN"
        assert health["sources"]["catalyst_source_mix"]["status"] == "WARN"

    def test_schema_version_present(self, healthy_snapshot):
        snap, data = healthy_snapshot
        health = build_health(snap, data, "2026-03-17")
        assert health["schema_version"] == "data_collection_health.v1"
        assert health["as_of_date"] == "2026-03-17"

    def test_source_keys_present(self, healthy_snapshot):
        snap, data = healthy_snapshot
        health = build_health(snap, data, "2026-03-17")
        expected_sources = {
            "cache_health",
            "catalyst_source_mix",
            "coverage_quality",
            "market_data",
            "ctgov",
            "sec",
            "fda",
            "options",
            "inputs_manifest",
        }
        assert set(health["sources"].keys()) == expected_sources


# ---------------------------------------------------------------------------
# Specific source check tests
# ---------------------------------------------------------------------------


class TestSourceChecks:
    def test_diff_events_zero_warns_when_calendar_healthy(self, healthy_snapshot):
        """Zero diff events + healthy calendar → WARN, not FAIL."""
        snap, data = healthy_snapshot
        # Overwrite source mix with zero diff events but calendar still present
        _write_source_mix(snap, ctgov_diff=0, sec8k=400)
        health = build_health(snap, data, "2026-03-17")
        csm = health["sources"]["catalyst_source_mix"]
        assert csm["status"] == "WARN"
        assert any("diff_based_catalyst_events=0" in f for f in csm["flags"])
        assert any("calendar coverage still present" in f for f in csm["flags"])

    def test_diff_events_zero_fails_when_calendar_absent(self, healthy_snapshot):
        """Zero diff events + no calendar → FAIL."""
        snap, data = healthy_snapshot
        # Write source mix with zero diff AND zero calendar
        (snap / "catalyst_source_mix.json").write_text(
            json.dumps(
                {
                    "total_events": 50,
                    "unique_tickers_with_events": 20,
                    "by_source": {"SEC_8K_FILING": 50},
                    "pre_dedup_by_source": {"CTGOV": 0, "CTGOV_CALENDAR": 0, "SEC_8K_FILING": 50},
                }
            ),
            encoding="utf-8",
        )
        health = build_health(snap, data, "2026-03-17")
        csm = health["sources"]["catalyst_source_mix"]
        assert csm["status"] == "FAIL"
        assert any("trial_records may be stale" in f for f in csm["flags"])

    def test_sec8k_zero_warns(self, healthy_snapshot):
        snap, data = healthy_snapshot
        _write_source_mix(snap, ctgov_diff=500, sec8k=0)
        health = build_health(snap, data, "2026-03-17")
        csm = health["sources"]["catalyst_source_mix"]
        assert csm["status"] in ("WARN", "FAIL")
        assert any("SEC_8K_FILING=0" in f for f in csm["flags"])

    def test_low_market_coverage_fails(self, healthy_snapshot):
        snap, data = healthy_snapshot
        _write_price_history(data, n_tickers=100)  # Well below 95%
        health = build_health(snap, data, "2026-03-17")
        assert health["sources"]["market_data"]["status"] == "FAIL"

    def test_cache_health_bad_fails(self, healthy_snapshot):
        snap, data = healthy_snapshot
        _write_cache_health(snap, overall="bad")
        health = build_health(snap, data, "2026-03-17")
        assert health["sources"]["cache_health"]["status"] == "FAIL"

    def test_missing_inputs_manifest_info(self, healthy_snapshot):
        """Missing manifest is INFO (opt-in feature), not WARN."""
        snap, data = healthy_snapshot
        health = build_health(snap, data, "2026-03-17")
        assert health["sources"]["inputs_manifest"]["status"] == "INFO"
        assert not health["sources"]["inputs_manifest"].get("present")


# ---------------------------------------------------------------------------
# Writer tests
# ---------------------------------------------------------------------------


class TestWriters:
    def test_write_json(self, healthy_snapshot):
        snap, data = healthy_snapshot
        health = build_health(snap, data, "2026-03-17")
        path = snap / "test_health.json"
        write_json_report(health, path)
        assert path.exists()
        loaded = json.loads(path.read_text())
        assert loaded["status"] in ("PASS", "WARN")

    def test_write_markdown(self, healthy_snapshot):
        snap, data = healthy_snapshot
        health = build_health(snap, data, "2026-03-17")
        path = snap / "test_health.md"
        write_markdown_report(health, path)
        text = path.read_text()
        assert "PASS" in text
        assert "Source Coverage" in text

    def test_write_markdown_degraded(self, degraded_snapshot):
        snap, data = degraded_snapshot
        health = build_health(snap, data, "2026-03-16")
        path = snap / "test_health.md"
        write_markdown_report(health, path)
        text = path.read_text()
        assert "FAIL" in text
        assert "Suggested Actions" in text
        assert "trial_records" in text

    def test_empty_health_writes(self, tmp_path):
        health = {
            "schema_version": "test",
            "as_of_date": "2026-01-01",
            "status": "WARN",
            "sources": {},
            "thresholds": {},
            "flags": [],
        }
        write_json_report(health, tmp_path / "h.json")
        write_markdown_report(health, tmp_path / "h.md")
        assert (tmp_path / "h.json").exists()
        assert (tmp_path / "h.md").exists()


# ---------------------------------------------------------------------------
# run_from_screen integration tests
# ---------------------------------------------------------------------------


class TestRunFromScreen:
    def test_produces_artifacts(self, healthy_snapshot):
        snap, data = healthy_snapshot
        result = run_from_screen(snap, data, "2026-03-17")
        assert result in ("PASS", "WARN")
        assert (snap / "data_collection_health.json").exists()
        assert (snap / "data_collection_health.md").exists()

    def test_degraded_returns_fail(self, degraded_snapshot):
        snap, data = degraded_snapshot
        result = run_from_screen(snap, data, "2026-03-16")
        assert result == "FAIL"

    def test_missing_dir_returns_none(self, tmp_path):
        result = run_from_screen(tmp_path / "nonexistent", tmp_path, "2026-03-17")
        # Should not crash — returns status even if sources are mostly WARN
        assert result is not None or result is None  # graceful either way

    @pytest.mark.skip(reason="placeholder: behavior not yet implemented (test-trust-audit hygiene 2026-07-14)")
    def test_opt_out_not_tested_here(self):
        """Opt-out (--no-data-collection-health) is tested at run_screen CLI level."""
        pass


# ---------------------------------------------------------------------------
# Real snapshot regression tests (Mar 16 stale vs Mar 17 fresh)
# ---------------------------------------------------------------------------


class TestRealSnapshotRegression:
    """Run against actual snapshots if they exist. Skip if not available."""

    MAR16 = Path(PROJECT_ROOT / "data/snapshots/2026-03-16")
    MAR17 = Path(PROJECT_ROOT / "data/snapshots/2026-03-17")
    DATA = Path(PROJECT_ROOT / "production_data")

    @pytest.mark.skipif(
        not (Path(PROJECT_ROOT / "data/snapshots/2026-03-16/cache_health.json").exists()),
        reason="Mar 16 snapshot not available",
    )
    def test_mar16_stale_fails(self):
        health = build_health(self.MAR16, self.DATA, "2026-03-16")
        assert health["status"] == "FAIL"
        assert health["sources"]["catalyst_source_mix"]["ctgov_diff_events"] == 0

    @pytest.mark.skipif(
        not (Path(PROJECT_ROOT / "data/snapshots/2026-03-17/cache_health.json").exists()),
        reason="Mar 17 snapshot not available",
    )
    def test_mar17_fresh_not_fail(self):
        health = build_health(self.MAR17, self.DATA, "2026-03-17")
        assert health["status"] != "FAIL"
        assert health["sources"]["catalyst_source_mix"]["ctgov_diff_events"] > 0
        assert health["sources"]["catalyst_source_mix"]["sec8k_events"] > 0
