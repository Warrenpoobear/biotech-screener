"""Tests for monitoring alert gate (thresholds + annotations)."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

import pytest

import sys

_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from scripts.evaluate_monitoring_gate import (
    evaluate,
    build_summary_table,
    main,
    _metric_status,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLDS: Dict[str, Any] = {
    "schema": "monitoring_thresholds.v1",
    "version": "v1",
    "stability": {
        "top60_overlap_warn": 70.0,
        "top60_overlap_fail": 50.0,
        "spearman_rho_warn": 0.80,
        "spearman_rho_fail": 0.60,
        "name_turnover_pct_warn": 35.0,
        "name_turnover_pct_fail": 50.0,
    },
    "coverage": {
        "eligible_pct_warn": 60.0,
        "eligible_pct_fail": 40.0,
    },
    "data_counts": {
        "sec_8k_events_warn_min": 50,
        "sec_8k_events_fail_min": 1,
        "ctgov_records_warn_min": 5000,
        "ctgov_records_fail_min": 1,
    },
    "degraded_mode": {
        "skip_stability_gates": True,
        "skip_reason": "cache_degraded_run",
    },
}


def _healthy_monitoring() -> Dict[str, Any]:
    """Return a monitoring dict where all metrics are healthy."""
    return {
        "schema": "monitoring.v1",
        "version": "1.0.0",
        "context": {
            "as_of_date": "2026-02-26",
            "cache_degraded_run": False,
            "cache_health_overall_status": "ok",
        },
        "coverage": {
            "n_universe": 300,
            "n_eligible": 210,
            "catalyst_signal_pct": 100.0,
            "coinvest_signal_pct": 100.0,
            "sec_8k_count": 220,
            "ctgov_count": 18000,
        },
        "stability": {
            "suppressed": False,
            "prior_date": "2026-02-25",
            "top_20": {
                "overlap_count": 18,
                "overlap_pct": 90.0,
                "top_k": 20,
            },
            "top_60": {
                "overlap_count": 54,
                "overlap_pct": 90.0,
                "top_k": 60,
            },
            "name_turnover_pct": 10.0,
            "rank_correlation": {
                "spearman": 0.92,
                "n_common": 200,
            },
        },
        "exposures": {},
    }


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Tests: evaluate()
# ---------------------------------------------------------------------------


class TestEvaluate:
    def test_pass_all_healthy(self):
        mon = _healthy_monitoring()
        verdict, fails, warns = evaluate(mon, DEFAULT_THRESHOLDS)
        assert verdict == "PASS"
        assert fails == []
        assert warns == []

    def test_warn_overlap_below_warn(self):
        mon = _healthy_monitoring()
        mon["stability"]["top_60"]["overlap_pct"] = 65.0
        verdict, fails, warns = evaluate(mon, DEFAULT_THRESHOLDS)
        assert verdict == "WARN"
        assert fails == []
        assert len(warns) == 1
        assert "top60_overlap" in warns[0]

    def test_fail_overlap_below_fail(self):
        mon = _healthy_monitoring()
        mon["stability"]["top_60"]["overlap_pct"] = 40.0
        verdict, fails, warns = evaluate(mon, DEFAULT_THRESHOLDS)
        assert verdict == "FAIL"
        assert len(fails) == 1
        assert "top60_overlap" in fails[0]

    def test_warn_spearman_below_warn(self):
        mon = _healthy_monitoring()
        mon["stability"]["rank_correlation"]["spearman"] = 0.75
        verdict, fails, warns = evaluate(mon, DEFAULT_THRESHOLDS)
        assert verdict == "WARN"
        assert any("spearman" in w for w in warns)

    def test_fail_spearman_below_fail(self):
        mon = _healthy_monitoring()
        mon["stability"]["rank_correlation"]["spearman"] = 0.50
        verdict, fails, warns = evaluate(mon, DEFAULT_THRESHOLDS)
        assert verdict == "FAIL"
        assert any("spearman" in f for f in fails)

    def test_warn_turnover_above_warn(self):
        mon = _healthy_monitoring()
        mon["stability"]["name_turnover_pct"] = 40.0
        verdict, fails, warns = evaluate(mon, DEFAULT_THRESHOLDS)
        assert verdict == "WARN"
        assert any("name_turnover" in w for w in warns)

    def test_fail_turnover_above_fail(self):
        mon = _healthy_monitoring()
        mon["stability"]["name_turnover_pct"] = 55.0
        verdict, fails, warns = evaluate(mon, DEFAULT_THRESHOLDS)
        assert verdict == "FAIL"
        assert any("name_turnover" in f for f in fails)

    def test_warn_eligible_pct_below_warn(self):
        mon = _healthy_monitoring()
        mon["coverage"]["n_eligible"] = 150  # 50% of 300
        verdict, fails, warns = evaluate(mon, DEFAULT_THRESHOLDS)
        assert verdict == "WARN"
        assert any("eligible_pct" in w for w in warns)

    def test_fail_eligible_pct_below_fail(self):
        mon = _healthy_monitoring()
        mon["coverage"]["n_eligible"] = 100  # 33% of 300
        verdict, fails, warns = evaluate(mon, DEFAULT_THRESHOLDS)
        assert verdict == "FAIL"
        assert any("eligible_pct" in f for f in fails)

    def test_warn_sec_8k_below_warn(self):
        mon = _healthy_monitoring()
        mon["coverage"]["sec_8k_count"] = 30
        verdict, fails, warns = evaluate(mon, DEFAULT_THRESHOLDS)
        assert verdict == "WARN"
        assert any("sec_8k" in w for w in warns)

    def test_fail_sec_8k_zero(self):
        mon = _healthy_monitoring()
        mon["coverage"]["sec_8k_count"] = 0
        verdict, fails, warns = evaluate(mon, DEFAULT_THRESHOLDS)
        assert verdict == "FAIL"
        assert any("sec_8k" in f for f in fails)

    def test_warn_ctgov_below_warn(self):
        mon = _healthy_monitoring()
        mon["coverage"]["ctgov_count"] = 3000
        verdict, fails, warns = evaluate(mon, DEFAULT_THRESHOLDS)
        assert verdict == "WARN"
        assert any("ctgov" in w for w in warns)

    def test_fail_ctgov_zero(self):
        mon = _healthy_monitoring()
        mon["coverage"]["ctgov_count"] = 0
        verdict, fails, warns = evaluate(mon, DEFAULT_THRESHOLDS)
        assert verdict == "FAIL"
        assert any("ctgov" in f for f in fails)

    def test_degraded_skips_stability(self):
        mon = _healthy_monitoring()
        mon["context"]["cache_degraded_run"] = True
        # Bad stability that would normally FAIL
        mon["stability"]["top_60"]["overlap_pct"] = 10.0
        mon["stability"]["rank_correlation"]["spearman"] = 0.10
        mon["stability"]["name_turnover_pct"] = 80.0

        verdict, fails, warns = evaluate(mon, DEFAULT_THRESHOLDS)
        assert verdict == "DEGRADED_SKIP"
        assert fails == []
        assert warns == []

    def test_degraded_still_checks_coverage(self):
        mon = _healthy_monitoring()
        mon["context"]["cache_degraded_run"] = True
        mon["coverage"]["n_eligible"] = 100  # 33% < 40% fail

        verdict, fails, warns = evaluate(mon, DEFAULT_THRESHOLDS)
        assert verdict == "FAIL"
        assert any("eligible_pct" in f for f in fails)

    def test_suppressed_stability_skips(self):
        mon = _healthy_monitoring()
        mon["stability"] = {"suppressed": True, "reason": "no_prior_snapshot"}

        verdict, fails, warns = evaluate(mon, DEFAULT_THRESHOLDS)
        assert verdict == "PASS"
        assert fails == []
        assert warns == []

    def test_multiple_warns_collected(self):
        mon = _healthy_monitoring()
        mon["stability"]["top_60"]["overlap_pct"] = 65.0
        mon["stability"]["rank_correlation"]["spearman"] = 0.75

        verdict, fails, warns = evaluate(mon, DEFAULT_THRESHOLDS)
        assert verdict == "WARN"
        assert len(warns) == 2

    def test_reasons_sorted_deterministic(self):
        mon = _healthy_monitoring()
        mon["stability"]["top_60"]["overlap_pct"] = 65.0
        mon["stability"]["rank_correlation"]["spearman"] = 0.75
        mon["coverage"]["sec_8k_count"] = 30

        _, _, warns1 = evaluate(mon, DEFAULT_THRESHOLDS)
        _, _, warns2 = evaluate(mon, DEFAULT_THRESHOLDS)
        assert warns1 == warns2
        # Verify sorted
        assert warns1 == sorted(warns1)


# ---------------------------------------------------------------------------
# Tests: _metric_status()
# ---------------------------------------------------------------------------


class TestMetricStatus:
    def test_lower_is_worse_ok(self):
        assert _metric_status(90.0, 70.0, 50.0, lower_is_worse=True) == "ok"

    def test_lower_is_worse_warn(self):
        assert _metric_status(60.0, 70.0, 50.0, lower_is_worse=True) == "WARN"

    def test_lower_is_worse_fail(self):
        assert _metric_status(40.0, 70.0, 50.0, lower_is_worse=True) == "FAIL"

    def test_higher_is_worse_ok(self):
        assert _metric_status(10.0, 35.0, 50.0, lower_is_worse=False) == "ok"

    def test_higher_is_worse_warn(self):
        assert _metric_status(40.0, 35.0, 50.0, lower_is_worse=False) == "WARN"

    def test_higher_is_worse_fail(self):
        assert _metric_status(55.0, 35.0, 50.0, lower_is_worse=False) == "FAIL"

    def test_none_thresholds_ok(self):
        assert _metric_status(10.0, None, None, lower_is_worse=True) == "ok"


# ---------------------------------------------------------------------------
# Tests: build_summary_table()
# ---------------------------------------------------------------------------


class TestBuildSummaryTable:
    def test_pass_summary_contains_key_fields(self):
        mon = _healthy_monitoring()
        summary = build_summary_table(
            mon, DEFAULT_THRESHOLDS, "PASS", [], [],
            as_of_date="2026-02-26",
        )
        assert "Monitoring Gate" in summary
        assert "2026-02-26" in summary
        assert "PASS" in summary
        assert "eligible_pct" in summary
        assert "top60_overlap" in summary
        assert "spearman_rho" in summary

    def test_fail_summary_includes_reasons(self):
        mon = _healthy_monitoring()
        summary = build_summary_table(
            mon, DEFAULT_THRESHOLDS, "FAIL",
            ["top60_overlap=40% < fail=50%"],
            [],
            as_of_date="2026-02-26",
        )
        assert "Fail reasons" in summary
        assert "top60_overlap" in summary

    def test_degraded_summary_shows_suppressed(self):
        mon = _healthy_monitoring()
        mon["stability"] = {"suppressed": True, "reason": "degraded_run"}
        summary = build_summary_table(
            mon, DEFAULT_THRESHOLDS, "DEGRADED_SKIP", [], [],
        )
        assert "suppressed" in summary
        assert "DEGRADED_SKIP" in summary


# ---------------------------------------------------------------------------
# Tests: CLI (main)
# ---------------------------------------------------------------------------


class TestCLI:
    def test_pass_exits_0(self, tmp_path):
        mon_path = tmp_path / "monitoring.json"
        thresh_path = tmp_path / "thresholds.json"
        _write_json(mon_path, _healthy_monitoring())
        _write_json(thresh_path, DEFAULT_THRESHOLDS)

        rc = main([
            "--monitoring", str(mon_path),
            "--thresholds", str(thresh_path),
            "--as-of-date", "2026-02-26",
            "--no-annotations",
        ])
        assert rc == 0

    def test_fail_exits_1(self, tmp_path):
        mon = _healthy_monitoring()
        mon["stability"]["top_60"]["overlap_pct"] = 30.0

        mon_path = tmp_path / "monitoring.json"
        thresh_path = tmp_path / "thresholds.json"
        _write_json(mon_path, mon)
        _write_json(thresh_path, DEFAULT_THRESHOLDS)

        rc = main([
            "--monitoring", str(mon_path),
            "--thresholds", str(thresh_path),
            "--no-annotations",
        ])
        assert rc == 1

    def test_bad_schema_exits_2(self, tmp_path):
        mon_path = tmp_path / "monitoring.json"
        _write_json(mon_path, {"schema": "wrong.v99"})
        thresh_path = tmp_path / "thresholds.json"
        _write_json(thresh_path, DEFAULT_THRESHOLDS)

        rc = main([
            "--monitoring", str(mon_path),
            "--thresholds", str(thresh_path),
            "--no-annotations",
        ])
        assert rc == 2

    def test_missing_file_exits_2(self, tmp_path):
        rc = main([
            "--monitoring", str(tmp_path / "nope.json"),
            "--thresholds", str(tmp_path / "also_nope.json"),
            "--no-annotations",
        ])
        assert rc == 2

    def test_degraded_exits_0(self, tmp_path):
        mon = _healthy_monitoring()
        mon["context"]["cache_degraded_run"] = True
        # Bad stability metrics
        mon["stability"]["top_60"]["overlap_pct"] = 10.0

        mon_path = tmp_path / "monitoring.json"
        thresh_path = tmp_path / "thresholds.json"
        _write_json(mon_path, mon)
        _write_json(thresh_path, DEFAULT_THRESHOLDS)

        rc = main([
            "--monitoring", str(mon_path),
            "--thresholds", str(thresh_path),
            "--no-annotations",
        ])
        assert rc == 0

    def test_writes_github_step_summary(self, tmp_path):
        summary_file = tmp_path / "step_summary.md"
        mon_path = tmp_path / "monitoring.json"
        thresh_path = tmp_path / "thresholds.json"
        _write_json(mon_path, _healthy_monitoring())
        _write_json(thresh_path, DEFAULT_THRESHOLDS)

        os.environ["GITHUB_STEP_SUMMARY"] = str(summary_file)
        try:
            rc = main([
                "--monitoring", str(mon_path),
                "--thresholds", str(thresh_path),
                "--no-annotations",
            ])
        finally:
            os.environ.pop("GITHUB_STEP_SUMMARY", None)

        assert rc == 0
        content = summary_file.read_text()
        assert "Monitoring Gate" in content
        assert "PASS" in content
