"""Tests for cache_health sentinel (SEC 8-K / CTGov anomaly detection)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cache_health import (
    compute_cache_health,
    load_prior_cache_health,
    SEC8K_MIN_RATIO_VS_PRIOR,
    CTGOV_BAD_RATIO_HIGH,
    CTGOV_BAD_RATIO_LOW,
    CTGOV_WARN_RATIO_HIGH,
    CTGOV_WARN_RATIO_LOW,
)


# =============================================================================
# compute_cache_health — unit tests
# =============================================================================


class TestSec8kZero:
    """SEC 8-K count of 0 is a clear outage."""

    def test_sec8k_zero_is_bad(self):
        h = compute_cache_health(sec8k_count=0, ctgov_count=1000)
        assert h["sec8k"]["status"] == "bad"
        assert h["overall_status"] == "bad"
        assert h["degraded_run"] is True

    def test_sec8k_zero_reason_nonempty(self):
        h = compute_cache_health(sec8k_count=0, ctgov_count=1000)
        assert "empty" in h["sec8k"]["reason"].lower()

    def test_sec8k_zero_no_prior(self):
        """Zero is bad even without a prior baseline."""
        h = compute_cache_health(sec8k_count=0, ctgov_count=1000)
        assert h["sec8k"]["status"] == "bad"
        assert h["comparisons"]["sec8k_ratio_vs_prior"] is None


class TestSec8kCollapse:
    """SEC 8-K sharp collapse vs prior."""

    def test_collapse_below_threshold(self):
        h = compute_cache_health(
            sec8k_count=5, ctgov_count=1000,
            prior_sec8k_count=100,
        )
        assert h["sec8k"]["status"] == "warning"
        assert h["comparisons"]["sec8k_ratio_vs_prior"] == 0.05
        assert h["degraded_run"] is True

    def test_moderate_drop_is_ok(self):
        """50% drop (0.50 > 0.30) should be ok."""
        h = compute_cache_health(
            sec8k_count=50, ctgov_count=1000,
            prior_sec8k_count=100,
        )
        assert h["sec8k"]["status"] == "ok"

    def test_prior_zero_no_ratio(self):
        """Prior sec8k=0 should not produce a ratio (avoid div by zero)."""
        h = compute_cache_health(
            sec8k_count=10, ctgov_count=1000,
            prior_sec8k_count=0,
        )
        assert h["sec8k"]["status"] == "ok"
        assert h["comparisons"]["sec8k_ratio_vs_prior"] is None


class TestCtgovBand:
    """CTGov ratio outside warning/bad bands."""

    def test_extreme_jump_is_bad(self):
        """Ratio > 1.50 is bad."""
        h = compute_cache_health(
            sec8k_count=100, ctgov_count=1600,
            prior_ctgov_count=1000,
        )
        assert h["ctgov"]["status"] == "bad"
        assert h["overall_status"] == "bad"

    def test_extreme_drop_is_bad(self):
        """Ratio < 0.60 is bad."""
        h = compute_cache_health(
            sec8k_count=100, ctgov_count=500,
            prior_ctgov_count=1000,
        )
        assert h["ctgov"]["status"] == "bad"

    def test_moderate_jump_is_warning(self):
        """Ratio 1.30 (between 1.25 and 1.50) is warning."""
        h = compute_cache_health(
            sec8k_count=100, ctgov_count=1300,
            prior_ctgov_count=1000,
        )
        assert h["ctgov"]["status"] == "warning"
        assert h["degraded_run"] is True

    def test_moderate_drop_is_warning(self):
        """Ratio 0.75 (between 0.60 and 0.80) is warning."""
        h = compute_cache_health(
            sec8k_count=100, ctgov_count=750,
            prior_ctgov_count=1000,
        )
        assert h["ctgov"]["status"] == "warning"

    def test_within_band_is_ok(self):
        """Ratio 1.05 is within [0.80, 1.25]."""
        h = compute_cache_health(
            sec8k_count=100, ctgov_count=1050,
            prior_ctgov_count=1000,
        )
        assert h["ctgov"]["status"] == "ok"
        assert h["degraded_run"] is False


class TestNoPrior:
    """No prior baselines available."""

    def test_no_prior_sec8k_nonzero(self):
        """Without priors, non-zero sec8k should be ok."""
        h = compute_cache_health(sec8k_count=50, ctgov_count=1000)
        assert h["sec8k"]["status"] == "ok"
        assert h["ctgov"]["status"] == "ok"
        assert h["overall_status"] == "ok"
        assert h["degraded_run"] is False

    def test_no_prior_comparisons_null(self):
        h = compute_cache_health(sec8k_count=50, ctgov_count=1000)
        assert h["comparisons"]["sec8k_ratio_vs_prior"] is None
        assert h["comparisons"]["ctgov_ratio_vs_prior"] is None


class TestAllOk:
    """Happy path — everything healthy."""

    def test_healthy_run(self):
        h = compute_cache_health(
            sec8k_count=200, ctgov_count=1050,
            prior_sec8k_count=180, prior_ctgov_count=1000,
            as_of_date="2026-02-25",
        )
        assert h["overall_status"] == "ok"
        assert h["degraded_run"] is False
        assert h["as_of_date"] == "2026-02-25"
        assert h["schema"] == "cache_health.v1"

    def test_thresholds_in_output(self):
        h = compute_cache_health(sec8k_count=100, ctgov_count=1000)
        t = h["thresholds"]
        assert "sec8k_min_ratio_vs_prior" in t
        assert "ctgov_warn_ratio_low" in t
        assert "ctgov_bad_ratio_high" in t


class TestWorstStatus:
    """Overall status should be the worst of sec8k and ctgov."""

    def test_bad_plus_ok(self):
        h = compute_cache_health(
            sec8k_count=0, ctgov_count=1000,
            prior_ctgov_count=1000,
        )
        assert h["sec8k"]["status"] == "bad"
        assert h["ctgov"]["status"] == "ok"
        assert h["overall_status"] == "bad"

    def test_warning_plus_ok(self):
        h = compute_cache_health(
            sec8k_count=10, ctgov_count=1050,
            prior_sec8k_count=100, prior_ctgov_count=1000,
        )
        assert h["sec8k"]["status"] == "warning"
        assert h["ctgov"]["status"] == "ok"
        assert h["overall_status"] == "warning"


# =============================================================================
# load_prior_cache_health — file discovery tests
# =============================================================================


class TestLoadPriorCacheHealth:

    def test_finds_most_recent_prior(self, tmp_path):
        for d in ("2026-02-20", "2026-02-21", "2026-02-22"):
            (tmp_path / d).mkdir()
        health = {"sec8k": {"count": 100}, "ctgov": {"count": 1000}}
        with open(tmp_path / "2026-02-21" / "cache_health.json", "w") as f:
            json.dump(health, f)
        result = load_prior_cache_health(tmp_path, "2026-02-22")
        assert result is not None
        assert result["sec8k"]["count"] == 100

    def test_returns_none_when_no_prior(self, tmp_path):
        (tmp_path / "2026-02-22").mkdir()
        result = load_prior_cache_health(tmp_path, "2026-02-22")
        assert result is None

    def test_returns_none_for_missing_dir(self, tmp_path):
        result = load_prior_cache_health(tmp_path / "nonexistent", "2026-02-22")
        assert result is None

    def test_skips_malformed_json(self, tmp_path):
        (tmp_path / "2026-02-21").mkdir()
        (tmp_path / "2026-02-21" / "cache_health.json").write_text("not json")
        result = load_prior_cache_health(tmp_path, "2026-02-22")
        assert result is None


# =============================================================================
# Gate integration (check_cache_health from run_daily_production)
# =============================================================================


class TestCacheHealthGate:

    def test_gate_pass_when_ok(self, tmp_path):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
        from run_daily_production import check_cache_health
        health = compute_cache_health(sec8k_count=100, ctgov_count=1000)
        (tmp_path / "cache_health.json").write_text(json.dumps(health))
        gate = check_cache_health(tmp_path)
        assert gate.status == "PASS"

    def test_gate_warn_when_degraded(self, tmp_path):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
        from run_daily_production import check_cache_health
        health = compute_cache_health(sec8k_count=0, ctgov_count=1000)
        (tmp_path / "cache_health.json").write_text(json.dumps(health))
        gate = check_cache_health(tmp_path)
        assert gate.status == "WARN"

    def test_gate_fail_when_bad_and_fail_flag(self, tmp_path):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
        from run_daily_production import check_cache_health
        health = compute_cache_health(sec8k_count=0, ctgov_count=1000)
        (tmp_path / "cache_health.json").write_text(json.dumps(health))
        gate = check_cache_health(tmp_path, fail_on_bad=True)
        assert gate.status == "FAIL"

    def test_gate_pass_when_no_sidecar(self, tmp_path):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
        from run_daily_production import check_cache_health
        gate = check_cache_health(tmp_path)
        assert gate.status == "PASS"
