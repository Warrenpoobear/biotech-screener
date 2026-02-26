"""Tests for cache refresh rejection → degraded cache health."""
from __future__ import annotations

import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from run_screen import load_cache_refresh_sidecar
from cache_health import compute_cache_health, _worst


def _make_sidecar(path: Path, results: list[dict]) -> None:
    """Write a cache_refresh sidecar file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "schema": "cache_refresh.v1",
        "as_of_date": "2026-02-25",
        "results": results,
    }))


def _baseline_health(**overrides) -> dict:
    """Build a baseline cache_health dict (all ok)."""
    health = compute_cache_health(
        sec8k_count=200,
        ctgov_count=18000,
        as_of_date="2026-02-25",
    )
    health.update(overrides)
    return health


def test_refresh_rejection_forces_degraded(tmp_path):
    """Sidecar with accepted=false → degraded_run=True, overall_status=bad."""
    sidecar_path = tmp_path / "cache_refresh_2026-02-25.json"
    _make_sidecar(sidecar_path, [
        {"source": "sec_8k", "count": 10, "prior_count": 200,
         "accepted": False, "committed": False,
         "reason": "collapse (ratio=0.05 < 0.3)"},
        {"source": "ctgov", "count": 18000, "prior_count": 18000,
         "accepted": True, "committed": True, "reason": ""},
    ])

    cr = load_cache_refresh_sidecar(sidecar_path)

    # Simulate the wiring in run_screen.py
    health = _baseline_health()
    health["cache_refresh_sidecar"] = cr["sidecar"]
    health["cache_refresh_had_rejections"] = cr["had_rejections"]
    health["cache_refresh_rejected_sources"] = cr["rejected_sources"]
    if cr["had_rejections"]:
        health["overall_status"] = _worst(health["overall_status"], "bad")
        health["degraded_run"] = True

    assert health["degraded_run"] is True
    assert health["overall_status"] == "bad"
    assert health["cache_refresh_had_rejections"] is True
    assert "sec_8k" in health["cache_refresh_rejected_sources"]


def test_refresh_all_accepted_does_not_force_degraded(tmp_path):
    """Sidecar with all accepted=true → no forced degradation."""
    sidecar_path = tmp_path / "cache_refresh_2026-02-25.json"
    _make_sidecar(sidecar_path, [
        {"source": "sec_8k", "count": 220, "prior_count": 219,
         "accepted": True, "committed": True, "reason": ""},
        {"source": "ctgov", "count": 18000, "prior_count": 18000,
         "accepted": True, "committed": True, "reason": ""},
    ])

    cr = load_cache_refresh_sidecar(sidecar_path)

    health = _baseline_health()
    health["cache_refresh_sidecar"] = cr["sidecar"]
    health["cache_refresh_had_rejections"] = cr["had_rejections"]
    health["cache_refresh_rejected_sources"] = cr["rejected_sources"]
    if cr["had_rejections"]:
        health["overall_status"] = _worst(health["overall_status"], "bad")
        health["degraded_run"] = True

    assert health["degraded_run"] is False
    assert health["overall_status"] == "ok"
    assert health["cache_refresh_had_rejections"] is False
    assert health["cache_refresh_rejected_sources"] == []


def test_missing_sidecar_does_not_force_degraded(tmp_path):
    """Missing sidecar → no effect on cache health."""
    cr = load_cache_refresh_sidecar(tmp_path / "nonexistent.json")

    health = _baseline_health()
    health["cache_refresh_sidecar"] = cr["sidecar"]
    health["cache_refresh_had_rejections"] = cr["had_rejections"]
    if cr["had_rejections"]:
        health["overall_status"] = _worst(health["overall_status"], "bad")
        health["degraded_run"] = True

    assert health["degraded_run"] is False
    assert health["overall_status"] == "ok"
    assert health["cache_refresh_sidecar"] is None


def test_rejection_escalates_warning_to_bad(tmp_path):
    """If health is already 'warning', rejection escalates to 'bad'."""
    sidecar_path = tmp_path / "cache_refresh_2026-02-25.json"
    _make_sidecar(sidecar_path, [
        {"source": "sec_8k", "count": 0, "prior_count": 200,
         "accepted": False, "committed": False,
         "reason": "empty_refresh"},
    ])

    cr = load_cache_refresh_sidecar(sidecar_path)

    # CTGov ratio 14000/18000 = 0.78 → below warn band [0.80, 1.25] → warning
    health = compute_cache_health(
        sec8k_count=200,
        ctgov_count=14000,
        prior_ctgov_count=18000,
        as_of_date="2026-02-25",
    )
    assert health["overall_status"] == "warning"  # precondition

    if cr["had_rejections"]:
        health["overall_status"] = _worst(health["overall_status"], "bad")
        health["degraded_run"] = True

    assert health["overall_status"] == "bad"
    assert health["degraded_run"] is True
