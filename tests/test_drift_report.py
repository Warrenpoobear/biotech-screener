"""Tests for scripts/run_drift_report.py — drift monitoring and guardrails."""
from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pandas as pd
import pytest

import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_drift_report import (
    DriftGuardrails,
    compute_drift_metrics,
    compute_snapshot_metrics,
    evaluate_guardrails,
    find_rollback_candidate,
    generate_drift_json,
    generate_drift_report_md,
    load_snapshot_window,
)
from run_phase2_snapshot_delta import SnapshotData


# ---------------------------------------------------------------------------
# Synthetic data builders
# ---------------------------------------------------------------------------
def _make_rankings(
    n_dev: int = 50,
    a_pct: float = 5.0,
    catalyst_missing_pct: float = 30.0,
    optionality_std: float = 0.30,
    ruleset_id: str = "eb833c56",
) -> pd.DataFrame:
    """Build a synthetic rankings DataFrame with controllable tier distribution."""
    n_a = max(1, int(n_dev * a_pct / 100))
    n_b = max(1, int(n_dev * 0.30))
    n_c = max(1, int(n_dev * 0.40))
    n_d = n_dev - n_a - n_b - n_c

    tiers = ["A"] * n_a + ["B"] * n_b + ["C"] * n_c + ["D"] * max(0, n_d)
    actual_n = len(tiers)

    # Build catalyst modes — some missing based on catalyst_missing_pct
    n_eligible = int(actual_n * 0.50)
    n_cat_missing = int(n_eligible * catalyst_missing_pct / 100)
    n_cat_present = n_eligible - n_cat_missing
    modes = (
        ["specific_days"] * n_cat_present
        + ["missing"] * n_cat_missing
        + ["no_upcoming"] * (actual_n - n_eligible)
    )

    # Eligible: first n_eligible are eligible
    eligible = ["1"] * n_eligible + ["0"] * (actual_n - n_eligible)

    # Optionality: generate values with roughly the requested std
    import random
    rng = random.Random(42)
    opt_vals = [round(0.5 + rng.gauss(0, optionality_std), 4) for _ in range(actual_n)]

    data = {
        "ticker": [f"T{i:03d}" for i in range(actual_n)],
        "archetype": ["drug_developer"] * actual_n,
        "tier_dev": tiers,
        "catalyst_mode": modes,
        "catalyst_days": list(range(10, 10 + actual_n)),
        "actionable_rank": list(range(1, actual_n + 1)),
        "risk_flags": [""] * actual_n,
        "target_weight_pct": [round(100 / actual_n, 2)] * actual_n,
        "decision_engine_ruleset_id": [ruleset_id] * actual_n,
        "composite_rank": list(range(1, actual_n + 1)),
        "composite_score": [50.0 - i * 0.3 for i in range(actual_n)],
        "clinical_optionality_pct_dev": opt_vals,
        "size_band": ["L"] * (actual_n // 2) + ["M"] * (actual_n - actual_n // 2),
        "decision_engine_version": ["v1.2.0"] * actual_n,
        "eligible": eligible,
    }
    return pd.DataFrame(data)


def _make_snapshot(
    date: str,
    rankings: pd.DataFrame | None = None,
    ruleset_id: str = "eb833c56",
) -> SnapshotData:
    if rankings is None:
        rankings = _make_rankings(ruleset_id=ruleset_id)
    portfolio = rankings.head(20).copy()
    return SnapshotData(
        date=date,
        path=Path(f"/fake/{date}"),
        rankings=rankings,
        portfolio=portfolio,
        metadata={"as_of_date": date},
        ruleset_id=ruleset_id,
        has_native_portfolio=True,
    )


def _write_minimal_snapshot(snap_dir: Path, date: str, ruleset_id: str = "eb833c56"):
    """Write a minimal loadable snapshot to disk."""
    d = snap_dir / date
    d.mkdir(parents=True, exist_ok=True)
    rankings = _make_rankings(ruleset_id=ruleset_id)
    rankings.to_csv(d / "rankings.csv", index=False)
    meta = {"as_of_date": date}
    with open(d / "metadata.json", "w") as f:
        json.dump(meta, f)
    return d


# ============================================================================
# TestLoadWindow
# ============================================================================
class TestLoadWindow:
    def test_loads_n_most_recent(self, tmp_path):
        """Should load the N most recent snapshots."""
        for date in ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05"]:
            _write_minimal_snapshot(tmp_path, date)
        snaps = load_snapshot_window(tmp_path, 3)
        assert len(snaps) == 3
        assert snaps[0].date == "2026-01-03"
        assert snaps[-1].date == "2026-01-05"

    def test_handles_fewer_than_n(self, tmp_path):
        """Should return what's available when < N snapshots exist."""
        for date in ["2026-01-01", "2026-01-02"]:
            _write_minimal_snapshot(tmp_path, date)
        snaps = load_snapshot_window(tmp_path, 5)
        assert len(snaps) == 2
        assert snaps[0].date == "2026-01-01"

    def test_empty_directory(self, tmp_path):
        """Empty directory returns empty list."""
        snaps = load_snapshot_window(tmp_path, 5)
        assert snaps == []


# ============================================================================
# TestDriftMetrics
# ============================================================================
class TestDriftMetrics:
    def test_tier_counts_correct(self):
        """Tier counts and percentages match input distribution."""
        rankings = _make_rankings(n_dev=100, a_pct=5.0)
        snap = _make_snapshot("2026-01-01", rankings)
        m = compute_snapshot_metrics(snap)
        assert m["tier_A_count"] == 5
        total = sum(m[f"tier_{t}_count"] for t in "ABCD")
        assert total == 100

    def test_top25_overlap_computed(self):
        """Overlap is computed between consecutive snapshots."""
        # Same rankings → 100% overlap
        rankings = _make_rankings()
        snaps = [
            _make_snapshot("2026-01-01", rankings.copy()),
            _make_snapshot("2026-01-02", rankings.copy()),
        ]
        metrics = compute_drift_metrics(snaps)
        assert metrics["snapshots"][0]["top25_overlap_pct"] is None
        assert metrics["snapshots"][1]["top25_overlap_pct"] == 100.0

    def test_rolling_aggregates(self):
        """Rolling min/max/mean computed across window."""
        snaps = []
        for i, date in enumerate(["2026-01-01", "2026-01-02", "2026-01-03"]):
            rankings = _make_rankings(n_dev=100, a_pct=5.0 + i * 2)
            snaps.append(_make_snapshot(date, rankings))
        metrics = compute_drift_metrics(snaps)
        rolling = metrics["rolling"]
        assert "tier_A_pct" in rolling
        assert rolling["tier_A_pct"]["min"] <= rolling["tier_A_pct"]["max"]
        assert rolling["tier_A_pct"]["current"] == metrics["current"]["tier_A_pct"]

    def test_single_snapshot_no_overlap(self):
        """Single snapshot has None overlap."""
        snap = _make_snapshot("2026-01-01")
        metrics = compute_drift_metrics([snap])
        assert metrics["current"]["top25_overlap_pct"] is None

    def test_mixed_rulesets_handled(self):
        """Mixed rulesets in window don't error — just annotated."""
        snaps = [
            _make_snapshot("2026-01-01", ruleset_id="aaa11111"),
            _make_snapshot("2026-01-02", ruleset_id="bbb22222"),
        ]
        metrics = compute_drift_metrics(snaps)
        rulesets = {s["ruleset_id"] for s in metrics["snapshots"]}
        assert len(rulesets) == 2


# ============================================================================
# TestGuardrailEvaluation
# ============================================================================
class TestGuardrailEvaluation:
    def _metrics_with_current(self, **overrides) -> dict:
        """Build a metrics dict with controllable current snapshot values."""
        current = {
            "tier_A_pct": 5.0,
            "catalyst_missing_pct": 30.0,
            "top25_overlap_pct": 80.0,
            "optionality_std": 0.30,
        }
        current.update(overrides)
        return {"current": current, "snapshots": [current], "rolling": {}}

    def test_fail_a_pct_low(self):
        status, reasons, _ = evaluate_guardrails(
            self._metrics_with_current(tier_A_pct=1.5),
            DriftGuardrails(),
        )
        assert status == "FAIL"
        assert any("A-tier" in r and "floor" in r for r in reasons)

    def test_fail_a_pct_high(self):
        status, reasons, _ = evaluate_guardrails(
            self._metrics_with_current(tier_A_pct=16.0),
            DriftGuardrails(),
        )
        assert status == "FAIL"
        assert any("A-tier" in r and "ceiling" in r for r in reasons)

    def test_fail_catalyst_missing_high(self):
        status, reasons, _ = evaluate_guardrails(
            self._metrics_with_current(catalyst_missing_pct=90.0),
            DriftGuardrails(),
        )
        assert status == "FAIL"
        assert any("Catalyst" in r for r in reasons)

    def test_fail_overlap_low(self):
        status, reasons, _ = evaluate_guardrails(
            self._metrics_with_current(top25_overlap_pct=40.0),
            DriftGuardrails(),
        )
        assert status == "FAIL"
        assert any("overlap" in r.lower() for r in reasons)

    def test_fail_dispersion_low(self):
        status, reasons, _ = evaluate_guardrails(
            self._metrics_with_current(optionality_std=0.05),
            DriftGuardrails(),
        )
        assert status == "FAIL"
        assert any("Optionality" in r for r in reasons)

    def test_ok_all_within_bounds(self):
        status, reasons, rollback = evaluate_guardrails(
            self._metrics_with_current(),
            DriftGuardrails(),
        )
        assert status == "OK"
        assert reasons == []
        assert rollback is None


# ============================================================================
# TestRollbackCandidate
# ============================================================================
class TestRollbackCandidate:
    def test_returns_most_recently_retired(self, tmp_path):
        manifest = {
            "schema_version": 1,
            "rulesets": [
                {"id": "old1", "file": "old.json", "status": "retired",
                 "updated_at": "2025-01-01T00:00:00Z"},
                {"id": "old2", "file": "newer.json", "status": "retired",
                 "updated_at": "2026-01-01T00:00:00Z"},
                {"id": "active1", "file": "active.json", "status": "active"},
            ],
        }
        mpath = tmp_path / "manifest.json"
        with open(mpath, "w") as f:
            json.dump(manifest, f)
        result = find_rollback_candidate(mpath)
        assert result is not None
        assert result["id"] == "old2"

    def test_returns_none_if_no_retired(self, tmp_path):
        manifest = {
            "schema_version": 1,
            "rulesets": [
                {"id": "active1", "file": "active.json", "status": "active"},
            ],
        }
        mpath = tmp_path / "manifest.json"
        with open(mpath, "w") as f:
            json.dump(manifest, f)
        result = find_rollback_candidate(mpath)
        assert result is None

    def test_correct_id_and_file(self, tmp_path):
        manifest = {
            "schema_version": 1,
            "rulesets": [
                {"id": "abc123", "file": "v_old.json", "status": "retired",
                 "updated_at": "2026-02-01T00:00:00Z"},
            ],
        }
        mpath = tmp_path / "manifest.json"
        with open(mpath, "w") as f:
            json.dump(manifest, f)
        result = find_rollback_candidate(mpath)
        assert result["id"] == "abc123"
        assert result["file"] == "v_old.json"


# ============================================================================
# TestDriftGuardrails
# ============================================================================
class TestDriftGuardrails:
    def test_frozen_immutability(self):
        g = DriftGuardrails()
        with pytest.raises(FrozenInstanceError):
            g.fail_a_pct_low = 999.0  # type: ignore[misc]

    def test_json_round_trip(self, tmp_path):
        g = DriftGuardrails(fail_a_pct_low=3.0, window_size=10)
        path = tmp_path / "guardrails.json"
        g.to_json(str(path))
        g2 = DriftGuardrails.from_json(str(path))
        assert g == g2
        assert g.guardrails_id == g2.guardrails_id

    def test_deterministic_id(self):
        g1 = DriftGuardrails()
        g2 = DriftGuardrails()
        assert g1.guardrails_id == g2.guardrails_id
        g3 = DriftGuardrails(fail_a_pct_low=3.0)
        assert g1.guardrails_id != g3.guardrails_id


# ============================================================================
# TestReportGeneration
# ============================================================================
class TestReportGeneration:
    def test_md_report_contains_key_sections(self):
        snap = _make_snapshot("2026-01-01")
        metrics = compute_drift_metrics([snap])
        guardrails = DriftGuardrails()
        md = generate_drift_report_md(metrics, "OK", [], guardrails)
        assert "# Daily Drift Report" in md
        assert "## Current Snapshot" in md
        assert "## Guardrails: OK" in md
        assert "## Guardrails Config" in md

    def test_json_report_structure(self):
        snap = _make_snapshot("2026-01-01")
        metrics = compute_drift_metrics([snap])
        guardrails = DriftGuardrails()
        report = generate_drift_json(metrics, "OK", [], guardrails)
        assert report["status"] == "OK"
        assert "current" in report
        assert "rolling" in report
        assert "guardrails" in report
        assert "guardrails_id" in report["guardrails"]

    def test_fail_report_includes_rollback(self):
        snap = _make_snapshot("2026-01-01")
        metrics = compute_drift_metrics([snap])
        guardrails = DriftGuardrails()
        rollback = {"id": "old123", "file": "old.json"}
        md = generate_drift_report_md(
            metrics, "FAIL", ["A-tier too low"], guardrails, rollback
        )
        assert "Rollback candidate" in md
        assert "old123" in md

    def test_mixed_rulesets_warning_in_md(self):
        snaps = [
            _make_snapshot("2026-01-01", ruleset_id="aaa11111"),
            _make_snapshot("2026-01-02", ruleset_id="bbb22222"),
        ]
        metrics = compute_drift_metrics(snaps)
        guardrails = DriftGuardrails()
        md = generate_drift_report_md(metrics, "OK", [], guardrails)
        assert "Mixed Rulesets" in md
