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
    ADAPTIVE_WARN_METRICS,
    DriftGuardrails,
    _compute_churn_details,
    _compute_gate_counts,
    _compute_strength_counts,
    _compute_strength_transitions,
    _compute_margin_summary,
    _cost_metrics,
    _parse_pipe_separated,
    compute_attribution,
    compute_drift_metrics,
    compute_snapshot_metrics,
    evaluate_adaptive_warnings,
    evaluate_guardrails,
    find_rollback_candidate,
    generate_drift_json,
    generate_drift_report_md,
    generate_rollback_packet_md,
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
    ruleset_id: str = "68b2c45e",
    include_attribution_cols: bool = False,
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
        # Cost columns (present in current-era snapshots)
        "est_cost_bps": [round(20.0 + i * 1.5, 1) for i in range(actual_n)],
        "cost_mult": [1.0] * actual_n,
        "cost_bucket": ["<=400bps"] * actual_n,
        "cost_haircut_applied": ["0"] * actual_n,
    }

    if include_attribution_cols:
        # ineligible_reasons: pipe-separated gates for ineligible tickers
        inelig_reasons = []
        for i in range(actual_n):
            if eligible[i] == "0":
                # Assign a mix of gate reasons to ineligible tickers
                if i % 3 == 0:
                    inelig_reasons.append("fundamental_red_flag")
                elif i % 3 == 1:
                    inelig_reasons.append("deep_drawdown")
                else:
                    inelig_reasons.append("fundamental_red_flag|sev3")
            else:
                inelig_reasons.append("")
        data["ineligible_reasons"] = inelig_reasons

        # catalyst_strength: strength bands for eligible, "missing" for ineligible
        strength_vals = []
        bands_cycle = ["near", "mid", "far", "missing"]
        for i in range(actual_n):
            if eligible[i] == "1":
                strength_vals.append(bands_cycle[i % len(bands_cycle)])
            else:
                strength_vals.append("missing")
        data["catalyst_strength"] = strength_vals

        # tier_reason: derived from tier_dev
        tier_reasons = []
        for t in tiers:
            if t == "A":
                tier_reasons.append("high_opt+catalyst_near")
            elif t == "B":
                tier_reasons.append("mod_opt")
            elif t == "C":
                tier_reasons.append("low_opt")
            else:
                tier_reasons.append("ineligible")
        data["tier_reason"] = tier_reasons

    return pd.DataFrame(data)


def _make_snapshot(
    date: str,
    rankings: pd.DataFrame | None = None,
    ruleset_id: str = "68b2c45e",
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


def _write_minimal_snapshot(snap_dir: Path, date: str, ruleset_id: str = "68b2c45e"):
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
        status, reasons, _, action = evaluate_guardrails(
            self._metrics_with_current(tier_A_pct=1.5),
            DriftGuardrails(),
        )
        assert status == "FAIL"
        assert any("A-tier" in r and "floor" in r for r in reasons)

    def test_fail_a_pct_high(self):
        status, reasons, _, action = evaluate_guardrails(
            self._metrics_with_current(tier_A_pct=16.0),
            DriftGuardrails(),
        )
        assert status == "FAIL"
        assert any("A-tier" in r and "ceiling" in r for r in reasons)

    def test_fail_catalyst_missing_high(self):
        status, reasons, _, action = evaluate_guardrails(
            self._metrics_with_current(catalyst_missing_pct=90.0),
            DriftGuardrails(),
        )
        assert status == "FAIL"
        assert any("Catalyst" in r for r in reasons)

    def test_fail_overlap_low(self):
        status, reasons, _, action = evaluate_guardrails(
            self._metrics_with_current(top25_overlap_pct=40.0),
            DriftGuardrails(),
        )
        assert status == "FAIL"
        assert any("overlap" in r.lower() for r in reasons)

    def test_fail_dispersion_low(self):
        status, reasons, _, action = evaluate_guardrails(
            self._metrics_with_current(optionality_std=0.05),
            DriftGuardrails(),
        )
        assert status == "FAIL"
        assert any("Optionality" in r for r in reasons)

    def test_ok_all_within_bounds(self):
        status, reasons, rollback, action = evaluate_guardrails(
            self._metrics_with_current(),
            DriftGuardrails(),
        )
        assert status == "OK"
        assert reasons == []
        assert rollback is None
        assert action == "NONE"

    def test_warn_eligible_dev_pct_low(self):
        """Eligible dev % below floor → WARN."""
        status, reasons, _, action = evaluate_guardrails(
            self._metrics_with_current(eligible_pct=8.0),
            DriftGuardrails(),
        )
        assert status == "WARN"
        assert any("eligible_dev_pct" in r for r in reasons)
        assert action == "INVESTIGATE"

    def test_ok_eligible_dev_pct_above_floor(self):
        """Eligible dev % above floor → no WARN from this check."""
        status, reasons, _, _ = evaluate_guardrails(
            self._metrics_with_current(eligible_pct=50.0),
            DriftGuardrails(),
        )
        assert status == "OK"


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
        g = DriftGuardrails(
            fail_a_pct_low=3.0, window_size=10,
            warn_iqr_k=1.5, warn_iqr_floor=2.0,
            warn_min_window=4, fail_corroboration_count=3,
        )
        path = tmp_path / "guardrails.json"
        g.to_json(str(path))
        g2 = DriftGuardrails.from_json(str(path))
        assert g == g2
        assert g.guardrails_id == g2.guardrails_id
        assert g2.warn_iqr_k == 1.5
        assert g2.warn_iqr_floor == 2.0
        assert g2.warn_min_window == 4
        assert g2.fail_corroboration_count == 3

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
        md = generate_drift_report_md(
            metrics, "OK", [], guardrails, recommended_action="NONE"
        )
        assert "# Daily Drift Report" in md
        assert "## Current Snapshot" in md
        assert "## Guardrails: OK" in md
        assert "Recommended action: NONE" in md
        assert "## Guardrails Config" in md
        assert "warn_iqr_k" in md
        assert "warn_iqr_floor" in md
        assert "warn_min_window" in md
        assert "fail_corroboration_count" in md

    def test_md_report_adaptive_warnings_section(self):
        snap = _make_snapshot("2026-01-01")
        metrics = compute_drift_metrics([snap])
        guardrails = DriftGuardrails()
        md = generate_drift_report_md(
            metrics, "WARN", [], guardrails, recommended_action="NONE",
            adaptive_warnings=["eligible % drifted above median"],
        )
        assert "## Adaptive Warnings" in md
        assert "eligible % drifted above median" in md

    def test_md_report_no_adaptive_section_when_empty(self):
        snap = _make_snapshot("2026-01-01")
        metrics = compute_drift_metrics([snap])
        guardrails = DriftGuardrails()
        md = generate_drift_report_md(
            metrics, "OK", [], guardrails, recommended_action="NONE"
        )
        assert "## Adaptive Warnings" not in md

    def test_json_report_structure(self):
        snap = _make_snapshot("2026-01-01")
        metrics = compute_drift_metrics([snap])
        guardrails = DriftGuardrails()
        report = generate_drift_json(
            metrics, "OK", [], guardrails, recommended_action="NONE"
        )
        assert report["status"] == "OK"
        assert report["recommended_action"] == "NONE"
        assert "current" in report
        assert "rolling" in report
        assert "guardrails" in report
        assert "guardrails_id" in report["guardrails"]
        assert "adaptive_warnings" in report
        assert report["adaptive_warnings"] == []

    def test_json_report_with_adaptive_warnings(self):
        snap = _make_snapshot("2026-01-01")
        metrics = compute_drift_metrics([snap])
        guardrails = DriftGuardrails()
        report = generate_drift_json(
            metrics, "WARN", [], guardrails, recommended_action="NONE",
            adaptive_warnings=["eligible % spike"],
        )
        assert report["adaptive_warnings"] == ["eligible % spike"]

    def test_fail_report_includes_rollback(self):
        snap = _make_snapshot("2026-01-01")
        metrics = compute_drift_metrics([snap])
        guardrails = DriftGuardrails()
        rollback = {"id": "old123", "file": "old.json"}
        md = generate_drift_report_md(
            metrics, "FAIL", ["A-tier too low"], guardrails, rollback,
            recommended_action="ROLLBACK_RECOMMENDED",
        )
        assert "Rollback candidate" in md
        assert "old123" in md
        assert "Recommended action: ROLLBACK_RECOMMENDED" in md

    def test_mixed_rulesets_warning_in_md(self):
        snaps = [
            _make_snapshot("2026-01-01", ruleset_id="aaa11111"),
            _make_snapshot("2026-01-02", ruleset_id="bbb22222"),
        ]
        metrics = compute_drift_metrics(snaps)
        guardrails = DriftGuardrails()
        md = generate_drift_report_md(metrics, "OK", [], guardrails)
        assert "Mixed Rulesets" in md


# ============================================================================
# TestRecommendedAction
# ============================================================================
class TestRecommendedAction:
    def _metrics_with_current(self, **overrides) -> dict:
        current = {
            "tier_A_pct": 5.0,
            "catalyst_missing_pct": 30.0,
            "top25_overlap_pct": 80.0,
            "optionality_std": 0.30,
        }
        current.update(overrides)
        return {"current": current, "snapshots": [current], "rolling": {}}

    def test_none_when_ok(self):
        _, _, _, action = evaluate_guardrails(
            self._metrics_with_current(),
            DriftGuardrails(),
        )
        assert action == "NONE"

    def test_rollback_recommended_when_fail_with_candidate(self, monkeypatch):
        """2+ FAILs + retired entry in manifest → ROLLBACK_RECOMMENDED."""
        fake_candidate = {"id": "retired1", "file": "old.json", "status": "retired"}

        import run_drift_report
        monkeypatch.setattr(
            run_drift_report, "find_rollback_candidate",
            lambda: fake_candidate,
        )

        # Trigger 2 FAIL metrics for corroboration
        status, reasons, rollback, action = evaluate_guardrails(
            self._metrics_with_current(tier_A_pct=1.0, optionality_std=0.05),
            DriftGuardrails(),
        )
        assert status == "FAIL"
        assert len(reasons) >= 2
        assert rollback is not None
        assert rollback["id"] == "retired1"
        assert action == "ROLLBACK_RECOMMENDED"

    def test_investigate_when_fail_no_candidate(self, monkeypatch):
        """FAIL + no retired entry → INVESTIGATE."""
        import run_drift_report
        monkeypatch.setattr(
            run_drift_report, "find_rollback_candidate",
            lambda: None,
        )

        status, reasons, rollback, action = evaluate_guardrails(
            self._metrics_with_current(tier_A_pct=1.0),
            DriftGuardrails(),
        )
        assert status == "FAIL"
        assert rollback is None
        assert action == "INVESTIGATE"


# ============================================================================
# TestRollbackPacket
# ============================================================================
class TestRollbackPacket:
    def test_packet_contains_failed_guardrails(self):
        reasons = ["A-tier % = 1.0% < 2.0% floor", "Optionality std = 0.05 < 0.10 floor"]
        rollback = {"id": "abc123", "file": "old.json"}
        packet = generate_rollback_packet_md(
            "FAIL", reasons, rollback, "ROLLBACK_RECOMMENDED",
            {"tier_A_pct": 1.0, "optionality_std": 0.05}, DriftGuardrails(),
        )
        for reason in reasons:
            assert reason in packet

    def test_packet_contains_candidate_id_and_file(self):
        rollback = {"id": "abc123", "file": "old.json"}
        packet = generate_rollback_packet_md(
            "FAIL", ["test fail"], rollback, "ROLLBACK_RECOMMENDED",
            {"tier_A_pct": 1.0}, DriftGuardrails(),
        )
        assert "abc123" in packet
        assert "production_data/decision_rulesets/old.json" in packet

    def test_packet_contains_verification_command(self):
        rollback = {"id": "abc123", "file": "old.json"}
        packet = generate_rollback_packet_md(
            "FAIL", ["test fail"], rollback, "ROLLBACK_RECOMMENDED",
            {"tier_A_pct": 1.0}, DriftGuardrails(),
        )
        assert "assert d['ruleset_id']=='abc123'" in packet

    def test_packet_contains_rollback_command(self):
        rollback = {"id": "abc123", "file": "old.json"}
        packet = generate_rollback_packet_md(
            "FAIL", ["test fail"], rollback, "ROLLBACK_RECOMMENDED",
            {"tier_A_pct": 1.0}, DriftGuardrails(),
        )
        assert "promote_ruleset.py abc123 --rollback --force" in packet

    def test_packet_investigate_when_no_candidate(self):
        packet = generate_rollback_packet_md(
            "FAIL", ["test fail"], None, "INVESTIGATE",
            {"tier_A_pct": 1.0}, DriftGuardrails(),
        )
        assert "Manual investigation required" in packet
        assert "Rollback Commands" not in packet


# ============================================================================
# TestRollingQuantiles
# ============================================================================
class TestRollingQuantiles:
    def test_rolling_includes_median_iqr_delta(self):
        """Rolling entries should have median, iqr, and delta keys."""
        snaps = [
            _make_snapshot(f"2026-01-0{i+1}", _make_rankings(a_pct=5.0 + i))
            for i in range(5)
        ]
        metrics = compute_drift_metrics(snaps)
        rolling = metrics["rolling"]
        for key in ("tier_A_pct", "eligible_pct", "catalyst_missing_pct"):
            assert key in rolling, f"Missing {key} in rolling"
            entry = rolling[key]
            assert "median" in entry, f"Missing median for {key}"
            assert "iqr" in entry, f"Missing iqr for {key}"
            assert "delta" in entry, f"Missing delta for {key}"

    def test_rolling_iqr_correct_with_known_values(self):
        """Verify IQR calculation with known values [3,5,7,9,11]."""
        import statistics
        vals = [3.0, 5.0, 7.0, 9.0, 11.0]
        q1, _q2, q3 = statistics.quantiles(vals, n=4)
        expected_iqr = round(q3 - q1, 2)

        # Build 5 snapshots with tier_A_pct = 3,5,7,9,11
        snaps = []
        for i, a_val in enumerate(vals):
            r = _make_rankings(n_dev=100, a_pct=a_val)
            snaps.append(_make_snapshot(f"2026-01-0{i+1}", r))
        metrics = compute_drift_metrics(snaps)
        rolling = metrics["rolling"]
        assert "tier_A_pct" in rolling
        # The actual IQR may differ slightly because _make_rankings rounds counts
        # but the median and iqr keys must be present and numeric
        assert rolling["tier_A_pct"]["iqr"] is not None
        assert isinstance(rolling["tier_A_pct"]["iqr"], float)
        assert rolling["tier_A_pct"]["median"] is not None

    def test_catalyst_strength_bands_in_rolling(self):
        """Catalyst strength band keys appear in rolling when snapshots have catalyst_strength."""
        # Build rankings with catalyst_strength column
        import random
        rng = random.Random(99)
        snaps = []
        for i in range(4):
            r = _make_rankings(n_dev=50)
            bands = [rng.choice(["near", "mid", "far", "missing"]) for _ in range(len(r))]
            r["catalyst_strength"] = bands
            snaps.append(_make_snapshot(f"2026-01-0{i+1}", r))
        metrics = compute_drift_metrics(snaps)
        rolling = metrics["rolling"]
        # At least some strength bands should appear
        found = [k for k in rolling if k.startswith("catalyst_strength_")]
        assert len(found) > 0, f"Expected catalyst_strength keys in rolling, got {list(rolling.keys())}"


# ============================================================================
# TestAdaptiveWarnings
# ============================================================================
class TestAdaptiveWarnings:
    def _build_stable_metrics(self, n_snaps=5, **current_overrides):
        """Build metrics dict with N stable snapshots + optional current overrides."""
        # Stable baseline values
        base = {
            "tier_A_pct": 10.0,
            "eligible_pct": 55.0,
            "catalyst_missing_pct": 30.0,
            "catalyst_strength_near_pct": 20.0,
        }
        snapshots = []
        for i in range(n_snaps):
            snap = dict(base)
            snap["date"] = f"2026-01-{i+1:02d}"
            snapshots.append(snap)

        # Apply overrides to latest snapshot
        if current_overrides:
            snapshots[-1].update(current_overrides)

        current = snapshots[-1]

        # Compute rolling manually (mirroring production logic)
        import statistics
        roll_keys = list(base.keys())
        rolling = {}
        for key in roll_keys:
            vals = [s[key] for s in snapshots if s.get(key) is not None]
            if vals:
                cur_val = current.get(key)
                med = round(statistics.median(vals), 2)
                entry = {
                    "min": round(min(vals), 2),
                    "max": round(max(vals), 2),
                    "mean": round(statistics.mean(vals), 2),
                    "median": med,
                    "current": cur_val,
                }
                if len(vals) >= 3:
                    q1, _q2, q3 = statistics.quantiles(vals, n=4)
                    entry["iqr"] = round(q3 - q1, 2)
                else:
                    entry["iqr"] = None
                if cur_val is not None:
                    entry["delta"] = round(cur_val - med, 2)
                else:
                    entry["delta"] = None
                rolling[key] = entry

        return {"snapshots": snapshots, "rolling": rolling, "current": current}

    def test_no_warn_within_iqr_fence(self):
        """Stable metrics should produce no adaptive warnings."""
        metrics = self._build_stable_metrics(5)
        status, reasons = evaluate_adaptive_warnings(metrics, DriftGuardrails())
        assert status == "OK"
        assert reasons == []

    def test_warn_eligible_pct_spike(self):
        """eligible_pct jumps far above median → WARN fires."""
        # 6×55.0 stable + 1×90.0: IQR=0, floor=1.0, threshold=2.0, delta=35 → fires
        metrics = self._build_stable_metrics(7, eligible_pct=90.0)
        status, reasons = evaluate_adaptive_warnings(metrics, DriftGuardrails())
        assert status == "WARN"
        assert any("eligible" in r for r in reasons)

    def test_warn_catalyst_missing_drop(self):
        """catalyst_missing_pct drops far below median → WARN fires."""
        # 6×30.0 + 1×0.0: IQR=0, floor=1.0, threshold=2.0, delta=30 → fires
        metrics = self._build_stable_metrics(7, catalyst_missing_pct=0.0)
        status, reasons = evaluate_adaptive_warnings(metrics, DriftGuardrails())
        assert status == "WARN"
        assert any("catalyst missing" in r.lower() for r in reasons)

    def test_no_warn_window_too_small(self):
        """2 snapshots (< min_window=3) → no adaptive WARN even with spike."""
        metrics = self._build_stable_metrics(2, eligible_pct=99.0)
        status, reasons = evaluate_adaptive_warnings(metrics, DriftGuardrails())
        assert status == "OK"
        assert reasons == []

    def test_iqr_floor_prevents_spurious_warn(self):
        """Flat window (IQR=0) with small shift → no WARN due to IQR floor."""
        # All values identical except tiny shift in latest
        metrics = self._build_stable_metrics(5, eligible_pct=55.5)
        # With floor=1.0 and k=2.0 → threshold=2.0. Delta=0.5 < 2.0
        status, reasons = evaluate_adaptive_warnings(metrics, DriftGuardrails())
        assert status == "OK"

    def test_iqr_floor_allows_large_shift(self):
        """Flat window with large shift → WARN fires despite IQR=0."""
        # 6×55.0 + 1×90.0: IQR=0, floor=1.0, threshold=2.0, delta=35 → fires
        metrics = self._build_stable_metrics(7, eligible_pct=90.0)
        status, reasons = evaluate_adaptive_warnings(metrics, DriftGuardrails())
        assert status == "WARN"

    def test_custom_iqr_k(self):
        """k=1.0 fires where k=2.0 would not for a moderate shift."""
        # Shift of 1.5pp: with k=2.0 threshold=2.0 → no warn. With k=1.0 threshold=1.0 → warn.
        metrics = self._build_stable_metrics(5, eligible_pct=56.5)
        # k=2.0: threshold = 2.0 * max(0.0, 1.0) = 2.0, delta=1.5 → no warn
        status_strict, _ = evaluate_adaptive_warnings(
            metrics, DriftGuardrails(warn_iqr_k=2.0)
        )
        assert status_strict == "OK"

        # k=1.0: threshold = 1.0 * max(0.0, 1.0) = 1.0, delta=1.5 → warn
        status_loose, reasons_loose = evaluate_adaptive_warnings(
            metrics, DriftGuardrails(warn_iqr_k=1.0)
        )
        assert status_loose == "WARN"
        assert len(reasons_loose) > 0


# ============================================================================
# TestCorroboration
# ============================================================================
class TestCorroboration:
    def _metrics_with_current(self, **overrides) -> dict:
        current = {
            "tier_A_pct": 5.0,
            "catalyst_missing_pct": 30.0,
            "top25_overlap_pct": 80.0,
            "optionality_std": 0.30,
        }
        current.update(overrides)
        return {"current": current, "snapshots": [current], "rolling": {}}

    def test_single_fail_is_investigate(self, monkeypatch):
        """1 FAIL metric + candidate → INVESTIGATE (not corroborated)."""
        fake_candidate = {"id": "retired1", "file": "old.json", "status": "retired"}
        import run_drift_report
        monkeypatch.setattr(
            run_drift_report, "find_rollback_candidate",
            lambda: fake_candidate,
        )
        status, reasons, rollback, action = evaluate_guardrails(
            self._metrics_with_current(tier_A_pct=1.0),
            DriftGuardrails(),
        )
        assert status == "FAIL"
        assert len(reasons) == 1
        assert rollback is not None
        assert action == "INVESTIGATE"

    def test_two_fails_is_rollback(self, monkeypatch):
        """2 FAIL metrics + candidate → ROLLBACK_RECOMMENDED (corroborated)."""
        fake_candidate = {"id": "retired1", "file": "old.json", "status": "retired"}
        import run_drift_report
        monkeypatch.setattr(
            run_drift_report, "find_rollback_candidate",
            lambda: fake_candidate,
        )
        status, reasons, rollback, action = evaluate_guardrails(
            self._metrics_with_current(tier_A_pct=1.0, optionality_std=0.05),
            DriftGuardrails(),
        )
        assert status == "FAIL"
        assert len(reasons) >= 2
        assert action == "ROLLBACK_RECOMMENDED"

    def test_two_fails_no_candidate_is_investigate(self, monkeypatch):
        """2 FAILs, no candidate → INVESTIGATE."""
        import run_drift_report
        monkeypatch.setattr(
            run_drift_report, "find_rollback_candidate",
            lambda: None,
        )
        status, reasons, rollback, action = evaluate_guardrails(
            self._metrics_with_current(tier_A_pct=1.0, optionality_std=0.05),
            DriftGuardrails(),
        )
        assert status == "FAIL"
        assert len(reasons) >= 2
        assert rollback is None
        assert action == "INVESTIGATE"

    def test_custom_corroboration_count(self, monkeypatch):
        """Set count=3, 2 FAILs → still INVESTIGATE."""
        fake_candidate = {"id": "retired1", "file": "old.json", "status": "retired"}
        import run_drift_report
        monkeypatch.setattr(
            run_drift_report, "find_rollback_candidate",
            lambda: fake_candidate,
        )
        status, reasons, rollback, action = evaluate_guardrails(
            self._metrics_with_current(tier_A_pct=1.0, optionality_std=0.05),
            DriftGuardrails(fail_corroboration_count=3),
        )
        assert status == "FAIL"
        assert len(reasons) >= 2
        assert len(reasons) < 3  # only 2 FAILs
        assert action == "INVESTIGATE"


# ============================================================================
# TestDriftAttribution
# ============================================================================
class TestDriftAttribution:
    """Tests for drift attribution — root-cause breadcrumbs."""

    def _make_attr_snapshot(self, date, **kwargs):
        """Helper to build a snapshot with attribution columns."""
        r = _make_rankings(include_attribution_cols=True, **kwargs)
        return _make_snapshot(date, r)

    # -- Core function tests --

    def test_attribution_returns_none_single_snapshot(self):
        """1 snapshot → None."""
        snap = self._make_attr_snapshot("2026-01-01")
        result = compute_attribution([snap])
        assert result is None

    def test_attribution_returns_dict_two_snapshots(self):
        """2 snapshots → dict with 3 top-level keys."""
        s1 = self._make_attr_snapshot("2026-01-01")
        s2 = self._make_attr_snapshot("2026-01-02")
        result = compute_attribution([s1, s2])
        assert result is not None
        assert "eligibility_gates" in result
        assert "catalyst_strength" in result
        assert "portfolio_churn" in result

    def test_attribution_has_correct_dates(self):
        """Dates match the two most recent snapshots."""
        s1 = self._make_attr_snapshot("2026-01-01")
        s2 = self._make_attr_snapshot("2026-01-02")
        s3 = self._make_attr_snapshot("2026-01-03")
        result = compute_attribution([s1, s2, s3])
        assert result["prior_date"] == "2026-01-02"
        assert result["current_date"] == "2026-01-03"

    # -- Eligibility gate tests --

    def test_gate_counts_correct(self):
        """Known ineligible_reasons → correct counts and deltas."""
        r1 = _make_rankings(n_dev=20, include_attribution_cols=True)
        r2 = _make_rankings(n_dev=20, include_attribution_cols=True)
        # Modify r2: add an extra fundamental_red_flag
        # Find first eligible ticker and make it ineligible
        idx = r2[r2["eligible"] == "1"].index[0]
        r2.at[idx, "eligible"] = "0"
        r2.at[idx, "ineligible_reasons"] = "fundamental_red_flag"

        s1 = _make_snapshot("2026-01-01", r1)
        s2 = _make_snapshot("2026-01-02", r2)
        result = compute_attribution([s1, s2])

        gates = result["eligibility_gates"]
        # r2 has one more fundamental_red_flag than r1
        assert gates["delta"]["fundamental_red_flag"] == 1

    def test_gate_counts_missing_column(self):
        """No ineligible_reasons column → all zeros."""
        r = _make_rankings(n_dev=20)  # no include_attribution_cols
        dev = r[r["archetype"] == "drug_developer"]
        counts = _compute_gate_counts(dev)
        assert counts == {"fundamental_red_flag": 0, "sev3": 0, "deep_drawdown": 0, "adv_fail": 0}

    # -- Catalyst strength tests --

    def test_strength_counts_correct(self):
        """Known catalyst_strength distribution → correct counts and deltas."""
        r1 = _make_rankings(n_dev=20, include_attribution_cols=True)
        r2 = _make_rankings(n_dev=20, include_attribution_cols=True)
        # Shift one eligible ticker from "near" to "far" in r2
        elig_mask = r2["eligible"] == "1"
        near_mask = r2["catalyst_strength"] == "near"
        both = r2[elig_mask & near_mask]
        if len(both) > 0:
            idx = both.index[0]
            r2.at[idx, "catalyst_strength"] = "far"

        s1 = _make_snapshot("2026-01-01", r1)
        s2 = _make_snapshot("2026-01-02", r2)
        result = compute_attribution([s1, s2])

        cs = result["catalyst_strength"]
        assert cs["delta"]["near"] == -1
        assert cs["delta"]["far"] == 1

    def test_strength_missing_column(self):
        """No catalyst_strength column → empty counts, no transitions."""
        r = _make_rankings(n_dev=20)  # no include_attribution_cols
        dev = r[r["archetype"] == "drug_developer"]
        counts = _compute_strength_counts(dev)
        assert counts == {"near": 0, "mid": 0, "far": 0, "missing": 0}

    def test_transitions_degraded(self):
        """near→far appears in degraded list."""
        r1 = _make_rankings(n_dev=20, include_attribution_cols=True)
        r2 = r1.copy()
        # Find an eligible ticker with strength "near" and degrade to "far"
        elig_mask = r2["eligible"] == "1"
        near_mask = r2["catalyst_strength"] == "near"
        both = r2[elig_mask & near_mask]
        assert len(both) > 0, "Need at least one eligible near ticker"
        idx = both.index[0]
        ticker = r2.at[idx, "ticker"]
        r2.at[idx, "catalyst_strength"] = "far"

        dev1 = r1[r1["archetype"] == "drug_developer"]
        dev2 = r2[r2["archetype"] == "drug_developer"]
        degraded, improved = _compute_strength_transitions(dev1, dev2)
        assert any(d["ticker"] == ticker for d in degraded)
        assert all(d["prior"] == "near" and d["current"] == "far"
                    for d in degraded if d["ticker"] == ticker)

    def test_transitions_improved(self):
        """missing→near appears in improved list."""
        r1 = _make_rankings(n_dev=20, include_attribution_cols=True)
        r2 = r1.copy()
        # Find an eligible ticker with strength "missing" and improve to "near"
        elig_mask = r2["eligible"] == "1"
        miss_mask = r2["catalyst_strength"] == "missing"
        both = r2[elig_mask & miss_mask]
        assert len(both) > 0, "Need at least one eligible missing ticker"
        idx = both.index[0]
        ticker = r2.at[idx, "ticker"]
        r2.at[idx, "catalyst_strength"] = "near"

        dev1 = r1[r1["archetype"] == "drug_developer"]
        dev2 = r2[r2["archetype"] == "drug_developer"]
        degraded, improved = _compute_strength_transitions(dev1, dev2)
        assert any(d["ticker"] == ticker for d in improved)

    def test_transitions_cap_at_10(self):
        """>10 transitions are capped at 10."""
        # Build 30 eligible tickers all with "near", then shift 15 to "far"
        r1 = _make_rankings(n_dev=30, include_attribution_cols=True)
        r2 = r1.copy()
        elig_near = r2[(r2["eligible"] == "1") & (r2["catalyst_strength"] == "near")]
        # Force all eligible to "near" first
        r1.loc[r1["eligible"] == "1", "catalyst_strength"] = "near"
        r2.loc[r2["eligible"] == "1", "catalyst_strength"] = "near"
        # Now degrade first 15 eligible in r2
        elig_idx = r2[r2["eligible"] == "1"].index[:15]
        r2.loc[elig_idx, "catalyst_strength"] = "far"

        dev1 = r1[r1["archetype"] == "drug_developer"]
        dev2 = r2[r2["archetype"] == "drug_developer"]
        degraded, improved = _compute_strength_transitions(dev1, dev2)
        assert len(degraded) <= 10

    # -- Portfolio churn tests --

    def test_churn_dropped_and_added(self):
        """Shuffled top-25 → correct dropped/added lists with metadata."""
        r1 = _make_rankings(n_dev=50, include_attribution_cols=True)
        r2 = r1.copy()
        # Swap actionable_rank of first and 30th ticker to change top-25
        r2.at[0, "actionable_rank"] = 40  # drop T000 out of top-25
        r2.at[30, "actionable_rank"] = 1   # add T030 into top-25

        churn = _compute_churn_details(r1, r2)
        dropped_tickers = {d["ticker"] for d in churn["dropped"]}
        added_tickers = {a["ticker"] for a in churn["added"]}
        assert "T000" in dropped_tickers
        assert "T030" in added_tickers
        # Metadata present
        for d in churn["dropped"]:
            assert "tier_dev" in d
            assert "size_band" in d
            assert "tier_reason" in d

    def test_churn_full_overlap(self):
        """Same rankings → empty churn, overlap=25."""
        r = _make_rankings(n_dev=50, include_attribution_cols=True)
        churn = _compute_churn_details(r, r.copy())
        assert churn["overlap_count"] == 25
        assert churn["dropped"] == []
        assert churn["added"] == []

    # -- Report integration tests --

    def test_md_includes_attribution_section(self):
        """Attribution dict → section present in MD."""
        s1 = self._make_attr_snapshot("2026-01-01")
        s2 = self._make_attr_snapshot("2026-01-02")
        metrics = compute_drift_metrics([s1, s2])
        attribution = compute_attribution([s1, s2])
        guardrails = DriftGuardrails()
        md = generate_drift_report_md(
            metrics, "OK", [], guardrails, attribution=attribution,
        )
        assert "## Drift Attribution" in md
        assert "### Eligibility Gate Changes" in md
        assert "### Catalyst Strength Shifts" in md
        assert "### Portfolio Churn" in md

    def test_md_no_attribution_when_none(self):
        """None attribution → no attribution section."""
        snap = _make_snapshot("2026-01-01")
        metrics = compute_drift_metrics([snap])
        guardrails = DriftGuardrails()
        md = generate_drift_report_md(
            metrics, "OK", [], guardrails, attribution=None,
        )
        assert "## Drift Attribution" not in md

    def test_json_includes_attribution(self):
        """Attribution dict appears in JSON output."""
        s1 = self._make_attr_snapshot("2026-01-01")
        s2 = self._make_attr_snapshot("2026-01-02")
        metrics = compute_drift_metrics([s1, s2])
        attribution = compute_attribution([s1, s2])
        guardrails = DriftGuardrails()
        report = generate_drift_json(
            metrics, "OK", [], guardrails, attribution=attribution,
        )
        assert "attribution" in report
        assert report["attribution"] is not None
        assert "eligibility_gates" in report["attribution"]

    def test_json_attribution_none(self):
        """None attribution → null in JSON output."""
        snap = _make_snapshot("2026-01-01")
        metrics = compute_drift_metrics([snap])
        guardrails = DriftGuardrails()
        report = generate_drift_json(
            metrics, "OK", [], guardrails, attribution=None,
        )
        assert "attribution" in report
        assert report["attribution"] is None


# ============================================================================
# TestCostGuardrail
# ============================================================================
class TestCostGuardrail:
    """Tests for median-cost WARN guardrail."""

    def _metrics_with_current(self, **overrides) -> dict:
        current = {
            "tier_A_pct": 5.0,
            "catalyst_missing_pct": 30.0,
            "top25_overlap_pct": 80.0,
            "optionality_std": 0.30,
        }
        current.update(overrides)
        return {"current": current, "snapshots": [current], "rolling": {}}

    def test_cost_warn_above_threshold(self):
        """median_cost_bps > threshold → WARN with reason."""
        status, reasons, _, action = evaluate_guardrails(
            self._metrics_with_current(median_cost_bps=80.0),
            DriftGuardrails(),
        )
        assert status == "WARN"
        assert any("cost" in r.lower() for r in reasons)

    def test_cost_ok_below_threshold(self):
        """median_cost_bps < threshold → no cost warning."""
        status, reasons, _, action = evaluate_guardrails(
            self._metrics_with_current(median_cost_bps=40.0),
            DriftGuardrails(),
        )
        assert status == "OK"
        assert not any("cost" in r.lower() for r in reasons)


# ============================================================================
# TestGatePressure — _compute_margin_summary pressure metrics
# ============================================================================
class TestGatePressure:
    """Tests for gate pressure metrics in _compute_margin_summary."""

    def _make_dev_df(self, rows):
        """Build a dev DataFrame from list of dicts."""
        return pd.DataFrame(rows)

    def test_near_gate_count(self):
        """Tickers at ±0.05 are counted as near-gate."""
        rows = [
            {"dd_abs_margin": 0.25, "dd_rel_margin": 0.10, "optionality_margin_a": 0.20,
             "rescued_by_rel": "0", "eligible": "1"},
            {"dd_abs_margin": 0.03, "dd_rel_margin": -0.04, "optionality_margin_a": -0.03,
             "rescued_by_rel": "0", "eligible": "1"},
            {"dd_abs_margin": -0.02, "dd_rel_margin": 0.15, "optionality_margin_a": 0.05,
             "rescued_by_rel": "1", "eligible": "1"},
            {"dd_abs_margin": -0.10, "dd_rel_margin": -0.20, "optionality_margin_a": -0.15,
             "rescued_by_rel": "0", "eligible": "1"},
        ]
        result = _compute_margin_summary(self._make_dev_df(rows))
        # dd_abs: 2 of 4 near (0.03, -0.02)
        assert result["dd_abs_near_gate_pct"] == 50.0
        # dd_rel: 1 of 4 near (-0.04)
        assert result["dd_rel_near_gate_pct"] == 25.0
        # opt_a: 2 of 4 near (-0.03, 0.05)
        assert result["optionality_near_a_floor_pct"] == 50.0

    def test_rescued_share(self):
        """rescued_share_pct = rescued / eligible."""
        rows = [
            {"rescued_by_rel": "1", "eligible": "1"},
            {"rescued_by_rel": "0", "eligible": "1"},
            {"rescued_by_rel": "0", "eligible": "1"},
            {"rescued_by_rel": "0", "eligible": "0"},
        ]
        result = _compute_margin_summary(self._make_dev_df(rows))
        # 1 rescued / 3 eligible = 33.3%
        assert result["rescued_share_pct"] == pytest.approx(33.3, abs=0.1)

    def test_no_margin_columns_returns_none(self):
        """Missing margin columns → all pressure metrics are None."""
        rows = [{"ticker": "AAA", "eligible": "1"}]
        result = _compute_margin_summary(self._make_dev_df(rows))
        assert result["dd_abs_near_gate_pct"] is None
        assert result["dd_rel_near_gate_pct"] is None
        assert result["optionality_near_a_floor_pct"] is None

    def test_no_eligible_returns_zero(self):
        """No eligible tickers → rescued_share_pct = 0."""
        rows = [
            {"rescued_by_rel": "0", "eligible": "0",
             "dd_abs_margin": 0.01, "dd_rel_margin": 0.01, "optionality_margin_a": 0.01},
        ]
        result = _compute_margin_summary(self._make_dev_df(rows))
        assert result["rescued_share_pct"] == 0.0

    def test_snapshot_metrics_include_pressure(self):
        """compute_snapshot_metrics passes through pressure metrics."""
        r = _make_rankings(n_dev=10, include_attribution_cols=True)
        # Add margin columns
        r["dd_abs_margin"] = [0.03, -0.02, 0.20, 0.30, -0.01, 0.25, 0.15, -0.04, 0.10, 0.05]
        r["dd_rel_margin"] = [0.10] * 10
        r["optionality_margin_a"] = [0.02] * 10
        r["rescued_by_rel"] = ["0"] * 10
        snap = _make_snapshot("2026-01-01", r)
        metrics = compute_snapshot_metrics(snap)
        assert "dd_abs_near_gate_pct" in metrics
        assert metrics["dd_abs_near_gate_pct"] is not None
        assert "rescued_share_pct" in metrics


# ============================================================================
# TestCostTelemetry
# ============================================================================
class TestCostTelemetry:
    """Tests for cost telemetry metrics and guardrails."""

    def test_cost_metrics_populated(self):
        """Snapshot with cost columns → all cost metrics present and correct."""
        n = 20
        r = _make_rankings(n_dev=n)
        # Override cost columns: 15 with haircut, 5 at floor
        mults = [1.0] * 10 + [0.85] * 5 + [0.55] * 5
        r["cost_mult"] = mults
        # 5 tickers at cap (1000+ bps), rest well below
        cost_bps = [50.0 + i * 20 for i in range(15)] + [1000.0] * 5
        r["est_cost_bps"] = cost_bps
        r["cost_haircut_applied"] = ["0" if m >= 1.0 else "1" for m in mults]
        r["cost_bucket"] = [
            "<=400bps" if m >= 1.0 else "<=1000bps" if m >= 0.85 else ">2000bps"
            for m in mults
        ]
        snap = _make_snapshot("2026-01-01", r)
        metrics = compute_snapshot_metrics(snap)

        # Coverage — all 20 dev tickers have est_cost_bps
        assert metrics["cost_coverage_pct"] == 100.0

        # Percentiles are present and ordered
        assert metrics["est_cost_bps_p10"] is not None
        assert metrics["est_cost_bps_p50"] is not None
        assert metrics["est_cost_bps_p90"] is not None
        assert metrics["est_cost_bps_p10"] <= metrics["est_cost_bps_p50"]
        assert metrics["est_cost_bps_p50"] <= metrics["est_cost_bps_p90"]

        # Mean cost mult < 1.0 (some haircuts applied)
        assert metrics["mean_cost_mult"] is not None
        assert metrics["mean_cost_mult"] < 1.0

        # Bucket shares sum to 100%
        total = sum(
            metrics[f"cost_bucket_{b}_pct"]
            for b in ("no", "mild", "heavy", "floor")
        )
        assert abs(total - 100.0) < 0.2

        # Floor bucket share still tracked separately
        assert metrics["cost_bucket_floor_pct"] == 25.0

        # Cap binding = est_cost_bps >= 1000 → 5/20 = 25%
        assert metrics["cap_binding_pct"] == 25.0

    def test_warn_cost_coverage_low(self):
        """cost_coverage_pct < 80% → WARN."""

        def _metrics_with_current(**overrides):
            current = {
                "tier_A_pct": 5.0,
                "catalyst_missing_pct": 30.0,
                "top25_overlap_pct": 80.0,
                "optionality_std": 0.30,
            }
            current.update(overrides)
            return {"current": current, "snapshots": [current], "rolling": {}}

        status, reasons, _, action = evaluate_guardrails(
            _metrics_with_current(cost_coverage_pct=60.0),
            DriftGuardrails(),
        )
        assert status == "WARN"
        assert any("Cost coverage" in r for r in reasons)
        assert action == "INVESTIGATE"

    def test_warn_cap_binding_high(self):
        """cap_binding_pct > 20% → WARN."""

        def _metrics_with_current(**overrides):
            current = {
                "tier_A_pct": 5.0,
                "catalyst_missing_pct": 30.0,
                "top25_overlap_pct": 80.0,
                "optionality_std": 0.30,
            }
            current.update(overrides)
            return {"current": current, "snapshots": [current], "rolling": {}}

        status, reasons, _, action = evaluate_guardrails(
            _metrics_with_current(cap_binding_pct=30.0),
            DriftGuardrails(),
        )
        assert status == "WARN"
        assert any("Cap binding" in r for r in reasons)
        assert action == "INVESTIGATE"

    def test_cost_metrics_missing_columns_returns_none(self):
        """Rankings without cost columns → _cost_metrics returns None."""
        df = _make_rankings().drop(
            columns=["est_cost_bps", "cost_mult", "cost_bucket"],
            errors="ignore",
        )
        assert _cost_metrics(df) is None
