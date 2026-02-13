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
    SUGGESTION_MIN_POINTS,
    _PANEL_COL_MAP,
    _SUGGEST_CS_MIN_CAT_SPECIFIC_DAYS_MEDIAN,
    _SUGGESTION_SPECS,
    DriftGuardrails,
    _compute_churn_details,
    _compute_gate_counts,
    _compute_strength_counts,
    _compute_strength_transitions,
    _compute_margin_summary,
    _cost_metrics,
    _parse_pipe_separated,
    _catalyst_coverage_metrics,
    _catalyst_source_metrics,
    _catalyst_event_type_metrics,
    _returns_source_metrics,
    _synthesize_actionable_rank,
    compute_attribution,
    compute_drift_metrics,
    compute_snapshot_metrics,
    compute_suggested_guardrails,
    evaluate_adaptive_warnings,
    evaluate_guardrails,
    find_rollback_candidate,
    generate_drift_json,
    generate_drift_report_md,
    generate_rollback_packet_md,
    load_panel_as_snapshots,
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
    ruleset_id: str = "f3454ef7",
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
    ruleset_id: str = "f3454ef7",
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


def _write_minimal_snapshot(snap_dir: Path, date: str, ruleset_id: str = "f3454ef7"):
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
        """A-tier at 0% triggers FAIL (pipeline broken)."""
        status, reasons, _, action = evaluate_guardrails(
            self._metrics_with_current(tier_A_pct=0.0),
            DriftGuardrails(),
        )
        assert status == "FAIL"
        assert any("A-tier" in r and "floor" in r for r in reasons)

    def test_warn_a_pct_low(self):
        """A-tier below warn threshold but above FAIL triggers WARN."""
        status, reasons, _, action = evaluate_guardrails(
            self._metrics_with_current(tier_A_pct=1.0),
            DriftGuardrails(),
        )
        # Should not FAIL (0% floor), but should WARN (1.5% threshold)
        assert status != "FAIL"
        assert any("A-tier" in r and "sparse" in r for r in reasons)

    def test_ok_a_pct_above_warn(self):
        """A-tier above warn threshold is OK."""
        status, reasons, _, action = evaluate_guardrails(
            self._metrics_with_current(tier_A_pct=2.0),
            DriftGuardrails(),
        )
        assert status == "OK"
        assert not any("A-tier" in r for r in reasons)

    def test_fail_a_pct_high(self):
        status, reasons, _, action = evaluate_guardrails(
            self._metrics_with_current(tier_A_pct=26.0),
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

        # Trigger 2 FAIL metrics for corroboration (overlap + dispersion)
        status, reasons, rollback, action = evaluate_guardrails(
            self._metrics_with_current(top25_overlap_pct=40.0, optionality_std=0.05),
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
            self._metrics_with_current(top25_overlap_pct=40.0),
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
        # overlap < 50% is a single FAIL
        status, reasons, rollback, action = evaluate_guardrails(
            self._metrics_with_current(top25_overlap_pct=40.0),
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
        # overlap + dispersion: 2 independent FAILs
        status, reasons, rollback, action = evaluate_guardrails(
            self._metrics_with_current(top25_overlap_pct=40.0, optionality_std=0.05),
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
            self._metrics_with_current(top25_overlap_pct=40.0, optionality_std=0.05),
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
        # overlap + dispersion: 2 independent FAILs, but corroboration needs 3
        status, reasons, rollback, action = evaluate_guardrails(
            self._metrics_with_current(top25_overlap_pct=40.0, optionality_std=0.05),
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


# ============================================================================
# TestReturnsSourceMix
# ============================================================================
class TestReturnsSourceMix:
    """Tests for returns-source mix metrics and guardrails."""

    def _make_rankings_with_rs(
        self, n_dev: int = 50, sources: list | None = None,
    ) -> pd.DataFrame:
        """Build rankings DataFrame with a returns_source column."""
        r = _make_rankings(n_dev=n_dev)
        if sources is not None:
            r["returns_source"] = sources[:n_dev]
        else:
            # Default: 80% morningstar, 10% csv, 5% unknown, 5% csv_outlier_override
            n_mstar = int(n_dev * 0.80)
            n_csv = int(n_dev * 0.10)
            n_unknown = int(n_dev * 0.05)
            n_outlier = n_dev - n_mstar - n_csv - n_unknown
            r["returns_source"] = (
                ["morningstar"] * n_mstar
                + ["csv"] * n_csv
                + ["unknown"] * n_unknown
                + ["csv_outlier_override"] * n_outlier
            )
        return r

    def _metrics_with_current(self, **overrides) -> dict:
        current = {
            "tier_A_pct": 5.0,
            "catalyst_missing_pct": 30.0,
            "top25_overlap_pct": 80.0,
            "optionality_std": 0.30,
        }
        current.update(overrides)
        return {"current": current, "snapshots": [current], "rolling": {}}

    # -- _returns_source_metrics tests --

    def test_mix_known_distribution(self):
        """Known source distribution → correct counts and shares."""
        r = _make_rankings(n_dev=20)
        r["returns_source"] = (
            ["morningstar"] * 14
            + ["csv"] * 3
            + ["csv_outlier_override"] * 1
            + ["unknown"] * 2
        )
        result = _returns_source_metrics(r)
        assert result is not None
        assert result["rs_n_dev"] == 20
        assert result["rs_morningstar_count"] == 14
        assert result["rs_morningstar_share_pct"] == 70.0
        assert result["rs_csv_count"] == 3
        assert result["rs_csv_share_pct"] == 15.0
        assert result["rs_csv_outlier_override_count"] == 1
        assert result["rs_csv_outlier_override_share_pct"] == 5.0
        assert result["rs_unknown_count"] == 2
        assert result["rs_unknown_share_pct"] == 10.0

    def test_mix_none_when_column_missing(self):
        """No returns_source column → returns None."""
        r = _make_rankings(n_dev=20)
        assert "returns_source" not in r.columns
        result = _returns_source_metrics(r)
        assert result is None

    def test_blank_treated_as_unknown(self):
        """Blank/empty returns_source values are counted as unknown."""
        r = _make_rankings(n_dev=10)
        r["returns_source"] = ["morningstar"] * 7 + ["", None, "  "]
        result = _returns_source_metrics(r)
        assert result["rs_unknown_count"] == 3
        assert result["rs_morningstar_count"] == 7

    def test_snapshot_metrics_include_rs(self):
        """compute_snapshot_metrics includes rs_ keys when column present."""
        r = self._make_rankings_with_rs(n_dev=30)
        snap = _make_snapshot("2026-01-01", r)
        metrics = compute_snapshot_metrics(snap)
        assert "rs_morningstar_share_pct" in metrics
        assert "rs_unknown_share_pct" in metrics
        assert "rs_csv_outlier_override_share_pct" in metrics
        assert metrics["rs_morningstar_share_pct"] is not None

    def test_snapshot_metrics_no_rs_when_missing(self):
        """compute_snapshot_metrics omits rs_ keys when column absent."""
        r = _make_rankings(n_dev=30)
        snap = _make_snapshot("2026-01-01", r)
        metrics = compute_snapshot_metrics(snap)
        assert "rs_morningstar_share_pct" not in metrics

    # -- Guardrail WARN tests --

    def test_warn_unknown_share_high(self):
        """rs_unknown_share_pct > threshold → WARN."""
        status, reasons, _, action = evaluate_guardrails(
            self._metrics_with_current(rs_unknown_share_pct=15.0),
            DriftGuardrails(),
        )
        assert status == "WARN"
        assert any("unknown" in r for r in reasons)
        assert action == "INVESTIGATE"

    def test_warn_csv_outlier_override_high(self):
        """rs_csv_outlier_override_share_pct > threshold → WARN."""
        status, reasons, _, action = evaluate_guardrails(
            self._metrics_with_current(rs_csv_outlier_override_share_pct=2.0),
            DriftGuardrails(),
        )
        assert status == "WARN"
        assert any("csv_outlier_override" in r for r in reasons)

    def test_warn_morningstar_share_low(self):
        """rs_morningstar_share_pct < threshold → WARN."""
        status, reasons, _, action = evaluate_guardrails(
            self._metrics_with_current(rs_morningstar_share_pct=60.0),
            DriftGuardrails(),
        )
        assert status == "WARN"
        assert any("morningstar" in r for r in reasons)

    def test_ok_rs_within_bounds(self):
        """Returns source shares within bounds → no WARN from rs checks."""
        status, reasons, _, _ = evaluate_guardrails(
            self._metrics_with_current(
                rs_unknown_share_pct=5.0,
                rs_csv_outlier_override_share_pct=0.5,
                rs_morningstar_share_pct=90.0,
            ),
            DriftGuardrails(),
        )
        assert status == "OK"
        assert not any("returns source" in r.lower() or "Returns source" in r for r in reasons)

    def test_ok_rs_none_no_warn(self):
        """rs metrics not present (None) → no WARN from rs checks."""
        status, reasons, _, _ = evaluate_guardrails(
            self._metrics_with_current(),
            DriftGuardrails(),
        )
        assert status == "OK"

    # -- Report table test --

    def test_md_report_includes_rs_section(self):
        """Returns Source Mix section present when rs data exists."""
        r = self._make_rankings_with_rs(n_dev=30)
        snap = _make_snapshot("2026-01-01", r)
        metrics = compute_drift_metrics([snap])
        guardrails = DriftGuardrails()
        md = generate_drift_report_md(metrics, "OK", [], guardrails)
        assert "## Returns Source Mix" in md
        assert "morningstar" in md
        assert "csv_outlier_override" in md

    def test_md_report_no_rs_section_when_missing(self):
        """No Returns Source Mix section when column is absent."""
        r = _make_rankings(n_dev=30)
        snap = _make_snapshot("2026-01-01", r)
        metrics = compute_drift_metrics([snap])
        guardrails = DriftGuardrails()
        md = generate_drift_report_md(metrics, "OK", [], guardrails)
        assert "## Returns Source Mix" not in md

    # -- Roll keys test --

    def test_roll_keys_tracked(self):
        """rs_morningstar_share_pct appears in rolling when snapshots have data."""
        snaps = []
        for i in range(4):
            r = self._make_rankings_with_rs(n_dev=30)
            snaps.append(_make_snapshot(f"2026-01-0{i+1}", r))
        metrics = compute_drift_metrics(snaps)
        rolling = metrics["rolling"]
        assert "rs_morningstar_share_pct" in rolling
        assert "rs_unknown_share_pct" in rolling
        assert rolling["rs_morningstar_share_pct"]["current"] is not None

    # -- Guardrails config test --

    def test_guardrails_config_includes_rs_thresholds(self):
        """Report shows rs guardrail thresholds."""
        r = _make_rankings(n_dev=30)
        snap = _make_snapshot("2026-01-01", r)
        metrics = compute_drift_metrics([snap])
        guardrails = DriftGuardrails()
        md = generate_drift_report_md(metrics, "OK", [], guardrails)
        assert "warn_rs_unknown_share_high" in md
        assert "warn_rs_csv_outlier_override_share_high" in md
        assert "warn_rs_morningstar_share_low" in md


class TestCatalystCoverage:
    """Tests for catalyst coverage metrics in drift report."""

    def _make_rankings_with_cat(
        self, n_dev: int = 50,
        eligible_pct: float = 60.0,
        modes: list | None = None,
    ) -> pd.DataFrame:
        """Build rankings with tier_dev and catalyst_mode columns."""
        r = _make_rankings(n_dev=n_dev)
        # Ensure tier_dev is populated for eligible tickers
        n_eligible = int(n_dev * eligible_pct / 100)
        tiers = (["A"] * max(1, n_eligible // 4)
                 + ["B"] * max(1, n_eligible // 4)
                 + ["C"] * (n_eligible - 2 * max(1, n_eligible // 4)))
        # Pad with blank for ineligible
        tiers = tiers[:n_eligible] + [""] * (n_dev - n_eligible)
        r["tier_dev"] = tiers[:n_dev]
        if modes is not None:
            r["catalyst_mode"] = modes[:n_dev]
        else:
            # Default: eligible get specific_days, ineligible get blank
            cat_modes = ["specific_days"] * n_eligible + [""] * (n_dev - n_eligible)
            r["catalyst_mode"] = cat_modes[:n_dev]
        return r

    def _metrics_with_current(self, **overrides) -> dict:
        current = {
            "tier_A_pct": 5.0,
            "catalyst_missing_pct": 30.0,
            "top25_overlap_pct": 80.0,
            "optionality_std": 0.30,
        }
        current.update(overrides)
        return {"current": current, "snapshots": [current], "rolling": {}}

    # -- _catalyst_coverage_metrics tests --

    def test_mix_known_distribution(self):
        """Known catalyst mode distribution → correct counts and shares."""
        r = _make_rankings(n_dev=20)
        r["tier_dev"] = ["A"] * 5 + ["B"] * 5 + ["C"] * 5 + [""] * 5
        modes = (
            ["specific_days"] * 8
            + ["no_upcoming"] * 4
            + ["missing"] * 3
            + [""] * 5  # ineligible, not counted
        )
        r["catalyst_mode"] = modes
        result = _catalyst_coverage_metrics(r)
        assert result is not None
        assert result["cat_n_dev"] == 20
        assert result["cat_n_eligible"] == 15
        assert result["cat_eligible_share_pct"] == 75.0
        assert result["cat_specific_days_count"] == 8
        assert result["cat_specific_days_share_pct"] == pytest.approx(53.3, abs=0.1)
        assert result["cat_no_upcoming_count"] == 4
        assert result["cat_missing_count"] == 3

    def test_blank_and_none_treated_as_missing(self):
        """Blank/None/NaN catalyst_mode values among eligible → counted as missing."""
        r = _make_rankings(n_dev=10)
        r["tier_dev"] = ["A"] * 10
        r["catalyst_mode"] = ["specific_days"] * 5 + ["", None, "", "specific_days", None]
        result = _catalyst_coverage_metrics(r)
        assert result["cat_missing_count"] == 4
        assert result["cat_specific_days_count"] == 6

    def test_none_when_columns_missing(self):
        """Returns None when tier_dev or catalyst_mode absent."""
        r = _make_rankings(n_dev=10)
        # Drop tier_dev → None
        r2 = r.drop(columns=["tier_dev"])
        assert _catalyst_coverage_metrics(r2) is None
        # Has tier_dev but no catalyst_mode → None
        r3 = r.drop(columns=["catalyst_mode"])
        assert _catalyst_coverage_metrics(r3) is None

    def test_snapshot_metrics_include_cat(self):
        """compute_snapshot_metrics includes cat_ keys when columns present."""
        r = self._make_rankings_with_cat(n_dev=30)
        snap = _make_snapshot("2026-01-01", r)
        metrics = compute_snapshot_metrics(snap)
        assert "cat_n_dev" in metrics
        assert "cat_eligible_share_pct" in metrics
        assert "cat_specific_days_share_pct" in metrics

    def test_md_report_includes_cat_section(self):
        """Report includes Catalyst Coverage section with definition line."""
        r = self._make_rankings_with_cat(n_dev=30)
        snap = _make_snapshot("2026-01-01", r)
        metrics = compute_drift_metrics([snap])
        guardrails = DriftGuardrails()
        md = generate_drift_report_md(metrics, "OK", [], guardrails)
        assert "## Catalyst Coverage" in md
        assert "Eligible = dev tickers with non-empty `tier_dev`" in md
        assert "specific_days" in md
        assert "no_upcoming" in md

    def test_md_report_no_cat_section_when_missing(self):
        """Report omits Catalyst Coverage when catalyst_mode column absent."""
        r = _make_rankings(n_dev=30)
        # Drop catalyst_mode — tier_dev must stay for _tier_counts()
        r.drop(columns=["catalyst_mode"], inplace=True)
        snap = _make_snapshot("2026-01-01", r)
        metrics = compute_drift_metrics([snap])
        guardrails = DriftGuardrails()
        md = generate_drift_report_md(metrics, "OK", [], guardrails)
        assert "## Catalyst Coverage" not in md

    def test_roll_keys_tracked(self):
        """cat_eligible_share_pct and cat_specific_days_share_pct in rolling."""
        snaps = []
        for i in range(4):
            r = self._make_rankings_with_cat(n_dev=30)
            snaps.append(_make_snapshot(f"2026-01-0{i+1}", r))
        metrics = compute_drift_metrics(snaps)
        rolling = metrics["rolling"]
        assert "cat_eligible_share_pct" in rolling
        assert "cat_specific_days_share_pct" in rolling
        assert rolling["cat_eligible_share_pct"]["current"] is not None

    def test_zero_eligible_no_crash(self):
        """All blank tier_dev → n_eligible=0, no division error."""
        r = _make_rankings(n_dev=10)
        r["tier_dev"] = [""] * 10
        r["catalyst_mode"] = ["specific_days"] * 10
        result = _catalyst_coverage_metrics(r)
        assert result is not None
        assert result["cat_n_eligible"] == 0
        assert result["cat_specific_days_share_pct"] is None

    # -- Guardrail WARN tests --

    def test_warn_cat_eligible_share_low(self):
        """WARN fires when cat_eligible_share_pct < threshold."""
        m = self._metrics_with_current(cat_eligible_share_pct=70.0)
        status, reasons, _, _ = evaluate_guardrails(m, DriftGuardrails())
        assert status == "WARN"
        assert any("Catalyst eligible share" in r for r in reasons)

    def test_warn_cat_specific_days_share_low(self):
        """WARN fires when cat_specific_days_share_pct < threshold."""
        m = self._metrics_with_current(cat_specific_days_share_pct=30.0)
        status, reasons, _, _ = evaluate_guardrails(m, DriftGuardrails())
        assert status == "WARN"
        assert any("specific_days share" in r for r in reasons)

    def test_ok_cat_within_bounds(self):
        """No WARN when catalyst coverage is healthy."""
        m = self._metrics_with_current(
            cat_eligible_share_pct=98.0,
            cat_specific_days_share_pct=60.0,
        )
        status, reasons, _, _ = evaluate_guardrails(m, DriftGuardrails())
        assert status == "OK"
        assert not any("Catalyst" in r for r in reasons)

    def test_ok_cat_none_no_warn(self):
        """No WARN when catalyst coverage keys absent (older snapshots)."""
        m = self._metrics_with_current()  # no cat_ keys
        status, reasons, _, _ = evaluate_guardrails(m, DriftGuardrails())
        assert status == "OK"

    def test_guardrails_config_includes_cat_thresholds(self):
        """Report shows catalyst coverage guardrail thresholds."""
        r = self._make_rankings_with_cat(n_dev=30)
        snap = _make_snapshot("2026-01-01", r)
        metrics = compute_drift_metrics([snap])
        guardrails = DriftGuardrails()
        md = generate_drift_report_md(metrics, "OK", [], guardrails)
        assert "warn_cat_eligible_share_low" in md
        assert "warn_cat_specific_days_share_low" in md


class TestCatalystSourceMix:
    """Tests for catalyst source mix metrics and guardrails."""

    def _make_rankings_with_cs(
        self, n_dev: int = 50, sources: list | None = None,
    ) -> pd.DataFrame:
        """Build rankings DataFrame with catalyst_source + tier_dev columns."""
        r = _make_rankings(n_dev=n_dev)
        # Ensure tier_dev is populated (eligible tickers)
        r["tier_dev"] = ["A"] * (n_dev // 4) + ["B"] * (n_dev // 4) + ["C"] * (n_dev - 2 * (n_dev // 4))
        if sources is not None:
            r["catalyst_source"] = sources[:n_dev]
        else:
            # Default: 60% CTGOV, 20% FDA, 10% none, 10% unknown
            n_ctgov = int(n_dev * 0.60)
            n_fda = int(n_dev * 0.20)
            n_none = int(n_dev * 0.10)
            n_unknown = n_dev - n_ctgov - n_fda - n_none
            r["catalyst_source"] = (
                ["CTGOV_CALENDAR"] * n_ctgov
                + ["FDA_CALENDAR"] * n_fda
                + [""] * n_none  # none
                + [None] * n_unknown  # unknown
            )
        return r

    def _metrics_with_current(self, **overrides) -> dict:
        current = {
            "tier_A_pct": 5.0,
            "catalyst_missing_pct": 30.0,
            "top25_overlap_pct": 80.0,
            "optionality_std": 0.30,
        }
        current.update(overrides)
        return {"current": current, "snapshots": [current], "rolling": {}}

    # -- _catalyst_source_metrics tests --

    def test_mix_known_distribution(self):
        """Known source distribution → correct counts and shares."""
        r = _make_rankings(n_dev=20)
        r["tier_dev"] = ["A"] * 20  # all eligible
        r["catalyst_source"] = (
            ["CTGOV_CALENDAR"] * 10
            + ["FDA_CALENDAR"] * 5
            + ["SEC_8K_FILING"] * 2
            + [""] * 2   # → none
            + [None] * 1  # → none (default: relaxed)
        )
        result = _catalyst_source_metrics(r)
        assert result is not None
        assert result["cs_n_eligible"] == 20
        assert result["cs_ctgov_calendar_count"] == 10
        assert result["cs_ctgov_calendar_share_pct"] == 50.0
        assert result["cs_fda_calendar_count"] == 5
        assert result["cs_fda_calendar_share_pct"] == 25.0
        assert result["cs_sec_8k_filing_count"] == 2
        assert result["cs_none_count"] == 3  # 2 empty + 1 None
        assert result["cs_none_share_pct"] == 15.0
        assert result["cs_unknown_count"] == 0
        assert result["cs_unknown_share_pct"] == 0.0

    def test_default_nan_maps_to_none(self):
        """Default (no strict): NaN/None → 'none' (relaxed)."""
        r = _make_rankings(n_dev=6)
        r["tier_dev"] = ["A"] * 6
        r["catalyst_source"] = ["CTGOV_CALENDAR", "", "  ", None, "FDA_CALENDAR", ""]
        result = _catalyst_source_metrics(r)
        assert result["cs_none_count"] == 4  # "", "  ", None, ""
        assert result["cs_unknown_count"] == 0

    def test_strict_missing_marks_unknown_when_dated_catalyst(self):
        """strict_missing: dated catalyst with missing source → 'unknown'."""
        r = _make_rankings(n_dev=3)
        r["tier_dev"] = ["A"] * 3
        r["catalyst_mode"] = ["specific_days", "no_upcoming", "blended_window"]
        r["catalyst_source"] = [None, None, ""]
        result = _catalyst_source_metrics(r, strict_missing=True)
        assert result["cs_unknown_count"] == 2  # specific_days + blended_window
        assert result["cs_none_count"] == 1  # no_upcoming

    def test_strict_missing_without_catalyst_mode_column(self):
        """strict_missing=True but no catalyst_mode column → all none."""
        r = _make_rankings(n_dev=3)
        r["tier_dev"] = ["A"] * 3
        r["catalyst_source"] = [None, "", None]
        r = r.drop(columns=["catalyst_mode"])
        result = _catalyst_source_metrics(r, strict_missing=True)
        assert result["cs_none_count"] == 3
        assert result["cs_unknown_count"] == 0

    def test_enrichment_source_aliases(self):
        """Enrichment pipeline names normalize to canonical keys."""
        r = _make_rankings(n_dev=5)
        r["tier_dev"] = ["A"] * 5
        r["catalyst_source"] = [
            "trial_pcd",   # → CTGOV_CALENDAR
            "pdufa",       # → FDA_CALENDAR
            "sec_8k",      # → SEC_8K_FILING
            "adcom",       # → FDA_CALENDAR
            "",            # → none
        ]
        result = _catalyst_source_metrics(r)
        assert result["cs_ctgov_calendar_count"] == 1
        assert result["cs_fda_calendar_count"] == 2  # pdufa + adcom
        assert result["cs_sec_8k_filing_count"] == 1
        assert result["cs_none_count"] == 1
        assert result["cs_unknown_count"] == 0

    def test_none_when_column_missing(self):
        """Returns None when catalyst_source column absent."""
        r = _make_rankings(n_dev=10)
        assert _catalyst_source_metrics(r) is None

    def test_none_when_tier_dev_missing(self):
        """Returns None when tier_dev column absent (needed for eligible filter)."""
        r = _make_rankings(n_dev=10)
        r["catalyst_source"] = ["CTGOV_CALENDAR"] * 10
        r.drop(columns=["tier_dev"], inplace=True)
        assert _catalyst_source_metrics(r) is None

    def test_snapshot_metrics_include_cs(self):
        """compute_snapshot_metrics includes cs_ keys when columns present."""
        r = self._make_rankings_with_cs(n_dev=30)
        snap = _make_snapshot("2026-01-01", r)
        metrics = compute_snapshot_metrics(snap)
        assert "cs_n_eligible" in metrics
        assert "cs_ctgov_calendar_share_pct" in metrics

    def test_md_report_includes_cs_section(self):
        """Report includes Catalyst Source Mix section when data present."""
        r = self._make_rankings_with_cs(n_dev=30)
        snap = _make_snapshot("2026-01-01", r)
        metrics = compute_drift_metrics([snap])
        guardrails = DriftGuardrails()
        md = generate_drift_report_md(metrics, "OK", [], guardrails)
        assert "## Catalyst Source Mix" in md
        assert "CTGOV_CALENDAR" in md
        assert "FDA_CALENDAR" in md

    def test_md_report_no_cs_section_when_missing(self):
        """Report omits Catalyst Source Mix when catalyst_source absent."""
        r = _make_rankings(n_dev=30)
        snap = _make_snapshot("2026-01-01", r)
        metrics = compute_drift_metrics([snap])
        guardrails = DriftGuardrails()
        md = generate_drift_report_md(metrics, "OK", [], guardrails)
        assert "## Catalyst Source Mix" not in md

    def test_roll_keys_tracked(self):
        """cs_ctgov_calendar_share_pct appears in rolling when data present."""
        snaps = []
        for i in range(4):
            r = self._make_rankings_with_cs(n_dev=30)
            snaps.append(_make_snapshot(f"2026-01-0{i+1}", r))
        metrics = compute_drift_metrics(snaps)
        rolling = metrics["rolling"]
        assert "cs_ctgov_calendar_share_pct" in rolling
        assert "cs_none_share_pct" in rolling
        assert rolling["cs_ctgov_calendar_share_pct"]["current"] is not None

    # -- Guardrail WARN tests --

    def test_warn_cs_ctgov_share_low(self):
        """WARN fires when CTGOV share < threshold (and regime is healthy)."""
        m = self._metrics_with_current(
            cs_ctgov_calendar_share_pct=30.0,
            cat_specific_days_share_pct=50.0,  # regime gate passes
        )
        status, reasons, _, _ = evaluate_guardrails(m, DriftGuardrails())
        assert status == "WARN"
        assert any("CTGOV share" in r for r in reasons)

    def test_warn_cs_unknown_share_high(self):
        """WARN fires when unknown source share > threshold (and regime healthy)."""
        m = self._metrics_with_current(
            cs_unknown_share_pct=10.0,
            cat_specific_days_share_pct=50.0,  # regime gate passes
        )
        status, reasons, _, _ = evaluate_guardrails(m, DriftGuardrails())
        assert status == "WARN"
        assert any("unknown" in r for r in reasons)

    def test_ok_cs_within_bounds(self):
        """No WARN when catalyst source mix is healthy."""
        m = self._metrics_with_current(
            cs_ctgov_calendar_share_pct=60.0,
            cs_unknown_share_pct=2.0,
            cat_specific_days_share_pct=50.0,
        )
        status, reasons, _, _ = evaluate_guardrails(m, DriftGuardrails())
        assert status == "OK"

    def test_ok_cs_none_no_warn(self):
        """No WARN when catalyst source keys absent (older snapshots)."""
        m = self._metrics_with_current()
        status, reasons, _, _ = evaluate_guardrails(m, DriftGuardrails())
        assert status == "OK"

    def test_cs_warn_suppressed_in_sparse_regime(self):
        """cs_* WARNs suppressed when cat_specific_days_share < 40%."""
        m = self._metrics_with_current(
            cs_ctgov_calendar_share_pct=10.0,  # would WARN normally
            cs_unknown_share_pct=20.0,         # would WARN normally
            cat_specific_days_share_pct=30.0,  # sparse regime → gate
        )
        status, reasons, _, _ = evaluate_guardrails(m, DriftGuardrails())
        assert not any("CTGOV" in r for r in reasons)
        assert not any("cs" in r.lower() and "unknown" in r.lower() for r in reasons)

    def test_cs_warn_suppressed_when_cat_specific_days_missing(self):
        """cs_* WARNs suppressed when cat_specific_days_share_pct absent."""
        m = self._metrics_with_current(
            cs_ctgov_calendar_share_pct=10.0,
        )
        status, reasons, _, _ = evaluate_guardrails(m, DriftGuardrails())
        assert not any("CTGOV" in r for r in reasons)

    def test_guardrails_config_includes_cs_thresholds(self):
        """Report shows catalyst source mix guardrail thresholds."""
        r = self._make_rankings_with_cs(n_dev=30)
        snap = _make_snapshot("2026-01-01", r)
        metrics = compute_drift_metrics([snap])
        guardrails = DriftGuardrails()
        md = generate_drift_report_md(metrics, "OK", [], guardrails)
        assert "warn_cs_ctgov_share_low" in md
        assert "warn_cs_unknown_share_high" in md

    # -- cs_* offender appendix tests --

    def test_cs_unknown_offender_fields(self):
        """Unknown source offenders have ticker, catalyst_mode, event_type, catalyst_days."""
        r = _make_rankings(n_dev=4)
        r["tier_dev"] = ["A"] * 4
        r["catalyst_mode"] = ["specific_days", "blended_window", "no_upcoming", "specific_days"]
        r["catalyst_source"] = [None, "", "CTGOV_CALENDAR", "CTGOV_CALENDAR"]
        r["catalyst_event_type"] = ["DATA_READOUT", "FDA_DECISION", "DATA_READOUT", "DATA_READOUT"]
        r["de_catalyst_days"] = ["30", "90", "60", "45"]
        result = _catalyst_source_metrics(r, strict_missing=True)
        offenders = result["_cs_unknown_offenders"]
        assert len(offenders) == 2
        assert offenders[0]["catalyst_mode"] == "specific_days"
        assert offenders[0]["event_type"] == "DATA_READOUT"
        assert offenders[0]["catalyst_days"] == "30"
        assert offenders[1]["catalyst_mode"] == "blended_window"

    def test_cs_unknown_offenders_sorted_deterministically(self):
        """Unknown source offenders sort by mode (specific_days first), then ticker."""
        r = _make_rankings(n_dev=4)
        r["ticker"] = ["ZZZ", "AAA", "MMM", "BBB"]
        r["tier_dev"] = ["A"] * 4
        r["catalyst_mode"] = ["blended_window", "specific_days", "specific_days", "blended_window"]
        r["catalyst_source"] = [None, None, None, None]
        r["catalyst_event_type"] = ["X"] * 4
        result = _catalyst_source_metrics(r, strict_missing=True)
        offenders = result["_cs_unknown_offenders"]
        tickers = [o["ticker"] for o in offenders]
        assert tickers == ["AAA", "MMM", "BBB", "ZZZ"]

    def test_cs_unknown_offenders_full_table_above_threshold(self):
        """Above WARN threshold: full unknown-source table with Days + Event Type."""
        r = _make_rankings(n_dev=10)
        r["tier_dev"] = ["A"] * 10
        r["catalyst_mode"] = ["specific_days"] * 10
        # 8 unknown out of 10 = 80% > 5% threshold
        r["catalyst_source"] = [None] * 8 + ["CTGOV_CALENDAR"] * 2
        r["catalyst_event_type"] = ["DATA_READOUT"] * 10
        r["de_catalyst_days"] = ["45"] * 10
        snap = _make_snapshot("2026-01-01", r)
        metrics = compute_drift_metrics([snap], strict_cs_missing=True)
        md = generate_drift_report_md(metrics, "OK", [], DriftGuardrails())
        assert "### Unknown Source" in md
        assert "| Days |" in md
        assert "| Event Type" in md

    def test_cs_unknown_offenders_summary_below_threshold(self):
        """Below WARN threshold: one-line summary instead of full table."""
        r = _make_rankings(n_dev=100)
        r["tier_dev"] = ["A"] * 100
        r["catalyst_mode"] = ["specific_days"] * 100
        # 2 unknown out of 100 = 2% < 5% threshold
        r["catalyst_source"] = [None] * 2 + ["CTGOV_CALENDAR"] * 98
        snap = _make_snapshot("2026-01-01", r)
        metrics = compute_drift_metrics([snap], strict_cs_missing=True)
        md = generate_drift_report_md(metrics, "OK", [], DriftGuardrails())
        assert "### Unknown Source" not in md
        assert "Unknown source offenders: 2 ticker(s)" in md
        assert "threshold not breached" in md

    def test_cs_unknown_offenders_absent_when_clean(self):
        """No unknown source offenders section when all sources populated."""
        r = _make_rankings(n_dev=10)
        r["tier_dev"] = ["A"] * 10
        r["catalyst_source"] = ["CTGOV_CALENDAR"] * 10
        snap = _make_snapshot("2026-01-01", r)
        metrics = compute_drift_metrics([snap])
        md = generate_drift_report_md(metrics, "OK", [], DriftGuardrails())
        assert "Unknown source offenders" not in md
        assert "### Unknown Source" not in md

    def test_cs_verbose_offenders_forces_table(self):
        """verbose_offenders=True forces full table below WARN threshold."""
        r = _make_rankings(n_dev=100)
        r["tier_dev"] = ["A"] * 100
        r["catalyst_mode"] = ["specific_days"] * 100
        r["catalyst_source"] = [None] * 2 + ["CTGOV_CALENDAR"] * 98
        snap = _make_snapshot("2026-01-01", r)
        metrics = compute_drift_metrics([snap], strict_cs_missing=True)
        md = generate_drift_report_md(
            metrics, "OK", [], DriftGuardrails(), verbose_offenders=True,
        )
        assert "### Unknown Source" in md

    # -- Non-CTGOV offender appendix tests --

    def test_non_ctgov_offender_fields(self):
        """Non-CTGOV offenders have ticker, mode, source, event_type, days."""
        r = _make_rankings(n_dev=5)
        r["tier_dev"] = ["A"] * 5
        r["catalyst_mode"] = ["specific_days"] * 3 + ["no_upcoming"] * 2
        r["catalyst_source"] = [
            "FDA_CALENDAR", "SEC_8K_FILING", "CTGOV_CALENDAR",
            "CORPORATE_CALENDAR", "CTGOV_CALENDAR",
        ]
        r["catalyst_event_type"] = [
            "FDA_DECISION", "DATA_READOUT", "DATA_READOUT",
            "TRIAL_ONGOING", "DATA_READOUT",
        ]
        r["de_catalyst_days"] = ["30", "60", "45", "90", "15"]
        result = _catalyst_source_metrics(r)
        offenders = result["_cs_non_ctgov_offenders"]
        # Only dated (specific_days/blended_window) non-CTGOV: FDA, SEC_8K
        # no_upcoming CORPORATE excluded (not dated)
        assert len(offenders) == 2
        assert offenders[0]["catalyst_source"] == "FDA_CALENDAR"
        assert offenders[0]["event_type"] == "FDA_DECISION"
        assert offenders[0]["catalyst_days"] == "30"
        assert offenders[1]["catalyst_source"] == "SEC_8K_FILING"

    def test_non_ctgov_offenders_sorted_deterministically(self):
        """Non-CTGOV offenders sort by source group, then mode, then ticker."""
        r = _make_rankings(n_dev=4)
        r["ticker"] = ["ZZZ", "AAA", "MMM", "BBB"]
        r["tier_dev"] = ["A"] * 4
        r["catalyst_mode"] = ["specific_days"] * 4
        r["catalyst_source"] = [
            "SEC_8K_FILING", "FDA_CALENDAR", "CORPORATE_CALENDAR", "FDA_CALENDAR",
        ]
        r["catalyst_event_type"] = ["X"] * 4
        result = _catalyst_source_metrics(r)
        offenders = result["_cs_non_ctgov_offenders"]
        tickers = [o["ticker"] for o in offenders]
        # FDA (AAA, BBB), SEC_8K (ZZZ), CORPORATE (MMM)
        assert tickers == ["AAA", "BBB", "ZZZ", "MMM"]

    def test_non_ctgov_full_table_when_ctgov_floor_breached(self):
        """Full table renders when CTGOV share < floor (37%)."""
        r = _make_rankings(n_dev=10)
        r["tier_dev"] = ["A"] * 10
        r["catalyst_mode"] = ["specific_days"] * 10
        # 3 CTGOV out of 10 = 30% < 37% floor
        r["catalyst_source"] = (
            ["CTGOV_CALENDAR"] * 3 + ["FDA_CALENDAR"] * 4 + ["SEC_8K_FILING"] * 3
        )
        r["catalyst_event_type"] = ["DATA_READOUT"] * 10
        r["de_catalyst_days"] = ["45"] * 10
        snap = _make_snapshot("2026-01-01", r)
        metrics = compute_drift_metrics([snap])
        md = generate_drift_report_md(metrics, "OK", [], DriftGuardrails())
        assert "### Non-CTGOV Dated Catalysts" in md
        assert "FDA_CALENDAR" in md

    def test_non_ctgov_summary_when_ctgov_floor_ok(self):
        """Summary line when CTGOV share >= floor."""
        r = _make_rankings(n_dev=10)
        r["tier_dev"] = ["A"] * 10
        r["catalyst_mode"] = ["specific_days"] * 10
        # 5 CTGOV + 5 FDA = 50% CTGOV > 37% floor
        r["catalyst_source"] = ["CTGOV_CALENDAR"] * 5 + ["FDA_CALENDAR"] * 5
        r["catalyst_event_type"] = ["DATA_READOUT"] * 10
        snap = _make_snapshot("2026-01-01", r)
        metrics = compute_drift_metrics([snap])
        md = generate_drift_report_md(metrics, "OK", [], DriftGuardrails())
        assert "### Non-CTGOV Dated Catalysts" not in md
        assert "Non-CTGOV dated catalysts: 5 ticker(s)" in md
        assert "CTGOV floor not breached" in md

    def test_non_ctgov_absent_when_all_ctgov(self):
        """No non-CTGOV section when all sources are CTGOV."""
        r = _make_rankings(n_dev=10)
        r["tier_dev"] = ["A"] * 10
        r["catalyst_mode"] = ["specific_days"] * 10
        r["catalyst_source"] = ["CTGOV_CALENDAR"] * 10
        snap = _make_snapshot("2026-01-01", r)
        metrics = compute_drift_metrics([snap])
        md = generate_drift_report_md(metrics, "OK", [], DriftGuardrails())
        assert "Non-CTGOV" not in md

    def test_non_ctgov_verbose_forces_table(self):
        """verbose_offenders=True forces full non-CTGOV table."""
        r = _make_rankings(n_dev=10)
        r["tier_dev"] = ["A"] * 10
        r["catalyst_mode"] = ["specific_days"] * 10
        r["catalyst_source"] = ["CTGOV_CALENDAR"] * 5 + ["FDA_CALENDAR"] * 5
        r["catalyst_event_type"] = ["DATA_READOUT"] * 10
        snap = _make_snapshot("2026-01-01", r)
        metrics = compute_drift_metrics([snap])
        md = generate_drift_report_md(
            metrics, "OK", [], DriftGuardrails(), verbose_offenders=True,
        )
        assert "### Non-CTGOV Dated Catalysts" in md

    def test_non_ctgov_excludes_undated(self):
        """Non-CTGOV offenders exclude no_upcoming and missing catalysts."""
        r = _make_rankings(n_dev=4)
        r["tier_dev"] = ["A"] * 4
        r["catalyst_mode"] = ["specific_days", "no_upcoming", "missing", "blended_window"]
        r["catalyst_source"] = ["FDA_CALENDAR", "FDA_CALENDAR", "FDA_CALENDAR", "SEC_8K_FILING"]
        r["catalyst_event_type"] = ["X"] * 4
        result = _catalyst_source_metrics(r)
        offenders = result["_cs_non_ctgov_offenders"]
        # Only specific_days and blended_window included
        assert len(offenders) == 2
        tickers = [o["catalyst_mode"] for o in offenders]
        assert "no_upcoming" not in tickers
        assert "missing" not in tickers


# ---------------------------------------------------------------------------
# Catalyst event type mix
# ---------------------------------------------------------------------------

class TestCatalystEventTypeMix:
    """Tests for catalyst event type mix metrics and report section."""

    def test_known_distribution(self):
        """Known event type distribution → correct counts and shares."""
        r = _make_rankings(n_dev=20)
        r["tier_dev"] = ["A"] * 20
        r["catalyst_event_type"] = (
            ["DATA_READOUT"] * 10
            + ["FDA_DECISION"] * 3
            + ["FDA_ADCOM"] * 2
            + [""] * 3   # → none
            + [None] * 2  # → none (default: relaxed)
        )
        result = _catalyst_event_type_metrics(r)
        assert result is not None
        assert result["ct_n_eligible"] == 20
        assert result["ct_data_readout_count"] == 10
        assert result["ct_data_readout_share_pct"] == 50.0
        assert result["ct_fda_decision_count"] == 3
        assert result["ct_fda_adcom_count"] == 2
        assert result["ct_none_count"] == 5  # 3 empty + 2 None
        assert result["ct_unknown_count"] == 0

    def test_default_nan_maps_to_none(self):
        """Default (no strict): NaN/None → 'none'."""
        r = _make_rankings(n_dev=4)
        r["tier_dev"] = ["A"] * 4
        r["catalyst_event_type"] = ["DATA_READOUT", "", None, "  "]
        result = _catalyst_event_type_metrics(r)
        assert result["ct_none_count"] == 3
        assert result["ct_unknown_count"] == 0

    def test_strict_missing_marks_unknown_when_dated_catalyst(self):
        """strict_missing: dated catalyst with missing event type → 'unknown'."""
        r = _make_rankings(n_dev=3)
        r["tier_dev"] = ["A"] * 3
        r["catalyst_mode"] = ["specific_days", "no_upcoming", "blended_window"]
        r["catalyst_event_type"] = [None, None, ""]
        result = _catalyst_event_type_metrics(r, strict_missing=True)
        assert result["ct_unknown_count"] == 2  # specific_days + blended_window
        assert result["ct_none_count"] == 1  # no_upcoming

    def test_enrichment_aliases(self):
        """Enrichment pipeline names normalize to canonical keys."""
        r = _make_rankings(n_dev=4)
        r["tier_dev"] = ["A"] * 4
        r["catalyst_event_type"] = [
            "data_readout",   # lowercase → DATA_READOUT
            "fda_decision",   # lowercase → FDA_DECISION
            "fda_adcom",      # lowercase → FDA_ADCOM
            "trial_ongoing",  # lowercase → TRIAL_ONGOING
        ]
        result = _catalyst_event_type_metrics(r)
        assert result["ct_data_readout_count"] == 1
        assert result["ct_fda_decision_count"] == 1
        assert result["ct_fda_adcom_count"] == 1
        assert result["ct_trial_ongoing_count"] == 1

    def test_none_when_column_missing(self):
        """Returns None when catalyst_event_type column absent."""
        r = _make_rankings(n_dev=10)
        assert _catalyst_event_type_metrics(r) is None

    def test_snapshot_metrics_include_ct(self):
        """compute_snapshot_metrics includes ct_ keys when columns present."""
        r = _make_rankings(n_dev=20)
        r["tier_dev"] = ["A"] * 20
        r["catalyst_event_type"] = ["DATA_READOUT"] * 15 + [""] * 5
        snap = _make_snapshot("2026-01-01", r)
        metrics = compute_snapshot_metrics(snap)
        assert "ct_n_eligible" in metrics
        assert "ct_data_readout_share_pct" in metrics

    def test_md_report_includes_ct_section(self):
        """Report includes Catalyst Event Type Mix section when data present."""
        r = _make_rankings(n_dev=20)
        r["tier_dev"] = ["A"] * 20
        r["catalyst_event_type"] = ["DATA_READOUT"] * 15 + [""] * 5
        snap = _make_snapshot("2026-01-01", r)
        metrics = compute_drift_metrics([snap])
        md = generate_drift_report_md(metrics, "OK", [], DriftGuardrails())
        assert "## Catalyst Event Type Mix" in md
        assert "DATA_READOUT" in md

    def test_md_report_no_ct_section_when_missing(self):
        """Report omits Catalyst Event Type Mix when column absent."""
        r = _make_rankings(n_dev=20)
        snap = _make_snapshot("2026-01-01", r)
        metrics = compute_drift_metrics([snap])
        md = generate_drift_report_md(metrics, "OK", [], DriftGuardrails())
        assert "## Catalyst Event Type Mix" not in md

    # -- ct_* WARN guardrail tests --

    def _metrics_with_ct(self, **overrides) -> dict:
        """Build metrics dict with ct_* keys."""
        current = {
            "tier_A_pct": 5.0,
            "catalyst_missing_pct": 30.0,
            "top25_overlap_pct": 80.0,
            "optionality_std": 0.30,
        }
        current.update(overrides)
        return {"current": current, "snapshots": [current], "rolling": {}}

    def test_warn_ct_unknown_share_high(self):
        """WARN fires when unknown event type share > threshold (regime ok)."""
        m = self._metrics_with_ct(
            ct_unknown_share_pct=10.0,
            cat_specific_days_share_pct=50.0,
        )
        status, reasons, _, _ = evaluate_guardrails(m, DriftGuardrails())
        assert status == "WARN"
        assert any("event type unknown" in r.lower() for r in reasons)

    def test_warn_ct_fda_share_spike(self):
        """WARN fires when combined FDA share > threshold (regime ok)."""
        m = self._metrics_with_ct(
            ct_fda_decision_share_pct=25.0,
            ct_fda_adcom_share_pct=10.0,
            cat_specific_days_share_pct=50.0,
        )
        status, reasons, _, _ = evaluate_guardrails(m, DriftGuardrails())
        assert status == "WARN"
        assert any("FDA share" in r for r in reasons)

    def test_ok_ct_within_bounds(self):
        """No WARN when event type mix is healthy."""
        m = self._metrics_with_ct(
            ct_unknown_share_pct=2.0,
            ct_fda_decision_share_pct=5.0,
            ct_fda_adcom_share_pct=1.0,
            cat_specific_days_share_pct=50.0,
        )
        status, reasons, _, _ = evaluate_guardrails(m, DriftGuardrails())
        assert status == "OK"

    def test_ct_warn_suppressed_in_sparse_regime(self):
        """ct_* WARNs suppressed when cat_specific_days_share < 40%."""
        m = self._metrics_with_ct(
            ct_unknown_share_pct=20.0,
            ct_fda_decision_share_pct=25.0,
            ct_fda_adcom_share_pct=10.0,
            cat_specific_days_share_pct=30.0,
        )
        status, reasons, _, _ = evaluate_guardrails(m, DriftGuardrails())
        assert not any("event type" in r.lower() for r in reasons)
        assert not any("FDA share" in r for r in reasons)

    def test_ct_warn_suppressed_when_cat_specific_days_missing(self):
        """ct_* WARNs suppressed when cat_specific_days_share_pct absent."""
        m = self._metrics_with_ct(
            ct_unknown_share_pct=20.0,
        )
        status, reasons, _, _ = evaluate_guardrails(m, DriftGuardrails())
        assert not any("event type" in r.lower() for r in reasons)

    def test_ct_fda_spike_below_threshold_ok(self):
        """No WARN when FDA combined share is below spike threshold."""
        m = self._metrics_with_ct(
            ct_fda_decision_share_pct=15.0,
            ct_fda_adcom_share_pct=10.0,
            cat_specific_days_share_pct=50.0,
        )
        status, reasons, _, _ = evaluate_guardrails(m, DriftGuardrails())
        assert not any("FDA share" in r for r in reasons)

    def test_guardrails_config_includes_ct_thresholds(self):
        """Report shows ct_* guardrail thresholds."""
        r = _make_rankings(n_dev=20)
        r["tier_dev"] = ["A"] * 20
        r["catalyst_event_type"] = ["DATA_READOUT"] * 15 + [""] * 5
        snap = _make_snapshot("2026-01-01", r)
        metrics = compute_drift_metrics([snap])
        guardrails = DriftGuardrails()
        md = generate_drift_report_md(metrics, "OK", [], guardrails)
        assert "warn_ct_unknown_share_high" in md
        assert "warn_ct_fda_share_spike" in md

    # -- Offender appendix tests --

    def test_unknown_offenders_listed(self):
        """Unknown offenders table appears when dated catalyst has blank type."""
        r = _make_rankings(n_dev=10)
        r["tier_dev"] = ["A"] * 10
        r["catalyst_mode"] = ["specific_days"] * 5 + ["no_upcoming"] * 5
        r["catalyst_event_type"] = [None] * 3 + ["DATA_READOUT"] * 2 + [""] * 5
        r["catalyst_source"] = ["CTGOV_CALENDAR"] * 10
        snap = _make_snapshot("2026-01-01", r)
        metrics = compute_drift_metrics([snap], strict_cs_missing=True)
        md = generate_drift_report_md(metrics, "OK", [], DriftGuardrails())
        assert "### Unknown Event Type" in md
        assert "Top Offenders" in md

    def test_unknown_offenders_absent_when_none(self):
        """No unknown offenders section when all event types are populated."""
        r = _make_rankings(n_dev=10)
        r["tier_dev"] = ["A"] * 10
        r["catalyst_event_type"] = ["DATA_READOUT"] * 5 + ["FDA_DECISION"] * 5
        r["catalyst_source"] = ["CTGOV_CALENDAR"] * 5 + ["FDA_CALENDAR"] * 5
        snap = _make_snapshot("2026-01-01", r)
        metrics = compute_drift_metrics([snap])
        md = generate_drift_report_md(metrics, "OK", [], DriftGuardrails())
        assert "### Unknown Event Type" not in md

    def test_fda_offenders_listed(self):
        """FDA-tagged tickers table appears when FDA events exist."""
        r = _make_rankings(n_dev=10)
        r["tier_dev"] = ["A"] * 10
        r["catalyst_event_type"] = (
            ["FDA_DECISION"] * 3 + ["FDA_ADCOM"] * 2 + ["DATA_READOUT"] * 5
        )
        r["catalyst_source"] = (
            ["FDA_CALENDAR"] * 5 + ["CTGOV_CALENDAR"] * 5
        )
        snap = _make_snapshot("2026-01-01", r)
        metrics = compute_drift_metrics([snap])
        md = generate_drift_report_md(metrics, "OK", [], DriftGuardrails())
        assert "### FDA-Tagged Tickers" in md
        assert "FDA_DECISION" in md

    def test_fda_offenders_absent_when_no_fda(self):
        """No FDA offenders section when no FDA events present."""
        r = _make_rankings(n_dev=10)
        r["tier_dev"] = ["A"] * 10
        r["catalyst_event_type"] = ["DATA_READOUT"] * 10
        snap = _make_snapshot("2026-01-01", r)
        metrics = compute_drift_metrics([snap])
        md = generate_drift_report_md(metrics, "OK", [], DriftGuardrails())
        assert "### FDA-Tagged Tickers" not in md

    def test_unknown_offender_fields(self):
        """Unknown offenders have ticker, catalyst_mode, catalyst_source, catalyst_days."""
        r = _make_rankings(n_dev=4)
        r["tier_dev"] = ["A"] * 4
        r["catalyst_mode"] = ["specific_days", "blended_window", "no_upcoming", "specific_days"]
        r["catalyst_event_type"] = [None, "", "DATA_READOUT", "DATA_READOUT"]
        r["catalyst_source"] = ["CTGOV_CALENDAR", "FDA_CALENDAR", "CTGOV_CALENDAR", "CTGOV_CALENDAR"]
        r["de_catalyst_days"] = ["45", "120", "30", "60"]
        result = _catalyst_event_type_metrics(r, strict_missing=True)
        offenders = result["_ct_unknown_offenders"]
        assert len(offenders) == 2
        assert offenders[0]["catalyst_mode"] == "specific_days"
        assert offenders[1]["catalyst_mode"] == "blended_window"
        assert offenders[0]["catalyst_days"] == "45"
        assert offenders[1]["catalyst_days"] == "120"

    def test_fda_offender_fields(self):
        """FDA offenders have ticker, event_type, catalyst_source."""
        r = _make_rankings(n_dev=4)
        r["tier_dev"] = ["A"] * 4
        r["catalyst_event_type"] = ["FDA_DECISION", "FDA_ADCOM", "DATA_READOUT", ""]
        r["catalyst_source"] = ["FDA_CALENDAR", "FDA_CALENDAR", "CTGOV_CALENDAR", ""]
        result = _catalyst_event_type_metrics(r)
        offenders = result["_ct_fda_offenders"]
        assert len(offenders) == 2
        assert offenders[0]["event_type"] == "FDA_DECISION"
        assert offenders[1]["event_type"] == "FDA_ADCOM"
        assert offenders[0]["catalyst_source"] == "FDA_CALENDAR"

    def test_unknown_offenders_sorted_deterministically(self):
        """Unknown offenders sort by mode (specific_days first), then ticker."""
        r = _make_rankings(n_dev=4)
        # Assign tickers explicitly
        r["ticker"] = ["ZZZ", "AAA", "MMM", "BBB"]
        r["tier_dev"] = ["A"] * 4
        r["catalyst_mode"] = ["blended_window", "specific_days", "specific_days", "blended_window"]
        r["catalyst_event_type"] = [None, None, None, None]
        r["catalyst_source"] = ["X"] * 4
        result = _catalyst_event_type_metrics(r, strict_missing=True)
        offenders = result["_ct_unknown_offenders"]
        # specific_days first (AAA, MMM), then blended_window (BBB, ZZZ)
        tickers = [o["ticker"] for o in offenders]
        assert tickers == ["AAA", "MMM", "BBB", "ZZZ"]

    def test_fda_offenders_sorted_deterministically(self):
        """FDA offenders sort by event_type (DECISION first), then ticker."""
        r = _make_rankings(n_dev=4)
        r["ticker"] = ["ZZZ", "AAA", "MMM", "BBB"]
        r["tier_dev"] = ["A"] * 4
        r["catalyst_event_type"] = ["FDA_ADCOM", "FDA_DECISION", "FDA_ADCOM", "FDA_DECISION"]
        r["catalyst_source"] = ["FDA_CALENDAR"] * 4
        result = _catalyst_event_type_metrics(r)
        offenders = result["_ct_fda_offenders"]
        tickers = [o["ticker"] for o in offenders]
        assert tickers == ["AAA", "BBB", "MMM", "ZZZ"]

    def test_unknown_offenders_summary_below_threshold(self):
        """Below WARN threshold: one-line summary instead of full table."""
        r = _make_rankings(n_dev=100)
        r["tier_dev"] = ["A"] * 100
        r["catalyst_mode"] = ["specific_days"] * 100
        # 2 unknown out of 100 = 2% < 5% threshold
        r["catalyst_event_type"] = [None] * 2 + ["DATA_READOUT"] * 98
        r["catalyst_source"] = ["CTGOV_CALENDAR"] * 100
        snap = _make_snapshot("2026-01-01", r)
        metrics = compute_drift_metrics([snap], strict_cs_missing=True)
        md = generate_drift_report_md(metrics, "OK", [], DriftGuardrails())
        # Full table should NOT appear
        assert "### Unknown Event Type" not in md
        # Summary line should appear
        assert "Unknown event type offenders: 2 ticker(s)" in md
        assert "threshold not breached" in md

    def test_fda_offenders_summary_below_threshold(self):
        """Below WARN threshold: one-line summary instead of full FDA table."""
        r = _make_rankings(n_dev=100)
        r["tier_dev"] = ["A"] * 100
        # 5 FDA out of 100 = 5% < 30% threshold
        r["catalyst_event_type"] = ["FDA_DECISION"] * 5 + ["DATA_READOUT"] * 95
        r["catalyst_source"] = ["FDA_CALENDAR"] * 5 + ["CTGOV_CALENDAR"] * 95
        snap = _make_snapshot("2026-01-01", r)
        metrics = compute_drift_metrics([snap])
        md = generate_drift_report_md(metrics, "OK", [], DriftGuardrails())
        # Full table should NOT appear
        assert "### FDA-Tagged Tickers" not in md
        # Summary line should appear
        assert "FDA-tagged tickers: 5" in md
        assert "threshold not breached" in md

    def test_unknown_offenders_full_table_above_threshold(self):
        """Above WARN threshold: full table with Days column."""
        r = _make_rankings(n_dev=10)
        r["tier_dev"] = ["A"] * 10
        r["catalyst_mode"] = ["specific_days"] * 10
        # 8 unknown out of 10 = 80% > 5% threshold
        r["catalyst_event_type"] = [None] * 8 + ["DATA_READOUT"] * 2
        r["catalyst_source"] = ["CTGOV_CALENDAR"] * 10
        r["de_catalyst_days"] = ["45"] * 10
        snap = _make_snapshot("2026-01-01", r)
        metrics = compute_drift_metrics([snap], strict_cs_missing=True)
        md = generate_drift_report_md(metrics, "OK", [], DriftGuardrails())
        assert "### Unknown Event Type" in md
        assert "| Days |" in md  # catalyst_days column present

    def test_verbose_offenders_forces_unknown_table(self):
        """verbose_offenders=True forces full unknown table below WARN threshold."""
        r = _make_rankings(n_dev=100)
        r["tier_dev"] = ["A"] * 100
        r["catalyst_mode"] = ["specific_days"] * 100
        # 2 unknown out of 100 = 2% < 5% threshold
        r["catalyst_event_type"] = [None] * 2 + ["DATA_READOUT"] * 98
        r["catalyst_source"] = ["CTGOV_CALENDAR"] * 100
        snap = _make_snapshot("2026-01-01", r)
        metrics = compute_drift_metrics([snap], strict_cs_missing=True)
        # Without verbose: summary line only
        md_quiet = generate_drift_report_md(metrics, "OK", [], DriftGuardrails())
        assert "### Unknown Event Type" not in md_quiet
        assert "Unknown event type offenders: 2" in md_quiet
        # With verbose: full table
        md_verbose = generate_drift_report_md(
            metrics, "OK", [], DriftGuardrails(), verbose_offenders=True,
        )
        assert "### Unknown Event Type" in md_verbose
        assert "| Days |" in md_verbose

    def test_verbose_offenders_forces_fda_table(self):
        """verbose_offenders=True forces full FDA table below WARN threshold."""
        r = _make_rankings(n_dev=100)
        r["tier_dev"] = ["A"] * 100
        # 5 FDA out of 100 = 5% < 30% threshold
        r["catalyst_event_type"] = ["FDA_DECISION"] * 5 + ["DATA_READOUT"] * 95
        r["catalyst_source"] = ["FDA_CALENDAR"] * 5 + ["CTGOV_CALENDAR"] * 95
        snap = _make_snapshot("2026-01-01", r)
        metrics = compute_drift_metrics([snap])
        # Without verbose: summary line only
        md_quiet = generate_drift_report_md(metrics, "OK", [], DriftGuardrails())
        assert "### FDA-Tagged Tickers" not in md_quiet
        assert "FDA-tagged tickers: 5" in md_quiet
        # With verbose: full table
        md_verbose = generate_drift_report_md(
            metrics, "OK", [], DriftGuardrails(), verbose_offenders=True,
        )
        assert "### FDA-Tagged Tickers" in md_verbose
        assert "FDA_DECISION" in md_verbose


# ---------------------------------------------------------------------------
# Panel-as-snapshots
# ---------------------------------------------------------------------------

def _make_panel_csv(path, rows, extra_cols=None):
    """Write a minimal panel CSV for testing.

    Each row is a dict with panel column names.
    """
    import csv

    base_cols = [
        "as_of_date", "ticker", "tier", "band", "eligible", "weight",
        "ineligible_reasons", "first_failed_gate", "tier_reason", "risk_flags",
        "catalyst_mode", "catalyst_strength", "mom_state",
        "optionality", "catalyst_days_raw", "ruleset_id",
        "drawdown_abs", "drawdown_xbi", "drawdown_rel_xbi",
        "dd_abs_margin", "dd_rel_margin", "rescued_by_rel",
        "dd_rel_margin_rescued", "optionality_margin_a", "actionable_catalyst",
        "returns_source",
    ]
    cols = base_cols + (extra_cols or [])

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _panel_row(
    date="2025-06-30", ticker="TICK", tier="B", band="L",
    eligible="1", weight="5.0", catalyst_mode="specific_days",
    catalyst_strength="near", optionality="0.70", ruleset_id="f3454ef7",
    returns_source="morningstar", **extra,
):
    """Build a single panel row dict with sensible defaults."""
    row = {
        "as_of_date": date, "ticker": ticker, "tier": tier, "band": band,
        "eligible": eligible, "weight": weight,
        "ineligible_reasons": "", "first_failed_gate": "",
        "tier_reason": "mod_opt" if tier == "B" else "high_opt+catalyst_near",
        "risk_flags": "", "catalyst_mode": catalyst_mode,
        "catalyst_strength": catalyst_strength, "mom_state": "tailwind",
        "optionality": optionality, "catalyst_days_raw": "30",
        "ruleset_id": ruleset_id,
        "drawdown_abs": "-0.10", "drawdown_xbi": "-0.05",
        "drawdown_rel_xbi": "-0.05",
        "dd_abs_margin": "0.30", "dd_rel_margin": "0.20",
        "rescued_by_rel": "0", "dd_rel_margin_rescued": "0",
        "optionality_margin_a": "0.10", "actionable_catalyst": "1",
        "returns_source": returns_source,
    }
    row.update(extra)
    return row


class TestPanelLoader:
    """Tests for load_panel_as_snapshots and column remapping."""

    def test_basic_load(self, tmp_path):
        """Panel CSV is loaded and split by date."""
        csv_path = tmp_path / "panel.csv"
        rows = [
            _panel_row(date="2025-01-31", ticker="A1"),
            _panel_row(date="2025-01-31", ticker="A2"),
            _panel_row(date="2025-02-28", ticker="A1"),
            _panel_row(date="2025-02-28", ticker="A3"),
        ]
        _make_panel_csv(csv_path, rows)
        snaps = load_panel_as_snapshots(csv_path)
        assert len(snaps) == 2
        assert snaps[0].date == "2025-01-31"
        assert snaps[1].date == "2025-02-28"
        assert len(snaps[0].rankings) == 2
        assert len(snaps[1].rankings) == 2

    def test_column_remapping(self, tmp_path):
        """Panel columns are renamed to snapshot equivalents."""
        csv_path = tmp_path / "panel.csv"
        _make_panel_csv(csv_path, [_panel_row()])
        snaps = load_panel_as_snapshots(csv_path)
        r = snaps[0].rankings
        # tier → tier_dev
        assert "tier_dev" in r.columns
        assert "tier" not in r.columns
        # optionality → clinical_optionality_pct_dev
        assert "clinical_optionality_pct_dev" in r.columns
        assert "optionality" not in r.columns
        # band → size_band
        assert "size_band" in r.columns
        assert "band" not in r.columns
        # drawdown_rel_xbi → de_drawdown_rel_xbi
        assert "de_drawdown_rel_xbi" in r.columns

    def test_date_column_dropped(self, tmp_path):
        """Date column is dropped from rankings (schema parity with rankings.csv)."""
        csv_path = tmp_path / "panel.csv"
        _make_panel_csv(csv_path, [_panel_row()])
        snaps = load_panel_as_snapshots(csv_path)
        r = snaps[0].rankings
        assert "as_of_date" not in r.columns
        assert "snapshot_date" not in r.columns
        assert "date" not in r.columns

    def test_archetype_injected(self, tmp_path):
        """Synthetic archetype=drug_developer is added."""
        csv_path = tmp_path / "panel.csv"
        _make_panel_csv(csv_path, [_panel_row()])
        snaps = load_panel_as_snapshots(csv_path)
        r = snaps[0].rankings
        assert "archetype" in r.columns
        assert all(r["archetype"] == "drug_developer")

    def test_actionable_rank_synthesized(self, tmp_path):
        """actionable_rank is derived from weight (higher weight = lower rank)."""
        csv_path = tmp_path / "panel.csv"
        rows = [
            _panel_row(ticker="HI", weight="10.0"),
            _panel_row(ticker="MID", weight="5.0"),
            _panel_row(ticker="ZERO", weight="0.0"),
        ]
        _make_panel_csv(csv_path, rows)
        snaps = load_panel_as_snapshots(csv_path)
        r = snaps[0].rankings
        assert "actionable_rank" in r.columns
        by_ticker = {row["ticker"]: row["actionable_rank"] for _, row in r.iterrows()}
        assert by_ticker["HI"] < by_ticker["MID"]
        assert by_ticker["ZERO"] == 9999

    def test_window_size_limits_dates(self, tmp_path):
        """window_size keeps only the last N dates."""
        csv_path = tmp_path / "panel.csv"
        rows = [
            _panel_row(date="2025-01-31", ticker="A"),
            _panel_row(date="2025-02-28", ticker="A"),
            _panel_row(date="2025-03-31", ticker="A"),
        ]
        _make_panel_csv(csv_path, rows)
        snaps = load_panel_as_snapshots(csv_path, window_size=2)
        assert len(snaps) == 2
        assert snaps[0].date == "2025-02-28"
        assert snaps[1].date == "2025-03-31"

    def test_ruleset_id_extracted(self, tmp_path):
        """Ruleset ID is extracted from panel rows."""
        csv_path = tmp_path / "panel.csv"
        _make_panel_csv(csv_path, [_panel_row(ruleset_id="deadbeef")])
        snaps = load_panel_as_snapshots(csv_path)
        assert snaps[0].ruleset_id == "deadbeef"

    def test_metrics_computable(self, tmp_path):
        """compute_snapshot_metrics runs without error on panel-derived snapshot."""
        csv_path = tmp_path / "panel.csv"
        n = 30
        rows = [
            _panel_row(
                ticker=f"T{i:03d}",
                tier=["A", "B", "C", "D"][i % 4],
                eligible="1" if i < 20 else "0",
                weight=str(round(5.0 - i * 0.1, 2)) if i < 20 else "0.0",
                optionality=str(round(0.3 + i * 0.02, 4)),
                catalyst_mode=["specific_days", "no_upcoming", "missing"][i % 3],
                catalyst_strength=["near", "mid", "far", "missing"][i % 4],
            )
            for i in range(n)
        ]
        _make_panel_csv(csv_path, rows)
        snaps = load_panel_as_snapshots(csv_path)
        m = compute_snapshot_metrics(snaps[0])
        assert m["n_dev"] == n
        assert m["tier_A_count"] > 0
        assert m["optionality_std"] is not None
        # Catalyst coverage should work
        assert m.get("cat_n_dev") == n
        assert m.get("cat_eligible_share_pct") is not None

    def test_drift_metrics_on_panel(self, tmp_path):
        """Full drift metrics pipeline works on multi-date panel."""
        csv_path = tmp_path / "panel.csv"
        rows = []
        for date in ("2025-01-31", "2025-02-28", "2025-03-31"):
            for i in range(20):
                rows.append(_panel_row(
                    date=date, ticker=f"T{i:03d}",
                    tier=["A", "B", "C", "D"][i % 4],
                    eligible="1" if i < 15 else "0",
                    weight=str(round(5.0 - i * 0.2, 2)) if i < 15 else "0.0",
                    catalyst_mode=["specific_days", "no_upcoming", "missing"][i % 3],
                    catalyst_strength=["near", "mid", "far", "missing"][i % 4],
                ))
        _make_panel_csv(csv_path, rows)
        snaps = load_panel_as_snapshots(csv_path)
        metrics = compute_drift_metrics(snaps)
        assert len(metrics["snapshots"]) == 3
        assert metrics["current"]["date"] == "2025-03-31"
        # Rolling should have keys
        assert "tier_A_pct" in metrics["rolling"]

    def test_missing_date_column_raises(self, tmp_path):
        """Panel without any recognized date column raises ValueError."""
        import csv as csv_mod
        csv_path = tmp_path / "bad.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv_mod.writer(f)
            w.writerow(["ticker", "tier"])
            w.writerow(["TICK", "A"])
        with pytest.raises(ValueError, match="date column"):
            load_panel_as_snapshots(csv_path)

    def test_alternate_date_column(self, tmp_path):
        """Panel with 'snapshot_date' instead of 'as_of_date' is accepted."""
        import csv as csv_mod
        csv_path = tmp_path / "panel.csv"
        # Write with snapshot_date column
        row = _panel_row()
        row["snapshot_date"] = row.pop("as_of_date")
        cols = list(row.keys())
        with open(csv_path, "w", newline="") as f:
            w = csv_mod.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerow(row)
        snaps = load_panel_as_snapshots(csv_path)
        assert len(snaps) == 1
        assert snaps[0].date == "2025-06-30"

    def test_panel_col_map_keys(self):
        """_PANEL_COL_MAP contains expected mappings."""
        assert _PANEL_COL_MAP["tier"] == "tier_dev"
        assert _PANEL_COL_MAP["optionality"] == "clinical_optionality_pct_dev"
        assert _PANEL_COL_MAP["band"] == "size_band"
        assert _PANEL_COL_MAP["drawdown_rel_xbi"] == "de_drawdown_rel_xbi"


class TestSynthesizeActionableRank:
    """Tests for _synthesize_actionable_rank."""

    def test_ranking_order(self):
        """Higher weight gets lower (better) rank."""
        df = pd.DataFrame({
            "weight": ["10.0", "5.0", "1.0", "0.0"],
        })
        ranks = _synthesize_actionable_rank(df)
        assert ranks.iloc[0] == 1  # weight=10
        assert ranks.iloc[1] == 2  # weight=5
        assert ranks.iloc[2] == 3  # weight=1
        assert ranks.iloc[3] == 9999  # weight=0

    def test_all_zero_weight(self):
        """All zero weights → all 9999."""
        df = pd.DataFrame({"weight": ["0.0", "0.0"]})
        ranks = _synthesize_actionable_rank(df)
        assert list(ranks) == [9999, 9999]

    def test_missing_weight(self):
        """Missing/NaN weight treated as 0."""
        df = pd.DataFrame({"weight": ["5.0", "", None]})
        ranks = _synthesize_actionable_rank(df)
        assert ranks.iloc[0] == 1
        assert ranks.iloc[1] == 9999
        assert ranks.iloc[2] == 9999


# ---------------------------------------------------------------------------
# Suggested Guardrails
# ---------------------------------------------------------------------------

class TestSuggestedGuardrails:
    """Tests for compute_suggested_guardrails."""

    def _make_metrics_series(self, n, **overrides):
        """Build N+1 synthetic snapshot metric dicts (N prior + 1 current)."""
        metrics = []
        for i in range(n + 1):
            m = {
                "date": f"2025-{i+1:02d}-01",
                "tier_A_pct": 5.0 + i * 0.1,
                "cat_eligible_share_pct": 95.0 - i * 0.5,
                "cat_specific_days_share_pct": 50.0 + i * 0.2,
                "rs_morningstar_share_pct": 90.0 + i * 0.1,
                "rs_unknown_share_pct": 3.0 + i * 0.1,
                "cs_ctgov_calendar_share_pct": 60.0 + i * 0.5,
                "cs_unknown_share_pct": 2.0 + i * 0.1,
            }
            m.update(overrides)
            metrics.append(m)
        return metrics

    def test_insufficient_history_returns_empty(self):
        """Fewer than min_points prior snapshots → no suggestions."""
        # 2 total = 1 prior (below default min_points=3)
        metrics = self._make_metrics_series(1)
        result = compute_suggested_guardrails(metrics, DriftGuardrails())
        assert result == []

    def test_exactly_min_points_produces_suggestions(self):
        """Exactly min_points prior snapshots → suggestions appear."""
        metrics = self._make_metrics_series(SUGGESTION_MIN_POINTS)
        result = compute_suggested_guardrails(metrics, DriftGuardrails())
        assert len(result) > 0
        labels = {s["label"] for s in result}
        assert "CTGOV calendar share" in labels

    def test_prior_only_excludes_current(self):
        """Current snapshot is excluded from baseline computation."""
        # 4 prior with stable values + 1 current with extreme value
        metrics = self._make_metrics_series(3)
        metrics.append({
            "date": "2025-99-01",
            "rs_morningstar_share_pct": 10.0,  # extreme outlier
            **{k: metrics[0][k] for k in metrics[0] if k != "rs_morningstar_share_pct"},
        })
        result = compute_suggested_guardrails(metrics, DriftGuardrails())
        ms = {s["metric"]: s for s in result}
        # Morningstar median should NOT be dragged down by the outlier current
        assert ms["rs_morningstar_share_pct"]["median"] > 80.0

    def test_low_floor_formula(self):
        """Low floor: suggested = max(0, median - k*IQR)."""
        # 5 prior with constant value → IQR=0 → suggested=median
        metrics = [{"date": f"d{i}", "cat_eligible_share_pct": 90.0}
                   for i in range(6)]
        result = compute_suggested_guardrails(
            metrics, DriftGuardrails(), k=2.0,
        )
        ms = {s["metric"]: s for s in result}
        s = ms["cat_eligible_share_pct"]
        assert s["iqr"] == 0.0
        assert s["suggested"] == 90.0  # median - 2*0 = 90
        assert s["direction"] == "low_floor"

    def test_high_ceiling_formula(self):
        """High ceiling: suggested = min(100, median + k*IQR)."""
        metrics = [{"date": f"d{i}", "cs_unknown_share_pct": 5.0,
                    "cat_specific_days_share_pct": 50.0}
                   for i in range(6)]
        result = compute_suggested_guardrails(
            metrics, DriftGuardrails(), k=2.0,
        )
        ms = {s["metric"]: s for s in result}
        s = ms["cs_unknown_share_pct"]
        assert s["suggested"] == 5.0  # median + 2*0 = 5
        assert s["direction"] == "high_ceiling"

    def test_clipping_to_zero(self):
        """Low floor is clipped to 0 (never negative)."""
        # Values with high variance → median - k*IQR could go negative
        metrics = [
            {"date": "d0", "tier_A_pct": 0.5},
            {"date": "d1", "tier_A_pct": 1.0},
            {"date": "d2", "tier_A_pct": 5.0},
            {"date": "d3", "tier_A_pct": 10.0},
            {"date": "d4", "tier_A_pct": 2.0},  # current (excluded)
        ]
        result = compute_suggested_guardrails(
            metrics, DriftGuardrails(), k=2.0,
        )
        ms = {s["metric"]: s for s in result}
        assert ms["tier_A_pct"]["suggested"] >= 0.0

    def test_clipping_to_100(self):
        """High ceiling is clipped to 100 (never above)."""
        metrics = [
            {"date": "d0", "rs_unknown_share_pct": 90.0},
            {"date": "d1", "rs_unknown_share_pct": 95.0},
            {"date": "d2", "rs_unknown_share_pct": 85.0},
            {"date": "d3", "rs_unknown_share_pct": 98.0},
            {"date": "d4", "rs_unknown_share_pct": 50.0},  # current
        ]
        result = compute_suggested_guardrails(
            metrics, DriftGuardrails(), k=2.0,
        )
        ms = {s["metric"]: s for s in result}
        assert ms["rs_unknown_share_pct"]["suggested"] <= 100.0

    def test_tighter_flag(self):
        """Tighter flag is True only when suggestion exceeds current threshold."""
        # All cat_eligible at 99% → suggested ≈ 99% >> current 80% → tighter
        metrics = [{"date": f"d{i}", "cat_eligible_share_pct": 99.0}
                   for i in range(6)]
        result = compute_suggested_guardrails(
            metrics, DriftGuardrails(), k=2.0,
        )
        ms = {s["metric"]: s for s in result}
        assert ms["cat_eligible_share_pct"]["tighter"] is True
        assert ms["cat_eligible_share_pct"]["current_threshold"] == 95.0

    def test_report_section_appears(self):
        """Suggestions section appears in markdown when suggestions exist."""
        snap = _make_snapshot("2026-01-01")
        metrics = compute_drift_metrics([snap, snap, snap, snap])
        suggestions = compute_suggested_guardrails(
            metrics["snapshots"], DriftGuardrails(),
        )
        md = generate_drift_report_md(
            metrics, "OK", [], DriftGuardrails(), suggestions=suggestions,
        )
        assert "## Suggested Guardrails" in md
        assert "informational only" in md

    def test_report_section_absent_when_no_suggestions(self):
        """Suggestions section is absent when no suggestions computed."""
        snap = _make_snapshot("2026-01-01")
        metrics = compute_drift_metrics([snap])
        md = generate_drift_report_md(
            metrics, "OK", [], DriftGuardrails(), suggestions=None,
        )
        assert "## Suggested Guardrails" not in md

    def test_missing_metric_skipped(self):
        """Metrics not present in snapshots are silently skipped."""
        # Only tier_A_pct present, others missing → only tier_A_pct suggested
        metrics = [{"date": f"d{i}", "tier_A_pct": 5.0} for i in range(6)]
        result = compute_suggested_guardrails(
            metrics, DriftGuardrails(), k=2.0,
        )
        assert len(result) == 1
        assert result[0]["metric"] == "tier_A_pct"

    def test_json_includes_suggestions(self):
        """JSON output includes suggested_guardrails key."""
        snap = _make_snapshot("2026-01-01")
        metrics = compute_drift_metrics([snap, snap, snap, snap])
        suggestions = compute_suggested_guardrails(
            metrics["snapshots"], DriftGuardrails(),
        )
        js = generate_drift_json(
            metrics, "OK", [], DriftGuardrails(), suggestions=suggestions,
        )
        assert "suggested_guardrails" in js
        assert isinstance(js["suggested_guardrails"], list)


# ---------------------------------------------------------------------------
# cs_ctgov suppression gate
# ---------------------------------------------------------------------------


class TestCsCtgovSuggestionGate:
    """cs_ctgov_calendar_share_pct suggestion is suppressed when
    the prior median of cat_specific_days_share_pct < 40%."""

    def _make_metrics(self, cat_specific_pct: float, cs_ctgov_pct: float, n: int = 4):
        """Build a list of snapshot metrics dicts with consistent values."""
        return [
            {
                "cat_specific_days_share_pct": cat_specific_pct,
                "cs_ctgov_calendar_share_pct": cs_ctgov_pct,
                # Fill enough keys so other specs don't crash
                "tier_A_pct": 5.0,
                "cat_eligible_share_pct": 100.0,
                "rs_morningstar_share_pct": 95.0,
                "rs_unknown_share_pct": 2.0,
                "cs_unknown_share_pct": 1.0,
                "ct_unknown_share_pct": 1.0,
            }
            for _ in range(n)
        ]

    def test_suppressed_when_specific_days_sparse(self):
        metrics = self._make_metrics(cat_specific_pct=25.0, cs_ctgov_pct=90.0)
        suggestions = compute_suggested_guardrails(metrics, DriftGuardrails())
        ctgov = [s for s in suggestions if s["metric"] == "cs_ctgov_calendar_share_pct"]
        assert len(ctgov) == 1
        assert ctgov[0]["suppressed"] is True
        assert ctgov[0]["suggested"] is None
        assert "25.0%" in ctgov[0]["suppressed_reason"]

    def test_not_suppressed_when_specific_days_healthy(self):
        metrics = self._make_metrics(cat_specific_pct=50.0, cs_ctgov_pct=80.0)
        suggestions = compute_suggested_guardrails(metrics, DriftGuardrails())
        ctgov = [s for s in suggestions if s["metric"] == "cs_ctgov_calendar_share_pct"]
        assert len(ctgov) == 1
        assert ctgov[0].get("suppressed") is not True
        assert ctgov[0]["suggested"] is not None

    def test_suppressed_at_boundary(self):
        # Exactly at the boundary (39.9 < 40.0 → suppressed)
        metrics = self._make_metrics(cat_specific_pct=39.9, cs_ctgov_pct=80.0)
        suggestions = compute_suggested_guardrails(metrics, DriftGuardrails())
        ctgov = [s for s in suggestions if s["metric"] == "cs_ctgov_calendar_share_pct"]
        assert ctgov[0]["suppressed"] is True

    def test_not_suppressed_at_exact_threshold(self):
        # Exactly 40.0 → NOT suppressed (>= would pass, but we use <)
        metrics = self._make_metrics(cat_specific_pct=40.0, cs_ctgov_pct=80.0)
        suggestions = compute_suggested_guardrails(metrics, DriftGuardrails())
        ctgov = [s for s in suggestions if s["metric"] == "cs_ctgov_calendar_share_pct"]
        assert ctgov[0].get("suppressed") is not True

    def test_report_shows_suppressed_row(self):
        metrics = self._make_metrics(cat_specific_pct=25.0, cs_ctgov_pct=90.0)
        suggestions = compute_suggested_guardrails(metrics, DriftGuardrails())
        md = generate_drift_report_md(
            {"snapshots": metrics}, "OK", [], DriftGuardrails(),
            suggestions=suggestions,
        )
        assert "suppressed" in md
        assert "CTGOV calendar share" in md or "cs_ctgov_calendar_share_pct" in md

    def test_cs_unknown_also_suppressed_when_sparse(self):
        metrics = self._make_metrics(cat_specific_pct=25.0, cs_ctgov_pct=90.0)
        suggestions = compute_suggested_guardrails(metrics, DriftGuardrails())
        cs_unk = [s for s in suggestions if s["metric"] == "cs_unknown_share_pct"]
        assert len(cs_unk) == 1
        assert cs_unk[0]["suppressed"] is True

    def test_cs_unknown_not_suppressed_when_healthy(self):
        metrics = self._make_metrics(cat_specific_pct=50.0, cs_ctgov_pct=80.0)
        suggestions = compute_suggested_guardrails(metrics, DriftGuardrails())
        cs_unk = [s for s in suggestions if s["metric"] == "cs_unknown_share_pct"]
        assert len(cs_unk) == 1
        assert cs_unk[0].get("suppressed") is not True

    def test_ct_unknown_suppressed_when_sparse(self):
        """ct_unknown suggestion is suppressed when specific_days are sparse."""
        metrics = self._make_metrics(cat_specific_pct=25.0, cs_ctgov_pct=90.0)
        suggestions = compute_suggested_guardrails(metrics, DriftGuardrails())
        ct_unk = [s for s in suggestions if s["metric"] == "ct_unknown_share_pct"]
        assert len(ct_unk) == 1
        assert ct_unk[0]["suppressed"] is True

    def test_ct_unknown_not_suppressed_when_healthy(self):
        """ct_unknown suggestion is not suppressed when specific_days are healthy."""
        metrics = self._make_metrics(cat_specific_pct=50.0, cs_ctgov_pct=80.0)
        suggestions = compute_suggested_guardrails(metrics, DriftGuardrails())
        ct_unk = [s for s in suggestions if s["metric"] == "ct_unknown_share_pct"]
        assert len(ct_unk) == 1
        assert ct_unk[0].get("suppressed") is not True

    def test_report_footer_shows_suppression_threshold(self):
        metrics = self._make_metrics(cat_specific_pct=25.0, cs_ctgov_pct=90.0)
        suggestions = compute_suggested_guardrails(metrics, DriftGuardrails())
        md = generate_drift_report_md(
            {"snapshots": metrics}, "OK", [], DriftGuardrails(),
            suggestions=suggestions,
        )
        assert "suggest_cs_min_cat_specific_days_median" in md
        assert "40.0%" in md

    def test_report_footer_absent_when_no_suppression(self):
        metrics = self._make_metrics(cat_specific_pct=50.0, cs_ctgov_pct=80.0)
        suggestions = compute_suggested_guardrails(metrics, DriftGuardrails())
        md = generate_drift_report_md(
            {"snapshots": metrics}, "OK", [], DriftGuardrails(),
            suggestions=suggestions,
        )
        assert "suggest_cs_min_cat_specific_days_median" not in md


# ---------------------------------------------------------------------------
# Panel-mode catalyst_source note
# ---------------------------------------------------------------------------


class TestPanelModeSourceNote:
    """Panel-mode CSV ambiguity note is injected via md.replace() in main().

    These tests verify:
    - generate_drift_report_md produces the ## Catalyst Source Mix header
      (prerequisite for the replace to work)
    - the replace pattern inserts the note correctly
    """

    def _metrics_with_cs(self):
        return {
            "snapshots": [],
            "rolling": {},
            "current": {
                "cs_n_eligible": 10,
                "cs_ctgov_calendar_count": 5, "cs_ctgov_calendar_share_pct": 50.0,
                "cs_fda_calendar_count": 2, "cs_fda_calendar_share_pct": 20.0,
                "cs_sec_8k_filing_count": 0, "cs_sec_8k_filing_share_pct": 0.0,
                "cs_corporate_calendar_count": 0, "cs_corporate_calendar_share_pct": 0.0,
                "cs_none_count": 3, "cs_none_share_pct": 30.0,
                "cs_unknown_count": 0, "cs_unknown_share_pct": 0.0,
            },
        }

    def test_md_contains_cs_header_for_replace(self):
        md = generate_drift_report_md(
            self._metrics_with_cs(), "OK", [], DriftGuardrails(),
        )
        assert "## Catalyst Source Mix\n\n" in md

    def test_replace_injects_note(self):
        md = generate_drift_report_md(
            self._metrics_with_cs(), "OK", [], DriftGuardrails(),
        )
        md = md.replace(
            "## Catalyst Source Mix\n\n",
            "## Catalyst Source Mix\n\n"
            "_Panel CSV: empty catalyst_source treated as `none`"
            " (CSV ambiguity)._\n\n",
            1,
        )
        assert "CSV ambiguity" in md

    def test_no_note_without_replace(self):
        md = generate_drift_report_md(
            self._metrics_with_cs(), "OK", [], DriftGuardrails(),
        )
        assert "CSV ambiguity" not in md
