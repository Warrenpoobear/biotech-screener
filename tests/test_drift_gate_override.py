"""Tests for the dated operator override on the blocking drift gate."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "scripts"))

from run_daily_production import DriftThresholds, check_drift_monitoring  # noqa: E402

from common.drift_override import (  # noqa: E402
    DEFAULT_OVERRIDES_PATH,
    fail_reason_metric,
    find_applicable_override,
    load_drift_overrides,
)

HEADER = ["ticker", "actionable_rank", "tier_dev", "eligible", "composite_rank", "composite_score"]

ENTRY = {
    "id": "test-order-switch",
    "prior_date": "2026-11-12",
    "valid_from": "2026-11-16",
    "valid_through": "2026-11-20",
    "metrics": ["top20_overlap_pct", "mean_abs_rank_delta_top60"],
    "reason": "Authorized order change.",
    "authorized_by": "operator, 2026-11-15",
    "memo": "docs/governance/example.md",
}


def _write_rankings(path: Path, order: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADER, lineterminator="\n")
        writer.writeheader()
        for rank, ticker in enumerate(order, start=1):
            writer.writerow(
                {
                    "ticker": ticker,
                    "actionable_rank": str(rank),
                    "tier_dev": "A",
                    "eligible": "True",
                    "composite_rank": str(rank),
                    "composite_score": str(round(1.0 - rank / 100.0, 4)),
                }
            )


def _write_overrides(path: Path, entries: list[dict]) -> Path:
    path.write_text(json.dumps({"overrides": entries}), encoding="utf-8")
    return path


def _reordered_run(tmp_path: Path, as_of: str = "2026-11-16", prior: str = "2026-11-12"):
    """Same 60 names; the top 20 swap places with ranks 21-40 (a reorder, not a feed break)."""
    names = [f"T{i:02d}" for i in range(60)]
    snapshot_dir = tmp_path / "snapshots"
    _write_rankings(snapshot_dir / prior / "rankings.csv", names)
    staging = tmp_path / "staging" / as_of
    _write_rankings(staging / "rankings.csv", names[20:40] + names[:20] + names[40:])
    return staging, snapshot_dir


# Only the top-20 overlap floor is active, so the scenario isolates that one FAIL reason.
TOP20_ONLY = DriftThresholds(
    fail_top60_overlap_pct=-1.0,
    fail_rank_spearman_rho=-2.0,
    fail_mean_abs_rank_delta_top60=9999.0,
    fail_tier_migration_count=9999,
    fail_eligibility_change_count=9999,
)


class TestGateWithOverride:
    def test_blocks_without_override(self, tmp_path):
        staging, snapshots = _reordered_run(tmp_path)
        overrides = _write_overrides(tmp_path / "overrides.json", [])
        gate = check_drift_monitoring(staging, snapshots, "2026-11-16", TOP20_ONLY, overrides_path=overrides)
        assert gate.status == "FAIL"
        assert "BLOCKED" in gate.detail

    def test_override_downgrades_to_warn_and_is_recorded(self, tmp_path):
        staging, snapshots = _reordered_run(tmp_path)
        overrides = _write_overrides(tmp_path / "overrides.json", [ENTRY])
        gate = check_drift_monitoring(staging, snapshots, "2026-11-16", TOP20_ONLY, overrides_path=overrides)
        assert gate.status == "WARN"
        assert "OPERATOR OVERRIDE test-order-switch" in gate.detail
        assert "FAIL top20_overlap_pct" in gate.detail
        report = json.loads((staging / "drift_report.json").read_text(encoding="utf-8"))
        assert report["status"] == "WARN"
        assert report["fail_reasons"], "the failing metric must stay visible"
        assert report["override_applied"]["id"] == "test-order-switch"
        assert "Operator override applied" in (staging / "drift_report.md").read_text(encoding="utf-8")

    def test_override_does_not_apply_outside_window(self, tmp_path):
        staging, snapshots = _reordered_run(tmp_path, as_of="2026-11-23")
        overrides = _write_overrides(tmp_path / "overrides.json", [ENTRY])
        gate = check_drift_monitoring(staging, snapshots, "2026-11-23", TOP20_ONLY, overrides_path=overrides)
        assert gate.status == "FAIL"

    def test_override_lapses_when_prior_snapshot_changes(self, tmp_path):
        staging, snapshots = _reordered_run(tmp_path, as_of="2026-11-17", prior="2026-11-16")
        overrides = _write_overrides(tmp_path / "overrides.json", [ENTRY])
        gate = check_drift_monitoring(staging, snapshots, "2026-11-17", TOP20_ONLY, overrides_path=overrides)
        assert gate.status == "FAIL"

    def test_uncovered_fail_reason_still_blocks(self, tmp_path):
        """A feed break (different names) also fails top-60 overlap, which no entry may cover."""
        snapshots = tmp_path / "snapshots"
        _write_rankings(snapshots / "2026-11-12" / "rankings.csv", [f"T{i:02d}" for i in range(60)])
        staging = tmp_path / "staging" / "2026-11-16"
        _write_rankings(staging / "rankings.csv", [f"X{i:02d}" for i in range(60)])
        overrides = _write_overrides(tmp_path / "overrides.json", [ENTRY])
        gate = check_drift_monitoring(staging, snapshots, "2026-11-16", DriftThresholds(), overrides_path=overrides)
        assert gate.status == "FAIL"
        assert "top60_overlap_pct" in gate.detail

    def test_no_drift_is_unaffected_by_an_override(self, tmp_path):
        names = [f"T{i:02d}" for i in range(60)]
        snapshots = tmp_path / "snapshots"
        _write_rankings(snapshots / "2026-11-12" / "rankings.csv", names)
        staging = tmp_path / "staging" / "2026-11-16"
        _write_rankings(staging / "rankings.csv", names)
        overrides = _write_overrides(tmp_path / "overrides.json", [ENTRY])
        gate = check_drift_monitoring(staging, snapshots, "2026-11-16", DriftThresholds(), overrides_path=overrides)
        assert gate.status == "PASS"
        report = json.loads((staging / "drift_report.json").read_text(encoding="utf-8"))
        assert report["override_applied"] is None


class TestOverrideFile:
    def test_missing_or_broken_file_means_no_overrides(self, tmp_path):
        assert load_drift_overrides(tmp_path / "absent.json") == []
        broken = tmp_path / "broken.json"
        broken.write_text("{not json", encoding="utf-8")
        assert load_drift_overrides(broken) == []

    def test_rejects_non_overridable_metric(self, tmp_path):
        for metric in ("top60_overlap_pct", "rank_spearman_rho", "eligibility_change_count"):
            path = _write_overrides(tmp_path / "o.json", [{**ENTRY, "metrics": [metric]}])
            assert load_drift_overrides(path) == []

    def test_rejects_long_or_inverted_window(self, tmp_path):
        path = _write_overrides(tmp_path / "o.json", [{**ENTRY, "valid_through": "2026-12-31"}])
        assert load_drift_overrides(path) == []
        path = _write_overrides(tmp_path / "o.json", [{**ENTRY, "valid_through": "2026-11-15"}])
        assert load_drift_overrides(path) == []

    def test_rejects_prior_date_inside_window(self, tmp_path):
        path = _write_overrides(tmp_path / "o.json", [{**ENTRY, "prior_date": "2026-11-16"}])
        assert load_drift_overrides(path) == []

    def test_rejects_missing_authorization_fields(self, tmp_path):
        for name in ("reason", "authorized_by", "memo", "id"):
            path = _write_overrides(tmp_path / "o.json", [{**ENTRY, name: " "}])
            assert load_drift_overrides(path) == []

    def test_fail_reason_metric(self):
        assert fail_reason_metric("FAIL top20_overlap_pct: 14.3% < 30.0%") == "top20_overlap_pct"
        assert fail_reason_metric("warn_top20_overlap_pct: 50.0% < 70.0%") == ""

    def test_no_fail_reasons_never_matches(self, tmp_path):
        entries = load_drift_overrides(_write_overrides(tmp_path / "o.json", [ENTRY]))
        assert find_applicable_override(entries, "2026-11-16", "2026-11-12", []) is None

    def test_committed_file_is_valid_and_narrow(self):
        raw = json.loads(DEFAULT_OVERRIDES_PATH.read_text(encoding="utf-8"))
        entries = load_drift_overrides()
        assert len(entries) == len(raw["overrides"]), "every committed entry must pass validation"
        friday = ["FAIL top20_overlap_pct: 14.3% < 30.0%"]
        assert find_applicable_override(entries, "2026-11-16", "2026-11-12", friday) is not None
        assert find_applicable_override(entries, "2026-11-17", "2026-11-16", friday) is None
        assert find_applicable_override(entries, "2026-11-23", "2026-11-12", friday) is None


# --- Turnover gate: same dated override, and "not measured" is a warning ----------

from run_daily_production import check_turnover  # noqa: E402

TURNOVER_ENTRY = {**ENTRY, "metrics": ["top20_overlap_pct", "name_turnover_pct"]}


def _write_delta(snap: Path, turnover: float | None, prior: str | None = "2026-11-12", suppressed: bool = False):
    snap.mkdir(parents=True, exist_ok=True)
    lines = ["PHASE-2 RUN DELTA REPORT", f"Prior:     {prior} (ruleset 8887576e)" if prior else "Prior:     (none)"]
    if suppressed:
        lines.append("*** CHURN COMPARISONS: SUPPRESSED ***")
    lines.append("1. PORTFOLIO TURNOVER")
    if turnover is None:
        lines.append("  Portfolio: 30 names (no prior for comparison)")
    else:
        lines.append(f"  Name turnover: {turnover}%")
    (snap / "phase2_run_delta_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return snap


class TestTurnoverGateOverride:
    def test_high_turnover_blocks_without_override(self, tmp_path):
        snap = _write_delta(tmp_path / "s", 43.3)
        overrides = _write_overrides(tmp_path / "o.json", [])
        gate = check_turnover(snap, 40.0, "2026-11-17", overrides_path=overrides)
        assert gate.status == "FAIL"

    def test_override_downgrades_to_warn(self, tmp_path):
        snap = _write_delta(tmp_path / "s", 43.3)
        overrides = _write_overrides(tmp_path / "o.json", [TURNOVER_ENTRY])
        gate = check_turnover(snap, 40.0, "2026-11-17", overrides_path=overrides)
        assert gate.status == "WARN"
        assert "OPERATOR OVERRIDE test-order-switch" in gate.detail
        assert gate.value == 43.3

    def test_override_needs_the_turnover_metric(self, tmp_path):
        snap = _write_delta(tmp_path / "s", 43.3)
        overrides = _write_overrides(tmp_path / "o.json", [ENTRY])  # drift metrics only
        assert check_turnover(snap, 40.0, "2026-11-17", overrides_path=overrides).status == "FAIL"

    def test_override_lapses_with_a_new_baseline_or_outside_the_window(self, tmp_path):
        overrides = _write_overrides(tmp_path / "o.json", [TURNOVER_ENTRY])
        new_base = _write_delta(tmp_path / "a", 43.3, prior="2026-11-17")
        assert check_turnover(new_base, 40.0, "2026-11-18", overrides_path=overrides).status == "FAIL"
        late = _write_delta(tmp_path / "b", 43.3)
        assert check_turnover(late, 40.0, "2026-11-23", overrides_path=overrides).status == "FAIL"

    def test_without_as_of_date_nothing_is_overridden(self, tmp_path):
        snap = _write_delta(tmp_path / "s", 43.3)
        overrides = _write_overrides(tmp_path / "o.json", [TURNOVER_ENTRY])
        assert check_turnover(snap, 40.0, overrides_path=overrides).status == "FAIL"

    def test_normal_turnover_passes(self, tmp_path):
        snap = _write_delta(tmp_path / "s", 10.0)
        gate = check_turnover(snap, 40.0, "2026-11-17")
        assert gate.status == "PASS" and gate.value == 10.0

    def test_unmeasured_turnover_is_a_warning_not_a_pass(self, tmp_path):
        suppressed = _write_delta(tmp_path / "a", None, prior=None, suppressed=True)
        gate = check_turnover(suppressed, 40.0, "2026-11-16")
        assert gate.status == "WARN"
        assert "NOT measured" in gate.detail and "suppressed" in gate.detail
        no_prior = _write_delta(tmp_path / "b", None, prior=None)
        assert check_turnover(no_prior, 40.0, "2026-11-16").status == "WARN"

    def test_missing_report_still_passes(self, tmp_path):
        (tmp_path / "s").mkdir()
        assert check_turnover(tmp_path / "s", 40.0, "2026-11-16").status == "PASS"

    def test_committed_entry_covers_turnover_against_2026_11_12(self):
        entries = load_drift_overrides()
        reason = ["FAIL name_turnover_pct: 43.3% > 40.0%"]
        assert find_applicable_override(entries, "2026-11-17", "2026-11-12", reason) is not None
        assert find_applicable_override(entries, "2026-11-18", "2026-11-17", reason) is None
