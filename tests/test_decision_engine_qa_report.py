"""Tests for the Decision Engine QA report generator."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Import the module under test
# ---------------------------------------------------------------------------
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.decision_engine_qa_report import (
    SECTION_NAMES,
    _section_coverage,
    _section_eligibility,
    _section_health_gate,
    _section_portfolio,
    _section_score_dispersion,
    _section_tier_distribution,
    _section_universe,
    compute_overall_status,
    format_markdown,
    generate_report,
)
from run_phase2_snapshot_delta import SnapshotData, load_snapshot

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SNAP_DIR = Path(__file__).resolve().parent.parent / "data" / "snapshots"


def _latest_snapshot_dir() -> Path | None:
    """Return the most recent snapshot directory, or None."""
    if not SNAP_DIR.is_dir():
        return None
    dirs = sorted(
        [d for d in SNAP_DIR.iterdir() if d.is_dir() and (d / "rankings.csv").exists()],
        key=lambda d: d.name,
    )
    return dirs[-1] if dirs else None


def _make_snap(rankings_data: list[dict], portfolio_data: list[dict] | None = None,
               health_json: dict | None = None) -> SnapshotData:
    """Build a minimal SnapshotData from dicts (no I/O)."""
    import pandas as pd
    import tempfile

    rankings = pd.DataFrame(rankings_data).astype(str)
    if portfolio_data is not None:
        portfolio = pd.DataFrame(portfolio_data).astype(str)
    else:
        portfolio = pd.DataFrame(columns=rankings.columns).astype(str)

    tmp = Path(tempfile.mkdtemp())
    if health_json is not None:
        with open(tmp / "phase2_health.json", "w") as f:
            json.dump(health_json, f)

    return SnapshotData(
        date="2026-01-01",
        path=tmp,
        rankings=rankings,
        portfolio=portfolio,
        metadata={},
        ruleset_id="test1234",
        has_native_portfolio=portfolio_data is not None,
    )


def _dev_row(**overrides) -> dict:
    """Generate a single drug_developer row with sensible defaults."""
    row = {
        "ticker": "TEST",
        "archetype": "drug_developer",
        "eligible": "1",
        "ineligible_reasons": "",
        "tier_dev": "B",
        "tier_reason": "moderate",
        "size_band": "M",
        "size_reasons": "",
        "catalyst_mode": "specific_days",
        "catalyst_days": "30",
        "mom_state": "neutral",
        "risk_flags": "",
        "composite_score": "50.0",
        "composite_rank": "100",
        "clinical_optionality_pct_dev": "0.5",
        "sponsor_tier1_count": "2",
        "target_weight_pct": "5.0",
        "actionable_rank": "10",
        "decision_engine_version": "v1.2.0",
        "decision_engine_ruleset_id": "test1234",
    }
    row.update(overrides)
    return row


# ---------------------------------------------------------------------------
# TestOverallStatus — pure logic, no I/O
# ---------------------------------------------------------------------------


class TestOverallStatus:
    def test_all_ok(self):
        sections = {n: {"status": "OK"} for n in SECTION_NAMES}
        assert compute_overall_status(sections) == "OK"

    def test_one_warn(self):
        sections = {n: {"status": "OK"} for n in SECTION_NAMES}
        sections["coverage"]["status"] = "WARN"
        assert compute_overall_status(sections) == "WARN"

    def test_one_fail(self):
        sections = {n: {"status": "OK"} for n in SECTION_NAMES}
        sections["universe"]["status"] = "FAIL"
        assert compute_overall_status(sections) == "FAIL"

    def test_fail_beats_warn(self):
        sections = {n: {"status": "OK"} for n in SECTION_NAMES}
        sections["coverage"]["status"] = "WARN"
        sections["portfolio"]["status"] = "FAIL"
        assert compute_overall_status(sections) == "FAIL"

    def test_multiple_warns(self):
        sections = {n: {"status": "WARN"} for n in SECTION_NAMES}
        assert compute_overall_status(sections) == "WARN"


# ---------------------------------------------------------------------------
# TestSectionStatusLogic — unit test each section's WARN/FAIL thresholds
# ---------------------------------------------------------------------------


class TestSectionStatusLogic:

    # -- Universe --
    def test_universe_ok(self):
        rows = [_dev_row(ticker=f"T{i}") for i in range(300)]
        snap = _make_snap(rows)
        status, metrics = _section_universe(snap)
        assert status == "OK"
        assert metrics["ranked"] == 300

    def test_universe_fail_empty(self):
        snap = _make_snap([])
        status, _ = _section_universe(snap)
        assert status == "FAIL"

    def test_universe_warn_low(self):
        rows = [_dev_row(ticker=f"T{i}") for i in range(250)]
        snap = _make_snap(rows)
        status, _ = _section_universe(snap)
        assert status == "WARN"

    def test_universe_warn_high(self):
        rows = [_dev_row(ticker=f"T{i}") for i in range(410)]
        snap = _make_snap(rows)
        status, _ = _section_universe(snap)
        assert status == "WARN"

    # -- Eligibility --
    def test_eligibility_ok(self):
        rows = [_dev_row(ticker=f"T{i}", eligible="1") for i in range(100)]
        # Add some with eligible=0 to bring dev_eligible_pct between 15-50
        rows += [_dev_row(ticker=f"I{i}", eligible="0",
                          ineligible_reasons="deep_drawdown") for i in range(200)]
        snap = _make_snap(rows)
        status, metrics = _section_eligibility(snap)
        # 100/(100+200) = 33.3% dev eligible
        assert status == "OK"
        assert metrics["dev_eligible_pct"] == pytest.approx(33.3, abs=0.1)

    def test_eligibility_fail_none(self):
        rows = [_dev_row(ticker=f"T{i}", eligible="0",
                         ineligible_reasons="fundamental_red_flag") for i in range(50)]
        snap = _make_snap(rows)
        status, _ = _section_eligibility(snap)
        assert status == "FAIL"

    def test_eligibility_warn_low_dev_pct(self):
        eligible = [_dev_row(ticker=f"E{i}", eligible="1") for i in range(10)]
        ineligible = [_dev_row(ticker=f"I{i}", eligible="0",
                               ineligible_reasons="x") for i in range(90)]
        snap = _make_snap(eligible + ineligible)
        status, metrics = _section_eligibility(snap)
        assert status == "WARN"
        assert metrics["dev_eligible_pct"] == 10.0

    def test_eligibility_warn_high_dev_pct(self):
        eligible = [_dev_row(ticker=f"E{i}", eligible="1") for i in range(60)]
        ineligible = [_dev_row(ticker=f"I{i}", eligible="0",
                               ineligible_reasons="x") for i in range(10)]
        snap = _make_snap(eligible + ineligible)
        status, metrics = _section_eligibility(snap)
        # 60/70 = 85.7% > 50%
        assert status == "WARN"

    # -- Tier Distribution --
    def test_tier_ok(self):
        rows = [_dev_row(ticker=f"A{i}", tier_dev="A") for i in range(5)]
        rows += [_dev_row(ticker=f"B{i}", tier_dev="B") for i in range(10)]
        rows += [_dev_row(ticker=f"C{i}", tier_dev="C") for i in range(20)]
        snap = _make_snap(rows)
        status, metrics = _section_tier_distribution(snap)
        assert status == "OK"
        assert metrics["tier_counts"]["A"] == 5

    def test_tier_warn_no_a(self):
        rows = [_dev_row(ticker=f"B{i}", tier_dev="B") for i in range(10)]
        rows += [_dev_row(ticker=f"C{i}", tier_dev="C") for i in range(20)]
        snap = _make_snap(rows)
        status, _ = _section_tier_distribution(snap)
        assert status == "WARN"

    def test_tier_warn_low_b(self):
        rows = [_dev_row(ticker=f"A{i}", tier_dev="A") for i in range(5)]
        rows += [_dev_row(ticker=f"B{i}", tier_dev="B") for i in range(2)]
        rows += [_dev_row(ticker=f"C{i}", tier_dev="C") for i in range(20)]
        snap = _make_snap(rows)
        status, _ = _section_tier_distribution(snap)
        assert status == "WARN"

    # -- Coverage --
    def test_coverage_ok(self):
        rows = [_dev_row(ticker=f"T{i}", catalyst_mode="specific_days",
                         sponsor_tier1_count="3") for i in range(80)]
        rows += [_dev_row(ticker=f"N{i}", catalyst_mode="no_upcoming",
                          sponsor_tier1_count="1") for i in range(20)]
        snap = _make_snap(rows)
        status, metrics = _section_coverage(snap)
        assert status == "OK"
        assert metrics["catalyst_specific_pct"] == 80.0

    def test_coverage_fail_no_catalyst(self):
        rows = [_dev_row(ticker=f"T{i}", catalyst_mode="no_upcoming",
                         sponsor_tier1_count="1") for i in range(50)]
        snap = _make_snap(rows)
        status, _ = _section_coverage(snap)
        # 0% specific_days → FAIL
        assert status == "FAIL"

    def test_coverage_warn_low_catalyst(self):
        rows = [_dev_row(ticker=f"T{i}", catalyst_mode="specific_days") for i in range(10)]
        rows += [_dev_row(ticker=f"N{i}", catalyst_mode="no_upcoming") for i in range(90)]
        snap = _make_snap(rows)
        status, metrics = _section_coverage(snap)
        # 10/100 = 10% < 20% → WARN
        assert status == "WARN"
        assert metrics["catalyst_specific_pct"] == 10.0

    def test_coverage_warn_low_sponsor(self):
        rows = [_dev_row(ticker=f"T{i}", catalyst_mode="specific_days",
                         sponsor_tier1_count="") for i in range(100)]
        snap = _make_snap(rows)
        status, _ = _section_coverage(snap)
        # 100% catalyst but 0% sponsor → WARN
        assert status == "WARN"

    # -- Score Dispersion --
    def test_dispersion_ok(self):
        import random
        random.seed(42)
        rows = [_dev_row(ticker=f"T{i}",
                         clinical_optionality_pct_dev=str(round(random.uniform(0.1, 0.9), 4)))
                for i in range(100)]
        snap = _make_snap(rows)
        status, metrics = _section_score_dispersion(snap)
        assert status == "OK"
        assert metrics["optionality_std"] > 0.15

    def test_dispersion_warn_compressed(self):
        rows = [_dev_row(ticker=f"T{i}",
                         clinical_optionality_pct_dev="0.50") for i in range(100)]
        snap = _make_snap(rows)
        status, metrics = _section_score_dispersion(snap)
        assert status == "WARN"
        assert metrics["optionality_std"] < 0.15

    # -- Portfolio --
    def test_portfolio_ok(self):
        port = [_dev_row(ticker=f"P{i}", target_weight_pct="5.0",
                         tier_dev="A", size_band="L") for i in range(20)]
        snap = _make_snap([_dev_row()], portfolio_data=port)
        status, metrics = _section_portfolio(snap)
        assert status == "OK"
        assert metrics["n_positions"] == 20
        assert metrics["top5_weight_pct"] == 25.0

    def test_portfolio_fail_empty(self):
        snap = _make_snap([_dev_row()], portfolio_data=[])
        status, _ = _section_portfolio(snap)
        assert status == "FAIL"

    def test_portfolio_warn_small(self):
        port = [_dev_row(ticker=f"P{i}", target_weight_pct="5.0",
                         tier_dev="A", size_band="L") for i in range(8)]
        snap = _make_snap([_dev_row()], portfolio_data=port)
        status, _ = _section_portfolio(snap)
        assert status == "WARN"

    def test_portfolio_warn_concentrated(self):
        port = [_dev_row(ticker=f"P{i}", target_weight_pct="12.0",
                         tier_dev="A", size_band="L") for i in range(10)]
        snap = _make_snap([_dev_row()], portfolio_data=port)
        status, metrics = _section_portfolio(snap)
        # top-5 weight = 60% > 50%
        assert status == "WARN"
        assert metrics["top5_weight_pct"] == 60.0

    # -- Health Gate --
    def test_health_gate_not_present(self):
        snap = _make_snap([_dev_row()])
        status, metrics = _section_health_gate(snap)
        assert status == "OK"
        assert metrics["available"] is False

    def test_health_gate_ok(self):
        snap = _make_snap([_dev_row()],
                          health_json={"status": "OK", "reasons": [], "metrics": {}})
        status, metrics = _section_health_gate(snap)
        assert status == "OK"
        assert metrics["available"] is True

    def test_health_gate_warn(self):
        snap = _make_snap([_dev_row()],
                          health_json={"status": "WARN", "reasons": ["low_a"], "metrics": {}})
        status, metrics = _section_health_gate(snap)
        assert status == "WARN"

    def test_health_gate_fail(self):
        snap = _make_snap([_dev_row()],
                          health_json={"status": "FAIL", "reasons": ["broken"], "metrics": {}})
        status, metrics = _section_health_gate(snap)
        assert status == "FAIL"


# ---------------------------------------------------------------------------
# TestQAReportGeneration — integration test with real snapshot
# ---------------------------------------------------------------------------


class TestQAReportGeneration:

    @pytest.fixture()
    def snapshot(self):
        snap_dir = _latest_snapshot_dir()
        if snap_dir is None:
            pytest.skip("No snapshot available")
        snap = load_snapshot(snap_dir)
        if snap is None:
            pytest.skip("Could not load snapshot")
        return snap

    def test_json_schema(self, snapshot):
        report = generate_report(snapshot)
        assert "generated_at" in report
        assert "snapshot_date" in report
        assert "overall_status" in report
        assert report["overall_status"] in ("OK", "WARN", "FAIL")
        assert "ruleset_id" in report
        assert "engine_version" in report
        assert "git_sha" in report
        assert "sections" in report
        for name in SECTION_NAMES:
            assert name in report["sections"], f"Missing section: {name}"
            sec = report["sections"][name]
            assert "status" in sec
            assert sec["status"] in ("OK", "WARN", "FAIL")
            assert "metrics" in sec
            assert isinstance(sec["metrics"], dict)

    def test_markdown_has_all_sections(self, snapshot):
        report = generate_report(snapshot)
        md = format_markdown(report)
        assert "# Decision Engine QA Report" in md
        assert "## Universe" in md
        assert "## Eligibility" in md
        assert "## Tier Distribution" in md
        assert "## Data Coverage" in md
        assert "## Score Dispersion" in md
        assert "## Portfolio" in md
        assert "## Health Gate" in md

    def test_json_round_trip(self, snapshot):
        report = generate_report(snapshot)
        serialized = json.dumps(report, indent=2)
        loaded = json.loads(serialized)
        assert loaded["overall_status"] == report["overall_status"]
        assert set(loaded["sections"].keys()) == set(report["sections"].keys())
