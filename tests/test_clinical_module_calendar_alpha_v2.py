"""
Tests for Clinical Calendar Alpha v2 — common/clinical_calendar_alpha.py

6 test groups:
  1. Program clustering stability
  2. Readout density correctness
  3. Execution momentum deltas
  4. Design + endpoint scoring
  5. Competitive intensity modifier
  6. Determinism (full pipeline reproducibility)
"""
import json
import sys
from pathlib import Path

import pytest

# Ensure project root is on sys.path
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from common.clinical_calendar_alpha import (
    CalendarAlphaConfig,
    cluster_programs,
    compose_clinical_score_v2,
    compute_design_quality,
    compute_endpoint_strength,
    compute_execution_momentum,
    compute_program_features,
    compute_readout_density,
    normalize_intervention,
    z_score_dict,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _trial(
    ticker="ACME",
    nct_id="NCT00000001",
    phase="PHASE2",
    status="RECRUITING",
    conditions=None,
    interventions=None,
    title="A Study of Drug X",
    pcd=None,
    cd=None,
    rfp=None,
    fp="2025-01-01",
    lup="2025-06-01",
):
    """Build a synthetic trial record."""
    return {
        "ticker": ticker,
        "nct_id": nct_id,
        "phase": phase,
        "status": status,
        "study_type": "INTERVENTIONAL",
        "conditions": conditions or ["Non-Small Cell Lung Cancer"],
        "interventions": interventions or ["Drug X 100mg", "Placebo"],
        "title": title,
        "primary_completion_date": pcd,
        "completion_date": cd,
        "results_first_posted": rfp,
        "first_posted": fp,
        "last_update_posted": lup,
    }


AS_OF = "2025-12-01"


# ===========================================================================
# 1. Program clustering stability
# ===========================================================================


class TestProgramClustering:
    """Same trials shuffled → identical program output; placebo filtered, dose grouped."""

    def test_shuffle_stability(self):
        """Same trials in different order produce identical programs."""
        from datetime import date

        trials = [
            _trial(nct_id="NCT001", interventions=["Drug X 50mg", "Placebo"], pcd="2026-03-01"),
            _trial(nct_id="NCT002", interventions=["Drug X 100mg", "Placebo"], pcd="2026-06-01"),
            _trial(nct_id="NCT003", interventions=["Drug Y"], conditions=["Breast Cancer"], pcd="2026-09-01"),
        ]
        as_of = date(2025, 12, 1)

        progs_a = cluster_programs(trials, "ACME", as_of)
        progs_b = cluster_programs(list(reversed(trials)), "ACME", as_of)

        # Same number of programs
        assert len(progs_a) == len(progs_b)
        # Same NCT IDs in each program (order-independent)
        for pa, pb in zip(progs_a, progs_b):
            assert sorted(pa["nct_ids"]) == sorted(pb["nct_ids"])

    def test_placebo_filtered_dose_grouped(self):
        """Placebo arms excluded; dose variants of same drug cluster together."""
        from datetime import date

        trials = [
            _trial(nct_id="NCT001", interventions=["Drug X 50mg", "Placebo"], pcd="2026-03-01"),
            _trial(nct_id="NCT002", interventions=["Drug X 100mg", "Placebo"], pcd="2026-06-01"),
        ]
        as_of = date(2025, 12, 1)

        progs = cluster_programs(trials, "ACME", as_of)
        # Both trials should cluster into one program (same drug, different doses)
        assert len(progs) == 1
        assert progs[0]["trial_count"] == 2

    def test_normalize_intervention(self):
        """Dosage patterns stripped, placebo returns None."""
        assert normalize_intervention("Placebo") is None
        assert normalize_intervention("Saline injection") is None
        normed = normalize_intervention("Drug X 100mg tablets")
        assert normed is not None
        assert "100" not in normed
        assert "tablet" not in normed


# ===========================================================================
# 2. Readout density correctness
# ===========================================================================


class TestReadoutDensity:
    """Known dates → correct density; K=8 cap enforced; past PCD excluded."""

    def test_density_windows(self):
        """Known future dates yield correct density_30/90/180 counts."""
        trials = [
            _trial(nct_id="NCT001", pcd="2025-12-15"),  # 14 days → in 30
            _trial(nct_id="NCT002", pcd="2026-02-01"),   # 62 days → in 90
            _trial(nct_id="NCT003", pcd="2026-04-01"),   # 121 days → in 180
            _trial(nct_id="NCT004", pcd="2026-09-01"),   # 274 days → outside 180
        ]
        result = compute_readout_density(trials, AS_OF)
        rd = result["ACME"]
        assert rd["readout_density_30"] == 1
        assert rd["readout_density_90"] == 2
        assert rd["readout_density_180"] == 3
        # Curve score should be monotonic (closer = higher contribution)
        assert rd["readout_curve_score"] > 0

    def test_k_cap(self):
        """Only top K=8 events contribute to curve score."""
        # Create 10 events, all within 180 days
        trials = [
            _trial(nct_id=f"NCT{i:03d}", pcd=f"2026-0{(i % 5) + 1}-15")
            for i in range(1, 11)
        ]
        result_k8 = compute_readout_density(trials, AS_OF, K=8)
        result_k4 = compute_readout_density(trials, AS_OF, K=4)
        # K=8 should have higher (or equal) curve score than K=4
        assert result_k8["ACME"]["readout_curve_score"] >= result_k4["ACME"]["readout_curve_score"]

    def test_past_pcd_excluded(self):
        """Past PCD (even with no results_first_posted) is NOT counted as imminent."""
        trials = [
            _trial(nct_id="NCT001", pcd="2025-06-01", rfp=None),  # past PCD, no results
        ]
        result = compute_readout_density(trials, AS_OF)
        rd = result.get("ACME", {})
        assert rd.get("readout_density_30", 0) == 0
        assert rd.get("readout_density_90", 0) == 0
        assert rd.get("readout_curve_score", 0.0) == 0.0


# ===========================================================================
# 3. Execution momentum deltas
# ===========================================================================


class TestExecutionMomentum:
    """Status transitions, PCD shifts, no-prior edge case."""

    def test_status_transitions(self):
        """RECRUITING→COMPLETED = advancement; RECRUITING→TERMINATED = regression."""
        current = [
            _trial(nct_id="NCT001", status="COMPLETED"),
            _trial(nct_id="NCT002", status="TERMINATED"),
        ]
        prior = [
            _trial(nct_id="NCT001", status="RECRUITING", fp="2024-01-01", lup="2024-06-01"),
            _trial(nct_id="NCT002", status="RECRUITING", fp="2024-01-01", lup="2024-06-01"),
        ]
        result = compute_execution_momentum(current, prior, AS_OF)
        m = result["ACME"]
        assert m["execution_advancements"] == 1
        # Regression: RECRUITING→TERMINATED with TERMINATED in regression set
        assert m["execution_momentum_score"] == 0.0  # 1 adv - 1 reg / 2 = 0

    def test_pcd_slippage(self):
        """PCD pushed out 30 days = slippage detected."""
        current = [
            _trial(nct_id="NCT001", pcd="2026-07-01"),
        ]
        prior = [
            _trial(nct_id="NCT001", pcd="2026-06-01", fp="2024-01-01", lup="2024-06-01"),
        ]
        result = compute_execution_momentum(current, prior, AS_OF)
        assert result["ACME"]["execution_slippages"] == 1

    def test_no_prior(self):
        """No prior trials → zeros with coverage='no_prior'."""
        current = [_trial(nct_id="NCT001")]
        result = compute_execution_momentum(current, None, AS_OF)
        m = result["ACME"]
        assert m["execution_momentum_score"] == 0.0
        assert m["execution_delta_coverage"] == "no_prior"


# ===========================================================================
# 4. Design + endpoint scoring
# ===========================================================================


class TestDesignQuality:
    """Title-derived design signals."""

    def test_strong_design(self):
        """Randomized, Double-Blind, Placebo-Controlled → design_quality ≥ 0.8."""
        trials = [
            _trial(
                nct_id="NCT001",
                title="A Randomized, Double-Blind, Placebo-Controlled Phase 3 Study",
            ),
        ]
        result = compute_design_quality(trials, AS_OF)
        dq = result["ACME"]
        assert dq["design_quality_score"] >= 0.8
        assert "randomized" in dq["design_signals"]
        assert "double_blind" in dq["design_signals"]

    def test_open_label(self):
        """Open-Label Safety Study → design_quality ≤ 0.5."""
        trials = [
            _trial(
                nct_id="NCT001",
                title="An Open-Label Safety Study of Drug X",
            ),
        ]
        result = compute_design_quality(trials, AS_OF)
        dq = result["ACME"]
        assert dq["design_quality_score"] <= 0.5


class TestEndpointStrength:
    """TA-based endpoint proxy."""

    def test_oncology_stronger(self):
        """Oncology conditions → higher endpoint strength than generic."""
        # Use multiple oncology keywords for unambiguous TA classification
        onc_trials = [
            _trial(nct_id="NCT001", conditions=["Metastatic Breast Cancer", "Carcinoma"], phase="PHASE3"),
        ]
        generic_trials = [
            _trial(nct_id="NCT001", conditions=["General Health"], phase="PHASE3", ticker="GENR"),
        ]
        onc_result = compute_endpoint_strength(onc_trials, AS_OF)
        gen_result = compute_endpoint_strength(generic_trials, AS_OF)
        assert onc_result["ACME"]["endpoint_strength_score"] > gen_result["GENR"]["endpoint_strength_score"]

    def test_ta_classification(self):
        """Oncology trials → therapeutic_area = 'oncology'."""
        trials = [
            _trial(nct_id="NCT001", conditions=["Breast Cancer", "Metastatic"]),
        ]
        result = compute_endpoint_strength(trials, AS_OF)
        assert result["ACME"]["therapeutic_area"] == "oncology"


# ===========================================================================
# 5. Competitive intensity modifier
# ===========================================================================


class TestCompetitiveIntensity:
    """Sizing dampen for crowded indications."""

    def test_compose_with_high_competition(self):
        """High competition z → sizing dampen applied when sizing enabled."""
        config = CalendarAlphaConfig(enable_sizing=True)
        features = {
            "design_quality_score": 0.5,
            "readout_curve_score": 0.0,
            "endpoint_strength_score": 0.5,
        }
        score_v2, sizing, tags = compose_clinical_score_v2(
            clinical_score=60.0,
            features=features,
            config=config,
            z_competition=1.5,  # high competition
        )
        assert sizing < 1.0
        assert "sizing_competition_dampen" in tags

    def test_compose_no_sizing_by_default(self):
        """Default config: sizing multiplier stays at 1.0."""
        config = CalendarAlphaConfig()
        features = {"design_quality_score": 0.5}
        _, sizing, _ = compose_clinical_score_v2(
            clinical_score=60.0,
            features=features,
            config=config,
            z_competition=1.5,
        )
        assert sizing == 1.0


# ===========================================================================
# 6. Determinism
# ===========================================================================


class TestDeterminism:
    """Full pipeline with synthetic trials → re-run → identical output."""

    def test_full_pipeline_deterministic(self):
        """Two runs with identical inputs produce byte-identical JSON output."""
        trials = [
            _trial(nct_id="NCT001", ticker="ACME", pcd="2026-03-01",
                   interventions=["Drug X 50mg", "Placebo"],
                   conditions=["Non-Small Cell Lung Cancer"],
                   title="Randomized Double-Blind Phase 2 Study"),
            _trial(nct_id="NCT002", ticker="ACME", pcd="2026-06-01",
                   interventions=["Drug X 100mg"],
                   conditions=["Non-Small Cell Lung Cancer"]),
            _trial(nct_id="NCT003", ticker="BETA", pcd="2026-02-15",
                   interventions=["Drug Y"],
                   conditions=["Breast Cancer"],
                   title="Open-Label Phase 3 Study"),
            _trial(nct_id="NCT004", ticker="BETA", pcd="2027-01-01",
                   interventions=["Drug Z"],
                   conditions=["Melanoma"],
                   phase="PHASE1"),
        ]

        def _run():
            prog = compute_program_features(trials, AS_OF)
            rd = compute_readout_density(trials, AS_OF)
            mom = compute_execution_momentum(trials, None, AS_OF)
            des = compute_design_quality(trials, AS_OF)
            ep = compute_endpoint_strength(trials, AS_OF)
            return {
                "program": prog,
                "readout": rd,
                "momentum": mom,
                "design": des,
                "endpoint": ep,
            }

        run1 = json.dumps(_run(), sort_keys=True, default=str)
        run2 = json.dumps(_run(), sort_keys=True, default=str)
        assert run1 == run2


# ===========================================================================
# Additional edge case tests
# ===========================================================================


class TestZScoreDict:
    """Cross-sectional z-score utility."""

    def test_basic_z(self):
        """Known values → correct z-scores."""
        vals = {"A": 10.0, "B": 20.0, "C": 30.0}
        z = z_score_dict(vals)
        assert z["A"] < 0
        assert abs(z["B"]) < 0.01  # mean
        assert z["C"] > 0

    def test_empty(self):
        assert z_score_dict({}) == {}

    def test_uniform(self):
        """All same values → all z = 0."""
        vals = {"A": 5.0, "B": 5.0, "C": 5.0}
        z = z_score_dict(vals)
        assert all(v == 0.0 for v in z.values())


class TestComposeScoreV2:
    """compose_clinical_score_v2 edge cases."""

    def test_none_clinical_score(self):
        """None clinical_score → (None, 1.0, [])."""
        config = CalendarAlphaConfig()
        v2, sizing, tags = compose_clinical_score_v2(None, {}, config)
        assert v2 is None
        assert sizing == 1.0
        assert tags == []

    def test_clamped_to_range(self):
        """Score stays in [0, 100] even with extreme adjustments."""
        config = CalendarAlphaConfig()
        v2, _, _ = compose_clinical_score_v2(
            98.0, {}, config,
            z_readout_curve=3.0, z_design=3.0, z_endpoint=3.0,
        )
        assert v2 is not None
        assert 0.0 <= v2 <= 100.0

    def test_sizing_max_deviation(self):
        """Sizing bounded by ±max_deviation."""
        config = CalendarAlphaConfig(enable_sizing=True, sizing_max_deviation=0.10)
        features = {"design_quality_score": 0.1, "readout_curve_score": 0.0, "endpoint_strength_score": 0.3}
        _, sizing, _ = compose_clinical_score_v2(
            50.0, features, config,
            z_competition=3.0,  # extreme
        )
        assert sizing >= 0.90
        assert sizing <= 1.10


class TestPITEnforcement:
    """Strict PIT < as_of_date (no same-day leak)."""

    def test_same_day_excluded(self):
        """Trial posted on as_of_date is excluded (strict <)."""
        trials = [
            _trial(nct_id="NCT001", fp="2025-12-01", lup="2025-12-01", pcd="2026-06-01"),
        ]
        result = compute_readout_density(trials, AS_OF)
        # Should be empty or zero — trial posted same day as as_of
        rd = result.get("ACME", {})
        assert rd.get("readout_density_30", 0) == 0

    def test_day_before_included(self):
        """Trial posted day before as_of_date is included."""
        trials = [
            _trial(nct_id="NCT001", fp="2025-11-30", lup="2025-11-30", pcd="2025-12-15"),
        ]
        result = compute_readout_density(trials, AS_OF)
        rd = result.get("ACME", {})
        assert rd["readout_density_30"] == 1


class TestRulesetSizingPlumbing:
    """Gap-2 fix: enable_clinical_sizing from DecisionRuleset wires into CalendarAlphaConfig."""

    def test_sizing_disabled_by_default(self):
        """DecisionRuleset default (enable_clinical_sizing=False) → all multipliers = 1.0."""
        from decision_engine import DecisionRuleset

        rs = DecisionRuleset()
        assert rs.enable_clinical_sizing is False

        # Build CalendarAlphaConfig the same way run_screen.py now does
        config = CalendarAlphaConfig(enable_sizing=rs.enable_clinical_sizing)
        assert config.enable_sizing is False

        features = {"design_quality_score": 0.5, "readout_curve_score": 0.8}
        _, sizing, tags = compose_clinical_score_v2(
            clinical_score=70.0,
            features=features,
            config=config,
            z_competition=2.0,  # would trigger dampen if sizing were enabled
        )
        assert sizing == 1.0, f"Expected 1.0 sizing with disabled flag, got {sizing}"
        assert not any("sizing" in t for t in tags)

    def test_sizing_enabled_from_ruleset(self):
        """DecisionRuleset with enable_clinical_sizing=True → at least one multiplier ≠ 1.0."""
        from decision_engine import DecisionRuleset

        rs = DecisionRuleset(enable_clinical_sizing=True)
        assert rs.enable_clinical_sizing is True

        config = CalendarAlphaConfig(enable_sizing=rs.enable_clinical_sizing)
        assert config.enable_sizing is True

        features = {"design_quality_score": 0.1, "readout_curve_score": 0.0}
        _, sizing, tags = compose_clinical_score_v2(
            clinical_score=50.0,
            features=features,
            config=config,
            z_competition=2.0,  # high competition → dampen
        )
        assert sizing != 1.0, f"Expected sizing != 1.0 when enabled with high competition, got {sizing}"
        assert any("sizing" in t for t in tags)


class TestCalendarAlphaSortIntegration:
    """Regression: clinical_score_v2_z must be injected for sort tilt to work."""

    def test_sort_tilt_nonzero_when_enabled(self):
        """With enable_calendar_alpha_sort=True, a ticker with high v2 z-score
        gets a different sort key than one with low v2 z-score."""
        from decision_engine import compute_actionable_sort_key, DecisionRuleset

        rs = DecisionRuleset(
            enable_calendar_alpha_sort=True,
            calendar_alpha_sort_weight=0.5,
            catalyst_priority_mode="tiebreaker",
        )

        base_fields = {
            "eligible": "1",
            "tier_dev": "A",
            "catalyst_mode": "specific_days",
            "catalyst_days": "30",
            "mom_state": "neutral",
            "sponsor_tier1_count": 3,
        }

        # High v2 z-score ticker
        high_fields = {**base_fields, "clinical_score_v2_z": 1.5}
        low_fields = {**base_fields, "clinical_score_v2_z": 0.0}

        key_high = compute_actionable_sort_key(
            high_fields, "drug_developer", 0.8, 10, "HIGH", ruleset=rs,
        )
        key_low = compute_actionable_sort_key(
            low_fields, "drug_developer", 0.8, 10, "LOW", ruleset=rs,
        )
        # Higher v2 z → lower effective_comp_rank → sorts first
        assert key_high < key_low, (
            f"Sort tilt should boost high v2 z-score ticker: {key_high} vs {key_low}"
        )
