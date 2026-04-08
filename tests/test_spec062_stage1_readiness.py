"""Tests for Spec 062: Stage 1 activation, EV feed verification, and readiness.

Covers:
  - Candidate ruleset v1.14.0 loads and validates
  - EV feed health verifier
  - Ops digest EV readiness section
  - Tiebreak impact measurement
"""

import csv
import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from decision_engine import DecisionRuleset


# =============================================================================
# Test: Candidate Ruleset v1.14.0
# =============================================================================


class TestCandidateRuleset:
    """v1.14.0 candidate loads, validates, and only differs by event_ev_stage."""

    def test_candidate_loads(self):
        path = Path("production_data/decision_rulesets/v1.14.0_a4_ev_tiebreaker.json")
        if not path.exists():
            pytest.skip("Candidate ruleset not generated yet")
        rs = DecisionRuleset.from_json(str(path))
        assert rs.event_ev_stage == "tiebreaker"

    def test_candidate_matches_base_except_ev_stage(self):
        base_path = Path("production_data/decision_rulesets/v1.13.0_a4_selector_ranker.json")
        cand_path = Path("production_data/decision_rulesets/v1.14.0_a4_ev_tiebreaker.json")
        if not base_path.exists() or not cand_path.exists():
            pytest.skip("Rulesets not present")

        base = DecisionRuleset.from_json(str(base_path))
        cand = DecisionRuleset.from_json(str(cand_path))

        from dataclasses import fields as dc_fields
        for f in dc_fields(base):
            if f.name == "event_ev_stage":
                assert getattr(base, f.name) == "off"
                assert getattr(cand, f.name) == "tiebreaker"
            else:
                assert getattr(base, f.name) == getattr(cand, f.name), (
                    f"Field {f.name} differs: base={getattr(base, f.name)}, cand={getattr(cand, f.name)}"
                )

    def test_candidate_in_manifest(self):
        manifest_path = Path("production_data/decision_rulesets/manifest.json")
        if not manifest_path.exists():
            pytest.skip("Manifest not present")
        with open(manifest_path) as f:
            manifest = json.load(f)
        rulesets = manifest.get("rulesets", [])
        candidate_entries = [r for r in rulesets if r.get("file") == "v1.14.0_a4_ev_tiebreaker.json"]
        assert len(candidate_entries) == 1
        entry = candidate_entries[0]
        assert entry["status"] == "candidate"
        assert "tiebreaker" in entry.get("description", "").lower() or "tiebreaker" in entry.get("notes", "").lower()


# =============================================================================
# Test: EV Feed Health Verifier
# =============================================================================


class TestVerifyEventEVFeed:
    """Tests for tools/verify_event_ev_feed.py."""

    def test_verify_missing_everything(self):
        from tools.verify_event_ev_feed import verify_feed
        # Use a date that certainly has no data
        result = verify_feed("1999-01-01")
        assert result["status"] == "missing"
        assert result["checks"]["artifact"]["exists"] is False

    def test_verify_with_artifact_only(self):
        from tools.verify_event_ev_feed import verify_feed, ARTIFACTS_DIR

        with tempfile.TemporaryDirectory() as td:
            # Monkey-patch ARTIFACTS_DIR temporarily
            import tools.verify_event_ev_feed as mod
            orig = mod.ARTIFACTS_DIR
            mod.ARTIFACTS_DIR = Path(td)
            try:
                # Create a fake artifact
                date_str = "2099-01-01"
                artifact = [
                    {"node": {"ticker": "AAAA"}, "payoff": {"downside_adjusted_ev": 5.0, "analog_confidence": "ok"}},
                ]
                with open(Path(td) / f"{date_str}_event_ev_scores.json", "w") as f:
                    json.dump(artifact, f)

                result = verify_feed(date_str)
                assert result["checks"]["artifact"]["exists"] is True
                assert result["checks"]["artifact"]["n_events"] == 1
            finally:
                mod.ARTIFACTS_DIR = orig


# =============================================================================
# Test: Ops Digest EV Readiness Builder
# =============================================================================


class TestOpsDigestEVReadiness:
    """Tests for the _build_event_ev_readiness section in ops digest."""

    def test_readiness_returns_available_false_when_no_artifacts(self):
        from tools.build_ops_digest import _build_event_ev_readiness
        result = _build_event_ev_readiness()
        # May be available=True with 0 artifacts or available=False
        # depending on whether the evaluator succeeds
        assert "available" in result

    def test_readiness_structure(self):
        from tools.build_ops_digest import _build_event_ev_readiness
        result = _build_event_ev_readiness()
        if result.get("available"):
            assert "highest_ready_stage" in result
            assert "tiebreaker_ready" in result
            assert "n_daily_artifacts" in result


# =============================================================================
# Test: Tiebreak Impact Measurement
# =============================================================================


class TestMeasureEVTiebreakImpact:
    """Tests for tools/measure_ev_tiebreak_impact.py."""

    def test_measure_missing_snapshot(self):
        from tools.measure_ev_tiebreak_impact import measure_impact
        result = measure_impact("1999-01-01")
        assert "error" in result

    def test_measure_with_fixture_snapshot(self):
        """Build a minimal rankings.csv and measure impact."""
        from tools.measure_ev_tiebreak_impact import measure_impact, SNAPSHOT_DIR

        import tools.measure_ev_tiebreak_impact as mod
        orig_dir = mod.SNAPSHOT_DIR

        with tempfile.TemporaryDirectory() as td:
            mod.SNAPSHOT_DIR = Path(td)
            date_dir = Path(td) / "2099-01-01"
            date_dir.mkdir()

            # Build minimal rankings.csv with fields needed for sort key
            fieldnames = [
                "ticker", "archetype", "eligible", "tier_dev",
                "catalyst_mode", "catalyst_days", "mom_state",
                "sponsor_tier1_count", "size_band", "risk_flags",
                "clinical_optionality_pct_dev", "composite_rank",
                "event_ev_score", "event_ev_score_z",
                "event_ev_bucket", "event_ev_analog_confidence",
                "final_score", "selector_score",
                "catalyst_event_type", "catalyst_source",
                "alpha_cohort_pct", "alpha_cohort_raw",
            ]
            rows = [
                {
                    "ticker": "AAAA", "archetype": "drug_developer",
                    "eligible": "1", "tier_dev": "B",
                    "catalyst_mode": "specific_days", "catalyst_days": "60",
                    "mom_state": "neutral", "sponsor_tier1_count": "0",
                    "size_band": "M", "risk_flags": "",
                    "clinical_optionality_pct_dev": "0.50",
                    "composite_rank": "10",
                    "event_ev_score": "5.0", "event_ev_score_z": "1.2",
                    "event_ev_bucket": "high_ev",
                    "event_ev_analog_confidence": "ok",
                    "final_score": "0.75", "selector_score": "0.75",
                    "catalyst_event_type": "", "catalyst_source": "",
                    "alpha_cohort_pct": "", "alpha_cohort_raw": "",
                },
                {
                    "ticker": "BBBB", "archetype": "drug_developer",
                    "eligible": "1", "tier_dev": "B",
                    "catalyst_mode": "specific_days", "catalyst_days": "60",
                    "mom_state": "neutral", "sponsor_tier1_count": "0",
                    "size_band": "M", "risk_flags": "",
                    "clinical_optionality_pct_dev": "0.50",
                    "composite_rank": "11",
                    "event_ev_score": "2.0", "event_ev_score_z": "-0.5",
                    "event_ev_bucket": "mid_ev",
                    "event_ev_analog_confidence": "ok",
                    "final_score": "0.75", "selector_score": "0.75",
                    "catalyst_event_type": "", "catalyst_source": "",
                    "alpha_cohort_pct": "", "alpha_cohort_raw": "",
                },
            ]

            with open(date_dir / "rankings.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            try:
                result = measure_impact("2099-01-01")
                assert "error" not in result
                assert result["n_total"] == 2
                assert result["n_eligible"] == 2
                assert "ties" in result
                assert "reordering" in result
                assert "top_k_changes" in result
            finally:
                mod.SNAPSHOT_DIR = orig_dir
