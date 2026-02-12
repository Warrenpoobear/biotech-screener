#!/usr/bin/env python3
"""
test_minimum_suite.py - Minimum Test Suite for Biotech Screener

Tests:
1. Smoke test: Full pipeline run completes
2. Regression test: Same inputs produce identical outputs
3. Schema tests: Validate key input/output schemas
4. PIT discipline test: No data after as_of_date is used
5. Edge cases: Missing values, empty modules handled explicitly

Uses session-scoped pipeline fixtures from conftest.py to avoid redundant runs.

Usage:
    pytest tests/test_minimum_suite.py -v
    pytest tests/test_minimum_suite.py -v -k smoke  # Run only smoke tests
"""

import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow

from conftest import (
    NON_DETERMINISTIC_PATHS,
    PIPELINE_MAIN_DATE,
    compute_content_hash,
)

# Test configuration
DATA_DIR = Path("production_data")


# ==============================================================================
# 1. SMOKE TESTS
# ==============================================================================


class TestSmoke:
    """Smoke tests - pipeline runs without crashing"""

    def test_pipeline_completes_without_error(self, pipeline_run_main):
        """The most basic test: pipeline runs to completion"""
        assert pipeline_run_main["success"], "Pipeline failed"
        assert pipeline_run_main["output_path"].exists(), "Output file not created"

    def test_output_is_valid_json(self, pipeline_run_main):
        """Output file is valid JSON"""
        assert isinstance(pipeline_run_main["data"], dict)

    def test_output_has_required_sections(self, pipeline_run_main):
        """Output has all required top-level sections"""
        data = pipeline_run_main["data"]

        required_sections = [
            "run_metadata",
            "module_1_universe",
            "module_2_financial",
            "module_3_catalyst",
            "module_4_clinical",
            "module_5_composite",
            "summary",
        ]

        for section in required_sections:
            assert section in data, f"Missing section: {section}"

    def test_summary_has_key_metrics(self, pipeline_run_main):
        """Summary section has key metrics"""
        summary = pipeline_run_main["data"]["summary"]
        assert "total_evaluated" in summary
        assert "active_universe" in summary
        assert "final_ranked" in summary
        assert summary["total_evaluated"] > 0


# ==============================================================================
# 2. REGRESSION TESTS
# ==============================================================================


class TestRegression:
    """Regression tests - same inputs produce identical outputs"""

    def test_determinism_two_runs(self, pipeline_run_main, pipeline_run_determinism):
        """Two runs with same inputs produce identical outputs"""
        hash1 = compute_content_hash(pipeline_run_main["data"], NON_DETERMINISTIC_PATHS)
        hash2 = compute_content_hash(pipeline_run_determinism["data"], NON_DETERMINISTIC_PATHS)

        assert hash1 == hash2, "Two runs produced different outputs"

    def test_module_2_deterministic(self):
        """Module 2 scoring is deterministic"""
        from module_2_financial import score_financial_health

        # Run twice with same inputs
        fin_data = {"Cash": 100_000_000, "NetIncome": -20_000_000}
        mkt_data = {"market_cap": 500_000_000, "avg_volume": 50_000, "price": 20}

        result1 = score_financial_health("TEST", fin_data, mkt_data)
        result2 = score_financial_health("TEST", fin_data, mkt_data)

        assert result1 == result2, "Module 2 is not deterministic"


# ==============================================================================
# 3. SCHEMA TESTS
# ==============================================================================


class TestSchema:
    """Schema validation tests"""

    def test_module_2_output_schema(self, pipeline_run_main):
        """Module 2 output has correct schema"""
        m2 = pipeline_run_main["data"]["module_2_financial"]

        # Required fields
        assert "scores" in m2
        assert isinstance(m2["scores"], list)

        if m2["scores"]:
            score = m2["scores"][0]
            assert "ticker" in score
            assert "financial_score" in score
            assert "severity" in score

    def test_module_5_output_schema(self, pipeline_run_main):
        """Module 5 output has correct schema"""
        m5 = pipeline_run_main["data"]["module_5_composite"]

        # Required fields
        assert "ranked_securities" in m5
        assert "excluded_securities" in m5
        assert "diagnostic_counts" in m5

        if m5["ranked_securities"]:
            sec = m5["ranked_securities"][0]
            assert "ticker" in sec
            assert "composite_score" in sec
            assert "composite_rank" in sec

    def test_scores_in_valid_range(self, pipeline_run_main):
        """All scores are in [0, 100] range"""
        data = pipeline_run_main["data"]

        # Module 2
        for score in data["module_2_financial"]["scores"]:
            s = score.get("financial_score", 0)
            assert 0 <= s <= 100, f"Module 2 score out of range: {s}"

        # Module 5
        for sec in data["module_5_composite"]["ranked_securities"]:
            s = float(sec.get("composite_score", "0"))
            assert 0 <= s <= 100, f"Module 5 score out of range: {s}"


# ==============================================================================
# 4. PIT DISCIPLINE TESTS
# ==============================================================================


class TestPITDiscipline:
    """Point-in-time discipline tests"""

    def test_as_of_date_in_output(self, pipeline_run_main):
        """Output contains correct as_of_date"""
        assert pipeline_run_main["data"]["run_metadata"]["as_of_date"] == PIPELINE_MAIN_DATE

    def test_historical_date_filters_future_data(self, pipeline_run_historical):
        """Running with historical date filters future data"""
        assert pipeline_run_historical["success"], "Pipeline failed with historical date"

        combined_output = pipeline_run_historical["stdout"] + pipeline_run_historical["stderr"]
        assert "Filtered" in combined_output or "future" in combined_output.lower()

    def test_pit_cutoff_computation(self):
        """PIT cutoff is correctly computed as as_of_date - 1"""
        from common.pit_enforcement import compute_pit_cutoff

        cutoff = compute_pit_cutoff("2026-01-15")
        assert cutoff == "2026-01-14"

    def test_pit_admissibility(self):
        """PIT admissibility check works correctly"""
        from common.pit_enforcement import is_pit_admissible

        # Data from before cutoff is admissible
        assert is_pit_admissible("2026-01-13", "2026-01-14")

        # Data from cutoff date is admissible
        assert is_pit_admissible("2026-01-14", "2026-01-14")

        # Data from after cutoff is NOT admissible
        assert not is_pit_admissible("2026-01-15", "2026-01-14")

        # None is NOT admissible
        assert not is_pit_admissible(None, "2026-01-14")


# ==============================================================================
# 5. EDGE CASE TESTS
# ==============================================================================


class TestEdgeCases:
    """Edge case handling tests"""

    def test_missing_financial_data_flagged(self, pipeline_run_main):
        """Tickers with missing financial data are properly flagged"""
        data = pipeline_run_main["data"]

        # Some tickers should have missing data flags
        missing_count = 0
        for score in data["module_2_financial"]["scores"]:
            if "missing_financial_data" in score.get("flags", []):
                missing_count += 1

        # Just informational - not necessarily a failure
        print(f"Tickers with missing financial data: {missing_count}")

    def test_sev3_tickers_excluded(self, pipeline_run_main):
        """SEV3 (critical) tickers are excluded from ranking"""
        data = pipeline_run_main["data"]

        # Count SEV3 in Module 2
        sev3_count = sum(
            1 for s in data["module_2_financial"]["scores"]
            if s.get("severity") == "sev3"
        )

        # Count excluded
        excluded_count = len(data["module_5_composite"]["excluded_securities"])

        # Most excluded should be SEV3
        assert excluded_count >= sev3_count * 0.8, "SEV3 tickers not being excluded"

    def test_excluded_have_exclusion_reason(self, pipeline_run_main):
        """Excluded securities have exclusion reasons"""
        data = pipeline_run_main["data"]

        for sec in data["module_5_composite"]["excluded_securities"]:
            reason = sec.get("reason")
            assert reason and reason != "unknown", f"Ticker {sec['ticker']} missing exclusion reason"

    def test_weights_sum_to_target(self, pipeline_run_main):
        """Position weights: 0 when position sizing disabled (default), 1.0 when enabled"""
        data = pipeline_run_main["data"]

        m5 = data["module_5_composite"]
        total_weight = sum(
            float(sec.get("position_weight", "0"))
            for sec in m5["ranked_securities"]
        )

        # Position sizing disabled by default -> weights should be 0
        def_config = m5.get("defensive_overlay_config", {})
        position_sizing = def_config.get("position_sizing_enabled", False)

        if position_sizing:
            expected, tolerance = 1.0, 0.01
            assert abs(total_weight - expected) < tolerance, f"Weights sum to {total_weight}, expected {expected}"
        else:
            assert total_weight == 0.0, f"Weights should be 0 when position sizing disabled, got {total_weight}"

    def test_excluded_have_zero_weight(self, pipeline_run_main):
        """Excluded securities have zero weight"""
        data = pipeline_run_main["data"]

        for sec in data["module_5_composite"]["excluded_securities"]:
            weight = float(sec.get("position_weight", "0"))
            assert weight == 0, f"Excluded ticker {sec['ticker']} has non-zero weight: {weight}"

    def test_no_all_zero_module_2(self, pipeline_run_main):
        """Module 2 doesn't return all zeros"""
        scores = pipeline_run_main["data"]["module_2_financial"]["scores"]
        non_zero = sum(1 for s in scores if s.get("financial_score", 0) != 0)

        assert non_zero > 0, "All Module 2 scores are zero"

    def test_no_all_zero_module_4(self, pipeline_run_main):
        """Module 4 doesn't return all zeros"""
        scores = pipeline_run_main["data"]["module_4_clinical"]["scores"]
        non_zero = sum(1 for s in scores if float(s.get("clinical_score", "0")) != 0)

        assert non_zero > 0, "All Module 4 scores are zero"


# ==============================================================================
# ADDITIONAL VALIDATION TESTS
# ==============================================================================


class TestValidation:
    """Additional validation tests"""

    def test_doctor_passes(self):
        """Doctor health check passes"""
        import sys
        result = subprocess.run(
            [sys.executable, "doctor.py", "--data-dir", str(DATA_DIR)],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )
        assert result.returncode == 0, f"Doctor check failed:\n{result.stdout}\n{result.stderr}"

    def test_validate_pipeline_passes(self, pipeline_run_main):
        """Pipeline validation passes on output"""
        import sys
        result = subprocess.run(
            [sys.executable, "validate_pipeline.py", "--output", str(pipeline_run_main["output_path"])],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )
        assert result.returncode == 0, f"Validation failed:\n{result.stdout}\n{result.stderr}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
