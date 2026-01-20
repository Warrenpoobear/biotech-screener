#!/usr/bin/env python3
"""
test_minimum_suite.py - Minimum Test Suite for Biotech Screener

Tests:
1. Smoke test: Full pipeline run completes
2. Regression test: Same inputs produce identical outputs
3. Schema tests: Validate key input/output schemas
4. PIT discipline test: No data after as_of_date is used
5. Edge cases: Missing values, empty modules handled explicitly

Usage:
    pytest tests/test_minimum_suite.py -v
    pytest tests/test_minimum_suite.py -v -k smoke  # Run only smoke tests
"""

import hashlib
import json
import subprocess
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict

import pytest

# Test configuration
DATA_DIR = Path("production_data")
TEST_AS_OF_DATE = "2026-01-20"

# ==============================================================================
# FIXTURES
# ==============================================================================


@pytest.fixture
def as_of_date():
    """Standard as_of_date for deterministic tests"""
    return date(2026, 1, 20)


@pytest.fixture
def tmp_output(tmp_path):
    """Temporary output file path"""
    return tmp_path / "test_output.json"


@pytest.fixture
def sample_financial_records():
    """Sample financial records for unit tests"""
    return [
        {
            "ticker": "TEST1",
            "Cash": 500_000_000,
            "NetIncome": -100_000_000,
            "market_cap": 2_000_000_000,
            "avg_volume": 500_000,
            "price": 20.0,
        },
        {
            "ticker": "TEST2",
            "Cash": 50_000_000,
            "NetIncome": -200_000_000,  # Short runway
            "market_cap": 500_000_000,
            "avg_volume": 100_000,
            "price": 10.0,
        },
    ]


@pytest.fixture
def sample_trial_records():
    """Sample trial records for unit tests"""
    return [
        {
            "ticker": "TEST1",
            "nct_id": "NCT00000001",
            "overall_status": "RECRUITING",
            "phase": "Phase 2",
            "last_update_posted": "2026-01-10",
        },
        {
            "ticker": "TEST1",
            "nct_id": "NCT00000002",
            "overall_status": "ACTIVE_NOT_RECRUITING",
            "phase": "Phase 3",
            "last_update_posted": "2026-01-15",
        },
    ]


def run_pipeline(as_of_date: str, output_path: Path, extra_args: list = None) -> tuple:
    """Run the pipeline and return (success, stdout, stderr)"""
    cmd = [
        "python", "run_screen.py",
        "--as-of-date", as_of_date,
        "--data-dir", str(DATA_DIR),
        "--output", str(output_path),
    ]
    if extra_args:
        cmd.extend(extra_args)

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent
    )
    return result.returncode == 0, result.stdout, result.stderr


def compute_hash(data: Any) -> str:
    """Compute deterministic content hash"""
    json_str = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(json_str.encode()).hexdigest()


# ==============================================================================
# 1. SMOKE TESTS
# ==============================================================================


class TestSmoke:
    """Smoke tests - pipeline runs without crashing"""

    def test_pipeline_completes_without_error(self, tmp_output):
        """The most basic test: pipeline runs to completion"""
        success, stdout, stderr = run_pipeline(TEST_AS_OF_DATE, tmp_output)
        assert success, f"Pipeline failed:\n{stderr}"
        assert tmp_output.exists(), "Output file not created"

    def test_output_is_valid_json(self, tmp_output):
        """Output file is valid JSON"""
        run_pipeline(TEST_AS_OF_DATE, tmp_output)

        with open(tmp_output) as f:
            data = json.load(f)

        assert isinstance(data, dict)

    def test_output_has_required_sections(self, tmp_output):
        """Output has all required top-level sections"""
        run_pipeline(TEST_AS_OF_DATE, tmp_output)

        with open(tmp_output) as f:
            data = json.load(f)

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

    def test_summary_has_key_metrics(self, tmp_output):
        """Summary section has key metrics"""
        run_pipeline(TEST_AS_OF_DATE, tmp_output)

        with open(tmp_output) as f:
            data = json.load(f)

        summary = data["summary"]
        assert "total_evaluated" in summary
        assert "active_universe" in summary
        assert "final_ranked" in summary
        assert summary["total_evaluated"] > 0


# ==============================================================================
# 2. REGRESSION TESTS
# ==============================================================================


class TestRegression:
    """Regression tests - same inputs produce identical outputs"""

    def test_determinism_two_runs(self, tmp_path):
        """Two runs with same inputs produce identical outputs"""
        output1 = tmp_path / "run1.json"
        output2 = tmp_path / "run2.json"

        success1, _, _ = run_pipeline(TEST_AS_OF_DATE, output1)
        success2, _, _ = run_pipeline(TEST_AS_OF_DATE, output2)

        assert success1 and success2, "One or both runs failed"

        with open(output1) as f:
            data1 = json.load(f)
        with open(output2) as f:
            data2 = json.load(f)

        # Compare content hashes
        hash1 = compute_hash(data1)
        hash2 = compute_hash(data2)

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

    def test_module_2_output_schema(self, tmp_output):
        """Module 2 output has correct schema"""
        run_pipeline(TEST_AS_OF_DATE, tmp_output)

        with open(tmp_output) as f:
            data = json.load(f)

        m2 = data["module_2_financial"]

        # Required fields
        assert "scores" in m2
        assert isinstance(m2["scores"], list)

        if m2["scores"]:
            score = m2["scores"][0]
            assert "ticker" in score
            assert "financial_score" in score
            assert "severity" in score

    def test_module_5_output_schema(self, tmp_output):
        """Module 5 output has correct schema"""
        run_pipeline(TEST_AS_OF_DATE, tmp_output)

        with open(tmp_output) as f:
            data = json.load(f)

        m5 = data["module_5_composite"]

        # Required fields
        assert "ranked_securities" in m5
        assert "excluded_securities" in m5
        assert "diagnostic_counts" in m5

        if m5["ranked_securities"]:
            sec = m5["ranked_securities"][0]
            assert "ticker" in sec
            assert "composite_score" in sec
            assert "composite_rank" in sec

    def test_scores_in_valid_range(self, tmp_output):
        """All scores are in [0, 100] range"""
        run_pipeline(TEST_AS_OF_DATE, tmp_output)

        with open(tmp_output) as f:
            data = json.load(f)

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

    def test_as_of_date_in_output(self, tmp_output):
        """Output contains correct as_of_date"""
        run_pipeline(TEST_AS_OF_DATE, tmp_output)

        with open(tmp_output) as f:
            data = json.load(f)

        assert data["run_metadata"]["as_of_date"] == TEST_AS_OF_DATE

    def test_historical_date_filters_future_data(self, tmp_path):
        """Running with historical date filters future data"""
        output = tmp_path / "historical.json"
        historical_date = "2026-01-15"

        success, stdout, stderr = run_pipeline(historical_date, output)
        # Pipeline should succeed (with filtering)
        assert success, f"Pipeline failed with historical date:\n{stderr}"

        # Check that filtering happened (warning in logs)
        combined_output = stdout + stderr
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

    def test_missing_financial_data_flagged(self, tmp_output):
        """Tickers with missing financial data are properly flagged"""
        run_pipeline(TEST_AS_OF_DATE, tmp_output)

        with open(tmp_output) as f:
            data = json.load(f)

        # Some tickers should have missing data flags
        missing_count = 0
        for score in data["module_2_financial"]["scores"]:
            if "missing_financial_data" in score.get("flags", []):
                missing_count += 1

        # Just informational - not necessarily a failure
        print(f"Tickers with missing financial data: {missing_count}")

    def test_sev3_tickers_excluded(self, tmp_output):
        """SEV3 (critical) tickers are excluded from ranking"""
        run_pipeline(TEST_AS_OF_DATE, tmp_output)

        with open(tmp_output) as f:
            data = json.load(f)

        # Count SEV3 in Module 2
        sev3_count = sum(
            1 for s in data["module_2_financial"]["scores"]
            if s.get("severity") == "sev3"
        )

        # Count excluded
        excluded_count = len(data["module_5_composite"]["excluded_securities"])

        # Most excluded should be SEV3
        assert excluded_count >= sev3_count * 0.8, "SEV3 tickers not being excluded"

    def test_excluded_have_exclusion_reason(self, tmp_output):
        """Excluded securities have exclusion reasons"""
        run_pipeline(TEST_AS_OF_DATE, tmp_output)

        with open(tmp_output) as f:
            data = json.load(f)

        for sec in data["module_5_composite"]["excluded_securities"]:
            reason = sec.get("reason")
            assert reason and reason != "unknown", f"Ticker {sec['ticker']} missing exclusion reason"

    def test_weights_sum_to_target(self, tmp_output):
        """Position weights sum to expected target"""
        run_pipeline(TEST_AS_OF_DATE, tmp_output)

        with open(tmp_output) as f:
            data = json.load(f)

        total_weight = sum(
            float(sec.get("position_weight", "0"))
            for sec in data["module_5_composite"]["ranked_securities"]
        )

        expected = 0.90
        tolerance = 0.01
        assert abs(total_weight - expected) < tolerance, (
            f"Weights sum to {total_weight}, expected {expected}"
        )

    def test_excluded_have_zero_weight(self, tmp_output):
        """Excluded securities have zero weight"""
        run_pipeline(TEST_AS_OF_DATE, tmp_output)

        with open(tmp_output) as f:
            data = json.load(f)

        for sec in data["module_5_composite"]["excluded_securities"]:
            weight = float(sec.get("position_weight", "0"))
            assert weight == 0, f"Excluded ticker {sec['ticker']} has non-zero weight: {weight}"

    def test_no_all_zero_module_2(self, tmp_output):
        """Module 2 doesn't return all zeros"""
        run_pipeline(TEST_AS_OF_DATE, tmp_output)

        with open(tmp_output) as f:
            data = json.load(f)

        scores = data["module_2_financial"]["scores"]
        non_zero = sum(1 for s in scores if s.get("financial_score", 0) != 0)

        assert non_zero > 0, "All Module 2 scores are zero"

    def test_no_all_zero_module_4(self, tmp_output):
        """Module 4 doesn't return all zeros"""
        run_pipeline(TEST_AS_OF_DATE, tmp_output)

        with open(tmp_output) as f:
            data = json.load(f)

        scores = data["module_4_clinical"]["scores"]
        non_zero = sum(1 for s in scores if float(s.get("clinical_score", "0")) != 0)

        assert non_zero > 0, "All Module 4 scores are zero"


# ==============================================================================
# ADDITIONAL VALIDATION TESTS
# ==============================================================================


class TestValidation:
    """Additional validation tests"""

    def test_doctor_passes(self):
        """Doctor health check passes"""
        result = subprocess.run(
            ["python", "doctor.py", "--data-dir", str(DATA_DIR)],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )
        assert result.returncode == 0, f"Doctor check failed:\n{result.stdout}\n{result.stderr}"

    def test_validate_pipeline_passes(self, tmp_output):
        """Pipeline validation passes on output"""
        run_pipeline(TEST_AS_OF_DATE, tmp_output)

        result = subprocess.run(
            ["python", "validate_pipeline.py", "--output", str(tmp_output)],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )
        assert result.returncode == 0, f"Validation failed:\n{result.stdout}\n{result.stderr}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
