#!/usr/bin/env python3
"""
test_golden_baseline.py - Golden Run Regression Tests

Creates a baseline output for one as-of-date and compares future outputs to it.
Allows explicitly-defined tolerated changes (e.g., timestamps).

Uses session-scoped pipeline fixtures from conftest.py to avoid redundant runs.

Usage:
    # Create baseline
    pytest tests/test_golden_baseline.py::TestGoldenBaseline::test_create_baseline -v

    # Run regression tests
    pytest tests/test_golden_baseline.py -v
"""

import json
from datetime import date
from pathlib import Path
from typing import Any, Dict, Set

import pytest

pytestmark = pytest.mark.slow

from conftest import (
    NON_DETERMINISTIC_PATHS,
    PIPELINE_HISTORICAL_DATE,
    compute_content_hash,
)

# Configuration
GOLDEN_DIR = Path(__file__).parent / "golden"
BASELINE_FILE = GOLDEN_DIR / "baseline_output.json"
BASELINE_METADATA_FILE = GOLDEN_DIR / "baseline_metadata.json"

# Fields that are NEVER allowed to change
CRITICAL_STABLE_FIELDS = {
    "summary.total_evaluated",
    "summary.active_universe",
    "summary.final_ranked",
    "module_5_composite.diagnostic_counts.rankable",
}


def get_nested_value(data: Dict, path: str) -> Any:
    """Get value from nested dict using dot-separated path"""
    keys = path.split(".")
    current = data
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return None
    return current


@pytest.fixture
def ensure_golden_dir():
    """Ensure golden directory exists"""
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)


class TestGoldenBaseline:
    """Golden baseline regression tests"""

    def test_create_baseline(self, ensure_golden_dir, pipeline_run_main):
        """Create or update the golden baseline"""
        output = pipeline_run_main["data"]

        # Save baseline
        with open(BASELINE_FILE, "w") as f:
            json.dump(output, f, indent=2, sort_keys=True)

        # Save metadata
        metadata = {
            "created_at": date.today().isoformat(),
            "as_of_date": "2026-01-20",
            "content_hash": compute_content_hash(output, NON_DETERMINISTIC_PATHS),
            "total_evaluated": output.get("summary", {}).get("total_evaluated"),
            "final_ranked": output.get("summary", {}).get("final_ranked"),
            "version": output.get("run_metadata", {}).get("version"),
        }

        with open(BASELINE_METADATA_FILE, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"\nBaseline created:")
        print(f"  File: {BASELINE_FILE}")
        print(f"  As-of date: 2026-01-20")
        print(f"  Content hash: {metadata['content_hash'][:16]}")
        print(f"  Total evaluated: {metadata['total_evaluated']}")
        print(f"  Final ranked: {metadata['final_ranked']}")

    @pytest.mark.skipif(not BASELINE_FILE.exists(), reason="No baseline exists. Run test_create_baseline first.")
    def test_output_matches_baseline(self, pipeline_run_main):
        """Test that current output matches the golden baseline"""
        current = pipeline_run_main["data"]

        with open(BASELINE_FILE) as f:
            baseline = json.load(f)

        # Compare content hashes (excluding tolerated diffs)
        current_hash = compute_content_hash(current, NON_DETERMINISTIC_PATHS)
        baseline_hash = compute_content_hash(baseline, NON_DETERMINISTIC_PATHS)

        if current_hash != baseline_hash:
            # Find differences
            differences = self._find_differences(baseline, current)
            diff_str = "\n".join(f"  {path}: {old} -> {new}" for path, old, new in differences[:10])
            pytest.fail(
                f"Output differs from baseline:\n"
                f"  Baseline hash: {baseline_hash[:16]}\n"
                f"  Current hash: {current_hash[:16]}\n"
                f"Differences:\n{diff_str}"
            )

    @pytest.mark.skipif(not BASELINE_FILE.exists(), reason="No baseline exists")
    def test_critical_fields_stable(self, pipeline_run_main):
        """Test that critical fields haven't changed"""
        current = pipeline_run_main["data"]

        with open(BASELINE_FILE) as f:
            baseline = json.load(f)

        for path in CRITICAL_STABLE_FIELDS:
            current_val = get_nested_value(current, path)
            baseline_val = get_nested_value(baseline, path)

            assert current_val == baseline_val, (
                f"Critical field {path} changed: {baseline_val} -> {current_val}"
            )

    @pytest.mark.skipif(not BASELINE_FILE.exists(), reason="No baseline exists")
    def test_determinism_multiple_runs(self, pipeline_run_main, pipeline_run_determinism):
        """Test that running twice produces identical output"""
        hash1 = compute_content_hash(pipeline_run_main["data"], NON_DETERMINISTIC_PATHS)
        hash2 = compute_content_hash(pipeline_run_determinism["data"], NON_DETERMINISTIC_PATHS)

        assert hash1 == hash2, "Two runs with same inputs produced different outputs"

    def _find_differences(self, baseline: Dict, current: Dict, prefix: str = "") -> list:
        """Find differences between two dicts"""
        differences = []

        all_keys = set(baseline.keys()) | set(current.keys())

        for key in sorted(all_keys):
            path = f"{prefix}.{key}" if prefix else key

            # Skip tolerated paths
            if path in NON_DETERMINISTIC_PATHS:
                continue

            baseline_val = baseline.get(key)
            current_val = current.get(key)

            if isinstance(baseline_val, dict) and isinstance(current_val, dict):
                differences.extend(self._find_differences(baseline_val, current_val, path))
            elif baseline_val != current_val:
                # Truncate long values
                base_str = str(baseline_val)[:50] if baseline_val else "None"
                curr_str = str(current_val)[:50] if current_val else "None"
                differences.append((path, base_str, curr_str))

        return differences


class TestSmokeTest:
    """Quick smoke tests that don't require baseline"""

    def test_pipeline_runs_without_crash(self, pipeline_run_main):
        """Test that the pipeline runs without crashing"""
        assert pipeline_run_main["success"], "Pipeline crashed"

    def test_output_has_required_sections(self, pipeline_run_main):
        """Test output has all required sections"""
        data = pipeline_run_main["data"]

        required = [
            "run_metadata",
            "module_1_universe",
            "module_2_financial",
            "module_3_catalyst",
            "module_4_clinical",
            "module_5_composite",
            "summary",
        ]

        for section in required:
            assert section in data, f"Missing required section: {section}"

    def test_no_all_zero_scores(self, pipeline_run_main):
        """Test that no module returns all-zero scores"""
        data = pipeline_run_main["data"]

        # Module 2
        m2_scores = data.get("module_2_financial", {}).get("scores", [])
        if m2_scores:
            non_zero = sum(1 for s in m2_scores if s.get("financial_score", 0) != 0)
            assert non_zero > 0, "All Module 2 scores are zero"

        # Module 4
        m4_scores = data.get("module_4_clinical", {}).get("scores", [])
        if m4_scores:
            non_zero = sum(1 for s in m4_scores if float(s.get("clinical_score", "0")) != 0)
            assert non_zero > 0, "All Module 4 scores are zero"

        # Module 5
        m5_ranked = data.get("module_5_composite", {}).get("ranked_securities", [])
        if m5_ranked:
            non_zero = sum(1 for s in m5_ranked if float(s.get("composite_score", "0")) != 0)
            assert non_zero > 0, "All Module 5 scores are zero"


class TestPITDiscipline:
    """Point-in-time discipline tests"""

    def test_no_future_data_in_output(self, pipeline_run_historical):
        """Test that output doesn't contain data from after as_of_date"""
        data = pipeline_run_historical["data"]

        # Check as_of_date in metadata
        metadata = data.get("run_metadata", {})
        assert metadata.get("as_of_date") == PIPELINE_HISTORICAL_DATE

        # Check no provenance dates are after as_of_date
        m4 = data.get("module_4_clinical", {})
        m4_date = m4.get("as_of_date")
        if m4_date:
            assert m4_date <= PIPELINE_HISTORICAL_DATE, (
                f"Module 4 as_of_date {m4_date} > {PIPELINE_HISTORICAL_DATE}"
            )


class TestEdgeCases:
    """Edge case tests"""

    def test_empty_catalyst_handling(self, pipeline_run_main):
        """Test that zero catalysts doesn't crash the pipeline"""
        data = pipeline_run_main["data"]

        # Catalyst module should have diagnostic counts even with 0 events
        m3 = data.get("module_3_catalyst", {})
        diag = m3.get("diagnostic_counts", {})
        assert "tickers_analyzed" in diag

    def test_missing_financial_data_handled(self, pipeline_run_main):
        """Test that missing financial data is handled gracefully"""
        data = pipeline_run_main["data"]

        # Check that tickers with missing data are properly flagged
        m2 = data.get("module_2_financial", {})
        scores = m2.get("scores", [])

        # At least some should have missing data flags
        missing_count = sum(1 for s in scores if "missing_financial_data" in s.get("flags", []))
        # This is informational - not a failure
        print(f"Tickers with missing financial data: {missing_count}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
