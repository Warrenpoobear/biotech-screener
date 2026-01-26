#!/usr/bin/env python3
"""
Tests for common/run_summary.py

Run summary artifact generation for backtest quality gates.
Tests cover:
- compute_coverage_stats (coverage statistics)
- compute_fallback_stats (cohort fallback statistics)
- extract_headline_metrics (IC, spread, monotonicity)
- generate_run_summary (full summary generation)
- print_run_summary (console output)
"""

import json
import pytest
import tempfile
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from common.run_summary import (
    compute_coverage_stats,
    compute_fallback_stats,
    extract_headline_metrics,
    generate_run_summary,
    print_run_summary,
)


class TestComputeCoverageStats:
    """Tests for compute_coverage_stats function."""

    def test_empty_snapshots(self):
        """Should handle empty snapshots."""
        result = compute_coverage_stats([], {})

        assert result["total_securities"] == 0
        assert result["rankable"] == 0
        assert result["with_returns"] == 0
        assert result["rankable_rate"] == 0
        assert result["return_coverage_rate"] == 0

    def test_counts_total_securities(self):
        """Should count ranked + excluded as total."""
        snapshots = [
            {
                "as_of_date": "2024-01-01",
                "ranked_securities": [{"ticker": "A"}, {"ticker": "B"}],
                "excluded_securities": [{"ticker": "C"}],
            }
        ]

        result = compute_coverage_stats(snapshots, {})

        assert result["total_securities"] == 3
        assert result["rankable"] == 2

    def test_counts_with_returns(self):
        """Should count securities with returns."""
        snapshots = [
            {
                "as_of_date": "2024-01-01",
                "ranked_securities": [
                    {"ticker": "A"},
                    {"ticker": "B"},
                    {"ticker": "C"},
                ],
                "excluded_securities": [],
            }
        ]
        returns_by_date = {
            "2024-01-01": {
                "A": "0.10",
                "B": "0.05",
                "C": None,  # No return
            }
        }

        result = compute_coverage_stats(snapshots, returns_by_date)

        assert result["with_returns"] == 2
        assert result["return_coverage_rate"] == round(2/3, 4)

    def test_computes_rankable_rate(self):
        """Should compute rankable rate correctly."""
        snapshots = [
            {
                "as_of_date": "2024-01-01",
                "ranked_securities": [{"ticker": "A"}],
                "excluded_securities": [{"ticker": "B"}, {"ticker": "C"}, {"ticker": "D"}],
            }
        ]

        result = compute_coverage_stats(snapshots, {})

        assert result["rankable_rate"] == 0.25  # 1 out of 4

    def test_multiple_snapshots(self):
        """Should aggregate across multiple snapshots."""
        snapshots = [
            {
                "as_of_date": "2024-01-01",
                "ranked_securities": [{"ticker": "A"}],
                "excluded_securities": [],
            },
            {
                "as_of_date": "2024-01-02",
                "ranked_securities": [{"ticker": "B"}, {"ticker": "C"}],
                "excluded_securities": [],
            },
        ]

        result = compute_coverage_stats(snapshots, {})

        assert result["total_securities"] == 3
        assert result["rankable"] == 3


class TestComputeFallbackStats:
    """Tests for compute_fallback_stats function."""

    def test_empty_snapshots(self):
        """Should handle empty snapshots."""
        result = compute_fallback_stats([])

        assert result["total_securities"] == 0
        assert result["normal_securities"] == 0
        assert result["fallback_securities"] == 0
        assert result["fallback_rate"] == 0

    def test_counts_normal_vs_fallback(self):
        """Should separate normal and fallback securities."""
        snapshots = [
            {
                "cohort_mode": "stage_only",
                "cohort_stats": {
                    "phase_3": {"count": 10, "normalization_fallback": "normal"},
                    "phase_2": {"count": 5, "normalization_fallback": "stage_only"},
                },
            }
        ]

        result = compute_fallback_stats(snapshots)

        assert result["total_securities"] == 15
        assert result["normal_securities"] == 10
        assert result["fallback_securities"] == 5
        assert result["fallback_rate"] == round(5/15, 4)

    def test_tracks_fallback_reasons(self):
        """Should track fallback reasons."""
        snapshots = [
            {
                "cohort_mode": "stage_only",
                "cohort_stats": {
                    "phase_3": {"count": 10, "normalization_fallback": "normal"},
                    "phase_2": {"count": 5, "normalization_fallback": "stage_only"},
                    "phase_1": {"count": 3, "normalization_fallback": "global"},
                },
            }
        ]

        result = compute_fallback_stats(snapshots)

        assert "stage_only" in result["fallback_reasons"]
        assert result["fallback_reasons"]["stage_only"] == 5
        assert result["fallback_reasons"]["global"] == 3

    def test_tracks_cohort_sizes(self):
        """Should track cohort size ranges."""
        snapshots = [
            {
                "cohort_mode": "stage_only",
                "cohort_stats": {
                    "phase_3": {"count": 10, "normalization_fallback": "normal"},
                },
            },
            {
                "cohort_mode": "stage_only",
                "cohort_stats": {
                    "phase_3": {"count": 15, "normalization_fallback": "normal"},
                },
            },
        ]

        result = compute_fallback_stats(snapshots)

        assert "phase_3" in result["cohort_size_ranges"]
        assert result["cohort_size_ranges"]["phase_3"]["min"] == 10
        assert result["cohort_size_ranges"]["phase_3"]["max"] == 15

    def test_identifies_stages_below_min(self):
        """Should identify stages below minimum size."""
        snapshots = [
            {
                "cohort_mode": "stage_only",
                "cohort_stats": {
                    "phase_3": {"count": 3, "normalization_fallback": "normal"},  # Below 5
                    "phase_2": {"count": 10, "normalization_fallback": "normal"},
                },
            }
        ]

        result = compute_fallback_stats(snapshots)

        assert "phase_3" in result["stages_below_min"]
        assert "phase_2" not in result["stages_below_min"]

    def test_min_stage_size_threshold(self):
        """Should include min stage size threshold."""
        result = compute_fallback_stats([])
        assert result["min_stage_size_threshold"] == 5


class TestExtractHeadlineMetrics:
    """Tests for extract_headline_metrics function."""

    def test_extracts_all_horizons(self):
        """Should extract metrics for all horizons."""
        backtest_result = {
            "aggregate_metrics": {
                "63d": {
                    "ic_mean": "0.05",
                    "ic_pos_frac": "0.60",
                    "bucket_spread_mean": "0.10",
                    "monotonicity_rate": "0.70",
                    "n_periods": 10,
                },
                "126d": {
                    "ic_mean": "0.08",
                    "ic_pos_frac": "0.65",
                    "bucket_spread_mean": "0.15",
                    "monotonicity_rate": "0.75",
                    "n_periods": 8,
                },
                "252d": {
                    "ic_mean": "0.12",
                    "ic_pos_frac": "0.70",
                    "bucket_spread_mean": "0.20",
                    "monotonicity_rate": "0.80",
                    "n_periods": 5,
                },
            }
        }

        result = extract_headline_metrics(backtest_result)

        assert "63d" in result
        assert "126d" in result
        assert "252d" in result

    def test_extracts_ic_mean(self):
        """Should extract IC mean."""
        backtest_result = {
            "aggregate_metrics": {
                "63d": {
                    "ic_mean": "0.05",
                }
            }
        }

        result = extract_headline_metrics(backtest_result)

        assert result["63d"]["ic_mean"] == 0.05

    def test_handles_missing_horizons(self):
        """Should handle missing horizons gracefully."""
        backtest_result = {
            "aggregate_metrics": {
                "63d": {"ic_mean": "0.05"},
                # 126d and 252d missing
            }
        }

        result = extract_headline_metrics(backtest_result)

        assert "63d" in result
        assert "126d" not in result
        assert "252d" not in result

    def test_handles_none_values(self):
        """Should handle None values."""
        backtest_result = {
            "aggregate_metrics": {
                "63d": {
                    "ic_mean": None,
                    "ic_pos_frac": None,
                    "bucket_spread_mean": None,
                    "monotonicity_rate": None,
                    "n_periods": 0,
                }
            }
        }

        result = extract_headline_metrics(backtest_result)

        assert result["63d"]["ic_mean"] is None
        assert result["63d"]["bucket_spread_mean"] is None

    def test_handles_empty_result(self):
        """Should handle empty backtest result."""
        result = extract_headline_metrics({})
        assert result == {}


class TestGenerateRunSummary:
    """Tests for generate_run_summary function."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        with tempfile.TemporaryDirectory() as d:
            yield Path(d)

    @pytest.fixture
    def minimal_inputs(self):
        """Minimal inputs for generate_run_summary."""
        return {
            "run_id": "test-run",
            "config": {"horizon": 90},
            "snapshots": [],
            "backtest_result": {
                "aggregate_metrics": {},
                "provenance": {"config_hash": "sha256:test"},
            },
            "validation_results": {"test_validation": True},
            "validation_details": {"test_validation": {"passed": True}},
        }

    def test_returns_dict(self, temp_dir, minimal_inputs):
        """Should return a dict."""
        result = generate_run_summary(**minimal_inputs, output_dir=temp_dir)
        assert isinstance(result, dict)

    def test_includes_run_id(self, temp_dir, minimal_inputs):
        """Should include run_id."""
        result = generate_run_summary(**minimal_inputs, output_dir=temp_dir)
        assert result["run_id"] == "test-run"

    def test_includes_generated_at(self, temp_dir, minimal_inputs):
        """Should include generated_at timestamp."""
        result = generate_run_summary(**minimal_inputs, output_dir=temp_dir)
        assert "generated_at" in result
        assert "T" in result["generated_at"]  # ISO format

    def test_includes_config(self, temp_dir, minimal_inputs):
        """Should include config."""
        result = generate_run_summary(**minimal_inputs, output_dir=temp_dir)
        assert result["config"]["horizon"] == 90

    def test_includes_validations(self, temp_dir, minimal_inputs):
        """Should include validations."""
        result = generate_run_summary(**minimal_inputs, output_dir=temp_dir)
        assert "validations" in result
        assert result["validations"]["all_passed"] is True
        assert result["validations"]["results"]["test_validation"] is True

    def test_includes_coverage(self, temp_dir, minimal_inputs):
        """Should include coverage stats."""
        result = generate_run_summary(**minimal_inputs, output_dir=temp_dir)
        assert "coverage" in result

    def test_includes_cohort_fallbacks(self, temp_dir, minimal_inputs):
        """Should include cohort fallback stats."""
        result = generate_run_summary(**minimal_inputs, output_dir=temp_dir)
        assert "cohort_fallbacks" in result

    def test_includes_headline_metrics(self, temp_dir, minimal_inputs):
        """Should include headline metrics."""
        result = generate_run_summary(**minimal_inputs, output_dir=temp_dir)
        assert "headline_metrics" in result

    def test_includes_hashes(self, temp_dir, minimal_inputs):
        """Should include hashes."""
        result = generate_run_summary(**minimal_inputs, output_dir=temp_dir)
        assert "hashes" in result
        assert "config" in result["hashes"]
        assert "results" in result["hashes"]

    def test_includes_quality_gates(self, temp_dir, minimal_inputs):
        """Should include quality gates."""
        result = generate_run_summary(**minimal_inputs, output_dir=temp_dir)
        assert "quality_gates" in result
        assert "return_coverage_ok" in result["quality_gates"]
        assert "fallback_rate_ok" in result["quality_gates"]
        assert "validations_ok" in result["quality_gates"]
        assert "all_gates_passed" in result["quality_gates"]

    def test_saves_to_file(self, temp_dir, minimal_inputs):
        """Should save summary to file."""
        generate_run_summary(**minimal_inputs, output_dir=temp_dir)

        summary_file = temp_dir / "run_summary.json"
        assert summary_file.exists()

        with open(summary_file) as f:
            saved = json.load(f)
        assert saved["run_id"] == "test-run"

    def test_creates_output_dir(self, temp_dir, minimal_inputs):
        """Should create output directory if missing."""
        new_dir = temp_dir / "nested" / "output"
        generate_run_summary(**minimal_inputs, output_dir=new_dir)
        assert new_dir.exists()

    def test_quality_gate_return_coverage(self, temp_dir, minimal_inputs):
        """Return coverage gate should check 80% threshold."""
        # With good coverage
        minimal_inputs["snapshots"] = [
            {
                "as_of_date": "2024-01-01",
                "ranked_securities": [{"ticker": f"T{i}"} for i in range(10)],
                "excluded_securities": [],
            }
        ]
        returns_by_date = {
            "2024-01-01": {f"T{i}": "0.10" for i in range(9)}  # 90% coverage
        }

        result = generate_run_summary(
            **minimal_inputs,
            output_dir=temp_dir,
            returns_by_date=returns_by_date,
        )

        assert result["quality_gates"]["return_coverage_ok"] is True

    def test_quality_gate_fallback_rate(self, temp_dir, minimal_inputs):
        """Fallback rate gate should check 20% threshold."""
        minimal_inputs["snapshots"] = [
            {
                "cohort_mode": "stage_only",
                "as_of_date": "2024-01-01",
                "ranked_securities": [],
                "excluded_securities": [],
                "cohort_stats": {
                    "phase_3": {"count": 90, "normalization_fallback": "normal"},
                    "phase_2": {"count": 10, "normalization_fallback": "stage_only"},
                },
            }
        ]

        result = generate_run_summary(**minimal_inputs, output_dir=temp_dir)

        # 10/100 = 10% fallback rate, should pass
        assert result["quality_gates"]["fallback_rate_ok"] is True


class TestPrintRunSummary:
    """Tests for print_run_summary function."""

    def test_prints_run_id(self, capsys):
        """Should print run ID."""
        summary = {
            "run_id": "print-test",
            "quality_gates": {"all_gates_passed": True},
            "validations": {"results": {}},
            "coverage": {},
            "cohort_fallbacks": {},
            "headline_metrics": {},
        }

        print_run_summary(summary)
        captured = capsys.readouterr()

        assert "print-test" in captured.out

    def test_prints_all_gates_passed(self, capsys):
        """Should print all gates passed status."""
        summary = {
            "run_id": "test",
            "quality_gates": {"all_gates_passed": True},
            "validations": {"results": {}},
            "coverage": {},
            "cohort_fallbacks": {},
            "headline_metrics": {},
        }

        print_run_summary(summary)
        captured = capsys.readouterr()

        assert "ALL GATES PASSED" in captured.out

    def test_prints_gates_failed(self, capsys):
        """Should print gates failed status."""
        summary = {
            "run_id": "test",
            "quality_gates": {
                "all_gates_passed": False,
                "return_coverage_ok": False,
            },
            "validations": {"results": {}},
            "coverage": {},
            "cohort_fallbacks": {},
            "headline_metrics": {},
        }

        print_run_summary(summary)
        captured = capsys.readouterr()

        assert "GATES FAILED" in captured.out

    def test_prints_coverage(self, capsys):
        """Should print coverage stats."""
        summary = {
            "run_id": "test",
            "quality_gates": {"all_gates_passed": True},
            "validations": {"results": {}},
            "coverage": {
                "rankable_rate": 0.85,
                "return_coverage_rate": 0.92,
            },
            "cohort_fallbacks": {},
            "headline_metrics": {},
        }

        print_run_summary(summary)
        captured = capsys.readouterr()

        assert "85.0%" in captured.out
        assert "92.0%" in captured.out

    def test_prints_headline_metrics(self, capsys):
        """Should print headline metrics."""
        summary = {
            "run_id": "test",
            "quality_gates": {"all_gates_passed": True},
            "validations": {"results": {}},
            "coverage": {},
            "cohort_fallbacks": {},
            "headline_metrics": {
                "63d": {"ic_mean": 0.05, "bucket_spread_mean": 0.10},
            },
        }

        print_run_summary(summary)
        captured = capsys.readouterr()

        assert "63d" in captured.out
        assert "IC=" in captured.out

    def test_prints_decision_proceed(self, capsys):
        """Should print proceed decision when gates pass."""
        summary = {
            "run_id": "test",
            "quality_gates": {"all_gates_passed": True},
            "validations": {"results": {}},
            "coverage": {},
            "cohort_fallbacks": {},
            "headline_metrics": {},
        }

        print_run_summary(summary)
        captured = capsys.readouterr()

        assert "Proceed with analysis" in captured.out

    def test_prints_decision_fix(self, capsys):
        """Should print fix decision when gates fail."""
        summary = {
            "run_id": "test",
            "quality_gates": {"all_gates_passed": False},
            "validations": {"results": {}},
            "coverage": {},
            "cohort_fallbacks": {},
            "headline_metrics": {},
        }

        print_run_summary(summary)
        captured = capsys.readouterr()

        assert "Fix data/coverage issues" in captured.out
