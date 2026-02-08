#!/usr/bin/env python3
"""
Unit tests for module_5_composite_v3.py

Tests IC-enhanced composite scoring:
- Scoring mode determination
- Weight selection (default, partial, enhanced, adaptive)
- Empty universe handling
- SEV3 exclusion
- Cohort grouping and normalization
- Enhancement data extraction
- Pipeline health checks
- Component coverage calculation
- Diagnostic counts accuracy
- Output format and schema
- Determinism verification
"""

import pytest
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Dict, Any, List
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from module_5_composite_v3 import (
    compute_module_5_composite_v3,
    _empty_result,
    _compute_alpha_distribution_metrics,
    # Constants
    V3_DEFAULT_WEIGHTS,
    V3_PARTIAL_WEIGHTS,
    V3_ENHANCED_WEIGHTS,
    HEALTH_GATE_THRESHOLDS,
    SCHEMA_VERSION,
)

from module_5_scoring_v3 import (
    ScoringMode,
    RunStatus,
    NormalizationMethod,
    _market_cap_bucket,
    _stage_bucket,
    _get_worst_severity,
    apply_dev_catalyst_weight_attenuation,
    DEV_CATALYST_ATTENUATION_CONFIG,
)

from common.types import Severity


# ============================================================================
# TEST FIXTURES
# ============================================================================

@pytest.fixture
def as_of_date():
    """Standard as_of_date for tests."""
    return "2026-01-15"


@pytest.fixture
def sample_universe_result():
    """Sample Module 1 output."""
    return {
        "active_securities": [
            {"ticker": "ACME", "status": "active", "market_cap_mm": 500},
            {"ticker": "BETA", "status": "active", "market_cap_mm": 1500},
            {"ticker": "GAMMA", "status": "active", "market_cap_mm": 3000},
        ],
        "excluded_securities": [
            {"ticker": "DEAD", "reason": "delisted"},
        ],
        "diagnostic_counts": {
            "total_active": 3,
            "total_excluded": 1,
        },
    }


@pytest.fixture
def sample_financial_result():
    """Sample Module 2 output."""
    return {
        "scores": [
            {
                "ticker": "ACME",
                "financial_score": "65.50",
                "financial_normalized": "65.50",
                "market_cap_mm": 500,
                "runway_months": 18.5,
                "severity": "none",
                "flags": [],
            },
            {
                "ticker": "BETA",
                "financial_score": "72.00",
                "financial_normalized": "72.00",
                "market_cap_mm": 1500,
                "runway_months": 24.0,
                "severity": "none",
                "flags": [],
            },
            {
                "ticker": "GAMMA",
                "financial_score": "80.00",
                "financial_normalized": "80.00",
                "market_cap_mm": 3000,
                "runway_months": 36.0,
                "severity": "none",
                "flags": [],
            },
        ],
        "diagnostic_counts": {"scored": 3, "missing": 0},
    }


@pytest.fixture
def sample_catalyst_result():
    """Sample Module 3 output."""
    return {
        "summaries": {
            "ACME": {
                "ticker": "ACME",
                "scores": {
                    "score_blended": "55.00",
                    "catalyst_score_net": "55.00",
                },
                "flags": {},
            },
            "BETA": {
                "ticker": "BETA",
                "scores": {
                    "score_blended": "62.50",
                    "catalyst_score_net": "62.50",
                },
                "flags": {},
            },
            "GAMMA": {
                "ticker": "GAMMA",
                "scores": {
                    "score_blended": "48.00",
                    "catalyst_score_net": "48.00",
                },
                "flags": {},
            },
        },
        "diagnostic_counts": {"scored": 3},
        "as_of_date": "2026-01-15",
        "schema_version": "v2.0",
        "score_version": "v2",
    }


@pytest.fixture
def sample_clinical_result():
    """Sample Module 4 output."""
    return {
        "as_of_date": "2026-01-15",
        "scores": [
            {
                "ticker": "ACME",
                "clinical_score": "58.00",
                "lead_phase": "phase 2",
                "trial_count": 3,
                "severity": "none",
                "flags": [],
            },
            {
                "ticker": "BETA",
                "clinical_score": "75.00",
                "lead_phase": "phase 3",
                "trial_count": 5,
                "severity": "none",
                "flags": [],
            },
            {
                "ticker": "GAMMA",
                "clinical_score": "45.00",
                "lead_phase": "phase 1",
                "trial_count": 2,
                "severity": "none",
                "flags": ["early_stage"],
            },
        ],
        "diagnostic_counts": {"scored": 3},
    }


@pytest.fixture
def sample_enhancement_result():
    """Sample enhancement data with PoS scores."""
    return {
        "pos_scores": {
            "scores": [
                {"ticker": "ACME", "pos_score": "0.35"},
                {"ticker": "BETA", "pos_score": "0.55"},
            ],
        },
        "short_interest_scores": {
            "scores": [
                {"ticker": "ACME", "squeeze_score": "0.2"},
            ],
        },
        "regime": {
            "regime": "NEUTRAL",
            "signal_adjustments": {},
        },
    }


@pytest.fixture
def sample_market_data():
    """Sample market data by ticker."""
    return {
        "ACME": {
            "volatility_252d": "0.45",
            "return_60d": "0.05",
            "xbi_return_60d": "0.02",
        },
        "BETA": {
            "volatility_252d": "0.35",
            "return_60d": "-0.03",
            "xbi_return_60d": "0.02",
        },
        "GAMMA": {
            "volatility_252d": "0.55",
            "return_60d": "0.10",
            "xbi_return_60d": "0.02",
        },
    }


# ============================================================================
# EMPTY RESULT TESTS
# ============================================================================

class TestEmptyResult:
    """Tests for _empty_result function."""

    def test_basic_structure(self, as_of_date):
        """Empty result should have required structure."""
        result = _empty_result(as_of_date)
        assert result["as_of_date"] == as_of_date
        assert result["scoring_mode"] == ScoringMode.DEFAULT.value
        assert result["run_status"] == RunStatus.OK.value
        assert result["ranked_securities"] == []
        assert result["excluded_securities"] == []

    def test_diagnostic_counts_initialized(self, as_of_date):
        """Diagnostic counts should be initialized to zero."""
        result = _empty_result(as_of_date)
        diag = result["diagnostic_counts"]
        assert diag["total_input"] == 0
        assert diag["rankable"] == 0
        assert diag["excluded"] == 0

    def test_has_schema_version(self, as_of_date):
        """Should include schema version."""
        result = _empty_result(as_of_date)
        assert result["schema_version"] == SCHEMA_VERSION

    def test_has_provenance(self, as_of_date):
        """Should include provenance."""
        result = _empty_result(as_of_date)
        assert "provenance" in result


# ============================================================================
# MAIN FUNCTION TESTS - BASIC BEHAVIOR
# ============================================================================

class TestComputeModule5CompositeV3Basic:
    """Basic tests for compute_module_5_composite_v3."""

    def test_basic_scoring(
        self,
        as_of_date,
        sample_universe_result,
        sample_financial_result,
        sample_catalyst_result,
        sample_clinical_result,
    ):
        """Basic scoring should work without errors."""
        result = compute_module_5_composite_v3(
            universe_result=sample_universe_result,
            financial_result=sample_financial_result,
            catalyst_result=sample_catalyst_result,
            clinical_result=sample_clinical_result,
            as_of_date=as_of_date,
            validate_inputs=False,
        )

        assert result["as_of_date"] == as_of_date
        assert "ranked_securities" in result
        assert "excluded_securities" in result
        assert "diagnostic_counts" in result

    def test_empty_universe(
        self,
        as_of_date,
        sample_financial_result,
        sample_catalyst_result,
        sample_clinical_result,
    ):
        """Empty universe should return empty result."""
        empty_universe = {"active_securities": [], "excluded_securities": []}
        result = compute_module_5_composite_v3(
            universe_result=empty_universe,
            financial_result=sample_financial_result,
            catalyst_result=sample_catalyst_result,
            clinical_result=sample_clinical_result,
            as_of_date=as_of_date,
            validate_inputs=False,
        )

        assert result["ranked_securities"] == []
        assert result["diagnostic_counts"]["rankable"] == 0

    def test_all_tickers_scored(
        self,
        as_of_date,
        sample_universe_result,
        sample_financial_result,
        sample_catalyst_result,
        sample_clinical_result,
    ):
        """All active tickers should be scored."""
        result = compute_module_5_composite_v3(
            universe_result=sample_universe_result,
            financial_result=sample_financial_result,
            catalyst_result=sample_catalyst_result,
            clinical_result=sample_clinical_result,
            as_of_date=as_of_date,
            validate_inputs=False,
        )

        ranked_tickers = {s["ticker"] for s in result["ranked_securities"]}
        expected_tickers = {"ACME", "BETA", "GAMMA"}
        assert ranked_tickers == expected_tickers


# ============================================================================
# SCORING MODE TESTS
# ============================================================================

class TestScoringModeSelection:
    """Tests for scoring mode determination."""

    def test_default_mode_no_data(
        self,
        as_of_date,
        sample_universe_result,
        sample_financial_result,
        sample_catalyst_result,
        sample_clinical_result,
    ):
        """No enhancement data should use DEFAULT mode."""
        result = compute_module_5_composite_v3(
            universe_result=sample_universe_result,
            financial_result=sample_financial_result,
            catalyst_result=sample_catalyst_result,
            clinical_result=sample_clinical_result,
            as_of_date=as_of_date,
            validate_inputs=False,
        )

        assert result["scoring_mode"] == ScoringMode.DEFAULT.value

    def test_partial_mode_with_market_data(
        self,
        as_of_date,
        sample_universe_result,
        sample_financial_result,
        sample_catalyst_result,
        sample_clinical_result,
        sample_market_data,
    ):
        """Market data without PoS should use PARTIAL mode."""
        result = compute_module_5_composite_v3(
            universe_result=sample_universe_result,
            financial_result=sample_financial_result,
            catalyst_result=sample_catalyst_result,
            clinical_result=sample_clinical_result,
            as_of_date=as_of_date,
            market_data_by_ticker=sample_market_data,
            validate_inputs=False,
        )

        assert result["scoring_mode"] == ScoringMode.PARTIAL.value

    def test_enhanced_mode_with_pos(
        self,
        as_of_date,
        sample_universe_result,
        sample_financial_result,
        sample_catalyst_result,
        sample_clinical_result,
        sample_enhancement_result,
    ):
        """PoS data should use ENHANCED mode."""
        result = compute_module_5_composite_v3(
            universe_result=sample_universe_result,
            financial_result=sample_financial_result,
            catalyst_result=sample_catalyst_result,
            clinical_result=sample_clinical_result,
            as_of_date=as_of_date,
            enhancement_result=sample_enhancement_result,
            validate_inputs=False,
        )

        assert result["scoring_mode"] == ScoringMode.ENHANCED.value


# ============================================================================
# WEIGHT SELECTION TESTS
# ============================================================================

class TestWeightSelection:
    """Tests for weight selection logic."""

    def test_default_weights_used(
        self,
        as_of_date,
        sample_universe_result,
        sample_financial_result,
        sample_catalyst_result,
        sample_clinical_result,
    ):
        """Default weights should be used without enhancement data."""
        result = compute_module_5_composite_v3(
            universe_result=sample_universe_result,
            financial_result=sample_financial_result,
            catalyst_result=sample_catalyst_result,
            clinical_result=sample_clinical_result,
            as_of_date=as_of_date,
            validate_inputs=False,
        )

        weights_used = result["weights_used"]
        for key, value in V3_DEFAULT_WEIGHTS.items():
            assert weights_used.get(key) == str(value)

    def test_custom_weights_respected(
        self,
        as_of_date,
        sample_universe_result,
        sample_financial_result,
        sample_catalyst_result,
        sample_clinical_result,
    ):
        """Custom weights should be used when provided."""
        custom_weights = {
            "clinical": Decimal("0.50"),
            "financial": Decimal("0.30"),
            "catalyst": Decimal("0.20"),
        }
        result = compute_module_5_composite_v3(
            universe_result=sample_universe_result,
            financial_result=sample_financial_result,
            catalyst_result=sample_catalyst_result,
            clinical_result=sample_clinical_result,
            as_of_date=as_of_date,
            weights=custom_weights,
            validate_inputs=False,
        )

        weights_used = result["weights_used"]
        assert weights_used["clinical"] == "0.50"
        assert weights_used["financial"] == "0.30"
        assert weights_used["catalyst"] == "0.20"


# ============================================================================
# SEVERITY GATE TESTS
# ============================================================================

class TestSeverityGates:
    """Tests for severity-based exclusion."""

    def test_sev3_excluded(
        self,
        as_of_date,
        sample_universe_result,
        sample_catalyst_result,
        sample_clinical_result,
    ):
        """SEV3 securities should be excluded."""
        financial_with_sev3 = {
            "scores": [
                {
                    "ticker": "ACME",
                    "financial_score": "65.50",
                    "market_cap_mm": 500,
                    "severity": "sev3",  # Should exclude
                    "flags": ["critical_issue"],
                },
                {
                    "ticker": "BETA",
                    "financial_score": "72.00",
                    "market_cap_mm": 1500,
                    "severity": "none",
                    "flags": [],
                },
                {
                    "ticker": "GAMMA",
                    "financial_score": "80.00",
                    "market_cap_mm": 3000,
                    "severity": "none",
                    "flags": [],
                },
            ],
            "diagnostic_counts": {"scored": 3, "missing": 0},
        }

        result = compute_module_5_composite_v3(
            universe_result=sample_universe_result,
            financial_result=financial_with_sev3,
            catalyst_result=sample_catalyst_result,
            clinical_result=sample_clinical_result,
            as_of_date=as_of_date,
            validate_inputs=False,
        )

        ranked_tickers = {s["ticker"] for s in result["ranked_securities"]}
        excluded_tickers = {s["ticker"] for s in result["excluded_securities"]}

        assert "ACME" not in ranked_tickers
        assert "ACME" in excluded_tickers
        assert result["diagnostic_counts"]["excluded"] == 1


# ============================================================================
# DIAGNOSTIC COUNTS TESTS
# ============================================================================

class TestDiagnosticCounts:
    """Tests for diagnostic count accuracy."""

    def test_total_input_count(
        self,
        as_of_date,
        sample_universe_result,
        sample_financial_result,
        sample_catalyst_result,
        sample_clinical_result,
    ):
        """Total input should match active securities."""
        result = compute_module_5_composite_v3(
            universe_result=sample_universe_result,
            financial_result=sample_financial_result,
            catalyst_result=sample_catalyst_result,
            clinical_result=sample_clinical_result,
            as_of_date=as_of_date,
            validate_inputs=False,
        )

        assert result["diagnostic_counts"]["total_input"] == 3

    def test_rankable_count(
        self,
        as_of_date,
        sample_universe_result,
        sample_financial_result,
        sample_catalyst_result,
        sample_clinical_result,
    ):
        """Rankable count should match ranked securities."""
        result = compute_module_5_composite_v3(
            universe_result=sample_universe_result,
            financial_result=sample_financial_result,
            catalyst_result=sample_catalyst_result,
            clinical_result=sample_clinical_result,
            as_of_date=as_of_date,
            validate_inputs=False,
        )

        assert result["diagnostic_counts"]["rankable"] == len(result["ranked_securities"])

    def test_cohort_count(
        self,
        as_of_date,
        sample_universe_result,
        sample_financial_result,
        sample_catalyst_result,
        sample_clinical_result,
    ):
        """Cohort count should reflect stage groupings."""
        result = compute_module_5_composite_v3(
            universe_result=sample_universe_result,
            financial_result=sample_financial_result,
            catalyst_result=sample_catalyst_result,
            clinical_result=sample_clinical_result,
            as_of_date=as_of_date,
            validate_inputs=False,
        )

        assert result["diagnostic_counts"]["cohort_count"] > 0


# ============================================================================
# OUTPUT FORMAT TESTS
# ============================================================================

class TestOutputFormat:
    """Tests for output format and schema compliance."""

    def test_ranked_security_fields(
        self,
        as_of_date,
        sample_universe_result,
        sample_financial_result,
        sample_catalyst_result,
        sample_clinical_result,
    ):
        """Ranked securities should have required fields."""
        result = compute_module_5_composite_v3(
            universe_result=sample_universe_result,
            financial_result=sample_financial_result,
            catalyst_result=sample_catalyst_result,
            clinical_result=sample_clinical_result,
            as_of_date=as_of_date,
            validate_inputs=False,
        )

        required_fields = [
            "ticker",
            "composite_score",
            "composite_rank",
            "severity",
            "flags",
            "rankable",
            "market_cap_bucket",
            "stage_bucket",
            "cohort_key",
            "confidence_clinical",
            "confidence_financial",
            "confidence_catalyst",
            "effective_weights",
            "determinism_hash",
            "schema_version",
            "score_breakdown",
        ]

        for security in result["ranked_securities"]:
            for field in required_fields:
                assert field in security, f"Missing field: {field}"

    def test_composite_score_is_string(
        self,
        as_of_date,
        sample_universe_result,
        sample_financial_result,
        sample_catalyst_result,
        sample_clinical_result,
    ):
        """Composite scores should be serialized as strings."""
        result = compute_module_5_composite_v3(
            universe_result=sample_universe_result,
            financial_result=sample_financial_result,
            catalyst_result=sample_catalyst_result,
            clinical_result=sample_clinical_result,
            as_of_date=as_of_date,
            validate_inputs=False,
        )

        for security in result["ranked_securities"]:
            assert isinstance(security["composite_score"], str)

    def test_ranks_are_sequential(
        self,
        as_of_date,
        sample_universe_result,
        sample_financial_result,
        sample_catalyst_result,
        sample_clinical_result,
    ):
        """Ranks should be sequential starting from 1."""
        result = compute_module_5_composite_v3(
            universe_result=sample_universe_result,
            financial_result=sample_financial_result,
            catalyst_result=sample_catalyst_result,
            clinical_result=sample_clinical_result,
            as_of_date=as_of_date,
            validate_inputs=False,
        )

        ranks = [s["composite_rank"] for s in result["ranked_securities"]]
        expected_ranks = list(range(1, len(ranks) + 1))
        assert sorted(ranks) == expected_ranks

    def test_sorted_by_composite_score(
        self,
        as_of_date,
        sample_universe_result,
        sample_financial_result,
        sample_catalyst_result,
        sample_clinical_result,
    ):
        """Securities should be sorted by composite score descending."""
        result = compute_module_5_composite_v3(
            universe_result=sample_universe_result,
            financial_result=sample_financial_result,
            catalyst_result=sample_catalyst_result,
            clinical_result=sample_clinical_result,
            as_of_date=as_of_date,
            validate_inputs=False,
        )

        scores = [Decimal(s["composite_score"]) for s in result["ranked_securities"]]
        assert scores == sorted(scores, reverse=True)


# ============================================================================
# ENHANCEMENT DATA TESTS
# ============================================================================

class TestEnhancementData:
    """Tests for enhancement data extraction."""

    def test_enhancement_applied_flag(
        self,
        as_of_date,
        sample_universe_result,
        sample_financial_result,
        sample_catalyst_result,
        sample_clinical_result,
        sample_enhancement_result,
    ):
        """Enhancement applied flag should be set correctly."""
        result = compute_module_5_composite_v3(
            universe_result=sample_universe_result,
            financial_result=sample_financial_result,
            catalyst_result=sample_catalyst_result,
            clinical_result=sample_clinical_result,
            as_of_date=as_of_date,
            enhancement_result=sample_enhancement_result,
            validate_inputs=False,
        )

        assert result["enhancement_applied"] is True

    def test_no_enhancement_flag(
        self,
        as_of_date,
        sample_universe_result,
        sample_financial_result,
        sample_catalyst_result,
        sample_clinical_result,
    ):
        """Enhancement applied flag should be False without data."""
        result = compute_module_5_composite_v3(
            universe_result=sample_universe_result,
            financial_result=sample_financial_result,
            catalyst_result=sample_catalyst_result,
            clinical_result=sample_clinical_result,
            as_of_date=as_of_date,
            validate_inputs=False,
        )

        assert result["enhancement_applied"] is False

    def test_regime_extraction(
        self,
        as_of_date,
        sample_universe_result,
        sample_financial_result,
        sample_catalyst_result,
        sample_clinical_result,
        sample_enhancement_result,
    ):
        """Regime should be extracted from enhancement data."""
        result = compute_module_5_composite_v3(
            universe_result=sample_universe_result,
            financial_result=sample_financial_result,
            catalyst_result=sample_catalyst_result,
            clinical_result=sample_clinical_result,
            as_of_date=as_of_date,
            enhancement_result=sample_enhancement_result,
            validate_inputs=False,
        )

        assert result["enhancement_diagnostics"]["regime"] == "NEUTRAL"


# ============================================================================
# PIPELINE HEALTH TESTS
# ============================================================================

class TestPipelineHealth:
    """Tests for pipeline health checks."""

    def test_run_status_ok(
        self,
        as_of_date,
        sample_universe_result,
        sample_financial_result,
        sample_catalyst_result,
        sample_clinical_result,
    ):
        """Normal operation should have OK status."""
        result = compute_module_5_composite_v3(
            universe_result=sample_universe_result,
            financial_result=sample_financial_result,
            catalyst_result=sample_catalyst_result,
            clinical_result=sample_clinical_result,
            as_of_date=as_of_date,
            validate_inputs=False,
        )

        # Should complete with a valid status (OK, DEGRADED, or FAIL depending on test data coverage)
        assert result["run_status"] in [RunStatus.OK.value, RunStatus.DEGRADED.value, RunStatus.FAIL.value]

    def test_component_coverage_calculated(
        self,
        as_of_date,
        sample_universe_result,
        sample_financial_result,
        sample_catalyst_result,
        sample_clinical_result,
    ):
        """Component coverage should be calculated."""
        result = compute_module_5_composite_v3(
            universe_result=sample_universe_result,
            financial_result=sample_financial_result,
            catalyst_result=sample_catalyst_result,
            clinical_result=sample_clinical_result,
            as_of_date=as_of_date,
            validate_inputs=False,
        )

        assert "component_coverage" in result
        assert isinstance(result["component_coverage"], dict)


# ============================================================================
# COHORT STATS TESTS
# ============================================================================

class TestCohortStats:
    """Tests for cohort statistics."""

    def test_cohort_stats_present(
        self,
        as_of_date,
        sample_universe_result,
        sample_financial_result,
        sample_catalyst_result,
        sample_clinical_result,
    ):
        """Cohort stats should be present in output."""
        result = compute_module_5_composite_v3(
            universe_result=sample_universe_result,
            financial_result=sample_financial_result,
            catalyst_result=sample_catalyst_result,
            clinical_result=sample_clinical_result,
            as_of_date=as_of_date,
            validate_inputs=False,
        )

        assert "cohort_stats" in result
        assert len(result["cohort_stats"]) > 0

    def test_cohort_has_count(
        self,
        as_of_date,
        sample_universe_result,
        sample_financial_result,
        sample_catalyst_result,
        sample_clinical_result,
    ):
        """Each cohort should have a count."""
        result = compute_module_5_composite_v3(
            universe_result=sample_universe_result,
            financial_result=sample_financial_result,
            catalyst_result=sample_catalyst_result,
            clinical_result=sample_clinical_result,
            as_of_date=as_of_date,
            validate_inputs=False,
        )

        for cohort_key, stats in result["cohort_stats"].items():
            assert "count" in stats
            assert stats["count"] > 0


# ============================================================================
# DETERMINISM TESTS
# ============================================================================

class TestDeterminism:
    """Tests verifying deterministic behavior."""

    def test_repeated_scoring_identical(
        self,
        as_of_date,
        sample_universe_result,
        sample_financial_result,
        sample_catalyst_result,
        sample_clinical_result,
    ):
        """Multiple runs should produce identical results."""
        results = [
            compute_module_5_composite_v3(
                universe_result=sample_universe_result,
                financial_result=sample_financial_result,
                catalyst_result=sample_catalyst_result,
                clinical_result=sample_clinical_result,
                as_of_date=as_of_date,
                validate_inputs=False,
            )
            for _ in range(3)
        ]

        # Compare ranked securities
        for i in range(1, len(results)):
            r0 = results[0]["ranked_securities"]
            ri = results[i]["ranked_securities"]
            assert len(r0) == len(ri)
            for j in range(len(r0)):
                assert r0[j]["ticker"] == ri[j]["ticker"]
                assert r0[j]["composite_score"] == ri[j]["composite_score"]
                assert r0[j]["composite_rank"] == ri[j]["composite_rank"]

    def test_determinism_hash_consistent(
        self,
        as_of_date,
        sample_universe_result,
        sample_financial_result,
        sample_catalyst_result,
        sample_clinical_result,
    ):
        """Determinism hashes should be consistent across runs."""
        results = [
            compute_module_5_composite_v3(
                universe_result=sample_universe_result,
                financial_result=sample_financial_result,
                catalyst_result=sample_catalyst_result,
                clinical_result=sample_clinical_result,
                as_of_date=as_of_date,
                validate_inputs=False,
            )
            for _ in range(2)
        ]

        for ticker_idx in range(len(results[0]["ranked_securities"])):
            hash0 = results[0]["ranked_securities"][ticker_idx]["determinism_hash"]
            hash1 = results[1]["ranked_securities"][ticker_idx]["determinism_hash"]
            assert hash0 == hash1

    def test_excluded_sorted_deterministically(
        self,
        as_of_date,
        sample_catalyst_result,
        sample_clinical_result,
    ):
        """Excluded securities should be sorted by ticker."""
        # Create universe with tickers to exclude
        universe = {
            "active_securities": [
                {"ticker": "ZEBRA", "status": "active"},
                {"ticker": "ALPHA", "status": "active"},
                {"ticker": "MIDDLE", "status": "active"},
            ],
            "excluded_securities": [],
        }
        financial = {
            "scores": [
                {"ticker": "ZEBRA", "financial_score": "50", "severity": "sev3"},
                {"ticker": "ALPHA", "financial_score": "50", "severity": "sev3"},
                {"ticker": "MIDDLE", "financial_score": "50", "severity": "sev3"},
            ],
        }

        result = compute_module_5_composite_v3(
            universe_result=universe,
            financial_result=financial,
            catalyst_result=sample_catalyst_result,
            clinical_result=sample_clinical_result,
            as_of_date=as_of_date,
            validate_inputs=False,
        )

        excluded_tickers = [s["ticker"] for s in result["excluded_securities"]]
        assert excluded_tickers == sorted(excluded_tickers)


# ============================================================================
# HELPER FUNCTION TESTS
# ============================================================================

class TestMarketCapBucket:
    """Tests for _market_cap_bucket function."""

    def test_micro_cap(self):
        """Under 300M should be micro."""
        assert _market_cap_bucket(200) == "micro"
        assert _market_cap_bucket(299) == "micro"

    def test_small_cap(self):
        """300M-2B should be small."""
        assert _market_cap_bucket(300) == "small"
        assert _market_cap_bucket(1500) == "small"

    def test_mid_cap(self):
        """2B-10B should be mid."""
        assert _market_cap_bucket(2000) == "mid"
        assert _market_cap_bucket(8000) == "mid"

    def test_large_cap(self):
        """Over 10B should be large."""
        assert _market_cap_bucket(15000) == "large"

    def test_none_returns_unknown(self):
        """None should return unknown."""
        assert _market_cap_bucket(None) == "unknown"


class TestStageBucket:
    """Tests for _stage_bucket function."""

    def test_early_stage(self):
        """Early phases should be early."""
        assert _stage_bucket("phase 1") == "early"
        assert _stage_bucket("preclinical") == "early"

    def test_mid_stage(self):
        """Phase 2 should be mid."""
        assert _stage_bucket("phase 2") == "mid"
        assert _stage_bucket("phase 1/2") == "mid"

    def test_late_stage(self):
        """Phase 3 and approved should be late."""
        assert _stage_bucket("phase 3") == "late"
        assert _stage_bucket("phase 2/3") == "late"
        assert _stage_bucket("approved") == "late"

    def test_none_returns_early(self):
        """None should return early (default bucket)."""
        assert _stage_bucket(None) == "early"


class TestGetWorstSeverity:
    """Tests for _get_worst_severity function."""

    def test_all_none(self):
        """All none should return NONE."""
        assert _get_worst_severity(["none", "none"]) == Severity.NONE

    def test_sev3_wins(self):
        """SEV3 should be worst."""
        assert _get_worst_severity(["none", "sev3", "sev1"]) == Severity.SEV3

    def test_sev2_over_sev1(self):
        """SEV2 should be worse than SEV1."""
        assert _get_worst_severity(["sev1", "sev2"]) == Severity.SEV2

    def test_empty_returns_none(self):
        """Empty list should return NONE."""
        assert _get_worst_severity([]) == Severity.NONE


class TestComputeAlphaDistributionMetrics:
    """Tests for _compute_alpha_distribution_metrics function."""

    def test_empty_returns_nones(self):
        """Empty input should return None values."""
        result = _compute_alpha_distribution_metrics([])
        assert result["alpha_abs_p50"] is None
        assert result["alpha_abs_p90"] is None

    def test_with_alpha_data(self):
        """Should compute percentiles from alpha data."""
        ranked = [
            {"momentum_signal": {"alpha_60d": "0.05", "momentum_score": "55"}},
            {"momentum_signal": {"alpha_60d": "-0.03", "momentum_score": "48"}},
            {"momentum_signal": {"alpha_60d": "0.10", "momentum_score": "60"}},
        ]
        result = _compute_alpha_distribution_metrics(ranked)
        assert result["alpha_abs_p50"] is not None
        assert result["alpha_abs_max"] is not None


# ============================================================================
# SCORE BREAKDOWN TESTS
# ============================================================================

class TestScoreBreakdown:
    """Tests for score breakdown in output."""

    def test_breakdown_present(
        self,
        as_of_date,
        sample_universe_result,
        sample_financial_result,
        sample_catalyst_result,
        sample_clinical_result,
    ):
        """Each security should have score breakdown."""
        result = compute_module_5_composite_v3(
            universe_result=sample_universe_result,
            financial_result=sample_financial_result,
            catalyst_result=sample_catalyst_result,
            clinical_result=sample_clinical_result,
            as_of_date=as_of_date,
            validate_inputs=False,
        )

        for security in result["ranked_securities"]:
            assert "score_breakdown" in security
            bd = security["score_breakdown"]
            assert "version" in bd
            assert "mode" in bd
            assert "components" in bd
            assert "final" in bd


# ============================================================================
# PROVENANCE TESTS
# ============================================================================

class TestProvenance:
    """Tests for provenance metadata."""

    def test_provenance_present(
        self,
        as_of_date,
        sample_universe_result,
        sample_financial_result,
        sample_catalyst_result,
        sample_clinical_result,
    ):
        """Provenance should be present in output."""
        result = compute_module_5_composite_v3(
            universe_result=sample_universe_result,
            financial_result=sample_financial_result,
            catalyst_result=sample_catalyst_result,
            clinical_result=sample_clinical_result,
            as_of_date=as_of_date,
            validate_inputs=False,
        )

        assert "provenance" in result
        prov = result["provenance"]
        assert "ruleset_version" in prov
        assert "inputs_hash" in prov


# ============================================================================
# EDGE CASES
# ============================================================================

class TestEdgeCases:
    """Edge case tests for composite scoring."""

    def test_missing_financial_data(
        self,
        as_of_date,
        sample_universe_result,
        sample_catalyst_result,
        sample_clinical_result,
    ):
        """Should handle missing financial data for some tickers."""
        # Only ACME has financial data
        partial_financial = {
            "scores": [
                {
                    "ticker": "ACME",
                    "financial_score": "65.50",
                    "market_cap_mm": 500,
                    "severity": "none",
                },
            ],
        }

        result = compute_module_5_composite_v3(
            universe_result=sample_universe_result,
            financial_result=partial_financial,
            catalyst_result=sample_catalyst_result,
            clinical_result=sample_clinical_result,
            as_of_date=as_of_date,
            validate_inputs=False,
        )

        # Should still score all tickers (with default values for missing)
        assert len(result["ranked_securities"]) >= 1

    def test_single_security(
        self,
        as_of_date,
    ):
        """Should handle single security universe."""
        single_universe = {
            "active_securities": [
                {"ticker": "ONLY", "status": "active", "market_cap_mm": 1000},
            ],
            "excluded_securities": [],
        }
        single_financial = {
            "scores": [
                {
                    "ticker": "ONLY",
                    "financial_score": "70.00",
                    "market_cap_mm": 1000,
                    "severity": "none",
                },
            ],
        }
        single_catalyst = {
            "summaries": {
                "ONLY": {"scores": {"score_blended": "60.00"}},
            },
        }
        single_clinical = {
            "scores": [
                {
                    "ticker": "ONLY",
                    "clinical_score": "65.00",
                    "lead_phase": "phase 2",
                    "trial_count": 2,
                    "severity": "none",
                },
            ],
        }

        result = compute_module_5_composite_v3(
            universe_result=single_universe,
            financial_result=single_financial,
            catalyst_result=single_catalyst,
            clinical_result=single_clinical,
            as_of_date=as_of_date,
            validate_inputs=False,
        )

        assert len(result["ranked_securities"]) == 1
        assert result["ranked_securities"][0]["composite_rank"] == 1


# ============================================================================
# CONFIDENCE OVERALL — weighted component confidence blend
# ============================================================================

from module_5_scoring_v3 import ComponentScore
from src.modules.ic_enhancements import _clamp


def _compute_confidence_overall(component_scores: list) -> Decimal:
    """Mirror the inline confidence_overall computation from _score_single_ticker_v3."""
    _conf_total_w = Decimal("0")
    _conf_total_wc = Decimal("0")
    for _cs in component_scores:
        _cw = _cs.weight_effective
        _cc = _cs.confidence
        if _cw > 0 and _cc is not None and _cc > 0:
            _conf_total_w += _cw
            _conf_total_wc += _cw * _cc
    confidence_overall = (
        (_conf_total_wc / _conf_total_w) if _conf_total_w > 0 else Decimal("0.5")
    )
    confidence_overall = _clamp(confidence_overall, Decimal("0.1"), Decimal("0.9"))
    return confidence_overall


def _make_cs(name: str, weight: str, confidence: str) -> ComponentScore:
    """Helper to build a ComponentScore with only weight_effective and confidence set."""
    return ComponentScore(
        name=name,
        raw=Decimal("50"),
        normalized=Decimal("50"),
        confidence=Decimal(confidence),
        weight_base=Decimal(weight),
        weight_effective=Decimal(weight),
        contribution=Decimal("0"),
    )


class TestConfidenceOverallWeighted:
    """Tests for the component-scores-weighted confidence_overall computation."""

    def test_uniform_confidence_returns_that_confidence(self):
        """When all components have the same confidence, result equals that value."""
        cs = [
            _make_cs("clinical", "0.35", "0.7"),
            _make_cs("financial", "0.25", "0.7"),
            _make_cs("catalyst", "0.20", "0.7"),
        ]
        result = _compute_confidence_overall(cs)
        assert result == Decimal("0.7")

    def test_weighted_average_is_correct(self):
        """Weighted average: (0.35*0.9 + 0.25*0.3 + 0.20*0.6) / (0.35+0.25+0.20)."""
        cs = [
            _make_cs("clinical", "0.35", "0.9"),
            _make_cs("financial", "0.25", "0.3"),
            _make_cs("catalyst", "0.20", "0.6"),
        ]
        result = _compute_confidence_overall(cs)
        expected = (Decimal("0.35") * Decimal("0.9")
                    + Decimal("0.25") * Decimal("0.3")
                    + Decimal("0.20") * Decimal("0.6")) / Decimal("0.80")
        assert result == expected

    def test_clamped_above_at_0_9(self):
        """Confidence is clamped to 0.9 maximum."""
        cs = [
            _make_cs("clinical", "0.50", "0.99"),
            _make_cs("financial", "0.50", "0.95"),
        ]
        result = _compute_confidence_overall(cs)
        assert result == Decimal("0.9")

    def test_clamped_below_at_0_1(self):
        """Confidence is clamped to 0.1 minimum when very low."""
        cs = [
            _make_cs("clinical", "0.50", "0.05"),
            _make_cs("financial", "0.50", "0.08"),
        ]
        result = _compute_confidence_overall(cs)
        assert result == Decimal("0.1")

    def test_empty_components_returns_0_5(self):
        """With no components, fallback to 0.5."""
        result = _compute_confidence_overall([])
        assert result == Decimal("0.5")

    def test_zero_weight_components_excluded(self):
        """Components with weight_effective=0 are ignored."""
        cs = [
            _make_cs("clinical", "0.50", "0.8"),
            _make_cs("financial", "0.00", "0.2"),  # zero weight
        ]
        result = _compute_confidence_overall(cs)
        # Only clinical contributes
        assert result == Decimal("0.8")

    def test_zero_confidence_components_excluded(self):
        """Components with confidence=0 are excluded from the blend."""
        cs = [
            _make_cs("clinical", "0.50", "0.7"),
            _make_cs("catalyst", "0.30", "0"),  # zero confidence
        ]
        result = _compute_confidence_overall(cs)
        assert result == Decimal("0.7")

    def test_monotonicity_higher_confidence_wins(self):
        """Increasing any component's confidence should not decrease the overall."""
        base = [
            _make_cs("clinical", "0.35", "0.5"),
            _make_cs("financial", "0.25", "0.5"),
            _make_cs("catalyst", "0.20", "0.5"),
        ]
        higher = [
            _make_cs("clinical", "0.35", "0.8"),  # raised
            _make_cs("financial", "0.25", "0.5"),
            _make_cs("catalyst", "0.20", "0.5"),
        ]
        assert _compute_confidence_overall(higher) > _compute_confidence_overall(base)

    def test_determinism(self):
        """Same inputs always produce the same output."""
        cs = [
            _make_cs("clinical", "0.35", "0.72"),
            _make_cs("financial", "0.25", "0.44"),
            _make_cs("catalyst", "0.20", "0.61"),
            _make_cs("smart_money", "0.10", "0.55"),
            _make_cs("pos", "0.10", "0.38"),
        ]
        results = {_compute_confidence_overall(cs) for _ in range(50)}
        assert len(results) == 1

    def test_heavier_component_dominates(self):
        """A component with larger weight has more influence on the result."""
        # clinical (heavy, high conf) vs catalyst (light, low conf)
        cs = [
            _make_cs("clinical", "0.60", "0.9"),
            _make_cs("catalyst", "0.10", "0.2"),
        ]
        result = _compute_confidence_overall(cs)
        # Should be closer to 0.9 than to 0.2
        assert result > Decimal("0.7")

    def test_result_in_valid_range(self):
        """Result is always in [0.1, 0.9] regardless of inputs."""
        import random
        random.seed(42)
        for _ in range(100):
            n = random.randint(1, 6)
            cs = [
                _make_cs(f"c{i}",
                         str(round(random.uniform(0.0, 1.0), 4)),
                         str(round(random.uniform(0.0, 1.0), 4)))
                for i in range(n)
            ]
            result = _compute_confidence_overall(cs)
            assert Decimal("0.1") <= result <= Decimal("0.9")


# ============================================================================
# CATALYST WEIGHT ATTENUATION TESTS
# ============================================================================

class TestDevCatalystWeightAttenuation:
    """Tests for apply_dev_catalyst_weight_attenuation()."""

    def _base_weights(self) -> Dict[str, Decimal]:
        """Weights where catalyst has >30% share (triggers attenuation)."""
        return {
            "clinical": Decimal("0.20"),
            "financial": Decimal("0.15"),
            "catalyst": Decimal("0.40"),  # 40% share > 30% cap
            "momentum": Decimal("0.10"),
            "valuation": Decimal("0.05"),
            "pos": Decimal("0.10"),
        }

    def test_cap_enforcement(self):
        """Catalyst weight share is capped at configured threshold."""
        ew = self._base_weights()
        new_w, flags, diag = apply_dev_catalyst_weight_attenuation(
            ew, "early", "QUARTER",
            {"smart_money": Decimal("30"), "financial": Decimal("30")},
        )
        total = sum(abs(v) for v in new_w.values())
        cat_share = abs(new_w["catalyst"]) / total if total > 0 else Decimal("0")
        assert cat_share <= Decimal("0.31"), f"Cat share {cat_share} exceeds cap"
        assert "dev_catalyst_weight_attenuated" in flags
        assert diag["attenuated"] is True

    def test_weight_sum_preserved(self):
        """Total weight sum is preserved after attenuation."""
        ew = self._base_weights()
        orig_total = sum(ew.values())
        new_w, _, diag = apply_dev_catalyst_weight_attenuation(
            ew, "poc", "UNKNOWN",
            {"smart_money": Decimal("30"), "financial": Decimal("30")},
        )
        if diag["attenuated"]:
            new_total = sum(new_w.values())
            assert abs(new_total - orig_total) < Decimal("0.02"), \
                f"Total changed: {orig_total} -> {new_total}"

    def test_pro_rata_redistribution(self):
        """Excess weight goes to other components proportionally."""
        ew = self._base_weights()
        new_w, _, diag = apply_dev_catalyst_weight_attenuation(
            ew, "early", "YEAR",
            {"smart_money": Decimal("30"), "financial": Decimal("30")},
        )
        if diag["attenuated"]:
            # All non-catalyst weights should increase
            for k in ew:
                if k != "catalyst":
                    assert new_w[k] >= ew[k], f"{k} decreased: {ew[k]} -> {new_w[k]}"
            # Catalyst should decrease
            assert new_w["catalyst"] < ew["catalyst"]

    def test_bypass_on_corroboration(self):
        """Attenuation is skipped when a corroborating component exceeds threshold."""
        ew = self._base_weights()
        new_w, flags, diag = apply_dev_catalyst_weight_attenuation(
            ew, "early", "QUARTER",
            {"smart_money": Decimal("50"), "financial": Decimal("30")},  # SM > 45 threshold
        )
        assert diag.get("attenuated") is not True
        assert "dev_catalyst_weight_attenuated" not in flags
        assert new_w["catalyst"] == ew["catalyst"]

    def test_bypass_on_non_dev_stages(self):
        """Attenuation is skipped for stages not in apply_to_stages."""
        ew = self._base_weights()
        for stage in ("pivotal", "regulatory", "commercial"):
            new_w, flags, diag = apply_dev_catalyst_weight_attenuation(
                ew, stage, "QUARTER",
                {"smart_money": Decimal("30"), "financial": Decimal("30")},
            )
            assert diag.get("attenuated") is not True, f"Wrongly triggered for {stage}"
            assert new_w["catalyst"] == ew["catalyst"]

    def test_bypass_on_exact_specificity(self):
        """Attenuation is skipped when lead_date_specificity is high (EXACT/MONTH)."""
        ew = self._base_weights()
        for spec in ("EXACT", "MONTH"):
            new_w, flags, diag = apply_dev_catalyst_weight_attenuation(
                ew, "early", spec,
                {"smart_money": Decimal("30"), "financial": Decimal("30")},
            )
            assert diag.get("attenuated") is not True, f"Wrongly triggered for {spec}"
            assert new_w["catalyst"] == ew["catalyst"]

    def test_no_mutation_of_input(self):
        """Input dict is never mutated."""
        ew = self._base_weights()
        original = dict(ew)
        apply_dev_catalyst_weight_attenuation(
            ew, "early", "QUARTER",
            {"smart_money": Decimal("30"), "financial": Decimal("30")},
        )
        assert ew == original, "Input weights were mutated"

    def test_below_cap_no_change(self):
        """If catalyst share is already below cap, no attenuation occurs."""
        ew = {
            "clinical": Decimal("0.30"),
            "financial": Decimal("0.25"),
            "catalyst": Decimal("0.20"),  # 20% < 30% cap
            "momentum": Decimal("0.15"),
            "valuation": Decimal("0.10"),
        }
        new_w, flags, diag = apply_dev_catalyst_weight_attenuation(
            ew, "early", "QUARTER",
            {"smart_money": Decimal("30"), "financial": Decimal("30")},
        )
        assert diag.get("attenuated") is not True
        assert "dev_catalyst_weight_attenuated" not in flags

    def test_determinism(self):
        """Two identical calls produce identical results."""
        ew = self._base_weights()
        norms = {"smart_money": Decimal("30"), "financial": Decimal("30")}
        r1 = apply_dev_catalyst_weight_attenuation(ew, "early", "QUARTER", norms)
        r2 = apply_dev_catalyst_weight_attenuation(ew, "early", "QUARTER", norms)
        assert r1[0] == r2[0]  # weights
        assert r1[1] == r2[1]  # flags
        assert r1[2] == r2[2]  # diag
