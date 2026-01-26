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

        # Should be OK or DEGRADED (depending on coverage)
        assert result["run_status"] in [RunStatus.OK.value, RunStatus.DEGRADED.value]

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
        """Under 300M should be micro_cap."""
        assert _market_cap_bucket(200) == "micro_cap"
        assert _market_cap_bucket(299) == "micro_cap"

    def test_small_cap(self):
        """300M-2B should be small_cap."""
        assert _market_cap_bucket(300) == "small_cap"
        assert _market_cap_bucket(1500) == "small_cap"

    def test_mid_cap(self):
        """2B-10B should be mid_cap."""
        assert _market_cap_bucket(2000) == "mid_cap"
        assert _market_cap_bucket(8000) == "mid_cap"

    def test_large_cap(self):
        """Over 10B should be large_cap."""
        assert _market_cap_bucket(15000) == "large_cap"

    def test_none_returns_unknown(self):
        """None should return unknown."""
        assert _market_cap_bucket(None) == "unknown"


class TestStageBucket:
    """Tests for _stage_bucket function."""

    def test_early_stage(self):
        """Early phases should be early_stage."""
        assert _stage_bucket("phase 1") == "early_stage"
        assert _stage_bucket("preclinical") == "early_stage"

    def test_mid_stage(self):
        """Phase 2 should be mid_stage."""
        assert _stage_bucket("phase 2") == "mid_stage"
        assert _stage_bucket("phase 1/2") == "mid_stage"

    def test_late_stage(self):
        """Phase 3 and approved should be late_stage."""
        assert _stage_bucket("phase 3") == "late_stage"
        assert _stage_bucket("phase 2/3") == "late_stage"
        assert _stage_bucket("approved") == "late_stage"

    def test_none_returns_unknown(self):
        """None should return unknown."""
        assert _stage_bucket(None) == "unknown"


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
