#!/usr/bin/env python3
"""
Unit tests for module_5_composite_with_defensive.py

Tests production wrapper for Module 5 with defensive overlays:
- V3/V2/V1 scoring mode selection
- Sanity override mechanism
- Defensive feature integration
- Universe file loading
- Feature flag handling
"""

import pytest
from decimal import Decimal
from pathlib import Path
from typing import Dict, Any
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from module_5_composite_with_defensive import (
    compute_module_5_composite_with_defensive,
    _apply_sanity_overrides,
    DEFAULT_UNIVERSE_PATHS,
    __version__,
)


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
        ],
        "excluded_securities": [],
        "diagnostic_counts": {"total_active": 2, "total_excluded": 0},
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
                "severity": "none",
                "flags": [],
            },
            {
                "ticker": "BETA",
                "financial_score": "72.00",
                "financial_normalized": "72.00",
                "market_cap_mm": 1500,
                "severity": "none",
                "flags": [],
            },
        ],
        "diagnostic_counts": {"scored": 2, "missing": 0},
    }


@pytest.fixture
def sample_catalyst_result():
    """Sample Module 3 output."""
    return {
        "summaries": {
            "ACME": {
                "ticker": "ACME",
                "scores": {"score_blended": "55.00", "catalyst_score_net": "55.00"},
                "flags": {},
            },
            "BETA": {
                "ticker": "BETA",
                "scores": {"score_blended": "62.50", "catalyst_score_net": "62.50"},
                "flags": {},
            },
        },
        "diagnostic_counts": {"scored": 2},
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
        ],
        "diagnostic_counts": {"scored": 2},
    }


# ============================================================================
# VERSION AND EXPORTS TESTS
# ============================================================================

class TestVersionAndExports:
    """Tests for module version and exports."""

    def test_version_defined(self):
        """Module version should be defined."""
        assert __version__ is not None
        assert isinstance(__version__, str)

    def test_default_universe_paths(self):
        """Default universe paths should be defined."""
        assert len(DEFAULT_UNIVERSE_PATHS) > 0


# ============================================================================
# BASIC FUNCTIONALITY TESTS
# ============================================================================

class TestBasicFunctionality:
    """Tests for basic wrapper functionality."""

    def test_basic_scoring_v3(
        self,
        as_of_date,
        sample_universe_result,
        sample_financial_result,
        sample_catalyst_result,
        sample_clinical_result,
    ):
        """Basic V3 scoring should work."""
        result = compute_module_5_composite_with_defensive(
            universe_result=sample_universe_result,
            financial_result=sample_financial_result,
            catalyst_result=sample_catalyst_result,
            clinical_result=sample_clinical_result,
            as_of_date=as_of_date,
            use_v3_scoring=True,
            enable_sanity_override=False,  # Disable to avoid v2 run
            validate=False,
        )

        assert "ranked_securities" in result
        assert len(result["ranked_securities"]) == 2

    def test_basic_scoring_v2(
        self,
        as_of_date,
        sample_universe_result,
        sample_financial_result,
        sample_catalyst_result,
        sample_clinical_result,
    ):
        """V2 scoring should work when v3 is disabled."""
        result = compute_module_5_composite_with_defensive(
            universe_result=sample_universe_result,
            financial_result=sample_financial_result,
            catalyst_result=sample_catalyst_result,
            clinical_result=sample_clinical_result,
            as_of_date=as_of_date,
            use_v3_scoring=False,
            use_v2_scoring=True,
            validate=False,
        )

        assert "ranked_securities" in result

    @pytest.mark.skip(reason="V1 scoring is deprecated and requires legacy schema formats")
    def test_basic_scoring_v1(
        self,
        as_of_date,
        sample_universe_result,
        sample_financial_result,
        sample_catalyst_result,
        sample_clinical_result,
    ):
        """V1 scoring should work when v3 and v2 are disabled."""
        result = compute_module_5_composite_with_defensive(
            universe_result=sample_universe_result,
            financial_result=sample_financial_result,
            catalyst_result=sample_catalyst_result,
            clinical_result=sample_clinical_result,
            as_of_date=as_of_date,
            use_v3_scoring=False,
            use_v2_scoring=False,
            validate=False,
        )

        assert "ranked_securities" in result


# ============================================================================
# SCORING MODE SELECTION TESTS
# ============================================================================

class TestScoringModeSelection:
    """Tests for scoring mode selection."""

    def test_v3_is_default(
        self,
        as_of_date,
        sample_universe_result,
        sample_financial_result,
        sample_catalyst_result,
        sample_clinical_result,
    ):
        """V3 should be the default scoring mode."""
        result = compute_module_5_composite_with_defensive(
            universe_result=sample_universe_result,
            financial_result=sample_financial_result,
            catalyst_result=sample_catalyst_result,
            clinical_result=sample_clinical_result,
            as_of_date=as_of_date,
            enable_sanity_override=False,
            validate=False,
        )

        # V3 should be used by default
        assert "ranked_securities" in result


# ============================================================================
# SANITY OVERRIDE TESTS
# ============================================================================

class TestSanityOverride:
    """Tests for sanity override mechanism."""

    def test_sanity_override_structure(self):
        """Sanity override function should handle normal cases."""
        v3_output = {
            "ranked_securities": [
                {"ticker": "ACME", "composite_rank": 1, "composite_score": "75.00", "score_breakdown": {}, "flags": []},
                {"ticker": "BETA", "composite_rank": 2, "composite_score": "65.00", "score_breakdown": {}, "flags": []},
            ],
        }
        v2_output = {
            "ranked_securities": [
                {"ticker": "BETA", "composite_rank": 1, "composite_score": "70.00", "flags": []},
                {"ticker": "ACME", "composite_rank": 2, "composite_score": "68.00", "flags": []},
            ],
        }

        result = _apply_sanity_overrides(v3_output, v2_output)

        assert "sanity_overrides" in result
        assert result["sanity_overrides"]["enabled"] is True

    def test_sanity_override_empty_outputs(self):
        """Sanity override should handle empty outputs."""
        v3_output = {"ranked_securities": []}
        v2_output = {"ranked_securities": []}

        result = _apply_sanity_overrides(v3_output, v2_output)

        assert result["sanity_overrides"]["overrides_count"] == 0


# ============================================================================
# DEFENSIVE OVERLAY TESTS
# ============================================================================

class TestDefensiveOverlay:
    """Tests for defensive overlay integration."""

    def test_apply_defensive_multiplier_flag(
        self,
        as_of_date,
        sample_universe_result,
        sample_financial_result,
        sample_catalyst_result,
        sample_clinical_result,
    ):
        """Defensive multiplier flag should be passed through."""
        result = compute_module_5_composite_with_defensive(
            universe_result=sample_universe_result,
            financial_result=sample_financial_result,
            catalyst_result=sample_catalyst_result,
            clinical_result=sample_clinical_result,
            as_of_date=as_of_date,
            apply_defensive_multiplier=True,
            enable_sanity_override=False,
            validate=False,
        )

        assert "ranked_securities" in result

    def test_apply_position_sizing_flag(
        self,
        as_of_date,
        sample_universe_result,
        sample_financial_result,
        sample_catalyst_result,
        sample_clinical_result,
    ):
        """Position sizing flag should be passed through."""
        result = compute_module_5_composite_with_defensive(
            universe_result=sample_universe_result,
            financial_result=sample_financial_result,
            catalyst_result=sample_catalyst_result,
            clinical_result=sample_clinical_result,
            as_of_date=as_of_date,
            apply_position_sizing=True,
            enable_sanity_override=False,
            validate=False,
        )

        assert "ranked_securities" in result


# ============================================================================
# DETERMINISM TESTS
# ============================================================================

class TestDeterminism:
    """Tests verifying deterministic behavior."""

    def test_repeated_runs_identical(
        self,
        as_of_date,
        sample_universe_result,
        sample_financial_result,
        sample_catalyst_result,
        sample_clinical_result,
    ):
        """Multiple runs should produce identical results."""
        results = [
            compute_module_5_composite_with_defensive(
                universe_result=sample_universe_result,
                financial_result=sample_financial_result,
                catalyst_result=sample_catalyst_result,
                clinical_result=sample_clinical_result,
                as_of_date=as_of_date,
                enable_sanity_override=False,
                validate=False,
            )
            for _ in range(3)
        ]

        # Check ranked securities are identical
        for i in range(1, len(results)):
            assert len(results[0]["ranked_securities"]) == len(results[i]["ranked_securities"])
            for j in range(len(results[0]["ranked_securities"])):
                assert results[0]["ranked_securities"][j]["ticker"] == results[i]["ranked_securities"][j]["ticker"]
                assert results[0]["ranked_securities"][j]["composite_score"] == results[i]["ranked_securities"][j]["composite_score"]


# ============================================================================
# ENHANCEMENT RESULT TESTS
# ============================================================================

class TestEnhancementResult:
    """Tests for enhancement result handling."""

    def test_enhancement_result_passed_through(
        self,
        as_of_date,
        sample_universe_result,
        sample_financial_result,
        sample_catalyst_result,
        sample_clinical_result,
    ):
        """Enhancement result should be passed to underlying scorer."""
        enhancement_result = {
            "pos_scores": {"scores": [{"ticker": "ACME", "pos_score": "0.45"}]},
            "regime": {"regime": "NEUTRAL"},
        }

        result = compute_module_5_composite_with_defensive(
            universe_result=sample_universe_result,
            financial_result=sample_financial_result,
            catalyst_result=sample_catalyst_result,
            clinical_result=sample_clinical_result,
            as_of_date=as_of_date,
            enhancement_result=enhancement_result,
            enable_sanity_override=False,
            validate=False,
        )

        assert "ranked_securities" in result


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

class TestEdgeCases:
    """Edge case tests for the defensive wrapper."""

    def test_empty_universe(
        self,
        as_of_date,
        sample_financial_result,
        sample_catalyst_result,
        sample_clinical_result,
    ):
        """Should handle empty universe."""
        empty_universe = {"active_securities": [], "excluded_securities": []}

        result = compute_module_5_composite_with_defensive(
            universe_result=empty_universe,
            financial_result=sample_financial_result,
            catalyst_result=sample_catalyst_result,
            clinical_result=sample_clinical_result,
            as_of_date=as_of_date,
            enable_sanity_override=False,
            validate=False,
        )

        assert result["ranked_securities"] == []

    def test_single_security(
        self,
        as_of_date,
    ):
        """Should handle single security."""
        single_universe = {
            "active_securities": [{"ticker": "ONLY", "status": "active"}],
            "excluded_securities": [],
        }
        single_financial = {
            "scores": [{"ticker": "ONLY", "financial_score": "70", "severity": "none"}],
        }
        single_catalyst = {"summaries": {"ONLY": {"scores": {"score_blended": "60"}}}}
        single_clinical = {
            "scores": [{"ticker": "ONLY", "clinical_score": "65", "lead_phase": "phase 2", "trial_count": 1, "severity": "none"}],
        }

        result = compute_module_5_composite_with_defensive(
            universe_result=single_universe,
            financial_result=single_financial,
            catalyst_result=single_catalyst,
            clinical_result=single_clinical,
            as_of_date=as_of_date,
            enable_sanity_override=False,
            validate=False,
        )

        assert len(result["ranked_securities"]) == 1
