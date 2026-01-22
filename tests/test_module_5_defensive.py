#!/usr/bin/env python3
"""
Tests for module_5_composite_with_defensive.py

The defensive wrapper adds position sizing and defensive overlays to Module 5.
These tests cover:
- Version selection (v1, v2, v3)
- Defensive overlay integration
- Sanity override mechanism
- Position sizing
"""

import pytest
from datetime import date
from decimal import Decimal
from unittest.mock import patch, MagicMock

from module_5_composite_with_defensive import (
    compute_module_5_composite_with_defensive,
    _apply_sanity_overrides,
)


class TestVersionSelection:
    """Tests for version selection logic."""

    @pytest.fixture
    def mock_inputs(self):
        """Create mock module inputs."""
        return {
            "universe_result": {
                "active_securities": [
                    {"ticker": "ACME", "market_cap_mm": 100},
                    {"ticker": "BETA", "market_cap_mm": 200},
                ],
                "excluded_securities": [],
            },
            "financial_result": {
                "scores": [
                    {"ticker": "ACME", "financial_score": "50.00"},
                    {"ticker": "BETA", "financial_score": "60.00"},
                ],
            },
            "catalyst_result": {
                "summaries": {
                    "ACME": {"catalyst_score_net": "30"},
                    "BETA": {"catalyst_score_net": "40"},
                },
            },
            "clinical_result": {
                "scores": [
                    {"ticker": "ACME", "clinical_score": "70.00", "lead_phase": "Phase 3"},
                    {"ticker": "BETA", "clinical_score": "65.00", "lead_phase": "Phase 2"},
                ],
            },
            "as_of_date": "2026-01-15",
        }

    @patch("module_5_composite_with_defensive.compute_module_5_composite_v3")
    @patch("module_5_composite_with_defensive.compute_module_5_composite_v2")
    def test_v3_is_default(self, mock_v2, mock_v3, mock_inputs):
        """V3 should be the default scorer."""
        mock_v3.return_value = {
            "ranked_securities": [
                {"ticker": "ACME", "composite_score": "65.00", "composite_rank": 1},
                {"ticker": "BETA", "composite_score": "60.00", "composite_rank": 2},
            ],
            "provenance": {"scorer": "v3"},
        }
        mock_v2.return_value = {
            "ranked_securities": [
                {"ticker": "ACME", "composite_score": "62.00", "composite_rank": 1},
                {"ticker": "BETA", "composite_score": "58.00", "composite_rank": 2},
            ],
            "provenance": {"scorer": "v2"},
        }

        result = compute_module_5_composite_with_defensive(**mock_inputs)

        mock_v3.assert_called_once()
        assert result["provenance"]["scorer"] == "v3"

    @patch("module_5_composite_with_defensive.compute_module_5_composite_v2")
    def test_v2_when_v3_disabled(self, mock_v2, mock_inputs):
        """V2 should be used when v3 disabled."""
        mock_v2.return_value = {
            "ranked_securities": [
                {"ticker": "ACME", "composite_score": "62.00", "composite_rank": 1},
            ],
            "provenance": {"scorer": "v2"},
        }

        result = compute_module_5_composite_with_defensive(
            **mock_inputs,
            use_v3_scoring=False,
        )

        mock_v2.assert_called()
        assert result["provenance"]["scorer"] == "v2"

    @patch("module_5_composite_with_defensive.compute_module_5_composite")
    def test_v1_when_both_disabled(self, mock_v1, mock_inputs):
        """V1 should be used when v2 and v3 disabled."""
        mock_v1.return_value = {
            "ranked_securities": [
                {"ticker": "ACME", "composite_score": "60.00", "composite_rank": 1},
            ],
            "provenance": {"scorer": "v1"},
        }

        result = compute_module_5_composite_with_defensive(
            **mock_inputs,
            use_v3_scoring=False,
            use_v2_scoring=False,
        )

        mock_v1.assert_called()


class TestSanityOverride:
    """Tests for sanity override mechanism."""

    def test_apply_sanity_overrides_no_overrides(self):
        """No overrides when ranks are similar."""
        output_v3 = {
            "ranked_securities": [
                {"ticker": "ACME", "composite_score": "65.00", "composite_rank": 1, "score_breakdown": {}},
                {"ticker": "BETA", "composite_score": "60.00", "composite_rank": 2, "score_breakdown": {}},
            ]
        }
        output_v2 = {
            "ranked_securities": [
                {"ticker": "ACME", "composite_score": "64.00", "composite_rank": 1},
                {"ticker": "BETA", "composite_score": "59.00", "composite_rank": 2},
            ]
        }

        result = _apply_sanity_overrides(output_v3, output_v2)

        assert result["sanity_overrides"]["overrides_count"] == 0
        # Ranks should be preserved (same order)
        assert result["ranked_securities"][0]["ticker"] == "ACME"

    @patch("module_5_composite_with_defensive.check_sanity_override")
    def test_override_applied_when_divergent(self, mock_check):
        """Override should be applied when ranks diverge significantly."""
        # Mock: ACME has massive rank divergence, fallback triggered
        mock_result = MagicMock()
        mock_result.fallback_to_v2 = True
        mock_result.override_reason = "massive_rank_divergence"
        mock_result.driving_factor = "experimental_signal"
        mock_result.confidence_level = Decimal("0.30")
        mock_result.override_applied = True
        mock_result.rank_divergence = 10
        mock_check.return_value = mock_result

        output_v3 = {
            "ranked_securities": [
                {"ticker": "ACME", "composite_score": "80.00", "composite_rank": 1, "score_breakdown": {}},
            ]
        }
        output_v2 = {
            "ranked_securities": [
                {"ticker": "ACME", "composite_score": "50.00", "composite_rank": 11},
            ]
        }

        result = _apply_sanity_overrides(output_v3, output_v2)

        assert result["sanity_overrides"]["overrides_count"] == 1
        assert result["ranked_securities"][0]["sanity_override"] is not None
        assert "sanity_override_applied" in result["ranked_securities"][0]["flags"]


class TestDefensiveOverlayIntegration:
    """Tests for defensive overlay integration."""

    @patch("module_5_composite_with_defensive.compute_module_5_composite_v3")
    @patch("module_5_composite_with_defensive.compute_module_5_composite_v2")
    @patch("module_5_composite_with_defensive.enrich_with_defensive_overlays")
    def test_defensive_overlays_applied(self, mock_enrich, mock_v2, mock_v3):
        """Defensive overlays should be applied to output."""
        mock_v3.return_value = {
            "ranked_securities": [{"ticker": "ACME", "composite_score": "65.00", "composite_rank": 1}],
            "provenance": {},
        }
        mock_v2.return_value = {
            "ranked_securities": [{"ticker": "ACME", "composite_score": "62.00", "composite_rank": 1}],
        }

        compute_module_5_composite_with_defensive(
            universe_result={"active_securities": [{"ticker": "ACME"}], "excluded_securities": []},
            financial_result={"scores": [{"ticker": "ACME", "financial_score": "50"}]},
            catalyst_result={"summaries": {"ACME": {"catalyst_score_net": "30"}}},
            clinical_result={"scores": [{"ticker": "ACME", "clinical_score": "70", "lead_phase": "Phase 3"}]},
            as_of_date="2026-01-15",
        )

        mock_enrich.assert_called_once()

    @patch("module_5_composite_with_defensive.compute_module_5_composite_v3")
    @patch("module_5_composite_with_defensive.compute_module_5_composite_v2")
    @patch("module_5_composite_with_defensive.enrich_with_defensive_overlays")
    def test_position_sizing_can_be_disabled(self, mock_enrich, mock_v2, mock_v3):
        """Position sizing should be controllable."""
        mock_v3.return_value = {"ranked_securities": [], "provenance": {}}
        mock_v2.return_value = {"ranked_securities": []}

        compute_module_5_composite_with_defensive(
            universe_result={"active_securities": [], "excluded_securities": []},
            financial_result={"scores": []},
            catalyst_result={"summaries": {}},
            clinical_result={"scores": []},
            as_of_date="2026-01-15",
            apply_position_sizing=False,
        )

        # Check that enrich was called with apply_position_sizing=False
        call_kwargs = mock_enrich.call_args.kwargs
        assert call_kwargs.get("apply_position_sizing") is False


class TestValidation:
    """Tests for validation functionality."""

    @patch("module_5_composite_with_defensive.compute_module_5_composite_v3")
    @patch("module_5_composite_with_defensive.compute_module_5_composite_v2")
    @patch("module_5_composite_with_defensive.enrich_with_defensive_overlays")
    @patch("module_5_composite_with_defensive.validate_defensive_integration")
    def test_validation_when_enabled(self, mock_validate, mock_enrich, mock_v2, mock_v3):
        """Validation should run when enabled."""
        mock_v3.return_value = {"ranked_securities": [], "provenance": {}}
        mock_v2.return_value = {"ranked_securities": []}

        compute_module_5_composite_with_defensive(
            universe_result={"active_securities": [], "excluded_securities": []},
            financial_result={"scores": []},
            catalyst_result={"summaries": {}},
            clinical_result={"scores": []},
            as_of_date="2026-01-15",
            validate=True,
        )

        mock_validate.assert_called_once()

    @patch("module_5_composite_with_defensive.compute_module_5_composite_v3")
    @patch("module_5_composite_with_defensive.compute_module_5_composite_v2")
    @patch("module_5_composite_with_defensive.enrich_with_defensive_overlays")
    @patch("module_5_composite_with_defensive.validate_defensive_integration")
    def test_validation_not_run_when_disabled(self, mock_validate, mock_enrich, mock_v2, mock_v3):
        """Validation should not run when disabled."""
        mock_v3.return_value = {"ranked_securities": [], "provenance": {}}
        mock_v2.return_value = {"ranked_securities": []}

        compute_module_5_composite_with_defensive(
            universe_result={"active_securities": [], "excluded_securities": []},
            financial_result={"scores": []},
            catalyst_result={"summaries": {}},
            clinical_result={"scores": []},
            as_of_date="2026-01-15",
            validate=False,
        )

        mock_validate.assert_not_called()


class TestWeightsPassthrough:
    """Tests for weights passthrough."""

    @patch("module_5_composite_with_defensive.compute_module_5_composite_v3")
    @patch("module_5_composite_with_defensive.compute_module_5_composite_v2")
    @patch("module_5_composite_with_defensive.enrich_with_defensive_overlays")
    def test_custom_weights_passed_to_v3(self, mock_enrich, mock_v2, mock_v3):
        """Custom weights should be passed to v3 scorer."""
        mock_v3.return_value = {"ranked_securities": [], "provenance": {}}
        mock_v2.return_value = {"ranked_securities": []}

        custom_weights = {
            "clinical": Decimal("0.50"),
            "financial": Decimal("0.30"),
            "catalyst": Decimal("0.20"),
        }

        compute_module_5_composite_with_defensive(
            universe_result={"active_securities": [], "excluded_securities": []},
            financial_result={"scores": []},
            catalyst_result={"summaries": {}},
            clinical_result={"scores": []},
            as_of_date="2026-01-15",
            weights=custom_weights,
        )

        call_kwargs = mock_v3.call_args.kwargs
        assert call_kwargs.get("weights") == custom_weights


class TestSanityOverrideToggle:
    """Tests for sanity override toggle."""

    @patch("module_5_composite_with_defensive.compute_module_5_composite_v3")
    @patch("module_5_composite_with_defensive.compute_module_5_composite_v2")
    @patch("module_5_composite_with_defensive.enrich_with_defensive_overlays")
    def test_sanity_override_disabled(self, mock_enrich, mock_v2, mock_v3):
        """Sanity override should be skippable."""
        mock_v3.return_value = {"ranked_securities": [], "provenance": {}}

        compute_module_5_composite_with_defensive(
            universe_result={"active_securities": [], "excluded_securities": []},
            financial_result={"scores": []},
            catalyst_result={"summaries": {}},
            clinical_result={"scores": []},
            as_of_date="2026-01-15",
            enable_sanity_override=False,
        )

        # V2 should NOT be called when sanity override is disabled
        mock_v2.assert_not_called()

    @patch("module_5_composite_with_defensive.compute_module_5_composite_v3")
    @patch("module_5_composite_with_defensive.compute_module_5_composite_v2")
    @patch("module_5_composite_with_defensive.enrich_with_defensive_overlays")
    def test_sanity_override_enabled(self, mock_enrich, mock_v2, mock_v3):
        """Sanity override should run v2 for comparison when enabled."""
        mock_v3.return_value = {"ranked_securities": [], "provenance": {}}
        mock_v2.return_value = {"ranked_securities": []}

        compute_module_5_composite_with_defensive(
            universe_result={"active_securities": [], "excluded_securities": []},
            financial_result={"scores": []},
            catalyst_result={"summaries": {}},
            clinical_result={"scores": []},
            as_of_date="2026-01-15",
            enable_sanity_override=True,  # Default
        )

        # V2 should be called for sanity comparison
        mock_v2.assert_called_once()
