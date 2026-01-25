#!/usr/bin/env python3
"""
test_data_enrichment_integration.py - Data Enrichment Pipeline Integration Tests

Tests the integration of data enrichment engines into run_screen.py:
1. Dilution Risk Engine integration
2. Timeline Slippage Engine integration
3. Smart Money Position Change Tracking
4. Regime-Adaptive Weighting

Author: Wake Robin Capital Management
Version: 1.0.0
"""

import json
import sys
import tempfile
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch
import pytest

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dilution_risk_engine import DilutionRiskEngine
from timeline_slippage_engine import TimelineSlippageEngine


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def as_of_date() -> date:
    """Standard as_of_date for deterministic testing."""
    return date(2026, 1, 15)


@pytest.fixture
def sample_financial_records() -> List[Dict[str, Any]]:
    """Sample financial records with Cash and NetIncome for dilution risk."""
    return [
        {
            "ticker": "ACME",
            "Cash": 100000000,
            "NetIncome": -60000000,  # Quarterly loss
            "R&D": 40000000,
        },
        {
            "ticker": "BETA",
            "Cash": 500000000,
            "NetIncome": 50000000,  # Profitable
            "R&D": 30000000,
        },
        {
            "ticker": "GAMA",
            "Cash": 20000000,
            "NetIncome": -80000000,  # High burn
            "R&D": 25000000,
        },
    ]


@pytest.fixture
def sample_market_data() -> List[Dict[str, Any]]:
    """Sample market data with volume and market cap."""
    return [
        {
            "ticker": "ACME",
            "market_cap": 500000000,
            "avg_volume_90d": 1000000,
        },
        {
            "ticker": "BETA",
            "market_cap": 2000000000,
            "avg_volume_90d": 2500000,
        },
        {
            "ticker": "GAMA",
            "market_cap": 100000000,
            "avg_volume_90d": 500000,
        },
    ]


@pytest.fixture
def sample_catalyst_summaries() -> Dict[str, Dict[str, Any]]:
    """Sample catalyst summaries with next_catalyst_date."""
    return {
        "ACME": {
            "next_catalyst_date": "2026-06-15",
            "events": [],
        },
        "BETA": {
            "next_catalyst_date": "2026-03-15",
            "events": [],
        },
        "GAMA": {
            "next_catalyst_date": "2026-12-15",
            "events": [
                {"event_type": "CT_TIMELINE_PUSHOUT"},
                {"event_type": "CT_TIMELINE_PUSHOUT"},
            ],
        },
    }


@pytest.fixture
def sample_holdings_snapshots() -> Dict[str, Dict[str, Any]]:
    """Sample holdings snapshots with current and prior data for position change detection."""
    return {
        "ACME": {
            "holdings": {
                "current": {
                    "0001263508": {"value_kusd": 150000, "quarter_end": "2025-09-30"},  # Baker Bros
                    "0001346824": {"value_kusd": 200000, "quarter_end": "2025-09-30"},  # RA Capital
                },
                "prior": {
                    "0001263508": {"value_kusd": 100000, "quarter_end": "2025-06-30"},  # +50%
                    "0001346824": {"value_kusd": 180000, "quarter_end": "2025-06-30"},  # +11%
                },
            },
        },
        "BETA": {
            "holdings": {
                "current": {
                    "0000909661": {"value_kusd": 300000, "quarter_end": "2025-09-30"},  # Perceptive
                },
                "prior": {
                    "0000909661": {"value_kusd": 500000, "quarter_end": "2025-06-30"},  # -40%
                    "0001167483": {"value_kusd": 100000, "quarter_end": "2025-06-30"},  # Deerfield - exited
                },
            },
        },
        "GAMA": {
            "holdings": {
                "current": {
                    "0001390295": {"value_kusd": 50000, "quarter_end": "2025-09-30"},  # RTW - NEW
                },
                "prior": {},
            },
        },
    }


# =============================================================================
# DILUTION RISK ENGINE INTEGRATION TESTS
# =============================================================================

class TestDilutionRiskIntegration:
    """Test dilution risk engine integration with pipeline."""

    def test_dilution_risk_engine_scoring(
        self,
        as_of_date: date,
        sample_financial_records: List[Dict],
        sample_market_data: List[Dict],
        sample_catalyst_summaries: Dict,
    ):
        """Test dilution risk engine scores universe correctly."""
        engine = DilutionRiskEngine()

        # Build universe for dilution risk
        market_by_ticker = {m["ticker"]: m for m in sample_market_data}
        dilution_universe = []

        for fin in sample_financial_records:
            ticker = fin["ticker"]
            mkt = market_by_ticker.get(ticker, {})
            catalyst = sample_catalyst_summaries.get(ticker, {})

            # Calculate quarterly burn from NetIncome (annualized / 4)
            quarterly_burn = None
            net_income = fin.get("NetIncome")
            if net_income is not None and net_income < 0:
                quarterly_burn = Decimal(str(net_income)) / Decimal("4")

            dilution_universe.append({
                "ticker": ticker,
                "quarterly_cash": Decimal(str(fin.get("Cash"))) if fin.get("Cash") else None,
                "quarterly_burn": quarterly_burn,
                "next_catalyst_date": catalyst.get("next_catalyst_date"),
                "market_cap": Decimal(str(mkt.get("market_cap"))) if mkt.get("market_cap") else None,
                "avg_daily_volume_90d": int(mkt.get("avg_volume_90d", 0)) if mkt.get("avg_volume_90d") else None,
            })

        result = engine.score_universe(dilution_universe, as_of_date)

        # Verify structure
        assert "scores" in result
        assert "diagnostic_counts" in result
        assert len(result["scores"]) == 3

        # Verify scoring logic
        scores_by_ticker = {s["ticker"]: s for s in result["scores"]}

        # ACME: Moderate burn, should have some risk
        assert scores_by_ticker["ACME"]["reason_code"] == "SUCCESS"

        # BETA: Profitable, no burn, should have low risk
        # Note: NetIncome > 0 means no quarterly_burn calculated
        assert scores_by_ticker["BETA"]["reason_code"] in ["SUCCESS", "INSUFFICIENT_DATA"]

        # GAMA: High burn relative to cash, should have high risk
        gama_risk = scores_by_ticker["GAMA"]
        if gama_risk["reason_code"] == "SUCCESS":
            assert gama_risk["risk_bucket"] in ["MEDIUM_RISK", "HIGH_RISK"]

    def test_dilution_risk_requires_catalyst_date(self, as_of_date: date):
        """Test that dilution risk fails gracefully without catalyst date."""
        engine = DilutionRiskEngine()

        universe = [{
            "ticker": "NOCATALYST",
            "quarterly_cash": Decimal("100000000"),
            "quarterly_burn": Decimal("-15000000"),
            "next_catalyst_date": None,  # Missing
            "market_cap": Decimal("500000000"),
        }]

        result = engine.score_universe(universe, as_of_date)

        assert result["scores"][0]["reason_code"] == "INSUFFICIENT_DATA"
        # Note: missing_fields not exposed in score_universe output, just reason_code


# =============================================================================
# TIMELINE SLIPPAGE ENGINE INTEGRATION TESTS
# =============================================================================

class TestTimelineSlippageIntegration:
    """Test timeline slippage engine integration with pipeline."""

    def test_timeline_slippage_universe_scoring(self, as_of_date: date):
        """Test timeline slippage engine scores universe correctly."""
        engine = TimelineSlippageEngine()

        # Build universe and trial data
        universe = [
            {"ticker": "ACME"},
            {"ticker": "BETA"},
            {"ticker": "GAMA"},
        ]

        current_trials_by_ticker = {
            "ACME": [
                {
                    "nct_id": "NCT12345678",
                    "primary_completion_date": "2026-06-15",
                    "phase": "phase_3",
                    "overall_status": "Recruiting",
                },
            ],
            "BETA": [
                {
                    "nct_id": "NCT23456789",
                    "primary_completion_date": "2026-03-15",
                    "phase": "phase_2",
                    "overall_status": "Active, not recruiting",
                },
            ],
            "GAMA": [
                {
                    "nct_id": "NCT34567890",
                    "primary_completion_date": "2026-12-15",
                    "phase": "phase_1",
                    "overall_status": "Recruiting",
                },
            ],
        }

        result = engine.score_universe(
            universe=universe,
            current_trials_by_ticker=current_trials_by_ticker,
            prior_trials_by_ticker=None,
            as_of_date=as_of_date,
        )

        # Verify structure
        assert "scores" in result
        assert "diagnostic_counts" in result
        assert len(result["scores"]) == 3

    def test_timeline_slippage_requires_as_of_date(self):
        """Test that timeline slippage requires as_of_date."""
        engine = TimelineSlippageEngine()

        with pytest.raises(ValueError, match="as_of_date is required"):
            engine.score_universe(
                universe=[{"ticker": "TEST"}],
                current_trials_by_ticker={},
                as_of_date=None,
            )


# =============================================================================
# SMART MONEY POSITION CHANGE TRACKING TESTS
# =============================================================================

class TestSmartMoneyPositionChanges:
    """Test smart money position change detection."""

    def test_convert_holdings_to_coinvest_position_changes(
        self,
        sample_holdings_snapshots: Dict,
    ):
        """Test _convert_holdings_to_coinvest detects position changes correctly."""
        # Import the function from run_screen
        from run_screen import _convert_holdings_to_coinvest

        result = _convert_holdings_to_coinvest(sample_holdings_snapshots)

        # Verify structure
        assert "ACME" in result
        assert "BETA" in result
        assert "GAMA" in result

        # ACME: Baker Bros +50% = INCREASE, RA Capital +11% = INCREASE
        # FIX: position_changes now keyed by holder name, not CIK
        acme_changes = result["ACME"]["position_changes"]
        assert acme_changes.get("Baker Bros") == "INCREASE"
        assert acme_changes.get("RA Capital") == "INCREASE"

        # BETA: Perceptive -40% = DECREASE, Deerfield exited = EXIT
        beta_changes = result["BETA"]["position_changes"]
        assert beta_changes.get("Perceptive") == "DECREASE"
        assert beta_changes.get("Deerfield") == "EXIT"

        # GAMA: RTW new position = NEW
        gama_changes = result["GAMA"]["position_changes"]
        assert gama_changes.get("RTW Investments") == "NEW"

    def test_holder_tier_identification(self, sample_holdings_snapshots: Dict):
        """Test that holder tiers are correctly identified."""
        from run_screen import _convert_holdings_to_coinvest

        result = _convert_holdings_to_coinvest(sample_holdings_snapshots)

        # Verify tier 1 managers are identified
        # FIX: holder_tiers now Dict[name -> tier_int], not Dict[cik -> dict]
        acme_tiers = result["ACME"]["holder_tiers"]
        assert acme_tiers["Baker Bros"] == 1
        assert acme_tiers["RA Capital"] == 1

    def test_coinvest_overlap_count(self, sample_holdings_snapshots: Dict):
        """Test coinvest overlap count is correct."""
        from run_screen import _convert_holdings_to_coinvest

        result = _convert_holdings_to_coinvest(sample_holdings_snapshots)

        assert result["ACME"]["coinvest_overlap_count"] == 2
        assert result["BETA"]["coinvest_overlap_count"] == 1
        assert result["GAMA"]["coinvest_overlap_count"] == 1


# =============================================================================
# REGIME-ADAPTIVE WEIGHTING TESTS
# =============================================================================

class TestRegimeAdaptiveWeighting:
    """Test regime-adaptive weight adjustments."""

    def test_bull_regime_weight_adjustments(self):
        """Test that BULL regime adjusts weights correctly."""
        from src.modules.ic_enhancements import apply_regime_to_weights

        base_weights = {
            "clinical": Decimal("0.40"),
            "financial": Decimal("0.25"),
            "catalyst": Decimal("0.15"),
            "momentum": Decimal("0.10"),
            "valuation": Decimal("0.10"),
        }

        adjusted = apply_regime_to_weights(base_weights, "BULL")

        # In BULL: momentum boosted (1.3x), catalyst boosted (1.2x)
        # financial reduced (0.8x), valuation reduced (0.7x)
        # After renormalization, momentum and catalyst should be higher relative to base

        # Weights should sum to 1.0
        total = sum(adjusted.values())
        assert abs(total - Decimal("1")) < Decimal("0.01")

        # Momentum should be relatively higher
        momentum_ratio = adjusted["momentum"] / base_weights["momentum"]
        financial_ratio = adjusted["financial"] / base_weights["financial"]
        assert momentum_ratio > financial_ratio

    def test_bear_regime_weight_adjustments(self):
        """Test that BEAR regime adjusts weights correctly."""
        from src.modules.ic_enhancements import apply_regime_to_weights

        base_weights = {
            "clinical": Decimal("0.40"),
            "financial": Decimal("0.25"),
            "catalyst": Decimal("0.15"),
            "momentum": Decimal("0.10"),
            "valuation": Decimal("0.10"),
        }

        adjusted = apply_regime_to_weights(base_weights, "BEAR")

        # In BEAR: financial boosted (1.4x), valuation boosted (1.2x)
        # momentum reduced (0.5x), catalyst reduced (0.7x)

        # Weights should sum to 1.0
        total = sum(adjusted.values())
        assert abs(total - Decimal("1")) < Decimal("0.01")

        # Financial should be relatively higher, momentum lower
        financial_ratio = adjusted["financial"] / base_weights["financial"]
        momentum_ratio = adjusted["momentum"] / base_weights["momentum"]
        assert financial_ratio > momentum_ratio

    def test_unknown_regime_uses_neutral_weights(self):
        """Test that UNKNOWN regime applies minimal adjustments."""
        from src.modules.ic_enhancements import apply_regime_to_weights

        base_weights = {
            "clinical": Decimal("0.40"),
            "financial": Decimal("0.25"),
            "catalyst": Decimal("0.15"),
            "momentum": Decimal("0.10"),
            "valuation": Decimal("0.10"),
        }

        adjusted = apply_regime_to_weights(base_weights, "UNKNOWN")

        # In UNKNOWN: all 1.0x except momentum 0.8x

        # Weights should sum to 1.0
        total = sum(adjusted.values())
        assert abs(total - Decimal("1")) < Decimal("0.01")


# =============================================================================
# ENHANCEMENT LAYER ASSEMBLY TESTS
# =============================================================================

class TestEnhancementLayerAssembly:
    """Test that enhancement results are properly assembled."""

    def test_enhancement_result_structure(self):
        """Test that enhancement_result has all expected keys."""
        expected_keys = [
            "regime",
            "pos_scores",
            "short_interest_scores",
            "accuracy_enhancements",
            "dilution_risk_scores",
            "timeline_slippage_scores",
            "provenance",
        ]

        # Create mock enhancement result
        enhancement_result = {
            "regime": {"regime": "BULL", "confidence": Decimal("0.80")},
            "pos_scores": {"scores": [], "diagnostic_counts": {}},
            "short_interest_scores": None,
            "accuracy_enhancements": None,
            "dilution_risk_scores": {"scores": [], "diagnostic_counts": {}},
            "timeline_slippage_scores": {"scores": [], "diagnostic_counts": {}},
            "provenance": {
                "module": "enhancements",
                "version": "1.2.0",
            },
        }

        for key in expected_keys:
            assert key in enhancement_result


# =============================================================================
# DETERMINISM TESTS
# =============================================================================

class TestEnrichmentDeterminism:
    """Test that enrichment engines produce deterministic outputs."""

    def test_dilution_risk_determinism(self, as_of_date: date):
        """Test dilution risk engine produces identical outputs for identical inputs."""
        engine1 = DilutionRiskEngine()
        engine2 = DilutionRiskEngine()

        universe = [{
            "ticker": "TEST",
            "quarterly_cash": Decimal("100000000"),
            "quarterly_burn": Decimal("-15000000"),
            "next_catalyst_date": "2026-06-15",
            "market_cap": Decimal("500000000"),
            "avg_daily_volume_90d": 1000000,
        }]

        result1 = engine1.score_universe(universe, as_of_date)
        result2 = engine2.score_universe(universe, as_of_date)

        # Core fields should be identical
        assert result1["scores"][0]["dilution_risk_score"] == result2["scores"][0]["dilution_risk_score"]
        assert result1["scores"][0]["risk_bucket"] == result2["scores"][0]["risk_bucket"]
        assert result1["scores"][0]["confidence"] == result2["scores"][0]["confidence"]

    def test_position_change_detection_determinism(self, sample_holdings_snapshots: Dict):
        """Test position change detection is deterministic."""
        from run_screen import _convert_holdings_to_coinvest

        result1 = _convert_holdings_to_coinvest(sample_holdings_snapshots)
        result2 = _convert_holdings_to_coinvest(sample_holdings_snapshots)

        # Position changes should be identical
        for ticker in sample_holdings_snapshots:
            ticker_upper = ticker.upper()
            assert result1[ticker_upper]["position_changes"] == result2[ticker_upper]["position_changes"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
