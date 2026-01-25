"""Tests for Modules 1-5."""
import os
import pytest
from decimal import Decimal

from module_1_universe import compute_module_1_universe
from module_2_financial import compute_module_2_financial
from module_3_catalyst import compute_module_3_catalyst
from module_4_clinical_dev import compute_module_4_clinical_dev
from module_5_composite import compute_module_5_composite


@pytest.fixture(autouse=True)
def set_validation_mode_warn():
    """Disable strict schema validation for test fixtures."""
    old_mode = os.environ.get("IC_VALIDATION_MODE")
    os.environ["IC_VALIDATION_MODE"] = "warn"
    yield
    if old_mode is None:
        os.environ.pop("IC_VALIDATION_MODE", None)
    else:
        os.environ["IC_VALIDATION_MODE"] = old_mode


class TestModule1Universe:
    def test_active_securities(self):
        records = [
            {"ticker": "AAPL", "company_name": "Apple Inc", "market_cap_mm": 3000000},
            {"ticker": "TINY", "company_name": "Tiny Corp", "market_cap_mm": 10},
        ]
        result = compute_module_1_universe(records, "2024-01-01")
        
        active = [s["ticker"] for s in result["active_securities"]]
        excluded = [s["ticker"] for s in result["excluded_securities"]]
        
        assert "AAPL" in active
        assert "TINY" in excluded
    
    def test_shell_exclusion(self):
        records = [
            {"ticker": "SPAC", "company_name": "Blank Check Acquisition Corp", "market_cap_mm": 500},
        ]
        result = compute_module_1_universe(records, "2024-01-01")
        
        assert len(result["active_securities"]) == 0
        assert result["excluded_securities"][0]["reason"] == "excluded_shell"


class TestModule2Financial:
    def test_financial_scoring(self):
        records = [
            {"ticker": "TEST", "cash_mm": 1000, "debt_mm": 100, "burn_rate_mm": 30,
             "market_cap_mm": 5000, "source_date": "2023-12-30"},
        ]
        result = compute_module_2_financial(records, ["TEST"], "2024-01-01")

        assert len(result["scores"]) == 1
        score = result["scores"][0]
        assert score["ticker"] == "TEST"
        assert float(score["financial_normalized"]) > 0
        assert float(score["runway_months"]) > 0
    
    def test_pit_filtering(self):
        records = [
            {"ticker": "OLD", "cash_mm": 100, "source_date": "2023-12-30"},
            {"ticker": "FUTURE", "cash_mm": 200, "source_date": "2024-01-02"},
        ]
        result = compute_module_2_financial(records, ["OLD", "FUTURE"], "2024-01-01")
        
        # FUTURE should be excluded (after PIT cutoff)
        tickers = [s["ticker"] for s in result["scores"]]
        assert "OLD" in tickers
        assert "FUTURE" not in tickers


class TestModule3Catalyst:
    """
    NOTE: Module 3 vNext uses a different API signature.
    These tests verify the legacy compatibility interface.
    For vNext tests, see tests/test_module_3_vnext.py
    """

    def test_catalyst_scoring_vnext(self):
        """Test that vNext API works (smoke test)."""
        from datetime import date
        from decimal import Decimal
        from module_3_schema import (
            EventType, EventSeverity, ConfidenceLevel, CatalystEventV2
        )
        from module_3_scoring import calculate_ticker_catalyst_score

        # Create sample events
        events = [
            CatalystEventV2(
                ticker="TEST",
                nct_id="NCT12345",
                event_type=EventType.CT_STATUS_UPGRADE,
                event_severity=EventSeverity.POSITIVE,
                event_date="2024-01-15",
                field_changed="status",
                prior_value="recruiting",
                new_value="active",
                source="CTGOV",
                confidence=ConfidenceLevel.HIGH,
                disclosed_at="2024-01-15",
            )
        ]

        as_of = date(2024, 1, 31)
        summary = calculate_ticker_catalyst_score("TEST", events, as_of)

        assert summary.ticker == "TEST"
        assert summary.score_override >= Decimal("0")
        assert summary.events_total == 1

    def test_no_catalyst_vnext(self):
        """Test empty events with vNext API."""
        from datetime import date
        from decimal import Decimal
        from module_3_scoring import calculate_ticker_catalyst_score, SCORE_DEFAULT

        as_of = date(2024, 1, 31)
        summary = calculate_ticker_catalyst_score("TEST", [], as_of)

        assert summary.ticker == "TEST"
        assert summary.score_override == SCORE_DEFAULT
        assert summary.events_total == 0


class TestModule4ClinicalDev:
    def test_clinical_scoring(self):
        records = [
            {"ticker": "TEST", "nct_id": "NCT12345", "phase": "phase 3",
             "status": "completed", "randomized": True, "blinded": "double",
             "primary_endpoint": "overall survival"},
        ]
        result = compute_module_4_clinical_dev(records, ["TEST"], "2024-01-01")
        
        assert len(result["scores"]) == 1
        score = result["scores"][0]
        assert score["lead_phase"] == "phase 3"
        assert float(score["clinical_score"]) > 50  # High score for P3 + good design
    
    def test_early_stage(self):
        records = [
            {"ticker": "EARLY", "nct_id": "NCT12345", "phase": "phase 1",
             "primary_endpoint": "safety"},
        ]
        result = compute_module_4_clinical_dev(records, ["EARLY"], "2024-01-01")
        
        score = result["scores"][0]
        assert score["lead_phase"] == "phase 1"
        assert "early_stage" in score["flags"]


class TestModule5Composite:
    @pytest.fixture
    def sample_inputs(self):
        universe = {
            "active_securities": [{"ticker": "A"}, {"ticker": "B"}]
        }
        financial = {
            "scores": [
                {"ticker": "A", "financial_score": "70", "market_cap_mm": "5000", "severity": "none", "flags": []},
                {"ticker": "B", "financial_score": "50", "market_cap_mm": "500", "severity": "none", "flags": []},
            ]
        }
        catalyst = {
            "scores": [
                {"ticker": "A", "catalyst_score": "40", "severity": "none", "flags": []},
                {"ticker": "B", "catalyst_score": "30", "severity": "none", "flags": []},
            ]
        }
        clinical = {
            "scores": [
                {"ticker": "A", "clinical_score": "80", "lead_phase": "phase 3", "severity": "none", "flags": []},
                {"ticker": "B", "clinical_score": "40", "lead_phase": "phase 1", "severity": "none", "flags": []},
            ]
        }
        return universe, financial, catalyst, clinical
    
    def test_composite_ranking(self, sample_inputs):
        u, f, c, cl = sample_inputs
        # validate_inputs=False for minimal test fixtures
        result = compute_module_5_composite(u, f, c, cl, "2024-01-01", validate_inputs=False)

        assert len(result["ranked_securities"]) == 2

        # A should rank higher (better scores across the board)
        ranked = result["ranked_securities"]
        assert ranked[0]["ticker"] == "A"
        assert ranked[0]["composite_rank"] == 1
        assert ranked[1]["ticker"] == "B"
        assert ranked[1]["composite_rank"] == 2

    def test_cohort_normalization(self, sample_inputs):
        u, f, c, cl = sample_inputs
        # validate_inputs=False for minimal test fixtures
        result = compute_module_5_composite(u, f, c, cl, "2024-01-01", validate_inputs=False)

        # Check normalized scores exist
        for sec in result["ranked_securities"]:
            assert sec["clinical_dev_normalized"] is not None
            assert sec["financial_normalized"] is not None
            assert sec["catalyst_normalized"] is not None


class TestRegimeWeightNormalization:
    """Tests for regime weight normalization (issue: weights must sum to 1.0 across regimes)."""

    @pytest.fixture
    def sample_inputs_with_regime(self):
        """Sample inputs for regime weight testing."""
        universe = {
            "active_securities": [{"ticker": "A"}, {"ticker": "B"}]
        }
        financial = {
            "scores": [
                {"ticker": "A", "financial_score": "70", "market_cap_mm": "5000", "severity": "none", "flags": []},
                {"ticker": "B", "financial_score": "50", "market_cap_mm": "500", "severity": "none", "flags": []},
            ]
        }
        catalyst = {
            "scores": [
                {"ticker": "A", "catalyst_score": "40", "severity": "none", "flags": []},
                {"ticker": "B", "catalyst_score": "30", "severity": "none", "flags": []},
            ]
        }
        clinical = {
            "scores": [
                {"ticker": "A", "clinical_score": "80", "lead_phase": "phase 3", "severity": "none", "flags": []},
                {"ticker": "B", "clinical_score": "40", "lead_phase": "phase 1", "severity": "none", "flags": []},
            ]
        }
        return universe, financial, catalyst, clinical

    def test_regime_weights_sum_to_one_bull(self, sample_inputs_with_regime):
        """BULL regime: weights must sum to 1.0 after normalization."""
        from decimal import Decimal
        u, f, c, cl = sample_inputs_with_regime

        # BULL regime adjustments (from regime_engine.py)
        enhancement_result = {
            "regime": {
                "regime": "BULL",
                "signal_adjustments": {
                    "momentum": Decimal("1.20"),
                    "clinical": Decimal("1.10"),
                    "financial": Decimal("0.95"),
                    "catalyst": Decimal("1.15"),
                },
            },
            "pos_scores": {"scores": []},  # No PoS data
        }

        result = compute_module_5_composite(
            u, f, c, cl, "2024-01-01",
            enhancement_result=enhancement_result,
            validate_inputs=False
        )

        # Weights must sum to 1.0
        weights_used = result.get("weights_used", {})
        total = sum(Decimal(v) for v in weights_used.values())
        assert abs(total - Decimal("1.0")) < Decimal("0.001"), f"Weights sum to {total}, expected 1.0"

        # Check audit trail includes regime info
        diag = result.get("enhancement_diagnostics", {})
        assert diag.get("regime_weights_applied") is True
        # Allow small rounding tolerance in audit sum (0.9999-1.0001 acceptable)
        effective_sum = Decimal(diag.get("effective_weights_sum", "0"))
        assert abs(effective_sum - Decimal("1.0")) < Decimal("0.001")

    def test_regime_weights_sum_to_one_bear(self, sample_inputs_with_regime):
        """BEAR regime: weights must sum to 1.0 after normalization."""
        from decimal import Decimal
        u, f, c, cl = sample_inputs_with_regime

        # BEAR regime adjustments (from regime_engine.py)
        enhancement_result = {
            "regime": {
                "regime": "BEAR",
                "signal_adjustments": {
                    "momentum": Decimal("0.80"),
                    "clinical": Decimal("1.05"),
                    "financial": Decimal("1.20"),
                    "catalyst": Decimal("0.90"),
                },
            },
            "pos_scores": {"scores": []},
        }

        result = compute_module_5_composite(
            u, f, c, cl, "2024-01-01",
            enhancement_result=enhancement_result,
            validate_inputs=False
        )

        weights_used = result.get("weights_used", {})
        total = sum(Decimal(v) for v in weights_used.values())
        assert abs(total - Decimal("1.0")) < Decimal("0.001"), f"Weights sum to {total}, expected 1.0"

    def test_regime_weights_sum_to_one_volatility_spike(self, sample_inputs_with_regime):
        """VOLATILITY_SPIKE regime: weights must sum to 1.0 after normalization."""
        from decimal import Decimal
        u, f, c, cl = sample_inputs_with_regime

        # VOLATILITY_SPIKE regime adjustments
        enhancement_result = {
            "regime": {
                "regime": "VOLATILITY_SPIKE",
                "signal_adjustments": {
                    "momentum": Decimal("0.70"),
                    "clinical": Decimal("0.90"),
                    "financial": Decimal("1.25"),
                    "catalyst": Decimal("0.80"),
                },
            },
            "pos_scores": {"scores": []},
        }

        result = compute_module_5_composite(
            u, f, c, cl, "2024-01-01",
            enhancement_result=enhancement_result,
            validate_inputs=False
        )

        weights_used = result.get("weights_used", {})
        total = sum(Decimal(v) for v in weights_used.values())
        assert abs(total - Decimal("1.0")) < Decimal("0.001"), f"Weights sum to {total}, expected 1.0"

    def test_regime_adjustments_without_pos_data(self, sample_inputs_with_regime):
        """Regime adjustments should apply even without PoS data."""
        from decimal import Decimal
        u, f, c, cl = sample_inputs_with_regime

        # Enhancement result with regime but NO PoS scores
        enhancement_result = {
            "regime": {
                "regime": "BEAR",
                "signal_adjustments": {
                    "momentum": Decimal("0.80"),
                    "clinical": Decimal("1.05"),
                    "financial": Decimal("1.20"),
                    "catalyst": Decimal("0.90"),
                },
            },
            "pos_scores": {"scores": []},  # Empty PoS
        }

        result = compute_module_5_composite(
            u, f, c, cl, "2024-01-01",
            enhancement_result=enhancement_result,
            validate_inputs=False
        )

        # Regime weights should still be applied
        diag = result.get("enhancement_diagnostics", {})
        assert diag.get("regime_weights_applied") is True
        assert diag.get("regime") == "BEAR"

    def test_regime_audit_trail_included(self, sample_inputs_with_regime):
        """Audit trail should include base weights and effective weights."""
        from decimal import Decimal
        u, f, c, cl = sample_inputs_with_regime

        enhancement_result = {
            "regime": {
                "regime": "BULL",
                "signal_adjustments": {
                    "momentum": Decimal("1.20"),
                    "clinical": Decimal("1.10"),
                    "financial": Decimal("0.95"),
                    "catalyst": Decimal("1.15"),
                },
            },
            "pos_scores": {"scores": []},
        }

        result = compute_module_5_composite(
            u, f, c, cl, "2024-01-01",
            enhancement_result=enhancement_result,
            validate_inputs=False
        )

        diag = result.get("enhancement_diagnostics", {})

        # Must include all audit fields
        assert "base_weights" in diag
        assert "effective_weights" in diag
        assert "effective_weights_sum" in diag
        assert "regime_weights_applied" in diag

        # Base weights should be the default weights
        base = diag["base_weights"]
        assert "clinical_dev" in base
        assert "financial" in base
        assert "catalyst" in base

    def test_all_regimes_produce_normalized_weights(self, sample_inputs_with_regime):
        """All regime types must produce weights that sum to 1.0."""
        from decimal import Decimal
        u, f, c, cl = sample_inputs_with_regime

        # Test all regime types
        regimes = {
            "BULL": {
                "momentum": Decimal("1.20"),
                "clinical": Decimal("1.10"),
                "financial": Decimal("0.95"),
                "catalyst": Decimal("1.15"),
            },
            "BEAR": {
                "momentum": Decimal("0.80"),
                "clinical": Decimal("1.05"),
                "financial": Decimal("1.20"),
                "catalyst": Decimal("0.90"),
            },
            "VOLATILITY_SPIKE": {
                "momentum": Decimal("0.70"),
                "clinical": Decimal("0.90"),
                "financial": Decimal("1.25"),
                "catalyst": Decimal("0.80"),
            },
            "SECTOR_ROTATION": {
                "momentum": Decimal("1.00"),
                "clinical": Decimal("1.00"),
                "financial": Decimal("1.05"),
                "catalyst": Decimal("1.00"),
            },
        }

        for regime_name, adjustments in regimes.items():
            enhancement_result = {
                "regime": {
                    "regime": regime_name,
                    "signal_adjustments": adjustments,
                },
                "pos_scores": {"scores": []},
            }

            result = compute_module_5_composite(
                u, f, c, cl, "2024-01-01",
                enhancement_result=enhancement_result,
                validate_inputs=False
            )

            weights_used = result.get("weights_used", {})
            total = sum(Decimal(v) for v in weights_used.values())
            assert abs(total - Decimal("1.0")) < Decimal("0.001"), \
                f"{regime_name}: Weights sum to {total}, expected 1.0"


class TestIntegration:
    def test_full_pipeline(self):
        """
        Test running all modules in sequence.

        Note: Module 3 vNext requires file-based inputs.
        This test verifies modules 1, 2, 4, 5 integration with mock M3 data.
        """
        from datetime import date
        from decimal import Decimal

        # Universe
        universe_data = [
            {"ticker": "TEST", "company_name": "Test Corp", "market_cap_mm": 5000},
        ]
        m1 = compute_module_1_universe(universe_data, "2024-01-01")
        active = [s["ticker"] for s in m1["active_securities"]]

        # Financial
        financial_data = [
            {"ticker": "TEST", "cash_mm": 500, "debt_mm": 50, "burn_rate_mm": 15,
             "market_cap_mm": 5000, "source_date": "2023-12-30"},
        ]
        m2 = compute_module_2_financial(financial_data, active, "2024-01-01")

        # Mock Catalyst result (since vNext requires file-based inputs)
        # Use the legacy format that Module 5 expects
        from catalyst_summary import TickerCatalystSummary
        from event_detector import CatalystEvent, EventType as LegacyEventType

        mock_summary = TickerCatalystSummary(
            ticker="TEST",
            as_of_date=date(2024, 1, 1),
            catalyst_score_pos=2.5,
            catalyst_score_neg=0.0,
            catalyst_score_net=2.5,
            nearest_positive_days=150,
            nearest_negative_days=None,
            severe_negative_flag=False,
            events=[],
        )

        m3 = {
            "summaries": {"TEST": mock_summary},
            "summaries_legacy": {"TEST": mock_summary},
            "diagnostic_counts": {"events_detected": 0, "severe_negatives": 0},
            "as_of_date": "2024-01-01",
        }

        # Clinical
        trial_data = [
            {"ticker": "TEST", "nct_id": "NCT123", "phase": "phase 2",
             "primary_completion_date": "2024-06-01", "status": "recruiting",
             "randomized": True, "blinded": "double", "primary_endpoint": "response"},
        ]
        m4 = compute_module_4_clinical_dev(trial_data, active, "2024-01-01")

        # Composite
        m5 = compute_module_5_composite(m1, m2, m3, m4, "2024-01-01")

        # Verify output
        assert m5["diagnostic_counts"]["rankable"] == 1
        assert m5["ranked_securities"][0]["ticker"] == "TEST"
        assert m5["provenance"]["ruleset_version"] == "1.2.2"
