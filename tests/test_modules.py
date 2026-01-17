"""Tests for Modules 1-5."""
import pytest
from decimal import Decimal

from module_1_universe import compute_module_1_universe
from module_2_financial import compute_module_2_financial
from module_3_catalyst import compute_module_3_catalyst
from module_4_clinical_dev import compute_module_4_clinical_dev
from module_5_composite import compute_module_5_composite


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
        assert m5["provenance"]["ruleset_version"] == "1.2.1"
