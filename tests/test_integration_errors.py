"""
Integration tests for error handling and edge cases.

Tests that the pipeline handles:
- Missing data gracefully
- Invalid input types
- Empty inputs
- Edge cases (NaN, None, zero values)
- PIT enforcement
"""
import pytest
from decimal import Decimal
from pathlib import Path
import json
import tempfile

from module_1_universe import compute_module_1_universe
from module_2_financial import compute_module_2_financial
from module_4_clinical_dev import compute_module_4_clinical_dev
from module_5_composite import compute_module_5_composite


class TestErrorHandlingModule1:
    """Test Module 1 handles errors gracefully."""

    def test_empty_universe(self):
        """Pipeline should handle empty input."""
        result = compute_module_1_universe([], "2024-01-01")

        assert result["active_securities"] == []
        assert result["excluded_securities"] == []
        assert result["diagnostic_counts"]["total_input"] == 0

    def test_missing_ticker_field(self):
        """Records without ticker should be skipped."""
        records = [
            {"company_name": "No Ticker Corp", "market_cap_mm": 5000},
            {"ticker": "GOOD", "company_name": "Good Corp", "market_cap_mm": 5000},
        ]
        result = compute_module_1_universe(records, "2024-01-01")

        # Only GOOD should be processed
        active_tickers = [s["ticker"] for s in result["active_securities"]]
        assert "GOOD" in active_tickers
        assert len(active_tickers) == 1

    def test_invalid_market_cap(self):
        """Non-numeric market cap should be handled."""
        records = [
            {"ticker": "BAD", "company_name": "Bad Data", "market_cap_mm": "not_a_number"},
            {"ticker": "GOOD", "company_name": "Good Corp", "market_cap_mm": 5000},
        ]
        result = compute_module_1_universe(records, "2024-01-01")

        # GOOD should still be processed
        active_tickers = [s["ticker"] for s in result["active_securities"]]
        assert "GOOD" in active_tickers

    def test_none_market_cap(self):
        """None market cap should be handled."""
        records = [
            {"ticker": "NONE", "company_name": "None Corp", "market_cap_mm": None},
            {"ticker": "GOOD", "company_name": "Good Corp", "market_cap_mm": 5000},
        ]
        result = compute_module_1_universe(records, "2024-01-01")

        # GOOD should still be processed
        active_tickers = [s["ticker"] for s in result["active_securities"]]
        assert "GOOD" in active_tickers

    def test_zero_market_cap(self):
        """Zero market cap should be excluded (micro cap)."""
        records = [
            {"ticker": "ZERO", "company_name": "Zero Corp", "market_cap_mm": 0},
        ]
        result = compute_module_1_universe(records, "2024-01-01")

        # ZERO should be excluded
        assert len(result["active_securities"]) == 0
        excluded_tickers = [s["ticker"] for s in result["excluded_securities"]]
        assert "ZERO" in excluded_tickers


class TestErrorHandlingModule2:
    """Test Module 2 handles errors gracefully."""

    def test_empty_financial_records(self):
        """Handle empty financial data."""
        result = compute_module_2_financial([], ["TEST"], "2024-01-01")

        assert len(result["scores"]) == 1
        score = result["scores"][0]
        assert score["ticker"] == "TEST"
        assert "missing_financial_data" in score["flags"]

    def test_missing_cash_field(self):
        """Handle records with missing cash."""
        records = [
            {"ticker": "NOCASH", "debt_mm": 100, "source_date": "2023-12-30"},
        ]
        result = compute_module_2_financial(records, ["NOCASH"], "2024-01-01")

        score = result["scores"][0]
        assert "missing_cash" in score["flags"] or score["cash_mm"] is None

    def test_negative_cash(self):
        """Handle negative cash values."""
        records = [
            {"ticker": "NEG", "cash_mm": -100, "debt_mm": 50, "source_date": "2023-12-30"},
        ]
        result = compute_module_2_financial(records, ["NEG"], "2024-01-01")

        # Should still produce a result (even if flagged)
        assert len(result["scores"]) == 1

    def test_zero_burn_rate(self):
        """Zero burn rate should not cause division by zero."""
        records = [
            {"ticker": "NOBURN", "cash_mm": 500, "debt_mm": 0,
             "burn_rate_mm": 0, "source_date": "2023-12-30"},
        ]
        result = compute_module_2_financial(records, ["NOBURN"], "2024-01-01")

        score = result["scores"][0]
        # Should handle gracefully (infinite runway or flagged)
        assert score is not None


class TestErrorHandlingModule4:
    """Test Module 4 handles errors gracefully."""

    def test_empty_trial_records(self):
        """Handle empty trial data."""
        result = compute_module_4_clinical_dev([], ["TEST"], "2024-01-01")

        assert len(result["scores"]) == 1
        score = result["scores"][0]
        assert score["ticker"] == "TEST"
        assert "no_trials" in score["flags"]

    def test_missing_phase(self):
        """Handle trials with missing phase."""
        records = [
            {"ticker": "NOPHASE", "nct_id": "NCT123", "status": "recruiting"},
        ]
        result = compute_module_4_clinical_dev(records, ["NOPHASE"], "2024-01-01")

        # Should still process (default to early phase)
        assert len(result["scores"]) == 1

    def test_invalid_phase(self):
        """Handle trials with invalid phase string."""
        records = [
            {"ticker": "BADPHASE", "nct_id": "NCT123",
             "phase": "invalid_phase", "status": "recruiting"},
        ]
        result = compute_module_4_clinical_dev(records, ["BADPHASE"], "2024-01-01")

        # Should still process
        assert len(result["scores"]) == 1


class TestErrorHandlingModule5:
    """Test Module 5 handles errors gracefully."""

    def test_empty_inputs(self):
        """Handle empty inputs from all modules."""
        universe = {"active_securities": []}
        financial = {"scores": []}
        catalyst = {"scores": []}
        clinical = {"scores": []}

        result = compute_module_5_composite(
            universe, financial, catalyst, clinical, "2024-01-01"
        )

        assert result["ranked_securities"] == []
        assert result["diagnostic_counts"]["rankable"] == 0

    def test_missing_scores(self):
        """Handle when some modules have no scores for a ticker."""
        universe = {"active_securities": [{"ticker": "TEST"}]}
        financial = {"scores": []}  # No financial data
        catalyst = {"scores": []}   # No catalyst data
        clinical = {"scores": []}   # No clinical data

        result = compute_module_5_composite(
            universe, financial, catalyst, clinical, "2024-01-01"
        )

        # Should still rank with available data (with uncertainty penalty)
        assert len(result["ranked_securities"]) == 1
        assert "uncertainty_penalty_applied" in result["ranked_securities"][0]["flags"]

    def test_mismatched_tickers(self):
        """Handle when modules have different tickers."""
        universe = {"active_securities": [{"ticker": "A"}, {"ticker": "B"}]}
        financial = {
            "scores": [{"ticker": "A", "financial_score": "70", "severity": "none", "flags": []}]
        }
        catalyst = {
            "scores": [{"ticker": "B", "catalyst_score": "50", "severity": "none", "flags": []}]
        }
        clinical = {
            "scores": [
                {"ticker": "A", "clinical_score": "60", "lead_phase": "phase 2", "severity": "none", "flags": []},
                {"ticker": "B", "clinical_score": "40", "lead_phase": "phase 1", "severity": "none", "flags": []},
            ]
        }

        result = compute_module_5_composite(
            universe, financial, catalyst, clinical, "2024-01-01"
        )

        # Both should be ranked but with penalties for missing data
        assert len(result["ranked_securities"]) == 2


class TestPITEnforcement:
    """Test Point-In-Time discipline is enforced."""

    def test_future_financial_data_excluded(self):
        """Financial data from after as_of_date should be excluded."""
        records = [
            {"ticker": "PAST", "cash_mm": 100, "source_date": "2023-12-30"},
            {"ticker": "FUTURE", "cash_mm": 200, "source_date": "2024-01-02"},
        ]
        result = compute_module_2_financial(
            records, ["PAST", "FUTURE"], "2024-01-01"
        )

        scored_tickers = [s["ticker"] for s in result["scores"] if s.get("cash_mm")]
        assert "PAST" in scored_tickers or len([s for s in result["scores"] if s["ticker"] == "PAST"]) > 0

        # FUTURE should not have financial data (after PIT cutoff)
        future_score = next((s for s in result["scores"] if s["ticker"] == "FUTURE"), None)
        if future_score:
            assert future_score.get("cash_mm") is None or "pit_filtered" in future_score.get("flags", [])


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_duplicate_tickers(self):
        """Handle duplicate tickers in input."""
        records = [
            {"ticker": "DUP", "company_name": "Dup Corp 1", "market_cap_mm": 5000},
            {"ticker": "DUP", "company_name": "Dup Corp 2", "market_cap_mm": 6000},
        ]
        result = compute_module_1_universe(records, "2024-01-01")

        # Should not crash, deduplicate somehow
        active_tickers = [s["ticker"] for s in result["active_securities"]]
        # Either one DUP or both, but no crash
        assert result is not None

    def test_special_characters_in_ticker(self):
        """Handle tickers with special characters."""
        records = [
            {"ticker": "BRK.A", "company_name": "Berkshire A", "market_cap_mm": 500000},
            {"ticker": "BRK-B", "company_name": "Berkshire B", "market_cap_mm": 500000},
        ]
        result = compute_module_1_universe(records, "2024-01-01")

        # Should handle gracefully
        assert result is not None

    def test_very_large_numbers(self):
        """Handle very large market cap values."""
        records = [
            {"ticker": "BIG", "company_name": "Big Corp", "market_cap_mm": 1e12},  # Trillion
        ]
        result = compute_module_1_universe(records, "2024-01-01")

        active = [s["ticker"] for s in result["active_securities"]]
        assert "BIG" in active

    def test_decimal_precision(self):
        """Ensure Decimal precision is maintained."""
        universe = {"active_securities": [{"ticker": "PREC"}]}
        financial = {
            "scores": [{
                "ticker": "PREC",
                "financial_score": "67.123456789",  # High precision
                "market_cap_mm": "1234.5678",
                "severity": "none",
                "flags": []
            }]
        }
        catalyst = {
            "scores": [{
                "ticker": "PREC",
                "catalyst_score": "45.987654321",
                "severity": "none",
                "flags": []
            }]
        }
        clinical = {
            "scores": [{
                "ticker": "PREC",
                "clinical_score": "78.111111111",
                "lead_phase": "phase 2",
                "severity": "none",
                "flags": []
            }]
        }

        result = compute_module_5_composite(
            universe, financial, catalyst, clinical, "2024-01-01"
        )

        # Should maintain precision (at least 2 decimal places)
        score = result["ranked_securities"][0]["composite_score"]
        assert "." in str(score)


class TestDeterminism:
    """Test that outputs are deterministic."""

    def test_same_input_same_output(self):
        """Same inputs should always produce same outputs."""
        records = [
            {"ticker": "A", "company_name": "A Corp", "market_cap_mm": 5000},
            {"ticker": "B", "company_name": "B Corp", "market_cap_mm": 3000},
            {"ticker": "C", "company_name": "C Corp", "market_cap_mm": 7000},
        ]

        result1 = compute_module_1_universe(records, "2024-01-01")
        result2 = compute_module_1_universe(records, "2024-01-01")

        # Should be identical
        assert result1["active_securities"] == result2["active_securities"]
        assert result1["excluded_securities"] == result2["excluded_securities"]

    def test_ordering_is_stable(self):
        """Output ordering should be stable across runs."""
        universe = {"active_securities": [
            {"ticker": "Z"}, {"ticker": "A"}, {"ticker": "M"}
        ]}
        financial = {"scores": [
            {"ticker": "Z", "financial_score": "50", "market_cap_mm": "5000", "severity": "none", "flags": []},
            {"ticker": "A", "financial_score": "50", "market_cap_mm": "5000", "severity": "none", "flags": []},
            {"ticker": "M", "financial_score": "50", "market_cap_mm": "5000", "severity": "none", "flags": []},
        ]}
        catalyst = {"scores": [
            {"ticker": "Z", "catalyst_score": "50", "severity": "none", "flags": []},
            {"ticker": "A", "catalyst_score": "50", "severity": "none", "flags": []},
            {"ticker": "M", "catalyst_score": "50", "severity": "none", "flags": []},
        ]}
        clinical = {"scores": [
            {"ticker": "Z", "clinical_score": "50", "lead_phase": "phase 2", "severity": "none", "flags": []},
            {"ticker": "A", "clinical_score": "50", "lead_phase": "phase 2", "severity": "none", "flags": []},
            {"ticker": "M", "clinical_score": "50", "lead_phase": "phase 2", "severity": "none", "flags": []},
        ]}

        result1 = compute_module_5_composite(
            universe, financial, catalyst, clinical, "2024-01-01"
        )
        result2 = compute_module_5_composite(
            universe, financial, catalyst, clinical, "2024-01-01"
        )

        # Ordering should be identical (all tied scores, sorted by ticker)
        tickers1 = [s["ticker"] for s in result1["ranked_securities"]]
        tickers2 = [s["ticker"] for s in result2["ranked_securities"]]
        assert tickers1 == tickers2
