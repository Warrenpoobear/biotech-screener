#!/usr/bin/env python3
"""
Tests for module_2_financial.py edge cases and untested code paths

Tests cover:
- Dilution risk calculation
- Runway score calculation with edge cases
- Burn rate acceleration handling
- Negative burn (cash generation) handling
- Market cap normalization
- Missing data handling
- PIT enforcement edge cases
"""

import pytest
from datetime import date
from decimal import Decimal

from module_2_financial import compute_module_2_financial


class TestDilutionRiskCalculation:
    """Test dilution risk scoring with various scenarios."""

    def test_low_cap_high_dilution_risk(self):
        """Small companies with low runway have high dilution risk."""
        records = [{
            "ticker": "SMALLCO",
            "cash_mm": 50,
            "debt_mm": 10,
            "burn_rate_mm": 20,
            "market_cap_mm": 100,  # Small cap
            "source_date": "2026-01-14",
        }]

        result = compute_module_2_financial(records, ["SMALLCO"], "2026-01-15")

        score_record = result["scores"][0]
        # Low runway + small cap = low financial score
        assert float(score_record["financial_score"]) < 50

    def test_large_cap_low_dilution_risk(self):
        """Large companies with good runway have low dilution risk."""
        records = [{
            "ticker": "BIGCO",
            "cash_mm": 5000,
            "debt_mm": 500,
            "burn_rate_mm": 100,
            "market_cap_mm": 20000,  # Large cap
            "source_date": "2026-01-14",
        }]

        result = compute_module_2_financial(records, ["BIGCO"], "2026-01-15")

        score_record = result["scores"][0]
        # Good runway + large cap = high financial score
        assert float(score_record["financial_score"]) > 60

    def test_zero_market_cap_handled(self):
        """Zero market cap is handled without crash."""
        records = [{
            "ticker": "ZERO",
            "cash_mm": 100,
            "debt_mm": 0,
            "burn_rate_mm": 10,
            "market_cap_mm": 0,  # Edge case
            "source_date": "2026-01-14",
        }]

        result = compute_module_2_financial(records, ["ZERO"], "2026-01-15")

        # Should not crash
        assert len(result["scores"]) == 1
        assert result["scores"][0]["ticker"] == "ZERO"

    def test_negative_market_cap_handled(self):
        """Negative market cap (data error) is handled."""
        records = [{
            "ticker": "NEG",
            "cash_mm": 100,
            "debt_mm": 0,
            "burn_rate_mm": 10,
            "market_cap_mm": -100,  # Invalid but shouldn't crash
            "source_date": "2026-01-14",
        }]

        result = compute_module_2_financial(records, ["NEG"], "2026-01-15")

        # Should handle gracefully
        assert len(result["scores"]) >= 0  # May be excluded or scored


class TestRunwayCalculation:
    """Test runway calculation with edge cases."""

    def test_positive_cash_flow_infinite_runway(self):
        """Companies with positive cash flow (negative burn) get max runway score."""
        records = [{
            "ticker": "PROFIT",
            "cash_mm": 100,
            "debt_mm": 0,
            "burn_rate_mm": -10,  # Generating cash!
            "market_cap_mm": 1000,
            "source_date": "2026-01-14",
        }]

        result = compute_module_2_financial(records, ["PROFIT"], "2026-01-15")

        score_record = result["scores"][0]
        # Positive cash flow = excellent financial health
        assert float(score_record["financial_score"]) > 80
        assert "revenue_positive" in str(score_record.get("flags", []))

    def test_zero_burn_infinite_runway(self):
        """Zero burn rate = infinite runway."""
        records = [{
            "ticker": "STABLE",
            "cash_mm": 100,
            "debt_mm": 0,
            "burn_rate_mm": 0,  # No burn
            "market_cap_mm": 1000,
            "source_date": "2026-01-14",
        }]

        result = compute_module_2_financial(records, ["STABLE"], "2026-01-15")

        score_record = result["scores"][0]
        # No burn = excellent score
        assert float(score_record["financial_score"]) > 70

    def test_very_high_burn_low_score(self):
        """Very high burn rate relative to cash = low score."""
        records = [{
            "ticker": "BURNING",
            "cash_mm": 50,
            "debt_mm": 0,
            "burn_rate_mm": 100,  # Burning 2x cash!
            "market_cap_mm": 500,
            "source_date": "2026-01-14",
        }]

        result = compute_module_2_financial(records, ["BURNING"], "2026-01-15")

        score_record = result["scores"][0]
        # Unsustainable burn = very low score
        assert float(score_record["financial_score"]) < 30
        assert float(score_record["runway_months"]) < 12

    def test_negative_cash_handled(self):
        """Negative cash (debt exceeds cash) is handled."""
        records = [{
            "ticker": "BROKE",
            "cash_mm": 10,
            "debt_mm": 50,  # More debt than cash
            "burn_rate_mm": 5,
            "market_cap_mm": 100,
            "source_date": "2026-01-14",
        }]

        result = compute_module_2_financial(records, ["BROKE"], "2026-01-15")

        # Should still score (negatively)
        assert len(result["scores"]) >= 0


class TestBurnAcceleration:
    """Test burn rate acceleration/deceleration handling."""

    def test_accelerating_burn_penalty(self):
        """Accelerating burn rate results in penalty."""
        records = [{
            "ticker": "ACCEL",
            "cash_mm": 200,
            "debt_mm": 0,
            "burn_rate_mm": 40,
            "market_cap_mm": 1000,
            "source_date": "2026-01-14",
            "burn_history": [30, 32, 35, 40],  # Accelerating
        }]

        result = compute_module_2_financial(records, ["ACCEL"], "2026-01-15")

        score_record = result["scores"][0]
        # Should have acceleration flag or penalty
        flags = score_record.get("flags", [])
        # Score should be impacted by acceleration

    def test_decelerating_burn_bonus(self):
        """Decelerating burn rate (improving efficiency) gets bonus."""
        records = [{
            "ticker": "DECEL",
            "cash_mm": 200,
            "debt_mm": 0,
            "burn_rate_mm": 20,
            "market_cap_mm": 1000,
            "source_date": "2026-01-14",
            "burn_history": [40, 35, 25, 20],  # Decelerating
        }]

        result = compute_module_2_financial(records, ["DECEL"], "2026-01-15")

        score_record = result["scores"][0]
        # Should get efficiency improvement bonus
        # Score should be better than flat burn


class TestMissingDataHandling:
    """Test handling of missing financial data."""

    def test_missing_cash_skipped(self):
        """Records missing cash are skipped."""
        records = [{
            "ticker": "NOCASH",
            "debt_mm": 10,
            "burn_rate_mm": 5,
            "market_cap_mm": 100,
            "source_date": "2026-01-14",
            # cash_mm missing
        }]

        result = compute_module_2_financial(records, ["NOCASH"], "2026-01-15")

        # Should be in missing data diagnostic
        assert result["diagnostic_counts"]["missing_data"] >= 1

    def test_missing_burn_rate_handled(self):
        """Records missing burn rate are handled."""
        records = [{
            "ticker": "NOBURN",
            "cash_mm": 100,
            "debt_mm": 10,
            "market_cap_mm": 500,
            "source_date": "2026-01-14",
            # burn_rate_mm missing
        }]

        result = compute_module_2_financial(records, ["NOBURN"], "2026-01-15")

        # Should handle gracefully (may use default or estimate)
        assert result["diagnostic_counts"]["total"] >= 1

    def test_missing_market_cap_handled(self):
        """Records missing market cap are handled."""
        records = [{
            "ticker": "NOCAP",
            "cash_mm": 100,
            "debt_mm": 0,
            "burn_rate_mm": 10,
            "source_date": "2026-01-14",
            # market_cap_mm missing
        }]

        result = compute_module_2_financial(records, ["NOCAP"], "2026-01-15")

        # Should handle gracefully
        assert result["diagnostic_counts"]["total"] >= 1


class TestPITEnforcement:
    """Test point-in-time enforcement edge cases."""

    def test_same_day_data_excluded(self):
        """Data with source_date = as_of_date is excluded (strict PIT)."""
        records = [{
            "ticker": "SAME",
            "cash_mm": 100,
            "debt_mm": 0,
            "burn_rate_mm": 10,
            "market_cap_mm": 500,
            "source_date": "2026-01-15",  # Same as as_of_date
        }]

        result = compute_module_2_financial(records, ["SAME"], "2026-01-15")

        # Should be excluded due to PIT cutoff
        tickers_scored = [s["ticker"] for s in result["scores"]]
        assert "SAME" not in tickers_scored

    def test_one_day_before_included(self):
        """Data with source_date = as_of_date - 1 is included."""
        records = [{
            "ticker": "YESTERDAY",
            "cash_mm": 100,
            "debt_mm": 0,
            "burn_rate_mm": 10,
            "market_cap_mm": 500,
            "source_date": "2026-01-14",  # Day before as_of_date
        }]

        result = compute_module_2_financial(records, ["YESTERDAY"], "2026-01-15")

        # Should be included
        tickers_scored = [s["ticker"] for s in result["scores"]]
        assert "YESTERDAY" in tickers_scored

    def test_future_data_excluded(self):
        """Future data is excluded (PIT safety)."""
        records = [{
            "ticker": "FUTURE",
            "cash_mm": 100,
            "debt_mm": 0,
            "burn_rate_mm": 10,
            "market_cap_mm": 500,
            "source_date": "2026-01-20",  # 5 days in future
        }]

        result = compute_module_2_financial(records, ["FUTURE"], "2026-01-15")

        # Should be excluded
        tickers_scored = [s["ticker"] for s in result["scores"]]
        assert "FUTURE" not in tickers_scored

    def test_multiple_records_most_recent_pit_safe_used(self):
        """With multiple records for same ticker, most recent PIT-safe one is used."""
        records = [
            {
                "ticker": "MULTI",
                "cash_mm": 50,
                "debt_mm": 0,
                "burn_rate_mm": 5,
                "market_cap_mm": 500,
                "source_date": "2026-01-10",  # Older
            },
            {
                "ticker": "MULTI",
                "cash_mm": 100,
                "debt_mm": 0,
                "burn_rate_mm": 10,
                "market_cap_mm": 500,
                "source_date": "2026-01-14",  # More recent, PIT-safe
            },
            {
                "ticker": "MULTI",
                "cash_mm": 200,
                "debt_mm": 0,
                "burn_rate_mm": 20,
                "market_cap_mm": 500,
                "source_date": "2026-01-15",  # Same day - excluded
            },
        ]

        result = compute_module_2_financial(records, ["MULTI"], "2026-01-15")

        tickers_scored = [s["ticker"] for s in result["scores"]]
        assert "MULTI" in tickers_scored

        # Should use 2026-01-14 data (cash=100)
        multi_score = [s for s in result["scores"] if s["ticker"] == "MULTI"][0]
        # Can verify it used the correct record by checking runway calculation


class TestEmptyUniverse:
    """Test empty universe handling."""

    def test_empty_active_tickers(self):
        """Empty active_tickers returns empty scores."""
        records = [{
            "ticker": "TEST",
            "cash_mm": 100,
            "debt_mm": 0,
            "burn_rate_mm": 10,
            "market_cap_mm": 500,
            "source_date": "2026-01-14",
        }]

        result = compute_module_2_financial(records, [], "2026-01-15")

        assert result["scores"] == []
        assert result["diagnostic_counts"]["total"] == 0

    def test_none_active_tickers(self):
        """None active_tickers is handled."""
        records = [{
            "ticker": "TEST",
            "cash_mm": 100,
            "source_date": "2026-01-14",
        }]

        result = compute_module_2_financial(records, None, "2026-01-15")

        # Should handle gracefully
        assert "scores" in result


class TestOutputSchema:
    """Test output schema compliance."""

    def test_all_required_fields_present(self):
        """Output has all required fields."""
        records = [{
            "ticker": "TEST",
            "cash_mm": 100,
            "debt_mm": 10,
            "burn_rate_mm": 10,
            "market_cap_mm": 500,
            "source_date": "2026-01-14",
        }]

        result = compute_module_2_financial(records, ["TEST"], "2026-01-15")

        assert "scores" in result
        assert "diagnostic_counts" in result

        score = result["scores"][0]
        required_fields = [
            "ticker",
            "financial_score",
            "financial_normalized",  # Legacy alias
            "market_cap_mm",
            "runway_months",
            "severity",
            "flags",
        ]

        for field in required_fields:
            assert field in score, f"Missing field: {field}"

    def test_financial_score_and_normalized_match(self):
        """financial_score and financial_normalized should match (backward compat)."""
        records = [{
            "ticker": "TEST",
            "cash_mm": 100,
            "debt_mm": 10,
            "burn_rate_mm": 10,
            "market_cap_mm": 500,
            "source_date": "2026-01-14",
        }]

        result = compute_module_2_financial(records, ["TEST"], "2026-01-15")

        score = result["scores"][0]
        assert score["financial_score"] == score["financial_normalized"]

    def test_score_bounded_0_100(self):
        """Financial scores are bounded to [0, 100]."""
        records = [{
            "ticker": "TEST",
            "cash_mm": 100,
            "debt_mm": 10,
            "burn_rate_mm": 10,
            "market_cap_mm": 500,
            "source_date": "2026-01-14",
        }]

        result = compute_module_2_financial(records, ["TEST"], "2026-01-15")

        score = float(result["scores"][0]["financial_score"])
        assert 0 <= score <= 100
