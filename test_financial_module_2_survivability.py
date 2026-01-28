#!/usr/bin/env python3
"""
test_financial_module_2_survivability.py - Unit tests for Financial Module 2

Tests all threshold boundaries, edge cases, and scoring logic.
"""

import unittest
from decimal import Decimal
from datetime import date, timedelta

from financial_module_2_survivability import (
    compute_survivability_score,
    compute_effective_runway,
    score_runway,
    score_burn_discipline,
    score_burn_acceleration,
    score_debt_fragility,
    clamp,
    to_decimal,
    FINAL_SCORE_MIN,
    FINAL_SCORE_MAX,
    SCORE_A_MIN,
    SCORE_A_MAX,
    SCORE_B_MIN,
    SCORE_B_MAX,
    SCORE_C_MIN,
    SCORE_C_MAX,
    SCORE_D_MIN,
    SCORE_D_MAX,
)


class TestHelpers(unittest.TestCase):
    """Test helper functions."""

    def test_to_decimal_valid(self):
        self.assertEqual(to_decimal(100), Decimal("100"))
        self.assertEqual(to_decimal("100.5"), Decimal("100.5"))
        self.assertEqual(to_decimal(100.5), Decimal("100.5"))
        self.assertEqual(to_decimal(Decimal("100")), Decimal("100"))

    def test_to_decimal_invalid(self):
        self.assertEqual(to_decimal(None), Decimal("0"))
        self.assertEqual(to_decimal("invalid"), Decimal("0"))
        self.assertEqual(to_decimal(None, Decimal("5")), Decimal("5"))

    def test_clamp(self):
        self.assertEqual(clamp(Decimal("5"), Decimal("0"), Decimal("10")), Decimal("5"))
        self.assertEqual(clamp(Decimal("-5"), Decimal("0"), Decimal("10")), Decimal("0"))
        self.assertEqual(clamp(Decimal("15"), Decimal("0"), Decimal("10")), Decimal("10"))


class TestRunwayScore(unittest.TestCase):
    """Test Score A: Effective Runway boundaries."""

    def test_runway_24_months_boundary(self):
        """Runway >= 24 months should score +2.0"""
        score, note = score_runway(Decimal("24"), Decimal("10"), Decimal("100"))
        self.assertEqual(score, Decimal("2.0"))
        self.assertIn(">=24mo", note)

        score, note = score_runway(Decimal("30"), Decimal("10"), Decimal("100"))
        self.assertEqual(score, Decimal("2.0"))

    def test_runway_18_24_boundary(self):
        """Runway [18, 24) months should score +1.0"""
        score, note = score_runway(Decimal("18"), Decimal("10"), Decimal("100"))
        self.assertEqual(score, Decimal("1.0"))
        self.assertIn("18-24mo", note)

        score, note = score_runway(Decimal("23.9"), Decimal("10"), Decimal("100"))
        self.assertEqual(score, Decimal("1.0"))

    def test_runway_12_18_boundary(self):
        """Runway [12, 18) months should score 0.0"""
        score, note = score_runway(Decimal("12"), Decimal("10"), Decimal("100"))
        self.assertEqual(score, Decimal("0.0"))
        self.assertIn("12-18mo", note)

        score, note = score_runway(Decimal("17.9"), Decimal("10"), Decimal("100"))
        self.assertEqual(score, Decimal("0.0"))

    def test_runway_9_12_boundary(self):
        """Runway [9, 12) months should score -1.0"""
        score, note = score_runway(Decimal("9"), Decimal("10"), Decimal("100"))
        self.assertEqual(score, Decimal("-1.0"))
        self.assertIn("9-12mo", note)

        score, note = score_runway(Decimal("11.9"), Decimal("10"), Decimal("100"))
        self.assertEqual(score, Decimal("-1.0"))

    def test_runway_6_9_boundary(self):
        """Runway [6, 9) months should score -3.0"""
        score, note = score_runway(Decimal("6"), Decimal("10"), Decimal("100"))
        self.assertEqual(score, Decimal("-3.0"))
        self.assertIn("6-9mo", note)

        score, note = score_runway(Decimal("8.9"), Decimal("10"), Decimal("100"))
        self.assertEqual(score, Decimal("-3.0"))

    def test_runway_below_6_boundary(self):
        """Runway < 6 months should score -6.0"""
        score, note = score_runway(Decimal("5.9"), Decimal("10"), Decimal("100"))
        self.assertEqual(score, Decimal("-6.0"))
        self.assertIn("<6mo", note)

        score, note = score_runway(Decimal("0"), Decimal("10"), Decimal("100"))
        self.assertEqual(score, Decimal("-6.0"))

    def test_non_burner_with_cash(self):
        """Non-burner with cash should score +2.0"""
        score, note = score_runway(Decimal("999"), Decimal("0"), Decimal("100"))
        self.assertEqual(score, Decimal("2.0"))
        self.assertIn("non_burner", note)


class TestBurnDisciplineScore(unittest.TestCase):
    """Test Score B: Burn Efficiency / Discipline boundaries."""

    def test_rd_ratio_055_boundary(self):
        """R&D ratio >= 0.55 should score +2.0"""
        score, note, missing = score_burn_discipline(
            Decimal("55"), Decimal("100"), Decimal("50")
        )
        self.assertEqual(score, Decimal("2.0"))
        self.assertFalse(missing)

        score, note, missing = score_burn_discipline(
            Decimal("70"), Decimal("100"), Decimal("50")
        )
        self.assertEqual(score, Decimal("2.0"))

    def test_rd_ratio_045_boundary(self):
        """R&D ratio [0.45, 0.55) should score +1.0"""
        score, note, missing = score_burn_discipline(
            Decimal("45"), Decimal("100"), Decimal("50")
        )
        self.assertEqual(score, Decimal("1.0"))

        score, note, missing = score_burn_discipline(
            Decimal("54"), Decimal("100"), Decimal("50")
        )
        self.assertEqual(score, Decimal("1.0"))

    def test_rd_ratio_030_boundary(self):
        """R&D ratio [0.30, 0.45) should score 0.0"""
        score, note, missing = score_burn_discipline(
            Decimal("30"), Decimal("100"), Decimal("50")
        )
        self.assertEqual(score, Decimal("0.0"))

        score, note, missing = score_burn_discipline(
            Decimal("44"), Decimal("100"), Decimal("50")
        )
        self.assertEqual(score, Decimal("0.0"))

    def test_rd_ratio_020_boundary(self):
        """R&D ratio [0.20, 0.30) should score -1.0"""
        score, note, missing = score_burn_discipline(
            Decimal("20"), Decimal("100"), Decimal("50")
        )
        self.assertEqual(score, Decimal("-1.0"))

        score, note, missing = score_burn_discipline(
            Decimal("29"), Decimal("100"), Decimal("50")
        )
        self.assertEqual(score, Decimal("-1.0"))

    def test_rd_ratio_below_020_boundary(self):
        """R&D ratio < 0.20 should score -2.0"""
        score, note, missing = score_burn_discipline(
            Decimal("19"), Decimal("100"), Decimal("50")
        )
        self.assertEqual(score, Decimal("-2.0"))

        score, note, missing = score_burn_discipline(
            Decimal("10"), Decimal("100"), Decimal("50")
        )
        self.assertEqual(score, Decimal("-2.0"))

    def test_fallback_rd_to_burn(self):
        """Fallback to R&D/burn ratio when opex missing."""
        # R&D/burn >= 0.60: +1.0
        score, note, missing = score_burn_discipline(
            Decimal("60"), None, Decimal("100")
        )
        self.assertEqual(score, Decimal("1.0"))

        # R&D/burn [0.40, 0.60): 0.0
        score, note, missing = score_burn_discipline(
            Decimal("50"), None, Decimal("100")
        )
        self.assertEqual(score, Decimal("0.0"))

        # R&D/burn < 0.40: -1.0
        score, note, missing = score_burn_discipline(
            Decimal("30"), None, Decimal("100")
        )
        self.assertEqual(score, Decimal("-1.0"))

    def test_missing_data(self):
        """Missing data should score 0.0 and flag missing."""
        score, note, missing = score_burn_discipline(None, None, Decimal("100"))
        self.assertEqual(score, Decimal("0.0"))
        self.assertTrue(missing)


class TestBurnAccelerationScore(unittest.TestCase):
    """Test Score C: Burn Acceleration / Cash Concentration boundaries."""

    def test_burn_accel_050_boundary(self):
        """Burn acceleration >= 0.50 should score -2.0"""
        # 100 -> 150 = 50% increase
        score, note, metric = score_burn_acceleration(
            Decimal("150"), Decimal("100"), Decimal("100"), Decimal("500")
        )
        self.assertEqual(score, Decimal("-2.0"))

    def test_burn_accel_025_boundary(self):
        """Burn acceleration [0.25, 0.50) should score -1.0"""
        # 100 -> 130 = 30% increase
        score, note, metric = score_burn_acceleration(
            Decimal("130"), Decimal("100"), Decimal("100"), Decimal("500")
        )
        self.assertEqual(score, Decimal("-1.0"))

    def test_burn_accel_below_025(self):
        """Burn acceleration < 0.25 should score 0.0"""
        # 100 -> 120 = 20% increase
        score, note, metric = score_burn_acceleration(
            Decimal("120"), Decimal("100"), Decimal("100"), Decimal("500")
        )
        self.assertEqual(score, Decimal("0.0"))

    def test_cash_concentration_025_boundary(self):
        """Cash concentration >= 0.25 should score -2.0"""
        # burn_ttm=400, cash=400 -> quarterly=100, conc=0.25
        score, note, metric = score_burn_acceleration(
            None, None, Decimal("400"), Decimal("400")
        )
        self.assertEqual(score, Decimal("-2.0"))

    def test_cash_concentration_015_boundary(self):
        """Cash concentration [0.15, 0.25) should score -1.0"""
        # burn_ttm=200, cash=500 -> quarterly=50, conc=0.10 (below threshold)
        # burn_ttm=300, cash=500 -> quarterly=75, conc=0.15
        score, note, metric = score_burn_acceleration(
            None, None, Decimal("300"), Decimal("500")
        )
        self.assertEqual(score, Decimal("-1.0"))

    def test_cash_concentration_below_015(self):
        """Cash concentration < 0.15 should score 0.0"""
        # burn_ttm=200, cash=500 -> quarterly=50, conc=0.10
        score, note, metric = score_burn_acceleration(
            None, None, Decimal("200"), Decimal("500")
        )
        self.assertEqual(score, Decimal("0.0"))

    def test_score_never_positive(self):
        """Score C should never be positive."""
        # Burn decreasing
        score, note, metric = score_burn_acceleration(
            Decimal("50"), Decimal("100"), Decimal("100"), Decimal("500")
        )
        self.assertLessEqual(score, Decimal("0.0"))


class TestDebtFragilityScore(unittest.TestCase):
    """Test Score D: Debt Fragility & Maturity Mismatch boundaries."""

    def test_debt_to_cash_010_boundary(self):
        """Debt/cash < 0.10 should score +1.0"""
        score, note, ratio = score_debt_fragility(
            Decimal("9"), Decimal("100"), Decimal("10"),
            None, None, None, Decimal("24")
        )
        self.assertEqual(score, Decimal("1.0"))

    def test_debt_to_cash_040_boundary(self):
        """Debt/cash [0.10, 0.40) should score 0.0"""
        score, note, ratio = score_debt_fragility(
            Decimal("30"), Decimal("100"), Decimal("10"),
            None, None, None, Decimal("24")
        )
        self.assertEqual(score, Decimal("0.0"))

    def test_debt_to_cash_080_boundary(self):
        """Debt/cash [0.40, 0.80) should score -1.0"""
        score, note, ratio = score_debt_fragility(
            Decimal("60"), Decimal("100"), Decimal("10"),
            None, None, None, Decimal("24")
        )
        self.assertEqual(score, Decimal("-1.0"))

    def test_debt_to_cash_above_080_burner(self):
        """Debt/cash >= 0.80 for burner should score -2.0"""
        score, note, ratio = score_debt_fragility(
            Decimal("90"), Decimal("100"), Decimal("10"),  # is_burner=True
            None, None, None, Decimal("24")
        )
        self.assertEqual(score, Decimal("-2.0"))

    def test_debt_to_cash_above_080_non_burner(self):
        """Debt/cash >= 0.80 for non-burner should score -1.0"""
        score, note, ratio = score_debt_fragility(
            Decimal("90"), Decimal("100"), Decimal("0"),  # is_burner=False
            None, None, None, Decimal("24")
        )
        self.assertEqual(score, Decimal("-1.0"))

    def test_maturity_before_catalyst_penalty(self):
        """Maturity before catalyst should add -1.0 penalty."""
        today = date.today()
        maturity = today + timedelta(days=90)
        catalyst = today + timedelta(days=180)

        # Base score would be 0.0 (debt/cash = 0.30)
        score, note, ratio = score_debt_fragility(
            Decimal("30"), Decimal("100"), Decimal("10"),
            maturity, catalyst, None, Decimal("24")
        )
        self.assertEqual(score, Decimal("-1.0"))  # 0.0 + (-1.0) penalty
        self.assertIn("maturity_before_catalyst", note)

    def test_debt_due_short_runway_penalty(self):
        """Debt due within 12m with short runway should add -1.0 penalty."""
        # Base score would be 0.0 (debt/cash = 0.30)
        score, note, ratio = score_debt_fragility(
            Decimal("30"), Decimal("100"), Decimal("10"),
            None, None, Decimal("20"), Decimal("10")  # runway < 12
        )
        self.assertEqual(score, Decimal("-1.0"))  # 0.0 + (-1.0) penalty
        self.assertIn("debt_due_12m_short_runway", note)

    def test_score_clamped_to_bounds(self):
        """Score D should be clamped to [-3, +1]."""
        # Maximum penalties: debt/cash >= 0.80 burner (-2) + maturity penalty (-1) = -3
        today = date.today()
        maturity = today + timedelta(days=90)
        catalyst = today + timedelta(days=180)

        score, note, ratio = score_debt_fragility(
            Decimal("90"), Decimal("100"), Decimal("10"),
            maturity, catalyst, None, Decimal("24")
        )
        self.assertEqual(score, Decimal("-3.0"))
        self.assertGreaterEqual(score, SCORE_D_MIN)
        self.assertLessEqual(score, SCORE_D_MAX)


class TestFinalScoreBounds(unittest.TestCase):
    """Test final score clamping."""

    def test_score_clamped_to_minus_10(self):
        """Final score should not go below -10.0"""
        # Create worst-case scenario
        result = compute_survivability_score({
            'Cash': 10e6,
            'CFO': -200e6,  # Massive burn
            'R&D': 10e6,
            'total_operating_expense_ttm': 200e6,
            'LongTermDebt': 100e6,
            'current_debt': 50e6,
        })
        self.assertGreaterEqual(result['score'], -10.0)

    def test_score_clamped_to_plus_5(self):
        """Final score should not exceed +5.0"""
        # Create best-case scenario
        result = compute_survivability_score({
            'Cash': 1000e6,
            'ShortTermInvestments': 500e6,
            'CFO': 100e6,  # Profitable
            'R&D': 150e6,
            'total_operating_expense_ttm': 200e6,
        })
        self.assertLessEqual(result['score'], 5.0)

    def test_subscores_within_bounds(self):
        """All subscores should be within their bounds."""
        result = compute_survivability_score({
            'Cash': 100e6,
            'CFO': -50e6,
            'R&D': 30e6,
            'total_operating_expense_ttm': 60e6,
        })

        subscores = result['subscores']
        self.assertGreaterEqual(subscores['runway'], float(SCORE_A_MIN))
        self.assertLessEqual(subscores['runway'], float(SCORE_A_MAX))
        self.assertGreaterEqual(subscores['discipline'], float(SCORE_B_MIN))
        self.assertLessEqual(subscores['discipline'], float(SCORE_B_MAX))
        self.assertGreaterEqual(subscores['accel'], float(SCORE_C_MIN))
        self.assertLessEqual(subscores['accel'], float(SCORE_C_MAX))
        self.assertGreaterEqual(subscores['debt'], float(SCORE_D_MIN))
        self.assertLessEqual(subscores['debt'], float(SCORE_D_MAX))


class TestMissingData(unittest.TestCase):
    """Test graceful degradation with missing data."""

    def test_missing_all_data(self):
        """Should not crash with empty data."""
        result = compute_survivability_score({})
        self.assertIn('score', result)
        self.assertIn('subscores', result)
        self.assertIn('coverage', result)
        self.assertGreaterEqual(result['score'], -10.0)
        self.assertLessEqual(result['score'], 5.0)

    def test_missing_cash_flagged(self):
        """Missing cash should be flagged."""
        result = compute_survivability_score({
            'CFO': -50e6,
        })
        self.assertIn('missing_cash', result['coverage'])

    def test_missing_burn_flagged(self):
        """Missing burn data should be flagged."""
        result = compute_survivability_score({
            'Cash': 100e6,
        })
        self.assertIn('missing_burn_data', result['coverage'])

    def test_missing_rd_flagged(self):
        """Missing R&D should be flagged."""
        result = compute_survivability_score({
            'Cash': 100e6,
            'CFO': -50e6,
        })
        self.assertIn('missing_rd_expense', result['coverage'])

    def test_missing_quarterly_burn_flagged(self):
        """Missing quarterly burn should be flagged."""
        result = compute_survivability_score({
            'Cash': 100e6,
            'CFO': -50e6,
        })
        self.assertIn('missing_quarterly_burn', result['coverage'])


class TestEffectiveRunwayCalculation(unittest.TestCase):
    """Test runway calculation logic."""

    def test_basic_runway_calculation(self):
        """Test basic runway = 12 * (cash - debt) / (burn + interest)"""
        # cash=600, debt=100, burn=100, interest=20 -> 12 * 500 / 120 = 50 months
        runway, method = compute_effective_runway(
            Decimal("600"), Decimal("100"), Decimal("20"), Decimal("100")
        )
        self.assertEqual(runway, Decimal("50"))

    def test_non_burner_runway(self):
        """Non-burner should get 999 months runway."""
        runway, method = compute_effective_runway(
            Decimal("100"), Decimal("0"), Decimal("0"), Decimal("0")
        )
        self.assertEqual(runway, Decimal("999"))
        self.assertEqual(method, "non_burner")

    def test_no_cash_runway(self):
        """No cash should get 0 months runway."""
        runway, method = compute_effective_runway(
            Decimal("0"), Decimal("0"), Decimal("0"), Decimal("0")
        )
        self.assertEqual(runway, Decimal("0"))
        self.assertEqual(method, "no_cash")

    def test_debt_reduces_effective_cash(self):
        """Near-term debt should reduce effective cash."""
        # cash=100, debt=50 -> effective=50
        runway_with_debt, _ = compute_effective_runway(
            Decimal("100"), Decimal("12"), Decimal("0"), Decimal("50")
        )
        # cash=100, debt=0 -> effective=100
        runway_no_debt, _ = compute_effective_runway(
            Decimal("100"), Decimal("12"), Decimal("0"), Decimal("0")
        )
        self.assertLess(runway_with_debt, runway_no_debt)


class TestIntegration(unittest.TestCase):
    """Integration tests with realistic data."""

    def test_healthy_biotech(self):
        """Healthy biotech should score positively."""
        result = compute_survivability_score({
            'Cash': 500e6,
            'ShortTermInvestments': 200e6,
            'CFO': -80e6,
            'R&D': 60e6,
            'total_operating_expense_ttm': 100e6,
            'LongTermDebt': 30e6,
        })
        # Should have good runway, good discipline, no acceleration, low debt
        self.assertGreater(result['score'], 0)
        self.assertEqual(result['subscores']['runway'], 2.0)  # 105 months
        self.assertGreaterEqual(result['subscores']['discipline'], 1.0)  # R&D ratio 60%

    def test_distressed_biotech(self):
        """Distressed biotech should score negatively."""
        result = compute_survivability_score({
            'Cash': 20e6,
            'CFO': -100e6,
            'R&D': 15e6,
            'total_operating_expense_ttm': 120e6,
            'LongTermDebt': 80e6,
            'current_debt': 30e6,
        })
        # Should have poor runway, poor discipline, high debt
        self.assertLess(result['score'], 0)
        self.assertEqual(result['subscores']['runway'], -6.0)  # ~2.4 months

    def test_profitable_company(self):
        """Profitable company should score well."""
        result = compute_survivability_score({
            'Cash': 1000e6,
            'CFO': 200e6,  # Positive
            'R&D': 150e6,
            'total_operating_expense_ttm': 250e6,
        })
        self.assertEqual(result['subscores']['runway'], 2.0)  # Non-burner
        self.assertGreater(result['score'], 0)


class TestOutputFormat(unittest.TestCase):
    """Test output format and structure."""

    def test_output_has_required_fields(self):
        """Output should have all required fields."""
        result = compute_survivability_score({'Cash': 100e6})

        self.assertIn('module', result)
        self.assertIn('score', result)
        self.assertIn('subscores', result)
        self.assertIn('metrics', result)
        self.assertIn('coverage', result)
        self.assertIn('notes', result)

        self.assertEqual(result['module'], 'financial_module_2_survivability')

    def test_subscores_has_all_components(self):
        """Subscores should have all four components."""
        result = compute_survivability_score({'Cash': 100e6})

        subscores = result['subscores']
        self.assertIn('runway', subscores)
        self.assertIn('discipline', subscores)
        self.assertIn('accel', subscores)
        self.assertIn('debt', subscores)

    def test_metrics_are_floats(self):
        """Numeric metrics should be float values."""
        result = compute_survivability_score({
            'Cash': 100e6,
            'CFO': -50e6,
        })

        # String fields that are expected
        string_fields = {'runway_confidence'}

        for key, value in result['metrics'].items():
            if value is not None and key not in string_fields:
                self.assertIsInstance(value, (int, float), f"{key} should be numeric")


if __name__ == '__main__':
    unittest.main()
