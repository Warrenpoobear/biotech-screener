"""Module 2 (financial health): solvency by operating cash flow, one cash total.

Two defects found in the 2026-11 financial-score review:

1. Positive net income marked a company "profitable" (monthly burn 0, runway
   999 months) even when operating cash flow was negative. A milestone or
   collaboration payment booked to income does not stop the cash burn.
2. Liquid assets were Cash plus the FIRST investment tag found, ignoring the
   deduped ``CashAndSecurities`` total the collectors already compute. The first
   tag can be years stale or the non-current portfolio.

Fixtures are synthetic; no live data is read.
"""

from decimal import Decimal

from module_2_financial_v2 import BurnSource, calculate_burn_rate_v2, calculate_liquid_assets, score_financial_health_v2

# --- solvency ---------------------------------------------------------------


def test_positive_net_income_does_not_erase_negative_cfo():
    result = calculate_burn_rate_v2({"CFO": -240_000_000, "NetIncome": 300_000_000, "Cash": 310_000_000})
    assert result.burn_source == BurnSource.CFO_ANNUAL
    assert result.monthly_burn == Decimal("20000000")
    assert result.raw_value == Decimal("-240000000")


def test_negative_cfo_with_positive_net_income_gets_a_finite_runway():
    fin = {"CFO": -240_000_000, "NetIncome": 300_000_000, "Cash": 310_000_000}
    mkt = {"market_cap": 2_000_000_000, "price": 20, "avg_volume": 1_000_000}
    scored = score_financial_health_v2("TEST", fin, mkt)
    assert float(scored["runway_months"]) == 15.5
    assert scored["burn_source"] == BurnSource.CFO_ANNUAL.value


def test_positive_cfo_is_still_profitable():
    result = calculate_burn_rate_v2({"CFO": 100_000_000, "NetIncome": -10_000_000})
    assert result.burn_source == BurnSource.PROFITABLE
    assert result.monthly_burn == Decimal("0")
    assert result.raw_value == Decimal("100000000")


def test_positive_net_income_counts_only_when_no_cash_flow_is_reported():
    result = calculate_burn_rate_v2({"NetIncome": 50_000_000})
    assert result.burn_source == BurnSource.PROFITABLE
    assert result.raw_value == Decimal("50000000")


def test_zero_cfo_does_not_fall_through_to_net_income():
    """CFO of exactly 0 is a reported figure, not a missing one."""
    result = calculate_burn_rate_v2({"CFO": 0, "NetIncome": 50_000_000})
    assert result.burn_source != BurnSource.PROFITABLE


# --- liquid assets ------------------------------------------------------------


def test_cash_and_securities_total_is_preferred():
    result = calculate_liquid_assets(
        {
            "Cash": 55_000_000,
            "CashAndSecurities": 110_000_000,
            # Larger, unrelated tag that must not be added on top of cash.
            "MarketableSecurities": 400_000_000,
        }
    )
    assert result.liquid_assets == Decimal("110000000")
    assert result.cash == Decimal("55000000")
    assert result.marketable_securities == Decimal("55000000")
    assert result.components_used == ["CashAndSecurities"]


def test_cash_and_securities_below_cash_does_not_go_negative():
    result = calculate_liquid_assets({"Cash": 80_000_000, "CashAndSecurities": 60_000_000})
    assert result.liquid_assets == Decimal("60000000")
    assert result.marketable_securities == Decimal("0")


def test_largest_investment_tag_is_used_without_a_total():
    result = calculate_liquid_assets(
        {
            "Cash": 10_000_000,
            "ShortTermInvestments": 5_000_000,
            "AvailableForSaleDebtCurrent": 70_000_000,
        }
    )
    assert result.liquid_assets == Decimal("80000000")
    assert result.components_used == ["Cash", "AvailableForSaleDebtCurrent"]


def test_cash_only_record_is_unchanged():
    result = calculate_liquid_assets({"Cash": 25_000_000})
    assert result.liquid_assets == Decimal("25000000")
    assert result.components_used == ["Cash"]


def test_zero_or_missing_total_falls_back_to_tags():
    for total in (0, None):
        result = calculate_liquid_assets(
            {"Cash": 10_000_000, "CashAndSecurities": total, "MarketableSecurities": 30_000_000}
        )
        assert result.liquid_assets == Decimal("40000000")
        assert result.components_used == ["Cash", "MarketableSecurities"]
