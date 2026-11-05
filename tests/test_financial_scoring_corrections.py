"""Corrections for cash, burn, market-cap units, and clinical endpoint tokens.

These lock the defects found in the 2026-11 review: millions-vs-dollars
market cap, net-income solvency, deduped cash, period length, undated
financials, stale PIT stores, and substring endpoint matches.
"""

from decimal import Decimal

from module_2_financial_v2 import (
    BurnSource,
    calculate_burn_rate_v2,
    calculate_liquid_assets,
    compute_module_2_financial,
    market_cap_to_usd,
    score_financial_health_v2,
)
from module_4_clinical_dev import _classify_endpoint, compute_module_4_clinical_dev
from pit_financials import pit_store_covers_as_of


def test_market_cap_millions_scale_to_dollars():
    # CRSP-shaped row: 4849.1 million, price x shares about $4.9 billion.
    usd = market_cap_to_usd(4849.1, price=49.99, shares=98_484_128)
    assert usd > Decimal("4000000000")
    assert usd < Decimal("6000000000")


def test_market_cap_already_in_dollars_stays():
    usd = market_cap_to_usd(5_252_712_960, price=50, shares=105_054_259)
    assert abs(usd - Decimal("5252712960")) / Decimal("5252712960") < Decimal("0.01")


def test_market_cap_without_shares_uses_magnitude():
    assert market_cap_to_usd(4849.1) == Decimal("4849.1") * Decimal("1000000")
    assert market_cap_to_usd(5_000_000_000) == Decimal("5000000000")


def test_positive_net_income_does_not_erase_negative_cfo():
    result = calculate_burn_rate_v2(
        {"CFO": -242_581_000, "CFO_date": "2026-03-31", "NetIncome": 307_698_000, "Cash": 144_090_000}
    )
    assert result.burn_source == BurnSource.CFO_ANNUAL
    assert result.monthly_burn > 0
    scored = score_financial_health_v2(
        "KYMR",
        {"CFO": -242_581_000, "CFO_date": "2026-03-31", "NetIncome": 307_698_000, "Cash": 144_090_000},
        {"market_cap": 2_000_000_000, "price": 20, "avg_volume": 1_000_000},
    )
    assert scored["runway_months"] < 12
    assert scored["runway_months"] != 999


def test_positive_cfo_is_still_solvent():
    result = calculate_burn_rate_v2({"CFO": 100e6, "NetIncome": -10e6})
    assert result.burn_source == BurnSource.PROFITABLE
    assert result.monthly_burn == 0


def test_june_cfo_is_not_divided_by_twelve():
    undated = calculate_burn_rate_v2({"CFO": -120e6})
    dated = calculate_burn_rate_v2({"CFO": -120e6, "CFO_date": "2026-06-30"})
    explicit = calculate_burn_rate_v2({"CFO": -120e6, "CFO_date": "2026-06-30", "CFO_period_months": 6})
    assert undated.monthly_burn == Decimal("10000000")
    assert dated.monthly_burn == explicit.monthly_burn
    assert dated.monthly_burn == Decimal("20000000")


def test_cash_and_securities_beats_stale_marketable_tag():
    liquid = calculate_liquid_assets(
        {
            "Cash": 192_027_000,
            "MarketableSecurities": 2_018_477_000,
            "CashAndSecurities": 353_630_000,
        }
    )
    assert liquid.liquid_assets == Decimal("353630000")
    assert liquid.components_used == ["CashAndSecurities"]


def test_largest_investment_tag_when_no_deduped_total():
    liquid = calculate_liquid_assets(
        {
            "Cash": 100,
            "MarketableSecurities": 50,
            "ShortTermInvestments": 400,
        }
    )
    assert liquid.liquid_assets == Decimal("500")


def test_wrapper_scales_market_cap_and_skips_undated_rows():
    result = compute_module_2_financial(
        financial_records=[
            {
                "ticker": "CRSP",
                "Cash": 192_027_000,
                "CashAndSecurities": 353_630_000,
                "CFO": -282_079_000,
                "CFO_date": "2026-06-30",
                "source_date": "2026-08-01",
            },
            {"ticker": "NODATE", "Cash": 10e6, "CFO": -10e6},
        ],
        market_records=[
            {
                "ticker": "CRSP",
                "market_cap": 4849.1,
                "price": 49.99,
                "shares_outstanding": 98_484_128,
                "avg_volume": 1_700_000,
            }
        ],
        active_tickers=["CRSP", "NODATE"],
        as_of_date="2026-11-04",
    )
    by_ticker = {row["ticker"]: row for row in result["scores"]}
    assert "NODATE" not in by_ticker
    crsp = by_ticker["CRSP"]
    assert Decimal("0.05") < Decimal(str(crsp["cash_to_mcap"])) < Decimal("2")
    assert crsp["dilution_score"] < 100 or crsp["cash_to_mcap"] < 0.5


def test_future_statement_date_is_excluded():
    result = compute_module_2_financial(
        [
            {"ticker": "OLD", "Cash": 10e6, "source_date": "2023-12-30"},
            {"ticker": "FUTURE", "Cash": 10e6, "CFO_date": "2026-06-30", "source_date": "2023-12-30"},
        ],
        ["OLD", "FUTURE"],
        "2024-01-01",
    )
    tickers = [row["ticker"] for row in result["scores"]]
    assert tickers == ["OLD"]


def test_share_count_alias_is_read():
    scored = score_financial_health_v2(
        "X",
        {
            "Cash": 100e6,
            "CFO": -50e6,
            "shares_outstanding": 110,
            "CommonStockSharesOutstanding": 50,
        },
        {"market_cap": 1e9, "price": 10, "avg_volume": 1e6},
    )
    # The explicit shares_outstanding field wins over the EDGAR alias.
    assert scored["share_count_growth"] is None

    result = compute_module_2_financial(
        financial_records=[
            {
                "ticker": "X",
                "Cash": 100e6,
                "CFO": -50e6,
                "CommonStockSharesOutstanding": 120,
                "shares_outstanding_prior": 100,
                "source_date": "2026-01-01",
            }
        ],
        active_tickers=["X"],
        as_of_date="2026-06-01",
        market_records=[{"ticker": "X", "market_cap": 1_000_000_000, "price": 10, "avg_volume": 1_000_000}],
    )
    assert result["scores"][0]["share_count_growth"] == 0.2


def test_endpoint_abbreviations_do_not_match_inside_words():
    assert _classify_endpoint("Percent change from baseline in serum exposure") == "neutral"
    assert _classify_endpoint("Mean increase in forced vital capacity") == "neutral"
    assert _classify_endpoint("Number of participants with dose-limiting toxicity") == "neutral"
    assert _classify_endpoint("Change in creatinine clearance") == "neutral"
    assert _classify_endpoint("Overall survival") == "strong"
    assert _classify_endpoint("OS at 12 months") == "strong"
    assert _classify_endpoint("Safety and tolerability") == "weak"


def test_updated_trial_after_cutoff_is_not_scored():
    result = compute_module_4_clinical_dev(
        trial_records=[
            {
                "ticker": "AAAA",
                "nct_id": "NCT1",
                "phase": "PHASE3",
                "status": "COMPLETED",
                "first_posted": "2019-01-01",
                "last_update_posted": "2025-06-01",
                "primary_endpoint": "Overall survival",
            },
            {
                "ticker": "AAAA",
                "nct_id": "NCT2",
                "phase": "PHASE2",
                "status": "RECRUITING",
                "first_posted": "2024-01-01",
                "last_update_posted": "2024-06-01",
                "primary_endpoint": "Overall survival",
            },
        ],
        active_tickers=["AAAA"],
        as_of_date="2024-12-31",
    )
    score = result["scores"][0]
    assert score["trial_count"] == 1
    assert score["lead_phase"] == "phase 2"


def test_stale_pit_store_does_not_cover_a_later_as_of():
    store = {"facts": {"cash": [{"end": "2025-12-31", "filed": "2026-03-23", "val": 1}]}}
    assert pit_store_covers_as_of(store, "2024-06-01") is True
    assert pit_store_covers_as_of(store, "2026-05-01") is True
    assert pit_store_covers_as_of(store, "2026-11-04") is False
    assert pit_store_covers_as_of({"facts": {}}, "2026-11-04") is False


def test_pit_gap_threshold_is_configurable():
    store = {"facts": {"cash": [{"end": "2025-12-31", "filed": "2026-03-23", "val": 1}]}}
    assert pit_store_covers_as_of(store, "2026-11-04", max_gap_days=120) is False
    assert pit_store_covers_as_of(store, "2026-11-04", max_gap_days=400) is True
