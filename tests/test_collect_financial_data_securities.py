"""Static financial_data.json collector: liquid-assets aggregation (2026-11-16).

GRAL reports its treasury portfolio only under
DebtSecuritiesAvailableForSaleExcludingAccruedInterestCurrent, which the collector
did not read, so CashAndSecurities equalled Cash ($29M vs ~$789M). The PIT store
(tools/build_pit_financials.py CURRENT_INVESTMENT_TAGS) already reads it.
"""

from collect_financial_data import CURRENT_INVESTMENT_FIELDS, aggregate_cash_and_securities


def test_reads_excluding_accrued_interest_tag():
    fd = {
        "Cash": 28_976_000,
        "Cash_date": "2026-09-30",
        "DebtSecuritiesAvailableForSaleExclAccruedCurrent": 759_700_000,
        "DebtSecuritiesAvailableForSaleExclAccruedCurrent_date": "2026-09-30",
    }
    aggregate_cash_and_securities(fd)
    assert fd["CashAndSecurities"] == 28_976_000 + 759_700_000
    assert fd["CashAndSecurities_date"] == "2026-09-30"


def test_reads_held_to_maturity_current():
    fd = {
        "Cash": 100,
        "Cash_date": "2026-09-30",
        "HeldToMaturitySecuritiesCurrent": 50,
        "HeldToMaturitySecuritiesCurrent_date": "2026-09-30",
    }
    aggregate_cash_and_securities(fd)
    assert fd["CashAndSecurities"] == 150


def test_takes_max_across_tags_not_sum():
    fd = {
        "Cash": 100,
        "Cash_date": "2026-09-30",
        "ShortTermInvestments": 40,
        "ShortTermInvestments_date": "2026-09-30",
        "AvailableForSaleDebtCurrent": 60,
        "AvailableForSaleDebtCurrent_date": "2026-09-30",
    }
    aggregate_cash_and_securities(fd)
    assert fd["CashAndSecurities"] == 160


def test_no_cash_date_keeps_old_behaviour():
    fd = {"Cash": 100, "ShortTermInvestments": 40, "ShortTermInvestments_date": "2026-09-30"}
    aggregate_cash_and_securities(fd)
    assert fd["CashAndSecurities"] == 140


def test_field_list_covers_pit_store_tags():
    for f in ("DebtSecuritiesAvailableForSaleExclAccruedCurrent", "HeldToMaturitySecuritiesCurrent"):
        assert f in CURRENT_INVESTMENT_FIELDS
