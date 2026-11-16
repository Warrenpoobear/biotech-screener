"""Liquid assets in the PIT store must include current marketable securities.

Before 2026-11-16 the builder kept only the first investment tag that had any facts
(MarketableSecuritiesCurrent / ShortTermInvestments / AvailableForSaleSecuritiesCurrent /
MarketableSecurities). Most filers report their treasury portfolio under
AvailableForSaleSecuritiesDebtSecuritiesCurrent, so the PIT store held cash only:
CYTK 2026-09-30 read $203M liquid instead of $1,130M (cash 203 + AFS current 927) and
was excluded by the sev3 gate.

Fixture values are CYTK / DNTH figures from SEC companyfacts ($M).
"""

from pit_financials import pit_financial_snapshot
from tools.build_pit_financials import extract_all_facts


def _fact(end, val_m, filed, form="10-Q"):
    return {"end": end, "val": val_m * 1e6, "filed": filed, "form": form, "accn": f"a-{end}-{filed}"}


def _companyfacts(tags):
    return {"facts": {"us-gaap": {t: {"units": {"USD": vals}} for t, vals in tags.items()}}}


def _snap(facts, as_of="2026-11-12"):
    return pit_financial_snapshot("TEST", as_of, preloaded_data={"ticker": "TEST", "cik": "1", "facts": facts})


def _m(x):
    return None if x is None else round(x / 1e6)


CASH = [_fact("2026-06-30", 255, "2026-08-06"), _fact("2026-09-30", 203, "2026-11-05")]


def test_builder_collects_every_current_investment_tag():
    cf = _companyfacts(
        {
            "CashAndCashEquivalentsAtCarryingValue": CASH,
            # An old tag with facts must not shadow the one the filer uses today.
            "MarketableSecuritiesCurrent": [_fact("2021-12-31", 50, "2022-02-20", "10-K")],
            "AvailableForSaleSecuritiesDebtSecuritiesCurrent": [_fact("2026-09-30", 927, "2026-11-05")],
            "DebtSecuritiesAvailableForSaleExcludingAccruedInterestCurrent": [_fact("2026-09-30", 760, "2026-11-05")],
        }
    )
    sti = extract_all_facts(cf)["short_term_investments"]
    assert {f["tag"] for f in sti} == {
        "MarketableSecuritiesCurrent",
        "AvailableForSaleSecuritiesDebtSecuritiesCurrent",
        "DebtSecuritiesAvailableForSaleExcludingAccruedInterestCurrent",
    }


def test_snapshot_adds_afs_debt_securities_to_cash():
    snap = _snap(
        {
            "cash": CASH,
            "short_term_investments": [
                dict(_fact("2026-09-30", 927, "2026-11-05"), tag="AvailableForSaleSecuritiesDebtSecuritiesCurrent")
            ],
        }
    )
    assert _m(snap["Cash"]) == 203
    assert _m(snap["ShortTermInvestments"]) == 927
    assert _m(snap["CashAndSecurities"]) == 1130


def test_overlapping_tags_take_the_max_not_the_sum():
    # Same portfolio under two tags (static collector rule: max, never add).
    sti = [
        dict(_fact("2026-09-30", 782, "2026-11-05"), tag="ShortTermInvestments"),
        dict(_fact("2026-09-30", 780, "2026-11-05"), tag="AvailableForSaleSecuritiesDebtSecuritiesCurrent"),
    ]
    snap = _snap({"cash": [_fact("2026-09-30", 110, "2026-11-05")], "short_term_investments": sti})
    assert _m(snap["CashAndSecurities"]) == 110 + 782


def test_investments_from_another_balance_sheet_date_are_not_added():
    # A stale tag (2021) must not be stacked on 2026 cash.
    sti = [dict(_fact("2021-12-31", 50, "2022-02-20", "10-K"), tag="MarketableSecuritiesCurrent")]
    snap = _snap({"cash": CASH, "short_term_investments": sti})
    assert _m(snap["CashAndSecurities"]) == 203
    assert snap["ShortTermInvestments"] is None


def test_point_in_time_uses_the_pair_known_on_as_of_date():
    sti = [
        dict(_fact("2026-06-30", 960, "2026-08-06"), tag="AvailableForSaleSecuritiesDebtSecuritiesCurrent"),
        dict(_fact("2026-09-30", 927, "2026-11-05"), tag="AvailableForSaleSecuritiesDebtSecuritiesCurrent"),
    ]
    # Before the Q3 10-Q is filed, both cash and securities come from 2026-06-30.
    snap = _snap({"cash": CASH, "short_term_investments": sti}, as_of="2026-10-15")
    assert _m(snap["CashAndSecurities"]) == 255 + 960
    assert snap["CashAndSecurities_date"] == "2026-06-30"


def test_legacy_records_without_tag_field_still_work():
    # Files written by the old builder carry one tag's facts with no "tag" key.
    snap = _snap({"cash": CASH, "short_term_investments": [_fact("2026-09-30", 38, "2026-11-05")]})
    assert _m(snap["CashAndSecurities"]) == 203 + 38


def test_builder_collects_every_cash_tag():
    cf = _companyfacts(
        {
            "CashAndCashEquivalentsAtCarryingValue": [_fact("2024-06-30", 69, "2024-08-08")],
            "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents": [_fact("2026-06-30", 120, "2026-08-04")],
        }
    )
    tags = {f["tag"] for f in extract_all_facts(cf)["cash"]}
    assert tags == {
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    }


def test_cash_uses_latest_balance_sheet_across_tags():
    # ARWR: the primary tag stopped in 2024; the filer now reports cash incl. restricted.
    cash = [
        dict(_fact("2024-06-30", 69, "2024-08-08"), tag="CashAndCashEquivalentsAtCarryingValue"),
        dict(
            _fact("2026-06-30", 120, "2026-08-04"), tag="CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"
        ),
    ]
    sti = [dict(_fact("2026-06-30", 1547, "2026-08-04"), tag="AvailableForSaleSecuritiesDebtSecuritiesCurrent")]
    snap = _snap({"cash": cash, "short_term_investments": sti})
    assert snap["Cash_date"] == "2026-06-30"
    assert _m(snap["CashAndSecurities"]) == 120 + 1547


def test_cash_prefers_unrestricted_tag_on_the_same_date():
    cash = [
        dict(_fact("2026-09-30", 203, "2026-11-05"), tag="CashAndCashEquivalentsAtCarryingValue"),
        dict(
            _fact("2026-09-30", 207, "2026-11-05"), tag="CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"
        ),
    ]
    assert _m(_snap({"cash": cash})["Cash"]) == 203


def test_ifrs_cash_and_equivalents_beats_narrow_cash_line():
    # ABVX (20-F, IFRS) reports CashAndCashEquivalents 517 and a narrower "Cash" 38 on the same date.
    cash = [
        dict(_fact("2025-12-31", 38, "2026-03-23", "20-F"), tag="Cash"),
        dict(_fact("2025-12-31", 517, "2026-03-23", "20-F"), tag="CashAndCashEquivalents"),
    ]
    assert _m(_snap({"cash": cash})["Cash"]) == 517
