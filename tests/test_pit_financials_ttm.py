"""TTM for duration metrics must respect each fact's period, not its form type.

10-Q cash-flow statements are year-to-date (Q2 = 6 months, Q3 = 9 months).
Treating every 10-Q fact as one quarter and summing four of them mixes 3/6/9-month
periods, which produced e.g. MDGL runway -1.6 months on 2026-10-15.

Fixture values are MDGL operating cash flow from SEC companyfacts ($M), with the
'filed' dates that make them PIT-available.
"""

import pytest

from pit_financials import _compute_ttm, pit_financial_snapshot


def _f(start, end, val_m, form, filed):
    return {"start": start, "end": end, "val": val_m * 1e6, "form": form, "filed": filed}


MDGL_OCF = [
    _f("2025-01-01", "2025-03-31", -88.9, "10-Q", "2025-05-01"),
    _f("2025-01-01", "2025-06-30", -135.9, "10-Q", "2025-08-05"),
    _f("2025-01-01", "2025-09-30", -56.1, "10-Q", "2025-11-04"),
    _f("2025-01-01", "2025-12-31", -189.6, "10-K", "2026-02-19"),
    _f("2026-01-01", "2026-03-31", -167.4, "10-Q", "2026-05-06"),
    _f("2025-01-01", "2025-03-31", -88.9, "10-Q", "2026-05-06"),  # prior-year comparative
    _f("2026-01-01", "2026-06-30", -144.1, "10-Q", "2026-07-30"),
    _f("2025-01-01", "2025-06-30", -135.9, "10-Q", "2026-07-30"),  # prior-year comparative
]


def _m(x):
    return round(x / 1e6, 1)


def test_ytd_q2_uses_annual_plus_ytd_minus_prior_ytd():
    # TTM to 2026-06-30 = H1'26 + FY'25 - H1'25 = -144.1 - 189.6 + 135.9
    assert _m(_compute_ttm(MDGL_OCF, "2026-10-15")) == -197.8


def test_ytd_q1_uses_annual_plus_ytd_minus_prior_ytd():
    # As of 2026-06-01 the latest is Q1'26: -167.4 - 189.6 + 88.9
    assert _m(_compute_ttm(MDGL_OCF, "2026-06-01")) == -268.1


def test_latest_annual_is_used_directly():
    # As of 2026-03-01 the 10-K is the latest fact: unchanged behaviour
    assert _m(_compute_ttm(MDGL_OCF, "2026-03-01")) == -189.6


def test_pit_excludes_facts_filed_after_as_of():
    # On 2026-07-29 the Q2 10-Q (filed 07-30) is not yet known -> Q1-based TTM
    assert _m(_compute_ttm(MDGL_OCF, "2026-07-29")) == -268.1


def test_discrete_quarters_are_summed():
    facts = [
        _f("2025-07-01", "2025-09-30", 10.0, "10-Q", "2025-11-01"),
        _f("2025-10-01", "2025-12-31", 20.0, "10-K", "2026-02-15"),
        _f("2026-01-01", "2026-03-31", 30.0, "10-Q", "2026-05-01"),
        _f("2026-04-01", "2026-06-30", 40.0, "10-Q", "2026-08-01"),
    ]
    assert _m(_compute_ttm(facts, "2026-10-15")) == 100.0


def test_ytd_without_annual_is_annualized_by_duration():
    facts = [_f("2026-01-01", "2026-06-30", -60.0, "10-Q", "2026-08-01")]
    # 181-day period annualized to 365 days
    assert _m(_compute_ttm(facts, "2026-10-15")) == pytest.approx(-121.0, abs=0.1)


def test_runway_for_mdgl_like_record_is_positive():
    data = {
        "ticker": "MDGL",
        "cik": "0001157601",
        "facts": {
            "cash": [{"end": "2026-06-30", "val": 292.7e6, "form": "10-Q", "filed": "2026-07-30"}],
            "operating_cash_flow": MDGL_OCF,
        },
    }
    rec = pit_financial_snapshot("MDGL", "2026-10-15", preloaded_data=data)
    assert _m(rec["burn_ttm"]) == 197.8
    assert rec["runway_months"] == pytest.approx(292.7 / (197.8 / 12), abs=0.1)
