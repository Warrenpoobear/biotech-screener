"""ETF holdings audit: IBB holdings must carry a fresh as-of date or be refused.

Fixtures are synthetic iShares-format CSV text; no live data is fetched.
"""

from datetime import date

import tools.audit_universe_against_xbi_ibb as audit


def _ishares_csv(as_of: str) -> str:
    return (
        "iShares Biotechnology ETF\n"
        f'Fund Holdings as of,"{as_of}"\n'
        'Inception Date,"Feb 05, 2001"\n'
        "\n"
        "Ticker,Name,Sector,Asset Class,Market Value,Weight (%),Notional Value,Quantity\n"
        '"GILD","GILEAD SCIENCES INC","Health Care","Equity","1","9.01","1","1"\n'
        '"AMGN","AMGEN INC","Health Care","Equity","1","8.50","1","1"\n'
        '"USD","USD CASH","Cash and/or Derivatives","Cash","1","0.10","1","1"\n'
    )


def test_parse_ishares_csv_reads_as_of_date_and_equity_rows():
    df, as_of = audit.parse_ishares_holdings_csv(_ishares_csv("Nov 14, 2026"))
    assert as_of == date(2026, 11, 14)
    assert list(df["ticker"]) == ["GILD", "AMGN"]
    assert list(df.columns[:3]) == ["ticker", "name", "weight_pct"]


def test_holdings_older_than_limit_are_stale():
    assert audit.holdings_are_stale(date(2026, 6, 26), today=date(2026, 11, 17))
    assert not audit.holdings_are_stale(date(2026, 11, 14), today=date(2026, 11, 17))
    assert audit.holdings_are_stale(None, today=date(2026, 11, 17))  # unknown date is never trusted


def test_stale_ibb_content_is_refused_not_used():
    df, source = audit.ibb_holdings_from_content(_ishares_csv("Jun 26, 2026"), "TEST", today=date(2026, 11, 17))
    assert df.empty
    assert source.startswith("FETCH_FAILED_STALE_HOLDINGS") and "2026-06-26" in source


def test_fresh_ibb_content_is_used_with_real_as_of_in_source():
    df, source = audit.ibb_holdings_from_content(_ishares_csv("Nov 14, 2026"), "TEST", today=date(2026, 11, 17))
    assert len(df) == 2
    assert "as_of=2026-11-14" in source and "20260626" not in source
