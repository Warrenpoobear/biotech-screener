"""
pit_financials.py - Point-in-time financial data query module.

Loads per-ticker PIT fact stores (built by tools/build_pit_financials.py)
and returns financial snapshots as they were known on any historical date,
eliminating look-ahead bias from backtests.

The key insight: SEC filings have both an 'end' date (the period the data
covers, e.g. 2024-12-31) and a 'filed' date (when it was actually filed
with the SEC, e.g. 2025-02-13). A backtest on 2025-01-15 should NOT see
data from the 2024-12-31 10-K because it wasn't filed until February.

Usage:
    from pit_financials import pit_financial_snapshot, pit_financial_audit

    snap = pit_financial_snapshot("ALNY", "2025-03-01", data_dir)
    audit = pit_financial_audit("ALNY", "2025-03-01", data_dir)
"""

import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Default PIT data directory
DEFAULT_PIT_DIR = Path(__file__).resolve().parent / "production_data" / "pit_financials"

# Duration forms: 10-K covers a full year, 10-Q covers a single quarter.
# When computing TTM from 10-K annual figures we divide by 4 as an
# approximation. When using 10-Q quarterly figures we multiply by 4.
# This is a known simplification; quarterly seasonality is ignored.
ANNUAL_FORMS = {"10-K", "10-K/A", "20-F", "20-F/A"}
QUARTERLY_FORMS = {"10-Q", "10-Q/A", "6-K", "6-K/A"}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _load_pit_data(ticker: str, data_dir: Optional[Path] = None) -> Optional[dict]:
    """Load the PIT JSON file for a ticker."""
    pit_dir = data_dir or DEFAULT_PIT_DIR
    path = pit_dir / f"{ticker.upper()}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _filter_facts_pit(
    facts: list[dict],
    as_of_date: str,
) -> list[dict]:
    """Filter facts to those filed on or before as_of_date.

    Only returns facts where the 'filed' date is known and <= as_of_date.
    """
    return [f for f in facts if f.get("filed") and f["filed"] <= as_of_date]


def _dedup_by_end_date(facts: list[dict]) -> list[dict]:
    """For each unique 'end' date, keep only the latest-filed fact.

    This handles amendments: if a company files a 10-K/A that supersedes
    the original 10-K, we want the amended values.
    """
    by_end: dict[str, dict] = {}
    for f in facts:
        end = f.get("end")
        if not end:
            continue
        existing = by_end.get(end)
        if existing is None or f.get("filed", "") > existing.get("filed", ""):
            by_end[end] = f
    return list(by_end.values())


def _latest_fact(
    facts: list[dict],
    as_of_date: str,
) -> Optional[dict]:
    """Return the most recent fact as known on as_of_date.

    Steps:
    1. Filter to facts filed on or before as_of_date
    2. De-duplicate by end date (take latest amendment)
    3. Return the fact with the most recent end date
    """
    pit_facts = _filter_facts_pit(facts, as_of_date)
    if not pit_facts:
        return None
    deduped = _dedup_by_end_date(pit_facts)
    if not deduped:
        return None
    # Sort by end date descending, return most recent
    deduped.sort(key=lambda x: x.get("end", ""), reverse=True)
    return deduped[0]


def _latest_n_facts(
    facts: list[dict],
    as_of_date: str,
    n: int = 4,
) -> list[dict]:
    """Return the N most recent facts (by end date) as known on as_of_date.

    Used for computing TTM from quarterly data.
    """
    pit_facts = _filter_facts_pit(facts, as_of_date)
    if not pit_facts:
        return []
    deduped = _dedup_by_end_date(pit_facts)
    deduped.sort(key=lambda x: x.get("end", ""), reverse=True)
    return deduped[:n]


def _is_quarterly(fact: dict) -> bool:
    """Heuristic: a fact is quarterly if it has a start date ~90 days before end,
    or if the form is a 10-Q variant."""
    form = (fact.get("form") or "").upper()
    if form in QUARTERLY_FORMS:
        return True
    if form in ANNUAL_FORMS:
        return False
    # Heuristic from start/end dates
    start = fact.get("start")
    end = fact.get("end")
    if start and end:
        try:
            s = datetime.strptime(start, "%Y-%m-%d")
            e = datetime.strptime(end, "%Y-%m-%d")
            days = (e - s).days
            # Quarterly: roughly 80-100 days; annual: roughly 350-370
            return days < 200
        except ValueError:
            pass
    return False


def _is_annual(fact: dict) -> bool:
    """Check if a fact covers an annual period."""
    form = (fact.get("form") or "").upper()
    if form in ANNUAL_FORMS:
        return True
    if form in QUARTERLY_FORMS:
        return False
    start = fact.get("start")
    end = fact.get("end")
    if start and end:
        try:
            s = datetime.strptime(start, "%Y-%m-%d")
            e = datetime.strptime(end, "%Y-%m-%d")
            days = (e - s).days
            return days > 200
        except ValueError:
            pass
    return False


def _compute_ttm(
    facts: list[dict],
    as_of_date: str,
) -> Optional[float]:
    """Compute trailing-twelve-month value for a duration metric.

    Strategy:
    1. If the most recent fact is annual (10-K), use it directly as TTM.
       NOTE: This is an approximation -- the 10-K value IS the full year,
       but it may be stale (e.g. 10-K for FY ending Dec 2024 used in
       March 2025 misses Q1 2025 activity). This is acceptable because
       quarterly data may not be available for all companies.
    2. If the most recent fact is quarterly (10-Q), sum the 4 most recent
       non-overlapping quarterly values.
    3. If we have fewer than 4 quarters, annualize what we have.
    """
    recent = _latest_n_facts(facts, as_of_date, n=8)
    if not recent:
        return None

    latest = recent[0]

    # If the latest is annual, use it directly
    if _is_annual(latest):
        return float(latest["val"])

    # Try to sum 4 quarters
    quarterly = [f for f in recent if _is_quarterly(f)]
    if len(quarterly) >= 4:
        return sum(float(f["val"]) for f in quarterly[:4])

    # Annualize from fewer quarters
    if quarterly:
        n_q = min(len(quarterly), 4)
        q_sum = sum(float(f["val"]) for f in quarterly[:n_q])
        return q_sum * (4 / n_q)

    # Last resort: if the latest fact exists, use it as-is
    # (could be a 10-K annual value from an older period)
    return float(latest["val"])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def pit_financial_snapshot(
    ticker: str,
    as_of_date: str,
    data_dir: Optional[Path] = None,
    preloaded_data: Optional[dict] = None,
) -> Optional[dict]:
    """Return financial record for ticker as known on as_of_date.

    Returns dict with fields matching financial_records.json schema:
        - Cash, ShortTermInvestments, CashAndSecurities (= cash_total)
        - Assets, Liabilities, ShareholdersEquity
        - Revenue, OperatingExpenses, R&D, NetIncome, CFO
        - CommonStockSharesOutstanding, LongTermDebt
        - runway_months (derived), has_revenue (derived)
        - CashAndSecurities_date, collected_at

    Returns None if no PIT data file exists for the ticker.

    Args:
        preloaded_data: Pre-loaded JSON dict for this ticker (skips file I/O).
    """
    pit_data = preloaded_data if preloaded_data is not None else _load_pit_data(ticker, data_dir)
    if pit_data is None:
        return None

    facts = pit_data.get("facts", {})
    if not facts:
        return None

    # Extract latest instant values (balance sheet items)
    def _get_instant(field_name: str) -> tuple[Optional[float], Optional[str]]:
        field_facts = facts.get(field_name)
        if not field_facts:
            return None, None
        latest = _latest_fact(field_facts, as_of_date)
        if latest is None:
            return None, None
        return float(latest["val"]), latest.get("end")

    # Extract TTM values (income statement / cash flow items)
    def _get_ttm(field_name: str) -> Optional[float]:
        field_facts = facts.get(field_name)
        if not field_facts:
            return None
        return _compute_ttm(field_facts, as_of_date)

    # --- Balance sheet (instant) ---
    cash, cash_date = _get_instant("cash")
    short_term_inv, sti_date = _get_instant("short_term_investments")
    assets, assets_date = _get_instant("assets")
    liabilities, liab_date = _get_instant("liabilities")
    equity, equity_date = _get_instant("stockholders_equity")
    shares, shares_date = _get_instant("shares_outstanding")
    lt_debt, debt_date = _get_instant("long_term_debt")

    # --- Duration metrics (TTM) ---
    # Each TTM field needs its end date so downstream consumers (Module 2)
    # can determine the period length for burn rate calculation.
    def _get_ttm_with_date(field_name: str):
        """Get TTM value and the latest fact's end date."""
        field_facts = facts.get(field_name)
        if not field_facts:
            return None, None
        recent = _latest_n_facts(field_facts, as_of_date, n=8)
        if not recent:
            return None, None
        ttm_val = _compute_ttm(field_facts, as_of_date)
        # Use the latest fact's end date as the period end
        latest_end = recent[0].get("end") if recent else None
        # Determine if TTM was computed from annual (12mo) or quarterly data
        return ttm_val, latest_end

    revenue_ttm, revenue_date = _get_ttm_with_date("revenue")
    opex_ttm, opex_date = _get_ttm_with_date("operating_expenses")
    rd_ttm, rd_date = _get_ttm_with_date("research_and_development")
    net_income_ttm, ni_date = _get_ttm_with_date("net_income")
    ocf_ttm, ocf_date = _get_ttm_with_date("operating_cash_flow")

    # --- Derived metrics ---
    cash_total = (cash or 0) + (short_term_inv or 0)
    if cash is None and short_term_inv is None:
        cash_total = None

    # Burn TTM: positive number representing cash consumed per year.
    # Primary: negative operating cash flow. Fallback: opex - revenue.
    burn_ttm = None
    if ocf_ttm is not None and ocf_ttm < 0:
        burn_ttm = abs(ocf_ttm)
    elif opex_ttm is not None:
        rev = revenue_ttm or 0
        approx = opex_ttm - rev
        if approx > 0:
            burn_ttm = approx

    # Runway months
    runway_months = None
    if cash_total is not None and cash_total > 0 and burn_ttm is not None and burn_ttm > 0:
        monthly_burn = burn_ttm / 12
        runway_months = round(cash_total / monthly_burn, 1)

    has_revenue = revenue_ttm is not None and revenue_ttm > 0

    # Use the most recent date across balance sheet items
    all_dates = [d for d in [cash_date, sti_date, assets_date, liab_date] if d]
    most_recent_date = max(all_dates) if all_dates else None

    # Build record matching financial_records.json schema
    record = {
        "ticker": ticker.upper(),
        "cik": pit_data.get("cik"),
        # Balance sheet
        "Assets": assets,
        "Assets_date": assets_date,
        "Liabilities": liabilities,
        "Liabilities_date": liab_date,
        "ShareholdersEquity": equity,
        "ShareholdersEquity_date": equity_date,
        "Cash": cash,
        "Cash_date": cash_date,
        "ShortTermInvestments": short_term_inv,
        "ShortTermInvestments_date": sti_date,
        "CashAndSecurities": cash_total,
        "CashAndSecurities_date": most_recent_date,
        "LongTermDebt": lt_debt,
        "LongTermDebt_date": debt_date,
        "CommonStockSharesOutstanding": shares,
        "CommonStockSharesOutstanding_date": shares_date,
        # Income / cash flow (TTM) — dates enable Module 2 period detection
        "Revenue": revenue_ttm,
        "Revenue_date": revenue_date,
        "OperatingExpenses": opex_ttm,
        "OperatingExpenses_date": opex_date,
        "R&D": rd_ttm,
        "R&D_date": rd_date,
        "NetIncome": net_income_ttm,
        "NetIncome_date": ni_date,
        "CFO": ocf_ttm,
        "CFO_date": ocf_date,
        # Derived
        "cash_total": cash_total,
        "burn_ttm": burn_ttm,
        "runway_months": runway_months,
        "has_revenue": has_revenue,
        # Provenance
        "pit_as_of": as_of_date,
        "collected_at": pit_data.get("collected_at"),
    }

    # Also provide fields matching survivability module expectations
    record["cash_and_equivalents"] = cash
    record["short_term_investments"] = short_term_inv
    record["operating_cash_flow_ttm"] = ocf_ttm
    record["total_operating_expense_ttm"] = opex_ttm
    record["revenue_ttm"] = revenue_ttm
    record["r_and_d_expense_ttm"] = rd_ttm
    record["long_term_debt"] = lt_debt

    return record


def pit_financial_audit(
    ticker: str,
    as_of_date: str,
    data_dir: Optional[Path] = None,
) -> dict:
    """Return audit info: which filing was used, filing date, staleness.

    Returns dict with:
        - ticker, as_of_date
        - fields: dict of field_name -> {end, filed, form, staleness_days}
        - most_recent_filing: the latest filed date across all fields
        - data_available: True if any PIT data was found
    """
    pit_data = _load_pit_data(ticker, data_dir)
    if pit_data is None:
        return {
            "ticker": ticker.upper(),
            "as_of_date": as_of_date,
            "data_available": False,
            "fields": {},
        }

    facts = pit_data.get("facts", {})
    try:
        ref_date = datetime.strptime(as_of_date, "%Y-%m-%d").date()
    except ValueError:
        ref_date = date.today()

    field_audit = {}
    for field_name, field_facts in facts.items():
        latest = _latest_fact(field_facts, as_of_date)
        if latest is None:
            field_audit[field_name] = {"available": False}
            continue

        filed_str = latest.get("filed")
        end_str = latest.get("end")
        staleness_days = None
        if end_str:
            try:
                end_dt = datetime.strptime(end_str, "%Y-%m-%d").date()
                staleness_days = (ref_date - end_dt).days
            except ValueError:
                pass

        filing_lag_days = None
        if filed_str and end_str:
            try:
                filed_dt = datetime.strptime(filed_str, "%Y-%m-%d").date()
                end_dt = datetime.strptime(end_str, "%Y-%m-%d").date()
                filing_lag_days = (filed_dt - end_dt).days
            except ValueError:
                pass

        field_audit[field_name] = {
            "available": True,
            "end": end_str,
            "filed": filed_str,
            "form": latest.get("form"),
            "val": latest.get("val"),
            "staleness_days": staleness_days,
            "filing_lag_days": filing_lag_days,
        }

    # Compute summary
    filed_dates = [v["filed"] for v in field_audit.values() if v.get("available") and v.get("filed")]
    most_recent_filing = max(filed_dates) if filed_dates else None

    end_dates = [v["end"] for v in field_audit.values() if v.get("available") and v.get("end")]
    most_recent_period = max(end_dates) if end_dates else None

    staleness_values = [v["staleness_days"] for v in field_audit.values() if v.get("staleness_days") is not None]
    min_staleness = min(staleness_values) if staleness_values else None
    max_staleness = max(staleness_values) if staleness_values else None

    return {
        "ticker": ticker.upper(),
        "as_of_date": as_of_date,
        "data_available": bool(filed_dates),
        "most_recent_filing": most_recent_filing,
        "most_recent_period": most_recent_period,
        "staleness_days_range": (min_staleness, max_staleness) if min_staleness is not None else None,
        "fields_available": sum(1 for v in field_audit.values() if v.get("available")),
        "fields_total": len(field_audit),
        "fields": field_audit,
    }


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    ticker = sys.argv[1] if len(sys.argv) > 1 else "ALNY"
    as_of = sys.argv[2] if len(sys.argv) > 2 else "2025-12-01"
    data_dir_arg = Path(sys.argv[3]) if len(sys.argv) > 3 else None

    print(f"PIT Financial Snapshot: {ticker} as of {as_of}")
    print("=" * 60)

    snap = pit_financial_snapshot(ticker, as_of, data_dir_arg)
    if snap is None:
        print(f"No PIT data found for {ticker}")
        print(f"Run: python tools/build_pit_financials.py --ticker {ticker}")
        sys.exit(1)

    # Print key fields
    key_fields = [
        ("Cash", "Cash"),
        ("Short-term investments", "ShortTermInvestments"),
        ("Cash + Securities", "CashAndSecurities"),
        ("Assets", "Assets"),
        ("Liabilities", "Liabilities"),
        ("Equity", "ShareholdersEquity"),
        ("Revenue (TTM)", "Revenue"),
        ("OpEx (TTM)", "OperatingExpenses"),
        ("R&D (TTM)", "R&D"),
        ("Net Income (TTM)", "NetIncome"),
        ("Operating CF (TTM)", "CFO"),
        ("Long-term Debt", "LongTermDebt"),
        ("Shares Outstanding", "CommonStockSharesOutstanding"),
        ("Burn TTM", "burn_ttm"),
        ("Runway (months)", "runway_months"),
        ("Has Revenue", "has_revenue"),
    ]

    for label, key in key_fields:
        val = snap.get(key)
        if val is None:
            print(f"  {label:30s}  --")
        elif isinstance(val, bool):
            print(f"  {label:30s}  {val}")
        elif isinstance(val, float) and abs(val) >= 1_000_000:
            print(f"  {label:30s}  ${val/1e6:,.1f}M")
        elif isinstance(val, float):
            print(f"  {label:30s}  {val:,.2f}")
        else:
            print(f"  {label:30s}  {val}")

    print()
    print("Audit trail:")
    print("-" * 60)
    audit = pit_financial_audit(ticker, as_of, data_dir_arg)
    print(f"  Data available:       {audit['data_available']}")
    print(f"  Fields available:     {audit['fields_available']}/{audit['fields_total']}")
    print(f"  Most recent filing:   {audit['most_recent_filing']}")
    print(f"  Most recent period:   {audit['most_recent_period']}")
    if audit.get("staleness_days_range"):
        lo, hi = audit["staleness_days_range"]
        print(f"  Staleness range:      {lo}-{hi} days")

    # Show per-field detail
    print()
    print("  Per-field detail:")
    for field, info in audit["fields"].items():
        if info.get("available"):
            lag = info.get("filing_lag_days", "?")
            print(
                f"    {field:28s}  end={info['end']}  filed={info['filed']}  "
                f"form={info.get('form', '?'):6s}  lag={lag}d  stale={info.get('staleness_days', '?')}d"
            )
        else:
            print(f"    {field:28s}  (not available)")
