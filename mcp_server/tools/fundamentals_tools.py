"""Fundamental financial health and cash-burn tools."""

import json
import logging

from mcp_server.app import mcp
from mcp_server.config import FINANCIAL_RECORDS_FILE, UNIVERSE_FILE, UNIVERSE_MAPPED_FILE

logger = logging.getLogger(__name__)


def _load_financial_records() -> list[dict]:
    """Load financial_records.json."""
    if FINANCIAL_RECORDS_FILE.exists():
        with open(FINANCIAL_RECORDS_FILE) as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    return []


def _load_universe() -> list[dict]:
    """Load the universe file for supplementary data."""
    for path in (UNIVERSE_MAPPED_FILE, UNIVERSE_FILE):
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
    return []


@mcp.tool()
def get_financial_health(ticker: str) -> str:
    """Return key financial health metrics for a ticker.

    Args:
        ticker: Equity ticker symbol (e.g. "VRTX").

    Reports cash, debt, equity, revenue, R&D, net income, and derived
    ratios (current ratio, debt-to-equity, cash runway estimate).
    """
    try:
        records = _load_financial_records()
        match = [r for r in records if r.get("ticker", "").upper() == ticker.upper()]

        if not match:
            # Fall back to universe data
            universe = _load_universe()
            uni_match = [
                u for u in universe
                if u.get("ticker", "").upper() == ticker.upper()
            ]
            if not uni_match:
                return json.dumps({"ticker": ticker, "error": "Ticker not found"})
            entry = uni_match[0]
            fin = entry.get("financial_data") or entry.get("financials") or {}
            return json.dumps({
                "ticker": ticker,
                "source": "universe",
                "data": {k: v for k, v in fin.items() if v is not None},
            }, default=str)

        rec = match[0]

        cash = rec.get("Cash") or rec.get("TotalLiquidity") or 0
        debt = rec.get("Debt") or rec.get("LongTermDebt") or 0
        equity = rec.get("ShareholdersEquity") or 0
        assets = rec.get("Assets") or 0
        current_assets = rec.get("CurrentAssets") or 0
        liabilities = rec.get("Liabilities") or 0
        current_liabilities = rec.get("CurrentLiabilities") or 0
        revenue = rec.get("Revenue") or 0
        rd = rec.get("R&D") or 0
        net_income = rec.get("NetIncome") or 0
        cfo = rec.get("CFO") or 0
        opex = rec.get("OperatingExpenses") or 0

        health: dict = {
            "ticker": ticker,
            "cash": cash,
            "debt": debt,
            "equity": equity,
            "assets": assets,
            "revenue": revenue,
            "r_and_d": rd,
            "net_income": net_income,
            "operating_cash_flow": cfo,
            "operating_expenses": opex,
            "cash_date": rec.get("Cash_date"),
            "data_source": rec.get("Cash_source", "SEC"),
        }

        # Derived ratios
        if current_liabilities and current_liabilities > 0:
            health["current_ratio"] = round(current_assets / current_liabilities, 2)
        if equity and equity > 0:
            health["debt_to_equity"] = round(debt / equity, 3)
        if cash and opex and opex > 0:
            # Quarterly burn → annual → runway in months
            quarterly_burn = opex - revenue if revenue < opex else 0
            if quarterly_burn > 0:
                annual_burn = quarterly_burn * 4
                runway_years = cash / annual_burn
                health["estimated_cash_runway_months"] = round(runway_years * 12, 1)

        return json.dumps(health, default=str)
    except Exception as e:
        logger.exception("get_financial_health failed for %s", ticker)
        return json.dumps({"ticker": ticker, "error": str(e)})


@mcp.tool()
def get_cash_burn_analysis(ticker: str) -> str:
    """Estimate quarterly cash burn and runway for a pre-revenue biotech.

    Args:
        ticker: Equity ticker symbol.

    Uses operating expenses, revenue, and cash position to estimate
    how many quarters of cash remain.
    """
    try:
        records = _load_financial_records()
        match = [r for r in records if r.get("ticker", "").upper() == ticker.upper()]

        if not match:
            return json.dumps({"ticker": ticker, "error": "No financial records found"})

        rec = match[0]
        cash = rec.get("Cash") or rec.get("TotalLiquidity") or 0
        revenue = rec.get("Revenue") or 0
        opex = rec.get("OperatingExpenses") or 0
        rd = rec.get("R&D") or 0
        cfo = rec.get("CFO") or 0
        net_income = rec.get("NetIncome") or 0

        # Net burn = opex - revenue (if burning cash)
        net_burn_quarterly = max(opex - revenue, 0)

        result: dict = {
            "ticker": ticker,
            "cash_position": cash,
            "quarterly_revenue": revenue,
            "quarterly_opex": opex,
            "quarterly_r_and_d": rd,
            "quarterly_net_burn": net_burn_quarterly,
            "operating_cash_flow": cfo,
            "net_income": net_income,
            "is_profitable": net_income > 0,
            "is_revenue_stage": revenue > 0,
        }

        if net_burn_quarterly > 0 and cash > 0:
            quarters_remaining = cash / net_burn_quarterly
            result["quarters_of_cash"] = round(quarters_remaining, 1)
            result["months_of_cash"] = round(quarters_remaining * 3, 1)
            if quarters_remaining < 4:
                result["warning"] = "Less than 1 year of cash remaining"
        elif net_income > 0:
            result["note"] = "Company is profitable; no cash burn"
        elif cash == 0:
            result["warning"] = "No cash data available"

        return json.dumps(result, default=str)
    except Exception as e:
        logger.exception("get_cash_burn_analysis failed for %s", ticker)
        return json.dumps({"ticker": ticker, "error": str(e)})
