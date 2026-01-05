"""
Module 2: Financial Health

Computes financial health score from balance sheet data.
Components: Cash runway, debt burden, market cap stability.

Input: Financial data records (cash, debt, burn rate, market cap)
Output: Financial scores per security
"""
from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

from common.provenance import create_provenance
from common.pit_enforcement import compute_pit_cutoff, is_pit_admissible
from common.types import Severity

RULESET_VERSION = "1.0.0"

# Scoring thresholds
RUNWAY_EXCELLENT_MONTHS = Decimal("36")
RUNWAY_GOOD_MONTHS = Decimal("24")
RUNWAY_ADEQUATE_MONTHS = Decimal("12")

DEBT_LOW_RATIO = Decimal("0.1")
DEBT_MODERATE_RATIO = Decimal("0.3")


def _compute_runway_score(runway_months: Optional[Decimal]) -> Decimal:
    """Score runway: 0-40 points."""
    if runway_months is None:
        return Decimal("20")  # Neutral
    
    if runway_months >= RUNWAY_EXCELLENT_MONTHS:
        return Decimal("40")
    elif runway_months >= RUNWAY_GOOD_MONTHS:
        return Decimal("32")
    elif runway_months >= RUNWAY_ADEQUATE_MONTHS:
        return Decimal("24")
    elif runway_months >= Decimal("6"):
        return Decimal("12")
    else:
        return Decimal("0")


def _compute_debt_score(debt_mm: Optional[Decimal], market_cap_mm: Optional[Decimal]) -> Decimal:
    """Score debt burden: 0-30 points."""
    if debt_mm is None or market_cap_mm is None or market_cap_mm == 0:
        return Decimal("15")  # Neutral
    
    ratio = debt_mm / market_cap_mm
    
    if ratio <= DEBT_LOW_RATIO:
        return Decimal("30")
    elif ratio <= DEBT_MODERATE_RATIO:
        return Decimal("22")
    elif ratio <= Decimal("0.5"):
        return Decimal("15")
    else:
        return Decimal("5")


def _compute_size_score(market_cap_mm: Optional[Decimal]) -> Decimal:
    """Score market cap (stability proxy): 0-30 points."""
    if market_cap_mm is None:
        return Decimal("15")
    
    if market_cap_mm >= Decimal("10000"):  # Large cap
        return Decimal("30")
    elif market_cap_mm >= Decimal("2000"):  # Mid cap
        return Decimal("25")
    elif market_cap_mm >= Decimal("500"):  # Small cap
        return Decimal("18")
    elif market_cap_mm >= Decimal("100"):  # Micro cap
        return Decimal("10")
    else:
        return Decimal("5")


def compute_module_2_financial(
    financial_records: List[Dict[str, Any]],
    active_tickers: List[str],
    as_of_date: str,
) -> Dict[str, Any]:
    """
    Compute financial health scores.
    
    Args:
        financial_records: List with ticker, cash_mm, debt_mm, burn_rate_mm, market_cap_mm, source_date
        active_tickers: Tickers from Module 1
        as_of_date: Analysis date
    
    Returns:
        {
            "as_of_date": str,
            "scores": [{ticker, financial_score, runway_months, components, severity, flags}],
            "missing_tickers": [str],
            "diagnostic_counts": {...},
            "provenance": {...}
        }
    """
    pit_cutoff = compute_pit_cutoff(as_of_date)
    
    # Index records by ticker (latest PIT-admissible)
    ticker_data: Dict[str, Dict[str, Any]] = {}
    for rec in financial_records:
        ticker = rec.get("ticker", "").upper()
        source_date = rec.get("source_date")
        
        if not is_pit_admissible(source_date, pit_cutoff):
            continue
        
        # Keep latest
        if ticker not in ticker_data or source_date > ticker_data[ticker].get("source_date", ""):
            ticker_data[ticker] = rec
    
    scores = []
    missing = []
    
    for ticker in active_tickers:
        if ticker not in ticker_data:
            missing.append(ticker)
            continue
        
        data = ticker_data[ticker]
        
        # Parse values
        cash_mm = Decimal(str(data["cash_mm"])) if data.get("cash_mm") else None
        debt_mm = Decimal(str(data["debt_mm"])) if data.get("debt_mm") else None
        burn_rate = Decimal(str(data["burn_rate_mm"])) if data.get("burn_rate_mm") else None
        market_cap = Decimal(str(data["market_cap_mm"])) if data.get("market_cap_mm") else None
        
        # Compute runway
        runway = None
        if cash_mm is not None and burn_rate is not None and burn_rate > 0:
            runway = cash_mm / burn_rate
        
        # Compute component scores
        runway_score = _compute_runway_score(runway)
        debt_score = _compute_debt_score(debt_mm, market_cap)
        size_score = _compute_size_score(market_cap)
        
        total = runway_score + debt_score + size_score
        
        # Severity flags
        severity = Severity.NONE
        flags = []
        
        if runway is not None and runway < Decimal("6"):
            severity = Severity.SEV2
            flags.append("low_runway")
        
        scores.append({
            "ticker": ticker,
            "financial_score": str(total.quantize(Decimal("0.01"))),
            "runway_months": str(runway.quantize(Decimal("0.1"))) if runway else None,
            "market_cap_mm": str(market_cap) if market_cap else None,
            "components": {
                "runway": str(runway_score),
                "debt": str(debt_score),
                "size": str(size_score),
            },
            "severity": severity.value,
            "flags": flags,
        })
    
    return {
        "as_of_date": as_of_date,
        "scores": sorted(scores, key=lambda x: x["ticker"]),
        "missing_tickers": sorted(missing),
        "diagnostic_counts": {
            "scored": len(scores),
            "missing": len(missing),
            "pit_cutoff": pit_cutoff,
        },
        "provenance": create_provenance(RULESET_VERSION, {"tickers": active_tickers}, pit_cutoff),
    }
