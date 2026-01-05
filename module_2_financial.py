"""
Module 2: Financial Health Assessment - Enhanced with Raw Value Pass-through

Architecture:
    - Computes financial health scores from financial.json records
    - Now ALSO passes through raw financial values for downstream transparency:
        * cash_usd
        * debt_usd  
        * ttm_revenue_usd
        * market_cap_usd (optional)
    
CCFT Compliance:
    - All raw values preserved exactly as input (no transformation)
    - Explicit None for missing values (never fabricate)
    - Coverage tracking for governance
"""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Set


# CCFT: Quantization for deterministic score representation  
DECIMAL_QUANTIZATION = Decimal("0.0001")


def _quantize(value: Decimal) -> Decimal:
    """CCFT: Quantize decimal values for deterministic representation."""
    return value.quantize(DECIMAL_QUANTIZATION, rounding=ROUND_HALF_UP)


def _safe_decimal(value: Any) -> Optional[Decimal]:
    """Convert value to Decimal safely, returning None if invalid/missing."""
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except (ValueError, TypeError):
        return None


def compute_module_2_financial(
    financial_records: List[Dict[str, Any]],
    active_tickers: Set[str],
    as_of_date: str,
) -> Dict[str, Any]:
    """
    Compute financial health scores AND pass through raw values.
    
    Args:
        financial_records: List of financial records from financial.json
        active_tickers: Set of tickers that passed Module 1 filter
        as_of_date: Point-in-time date for the analysis (YYYY-MM-DD)
    
    Returns:
        Dict containing:
            - securities: Dict[ticker, security_financial_data]
            - coverage_stats: Coverage metrics for governance
            - as_of_date: Threading the analysis date
    """
    # Index financial records by ticker for O(1) lookup
    financial_by_ticker: Dict[str, Dict[str, Any]] = {}
    for record in financial_records:
        ticker = record.get("ticker")
        if ticker:
            financial_by_ticker[ticker] = record
    
    securities: Dict[str, Dict[str, Any]] = {}
    
    # Coverage counters
    coverage_stats = {
        "total_active": len(active_tickers),
        "has_cash": 0,
        "has_debt": 0,
        "has_revenue": 0,
        "has_market_cap": 0,
        "has_any_financial": 0,
    }
    
    for ticker in sorted(active_tickers):  # CCFT: Deterministic iteration
        fin_record = financial_by_ticker.get(ticker, {})
        
        # Extract raw values - NEVER fabricate, explicit None if missing
        cash_usd = _safe_decimal(fin_record.get("cash_usd"))
        debt_usd = _safe_decimal(fin_record.get("debt_usd"))
        ttm_revenue_usd = _safe_decimal(fin_record.get("ttm_revenue_usd"))
        market_cap_usd = _safe_decimal(fin_record.get("market_cap_usd"))
        
        # Track coverage
        if cash_usd is not None:
            coverage_stats["has_cash"] += 1
        if debt_usd is not None:
            coverage_stats["has_debt"] += 1
        if ttm_revenue_usd is not None:
            coverage_stats["has_revenue"] += 1
        if market_cap_usd is not None:
            coverage_stats["has_market_cap"] += 1
        if any([cash_usd, debt_usd, ttm_revenue_usd, market_cap_usd]):
            coverage_stats["has_any_financial"] += 1
        
        # Compute derived metrics (where possible)
        net_cash_usd = None
        if cash_usd is not None and debt_usd is not None:
            net_cash_usd = cash_usd - debt_usd
        
        # Compute financial health score
        health_score, health_flags = _compute_health_score(
            cash_usd=cash_usd,
            debt_usd=debt_usd,
            ttm_revenue_usd=ttm_revenue_usd,
            market_cap_usd=market_cap_usd,
        )
        
        # Build security output with raw pass-through values
        securities[ticker] = {
            # Raw pass-through values (Tier-0 provenance)
            "cash_usd": str(cash_usd) if cash_usd is not None else None,
            "debt_usd": str(debt_usd) if debt_usd is not None else None,
            "ttm_revenue_usd": str(ttm_revenue_usd) if ttm_revenue_usd is not None else None,
            "market_cap_usd": str(market_cap_usd) if market_cap_usd is not None else None,
            
            # Derived metrics
            "net_cash_usd": str(net_cash_usd) if net_cash_usd is not None else None,
            
            # Health assessment
            "health_score": str(health_score) if health_score is not None else None,
            "health_flags": health_flags,
            
            # Data quality metadata
            "financial_fields_present": sum([
                cash_usd is not None,
                debt_usd is not None,
                ttm_revenue_usd is not None,
                market_cap_usd is not None,
            ]),
            "data_source": fin_record.get("data_source", "unknown"),
        }
    
    # Compute coverage percentages
    total = coverage_stats["total_active"] or 1  # Avoid division by zero
    coverage_stats["cash_coverage_pct"] = round(100 * coverage_stats["has_cash"] / total, 1)
    coverage_stats["debt_coverage_pct"] = round(100 * coverage_stats["has_debt"] / total, 1)
    coverage_stats["revenue_coverage_pct"] = round(100 * coverage_stats["has_revenue"] / total, 1)
    coverage_stats["market_cap_coverage_pct"] = round(100 * coverage_stats["has_market_cap"] / total, 1)
    
    return {
        "securities": securities,
        "coverage_stats": coverage_stats,
        "as_of_date": as_of_date,
    }


def _compute_health_score(
    cash_usd: Optional[Decimal],
    debt_usd: Optional[Decimal],
    ttm_revenue_usd: Optional[Decimal],
    market_cap_usd: Optional[Decimal],
) -> tuple[Optional[Decimal], List[str]]:
    """
    Compute financial health score with explicit flag generation.
    
    Returns:
        Tuple of (score, flags) where:
            - score: 0-100 health score (or None if insufficient data)
            - flags: List of pattern/warning flags detected
    """
    flags: List[str] = []
    
    # Track what we're missing
    if cash_usd is None:
        flags.append("MISSING_CASH")
    if debt_usd is None:
        flags.append("MISSING_DEBT")
    if ttm_revenue_usd is None:
        flags.append("MISSING_REVENUE")
    
    # Cannot compute meaningful score without cash
    if cash_usd is None:
        return None, flags
    
    score = Decimal("50")  # Base score
    
    # Pattern detection
    debt = debt_usd or Decimal("0")
    net_cash = cash_usd - debt
    
    # Cash Fortress pattern: net cash > $1B
    if net_cash >= Decimal("1000000000"):
        flags.append("CASH_FORTRESS")
        score += Decimal("20")
    
    # Net debt warning
    if net_cash < Decimal("0"):
        flags.append("NET_DEBT_WARNING")
        score -= Decimal("15")
    
    # Commercial patterns (revenue-based)
    if ttm_revenue_usd is not None:
        if ttm_revenue_usd >= Decimal("500000000"):
            flags.append("COMMERCIAL_CASH_COW")
            score += Decimal("15")
        elif ttm_revenue_usd >= Decimal("100000000"):
            flags.append("COMMERCIAL_RAMP")
            score += Decimal("10")
        elif ttm_revenue_usd > Decimal("0"):
            flags.append("COMMERCIAL_LAUNCH")
            score += Decimal("5")
        else:
            flags.append("PRE_REVENUE")
    
    # Normalize to 0-100 range
    score = max(Decimal("0"), min(Decimal("100"), score))
    
    return _quantize(score), flags


# Example usage / self-test
if __name__ == "__main__":
    # Test with sample data
    sample_financial = [
        {"ticker": "ABBV", "cash_usd": 12500000000, "debt_usd": 55000000000, "ttm_revenue_usd": 58000000000},
        {"ticker": "VRTX", "cash_usd": 11000000000, "debt_usd": 500000000, "ttm_revenue_usd": 9000000000},
        {"ticker": "SRPT", "cash_usd": 2500000000, "debt_usd": 1200000000, "ttm_revenue_usd": 1500000000},
        {"ticker": "ALNY", "cash_usd": 2100000000, "debt_usd": 1500000000, "ttm_revenue_usd": 2500000000},
        {"ticker": "BEAM", "cash_usd": 1100000000, "debt_usd": 0},  # Pre-revenue
        {"ticker": "RARE", "cash_usd": 450000000},  # Missing debt/revenue
    ]
    
    active = {"ABBV", "VRTX", "SRPT", "ALNY", "BEAM", "RARE"}
    
    result = compute_module_2_financial(
        financial_records=sample_financial,
        active_tickers=active,
        as_of_date="2025-01-05",
    )
    
    import json
    print(json.dumps(result, indent=2))
