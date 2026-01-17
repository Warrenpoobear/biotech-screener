#!/usr/bin/env python3
"""
module_2_financial.py - Financial Health Scoring with Burn Acceleration & Catalyst Timing

ENHANCEMENTS (v2.0):
- Burn acceleration detection (4Q trailing vs recent)
- Catalyst timing integration (coverage ratio when available)
- Recent financing dampener
- Enhanced diagnostic output

Scores tickers on financial health using:
1. Cash runway (50% weight)
2. Dilution risk (30% weight)  
3. Liquidity (20% weight)

Usage:
    from module_2_financial import run_module_2
    results = run_module_2(universe, financial_data, market_data, catalyst_summaries)
"""

import json
from pathlib import Path
import math
from typing import Dict, List, Optional
from datetime import datetime, timedelta

# NOTE (SEC 10-Q gotcha):
# Many 10-Qs report CFO as year-to-date (YTD). If you pass YTD CFO fields plus
# fiscal_period, we convert to true quarterly CFO deterministically below.
# This also supports non-calendar fiscal years because it keys off fiscal_period.

# Epsilon for numerical stability
EPS = 1e-6

# ============================================================================
# NEW: Burn Acceleration & Catalyst Timing Functions
# ============================================================================

def quarterly_cfo_from_ytd(
    fiscal_period: str,
    cfo_ytd_current: float,
    cfo_ytd_prev: Optional[float] = None,
    cfo_fy_annual: Optional[float] = None,
    cfo_ytd_q3: Optional[float] = None,
) -> Optional[float]:
    """
    Convert YTD CFO to true quarterly CFO using fiscal_period (Q1/Q2/Q3/FY).
    - Q1: quarterly == YTD(Q1)
    - Q2: quarterly == YTD(Q2) - YTD(Q1)
    - Q3: quarterly == YTD(Q3) - YTD(Q2)
    - Q4: quarterly == FY(annual) - YTD(Q3)  (requires cfo_fy_annual and cfo_ytd_q3)
    Returns None if inputs are insufficient.
    """
    fp = (fiscal_period or "").upper()
    if fp == "Q1":
        return cfo_ytd_current
    if fp in ("Q2", "Q3"):
        if cfo_ytd_prev is None:
            return None
        return cfo_ytd_current - cfo_ytd_prev
    if fp in ("FY", "Q4"):
        if cfo_fy_annual is None or cfo_ytd_q3 is None:
            return None
        return cfo_fy_annual - cfo_ytd_q3
    return None

def calculate_burn_acceleration(
    cfo_recent_q: float,
    cfo_last_4q: List[float]
) -> tuple:
    """
    Calculate burn acceleration ratio.
    
    Args:
        cfo_recent_q: Most recent quarterly CFO (negative = burn)
        cfo_last_4q: List of last 4 quarterly CFO values
    
    Returns:
        (burn_recent_m, burn_4q_avg_m, burn_acceleration)
    """
    # Convert to monthly burn (positive values)
    burn_recent_m = max(0, -cfo_recent_q) / 3.0
    
    # Calculate 4Q trailing average
    burn_4q_total = sum(max(0, -cfo) for cfo in cfo_last_4q)
    burn_4q_avg_m = burn_4q_total / 12.0  # 4 quarters = 12 months
    
    # Compute acceleration with floor
    if burn_4q_avg_m < EPS:
        burn_acceleration = 1.0  # No meaningful baseline
    else:
        burn_acceleration = burn_recent_m / max(EPS, burn_4q_avg_m)
        burn_acceleration = min(3.0, burn_acceleration)  # Cap at 3x
    
    return burn_recent_m, burn_4q_avg_m, burn_acceleration


def calculate_catalyst_coverage(
    runway_months: float,
    next_catalyst_date: Optional[str],
    as_of_date: str
) -> tuple:
    """
    Calculate runway vs catalyst timing coverage.
    
    Args:
        runway_months: Months of cash runway
        next_catalyst_date: ISO date string of next major catalyst
        as_of_date: Current analysis date
    
    Returns:
        (ttc_months, coverage, catalyst_timing_flag)
    """
    if not next_catalyst_date or runway_months is None:
        return None, None, "UNKNOWN"
    
    try:
        # Parse dates
        catalyst_dt = datetime.fromisoformat(next_catalyst_date.replace('Z', ''))
        as_of_dt = datetime.fromisoformat(as_of_date) if isinstance(as_of_date, str) else as_of_date
        
        # Calculate months to catalyst
        days_to_catalyst = (catalyst_dt.date() - as_of_dt.date()).days
        ttc_months = days_to_catalyst / 30.0
        
        # Classify timing
        if ttc_months < 0:
            return None, None, "UNKNOWN"  # Past date
        elif ttc_months > 24:
            return ttc_months, None, "FAR"  # Too far out
        
        # Calculate coverage
        coverage = runway_months / max(1.0, ttc_months)
        
        return ttc_months, coverage, "KNOWN"
        
    except (ValueError, TypeError, AttributeError):
        return None, None, "UNKNOWN"


def apply_dilution_penalty_enhanced(
    base_dilution_score: float,
    runway_months: Optional[float],
    burn_acceleration: float,
    coverage: Optional[float],
    days_since_raise: Optional[int]
) -> float:
    """
    Apply enhanced dilution penalty with burn acceleration and raise dampener.
    
    Args:
        base_dilution_score: Score from cash/mcap ratio
        runway_months: Months of cash runway
        burn_acceleration: Burn acceleration ratio
        coverage: Coverage ratio (runway / ttc) if available
        days_since_raise: Days since most recent equity raise
    
    Returns:
        Adjusted dilution score
    """
    score = base_dilution_score
    
    # Base runway penalty (existing logic)
    if runway_months is not None and runway_months < 12:
        penalty_factor = 0.5 + (runway_months / 24.0)
        score *= min(1.0, penalty_factor)
    
    # Burn acceleration amplifier
    if burn_acceleration >= 1.3:
        if burn_acceleration >= 1.6:
            score *= 0.57  # 1.75x penalty = divide by 1.75
        else:
            score *= 0.67  # 1.5x penalty = divide by 1.5
    
    # Coverage penalty (if catalyst timing known)
    if coverage is not None and coverage < 1.0:
        # They need money before catalyst
        score *= 0.7
    
    # Recent raise dampener (with floor)
    if days_since_raise is not None and days_since_raise < 90:
        score *= 1.5  # Dampen penalty
        # But maintain floor if runway still critical
        if runway_months is not None and runway_months < 9:
            min_floor = 25.0
            score = max(score, min_floor)
    
    return score


# ============================================================================
# ORIGINAL FUNCTIONS (Unchanged)
# ============================================================================

def calculate_cash_runway(financial_data: Dict, market_data: Dict) -> tuple:
    """
    Calculate months of cash runway.
    
    Returns:
        (runway_months, monthly_burn, score)
    """
    cash = financial_data.get('Cash', 0) or 0
    net_income = financial_data.get('NetIncome', 0) or 0
    rd_expense = financial_data.get('R&D', 0) or 0
    
    # CASE 1: Profitable companies
    if net_income and net_income > 0:
        return 999.0, 0, 95.0
    
    # CASE 2: Burning cash
    if net_income and net_income < 0:
        quarterly_burn = abs(net_income)
        monthly_burn = quarterly_burn / 3.0
    
    # CASE 3: Estimate from R&D
    elif rd_expense and rd_expense > 0:
        quarterly_burn = rd_expense * 1.5
        monthly_burn = quarterly_burn / 3.0
    
    else:
        return None, None, 50.0
    
    # Calculate runway
    if monthly_burn > 0:
        runway_months = cash / monthly_burn
    else:
        runway_months = 999.0
    
    # Score based on runway
    if runway_months >= 24:
        score = 100.0
    elif runway_months >= 18:
        score = 90.0
    elif runway_months >= 12:
        score = 70.0
    elif runway_months >= 6:
        score = 40.0
    else:
        score = 10.0
    
    return runway_months, monthly_burn, score


def calculate_dilution_risk(financial_data: Dict, market_data: Dict, runway_months: Optional[float]) -> tuple:
    """Score dilution risk with CONTINUOUS scoring"""
    cash = financial_data.get('Cash', 0) or 0
    market_cap = market_data.get('market_cap', 0) or 0
    
    if not market_cap or market_cap <= 0:
        return None, 50.0
    
    cash_to_mcap = cash / market_cap
    
    # Sigmoid scoring
    if cash_to_mcap >= 0.40:
        dilution_score = 100.0
    elif cash_to_mcap <= 0:
        dilution_score = 0.0
    else:
        k = 15.0
        midpoint = 0.15
        dilution_score = 100.0 / (1.0 + math.exp(-k * (cash_to_mcap - midpoint)))
    
    # NOTE: Runway penalty now handled in apply_dilution_penalty_enhanced()
    return cash_to_mcap, dilution_score


def score_liquidity(market_data: Dict) -> float:
    """Score based on market cap and trading volume"""
    market_cap = market_data.get('market_cap', 0) or 0
    avg_volume = market_data.get('avg_volume', 0) or 0
    price = market_data.get('price', 0) or 0
    
    if not all([market_cap, avg_volume, price]):
        return 50.0
    
    dollar_volume = avg_volume * price
    
    # Market cap tiers
    if market_cap > 10e9:
        mcap_score = 100.0
    elif market_cap > 2e9:
        mcap_score = 80.0
    elif market_cap > 500e6:
        mcap_score = 60.0
    elif market_cap > 200e6:
        mcap_score = 40.0
    else:
        mcap_score = 20.0
    
    # Volume tiers
    if dollar_volume > 10e6:
        volume_score = 100.0
    elif dollar_volume > 5e6:
        volume_score = 80.0
    elif dollar_volume > 1e6:
        volume_score = 60.0
    elif dollar_volume > 500e3:
        volume_score = 40.0
    else:
        volume_score = 20.0
    
    return mcap_score * 0.6 + volume_score * 0.4


# ============================================================================
# ENHANCED SCORING FUNCTION
# ============================================================================

def score_financial_health_enhanced(
    ticker: str,
    financial_data: Dict,
    market_data: Dict,
    catalyst_summary: Optional[Dict] = None,
    as_of_date: Optional[str] = None
) -> Dict:
    """
    Enhanced financial health scoring with burn acceleration and catalyst timing.
    
    Args:
        ticker: Ticker symbol
        financial_data: Financial metrics
        market_data: Market data
        catalyst_summary: Optional catalyst timing from Module 3
        as_of_date: Optional analysis date for catalyst timing
    
    Returns:
        Enhanced score dict with diagnostic fields
    """
    # Component 1: Cash Runway (50%)
    runway_months, burn_rate, runway_score = calculate_cash_runway(financial_data, market_data)
    
    # Component 2: Dilution Risk (30%) - Base calculation
    cash_to_mcap, base_dilution_score = calculate_dilution_risk(financial_data, market_data, runway_months)
    
    # NEW: Burn acceleration (if CFO history available)
    burn_recent_m = None
    burn_4q_avg_m = None
    burn_acceleration = 1.0
    cfo_quality_flag = "OK"
    
    # Prefer explicit quarterly CFO history if provided
    if 'CFO_recent_q' in financial_data and 'CFO_last_4q' in financial_data:
        try:
            cfo_recent = financial_data['CFO_recent_q']
            cfo_history = financial_data['CFO_last_4q']
            burn_recent_m, burn_4q_avg_m, burn_acceleration = calculate_burn_acceleration(
                cfo_recent, cfo_history
            )
        except (TypeError, ValueError):
            cfo_quality_flag = "NOISY"
    # Fallback: derive quarterly CFO from YTD CFO (10-Q style) if available
    elif 'CFO_ytd_current' in financial_data and 'fiscal_period' in financial_data:
        try:
            cfo_recent = quarterly_cfo_from_ytd(
                fiscal_period=str(financial_data.get('fiscal_period')),
                cfo_ytd_current=float(financial_data.get('CFO_ytd_current')),
                cfo_ytd_prev=financial_data.get('CFO_ytd_prev'),
                cfo_fy_annual=financial_data.get('CFO_fy_annual'),
                cfo_ytd_q3=financial_data.get('CFO_ytd_q3'),
            )
            if cfo_recent is None:
                cfo_quality_flag = "MISSING"
            else:
                # Use recent quarter only; burn acceleration stays neutral unless full history is supplied
                burn_recent_m, burn_4q_avg_m, burn_acceleration = calculate_burn_acceleration(
                    cfo_recent, [cfo_recent, cfo_recent, cfo_recent, cfo_recent]
                )
        except (TypeError, ValueError):
            cfo_quality_flag = "NOISY"
    else:
        cfo_quality_flag = "MISSING"
    
    # NEW: Catalyst timing coverage (if available)
    ttc_months = None
    coverage = None
    catalyst_timing_flag = "UNKNOWN"
    
    if catalyst_summary and as_of_date:
        next_catalyst_date = catalyst_summary.get('next_major_catalyst_date')
        if next_catalyst_date:
            ttc_months, coverage, catalyst_timing_flag = calculate_catalyst_coverage(
                runway_months, next_catalyst_date, as_of_date
            )
    
    # NEW: Recent financing check (if available)
    days_since_raise = financial_data.get('days_since_last_raise')
    
    # Apply enhanced dilution penalty
    if base_dilution_score is not None:
        dilution_score = apply_dilution_penalty_enhanced(
            base_dilution_score,
            runway_months,
            burn_acceleration,
            coverage,
            days_since_raise
        )
    else:
        dilution_score = 50.0
    
    # Component 3: Liquidity (20%)
    liquidity_score = score_liquidity(market_data)
    
    # Composite score
    if all([runway_score is not None, dilution_score is not None, liquidity_score is not None]):
        composite = runway_score * 0.50 + dilution_score * 0.30 + liquidity_score * 0.20
        has_data = True
    else:
        composite = 50.0
        has_data = False
    
    # Enhanced output with diagnostics
    return {
        # Core outputs
        "ticker": ticker,
        "financial_normalized": float(composite),
        "runway_months": float(runway_months) if runway_months else None,
        "runway_score": float(runway_score) if runway_score else None,
        "dilution_score": float(dilution_score) if dilution_score else None,
        "liquidity_score": float(liquidity_score) if liquidity_score else None,
        "cash_to_mcap": float(cash_to_mcap) if cash_to_mcap else None,
        "monthly_burn": float(burn_rate) if burn_rate else None,
        "has_financial_data": has_data,
        
        # NEW: Burn acceleration fields
        "burn_recent_m": float(burn_recent_m) if burn_recent_m else None,
        "burn_4q_avg_m": float(burn_4q_avg_m) if burn_4q_avg_m else None,
        "burn_acceleration": float(burn_acceleration),
        "cfo_quality_flag": cfo_quality_flag,
        
        # NEW: Catalyst timing fields
        "ttc_months": float(ttc_months) if ttc_months else None,
        "coverage": float(coverage) if coverage else None,
        "catalyst_timing_flag": catalyst_timing_flag,
        
        # NEW: Financing context
        "days_since_raise": int(days_since_raise) if days_since_raise else None,
    }


# ============================================================================
# MAIN ENTRY POINTS
# ============================================================================

def run_module_2(
    universe: List[str],
    financial_data: List[Dict],
    market_data: List[Dict],
    catalyst_summaries: Optional[Dict] = None,
    as_of_date: Optional[str] = None
) -> List[Dict]:
    """
    Main entry point for Module 2 financial health scoring.
    
    Args:
        universe: List of tickers to score
        financial_data: List of dicts from financial_data.json
        market_data: List of dicts from market_data.json
        catalyst_summaries: Optional dict of ticker -> catalyst summary (from Module 3)
        as_of_date: Optional analysis date for catalyst timing
    
    Returns:
        List of dicts with financial health scores
    """
    print(f"\n=== DEBUG run_module_2 (ENHANCED) ===")
    print(f"Universe: {len(universe)} tickers")
    print(f"Catalyst summaries available: {bool(catalyst_summaries)}")
    print("="*50)
    
    # Create lookup dicts
    fin_lookup = {f['ticker']: f for f in financial_data if 'ticker' in f}
    mkt_lookup = {m['ticker']: m for m in market_data if 'ticker' in m}
    
    results = []
    for ticker in universe:
        fin_data = fin_lookup.get(ticker, {})
        mkt_data = mkt_lookup.get(ticker, {})
        
        # Get catalyst summary if available
        catalyst_summary = None
        if catalyst_summaries:
            catalyst_summary = catalyst_summaries.get(ticker)
        
        score_result = score_financial_health_enhanced(
            ticker, fin_data, mkt_data, catalyst_summary, as_of_date
        )
        results.append(score_result)
    
    return results


# Backward compatibility wrapper
def compute_module_2_financial(*args, **kwargs):
    """Ultra-flexible wrapper for run_screen.py integration"""
    # Extract parameters
    universe = kwargs.get('universe', kwargs.get('active_tickers', kwargs.get('active_universe', [])))
    financial_data = kwargs.get('financial_records', kwargs.get('financial_data', []))
    market_data = kwargs.get('market_records', kwargs.get('market_data', []))
    catalyst_summaries = kwargs.get('catalyst_summaries')
    as_of_date = kwargs.get('as_of_date')
    
    # Convert set to list if needed
    if isinstance(universe, set):
        universe = list(universe)
    
    # Extract market data from raw_universe if needed
    if not market_data and 'raw_universe' in kwargs:
        raw_universe = kwargs['raw_universe']
        market_data = []
        for record in raw_universe:
            if 'market_data' in record and record.get('ticker'):
                mkt = record['market_data'].copy()
                mkt['ticker'] = record['ticker']
                market_data.append(mkt)
    
    # Map market data field names
    mapped_market = []
    for rec in market_data:
        mapped_market.append({
            'ticker': rec.get('ticker'),
            'market_cap': rec.get('market_cap'),
            'price': rec.get('price'),
            'avg_volume': rec.get('volume_avg_30d'),
        })
    
    # Run enhanced module
    result = run_module_2(universe, financial_data, mapped_market, catalyst_summaries, as_of_date)
    
    # Wrap in expected format
    return {
        'scores': result,
        'diagnostic_counts': {
            'scored': len(result),
            'missing': 0
        }
    }


# ============================================================================
# TEST/DEMO
# ============================================================================

def main():
    """Test enhanced Module 2 with burn acceleration"""
    
    universe = ['CVAC', 'RYTM']
    
    # Enhanced financial data with CFO history
    financial_data = [
        {
            'ticker': 'CVAC',
            'Cash': 500e6,
            'NetIncome': -100e6,
            'R&D': 80e6,
            'CFO_recent_q': -110e6,  # Accelerating burn
            'CFO_last_4q': [-90e6, -95e6, -100e6, -110e6],
            'days_since_last_raise': None
        },
        {
            'ticker': 'RYTM',
            'Cash': 200e6,
            'NetIncome': -50e6,
            'R&D': 40e6,
            'CFO_recent_q': -45e6,  # Stable burn
            'CFO_last_4q': [-48e6, -50e6, -47e6, -45e6],
            'days_since_last_raise': 60  # Recent raise
        }
    ]
    
    market_data = [
        {'ticker': 'CVAC', 'market_cap': 2e9, 'avg_volume': 500000, 'price': 20.0},
        {'ticker': 'RYTM', 'market_cap': 800e6, 'avg_volume': 200000, 'price': 15.0}
    ]
    
    # Mock catalyst summaries
    catalyst_summaries = {
        'CVAC': {'next_major_catalyst_date': '2026-06-15'},
        'RYTM': {'next_major_catalyst_date': '2026-12-01'}
    }
    
    results = run_module_2(universe, financial_data, market_data, catalyst_summaries, '2026-01-08')
    
    print("="*80)
    print("MODULE 2 ENHANCED: TEST RUN")
    print("="*80)
    
    for r in results:
        print(f"\n{r['ticker']}:")
        print(f"  Financial Score: {r['financial_normalized']:.2f}")
        print(f"  Runway: {r['runway_months']:.1f} months")
        print(f"  Burn Acceleration: {r['burn_acceleration']:.2f}x")
        if r['coverage']:
            print(f"  Coverage: {r['coverage']:.2f} ({r['catalyst_timing_flag']})")
        if r['days_since_raise']:
            print(f"  Days Since Raise: {r['days_since_raise']}")
    
    print("\n" + "="*80)
    print("✅ Enhanced Module 2 test complete!")
    print("="*80)


if __name__ == "__main__":
    main()
