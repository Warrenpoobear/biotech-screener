#!/usr/bin/env python3
"""
module_2_financial.py - Financial Health Scoring

Scores tickers on financial health using:
1. Cash runway (50% weight)
2. Dilution risk (30% weight)
3. Liquidity (20% weight)

Severity levels based on financial health:
- SEV3: Runway < 6 months (critical)
- SEV2: Runway 6-12 months (warning)
- SEV1: Runway 12-18 months (caution)
- NONE: Runway >= 18 months (healthy)

Usage:
    from module_2_financial import run_module_2
    results = run_module_2(universe, financial_data, market_data)
"""

import json
import logging
from pathlib import Path
import math
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def calculate_cash_runway(financial_data: Dict, market_data: Dict) -> tuple:
    """
    Calculate months of cash runway.
    
    Returns:
        (runway_months, monthly_burn, score)
    """
    
    cash = financial_data.get('Cash', 0) or 0
    net_income = financial_data.get('NetIncome', 0) or 0
    rd_expense = financial_data.get('R&D', 0) or 0
    
    # CASE 1: Profitable companies (positive net income)
    if net_income and net_income > 0:
        # Cash positive - assign high score based on profitability
        return 999.0, 0, 95.0  # Strong position
    
    # CASE 2: Burning cash (negative net income)
    if net_income and net_income < 0:
        quarterly_burn = abs(net_income)
        monthly_burn = quarterly_burn / 3.0
    
    # CASE 3: Estimate from R&D (pre-revenue companies)
    elif rd_expense and rd_expense > 0:
        # Assume total opex = R&D × 1.5 (add G&A overhead)
        quarterly_burn = rd_expense * 1.5
        monthly_burn = quarterly_burn / 3.0
    
    else:
        # No burn data - return neutral
        return None, None, 50.0
    
    # Calculate runway for burning companies
    if monthly_burn > 0:
        runway_months = cash / monthly_burn
    else:
        # Cash positive
        runway_months = 999.0
    
    # Score based on runway
    if runway_months >= 24:
        score = 100.0  # 2+ years
    elif runway_months >= 18:
        score = 90.0   # 18-24 months
    elif runway_months >= 12:
        score = 70.0   # 12-18 months
    elif runway_months >= 6:
        score = 40.0   # 6-12 months
    else:
        score = 10.0   # < 6 months
    
    return runway_months, monthly_burn, score


def calculate_dilution_risk(financial_data: Dict, market_data: Dict, runway_months: Optional[float]) -> tuple:
    """Score dilution risk with CONTINUOUS scoring"""
    
    cash = financial_data.get('Cash', 0) or 0
    market_cap = market_data.get('market_cap', 0) or 0
    
    if not market_cap or market_cap <= 0:
        return None, 50.0
    
    cash_to_mcap = cash / market_cap
    
    # CONTINUOUS scoring: sigmoid curve for cash/mcap ratio
    # 0% ? 0, 5% ? 20, 10% ? 40, 20% ? 70, 30%+ ? 95
    if cash_to_mcap >= 0.40:
        dilution_score = 100.0
    elif cash_to_mcap <= 0:
        dilution_score = 0.0
    else:
        # Sigmoid: maps 0-40% cash/mcap to 0-100 score
        k = 15.0  # Steepness
        midpoint = 0.15  # Inflection at 15%
        dilution_score = 100.0 / (1.0 + math.exp(-k * (cash_to_mcap - midpoint)))
    
    # Continuous penalty for near-term financing needs
    if runway_months is not None and runway_months < 12:
        # Smooth penalty: 0mo?0.5x, 12mo?1.0x
        penalty_factor = 0.5 + (runway_months / 24.0)
        dilution_score *= min(1.0, penalty_factor)
    
    return cash_to_mcap, dilution_score


def score_liquidity(market_data: Dict) -> float:
    """Score based on market cap and trading volume"""

    market_cap = market_data.get('market_cap', 0) or 0
    avg_volume = market_data.get('avg_volume', 0) or 0
    price = market_data.get('price', 0) or 0

    if not all([market_cap, avg_volume, price]):
        return 50.0

    # Dollar volume
    dollar_volume = avg_volume * price

    # Market cap tiers (60% weight)
    if market_cap > 10e9:
        mcap_score = 100.0  # >$10B
    elif market_cap > 2e9:
        mcap_score = 80.0   # $2-10B
    elif market_cap > 500e6:
        mcap_score = 60.0   # $500M-2B
    elif market_cap > 200e6:
        mcap_score = 40.0   # $200M-500M
    else:
        mcap_score = 20.0   # <$200M

    # Volume tiers (40% weight)
    if dollar_volume > 10e6:
        volume_score = 100.0  # $10M+
    elif dollar_volume > 5e6:
        volume_score = 80.0   # $5-10M
    elif dollar_volume > 1e6:
        volume_score = 60.0   # $1-5M
    elif dollar_volume > 500e3:
        volume_score = 40.0   # $500K-1M
    else:
        volume_score = 20.0   # <$500K

    # Composite
    return mcap_score * 0.6 + volume_score * 0.4


def determine_financial_severity(runway_months: Optional[float], cash_to_mcap: Optional[float]) -> str:
    """
    Determine severity level based on financial health metrics.

    Args:
        runway_months: Estimated months of cash runway
        cash_to_mcap: Cash to market cap ratio

    Returns:
        Severity string: "none", "sev1", "sev2", or "sev3"
    """
    # If no runway data, check cash/mcap ratio
    if runway_months is None:
        if cash_to_mcap is not None and cash_to_mcap < 0.05:
            return "sev2"  # Very low cash relative to market cap
        return "none"  # Unknown - no penalty

    # Severity based on runway
    if runway_months < 6:
        return "sev3"  # Critical - less than 6 months
    elif runway_months < 12:
        return "sev2"  # Warning - 6-12 months
    elif runway_months < 18:
        return "sev1"  # Caution - 12-18 months
    else:
        return "none"  # Healthy - 18+ months


def score_financial_health(ticker: str, financial_data: Dict, market_data: Dict) -> Dict:
    """Main scoring function for Module 2"""

    # Component 1: Cash Runway (50%)
    runway_months, burn_rate, runway_score = calculate_cash_runway(financial_data, market_data)

    # Component 2: Dilution Risk (30%)
    cash_to_mcap, dilution_score = calculate_dilution_risk(financial_data, market_data, runway_months)

    # Component 3: Liquidity (20%)
    liquidity_score = score_liquidity(market_data)

    # Composite score
    if all([runway_score is not None, dilution_score is not None, liquidity_score is not None]):
        composite = runway_score * 0.50 + dilution_score * 0.30 + liquidity_score * 0.20
        has_data = True
    else:
        composite = 50.0
        has_data = False

    # Determine severity based on financial health
    severity = determine_financial_severity(runway_months, cash_to_mcap)

    # Build flags list
    flags = []
    if runway_months is not None and runway_months < 12:
        flags.append("low_runway")
    if cash_to_mcap is not None and cash_to_mcap < 0.10:
        flags.append("low_cash_ratio")
    if not has_data:
        flags.append("missing_financial_data")

    return {
        "ticker": ticker,
        "financial_normalized": float(composite),
        "runway_months": float(runway_months) if runway_months else None,
        "runway_score": float(runway_score) if runway_score else None,
        "dilution_score": float(dilution_score) if dilution_score else None,
        "liquidity_score": float(liquidity_score) if liquidity_score else None,
        "cash_to_mcap": float(cash_to_mcap) if cash_to_mcap else None,
        "monthly_burn": float(burn_rate) if burn_rate else None,
        "has_financial_data": has_data,
        "severity": severity,
        "flags": flags,
    }

def run_module_2(universe: List[str], financial_data: List[Dict], market_data: List[Dict]) -> List[Dict]:
    """
    Main entry point for Module 2 financial health scoring.

    Args:
        universe: List of tickers to score
        financial_data: List of dicts from financial_data.json
        market_data: List of dicts from market_data.json

    Returns:
        List of dicts with financial health scores
    """
    logger.info(f"Module 2: Scoring {len(universe)} tickers")
    logger.debug(f"Financial records: {len(financial_data)}, Market records: {len(market_data)}")

    # Create lookup dicts
    fin_lookup = {f['ticker']: f for f in financial_data if 'ticker' in f}
    mkt_lookup = {m['ticker']: m for m in market_data if 'ticker' in m}

    results = []
    for ticker in universe:
        fin_data = fin_lookup.get(ticker, {})
        mkt_data = mkt_lookup.get(ticker, {})
        score_result = score_financial_health(ticker, fin_data, mkt_data)
        results.append(score_result)

    # Log severity distribution
    severity_counts = {}
    for r in results:
        sev = r.get("severity", "none")
        severity_counts[sev] = severity_counts.get(sev, 0) + 1
    logger.info(f"Module 2 severity distribution: {severity_counts}")

    return results


def main():
    """Test Module 2 on sample data"""
    
    universe = ['CVAC', 'RYTM', 'IMMP']
    
    financial_data = [
        {'ticker': 'CVAC', 'Cash': 500e6, 'NetIncome': -100e6, 'R&D': 80e6},
        {'ticker': 'RYTM', 'Cash': 200e6, 'NetIncome': -50e6, 'R&D': 40e6},
        {'ticker': 'IMMP', 'Cash': 1000e6, 'NetIncome': -150e6, 'R&D': 120e6}
    ]
    
    market_data = [
        {'ticker': 'CVAC', 'market_cap': 2e9, 'avg_volume': 500000, 'price': 20.0},
        {'ticker': 'RYTM', 'market_cap': 800e6, 'avg_volume': 200000, 'price': 15.0},
        {'ticker': 'IMMP', 'market_cap': 5e9, 'avg_volume': 1000000, 'price': 50.0}
    ]
    
    results = run_module_2(universe, financial_data, market_data)
    
    print("="*80)
    print("MODULE 2: FINANCIAL HEALTH SCORING - TEST RUN")
    print("="*80)
    
    for r in results:
        print(f"\n{r['ticker']}:")
        print(f"  Financial Score: {r['financial_normalized']:.2f}")
        if r['runway_months']:
            print(f"  Runway: {r['runway_months']:.1f} months ({r['runway_score']:.0f} pts)")
        if r['cash_to_mcap']:
            print(f"  Dilution: {r['cash_to_mcap']:.1%} cash/mcap ({r['dilution_score']:.0f} pts)")
        if r['liquidity_score']:
            print(f"  Liquidity: {r['liquidity_score']:.0f} pts")
    
    print("\n" + "="*80)
    print("✅ Module 2 test complete!")
    print("="*80)


if __name__ == "__main__":
    main()

# Alias for run_screen.py
def compute_module_2_financial(*args, **kwargs):
    """Ultra-flexible wrapper - NO field mapping needed for SEC data"""
    # Extract parameters
    universe = kwargs.get('universe', kwargs.get('active_tickers', kwargs.get('active_universe', [])))
    financial_data = kwargs.get('financial_records', kwargs.get('financial_data', []))
    market_data = kwargs.get('market_records', kwargs.get('market_data', []))
    
    # Convert set to list if needed
    if isinstance(universe, set):
        universe = list(universe)
    
    # If market_data is empty but we have raw_universe, extract it
    if not market_data and 'raw_universe' in kwargs:
        raw_universe = kwargs['raw_universe']
        market_data = []
        for record in raw_universe:
            if 'market_data' in record and record.get('ticker'):
                mkt = record['market_data'].copy()
                mkt['ticker'] = record['ticker']
                market_data.append(mkt)
    
    # Map market data field names only (volume_avg_30d -> avg_volume)
    mapped_market = []
    for rec in market_data:
        mapped_market.append({
            'ticker': rec.get('ticker'),
            'market_cap': rec.get('market_cap'),
            'price': rec.get('price'),
            'avg_volume': rec.get('volume_avg_30d'),
        })
    
    # Financial data already has correct field names (Cash, NetIncome, R&D)
    # Just pass it through!
    result = run_module_2(universe, financial_data, mapped_market)
    
    # Wrap in expected format
    return {
        'scores': result,
        'diagnostic_counts': {
            'scored': len(result),
            'missing': 0
        }
    }