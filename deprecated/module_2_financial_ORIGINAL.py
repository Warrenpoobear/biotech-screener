#!/usr/bin/env python3
"""
module_2_financial.py - Financial Health Scoring

Scores tickers on financial health using:
1. Cash runway (50% weight)
2. Dilution risk (30% weight)
3. Liquidity (20% weight)

Usage:
    from module_2_financial import run_module_2
    results = run_module_2(universe, financial_data, market_data)
"""

import json
from pathlib import Path
from typing import Dict, List, Optional


def calculate_cash_runway(financial_data: Dict, market_data: Dict) -> tuple:
    """
    Calculate months of cash runway.
    
    Returns:
        (runway_months, monthly_burn, score)
    """
    
    cash = financial_data.get('Cash', 0) or 0
    net_income = financial_data.get('NetIncome', 0) or 0
    rd_expense = financial_data.get('R&D', 0) or 0
    
    # Method 1: Use net income if available and negative
    if net_income and net_income < 0:
        quarterly_burn = abs(net_income)
        monthly_burn = quarterly_burn / 3.0
    
    # Method 2: Estimate from R&D (pre-revenue companies)
    elif rd_expense and rd_expense > 0:
        # Assume total opex = R&D × 1.5 (add G&A overhead)
        quarterly_burn = rd_expense * 1.5
        monthly_burn = quarterly_burn / 3.0
    
    else:
        # No burn data - return neutral
        return None, None, 50.0
    
    # Calculate runway
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
    """Score dilution risk based on cash as % of market cap"""
    
    cash = financial_data.get('Cash', 0) or 0
    market_cap = market_data.get('market_cap', 0) or 0
    
    if not market_cap or market_cap <= 0:
        return None, 50.0
    
    cash_to_mcap = cash / market_cap
    
    # Base scoring
    if cash_to_mcap >= 0.30:
        dilution_score = 100.0  # Strong
    elif cash_to_mcap >= 0.20:
        dilution_score = 80.0   # Adequate
    elif cash_to_mcap >= 0.10:
        dilution_score = 60.0   # Moderate
    elif cash_to_mcap >= 0.05:
        dilution_score = 30.0   # High risk
    else:
        dilution_score = 10.0   # Critical
    
    # Penalize near-term financing needs
    if runway_months is not None and runway_months < 12:
        dilution_score *= 0.7
    
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
    
    return {
        "ticker": ticker,
        "financial_normalized": float(composite),
        "runway_months": float(runway_months) if runway_months else None,
        "runway_score": float(runway_score) if runway_score else None,
        "dilution_score": float(dilution_score) if dilution_score else None,
        "liquidity_score": float(liquidity_score) if liquidity_score else None,
        "cash_to_mcap": float(cash_to_mcap) if cash_to_mcap else None,
        "monthly_burn": float(burn_rate) if burn_rate else None,
        "has_financial_data": has_data
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
    
    # Create lookup dicts
    fin_lookup = {f['ticker']: f for f in financial_data if 'ticker' in f}
    mkt_lookup = {m['ticker']: m for m in market_data if 'ticker' in m}
    
    results = []
    for ticker in universe:
        fin_data = fin_lookup.get(ticker, {})
        mkt_data = mkt_lookup.get(ticker, {})
        score_result = score_financial_health(ticker, fin_data, mkt_data)
        results.append(score_result)
    
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
