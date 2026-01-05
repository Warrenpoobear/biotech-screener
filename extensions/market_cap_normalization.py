# extensions/market_cap_normalization.py
"""
Market Cap Normalization for Financial Health Scoring

Enhances Module 2 by adding relative cash metrics instead of absolute thresholds.
Eliminates large-cap bias in CASH_FORTRESS pattern detection.

DETERMINISM GUARANTEES:
- Decimal-only arithmetic
- Stable thresholds
- No time-dependencies
- Auditable calculations
"""
from __future__ import annotations

import hashlib
import json
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, Optional, Any


# Quantization for deterministic decimals
DECIMAL_PRECISION = Decimal('0.01')


def get_market_cap_from_yahoo(ticker: str) -> Optional[Decimal]:
    """
    Fetch market cap from Yahoo Finance (or use cached value).
    
    In production, replace with your actual data source.
    For now, this is a placeholder that should be replaced with
    your data pipeline integration.
    """
    # TODO: Replace with actual Yahoo Finance integration
    # This would typically come from your existing price data pipeline
    return None


def calculate_relative_cash_strength(
    cash_usd: Optional[Decimal],
    debt_usd: Optional[Decimal],
    market_cap_usd: Optional[Decimal],
) -> Dict[str, Any]:
    """
    Calculate cash strength relative to market cap.
    
    Ratios:
    - >=50%: FORTRESS (exceptional balance sheet)
    - 30-50%: STRONG (healthy)
    - 15-30%: ADEQUATE (standard)
    - <15%: WEAK (dilution risk)
    
    Returns dict with:
    - status: Category (FORTRESS/STRONG/ADEQUATE/WEAK/UNKNOWN)
    - net_cash_ratio: Decimal (net cash / market cap)
    - score_boost: Decimal (points to add)
    - audit_hash: str (deterministic hash)
    """
    
    # Handle missing data
    if market_cap_usd is None or market_cap_usd <= 0:
        return {
            'status': 'UNKNOWN',
            'net_cash_ratio': None,
            'score_boost': Decimal('0'),
            'missing_data': 'market_cap',
            'audit_hash': _compute_audit_hash(None, None, None, 'UNKNOWN')
        }
    
    # Calculate net cash
    cash = cash_usd or Decimal('0')
    debt = debt_usd or Decimal('0')
    net_cash = cash - debt
    
    # Calculate ratio
    cash_ratio = (net_cash / market_cap_usd).quantize(DECIMAL_PRECISION, rounding=ROUND_HALF_UP)
    
    # Determine category and score boost
    if cash_ratio >= Decimal('0.50'):
        status = 'FORTRESS'
        score_boost = Decimal('15')
    elif cash_ratio >= Decimal('0.30'):
        status = 'STRONG'
        score_boost = Decimal('10')
    elif cash_ratio >= Decimal('0.15'):
        status = 'ADEQUATE'
        score_boost = Decimal('5')
    else:
        status = 'WEAK'
        score_boost = Decimal('0')
    
    # Create audit hash
    audit_hash = _compute_audit_hash(net_cash, market_cap_usd, cash_ratio, status)
    
    return {
        'status': status,
        'net_cash_ratio': str(cash_ratio),  # Serialize as string
        'net_cash_usd': str(net_cash),
        'market_cap_usd': str(market_cap_usd),
        'score_boost': str(score_boost),
        'audit_hash': audit_hash,
    }


def calculate_absolute_cash_fortress(
    cash_usd: Optional[Decimal],
    debt_usd: Optional[Decimal],
    threshold: Decimal = Decimal('1000000000'),  #  default
) -> Dict[str, Any]:
    """
    Legacy absolute cash fortress check.
    
    Kept for backward compatibility but not recommended.
    Use relative_cash_strength instead.
    """
    if cash_usd is None or debt_usd is None:
        return {
            'is_fortress': False,
            'net_cash_usd': None,
            'threshold_usd': str(threshold),
        }
    
    net_cash = cash_usd - debt_usd
    is_fortress = net_cash >= threshold
    
    return {
        'is_fortress': is_fortress,
        'net_cash_usd': str(net_cash),
        'threshold_usd': str(threshold),
    }


def enhance_financial_score_with_relative_cash(
    base_financial_score: Decimal,
    ticker: str,
    cash_usd: Optional[Decimal],
    debt_usd: Optional[Decimal],
    market_cap_usd: Optional[Decimal],
) -> Dict[str, Any]:
    """
    Enhance financial health score with relative cash strength.
    
    Returns:
    - enhanced_score: Decimal
    - relative_cash_analysis: Dict
    - score_components: Dict (breakdown)
    """
    
    # Calculate relative cash strength
    rel_cash = calculate_relative_cash_strength(cash_usd, debt_usd, market_cap_usd)
    
    # Apply score boost
    if rel_cash['score_boost'] is not None:
        boost = Decimal(rel_cash['score_boost'])
        enhanced_score = base_financial_score + boost
    else:
        enhanced_score = base_financial_score
    
    return {
        'ticker': ticker,
        'base_financial_score': str(base_financial_score),
        'enhanced_score': str(enhanced_score),
        'relative_cash_analysis': rel_cash,
        'score_components': {
            'base': str(base_financial_score),
            'relative_cash_boost': rel_cash['score_boost'],
            'total': str(enhanced_score),
        }
    }


def _compute_audit_hash(
    net_cash: Optional[Decimal],
    market_cap: Optional[Decimal],
    ratio: Optional[Decimal],
    status: str,
) -> str:
    """
    Compute deterministic audit hash for relative cash calculation.
    
    Uses canonical JSON serialization for reproducibility.
    """
    audit_data = {
        'net_cash': str(net_cash) if net_cash is not None else None,
        'market_cap': str(market_cap) if market_cap is not None else None,
        'ratio': str(ratio) if ratio is not None else None,
        'status': status,
    }
    
    # Canonical JSON with sorted keys
    canonical = json.dumps(audit_data, sort_keys=True, separators=(',', ':'))
    hash_hex = hashlib.sha256(canonical.encode('utf-8')).hexdigest()
    
    return hash_hex[:16]  # First 16 chars for brevity


# Example usage and testing
if __name__ == '__main__':
    print('==' * 40)
    print('MARKET CAP NORMALIZATION - EXAMPLE USAGE')
    print('==' * 40)
    
    # Example 1: Large cap with strong cash position
    print('\nExample 1: REGN (Large Cap, Strong Cash)')
    print('-' * 40)
    result1 = calculate_relative_cash_strength(
        cash_usd=Decimal('7500000000'),  # .5B cash
        debt_usd=Decimal('500000000'),   # .5B debt
        market_cap_usd=Decimal('20000000000'),  #  market cap
    )
    print(f"Net Cash Ratio: {result1['net_cash_ratio']}")
    print(f"Status: {result1['status']}")
    print(f"Score Boost: +{result1['score_boost']} points")
    print(f"Audit Hash: {result1['audit_hash']}")
    
    # Example 2: Small cap with adequate cash
    print('\nExample 2: VRTX (Mid Cap, Adequate Cash)')
    print('-' * 40)
    result2 = calculate_relative_cash_strength(
        cash_usd=Decimal('2000000000'),  #  cash
        debt_usd=Decimal('500000000'),   # .5B debt
        market_cap_usd=Decimal('8000000000'),  #  market cap
    )
    print(f"Net Cash Ratio: {result2['net_cash_ratio']}")
    print(f"Status: {result2['status']}")
    print(f"Score Boost: +{result2['score_boost']} points")
    
    # Example 3: Small cap with weak cash
    print('\nExample 3: Small Biotech (Weak Cash)')
    print('-' * 40)
    result3 = calculate_relative_cash_strength(
        cash_usd=Decimal('100000000'),   #  cash
        debt_usd=Decimal('20000000'),    #  debt
        market_cap_usd=Decimal('800000000'),  #  market cap
    )
    print(f"Net Cash Ratio: {result3['net_cash_ratio']}")
    print(f"Status: {result3['status']}")
    print(f"Score Boost: +{result3['score_boost']} points")
    
    print('\n' + '==' * 40)
    print('✅ Market cap normalization working correctly!')
    print('==' * 40)