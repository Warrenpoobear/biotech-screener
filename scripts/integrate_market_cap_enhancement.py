#!/usr/bin/env python3
"""
Integration script to add market cap normalization to screening results.

Usage:
    python scripts/integrate_market_cap_enhancement.py \
        --input results.json \
        --output results_enhanced.json \
        --market-caps market_caps.json
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
from decimal import Decimal
from pathlib import Path
from typing import Dict, Any

from extensions.market_cap_normalization import (
    calculate_relative_cash_strength,
    enhance_financial_score_with_relative_cash,
)


def load_market_caps(filepath: Path) -> Dict[str, Decimal]:
    """
    Load market cap data from JSON file.
    
    Expected format:
    {
        "REGN": "20000000000",
        "VRTX": "8000000000",
        ...
    }
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    return {
        ticker: Decimal(cap_str)
        for ticker, cap_str in data.items()
    }


def enhance_screening_results(
    results: Dict[str, Any],
    market_caps: Dict[str, Decimal],
) -> Dict[str, Any]:
    """
    Add market cap normalization to existing screening results.
    
    Modifies the 'module_5_composite' section to include:
    - relative_cash_analysis for each security
    - enhanced scores with relative cash boost
    """
    
    # Get ranked securities from Module 5
    if 'module_5_composite' not in results:
        print('Warning: No module_5_composite found in results')
        return results
    
    ranked = results['module_5_composite'].get('ranked_securities', [])
    
    enhanced_count = 0
    missing_market_cap = []
    
    for security in ranked:
        ticker = security.get('ticker')
        
        if ticker not in market_caps:
            missing_market_cap.append(ticker)
            continue
        
        # Get financial data
        module_2 = security.get('module_2_financial', {})
        cash_usd = module_2.get('cash_usd')
        debt_usd = module_2.get('debt_usd')
        
        if cash_usd is not None:
            cash_usd = Decimal(str(cash_usd))
        if debt_usd is not None:
            debt_usd = Decimal(str(debt_usd))
        
        market_cap = market_caps[ticker]
        
        # Calculate relative cash strength
        rel_cash = calculate_relative_cash_strength(
            cash_usd=cash_usd,
            debt_usd=debt_usd,
            market_cap_usd=market_cap,
        )
        
        # Add to security data
        security['relative_cash_analysis'] = rel_cash
        
        # Optionally enhance the composite score
        if 'composite_score' in security:
            base_score = Decimal(str(security['composite_score']))
            boost = Decimal(rel_cash['score_boost'])
            
            security['composite_score_original'] = str(base_score)
            security['composite_score'] = str(base_score + boost)
            security['relative_cash_boost'] = str(boost)
        
        enhanced_count += 1
    
    # Add summary
    results['market_cap_enhancement'] = {
        'enhanced_count': enhanced_count,
        'missing_market_cap': missing_market_cap,
        'total_securities': len(ranked),
    }
    
    print(f'\n✅ Enhanced {enhanced_count} securities with market cap normalization')
    if missing_market_cap:
        print(f'⚠️  Missing market cap data for: {missing_market_cap}')
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Add market cap normalization to screening results'
    )
    parser.add_argument(
        '--input',
        type=Path,
        required=True,
        help='Input screening results JSON',
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output enhanced results JSON',
    )
    parser.add_argument(
        '--market-caps',
        type=Path,
        required=True,
        help='Market cap data JSON file',
    )
    parser.add_argument(
        '--no-score-adjustment',
        action='store_true',
        help='Add analysis but do not adjust composite scores',
    )
    
    args = parser.parse_args()
    
    # Load inputs
    print(f'Loading screening results from {args.input}...')
    with open(args.input, 'r') as f:
        results = json.load(f)
    
    print(f'Loading market cap data from {args.market_caps}...')
    market_caps = load_market_caps(args.market_caps)
    print(f'  Loaded market caps for {len(market_caps)} tickers')
    
    # Enhance results
    print('\nEnhancing results with market cap normalization...')
    enhanced_results = enhance_screening_results(results, market_caps)
    
    # Save output
    print(f'\nWriting enhanced results to {args.output}...')
    with open(args.output, 'w') as f:
        json.dump(enhanced_results, f, indent=2)
    
    print(f'\n✅ Enhancement complete!')
    print(f'   Input:  {args.input}')
    print(f'   Output: {args.output}')


if __name__ == '__main__':
    main()