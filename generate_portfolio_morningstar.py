#!/usr/bin/env python3
"""
Portfolio Construction from Rankings - MORNINGSTAR VERSION
Uses institutional-grade Morningstar Direct returns data

Usage:
    python scripts/generate_portfolio_morningstar.py --date 2026-01-15 --top-n 60
"""

import json
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from decimal import Decimal
import math
from typing import Dict, List, Tuple

def load_rankings(path: Path) -> List[Dict]:
    """Load ranked tickers with momentum - handles multiple formats"""
    # Use utf-8 encoding to handle special characters
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    
    # Format 1: Already a list
    if isinstance(data, list):
        return data
    
    # Format 2: Multi-module pipeline output (has module_5_composite key)
    if isinstance(data, dict) and 'module_5_composite' in data:
        module5_data = data['module_5_composite']
        
        # Try different nested structures
        if 'ranked_securities' in module5_data:
            securities = module5_data['ranked_securities']
        elif 'securities' in module5_data:
            securities = module5_data['securities']
        elif 'rankings' in module5_data:
            securities = module5_data['rankings']
        elif isinstance(module5_data, list):
            securities = module5_data
        elif isinstance(module5_data, dict):
            # Maybe it's a dict of ticker -> data
            securities = []
            for ticker, ticker_data in module5_data.items():
                if isinstance(ticker_data, dict) and ticker not in ['as_of_date', 'metadata', 'provenance']:
                    if 'ticker' not in ticker_data:
                        ticker_data['ticker'] = ticker
                    securities.append(ticker_data)
            
            # Sort by score
            if securities and 'final_score' in securities[0]:
                securities.sort(key=lambda x: x.get('final_score', 0), reverse=True)
        else:
            securities = []
        
        if securities:
            return securities
    
    # Format 3: Has 'securities' key
    if isinstance(data, dict) and 'securities' in data:
        return data['securities']
    
    # Format 4: Has 'rankings' key
    if isinstance(data, dict) and 'rankings' in data:
        return data['rankings']
    
    # Format 5: Has 'ranked_securities' key
    if isinstance(data, dict) and 'ranked_securities' in data:
        return data['ranked_securities']
    
    # Format 6: Dict with ticker keys at top level
    if isinstance(data, dict):
        # Check if keys look like tickers (all caps, 2-5 chars)
        ticker_like_keys = [k for k in data.keys() if isinstance(k, str) and k.isupper() and 2 <= len(k) <= 5]
        
        if len(ticker_like_keys) > 10:  # At least 10 ticker-like keys
            result = []
            for ticker in ticker_like_keys:
                ticker_data = data[ticker]
                if isinstance(ticker_data, dict):
                    if 'ticker' not in ticker_data:
                        ticker_data['ticker'] = ticker
                    result.append(ticker_data)
                else:
                    result.append({'ticker': ticker})
            
            # Sort by rank or score
            if result and 'rank' in result[0]:
                result.sort(key=lambda x: x.get('rank', 999))
            elif result and 'final_score' in result[0]:
                result.sort(key=lambda x: x.get('final_score', 0), reverse=True)
            
            return result
    
    # If we get here, format is unknown
    print(f"⚠️  Unknown JSON format in {path}")
    print(f"   Type: {type(data)}")
    if isinstance(data, dict):
        print(f"   Keys: {list(data.keys())[:10]}")
        # Try to show structure of first module
        if 'module_5_composite' in data:
            print(f"   module_5_composite keys: {list(data['module_5_composite'].keys())[:10]}")
    raise ValueError(f"Unexpected format in {path}")

def load_morningstar_returns(returns_db_path: Path) -> Dict:
    """Load Morningstar returns database"""
    with open(returns_db_path, encoding='utf-8') as f:
        data = json.load(f)
    
    # Convert list format to dict by ticker for easy lookup
    returns_by_ticker = {}
    
    for ticker, ticker_data in data.items():
        returns_by_ticker[ticker] = ticker_data
    
    return returns_by_ticker

def get_current_price(ticker: str, returns_data: Dict, as_of_date: str = None) -> float:
    """
    Get current price from Morningstar return index
    Price = most recent return index value
    """
    if ticker not in returns_data:
        return None
    
    ticker_data = returns_data[ticker]
    returns_list = ticker_data.get('returns', [])
    
    if not returns_list:
        return None
    
    # Get most recent return
    if as_of_date:
        # Find closest date <= as_of_date
        target_date = datetime.strptime(as_of_date, '%Y-%m-%d')
        valid_returns = [r for r in returns_list 
                        if datetime.strptime(r['date'], '%Y-%m-%d') <= target_date]
        if not valid_returns:
            return None
        latest = max(valid_returns, key=lambda x: x['date'])
    else:
        latest = returns_list[-1]
    
    # Return index value as proxy for price level
    # Note: This is relative, we'll use it for volatility calc
    return float(latest['return_index'])

def calculate_volatility_from_returns(ticker: str, returns_data: Dict, days: int = 30) -> float:
    """
    Calculate annualized volatility from Morningstar returns
    Uses last N days of daily returns
    """
    if ticker not in returns_data:
        return 0.50  # Default 50% vol
    
    ticker_data = returns_data[ticker]
    returns_list = ticker_data.get('returns', [])
    
    if len(returns_list) < days:
        return 0.50  # Default if insufficient data
    
    # Get last N days
    recent_returns = returns_list[-days:]
    
    # Calculate return series
    return_pcts = []
    for i in range(1, len(recent_returns)):
        prev_idx = float(recent_returns[i-1]['return_index'])
        curr_idx = float(recent_returns[i]['return_index'])
        
        if prev_idx > 0:
            daily_return = (curr_idx - prev_idx) / prev_idx
            return_pcts.append(daily_return)
    
    if len(return_pcts) < 20:
        return 0.50  # Default if insufficient returns
    
    # Calculate standard deviation
    mean_return = sum(return_pcts) / len(return_pcts)
    variance = sum((r - mean_return) ** 2 for r in return_pcts) / (len(return_pcts) - 1)
    daily_vol = math.sqrt(variance)
    
    # Annualize (252 trading days)
    annual_vol = daily_vol * math.sqrt(252)
    
    # Cap at reasonable bounds (10% - 200%)
    return max(0.10, min(2.0, annual_vol))

def get_latest_price_from_external(ticker: str, returns_data: Dict) -> float:
    """
    Get actual trading price (not return index)
    For production, you'd query Morningstar API for latest quote
    For now, use a scaling heuristic based on typical biotech prices
    """
    if ticker not in returns_data:
        return None
    
    ticker_data = returns_data[ticker]
    returns_list = ticker_data.get('returns', [])
    
    if not returns_list:
        return None
    
    # Get latest return index
    latest_idx = float(returns_list[-1]['return_index'])
    
    # Heuristic: Assume base price of $50 at index = 100
    # This is just for share count estimation
    # In production, you'd query actual prices from Morningstar quote API
    estimated_price = latest_idx * 0.50  # Scale factor
    
    # Cap at reasonable biotech price range ($5 - $500)
    return max(5.0, min(500.0, estimated_price))

def calculate_inverse_vol_weights(
    tickers: List[str],
    returns_data: Dict
) -> Dict[str, float]:
    """
    Inverse volatility weighting using Morningstar data
    - Lower volatility stocks get higher weight
    - Smooths out risk contribution across positions
    """
    # Calculate volatility for each ticker
    volatilities = {}
    for ticker in tickers:
        vol = calculate_volatility_from_returns(ticker, returns_data, days=30)
        volatilities[ticker] = vol
    
    # Calculate inverse volatility
    inv_vols = {t: 1.0 / v for t, v in volatilities.items()}
    
    # Normalize to sum to 1.0
    total_inv_vol = sum(inv_vols.values())
    weights = {t: inv_vol / total_inv_vol for t, inv_vol in inv_vols.items()}
    
    return weights, volatilities

def apply_concentration_limits(
    weights: Dict[str, float],
    max_position_pct: float = 3.0
) -> Dict[str, float]:
    """
    Apply concentration limits
    - No single position > max_position_pct
    - Redistribute excess to other positions
    """
    max_weight = max_position_pct / 100.0
    
    # Cap oversized positions
    capped = {}
    excess = 0.0
    
    for ticker, weight in weights.items():
        if weight > max_weight:
            capped[ticker] = max_weight
            excess += (weight - max_weight)
        else:
            capped[ticker] = weight
    
    # Redistribute excess proportionally to uncapped positions
    if excess > 0:
        uncapped_tickers = [t for t, w in capped.items() if w < max_weight]
        if uncapped_tickers:
            uncapped_total = sum(capped[t] for t in uncapped_tickers)
            if uncapped_total > 0:
                for ticker in uncapped_tickers:
                    boost = excess * (capped[ticker] / uncapped_total)
                    capped[ticker] = min(max_weight, capped[ticker] + boost)
    
    # Renormalize to ensure 100%
    total = sum(capped.values())
    normalized = {t: w / total for t, w in capped.items()}
    
    return normalized

def calculate_portfolio_metrics(
    positions: List[Dict],
    volatilities: Dict[str, float]
) -> Dict:
    """Calculate portfolio-level risk metrics"""
    
    # Weighted average momentum
    total_weight = sum(p['weight_pct'] for p in positions)
    avg_momentum = sum(
        p.get('momentum_score', 50) * p['weight_pct'] / total_weight
        for p in positions
    )
    
    # Weighted average volatility
    weighted_vol = 0.0
    for pos in positions:
        ticker = pos['ticker']
        if ticker in volatilities:
            weighted_vol += volatilities[ticker] * pos['weight_pct'] / 100.0
    
    # Portfolio beta estimate
    portfolio_beta = weighted_vol / 0.60 if weighted_vol > 0 else 0.0
    
    return {
        'num_positions': len(positions),
        'total_weight_pct': sum(p['weight_pct'] for p in positions),
        'avg_momentum': round(avg_momentum, 1),
        'max_position_pct': max(p['weight_pct'] for p in positions),
        'min_position_pct': min(p['weight_pct'] for p in positions),
        'portfolio_volatility': round(weighted_vol * 100, 1),
        'portfolio_beta_estimate': round(portfolio_beta, 2),
        'data_source': 'morningstar_direct',
        'tickers_with_data': sum(1 for p in positions if p.get('current_price')),
        'tickers_missing_data': sum(1 for p in positions if not p.get('current_price'))
    }

def generate_portfolio(
    ranked_path: Path,
    returns_db_path: Path,
    top_n: int,
    portfolio_value: float,
    date_str: str,
    output_path: Path,
    max_position_pct: float = 3.0
):
    """Main portfolio generation logic"""
    
    print("="*70)
    print("PORTFOLIO GENERATION - MORNINGSTAR POWERED")
    print("="*70)
    print(f"Date: {date_str}")
    print(f"Portfolio value: ${portfolio_value:,.0f}")
    print(f"Target positions: {top_n}")
    print(f"Max position: {max_position_pct}%")
    print(f"Data source: Morningstar Direct")
    print()
    
    # Load rankings
    print("1. Loading rankings...")
    securities = load_rankings(ranked_path)
    print(f"   Found {len(securities)} ranked securities")
    
    # Load Morningstar returns
    print("2. Loading Morningstar returns database...")
    returns_data = load_morningstar_returns(returns_db_path)
    print(f"   Loaded {len(returns_data)} tickers from Morningstar")
    print()
    
    # Take top N
    top_securities = securities[:top_n]
    tickers = [s['ticker'] for s in top_securities]
    print(f"   Selected top {len(tickers)} for portfolio")
    
    # Check coverage
    tickers_with_data = [t for t in tickers if t in returns_data]
    print(f"   Morningstar coverage: {len(tickers_with_data)}/{len(tickers)} ({100*len(tickers_with_data)/len(tickers):.1f}%)")
    print()
    
    # Calculate weights
    print("3. Calculating position sizes...")
    weights, volatilities = calculate_inverse_vol_weights(tickers_with_data, returns_data)
    
    # Apply concentration limits
    final_weights = apply_concentration_limits(weights, max_position_pct)
    print(f"   Applied {max_position_pct}% concentration limit")
    print()
    
    # Build position list
    print("4. Building position list...")
    positions = []
    
    for i, security in enumerate(top_securities):
        ticker = security.get('ticker')
        
        if not ticker:
            continue
        
        if ticker not in tickers_with_data:
            print(f"   ⚠️  Skipping {ticker} (no Morningstar data)")
            continue
        
        weight_pct = final_weights.get(ticker, 0) * 100
        position_value = portfolio_value * weight_pct / 100
        
        # Get price estimate for share count
        price_estimate = get_latest_price_from_external(ticker, returns_data)
        shares = int(position_value / price_estimate) if price_estimate and price_estimate > 0 else 0
        
        # Map field names (handle different JSON formats)
        original_score = float(security.get('original_score') or 
                              security.get('composite_score_original') or 
                              security.get('composite_score') or 0)
        
        final_score = float(security.get('final_score') or 
                           security.get('composite_score_with_momentum') or 
                           security.get('composite_score') or 0)
        
        momentum_score = float(security.get('momentum_score', 50))
        
        position = {
            'rank': i + 1,
            'ticker': ticker,
            'weight_pct': round(weight_pct, 2),
            'position_value_usd': round(position_value, 2),
            'original_score': round(original_score, 2),
            'momentum_score': round(momentum_score, 1),
            'final_score': round(final_score, 2),
            'volatility': round(volatilities.get(ticker, 0.50), 3),
            'price_estimate': round(price_estimate, 2) if price_estimate else None,
            'shares_estimate': shares,
            'data_source': 'morningstar_direct'
        }
        
        positions.append(position)
    
    print(f"   Generated {len(positions)} positions")
    print()
    
    # Calculate portfolio metrics
    print("5. Calculating portfolio metrics...")
    metrics = calculate_portfolio_metrics(positions, volatilities)
    print(f"   Portfolio beta estimate: {metrics['portfolio_beta_estimate']}")
    print(f"   Portfolio volatility: {metrics['portfolio_volatility']}%")
    print(f"   Average momentum: {metrics['avg_momentum']}")
    print()
    
    # Build output
    output = {
        'date': date_str,
        'generated_at': datetime.now().isoformat(),
        'portfolio_value_usd': portfolio_value,
        'num_positions': len(positions),
        'positions': positions,
        'metrics': metrics,
        'parameters': {
            'top_n': top_n,
            'max_position_pct': max_position_pct,
            'weighting_method': 'inverse_volatility',
            'data_source': 'morningstar_direct',
            'returns_db_path': str(returns_db_path)
        },
        'data_quality': {
            'tickers_requested': len(tickers),
            'tickers_with_morningstar_data': len(tickers_with_data),
            'coverage_pct': round(100 * len(tickers_with_data) / len(tickers), 1)
        }
    }
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"✅ Portfolio saved to {output_path}")
    print()
    
    # Print summary
    print("="*70)
    print("TOP 10 POSITIONS")
    print("="*70)
    print(f"{'Rank':<6}{'Ticker':<8}{'Weight':>8}{'Value':>12}{'Vol':>8}{'Score':>8}")
    print("-"*70)
    
    for pos in positions[:10]:
        print(
            f"{pos['rank']:<6}"
            f"{pos['ticker']:<8}"
            f"{pos['weight_pct']:>7.2f}%"
            f"${pos['position_value_usd']:>10,.0f}"
            f"{pos['volatility']*100:>7.1f}%"
            f"{pos['final_score']:>8.2f}"
        )
    
    print()
    print("="*70)
    print("PORTFOLIO METRICS")
    print("="*70)
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    print()
    
    return output

def main():
    parser = argparse.ArgumentParser(description='Generate portfolio from rankings (Morningstar)')
    parser.add_argument('--date', required=True, help='Portfolio date (YYYY-MM-DD)')
    parser.add_argument('--top-n', type=int, default=60, help='Number of positions (default: 60)')
    parser.add_argument('--portfolio-value', type=float, default=10_000_000, 
                       help='Portfolio value in USD (default: 10M)')
    parser.add_argument('--ranked-path', default='outputs/ranked_with_momentum.json',
                       help='Path to ranked JSON file')
    parser.add_argument('--returns-db', default='data/returns/returns_db_daily.json',
                       help='Path to Morningstar returns database')
    parser.add_argument('--output-path', help='Output path (default: outputs/portfolio_YYYYMMDD.json)')
    parser.add_argument('--max-position-pct', type=float, default=3.0,
                       help='Max position size as % of portfolio (default: 3.0)')
    
    args = parser.parse_args()
    
    # Parse date
    date_str = args.date
    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    date_compact = date_obj.strftime('%Y%m%d')
    
    # Set paths
    ranked_path = Path(args.ranked_path)
    returns_db_path = Path(args.returns_db)
    
    if args.output_path:
        output_path = Path(args.output_path)
    else:
        output_path = Path(f'outputs/portfolio_{date_compact}_morningstar.json')
    
    # Verify returns DB exists
    if not returns_db_path.exists():
        print(f"❌ Morningstar returns database not found: {returns_db_path}")
        print("   Run: python scripts/build_returns_database.py")
        return 1
    
    # Generate
    generate_portfolio(
        ranked_path=ranked_path,
        returns_db_path=returns_db_path,
        top_n=args.top_n,
        portfolio_value=args.portfolio_value,
        date_str=date_str,
        output_path=output_path,
        max_position_pct=args.max_position_pct
    )
    
    return 0

if __name__ == '__main__':
    exit(main())
