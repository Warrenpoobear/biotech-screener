#!/usr/bin/env python3
"""
Portfolio Construction from Rankings
Takes momentum-integrated rankings and generates trade-ready portfolio

Usage:
    python scripts/generate_portfolio_from_rankings.py --date 2026-01-15 --top-n 60
"""

import json
import argparse
from datetime import datetime
from pathlib import Path
from decimal import Decimal
import yfinance as yf
from typing import Dict, List, Tuple

def load_rankings(path: Path) -> List[Dict]:
    """Load ranked tickers with momentum"""
    with open(path) as f:
        data = json.load(f)
    
    if isinstance(data, list):
        return data
    elif 'securities' in data:
        return data['securities']
    else:
        raise ValueError(f"Unexpected format in {path}")

def fetch_current_prices(tickers: List[str]) -> Dict[str, float]:
    """Fetch current prices and volatility for position sizing"""
    prices = {}
    volatilities = {}
    
    print(f"Fetching prices for {len(tickers)} tickers...")
    
    # Batch fetch for efficiency
    batch_size = 50
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        batch_str = " ".join(batch)
        
        try:
            data = yf.download(batch_str, period="90d", progress=False, show_errors=False)
            
            for ticker in batch:
                try:
                    if len(batch) == 1:
                        close_prices = data['Close']
                    else:
                        close_prices = data['Close'][ticker]
                    
                    if len(close_prices) > 0:
                        # Current price
                        prices[ticker] = float(close_prices.iloc[-1])
                        
                        # 30-day volatility (annualized)
                        returns = close_prices.pct_change().dropna()
                        if len(returns) >= 20:
                            volatilities[ticker] = float(returns.std() * (252 ** 0.5))
                        else:
                            volatilities[ticker] = 0.50  # Default 50% volatility
                    
                except Exception as e:
                    print(f"  ⚠️  Error fetching {ticker}: {e}")
                    continue
        
        except Exception as e:
            print(f"  ⚠️  Batch error: {e}")
            continue
    
    print(f"  ✅ Fetched {len(prices)} prices")
    print(f"  ✅ Calculated {len(volatilities)} volatilities")
    
    return prices, volatilities

def calculate_inverse_vol_weights(
    tickers: List[str],
    volatilities: Dict[str, float],
    prices: Dict[str, float]
) -> Dict[str, float]:
    """
    Inverse volatility weighting
    - Lower volatility stocks get higher weight
    - Smooths out risk contribution across positions
    """
    # Calculate inverse volatility for each ticker
    inv_vols = {}
    for ticker in tickers:
        if ticker in volatilities and ticker in prices:
            vol = volatilities[ticker]
            # Avoid division by zero, cap at 200% vol
            vol = max(0.10, min(2.0, vol))
            inv_vols[ticker] = 1.0 / vol
        else:
            # If no data, use average inverse vol
            inv_vols[ticker] = 2.0  # ~50% vol
    
    # Normalize to sum to 1.0
    total_inv_vol = sum(inv_vols.values())
    weights = {t: inv_vol / total_inv_vol for t, inv_vol in inv_vols.items()}
    
    return weights

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
                    # Proportional redistribution
                    boost = excess * (capped[ticker] / uncapped_total)
                    capped[ticker] = min(max_weight, capped[ticker] + boost)
    
    # Renormalize to ensure 100%
    total = sum(capped.values())
    normalized = {t: w / total for t, w in capped.items()}
    
    return normalized

def calculate_portfolio_metrics(
    positions: List[Dict],
    prices: Dict[str, float],
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
    
    # Portfolio beta estimate (using weighted vol as proxy)
    # Typical biotech vol ~60%, market vol ~20%, so biotech beta ~0.6-0.8
    portfolio_beta = weighted_vol / 0.60 if weighted_vol > 0 else 0.0
    
    return {
        'num_positions': len(positions),
        'total_weight_pct': sum(p['weight_pct'] for p in positions),
        'avg_momentum': round(avg_momentum, 1),
        'max_position_pct': max(p['weight_pct'] for p in positions),
        'min_position_pct': min(p['weight_pct'] for p in positions),
        'portfolio_volatility': round(weighted_vol * 100, 1),
        'portfolio_beta_estimate': round(portfolio_beta, 2),
        'tickers_with_prices': sum(1 for p in positions if p.get('current_price')),
        'tickers_missing_prices': sum(1 for p in positions if not p.get('current_price'))
    }

def generate_portfolio(
    ranked_path: Path,
    top_n: int,
    portfolio_value: float,
    date_str: str,
    output_path: Path,
    max_position_pct: float = 3.0
):
    """Main portfolio generation logic"""
    
    print("="*70)
    print("PORTFOLIO GENERATION")
    print("="*70)
    print(f"Date: {date_str}")
    print(f"Portfolio value: ${portfolio_value:,.0f}")
    print(f"Target positions: {top_n}")
    print(f"Max position: {max_position_pct}%")
    print()
    
    # Load rankings
    print("1. Loading rankings...")
    securities = load_rankings(ranked_path)
    print(f"   Found {len(securities)} ranked securities")
    
    # Take top N
    top_securities = securities[:top_n]
    tickers = [s['ticker'] for s in top_securities]
    print(f"   Selected top {len(tickers)} for portfolio")
    print()
    
    # Fetch prices and volatility
    print("2. Fetching market data...")
    prices, volatilities = fetch_current_prices(tickers)
    print()
    
    # Calculate weights
    print("3. Calculating position sizes...")
    weights = calculate_inverse_vol_weights(tickers, volatilities, prices)
    
    # Apply concentration limits
    final_weights = apply_concentration_limits(weights, max_position_pct)
    print(f"   Applied {max_position_pct}% concentration limit")
    print()
    
    # Build position list
    print("4. Building position list...")
    positions = []
    
    for i, security in enumerate(top_securities):
        ticker = security['ticker']
        weight_pct = final_weights.get(ticker, 0) * 100
        position_value = portfolio_value * weight_pct / 100
        
        position = {
            'rank': i + 1,
            'ticker': ticker,
            'weight_pct': round(weight_pct, 2),
            'position_value_usd': round(position_value, 2),
            'original_score': security.get('original_score', 0),
            'momentum_score': security.get('momentum_score', 50),
            'final_score': security.get('final_score', 0),
            'current_price': prices.get(ticker),
            'volatility': volatilities.get(ticker),
            'shares': int(position_value / prices[ticker]) if ticker in prices and prices[ticker] > 0 else 0
        }
        
        positions.append(position)
    
    print(f"   Generated {len(positions)} positions")
    print()
    
    # Calculate portfolio metrics
    print("5. Calculating portfolio metrics...")
    metrics = calculate_portfolio_metrics(positions, prices, volatilities)
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
            'weighting_method': 'inverse_volatility'
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
    print(f"{'Rank':<6}{'Ticker':<8}{'Weight':>8}{'Value':>12}{'Shares':>8}{'Price':>10}")
    print("-"*70)
    
    for pos in positions[:10]:
        print(
            f"{pos['rank']:<6}"
            f"{pos['ticker']:<8}"
            f"{pos['weight_pct']:>7.2f}%"
            f"${pos['position_value_usd']:>10,.0f}"
            f"{pos['shares']:>8}"
            f"${pos['current_price']:>9.2f}" if pos['current_price'] else "N/A"
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
    parser = argparse.ArgumentParser(description='Generate portfolio from rankings')
    parser.add_argument('--date', required=True, help='Portfolio date (YYYY-MM-DD)')
    parser.add_argument('--top-n', type=int, default=60, help='Number of positions (default: 60)')
    parser.add_argument('--portfolio-value', type=float, default=10_000_000, 
                       help='Portfolio value in USD (default: 10M)')
    parser.add_argument('--ranked-path', default='outputs/ranked_with_momentum.json',
                       help='Path to ranked JSON file')
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
    if args.output_path:
        output_path = Path(args.output_path)
    else:
        output_path = Path(f'outputs/portfolio_{date_compact}.json')
    
    # Generate
    generate_portfolio(
        ranked_path=ranked_path,
        top_n=args.top_n,
        portfolio_value=args.portfolio_value,
        date_str=date_str,
        output_path=output_path,
        max_position_pct=args.max_position_pct
    )

if __name__ == '__main__':
    main()
