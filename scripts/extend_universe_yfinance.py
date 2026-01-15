#!/usr/bin/env python3
"""
extend_universe_yfinance.py

Extends your existing 21-stock universe with new stocks using Yahoo Finance data.
Adds 29 additional large/mid-cap biotech stocks to reach 50 total.

Prerequisites:
    pip install yfinance pandas --break-system-packages

Usage:
    python extend_universe_yfinance.py --base production_data/universe.json --output top50_universe.json
"""

import argparse
import json
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import time

try:
    import yfinance as yf
    import pandas as pd
except ImportError:
    print("ERROR: Required packages not installed")
    print("Please run: pip install yfinance pandas --break-system-packages")
    sys.exit(1)


# 29 additional stocks to add (large/mid caps with good data availability)
NEW_TICKERS = [
    # Mega caps (easy data)
    "AMGN", "GILD", "BIIB", "MRNA", "BNTX",
    
    # Large caps
    "SGEN", "LGND", "TECH", "BGNE", "RGEN",
    
    # Mid caps
    "ROIV", "NTRA", "HZNP", "DAWN", "PCVX",
    "ARVN", "LEGN", "IMMU", "BLUE", "FATE",
    "NTLA", "EDIT", "BEAM", "VCYT", "NSTG",
    "AXSM", "PTGX", "CDNA", "PRVA"
]


def collect_market_data(ticker: str, end_date: datetime) -> Optional[Dict]:
    """
    Collect defensive features for one ticker using Yahoo Finance.
    
    Returns defensive_features dict or None if data unavailable.
    """
    try:
        print(f"  Collecting {ticker}...", end=" ")
        
        stock = yf.Ticker(ticker)
        
        # Get 90 days of history (to ensure 60 trading days)
        start_date = end_date - timedelta(days=120)
        hist = stock.history(start=start_date, end=end_date)
        
        if len(hist) < 30:
            print(f"✗ Insufficient data ({len(hist)} days)")
            return None
        
        # Calculate features
        current_price = float(hist['Close'].iloc[-1])
        
        # Get price 60 trading days ago (or as close as possible)
        if len(hist) >= 60:
            price_60d_ago = float(hist['Close'].iloc[-60])
        else:
            price_60d_ago = float(hist['Close'].iloc[0])
        
        returns = hist['Close'].pct_change().dropna()
        
        # Get XBI for correlation calculation
        xbi = yf.Ticker('XBI')
        xbi_hist = xbi.history(start=start_date, end=end_date)
        
        if len(xbi_hist) < 30:
            print(f"✗ XBI data unavailable")
            return None
        
        xbi_returns = xbi_hist['Close'].pct_change().dropna()
        
        # Align dates for correlation
        common_dates = returns.index.intersection(xbi_returns.index)
        
        if len(common_dates) < 20:
            print(f"✗ Insufficient overlap with XBI")
            return None
        
        correlation = float(returns.loc[common_dates].corr(xbi_returns.loc[common_dates]))
        
        # Calculate drawdown
        peak = hist['Close'].max()
        trough = hist['Close'].min()
        drawdown = float((trough / peak) - 1.0)
        
        # Calculate annualized volatility
        vol_60d = float(returns.std() * (252 ** 0.5))
        
        # Simple RSI calculation (14-day)
        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        rsi_current = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0
        
        # Determine volatility regime
        vol_regime = "normal"
        if vol_60d > 0.50:
            vol_regime = "high"
        elif vol_60d > 0.40:
            vol_regime = "elevated"
        
        features = {
            "price_current": f"{current_price:.2f}",
            "price_60d_ago": f"{price_60d_ago:.2f}",
            "return_60d": f"{(current_price / price_60d_ago - 1.0):.4f}",
            "vol_60d": f"{vol_60d:.4f}",
            "drawdown_60d": f"{drawdown:.4f}",
            "corr_xbi": f"{correlation:.4f}",
            "rsi_14d": f"{rsi_current:.1f}",
            "vol_regime": vol_regime,
            "timestamp": datetime.now().isoformat() + "Z"
        }
        
        print(f"✓ (vol: {vol_60d:.2%}, corr: {correlation:.2f})")
        return features
        
    except Exception as e:
        print(f"✗ Error: {str(e)[:50]}")
        return None


def get_basic_info(ticker: str) -> Dict:
    """Get basic company info from Yahoo Finance."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        return {
            "name": info.get('longName', f"{ticker} Inc"),
            "market_cap": info.get('marketCap', 1000000000),
            "price": info.get('currentPrice', 50.0),
            "shares": info.get('sharesOutstanding', 20000000)
        }
    except:
        return {
            "name": f"{ticker} Inc",
            "market_cap": 1000000000,
            "price": 50.0,
            "shares": 20000000
        }


def create_security_entry(ticker: str, defensive_features: Dict, info: Dict) -> Dict:
    """Create a complete security entry for the universe."""
    return {
        "ticker": ticker,
        "name": info["name"],
        "status": "active",
        "status_reason": "active",
        "market_cap_usd": str(info["market_cap"]),
        "price_usd": f"{info['price']:.2f}",
        "shares_outstanding": str(info["shares"]),
        "defensive_features": defensive_features,
        
        # Placeholder financial data (would need SEC filings for real data)
        "financial_health": {
            "cash_usd": str(int(info["market_cap"] * 0.3)),  # Rough estimate: 30% of market cap
            "total_debt_usd": str(int(info["market_cap"] * 0.1)),  # Rough estimate: 10% of market cap
            "quarterly_burn_usd": str(int(info["market_cap"] * 0.02)),  # Rough estimate: 2% per quarter
            "runway_quarters": "15"  # Placeholder
        },
        
        # Placeholder clinical data (would need ClinicalTrials.gov for real data)
        "lead_program": {
            "phase": "phase_2",  # Placeholder - varies by company
            "indication": "oncology",  # Placeholder - varies by company
            "trial_id": "NCT00000000"  # Placeholder
        }
    }


def load_existing_universe(filepath: str) -> List[Dict]:
    """Load your existing 21-stock universe."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Handle both array and dict formats
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and "active_securities" in data:
        return data["active_securities"]
    else:
        return data
    
    return []


def main():
    parser = argparse.ArgumentParser(description="Extend universe with Yahoo Finance data")
    parser.add_argument("--base", default="production_data/universe.json",
                       help="Your existing universe file (21 stocks)")
    parser.add_argument("--output", default="top50_universe.json",
                       help="Output file for extended universe (50 stocks)")
    parser.add_argument("--as-of-date", default=datetime.now().strftime("%Y-%m-%d"),
                       help="As-of date for data collection")
    parser.add_argument("--delay", type=float, default=0.5,
                       help="Delay between API calls (seconds)")
    args = parser.parse_args()
    
    print("="*80)
    print("EXTEND UNIVERSE WITH YAHOO FINANCE DATA")
    print("="*80)
    print(f"Base universe: {args.base}")
    print(f"Output: {args.output}")
    print(f"As of date: {args.as_of_date}")
    print(f"Adding {len(NEW_TICKERS)} new stocks")
    print()
    
    # Load existing universe
    print("Loading existing universe...")
    existing = load_existing_universe(args.base)
    existing_tickers = {sec['ticker'] for sec in existing}
    print(f"✓ Loaded {len(existing)} existing securities")
    print()
    
    # Collect data for new tickers
    print("Collecting data for new tickers...")
    print(f"(This will take ~{len(NEW_TICKERS) * args.delay:.0f} seconds with rate limiting)")
    print()
    
    end_date = datetime.strptime(args.as_of_date, "%Y-%m-%d")
    new_securities = []
    failed_tickers = []
    
    for i, ticker in enumerate(NEW_TICKERS, 1):
        print(f"[{i}/{len(NEW_TICKERS)}] ", end="")
        
        # Get defensive features
        defensive_features = collect_market_data(ticker, end_date)
        
        if defensive_features is None:
            failed_tickers.append(ticker)
            time.sleep(args.delay)
            continue
        
        # Get basic info
        info = get_basic_info(ticker)
        
        # Create security entry
        security = create_security_entry(ticker, defensive_features, info)
        new_securities.append(security)
        
        # Rate limiting
        time.sleep(args.delay)
    
    # Combine
    extended_universe = existing + new_securities
    
    print()
    print("="*80)
    print("COLLECTION SUMMARY")
    print("="*80)
    print(f"Existing securities: {len(existing)}")
    print(f"New securities collected: {len(new_securities)}")
    print(f"Failed: {len(failed_tickers)}")
    if failed_tickers:
        print(f"  Failed tickers: {', '.join(failed_tickers)}")
    print(f"Total universe size: {len(extended_universe)}")
    print()
    
    # Save
    with open(args.output, 'w') as f:
        json.dump(extended_universe, f, indent=2)
    
    print(f"✓ Extended universe saved to: {args.output}")
    print()
    
    # Summary statistics
    if new_securities:
        vols = [float(s['defensive_features']['vol_60d']) for s in new_securities]
        corrs = [float(s['defensive_features']['corr_xbi']) for s in new_securities]
        
        print("="*80)
        print("NEW SECURITIES STATISTICS")
        print("="*80)
        print(f"Volatility range: {min(vols):.2%} to {max(vols):.2%}")
        print(f"Average volatility: {sum(vols)/len(vols):.2%}")
        print(f"Correlation range: {min(corrs):.2f} to {max(corrs):.2f}")
        print(f"Average correlation: {sum(corrs)/len(corrs):.2f}")
        print()
    
    print("="*80)
    print("NEXT STEPS")
    print("="*80)
    print()
    print("1. Copy to production data directory:")
    print(f"   Copy-Item {args.output} production_data/universe.json")
    print()
    print("2. Run screener:")
    print("   python run_screen.py \\")
    print("       --as-of-date 2026-01-06 \\")
    print("       --data-dir production_data \\")
    print("       --output top50_screening_results.json")
    print()
    print("3. Analyze results:")
    print("   python analyze_defensive_impact.py \\")
    print("       --output top50_screening_results.json")
    print()
    print("="*80)


if __name__ == "__main__":
    main()
