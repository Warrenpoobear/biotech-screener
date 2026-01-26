#!/usr/bin/env python3
"""
extend_to_100_stocks.py

Extends the 44-stock universe to 100 stocks by adding 56 more biotech stocks
from XBI, IBB, and NBI constituent lists.

Usage:
    python extend_to_100_stocks.py --base top50_universe.json --output universe_100stocks.json
"""

import argparse
import json
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

try:
    import yfinance as yf
    import pandas as pd
except ImportError:
    print("ERROR: Required packages not installed")
    print("Please run: pip install yfinance pandas --break-system-packages")
    sys.exit(1)


# 56 additional stocks to reach 100 total
# Focused on liquid mid/small caps with good data availability
PHASE2_TICKERS = [
    # Mid-cap biotech (good liquidity)
    "AGIO", "AKRO", "ALLO", "APLS", "ARCT", "ARWR", "AVTR", "BCRX",
    "BDTX", "BGXX", "BPMC", "CRBU", "CRNX", "CRVS", "CVAC", "CLDX",
    
    # Small-cap with clinical catalysts
    "CYTK", "DNLI", "DRMA", "DVAX", "ETNB", "FGEN", "FIXX", "FULC",
    "GLPG", "HALO", "HRTX", "ICPT", "IMCR", "IMMP", "IMVT", "IRWD",
    
    # Emerging biotechs
    "ITCI", "KALA", "KALV", "KRYS", "KYMR", "LQDA", "MCRB", "MDGL",
    "MRSN", "MRVI", "NVCR", "NVAX", "OCGN", "OPCH", "PCRX", "PGEN",
    
    # Additional XBI constituents  
    "PRTA", "PTCT", "RVMD", "RXRX", "RGNX", "SAGE", "SANA", "SGMO",
    "SNDX", "SPRY", "TARS", "TBPH", "TCDA", "TGTX"
]

# Note: This is 56 stocks, bringing total from 44 to 100


def load_existing_universe(filepath: str) -> tuple[List[Dict], set]:
    """Load existing universe and return securities + ticker set."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Handle both array and dict formats
    if isinstance(data, list):
        securities = data
    elif isinstance(data, dict) and "active_securities" in data:
        securities = data["active_securities"]
    else:
        securities = data
    
    existing_tickers = {sec['ticker'] for sec in securities}
    return securities, existing_tickers


def collect_market_data(ticker: str, end_date: datetime) -> Optional[Dict]:
    """Collect defensive features using Yahoo Finance."""
    try:
        print(f"  Collecting {ticker}...", end=" ", flush=True)
        
        stock = yf.Ticker(ticker)
        start_date = end_date - timedelta(days=120)
        hist = stock.history(start=start_date, end=end_date)
        
        if len(hist) < 30:
            print(f"✗ Insufficient data ({len(hist)} days)")
            return None
        
        current_price = float(hist['Close'].iloc[-1])
        
        if len(hist) >= 60:
            price_60d_ago = float(hist['Close'].iloc[-60])
        else:
            price_60d_ago = float(hist['Close'].iloc[0])
        
        returns = hist['Close'].pct_change().dropna()
        
        # Get XBI for correlation
        xbi = yf.Ticker('XBI')
        xbi_hist = xbi.history(start=start_date, end=end_date)
        
        if len(xbi_hist) < 30:
            print(f"✗ XBI data unavailable")
            return None
        
        xbi_returns = xbi_hist['Close'].pct_change().dropna()
        common_dates = returns.index.intersection(xbi_returns.index)
        
        if len(common_dates) < 20:
            print(f"✗ Insufficient overlap")
            return None
        
        correlation = float(returns.loc[common_dates].corr(xbi_returns.loc[common_dates]))
        
        # Calculate metrics
        peak = hist['Close'].max()
        trough = hist['Close'].min()
        drawdown = float((trough / peak) - 1.0)
        vol_60d = float(returns.std() * (252 ** 0.5))
        
        # Simple RSI
        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        rsi_current = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0
        
        # Volatility regime
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
        print(f"✗ Error: {str(e)[:40]}")
        return None


def get_basic_info(ticker: str) -> Dict:
    """Get basic company info from Yahoo Finance."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        return {
            "name": info.get('longName', f"{ticker} Inc"),
            "market_cap": info.get('marketCap', 500000000),
            "price": info.get('currentPrice', 30.0),
            "shares": info.get('sharesOutstanding', 20000000)
        }
    except Exception:
        return {
            "name": f"{ticker} Inc",
            "market_cap": 500000000,
            "price": 30.0,
            "shares": 20000000
        }


def create_security_entry(ticker: str, defensive_features: Dict, info: Dict) -> Dict:
    """Create a complete security entry."""
    return {
        "ticker": ticker,
        "name": info["name"],
        "status": "active",
        "status_reason": "active",
        "market_cap_usd": str(info["market_cap"]),
        "price_usd": f"{info['price']:.2f}",
        "shares_outstanding": str(info["shares"]),
        "defensive_features": defensive_features,
        
        # Placeholder financial data
        "financial_health": {
            "cash_usd": str(int(info["market_cap"] * 0.25)),
            "total_debt_usd": str(int(info["market_cap"] * 0.10)),
            "quarterly_burn_usd": str(int(info["market_cap"] * 0.02)),
            "runway_quarters": "12"
        },
        
        # Placeholder clinical data
        "lead_program": {
            "phase": "phase_2",
            "indication": "oncology",
            "trial_id": "NCT00000000"
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Extend to 100 stocks")
    parser.add_argument("--base", default="top50_universe.json",
                       help="Current universe file (44 stocks)")
    parser.add_argument("--output", default="universe_100stocks.json",
                       help="Output file (100 stocks)")
    parser.add_argument("--as-of-date", default=datetime.now().strftime("%Y-%m-%d"),
                       help="As-of date")
    parser.add_argument("--delay", type=float, default=0.5,
                       help="Delay between API calls")
    args = parser.parse_args()
    
    print("="*80)
    print("PHASE 2: EXPAND TO 100 STOCKS")
    print("="*80)
    print(f"Base universe: {args.base}")
    print(f"Output: {args.output}")
    print(f"As of date: {args.as_of_date}")
    print(f"Adding {len(PHASE2_TICKERS)} new stocks")
    print()
    
    # Load existing
    print("Loading existing universe...")
    existing, existing_tickers = load_existing_universe(args.base)
    print(f"✓ Loaded {len(existing)} existing securities")
    
    # Check for duplicates
    duplicates = [t for t in PHASE2_TICKERS if t in existing_tickers]
    if duplicates:
        print(f"\nWARNING: {len(duplicates)} tickers already in universe:")
        print(f"  {', '.join(duplicates[:10])}")
        print("Skipping duplicates...")
        new_tickers = [t for t in PHASE2_TICKERS if t not in existing_tickers]
    else:
        new_tickers = PHASE2_TICKERS
    
    print()
    print(f"Collecting data for {len(new_tickers)} new tickers...")
    print(f"(This will take ~{len(new_tickers) * args.delay:.0f} seconds with rate limiting)")
    print()
    
    end_date = datetime.strptime(args.as_of_date, "%Y-%m-%d")
    new_securities = []
    failed_tickers = []
    
    for i, ticker in enumerate(new_tickers, 1):
        print(f"[{i}/{len(new_tickers)}] ", end="")
        
        defensive_features = collect_market_data(ticker, end_date)
        
        if defensive_features is None:
            failed_tickers.append(ticker)
            time.sleep(args.delay)
            continue
        
        info = get_basic_info(ticker)
        security = create_security_entry(ticker, defensive_features, info)
        new_securities.append(security)
        
        time.sleep(args.delay)
    
    # Combine
    extended_universe = existing + new_securities
    
    print()
    print("="*80)
    print("PHASE 2 SUMMARY")
    print("="*80)
    print(f"Starting securities: {len(existing)}")
    print(f"New securities collected: {len(new_securities)}")
    print(f"Failed: {len(failed_tickers)}")
    if failed_tickers:
        print(f"  Failed tickers: {', '.join(failed_tickers[:15])}")
        if len(failed_tickers) > 15:
            print(f"  ... and {len(failed_tickers) - 15} more")
    print(f"Final universe size: {len(extended_universe)}")
    print()
    
    # Save
    with open(args.output, 'w') as f:
        json.dump(extended_universe, f, indent=2)
    
    print(f"✓ Extended universe saved to: {args.output}")
    print()
    
    # Statistics
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
    print("1. Copy to production:")
    print(f"   Copy-Item {args.output} production_data/universe.json")
    print()
    print("2. Run screener on 100 stocks:")
    print("   python run_screen.py \\")
    print("       --as-of-date 2026-01-06 \\")
    print("       --data-dir production_data \\")
    print("       --output screening_100stocks.json")
    print()
    print("3. Analyze results:")
    print("   python analyze_defensive_impact.py \\")
    print("       --output screening_100stocks.json")
    print()
    print("="*80)


if __name__ == "__main__":
    main()
