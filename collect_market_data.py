#!/usr/bin/env python3
"""
collect_market_data.py - Collect Market Data from Yahoo Finance

Fetches price, volume, market cap, and other market data for all tickers in universe.

Usage:
    python collect_market_data.py --universe production_data/universe.json
"""

import json
import time
from pathlib import Path
from datetime import date, datetime, timedelta
from typing import Dict, Optional
import argparse


def get_market_data(ticker: str) -> Optional[Dict]:
    """Get comprehensive market data for a ticker using yfinance"""
    try:
        import yfinance as yf
        
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Get 90-day history for volatility/returns
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)
        hist = stock.history(start=start_date, end=end_date)
        
        # Calculate metrics
        if not hist.empty and len(hist) > 0:
            current_price = float(hist['Close'].iloc[-1])
            avg_volume_90d = float(hist['Volume'].mean())
            
            # Volatility (annualized)
            returns = hist['Close'].pct_change()
            volatility_90d = float(returns.std() * (252 ** 0.5)) if len(returns) > 1 else None
            
            # Returns
            returns_1m = float((hist['Close'].iloc[-1] / hist['Close'].iloc[-21]) - 1) if len(hist) >= 21 else None
            returns_3m = float((hist['Close'].iloc[-1] / hist['Close'].iloc[0]) - 1) if len(hist) >= 63 else None
            
            high_90d = float(hist['High'].max())
            low_90d = float(hist['Low'].min())
        else:
            current_price = None
            avg_volume_90d = None
            volatility_90d = None
            returns_1m = None
            returns_3m = None
            high_90d = None
            low_90d = None
        
        return {
            "ticker": ticker,
            "price": current_price,
            "market_cap": info.get('marketCap'),
            "enterprise_value": info.get('enterpriseValue'),
            "shares_outstanding": info.get('sharesOutstanding'),
            "float_shares": info.get('floatShares'),
            "avg_volume": info.get('averageVolume'),
            "avg_volume_90d": avg_volume_90d,
            "beta": info.get('beta'),
            "short_percent": info.get('shortPercentOfFloat'),
            "short_ratio": info.get('shortRatio'),
            "52w_high": info.get('fiftyTwoWeekHigh'),
            "52w_low": info.get('fiftyTwoWeekLow'),
            "high_90d": high_90d,
            "low_90d": low_90d,
            "volatility_90d": volatility_90d,
            "returns_1m": returns_1m,
            "returns_3m": returns_3m,
            "exchange": info.get('exchange'),
            "sector": info.get('sector'),
            "industry": info.get('industry'),
            "collected_at": date.today().isoformat()
        }
    except ImportError as e:
        print(f"  ERROR: yfinance not installed: {e}")
        return None
    except (ValueError, TypeError, KeyError, AttributeError) as e:
        print(f"  ERROR: Data extraction failed for {ticker}: {e}")
        return None
    except Exception as e:
        # Network errors, API errors, etc. - log and continue
        print(f"  ERROR: Failed to fetch data for {ticker}: {type(e).__name__}: {e}")
        return None


def collect_all_market_data(universe_file: Path, output_file: Path):
    """Collect market data for all tickers"""
    
    print("="*80)
    print("MARKET DATA COLLECTION (Yahoo Finance)")
    print("="*80)
    print(f"Date: {date.today()}")
    
    # Check yfinance
    try:
        import yfinance as yf
        print("✅ yfinance library found")
    except ImportError:
        print("\n⚠️  Installing yfinance...")
        import subprocess
        subprocess.run(['pip', 'install', 'yfinance'], check=True)
        print("✅ Installed yfinance")
    
    # Load universe
    with open(universe_file) as f:
        universe = json.load(f)
    
    tickers = [s['ticker'] for s in universe if s.get('ticker') and s['ticker'] != '_XBI_BENCHMARK_']
    
    print(f"\nUniverse: {len(tickers)} tickers")
    print(f"Output: {output_file}")
    print(f"Estimated time: {len(tickers) * 0.3 / 60:.1f} minutes")
    
    # Collect
    all_data = []
    stats = {'total': len(tickers), 'successful': 0, 'failed': 0, 'no_data': []}
    
    print(f"\n{'='*80}")
    print("COLLECTING MARKET DATA")
    print(f"{'='*80}\n")
    
    for i, ticker in enumerate(tickers, 1):
        print(f"[{i:3d}/{len(tickers)}] {ticker:6s}", end=" ", flush=True)
        
        data = get_market_data(ticker)
        
        if data and data.get('price'):
            all_data.append(data)
            stats['successful'] += 1
            
            price = data['price']
            mcap = data.get('market_cap', 0)
            mcap_str = f"${mcap/1e9:.1f}B" if mcap and mcap > 1e9 else f"${mcap/1e6:.0f}M" if mcap else "N/A"
            
            print(f"✅ ${price:7.2f}  MCap: {mcap_str:>8s}")
        else:
            stats['failed'] += 1
            stats['no_data'].append(ticker)
            print("❌ No data")
        
        time.sleep(0.2)  # Rate limiting
        
        if i % 50 == 0:
            print(f"\n  Progress: {i}/{len(tickers)} ({i/len(tickers)*100:.1f}%), Success: {stats['successful']/i*100:.1f}%\n")
    
    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(all_data, f, indent=2)
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total: {stats['total']}, Successful: {stats['successful']}, Failed: {stats['failed']}")
    print(f"Coverage: {stats['successful'] / stats['total'] * 100:.1f}%")
    print(f"✅ Saved to: {output_file}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Collect market data from Yahoo Finance")
    parser.add_argument('--universe', type=Path, default=Path('production_data/universe.json'))
    parser.add_argument('--output', type=Path, default=Path('production_data/market_data.json'))
    args = parser.parse_args()
    
    if not args.universe.exists():
        print(f"❌ Universe file not found: {args.universe}")
        return 1
    
    try:
        collect_all_market_data(args.universe, args.output)
        return 0
    except KeyboardInterrupt:
        print("\n\n❌ Cancelled by user")
        return 1
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
