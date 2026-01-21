#!/usr/bin/env python3
"""
collect_market_data.py - Collect Market Data from Yahoo Finance

Fetches price, volume, market cap, and other market data for all tickers in universe.
Handles network/proxy issues gracefully by falling back to cached data.

Usage:
    python collect_market_data.py --universe production_data/universe.json
    python collect_market_data.py --use-cache  # Use existing data without fetching
    python collect_market_data.py --test-connection  # Test if Yahoo Finance is reachable
"""

import json
import time
from pathlib import Path
from datetime import date, datetime, timedelta
from typing import Dict, Optional, Tuple
import argparse


def test_network_connectivity() -> Tuple[bool, str]:
    """
    Test if Yahoo Finance API is reachable.
    Returns (is_reachable, error_message).
    """
    try:
        import yfinance as yf
        # Try to fetch a well-known ticker
        stock = yf.Ticker("AAPL")
        hist = stock.history(period="1d")
        if len(hist) > 0:
            return True, ""
        # If no history, try info endpoint
        info = stock.info
        if info and ('currentPrice' in info or 'regularMarketPrice' in info):
            return True, ""
        return False, "Yahoo Finance returned empty data - may be rate limited or blocked"
    except Exception as e:
        error_msg = str(e)
        if "ProxyError" in error_msg or "403 Forbidden" in error_msg:
            return False, "Network proxy blocking Yahoo Finance API (fc.yahoo.com not in allowed hosts)"
        if "Max retries exceeded" in error_msg:
            return False, "Network timeout - Yahoo Finance API unreachable"
        if "No module named 'yfinance'" in error_msg:
            return False, "yfinance not installed - run: pip install yfinance"
        return False, f"Connection error: {error_msg[:100]}"


def load_cached_market_data(cache_file: Path) -> Tuple[list, str]:
    """
    Load cached market data if available.
    Returns (data_list, collection_date).
    """
    if not cache_file.exists():
        return [], ""
    try:
        with open(cache_file) as f:
            data = json.load(f)
        if not data:
            return [], ""
        # Get collection date from first record
        collection_date = data[0].get("collected_at", "unknown")
        return data, collection_date
    except Exception:
        return [], ""


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
    except Exception as e:
        return None


def collect_all_market_data(universe_file: Path, output_file: Path, use_cache: bool = False, force_refresh: bool = False):
    """Collect market data for all tickers with graceful fallback to cached data"""

    print("="*80)
    print("MARKET DATA COLLECTION (Yahoo Finance)")
    print("="*80)
    print(f"Date: {date.today()}")

    # Check for cached data
    cached_data, cache_date = load_cached_market_data(output_file)
    cache_available = len(cached_data) > 0
    if cache_available:
        cache_age = (date.today() - date.fromisoformat(cache_date)).days if cache_date and cache_date != "unknown" else -1
        print(f"📦 Cached data: {len(cached_data)} records from {cache_date} ({cache_age} days old)")
    else:
        print("📦 No cached data available")

    # If use_cache flag is set, skip network collection
    if use_cache:
        if cache_available:
            print("\n✅ Using cached data (--use-cache flag set)")
            print(f"{'='*80}\n")
            return
        else:
            print("\n❌ Cannot use cache - no cached data found")
            return

    # Check yfinance
    try:
        import yfinance as yf
        print("✅ yfinance library found")
    except ImportError:
        print("\n❌ yfinance not installed")
        print("   Install with: pip install yfinance")
        if cache_available:
            print(f"\n⚠️  Using cached data from {cache_date} (yfinance unavailable)")
            return
        else:
            raise ImportError("yfinance not installed and no cached data available")

    # Test network connectivity
    print("\n🔌 Testing network connectivity...")
    is_reachable, error_msg = test_network_connectivity()

    if not is_reachable:
        print(f"❌ Network test failed: {error_msg}")
        if cache_available:
            print(f"\n⚠️  FALLING BACK TO CACHED DATA")
            print(f"   Cache date: {cache_date}")
            print(f"   Cache age: {cache_age} days")
            if cache_age > 7:
                print(f"   ⚠️  WARNING: Cache is stale (>{7} days old)")
            print(f"\n   To force refresh when network is available, run:")
            print(f"   python collect_market_data.py --force-refresh")
            print(f"{'='*80}\n")
            return
        else:
            raise ConnectionError(f"Cannot fetch market data: {error_msg}")

    print("✅ Yahoo Finance API reachable")

    # Load universe
    with open(universe_file) as f:
        universe = json.load(f)

    tickers = [s['ticker'] for s in universe if s.get('ticker') and s['ticker'] != '_XBI_BENCHMARK_']

    print(f"\nUniverse: {len(tickers)} tickers")
    print(f"Output: {output_file}")
    print(f"Estimated time: {len(tickers) * 0.3 / 60:.1f} minutes")

    # Collect
    all_data = []
    stats = {'total': len(tickers), 'successful': 0, 'failed': 0, 'no_data': [], 'network_errors': 0}

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

        # Check for network issues after first 5 tickers
        if i == 5 and stats['successful'] == 0:
            print("\n⚠️  First 5 tickers all failed - likely network issue")
            if cache_available:
                print(f"   Falling back to cached data from {cache_date}")
                return

    # Check if collection was mostly failures (likely network issue mid-run)
    if stats['total'] > 0 and stats['successful'] / stats['total'] < 0.1:
        print(f"\n⚠️  Very low success rate ({stats['successful']}/{stats['total']})")
        if cache_available and not force_refresh:
            print(f"   Using cached data from {cache_date} instead")
            return

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
    parser.add_argument('--use-cache', action='store_true',
                       help='Use existing cached data without attempting network fetch')
    parser.add_argument('--force-refresh', action='store_true',
                       help='Force refresh even if network appears unavailable')
    parser.add_argument('--test-connection', action='store_true',
                       help='Test Yahoo Finance connectivity and exit')
    args = parser.parse_args()

    # Handle test-connection mode
    if args.test_connection:
        print("="*80)
        print("TESTING YAHOO FINANCE CONNECTIVITY")
        print("="*80)

        try:
            import yfinance as yf
            print("✅ yfinance library found")
        except ImportError:
            print("❌ yfinance not installed")
            print("   Install with: pip install yfinance")
            return 1

        is_reachable, error_msg = test_network_connectivity()
        if is_reachable:
            print("✅ Yahoo Finance API is reachable")
            return 0
        else:
            print(f"❌ Yahoo Finance API is NOT reachable")
            print(f"   Reason: {error_msg}")
            print("\n   Possible solutions:")
            print("   1. Check your network/proxy configuration")
            print("   2. Ensure fc.yahoo.com and query1.finance.yahoo.com are accessible")
            print("   3. Use --use-cache to use existing cached market data")
            return 1

    if not args.universe.exists():
        print(f"❌ Universe file not found: {args.universe}")
        return 1

    try:
        collect_all_market_data(
            args.universe,
            args.output,
            use_cache=args.use_cache,
            force_refresh=args.force_refresh
        )
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
