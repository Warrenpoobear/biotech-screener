"""
yahoo_collector.py - Collect market data from Yahoo Finance
Free, no API key required. Rate: reasonable (1 req/sec safe)
"""
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from decimal import Decimal
import hashlib

def get_cache_path(ticker: str) -> Path:
    """Get cache file path for ticker."""
    cache_dir = Path(__file__).parent.parent / "cache" / "yahoo"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{ticker}.json"

def is_cache_valid(cache_path: Path, max_age_hours: int = 24) -> bool:
    """Check if cache is fresh enough."""
    if not cache_path.exists():
        return False
    
    age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
    return age < timedelta(hours=max_age_hours)

def fetch_yahoo_data(ticker: str) -> dict:
    """
    Fetch data from Yahoo Finance using yfinance library.
    Returns structured data with provenance.
    """
    try:
        import yfinance as yf
    except ImportError:
        return {
            "ticker": ticker,
            "success": False,
            "error": "yfinance not installed (pip install yfinance)",
            "timestamp": datetime.now().isoformat()
        }
    
    try:
        stock = yf.Ticker(ticker)
        
        # Try fast_info first (more reliable)
        try:
            fast_info = stock.fast_info
            price = float(fast_info.get('last_price', 0))
            market_cap = float(fast_info.get('market_cap', 0))
            shares = float(fast_info.get('shares', 0))
        except:
            # Fallback to info dict
            info = stock.info
            price = float(info.get('currentPrice', info.get('regularMarketPrice', 0)))
            market_cap = float(info.get('marketCap', 0))
            shares = float(info.get('sharesOutstanding', 0))
            fast_info = None
        
        # Always get info for additional fields
        info = stock.info
        
        # Get historical data for volume metrics
        hist = stock.history(period="1mo")
        avg_volume = int(hist['Volume'].mean()) if not hist.empty else 0
        
        data = {
            "ticker": ticker,
            "success": True,
            "price": {
                "current": price,
                "currency": "USD",
                "day_high": float(info.get('dayHigh', 0)),
                "day_low": float(info.get('dayLow', 0)),
                "52_week_high": float(info.get('fiftyTwoWeekHigh', 0)),
                "52_week_low": float(info.get('fiftyTwoWeekLow', 0))
            },
            "market_cap": {
                "value": market_cap,
                "currency": "USD"
            },
            "shares_outstanding": shares,
            "volume": {
                "last": int(info.get('volume', 0)),
                "average_30d": avg_volume
            },
            "valuation": {
                "pe_ratio": float(info.get('trailingPE', 0)) if info.get('trailingPE') else None,
                "forward_pe": float(info.get('forwardPE', 0)) if info.get('forwardPE') else None,
                "price_to_book": float(info.get('priceToBook', 0)) if info.get('priceToBook') else None
            },
            "company_info": {
                "name": info.get('longName', info.get('shortName', '')),
                "sector": info.get('sector', ''),
                "industry": info.get('industry', '')
            },
            "provenance": {
                "source": "Yahoo Finance via yfinance",
                "timestamp": datetime.now().isoformat(),
                "url": f"https://finance.yahoo.com/quote/{ticker}",
                "data_hash": hashlib.sha256(json.dumps({
                    "price": price,
                    "market_cap": market_cap
                }).encode()).hexdigest()[:16]
            }
        }
        
        return data
        
    except Exception as e:
        return {
            "ticker": ticker,
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

def collect_yahoo_data(ticker: str, force_refresh: bool = False) -> dict:
    """
    Main entry point: collect Yahoo Finance data with caching.
    
    Args:
        ticker: Stock ticker symbol
        force_refresh: Bypass cache and fetch fresh data
        
    Returns:
        dict with market data or error info
    """
    cache_path = get_cache_path(ticker)
    
    # Check cache first
    if not force_refresh and is_cache_valid(cache_path):
        with open(cache_path) as f:
            cached = json.load(f)
            cached['from_cache'] = True
            return cached
    
    # Fetch fresh data
    data = fetch_yahoo_data(ticker)
    
    # Cache successful results
    if data.get('success'):
        with open(cache_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    data['from_cache'] = False
    return data

def collect_batch(tickers: list[str], delay_seconds: float = 1.0) -> dict:
    """
    Collect data for multiple tickers with rate limiting.
    
    Args:
        tickers: List of ticker symbols
        delay_seconds: Delay between requests (respect rate limits)
        
    Returns:
        dict mapping ticker to data
    """
    results = {}
    total = len(tickers)
    
    print(f"\n📊 Collecting Yahoo Finance data for {total} tickers...")
    
    for i, ticker in enumerate(tickers, 1):
        print(f"[{i}/{total}] Fetching {ticker}...", end=" ")
        
        data = collect_yahoo_data(ticker)
        results[ticker] = data
        
        if data.get('success'):
            price = data['price']['current']
            mcap = data['market_cap']['value'] / 1e9  # Convert to billions
            cached = " (cached)" if data.get('from_cache') else ""
            print(f"✓ ${price:.2f}, MCap: ${mcap:.2f}B{cached}")
        else:
            print(f"✗ {data.get('error', 'Unknown error')}")
        
        # Rate limiting (except on last iteration)
        if i < total and not data.get('from_cache'):
            time.sleep(delay_seconds)
    
    # Summary
    successful = sum(1 for d in results.values() if d.get('success'))
    print(f"\n✓ Successfully collected data for {successful}/{total} tickers")
    
    return results

if __name__ == "__main__":
    # Test with a single ticker
    test_ticker = "VRTX"
    print(f"Testing Yahoo Finance collector with {test_ticker}...")
    
    data = collect_yahoo_data(test_ticker, force_refresh=True)
    print(json.dumps(data, indent=2))
