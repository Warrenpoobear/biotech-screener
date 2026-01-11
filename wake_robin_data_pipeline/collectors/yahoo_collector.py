"""
yahoo_collector.py - Collect market data from Yahoo Finance
Free, no API key required. Rate: reasonable (1 req/sec safe)
"""
import json
import logging
import time
from pathlib import Path
from datetime import datetime, timedelta
import hashlib

logger = logging.getLogger(__name__)


def get_cache_path(ticker: str) -> Path:
    cache_dir = Path(__file__).parent.parent / "cache" / "yahoo"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{ticker}.json"

def is_cache_valid(cache_path: Path, max_age_hours: int = 24) -> bool:
    if not cache_path.exists():
        return False
    age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
    return age < timedelta(hours=max_age_hours)

def fetch_yahoo_data(ticker: str) -> dict:
    try:
        import yfinance as yf
    except ImportError:
        logger.error("yfinance not installed - run: pip install yfinance")
        return {
            "ticker": ticker,
            "success": False,
            "error": "yfinance not installed",
            "timestamp": datetime.now().isoformat()
        }

    try:
        stock = yf.Ticker(ticker)
        info = stock.info  # Use info dict directly (more reliable)

        # Get historical data for volume
        avg_volume = 0
        volume_error = None
        try:
            hist = stock.history(period="1mo")
            if not hist.empty:
                avg_volume = int(hist['Volume'].mean())
            else:
                volume_error = "empty_history"
        except Exception as e:
            volume_error = str(e)
            logger.warning(f"Failed to get volume history for {ticker}: {e}")

        # Extract data (prioritize info dict)
        price = float(info.get('currentPrice') or info.get('regularMarketPrice') or 0)
        market_cap = float(info.get('marketCap') or 0)
        shares = float(info.get('sharesOutstanding') or 0)

        # Build flags for data quality tracking
        flags = []
        if price == 0:
            flags.append("missing_price")
        if market_cap == 0:
            flags.append("missing_market_cap")
        if avg_volume == 0:
            flags.append("missing_volume")
        if volume_error:
            flags.append(f"volume_error:{volume_error}")

        data = {
            "ticker": ticker,
            "success": True,
            "price": {
                "current": price,
                "currency": "USD",
                "day_high": float(info.get('dayHigh') or 0),
                "day_low": float(info.get('dayLow') or 0),
                "52_week_high": float(info.get('fiftyTwoWeekHigh') or 0),
                "52_week_low": float(info.get('fiftyTwoWeekLow') or 0)
            },
            "market_cap": {
                "value": market_cap,
                "currency": "USD"
            },
            "shares_outstanding": shares,
            "volume": {
                "last": int(info.get('volume') or 0),
                "average_30d": avg_volume
            },
            "valuation": {
                "pe_ratio": float(info.get('trailingPE')) if info.get('trailingPE') else None,
                "forward_pe": float(info.get('forwardPE')) if info.get('forwardPE') else None,
                "price_to_book": float(info.get('priceToBook')) if info.get('priceToBook') else None
            },
            "company_info": {
                "name": info.get('longName', info.get('shortName', '')),
                "sector": info.get('sector', ''),
                "industry": info.get('industry', '')
            },
            "data_quality": {
                "flags": flags,
                "has_price": price > 0,
                "has_market_cap": market_cap > 0,
                "has_volume": avg_volume > 0,
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
        logger.error(f"Failed to fetch Yahoo data for {ticker}: {e}")
        return {
            "ticker": ticker,
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

def collect_yahoo_data(ticker: str, force_refresh: bool = False) -> dict:
    cache_path = get_cache_path(ticker)
    
    if not force_refresh and is_cache_valid(cache_path):
        with open(cache_path) as f:
            cached = json.load(f)
            cached['from_cache'] = True
            return cached
    
    data = fetch_yahoo_data(ticker)
    
    if data.get('success'):
        with open(cache_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    data['from_cache'] = False
    return data

def collect_batch(tickers: list, delay_seconds: float = 1.0) -> dict:
    results = {}
    total = len(tickers)
    
    print(f"\n📊 Collecting Yahoo Finance data for {total} tickers...")
    
    for i, ticker in enumerate(tickers, 1):
        print(f"[{i}/{total}] Fetching {ticker}...", end=" ")
        
        data = collect_yahoo_data(ticker)
        results[ticker] = data
        
        if data.get('success'):
            price = data['price']['current']
            mcap = data['market_cap']['value'] / 1e9
            cached = " (cached)" if data.get('from_cache') else ""
            print(f"✓ ${price:.2f}, MCap: ${mcap:.2f}B{cached}")
        else:
            print(f"✗ {data.get('error', 'Unknown error')}")
        
        if i < total and not data.get('from_cache'):
            time.sleep(delay_seconds)
    
    successful = sum(1 for d in results.values() if d.get('success'))
    print(f"\n✓ Successfully collected data for {successful}/{total} tickers")
    
    return results

if __name__ == "__main__":
    test_ticker = "VRTX"
    print(f"Testing Yahoo Finance collector with {test_ticker}...")
    data = collect_yahoo_data(test_ticker, force_refresh=True)
    print(json.dumps(data, indent=2))
