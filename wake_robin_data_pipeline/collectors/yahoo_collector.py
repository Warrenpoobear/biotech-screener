"""
yahoo_collector.py - Collect market data from Yahoo Finance
Free, no API key required. Rate: reasonable (1 req/sec safe)

Enhanced to include balance sheet data for supplementing SEC financials.
"""
import json
import logging
import time
from pathlib import Path
from datetime import datetime, timedelta
import hashlib
import pandas as pd

logger = logging.getLogger(__name__)


def extract_balance_sheet_data(stock) -> dict:
    """
    Extract balance sheet data from yfinance quarterly_balance_sheet.

    Returns dict with:
    - total_assets, total_liabilities, stockholders_equity
    - cash, total_debt, long_term_debt
    - period_date (the fiscal period end date)
    """
    result = {
        "success": False,
        "total_assets": None,
        "total_liabilities": None,
        "stockholders_equity": None,
        "cash": None,
        "marketable_securities": None,
        "total_liquidity": None,
        "total_debt": None,
        "long_term_debt": None,
        "current_debt": None,
        "period_date": None,
        "error": None
    }

    try:
        bs = stock.quarterly_balance_sheet

        if bs is None or bs.empty:
            result["error"] = "no_balance_sheet_data"
            return result

        # Get most recent column (most recent quarter)
        latest_col = bs.columns[0]
        period_date = latest_col.strftime('%Y-%m-%d') if hasattr(latest_col, 'strftime') else str(latest_col)
        result["period_date"] = period_date

        # Helper to safely extract a value
        def get_value(keys):
            """Try multiple possible row names to find the value."""
            if isinstance(keys, str):
                keys = [keys]
            for key in keys:
                if key in bs.index:
                    val = bs.loc[key, latest_col]
                    if pd.notna(val):
                        return float(val)
            return None

        # Extract key balance sheet items (using exact yfinance field names)
        result["total_assets"] = get_value(['Total Assets'])

        result["total_liabilities"] = get_value([
            'Total Liabilities Net Minority Interest',
            'Current Liabilities'  # Fallback to current only
        ])

        result["stockholders_equity"] = get_value([
            'Stockholders Equity',
            'Common Stock Equity',
            'Total Equity Gross Minority Interest'
        ])

        # Cash only (not including short-term investments)
        result["cash"] = get_value(['Cash And Cash Equivalents'])

        # Total liquidity (cash + short-term investments combined field)
        result["total_liquidity"] = get_value(['Cash Cash Equivalents And Short Term Investments'])

        # If no combined field, use cash as total liquidity
        if result["total_liquidity"] is None and result["cash"] is not None:
            result["total_liquidity"] = result["cash"]

        # Calculate marketable securities as difference if both fields exist
        if result["total_liquidity"] is not None and result["cash"] is not None:
            ms = result["total_liquidity"] - result["cash"]
            if ms > 0:
                result["marketable_securities"] = ms

        result["total_debt"] = get_value([
            'Total Debt',
            'TotalDebt'
        ])

        result["long_term_debt"] = get_value([
            'Long Term Debt',
            'Long Term Debt And Capital Lease Obligation',
            'LongTermDebt'
        ])

        result["current_debt"] = get_value([
            'Current Debt',
            'Current Debt And Capital Lease Obligation',
            'CurrentDebt'
        ])

        # Mark success if we got at least some core data
        if result["total_assets"] or result["total_liabilities"] or result["cash"]:
            result["success"] = True
        else:
            result["error"] = "no_usable_data_extracted"

    except Exception as e:
        result["error"] = str(e)
        logger.warning(f"Failed to extract balance sheet: {e}")

    return result


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

        # Get balance sheet data
        balance_sheet = extract_balance_sheet_data(stock)

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
        if not balance_sheet.get('success'):
            flags.append(f"balance_sheet_error:{balance_sheet.get('error', 'unknown')}")

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
            "balance_sheet": {
                "success": balance_sheet.get('success', False),
                "period_date": balance_sheet.get('period_date'),
                "total_assets": balance_sheet.get('total_assets'),
                "total_liabilities": balance_sheet.get('total_liabilities'),
                "stockholders_equity": balance_sheet.get('stockholders_equity'),
                "cash": balance_sheet.get('cash'),
                "marketable_securities": balance_sheet.get('marketable_securities'),
                "total_liquidity": balance_sheet.get('total_liquidity'),
                "total_debt": balance_sheet.get('total_debt'),
                "long_term_debt": balance_sheet.get('long_term_debt'),
                "current_debt": balance_sheet.get('current_debt'),
                "error": balance_sheet.get('error')
            },
            "data_quality": {
                "flags": flags,
                "has_price": price > 0,
                "has_market_cap": market_cap > 0,
                "has_volume": avg_volume > 0,
                "has_balance_sheet": balance_sheet.get('success', False),
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

def collect_batch(tickers: list, delay_seconds: float = 1.0, force_refresh: bool = False) -> dict:
    results = {}
    total = len(tickers)
    bs_success = 0

    print(f"\n📊 Collecting Yahoo Finance data for {total} tickers...")
    if force_refresh:
        print("   (force_refresh=True, bypassing cache)")

    for i, ticker in enumerate(tickers, 1):
        print(f"[{i}/{total}] Fetching {ticker}...", end=" ")

        data = collect_yahoo_data(ticker, force_refresh=force_refresh)
        results[ticker] = data

        if data.get('success'):
            price = data['price']['current']
            mcap = data['market_cap']['value'] / 1e9
            cached = " (cached)" if data.get('from_cache') else ""
            bs_ok = "✓BS" if data.get('balance_sheet', {}).get('success') else ""
            if data.get('balance_sheet', {}).get('success'):
                bs_success += 1
            print(f"✓ ${price:.2f}, MCap: ${mcap:.2f}B {bs_ok}{cached}")
        else:
            print(f"✗ {data.get('error', 'Unknown error')}")

        if i < total and not data.get('from_cache'):
            time.sleep(delay_seconds)

    successful = sum(1 for d in results.values() if d.get('success'))
    print(f"\n✓ Successfully collected data for {successful}/{total} tickers")
    print(f"✓ Balance sheet data available for {bs_success}/{total} tickers")

    return results

if __name__ == "__main__":
    test_ticker = "VRTX"
    print(f"Testing Yahoo Finance collector with {test_ticker}...")
    data = collect_yahoo_data(test_ticker, force_refresh=True)
    print(json.dumps(data, indent=2))
