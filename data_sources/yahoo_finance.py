#!/usr/bin/env python3
"""
Yahoo Finance data fetcher - stdlib only.

Free, no API key required.

Fetches real-time market data including:
- Current price
- Volume
- Market cap
- Shares outstanding
- Beta

Uses the Yahoo Finance query1 API which is publicly accessible.

Author: Wake Robin Capital Management
Version: 1.0.0
"""

import urllib.request
import urllib.parse
import json
import ssl
from decimal import Decimal
from datetime import datetime
from typing import Dict, Optional, Any


# Module metadata
__version__ = "1.0.0"


def _create_ssl_context() -> ssl.SSLContext:
    """Create SSL context for HTTPS requests."""
    context = ssl.create_default_context()
    return context


def fetch_quote(ticker: str) -> Optional[Dict[str, Any]]:
    """
    Fetch current quote data for ticker.

    Uses Yahoo Finance chart API (free, public).

    Args:
        ticker: Stock ticker symbol (e.g., "ABBV", "NVAX")

    Returns:
        Dict containing:
        - ticker: str
        - price: Decimal
        - volume: int
        - market_cap: Decimal
        - currency: str
        - exchange: str
        - timestamp: int (Unix timestamp)
        - source: str

        Returns None if fetch fails.
    """
    try:
        # Yahoo Finance chart API (free, public)
        base_url = "https://query1.finance.yahoo.com/v8/finance/chart/"
        url = f"{base_url}{urllib.parse.quote(ticker)}?interval=1d&range=1d"

        # Create request with headers
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; WakeRobinBot/1.0)"
        }
        req = urllib.request.Request(url, headers=headers)

        with urllib.request.urlopen(req, timeout=10, context=_create_ssl_context()) as response:
            data = json.loads(response.read().decode('utf-8'))

        # Check for errors
        if "chart" not in data or "result" not in data["chart"]:
            print(f"  Yahoo Finance: Invalid response for {ticker}")
            return None

        result = data["chart"]["result"]
        if not result:
            print(f"  Yahoo Finance: No data for {ticker}")
            return None

        result = result[0]
        meta = result.get("meta", {})

        # Extract market data
        price = meta.get("regularMarketPrice")
        volume = meta.get("regularMarketVolume", 0)
        market_cap = meta.get("marketCap", 0)
        currency = meta.get("currency", "USD")
        exchange = meta.get("exchangeName", "")
        timestamp = meta.get("regularMarketTime", 0)

        if price is None:
            print(f"  Yahoo Finance: No price data for {ticker}")
            return None

        return {
            "ticker": ticker,
            "price": Decimal(str(price)),
            "volume": int(volume) if volume else 0,
            "market_cap": Decimal(str(market_cap)) if market_cap else Decimal("0"),
            "currency": currency,
            "exchange": exchange,
            "timestamp": timestamp,
            "collected_at": datetime.now().isoformat(),
            "source": "yahoo_finance"
        }

    except urllib.error.HTTPError as e:
        print(f"  Yahoo Finance HTTP error for {ticker}: {e.code}")
        return None
    except urllib.error.URLError as e:
        print(f"  Yahoo Finance URL error for {ticker}: {e.reason}")
        return None
    except json.JSONDecodeError as e:
        print(f"  Yahoo Finance JSON error for {ticker}: {e}")
        return None
    except Exception as e:
        print(f"  Yahoo Finance fetch failed for {ticker}: {e}")
        return None


def fetch_key_statistics(ticker: str) -> Optional[Dict[str, Any]]:
    """
    Fetch key statistics (shares outstanding, beta, etc.).

    Uses Yahoo Finance quoteSummary API.

    Args:
        ticker: Stock ticker symbol

    Returns:
        Dict containing key statistics, or None if unavailable.
    """
    try:
        base_url = "https://query1.finance.yahoo.com/v10/finance/quoteSummary/"
        modules = "defaultKeyStatistics,summaryDetail"
        url = f"{base_url}{urllib.parse.quote(ticker)}?modules={modules}"

        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; WakeRobinBot/1.0)"
        }
        req = urllib.request.Request(url, headers=headers)

        with urllib.request.urlopen(req, timeout=10, context=_create_ssl_context()) as response:
            data = json.loads(response.read().decode('utf-8'))

        result = data.get("quoteSummary", {}).get("result", [])
        if not result:
            return None

        result = result[0]
        stats = result.get("defaultKeyStatistics", {})
        detail = result.get("summaryDetail", {})

        return {
            "ticker": ticker,
            "shares_outstanding": _extract_raw(stats.get("sharesOutstanding")),
            "float_shares": _extract_raw(stats.get("floatShares")),
            "beta": _extract_raw(stats.get("beta")),
            "short_percent": _extract_raw(stats.get("shortPercentOfFloat")),
            "short_ratio": _extract_raw(stats.get("shortRatio")),
            "pe_ratio": _extract_raw(detail.get("trailingPE")),
            "market_cap": _extract_raw(detail.get("marketCap")),
            "collected_at": datetime.now().isoformat(),
            "source": "yahoo_finance"
        }

    except Exception as e:
        print(f"  Yahoo Finance key stats failed for {ticker}: {e}")
        return None


def _extract_raw(field: Optional[Dict]) -> Optional[Any]:
    """Extract raw value from Yahoo Finance field dict."""
    if field is None:
        return None
    if isinstance(field, dict):
        return field.get("raw")
    return field


def fetch_historical(ticker: str, days: int = 90) -> Optional[Dict[str, Any]]:
    """
    Fetch historical price data.

    Args:
        ticker: Stock ticker symbol
        days: Number of days of history to fetch

    Returns:
        Dict containing historical data including:
        - prices: List of daily closes
        - volumes: List of daily volumes
        - high_90d: 90-day high
        - low_90d: 90-day low
        - volatility: Annualized volatility
    """
    try:
        base_url = "https://query1.finance.yahoo.com/v8/finance/chart/"
        url = f"{base_url}{urllib.parse.quote(ticker)}?interval=1d&range={days}d"

        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; WakeRobinBot/1.0)"
        }
        req = urllib.request.Request(url, headers=headers)

        with urllib.request.urlopen(req, timeout=15, context=_create_ssl_context()) as response:
            data = json.loads(response.read().decode('utf-8'))

        result = data["chart"]["result"][0]
        indicators = result.get("indicators", {}).get("quote", [{}])[0]

        closes = indicators.get("close", [])
        highs = indicators.get("high", [])
        lows = indicators.get("low", [])
        volumes = indicators.get("volume", [])

        # Filter out None values
        closes = [c for c in closes if c is not None]
        highs = [h for h in highs if h is not None]
        lows = [l for l in lows if l is not None]
        volumes = [v for v in volumes if v is not None]

        if not closes:
            return None

        # Calculate metrics
        high_90d = max(highs) if highs else None
        low_90d = min(lows) if lows else None

        # Calculate volatility (annualized std dev of returns)
        volatility = None
        if len(closes) > 1:
            returns = [(closes[i] - closes[i-1]) / closes[i-1]
                       for i in range(1, len(closes)) if closes[i-1] != 0]
            if returns:
                import math
                mean_return = sum(returns) / len(returns)
                variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
                daily_vol = math.sqrt(variance)
                volatility = daily_vol * math.sqrt(252)  # Annualize

        return {
            "ticker": ticker,
            "prices": closes,
            "volumes": volumes,
            "high_90d": high_90d,
            "low_90d": low_90d,
            "volatility": volatility,
            "data_points": len(closes),
            "collected_at": datetime.now().isoformat(),
            "source": "yahoo_finance"
        }

    except Exception as e:
        print(f"  Yahoo Finance historical failed for {ticker}: {e}")
        return None


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("YAHOO FINANCE DATA FETCHER TEST")
    print("=" * 60)

    # Test with a biotech stock
    test_ticker = "ABBV"

    print(f"\nFetching quote for {test_ticker}...")
    quote = fetch_quote(test_ticker)
    if quote:
        print(f"  Price: ${quote['price']}")
        print(f"  Volume: {quote['volume']:,}")
        print(f"  Market Cap: ${quote['market_cap']:,.0f}")
        print(f"  Exchange: {quote['exchange']}")
    else:
        print("  Failed to fetch quote")

    print(f"\nFetching key statistics for {test_ticker}...")
    stats = fetch_key_statistics(test_ticker)
    if stats:
        print(f"  Beta: {stats.get('beta')}")
        print(f"  Short %: {stats.get('short_percent')}")
        print(f"  P/E: {stats.get('pe_ratio')}")
    else:
        print("  Failed to fetch key statistics")

    print(f"\nFetching historical data for {test_ticker}...")
    hist = fetch_historical(test_ticker, days=30)
    if hist:
        print(f"  Data points: {hist['data_points']}")
        print(f"  90-day high: ${hist['high_90d']:.2f}" if hist['high_90d'] else "  90-day high: N/A")
        print(f"  90-day low: ${hist['low_90d']:.2f}" if hist['low_90d'] else "  90-day low: N/A")
        print(f"  Volatility: {hist['volatility']*100:.1f}%" if hist['volatility'] else "  Volatility: N/A")
    else:
        print("  Failed to fetch historical data")
