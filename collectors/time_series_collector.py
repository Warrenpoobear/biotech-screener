"""
time_series_collector.py - Collect historical price/return series for time-series analysis
Uses market_data_provider.py for PIT-safe data with caching
"""
import json
import time
from pathlib import Path
from datetime import datetime, date
from typing import Dict, List

# Import the market data provider (in parent directory)
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from market_data_provider import PriceDataProvider, BatchPriceProvider


def collect_time_series_data(
    ticker: str, 
    as_of: date = None,
    lookback_days: int = 365
) -> dict:
    """
    Collect historical price/return series for a single ticker.
    
    Args:
        ticker: Stock ticker symbol
        as_of: Point-in-time date (defaults to today)
        lookback_days: How far back to fetch (default 365)
    
    Returns:
        Dict with prices, returns, volumes and metadata
    """
    if as_of is None:
        as_of = date.today()
    
    try:
        provider = PriceDataProvider()
        
        # Get complete data package
        ticker_data = provider.get_ticker_data(ticker, as_of, lookback_days)
        
        if ticker_data["num_days"] == 0:
            return {
                "ticker": ticker,
                "success": False,
                "error": "No historical data available",
                "timestamp": datetime.now().isoformat()
            }
        
        # Get ADV for liquidity gate
        adv = provider.get_adv(ticker, as_of, window=20)
        
        # Convert to JSON-safe format
        data = {
            "ticker": ticker,
            "success": True,
            "as_of_date": as_of.isoformat(),
            "time_series": {
                "prices": [float(p) for p in ticker_data["prices"]],
                "returns": ticker_data["returns"],  # Already floats
                "volumes": ticker_data["volumes"],
                "num_days": ticker_data["num_days"],
                "lookback_days": lookback_days
            },
            "liquidity": {
                "adv_20d": float(adv) if adv else None,
                "adv_20d_formatted": f"${float(adv):,.0f}" if adv else None
            },
            "provenance": {
                "source": "Yahoo Finance via yfinance (historical)",
                "timestamp": datetime.now().isoformat(),
                "method": "market_data_provider",
                "pit_safe": True,
                "cached": True  # Provider uses 24h cache
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


def collect_batch(
    tickers: List[str],
    as_of: date = None,
    lookback_days: int = 365,
    delay_seconds: float = 0.1  # Faster due to caching
) -> Dict[str, dict]:
    """
    Collect time-series data for multiple tickers efficiently.
    
    Also fetches XBI benchmark for correlation/beta calculations.
    """
    if as_of is None:
        as_of = date.today()
    
    results = {}
    total = len(tickers)
    
    print(f"\n📈 Collecting time-series data for {total} tickers...")
    print(f"   As-of date: {as_of.isoformat()}")
    print(f"   Lookback: {lookback_days} days")
    
    # Use batch provider for efficiency
    batch_provider = BatchPriceProvider()
    
    try:
        # Fetch all tickers + XBI in one batch
        market_data = batch_provider.get_batch_data(
            tickers=tickers,
            as_of=as_of,
            lookback_days=lookback_days,
            include_xbi=True
        )
        
        # Convert to standard format
        for ticker in tickers:
            if ticker in market_data:
                ticker_data = market_data[ticker]
                
                # Get ADV
                provider = PriceDataProvider()
                adv = provider.get_adv(ticker, as_of, window=20)
                
                results[ticker] = {
                    "ticker": ticker,
                    "success": True,
                    "as_of_date": as_of.isoformat(),
                    "time_series": {
                        "prices": [float(p) for p in ticker_data["prices"]],
                        "returns": ticker_data["returns"],
                        "volumes": ticker_data["volumes"],
                        "num_days": ticker_data["num_days"],
                        "lookback_days": lookback_days
                    },
                    "liquidity": {
                        "adv_20d": float(adv) if adv else None,
                        "adv_20d_formatted": f"${float(adv):,.0f}" if adv else None
                    },
                    "provenance": {
                        "source": "Yahoo Finance via yfinance (historical)",
                        "timestamp": datetime.now().isoformat(),
                        "method": "market_data_provider",
                        "pit_safe": True,
                        "cached": True
                    }
                }
                
                # Progress indicator
                days = ticker_data["num_days"]
                print(f"✓ {ticker}: {days} days of data")
            else:
                results[ticker] = {
                    "ticker": ticker,
                    "success": False,
                    "error": "No data available",
                    "timestamp": datetime.now().isoformat()
                }
                print(f"✗ {ticker}: No data")
        
        # Store XBI benchmark separately (available to all tickers)
        if "_xbi_" in market_data:
            xbi_data = market_data["_xbi_"]
            results["_XBI_BENCHMARK_"] = {
                "ticker": "XBI",
                "success": True,
                "as_of_date": as_of.isoformat(),
                "time_series": {
                    "prices": [float(p) for p in xbi_data["prices"]],
                    "returns": xbi_data["returns"],
                    "volumes": xbi_data.get("volumes", []),
                    "num_days": xbi_data["num_days"],
                    "lookback_days": lookback_days
                },
                "provenance": {
                    "source": "Yahoo Finance (XBI ETF)",
                    "timestamp": datetime.now().isoformat(),
                    "purpose": "Benchmark for correlation/beta/regime detection"
                }
            }
            print(f"✓ XBI benchmark: {xbi_data['num_days']} days")
    
    except Exception as e:
        print(f"✗ Batch collection error: {e}")
        # Fall back to individual collection
        for ticker in tickers:
            results[ticker] = collect_time_series_data(ticker, as_of, lookback_days)
            time.sleep(delay_seconds)
    
    successful = sum(1 for d in results.values() 
                    if d.get('success') and d['ticker'] != 'XBI')
    print(f"\n✓ Successfully collected time-series data for {successful}/{total} tickers")
    
    return results


if __name__ == "__main__":
    # Test with a single ticker
    test_ticker = "VRTX"
    print(f"Testing time-series collector with {test_ticker}...")
    
    data = collect_time_series_data(test_ticker, as_of=date(2024, 12, 31))
    
    if data.get('success'):
        print(f"\n✓ Success!")
        print(f"  - Days of data: {data['time_series']['num_days']}")
        print(f"  - First price: ${data['time_series']['prices'][0]:.2f}")
        print(f"  - Last price: ${data['time_series']['prices'][-1]:.2f}")
        print(f"  - ADV (20d): {data['liquidity']['adv_20d_formatted']}")
    else:
        print(f"\n✗ Failed: {data.get('error')}")
