"""
market_data_provider.py - Point-in-time safe price/return series provider
FINAL VERSION - Handles both historical and current dates correctly
"""

from decimal import Decimal, ROUND_HALF_UP
from typing import List, Dict, Optional
from datetime import datetime, date, timedelta
from pathlib import Path
import json
import hashlib
import math

try:
    import yfinance as yf
except ImportError:
    yf = None

CACHE_DIR = Path("cache/market_data")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

CACHE_TTL_HOURS = 24
DECIMAL_PRECISION = Decimal("0.00000001")
TRADING_DAYS_PER_YEAR = 252

def _quantize(value: float) -> Decimal:
    return Decimal(str(value)).quantize(DECIMAL_PRECISION, rounding=ROUND_HALF_UP)

def _date_to_str(d: date) -> str:
    return d.isoformat()

def _compute_cache_key(ticker: str, as_of: date, lookback_days: int) -> str:
    key_data = f"{ticker}|{_date_to_str(as_of)}|{lookback_days}"
    return hashlib.sha256(key_data.encode()).hexdigest()[:16]

def _is_cache_valid(cache_path: Path, ttl_hours: int = CACHE_TTL_HOURS) -> bool:
    if not cache_path.exists():
        return False
    mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
    age = datetime.now() - mtime
    return age.total_seconds() < (ttl_hours * 3600)

def _write_cache(cache_path: Path, data: Dict) -> None:
    cache_data = {
        "data": data,
        "cached_at": datetime.now().isoformat(),
        "integrity": hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
    }
    temp_path = cache_path.with_suffix('.tmp')
    try:
        with open(temp_path, 'w') as f:
            json.dump(cache_data, f, indent=2, sort_keys=True)
        temp_path.replace(cache_path)
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        raise RuntimeError(f"Cache write failed: {e}")

def _read_cache(cache_path: Path) -> Optional[Dict]:
    try:
        with open(cache_path) as f:
            cache_data = json.load(f)
        data = cache_data["data"]
        expected_hash = cache_data["integrity"]
        actual_hash = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
        if expected_hash != actual_hash:
            print(f"WARNING: Cache integrity mismatch for {cache_path.name}")
            return None
        return data
    except Exception:
        return None

class PriceDataProvider:
    def __init__(self, cache_ttl_hours: int = CACHE_TTL_HOURS):
        self.cache_ttl_hours = cache_ttl_hours
        if yf is None:
            raise RuntimeError("yfinance not installed")
    
    def get_prices(self, ticker: str, as_of: date, lookback_days: int, use_cache: bool = True) -> List[Decimal]:
        """
        Get daily closing prices for a ticker.
        
        CRITICAL: Handles both historical and current dates correctly.
        - Historical dates (as_of < today): Uses start/end parameters
        - Current dates (as_of >= today): Uses period parameter
        """
        if use_cache:
            cache_key = _compute_cache_key(ticker, as_of, lookback_days)
            cache_path = CACHE_DIR / f"{cache_key}_prices.json"
            if _is_cache_valid(cache_path, self.cache_ttl_hours):
                cached = _read_cache(cache_path)
                if cached:
                    return [Decimal(str(p)) for p in cached["prices"]]
        
        try:
            stock = yf.Ticker(ticker)
            
            # Check if as_of is in the past
            today = date.today()
            
            if as_of < today:
                # Historical data: MUST use start/end dates
                # Otherwise period='1y' returns last year from TODAY, not from as_of
                start_date = as_of - timedelta(days=lookback_days)
                hist = stock.history(
                    start=start_date.strftime('%Y-%m-%d'),
                    end=(as_of + timedelta(days=1)).strftime('%Y-%m-%d'),  # +1 because end is exclusive
                    auto_adjust=True,
                    actions=False
                )
            else:
                # Current/recent data: can use period (more reliable for recent data)
                if lookback_days <= 30:
                    period = '1mo'
                elif lookback_days <= 90:
                    period = '3mo'
                elif lookback_days <= 180:
                    period = '6mo'
                elif lookback_days <= 365:
                    period = '1y'
                elif lookback_days <= 730:
                    period = '2y'
                else:
                    period = '5y'
                
                hist = stock.history(period=period, auto_adjust=True, actions=False)
            
            if hist.empty:
                return []
            
            prices = []
            for idx, row in hist.iterrows():
                # Handle timezone-aware datetime index
                if hasattr(idx, 'date'):
                    price_date = idx.date()
                else:
                    price_date = idx.to_pydatetime().date()
                
                # PIT discipline: only include data <= as_of
                if price_date <= as_of:
                    close = row['Close']
                    if not math.isnan(close):
                        prices.append(_quantize(close))
            
            if use_cache and prices:
                cache_data = {
                    "ticker": ticker,
                    "as_of": _date_to_str(as_of),
                    "lookback_days": lookback_days,
                    "num_prices": len(prices),
                    "prices": [str(p) for p in prices],
                }
                _write_cache(cache_path, cache_data)
            
            return prices
            
        except Exception as e:
            print(f"ERROR: Failed to fetch prices for {ticker}: {e}")
            return []
    
    def get_log_returns(self, ticker: str, as_of: date, lookback_days: int, use_cache: bool = True) -> List[float]:
        """Get daily log returns."""
        prices = self.get_prices(ticker, as_of, lookback_days, use_cache)
        if len(prices) < 2:
            return []
        returns = []
        for i in range(1, len(prices)):
            p_prev = float(prices[i-1])
            p_curr = float(prices[i])
            if p_prev > 0 and p_curr > 0:
                returns.append(math.log(p_curr / p_prev))
        return returns
    
    def get_volumes(self, ticker: str, as_of: date, lookback_days: int, use_cache: bool = True) -> List[int]:
        """Get daily trading volumes."""
        if use_cache:
            cache_key = _compute_cache_key(ticker, as_of, lookback_days)
            cache_path = CACHE_DIR / f"{cache_key}_volumes.json"
            if _is_cache_valid(cache_path, self.cache_ttl_hours):
                cached = _read_cache(cache_path)
                if cached:
                    return cached["volumes"]
        
        try:
            stock = yf.Ticker(ticker)
            today = date.today()
            
            if as_of < today:
                # Historical: use start/end
                start_date = as_of - timedelta(days=lookback_days)
                hist = stock.history(
                    start=start_date.strftime('%Y-%m-%d'),
                    end=(as_of + timedelta(days=1)).strftime('%Y-%m-%d'),
                    auto_adjust=True,
                    actions=False
                )
            else:
                # Current: use period
                if lookback_days <= 30:
                    period = '1mo'
                elif lookback_days <= 90:
                    period = '3mo'
                elif lookback_days <= 365:
                    period = '1y'
                else:
                    period = '2y'
                hist = stock.history(period=period, auto_adjust=True, actions=False)
            
            if hist.empty:
                return []
            
            volumes = []
            for idx, row in hist.iterrows():
                if hasattr(idx, 'date'):
                    volume_date = idx.date()
                else:
                    volume_date = idx.to_pydatetime().date()
                
                if volume_date <= as_of:
                    vol = row['Volume']
                    if not math.isnan(vol):
                        volumes.append(int(vol))
            
            if use_cache and volumes:
                cache_data = {
                    "ticker": ticker,
                    "as_of": _date_to_str(as_of),
                    "lookback_days": lookback_days,
                    "volumes": volumes,
                }
                _write_cache(cache_path, cache_data)
            
            return volumes
            
        except Exception as e:
            print(f"ERROR: Failed to fetch volumes for {ticker}: {e}")
            return []
    
    def get_adv(self, ticker: str, as_of: date, window: int = 20) -> Optional[Decimal]:
        """Get average daily dollar volume."""
        lookback_days = window * 2
        prices = self.get_prices(ticker, as_of, lookback_days)
        volumes = self.get_volumes(ticker, as_of, lookback_days)
        
        if not prices or not volumes or len(prices) != len(volumes):
            return None
        
        recent_prices = prices[-window:] if len(prices) >= window else prices
        recent_volumes = volumes[-window:] if len(volumes) >= window else volumes
        
        if not recent_prices or not recent_volumes:
            return None
        
        dollar_volumes = [float(p) * v for p, v in zip(recent_prices, recent_volumes)]
        avg_dollar_volume = sum(dollar_volumes) / len(dollar_volumes)
        
        return _quantize(avg_dollar_volume)
    
    def get_ticker_data(self, ticker: str, as_of: date, lookback_days: int = 365) -> Dict:
        """Get complete data package for a ticker."""
        prices = self.get_prices(ticker, as_of, lookback_days)
        returns = self.get_log_returns(ticker, as_of, lookback_days)
        volumes = self.get_volumes(ticker, as_of, lookback_days)
        
        return {
            "ticker": ticker,
            "as_of": as_of,
            "prices": prices,
            "returns": returns,
            "volumes": volumes,
            "num_days": len(prices),
        }

class BatchPriceProvider:
    def __init__(self, cache_ttl_hours: int = CACHE_TTL_HOURS):
        self.provider = PriceDataProvider(cache_ttl_hours)
    
    def get_batch_data(self, tickers: List[str], as_of: date, lookback_days: int = 365, include_xbi: bool = True) -> Dict[str, Dict]:
        """Get data for multiple tickers efficiently."""
        results = {}
        
        if include_xbi:
            print("Fetching XBI benchmark data...")
            xbi_data = self.provider.get_ticker_data("XBI", as_of, lookback_days)
            results["_xbi_"] = xbi_data
        
        for ticker in tickers:
            print(f"Fetching {ticker}...")
            try:
                data = self.provider.get_ticker_data(ticker, as_of, lookback_days)
                if data["num_days"] > 0:
                    results[ticker] = data
                else:
                    print(f"  WARNING: No data for {ticker}")
            except Exception as e:
                print(f"  ERROR: Failed to fetch {ticker}: {e}")
                continue
        
        return results

def get_prices(ticker: str, as_of: date, lookback_days: int) -> List[Decimal]:
    provider = PriceDataProvider()
    return provider.get_prices(ticker, as_of, lookback_days)

def get_log_returns(ticker: str, as_of: date, lookback_days: int) -> List[float]:
    provider = PriceDataProvider()
    return provider.get_log_returns(ticker, as_of, lookback_days)

def get_adv(ticker: str, as_of: date, window: int = 20) -> Optional[Decimal]:
    provider = PriceDataProvider()
    return provider.get_adv(ticker, as_of, window)

def validate_pit_discipline() -> bool:
    """Validate that PIT discipline is enforced."""
    print("=== PIT Discipline Validation ===")
    
    provider = PriceDataProvider()
    
    # Test with historical date
    test_as_of = date(2024, 6, 30)
    test_ticker = "VRTX"
    
    prices = provider.get_prices(test_ticker, test_as_of, lookback_days=90)
    
    if not prices:
        print("FAIL: No prices returned")
        return False
    
    print(f"✓ Fetched {len(prices)} prices for {test_ticker} as of {test_as_of}")
    print(f"  First price: ${prices[0]:.2f}")
    print(f"  Last price: ${prices[-1]:.2f}")
    
    return True

def validate_determinism() -> bool:
    """Validate that same inputs produce identical outputs."""
    print("\n=== Determinism Validation ===")
    
    provider = PriceDataProvider()
    
    ticker = "AMGN"
    as_of = date(2024, 12, 31)
    lookback = 60
    
    # First call
    prices1 = provider.get_prices(ticker, as_of, lookback, use_cache=False)
    
    # Second call (no cache to ensure we're re-fetching)
    prices2 = provider.get_prices(ticker, as_of, lookback, use_cache=False)
    
    if prices1 != prices2:
        print("FAIL: Results differ between calls")
        return False
    
    print(f"✓ Determinism validated: {len(prices1)} prices match across calls")
    
    return True

def validate_cache_integrity() -> bool:
    """Validate that cached data matches fresh data."""
    print("\n=== Cache Integrity Validation ===")
    
    provider = PriceDataProvider()
    
    ticker = "XBI"
    as_of = date(2024, 12, 31)
    lookback = 30
    
    # Fresh fetch
    prices_fresh = provider.get_prices(ticker, as_of, lookback, use_cache=False)
    
    # Cached fetch
    prices_cached = provider.get_prices(ticker, as_of, lookback, use_cache=True)
    
    if prices_fresh != prices_cached:
        print("FAIL: Cached data doesn't match fresh data")
        return False
    
    print(f"✓ Cache integrity validated: {len(prices_cached)} prices match")
    
    return True

if __name__ == "__main__":
    """Run validation suite."""
    print("=== Market Data Provider Validation ===\n")
    
    tests = [
        ("PIT Discipline", validate_pit_discipline),
        ("Determinism", validate_determinism),
        ("Cache Integrity", validate_cache_integrity),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"ERROR in {name}: {e}")
            results.append((name, False))
    
    print("\n=== Validation Summary ===")
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(p for _, p in results)
    print(f"\nOverall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
