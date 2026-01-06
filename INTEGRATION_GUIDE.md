"""
INTEGRATION_GUIDE.md - How to wire market_data_provider.py into Wake Robin

This guide shows how to integrate the PIT-safe price provider into your
existing Wake Robin pipeline to enable time-series defensive overlays.
"""

# =============================================================================
# QUICK START (5 minutes)
# =============================================================================

## Step 1: Install dependency

```bash
pip install yfinance
```

## Step 2: Copy file to your project

```bash
cp market_data_provider.py /path/to/wake_robin/
```

## Step 3: Test it

```python
from market_data_provider import PriceDataProvider
from datetime import date

provider = PriceDataProvider()

# Get prices for AAPL as of Dec 31, 2024
prices = provider.get_prices("AAPL", date(2024, 12, 31), lookback_days=60)
print(f"Fetched {len(prices)} prices")

# Get log returns for XBI
returns = provider.get_log_returns("XBI", date(2024, 12, 31), lookback_days=365)
print(f"Fetched {len(returns)} returns")
```

# =============================================================================
# INTEGRATION PATTERNS
# =============================================================================

## Pattern A: Add to existing module_2_financial.py

If you have a `module_2_financial.py` that enriches securities with
financial data, add time-series features there:

```python
# At top of module_2_financial.py
from market_data_provider import PriceDataProvider
from datetime import date

# Initialize provider (reuse across securities)
price_provider = PriceDataProvider()

# Inside your securities processing loop
def enrich_securities(securities: List[Dict], as_of: date) -> List[Dict]:
    """Enrich securities with market data."""
    
    # Get XBI benchmark data once (reused for all tickers)
    xbi_data = price_provider.get_ticker_data("XBI", as_of, lookback_days=365)
    
    for sec in securities:
        ticker = sec["ticker"]
        
        # Get price/return data
        ticker_data = price_provider.get_ticker_data(ticker, as_of, lookback_days=365)
        
        # Add to security record (as placeholders for now)
        sec["price_history"] = {
            "prices": ticker_data["prices"],
            "returns": ticker_data["returns"],
            "volumes": ticker_data["volumes"],
            "num_days": ticker_data["num_days"],
        }
        
        # XBI reference for later correlation/beta calculations
        sec["_xbi_reference"] = xbi_data
    
    return securities
```

## Pattern B: Separate market data enrichment step

Create a new module that runs after universe/financial but before
time-series calculations:

```python
# market_enrichment.py

from market_data_provider import BatchPriceProvider
from datetime import date
from typing import List, Dict

def enrich_universe_with_market_data(
    securities: List[Dict],
    as_of: date,
) -> List[Dict]:
    """
    Enrich entire universe with market data in one batch.
    
    This is more efficient than individual calls.
    """
    # Get all tickers
    tickers = [sec["ticker"] for sec in securities]
    
    # Batch fetch
    batch_provider = BatchPriceProvider()
    market_data = batch_provider.get_batch_data(
        tickers=tickers,
        as_of=as_of,
        lookback_days=365,
        include_xbi=True,
    )
    
    # Attach to securities
    xbi_data = market_data.get("_xbi_")
    
    for sec in securities:
        ticker = sec["ticker"]
        
        if ticker in market_data:
            sec["market_data"] = market_data[ticker]
            sec["_xbi_reference"] = xbi_data
        else:
            # No market data available (illiquid, delisted, etc.)
            sec["market_data"] = None
            sec["_xbi_reference"] = xbi_data
    
    return securities
```

## Pattern C: Lazy loading (for development)

If you want to develop time-series features without blocking on data
fetching during every test run:

```python
# Use a wrapper that caches aggressively

from market_data_provider import PriceDataProvider
from functools import lru_cache

class CachedPriceProvider:
    def __init__(self):
        self.provider = PriceDataProvider(cache_ttl_hours=720)  # 30 days
    
    @lru_cache(maxsize=1000)
    def get_prices_cached(self, ticker: str, as_of_str: str, lookback: int):
        """In-memory + disk cache."""
        from datetime import date
        as_of = date.fromisoformat(as_of_str)
        return self.provider.get_prices(ticker, as_of, lookback)
```

# =============================================================================
# WIRING INTO TIME-SERIES CALCULATIONS
# =============================================================================

Once you have market data attached to securities, wire it into the
time-series defensive overlay module:

```python
# In time_series_manager.py (from the specification document)

from market_data_provider import PriceDataProvider

class TimeSeriesManager:
    def __init__(self):
        self.price_provider = PriceDataProvider()
        # ... rest of init
    
    def enrich_securities_with_time_series(
        self,
        securities: List[Dict[str, Any]],
        as_of: date,
    ) -> List[PositionRecommendation]:
        """
        Main entry point: enrich fundamental securities with time-series features.
        
        Now using the PIT-safe price provider instead of requiring
        pre-fetched data.
        """
        
        # Fetch XBI data once
        xbi_data = self.price_provider.get_ticker_data("XBI", as_of, lookback_days=365)
        
        recommendations = []
        
        for sec in securities:
            ticker = sec["ticker"]
            
            # Fetch ticker data (cached if available)
            ticker_data = self.price_provider.get_ticker_data(ticker, as_of, lookback_days=365)
            
            # Calculate time-series features
            ts_features = self._calculate_security_features(
                ticker,
                ticker_data,
                xbi_data,
            )
            
            # ... rest of processing
        
        return recommendations
```

# =============================================================================
# MIGRATION PATH (from existing Yahoo Finance code)
# =============================================================================

If you currently have code like this:

```python
# OLD: Current snapshot only
import yfinance as yf

stock = yf.Ticker(ticker)
info = stock.info
price = info.get('currentPrice')
```

Migrate to:

```python
# NEW: Historical series with PIT discipline
from market_data_provider import PriceDataProvider
from datetime import date

provider = PriceDataProvider()

# Get as-of date (e.g., today or end of last quarter)
as_of = date.today()

# Get recent prices (last 60 days for current price context)
prices = provider.get_prices(ticker, as_of, lookback_days=60)

if prices:
    current_price = prices[-1]  # Most recent price
    
    # Also get longer history for time-series calculations
    prices_1y = provider.get_prices(ticker, as_of, lookback_days=365)
    returns_1y = provider.get_log_returns(ticker, as_of, lookback_days=365)
```

# =============================================================================
# TESTING CHECKLIST
# =============================================================================

Before going live, verify:

1. **PIT Discipline**: 
   ```python
   # Test: as_of in the past should never return future data
   from market_data_provider import validate_pit_discipline
   validate_pit_discipline()
   ```

2. **Determinism**:
   ```python
   # Test: same inputs -> same outputs
   from market_data_provider import validate_determinism
   validate_determinism()
   ```

3. **Cache Performance**:
   ```python
   # First call: fetches from Yahoo (slow)
   prices1 = provider.get_prices("AAPL", date(2024, 12, 31), 365)
   
   # Second call: uses cache (fast)
   prices2 = provider.get_prices("AAPL", date(2024, 12, 31), 365)
   
   assert prices1 == prices2
   ```

4. **Edge Cases**:
   ```python
   # Delisted ticker
   prices = provider.get_prices("INVALIDTICKER", date(2024, 12, 31), 365)
   assert prices == []  # Graceful failure
   
   # Very new ticker (IPO in last 60 days)
   prices = provider.get_prices("NEWTICKER", date(2024, 12, 31), 365)
   # Should return whatever is available, even if < 365 days
   ```

# =============================================================================
# PRODUCTION DEPLOYMENT CHECKLIST
# =============================================================================

## Pre-deployment

- [ ] Run full validation suite: `python market_data_provider.py`
- [ ] Test with your actual biotech universe (20+ tickers)
- [ ] Verify cache directory permissions: `cache/market_data/`
- [ ] Set up cache cleanup cron (optional): delete files >30 days old

## Post-deployment monitoring

Monitor these metrics:

1. **Cache hit rate**: Should be >80% after initial run
2. **Data coverage**: % of tickers with sufficient history
3. **Fetch failures**: Track which tickers fail consistently
4. **PIT violations**: Should be ZERO (add assertion in code)

## Failure modes

| Failure | Symptom | Fix |
|---------|---------|-----|
| No internet | Empty price lists | Add offline mode with stale cache |
| Yahoo rate limit | 429 errors | Add exponential backoff |
| Delisted ticker | Empty results | Handle gracefully in downstream code |
| Cache corruption | Integrity mismatch | Auto-delete and re-fetch |

# =============================================================================
# ADVANCED: HISTORICAL BACKTESTING
# =============================================================================

For walk-forward validation, you need to simulate running the system
at different historical dates:

```python
from datetime import date, timedelta

# Backtest dates: end of each quarter in 2023-2024
backtest_dates = [
    date(2023, 3, 31),
    date(2023, 6, 30),
    date(2023, 9, 30),
    date(2023, 12, 31),
    date(2024, 3, 31),
    date(2024, 6, 30),
    date(2024, 9, 30),
    date(2024, 12, 31),
]

provider = PriceDataProvider()

for as_of in backtest_dates:
    print(f"\n=== Running as of {as_of} ===")
    
    # Get universe securities (from your fundamental analysis)
    securities = load_securities_as_of(as_of)
    
    # Enrich with market data
    for sec in securities:
        ticker = sec["ticker"]
        
        # This respects PIT: only gets data <= as_of
        data = provider.get_ticker_data(ticker, as_of, lookback_days=365)
        sec["market_data"] = data
    
    # Run time-series calculations
    # ... your analysis code here
```

The key: `as_of` parameter ensures you never use future data, making
backtest results valid.

# =============================================================================
# FAQ
# =============================================================================

**Q: What if Yahoo Finance changes their API?**
A: The provider is designed to be modular. You can swap in a different
data source (Alpha Vantage, Polygon, etc.) by implementing the same
interface.

**Q: Can I use this for intraday data?**
A: No, this is designed for daily OHLC data. For intraday, you'd need
a different provider (and probably pay for it).

**Q: What about corporate actions (splits, dividends)?**
A: We use `auto_adjust=True` in yfinance, which handles splits and
dividends automatically. The prices you get are adjusted.

**Q: How do I handle tickers that changed (e.g., ABBV spinoff from ABT)?**
A: You'll need to track ticker history separately. This provider doesn't
handle corporate structure changes automatically.

**Q: What's the performance like for 100+ tickers?**
A: With caching:
- First run: ~30-60 seconds (fetching from Yahoo)
- Subsequent runs (cache hit): ~1-2 seconds
- Batch fetch all at once vs. sequential doesn't matter much due to Yahoo's rate limits

**Q: Can I use this in production without yfinance?**
A: You'd need to implement an adapter for your production data source.
The interface is simple: `get_prices(ticker, as_of, lookback) -> List[Decimal]`

# =============================================================================
# NEXT STEPS
# =============================================================================

1. **Run validation**: `python market_data_provider.py`
2. **Test with your tickers**: Create a simple script with your biotech universe
3. **Wire into Module 2**: Add market data enrichment to your pipeline
4. **Enable time-series overlays**: Now you can use the defensive overlay specs

Once this is working, you can move to Option A (wiring the defensive
overlays into your scoring pipeline).
"""