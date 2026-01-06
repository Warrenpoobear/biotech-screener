"""
MINIMAL_INTEGRATION_PATCH.md

Exact diffs to wire market_data_provider into your existing pipeline.
These are minimal, bounded changes following your "diff-only" workflow.
"""

# =============================================================================
# PATCH 1: Add market data fields to security schema
# =============================================================================

# File: module_2_financial.py (or wherever you build the per-ticker dict)
# Location: Inside the loop that builds securities[ticker] dict

## BEFORE:
```python
securities[ticker] = {
    "ticker": ticker,
    "composite_score": score,
    "cash_runway_months": runway,
    # ... other fundamental fields
}
```

## AFTER:
```python
securities[ticker] = {
    "ticker": ticker,
    "composite_score": score,
    "cash_runway_months": runway,
    # ... other fundamental fields
    
    # Time-series placeholders (populated by market data enrichment)
    "market_data": None,  # Will hold {prices, returns, volumes, num_days}
    "_xbi_reference": None,  # XBI benchmark data for correlation/beta
    
    # Time-series features (populated by defensive overlay module)
    "vol_60d": None,
    "vol_20d": None,
    "corr_xbi_120d": None,
    "adv_20d": None,
    "drawdown_current": None,
    "beta_xbi_60d": None,
    "rsi_14d": None,
    
    # Position sizing outputs
    "suggested_weight_pct": None,
    "position_flags": [],
    "passed_gates": True,
    "rejection_reason": None,
}
```

**Lines changed:** +17
**Why:** Adds schema fields for market data and time-series features

# =============================================================================
# PATCH 2: Add market data enrichment step
# =============================================================================

# File: module_2_financial.py
# Location: After fundamental analysis, before returning securities

## ADD THIS FUNCTION:
```python
def enrich_with_market_data(
    securities: List[Dict],
    as_of: date,
) -> List[Dict]:
    """
    Enrich securities with market data for time-series calculations.
    
    This is called after fundamental analysis and before time-series overlay.
    """
    from market_data_provider import BatchPriceProvider
    
    # Get all tickers
    tickers = [sec["ticker"] for sec in securities]
    
    # Batch fetch (includes XBI automatically)
    print(f"Fetching market data for {len(tickers)} securities...")
    batch_provider = BatchPriceProvider()
    market_data = batch_provider.get_batch_data(
        tickers=tickers,
        as_of=as_of,
        lookback_days=365,  # 1 year history for time-series
        include_xbi=True,
    )
    
    # Attach to securities
    xbi_data = market_data.get("_xbi_")
    enriched_count = 0
    
    for sec in securities:
        ticker = sec["ticker"]
        
        if ticker in market_data:
            # Convert Decimals to floats for JSON serialization
            ticker_data = market_data[ticker]
            sec["market_data"] = {
                "prices": [float(p) for p in ticker_data["prices"]],
                "returns": ticker_data["returns"],
                "volumes": ticker_data["volumes"],
                "num_days": ticker_data["num_days"],
            }
            
            # XBI reference
            if xbi_data:
                sec["_xbi_reference"] = {
                    "prices": [float(p) for p in xbi_data["prices"]],
                    "returns": xbi_data["returns"],
                    "num_days": xbi_data["num_days"],
                }
            
            enriched_count += 1
        else:
            print(f"  WARNING: No market data for {ticker}")
    
    print(f"✓ Enriched {enriched_count}/{len(securities)} securities with market data")
    return securities
```

## THEN CALL IT:
```python
# At the end of your main processing function

## BEFORE:
```python
def run_module_2_financial(securities, as_of_date):
    # ... existing fundamental analysis code
    
    return securities  # Return immediately
```

## AFTER:
```python
def run_module_2_financial(securities, as_of_date):
    # ... existing fundamental analysis code
    
    # Enrich with market data
    securities = enrich_with_market_data(securities, as_of_date)
    
    return securities
```

**Lines changed:** +45 (new function) + 2 (call site)
**Why:** Adds market data to every security for downstream time-series calculations

# =============================================================================
# PATCH 3: Add feature flag for market data enrichment (optional)
# =============================================================================

# File: config.py (or wherever you store configuration)

## ADD:
```python
# Market data configuration
ENABLE_MARKET_DATA = True  # Set to False to skip market data enrichment
MARKET_DATA_CACHE_TTL_HOURS = 24  # Cache TTL
MARKET_DATA_LOOKBACK_DAYS = 365  # How far back to fetch
```

## THEN MODIFY THE CALL:
```python
def run_module_2_financial(securities, as_of_date):
    # ... existing fundamental analysis code
    
    # Enrich with market data (if enabled)
    if config.ENABLE_MARKET_DATA:
        securities = enrich_with_market_data(securities, as_of_date)
    else:
        print("Market data enrichment disabled")
    
    return securities
```

**Lines changed:** +3 (config) + 4 (conditional call)
**Why:** Allows toggling market data enrichment without code changes

# =============================================================================
# PATCH 4: Update requirements/dependencies
# =============================================================================

# File: requirements.txt (or install instructions)

## ADD:
```
yfinance>=0.2.32
```

## Or for conda/pip install:
```bash
pip install yfinance
```

**Why:** Adds the only external dependency needed

# =============================================================================
# PATCH 5: Add error handling for missing market data
# =============================================================================

# File: module_5_composite.py (or wherever you use market data)
# Location: Before calculating time-series features

## ADD THIS CHECK:
```python
def calculate_time_series_features(security):
    """Calculate time-series defensive features."""
    
    # Check if market data is available
    if security["market_data"] is None:
        security["position_flags"].append("NO_MARKET_DATA")
        security["rejection_reason"] = "Insufficient market data for analysis"
        security["passed_gates"] = False
        return security
    
    # Check if we have sufficient history
    if security["market_data"]["num_days"] < 60:
        security["position_flags"].append("INSUFFICIENT_HISTORY")
        print(f"  WARNING: {security['ticker']} has only {security['market_data']['num_days']} days of history")
    
    # Proceed with time-series calculations
    # ... (your time-series overlay code here)
    
    return security
```

**Lines changed:** +15
**Why:** Graceful handling of missing/insufficient market data

# =============================================================================
# COMPLETE MINIMAL INTEGRATION
# =============================================================================

## Summary of changes:

1. **Schema update**: +17 lines (add fields to security dict)
2. **Enrichment function**: +45 lines (new function)
3. **Call site**: +2 lines (call enrichment)
4. **Config flag**: +7 lines (optional feature flag)
5. **Error handling**: +15 lines (graceful degradation)

**Total**: ~86 lines of new code + 1 dependency (yfinance)

## Testing the integration:

```python
# Quick test script
from datetime import date

# Your existing code
securities = run_module_1_universe()  # Get initial universe
securities = run_module_2_financial(securities, as_of=date(2024, 12, 31))

# Verify market data was added
for sec in securities:
    print(f"{sec['ticker']}: {sec['market_data']['num_days']} days of data")
```

## Rollback if needed:

If something breaks:
1. Set `ENABLE_MARKET_DATA = False` in config
2. System continues without time-series features
3. Debug in isolation

## Next steps after integration:

Once market data enrichment is working:
1. Wire in the time-series defensive overlay (time_series_manager.py)
2. Add gates and sizing rules
3. Enable regime detection
4. Run validation suite

---

# =============================================================================
# ALTERNATIVE: Minimal "just get it working" patch (10 lines)
# =============================================================================

If you want the absolute minimum to unblock development:

```python
# At the end of module_2_financial.py

from market_data_provider import PriceDataProvider
from datetime import date

def quick_add_prices(securities, as_of):
    """Minimal market data for development."""
    provider = PriceDataProvider()
    for sec in securities:
        sec["prices_1y"] = provider.get_prices(sec["ticker"], as_of, 365)
        sec["returns_1y"] = provider.get_log_returns(sec["ticker"], as_of, 365)
    return securities

# Call it:
securities = quick_add_prices(securities, date(2024, 12, 31))
```

This gives you prices/returns immediately so you can start developing
time-series features. Optimize later.
"""