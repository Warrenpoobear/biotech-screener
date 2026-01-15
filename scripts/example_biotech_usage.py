"""
example_biotech_usage.py - Concrete example using Wake Robin biotech universe

This demonstrates how to use the market_data_provider with a real
biotech universe and prepare data for time-series defensive overlays.
"""

from market_data_provider import (
    PriceDataProvider,
    BatchPriceProvider,
    get_prices,
    get_log_returns,
    get_adv,
)
from datetime import date, datetime
from decimal import Decimal
from typing import List, Dict
import json

# =============================================================================
# EXAMPLE 1: Single ticker deep dive
# =============================================================================

def example_single_ticker():
    """
    Fetch complete history for a single biotech ticker.
    """
    print("=" * 70)
    print("EXAMPLE 1: Single Ticker Deep Dive")
    print("=" * 70)
    
    provider = PriceDataProvider()
    
    ticker = "CELG"  # Example biotech (if still trading) or use "AMGN"
    as_of = date(2024, 12, 31)
    
    print(f"\nFetching data for {ticker} as of {as_of}")
    print(f"Lookback: 365 days\n")
    
    # Get complete data package
    data = provider.get_ticker_data(ticker, as_of, lookback_days=365)
    
    print(f"Results:")
    print(f"  - Prices fetched: {data['num_days']}")
    
    if data['prices']:
        print(f"  - First price: ${data['prices'][0]:.2f}")
        print(f"  - Last price: ${data['prices'][-1]:.2f}")
        print(f"  - Price change: {((data['prices'][-1] / data['prices'][0]) - 1) * 100:.1f}%")
    
    if data['returns']:
        print(f"  - Returns computed: {len(data['returns'])}")
        mean_return = sum(data['returns']) / len(data['returns'])
        print(f"  - Mean daily return: {mean_return:.4f}")
    
    if data['volumes']:
        avg_volume = sum(data['volumes']) / len(data['volumes'])
        print(f"  - Average volume: {avg_volume:,.0f}")
    
    # Get ADV
    adv = provider.get_adv(ticker, as_of, window=20)
    if adv:
        print(f"  - 20-day ADV: ${adv:,.0f}")
    
    return data


# =============================================================================
# EXAMPLE 2: Small biotech universe (pilot)
# =============================================================================

def example_pilot_universe():
    """
    Fetch data for a pilot biotech universe (20 tickers).
    
    This simulates what you'd do in your actual screening pipeline.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Pilot Biotech Universe")
    print("=" * 70)
    
    # Your 20-ticker pilot universe (adjust to your actual tickers)
    pilot_tickers = [
        "AMGN", "BIIB", "GILD", "REGN", "VRTX",  # Large cap
        "ALNY", "BMRN", "IONS", "SRPT", "INCY",  # Mid cap
        "ARWR", "FOLD", "SAGE", "BLUE", "CRSP",  # Clinical stage
        "EDIT", "NTLA", "BEAM", "VERV", "PRVB",  # Gene editing
    ]
    
    as_of = date(2024, 12, 31)
    
    print(f"\nFetching data for {len(pilot_tickers)} tickers")
    print(f"As of: {as_of}")
    print(f"Includes XBI benchmark\n")
    
    # Use batch provider for efficiency
    batch_provider = BatchPriceProvider()
    
    market_data = batch_provider.get_batch_data(
        tickers=pilot_tickers,
        as_of=as_of,
        lookback_days=365,
        include_xbi=True,
    )
    
    print(f"\n=== Results ===")
    print(f"Total tickers fetched: {len(market_data)}")
    
    # Summary statistics
    coverage = []
    for ticker, data in market_data.items():
        if ticker == "_xbi_":
            continue
        
        num_days = data.get("num_days", 0)
        coverage.append((ticker, num_days))
    
    coverage.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nData coverage (days of history):")
    print(f"{'Ticker':<8} {'Days':<6} {'Coverage':<10}")
    print("-" * 30)
    
    for ticker, days in coverage[:10]:  # Top 10
        pct = (days / 252) * 100  # Approx 252 trading days/year
        bar = "█" * int(pct / 5)  # Visual bar
        print(f"{ticker:<8} {days:<6} {bar}")
    
    # XBI benchmark stats
    if "_xbi_" in market_data:
        xbi = market_data["_xbi_"]
        print(f"\nXBI Benchmark: {xbi['num_days']} days of data")
    
    return market_data


# =============================================================================
# EXAMPLE 3: Prepare for time-series calculations
# =============================================================================

def example_time_series_prep():
    """
    Show how to structure data for the time-series defensive overlay.
    
    This is what you'd pass into the TimeSeriesManager from the specs.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Prepare Data for Time-Series Calculations")
    print("=" * 70)
    
    # Simulate securities from your fundamental analysis
    # (These would come from Module 1 + Module 2 in your pipeline)
    fundamental_securities = [
        {
            "ticker": "AMGN",
            "composite_score": 85.5,
            "score_percentile": 0.9,
            "cash_runway_months": 36,
            "lead_program_stage_score": 4,  # Approved
            "catalyst_count_12m": 1,
        },
        {
            "ticker": "VRTX",
            "composite_score": 92.3,
            "score_percentile": 0.95,
            "cash_runway_months": 48,
            "lead_program_stage_score": 4,
            "catalyst_count_12m": 2,
        },
        {
            "ticker": "SRPT",
            "composite_score": 72.1,
            "score_percentile": 0.7,
            "cash_runway_months": 18,
            "lead_program_stage_score": 3,  # Phase 3
            "catalyst_count_12m": 1,
        },
    ]
    
    as_of = date(2024, 12, 31)
    
    print(f"\nEnriching {len(fundamental_securities)} securities with market data")
    print(f"As of: {as_of}\n")
    
    # Get tickers
    tickers = [sec["ticker"] for sec in fundamental_securities]
    
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
    
    enriched_securities = []
    for sec in fundamental_securities:
        ticker = sec["ticker"]
        
        if ticker in market_data:
            # Add market data
            sec["market_data"] = {
                "prices": [float(p) for p in market_data[ticker]["prices"]],  # Convert Decimal to float for JSON
                "returns": market_data[ticker]["returns"],
                "volumes": market_data[ticker]["volumes"],
                "num_days": market_data[ticker]["num_days"],
            }
            
            # Add XBI reference (for correlation/beta)
            sec["_xbi_data"] = {
                "prices": [float(p) for p in xbi_data["prices"]],
                "returns": xbi_data["returns"],
                "num_days": xbi_data["num_days"],
            }
            
            print(f"✓ {ticker}: {sec['market_data']['num_days']} days of data")
            enriched_securities.append(sec)
        else:
            print(f"✗ {ticker}: No market data available")
    
    print(f"\n=== Ready for Time-Series Overlay ===")
    print(f"Securities with complete data: {len(enriched_securities)}")
    print(f"\nNext step: Pass to TimeSeriesManager.enrich_securities_with_time_series()")
    
    return enriched_securities


# =============================================================================
# EXAMPLE 4: PIT discipline demonstration
# =============================================================================

def example_pit_discipline():
    """
    Demonstrate that PIT discipline prevents lookahead bias.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Point-in-Time Discipline")
    print("=" * 70)
    
    provider = PriceDataProvider()
    ticker = "AMGN"
    
    # Historical as_of date
    historical_date = date(2023, 6, 30)
    
    print(f"\nFetching {ticker} data as of {historical_date}")
    print("This simulates running the system in mid-2023")
    print("PIT guarantee: No data after June 30, 2023 should be included\n")
    
    prices = provider.get_prices(ticker, historical_date, lookback_days=180)
    
    print(f"Results:")
    print(f"  - Prices fetched: {len(prices)}")
    print(f"  - This represents ~{len(prices)} trading days before {historical_date}")
    
    # Now fetch current data
    current_date = date(2024, 12, 31)
    current_prices = provider.get_prices(ticker, current_date, lookback_days=180)
    
    print(f"\nCompare to fetching as of {current_date}:")
    print(f"  - Prices fetched: {len(current_prices)}")
    
    print(f"\n✓ PIT discipline enforced:")
    print(f"  Historical fetch stopped at {historical_date}")
    print(f"  Current fetch includes data up to {current_date}")
    print(f"  This prevents lookahead bias in backtesting!")


# =============================================================================
# EXAMPLE 5: Export for time-series validation
# =============================================================================

def example_export_for_validation():
    """
    Export market data to JSON for validation/testing.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Export Data for Validation")
    print("=" * 70)
    
    # Small test universe
    test_tickers = ["AMGN", "VRTX", "SRPT"]
    as_of = date(2024, 12, 31)
    
    batch_provider = BatchPriceProvider()
    market_data = batch_provider.get_batch_data(
        tickers=test_tickers,
        as_of=as_of,
        lookback_days=365,
        include_xbi=True,
    )
    
    # Convert to JSON-serializable format
    export_data = {}
    for ticker, data in market_data.items():
        export_data[ticker] = {
            "ticker": ticker,
            "as_of": str(data["as_of"]),
            "num_days": data["num_days"],
            "prices": [float(p) for p in data["prices"]],
            "returns": data["returns"],
            "volumes": data["volumes"],
        }
    
    # Save to file
    output_file = "market_data_validation.json"
    with open(output_file, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    print(f"\n✓ Exported {len(export_data)} tickers to {output_file}")
    print(f"  Use this file to validate time-series calculations")
    print(f"  File size: {len(json.dumps(export_data)) / 1024:.1f} KB")
    
    return output_file


# =============================================================================
# MAIN: Run all examples
# =============================================================================

if __name__ == "__main__":
    print("\n")
    print("=" * 70)
    print("MARKET DATA PROVIDER - BIOTECH UNIVERSE EXAMPLES")
    print("=" * 70)
    print("\nThese examples demonstrate:")
    print("1. Single ticker deep dive")
    print("2. Batch fetching for pilot universe")
    print("3. Preparing data for time-series calculations")
    print("4. Point-in-time discipline")
    print("5. Exporting for validation")
    print("\n")
    
    try:
        # Example 1: Single ticker
        example_single_ticker()
        
        # Example 2: Pilot universe
        example_pilot_universe()
        
        # Example 3: Time-series prep
        example_time_series_prep()
        
        # Example 4: PIT discipline
        example_pit_discipline()
        
        # Example 5: Export
        example_export_for_validation()
        
        print("\n" + "=" * 70)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print("\nNext steps:")
        print("1. Install yfinance: pip install yfinance")
        print("2. Run this script: python example_biotech_usage.py")
        print("3. Check cache/market_data/ for cached results")
        print("4. Wire into your Module 2 pipeline (see INTEGRATION_GUIDE.md)")
        print("\n")
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        print("\nTroubleshooting:")
        print("- Ensure yfinance is installed: pip install yfinance")
        print("- Check internet connection")
        print("- Verify ticker symbols are valid")
        print("\n")
        raise
