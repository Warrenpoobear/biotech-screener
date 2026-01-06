"""
test_time_series_integration.py - Test time-series collector integration

Run this BEFORE modifying collect_universe_data.py to verify the 
time_series_collector works independently.
"""
import sys
from pathlib import Path
from datetime import date

# Add collectors to path
sys.path.insert(0, str(Path(__file__).parent))

from collectors import time_series_collector

def test_single_ticker():
    """Test time-series collection for a single ticker."""
    print("=" * 60)
    print("TEST 1: Single Ticker Collection")
    print("=" * 60)
    
    ticker = "VRTX"
    as_of = date(2024, 12, 31)
    
    print(f"\nFetching time-series data for {ticker}...")
    data = time_series_collector.collect_time_series_data(ticker, as_of=as_of)
    
    if data.get('success'):
        print(f"\n✓ SUCCESS!")
        print(f"  Ticker: {data['ticker']}")
        print(f"  Days of data: {data['time_series']['num_days']}")
        print(f"  First price: ${data['time_series']['prices'][0]:.2f}")
        print(f"  Last price: ${data['time_series']['prices'][-1]:.2f}")
        print(f"  ADV (20d): {data['liquidity']['adv_20d_formatted']}")
        print(f"  PIT-safe: {data['provenance']['pit_safe']}")
        return True
    else:
        print(f"\n✗ FAILED: {data.get('error')}")
        return False


def test_batch_collection():
    """Test batch collection for multiple tickers."""
    print("\n" + "=" * 60)
    print("TEST 2: Batch Collection")
    print("=" * 60)
    
    tickers = ["VRTX", "AMGN", "SRPT"]
    as_of = date(2024, 12, 31)
    
    print(f"\nFetching time-series data for {len(tickers)} tickers...")
    results = time_series_collector.collect_batch(
        tickers,
        as_of=as_of,
        lookback_days=365
    )
    
    successful = sum(1 for d in results.values() 
                    if d.get('success') and d['ticker'] != 'XBI')
    
    print(f"\n{'='*60}")
    print(f"Results: {successful}/{len(tickers)} successful")
    
    # Check for XBI benchmark
    if "_XBI_BENCHMARK_" in results:
        xbi = results["_XBI_BENCHMARK_"]
        print(f"XBI benchmark: {xbi['time_series']['num_days']} days")
        print("✓ Benchmark data available for correlation/beta calculations")
    
    return successful == len(tickers)


def test_integration_format():
    """Test that output format matches what collect_universe_data expects."""
    print("\n" + "=" * 60)
    print("TEST 3: Integration Format Check")
    print("=" * 60)
    
    ticker = "VRTX"
    data = time_series_collector.collect_time_series_data(ticker)
    
    # Check required fields
    required_fields = [
        'ticker', 'success', 'time_series', 'liquidity', 'provenance'
    ]
    
    missing = [f for f in required_fields if f not in data]
    
    if missing:
        print(f"\n✗ Missing fields: {missing}")
        return False
    
    # Check time_series structure
    if data.get('success'):
        ts_fields = ['prices', 'returns', 'volumes', 'num_days']
        ts_missing = [f for f in ts_fields if f not in data['time_series']]
        
        if ts_missing:
            print(f"\n✗ Missing time_series fields: {ts_missing}")
            return False
    
    print("\n✓ All required fields present")
    print("✓ Format matches collect_universe_data.py expectations")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("TIME-SERIES COLLECTOR INTEGRATION TESTS")
    print("=" * 60)
    
    tests = [
        ("Single Ticker", test_single_ticker),
        ("Batch Collection", test_batch_collection),
        ("Integration Format", test_integration_format)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n✗ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(p for _, p in results)
    
    if all_passed:
        print("\n✓ ALL TESTS PASSED")
        print("\nYou're ready to integrate into collect_universe_data.py!")
        print("Follow the instructions in PATCH_collect_universe_data.txt")
    else:
        print("\n✗ SOME TESTS FAILED")
        print("\nFix the issues above before integrating")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
