#!/usr/bin/env python3
"""Debug script to diagnose Morningstar returns fetch issues."""

import os
import sys

# Check token
token = os.environ.get('MD_AUTH_TOKEN', '')
if not token:
    print("ERROR: MD_AUTH_TOKEN not set")
    sys.exit(1)
print(f"Token present: {len(token)} chars")

import morningstar_data as md
print(f"morningstar-data version: {md.__version__}")

# Test tickers
test_tickers = ['VRTX', 'BMRN', 'ALNY']

print("\n=== Step 1: Explore investments() API ===")

# Try different API calls to find SecIds
for ticker in test_tickers:
    print(f"\n--- Looking up {ticker} ---")

    # Try 1: Search by ticker string
    try:
        result = md.direct.investments(ticker)
        print(f"investments('{ticker}'): type={type(result)}")
        if result is not None and hasattr(result, 'empty') and not result.empty:
            print(f"  Columns: {list(result.columns)}")
            print(f"  First rows:\n{result.head(3).to_string()}")
        elif result is not None:
            print(f"  Result: {result}")
    except Exception as e:
        print(f"  Error: {e}")

    # Try 2: Search with exchange
    try:
        result = md.direct.investments(f"{ticker}:US")
        print(f"investments('{ticker}:US'): type={type(result)}")
        if result is not None and hasattr(result, 'empty') and not result.empty:
            print(f"  Columns: {list(result.columns)}")
            print(f"  First rows:\n{result.head(3).to_string()}")
    except Exception as e:
        print(f"  Error: {e}")

# Try 3: Check for other functions
print("\n=== Available md.direct functions ===")
funcs = [f for f in dir(md.direct) if not f.startswith('_')]
print(funcs)

# Try search function if available
if hasattr(md.direct, 'search'):
    print("\n=== Trying md.direct.search() ===")
    try:
        result = md.direct.search("VRTX")
        print(f"search('VRTX'): {type(result)}")
        if hasattr(result, 'head'):
            print(result.head().to_string())
    except Exception as e:
        print(f"Error: {e}")

# Try get_security_details or similar
if hasattr(md.direct, 'get_security_details'):
    print("\n=== Trying md.direct.get_security_details() ===")
    try:
        result = md.direct.get_security_details(['VRTX:US'])
        print(f"Result: {type(result)}")
    except Exception as e:
        print(f"Error: {e}")

print("\n=== Step 2: Fetch returns using SecIds ===")
# Use known working SecId from earlier testing
test_secids = ['0P000005R7']  # VRTX
print(f"Testing with known SecId: {test_secids}")

try:
    from morningstar_data.direct.data_type import Frequency

    df = md.direct.get_returns(
        investments=test_secids,
        start_date='2024-01-01',
        end_date='2024-12-31',
        freq=Frequency.monthly,
    )

    print(f"Result type: {type(df)}")
    if df is not None and not df.empty:
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"\nFirst 5 rows:")
        print(df.head().to_string())
    else:
        print("Empty or None result")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Step 3: Try ticker:US format directly ===")
try:
    df = md.direct.get_returns(
        investments=['VRTX:US'],
        start_date='2024-01-01',
        end_date='2024-12-31',
        freq=Frequency.monthly,
    )

    print(f"Result type: {type(df)}")
    if df is not None and not df.empty:
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"\nFirst 5 rows:")
        print(df.head().to_string())
    else:
        print("Empty or None result")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Done ===")
