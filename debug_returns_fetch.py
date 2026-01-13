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

print("\n=== Step 1: Look up SecIds using investments() ===")
ticker_queries = [f"{t}:US" for t in test_tickers]
print(f"Querying: {ticker_queries}")

try:
    inv_df = md.direct.investments(ticker_queries)
    print(f"Result type: {type(inv_df)}")
    if inv_df is not None and not inv_df.empty:
        print(f"Columns: {list(inv_df.columns)}")
        print(f"Index: {inv_df.index.tolist()}")
        print(f"\nFull DataFrame:")
        print(inv_df.to_string())

        # Extract SecIds
        ticker_to_secid = {}
        for idx, row in inv_df.iterrows():
            print(f"\n  Row index: {idx}, type: {type(idx)}")
            print(f"  Row data: {dict(row)}")
            sec_id = row.get('SecId', None)
            if sec_id:
                ticker_to_secid[str(idx).split(':')[0]] = str(sec_id)

        print(f"\nTicker to SecId mapping: {ticker_to_secid}")
    else:
        print("Empty or None result")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

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
