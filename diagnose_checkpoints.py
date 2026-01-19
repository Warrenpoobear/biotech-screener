"""
Diagnostic script to check checkpoint file structure.

This will help identify the correct keys to extract tickers.
"""

import json
from pathlib import Path

checkpoint_dir = Path('checkpoints')

print("="*70)
print("CHECKPOINT FILE DIAGNOSTIC")
print("="*70)
print()

# Find checkpoint files
checkpoint_files = sorted(checkpoint_dir.glob('module_5_*.json'))

if not checkpoint_files:
    print(f"❌ No checkpoint files found in {checkpoint_dir}")
    print()
    print("Looking for files matching: module_5_*.json")
    exit(1)

print(f"Found {len(checkpoint_files)} checkpoint files")
print()

# Check first file in detail
first_file = checkpoint_files[0]
print(f"Examining: {first_file.name}")
print("-" * 70)

try:
    with open(first_file) as f:
        data = json.load(f)
    
    print(f"Top-level keys in JSON:")
    for key in data.keys():
        value = data[key]
        if isinstance(value, list):
            print(f"  - {key}: list with {len(value)} items")
            if value and isinstance(value[0], dict):
                print(f"    First item keys: {list(value[0].keys())[:5]}")
        elif isinstance(value, dict):
            print(f"  - {key}: dict with {len(value)} keys")
            print(f"    Keys: {list(value.keys())[:5]}")
        else:
            print(f"  - {key}: {type(value).__name__}")
    
    print()
    print("Checking for ticker locations...")
    print("-" * 70)
    
    # Try different possible locations
    tickers_found = []
    
    # Method 1: ranked_securities
    if 'ranked_securities' in data:
        securities = data['ranked_securities']
        if securities:
            print(f"✓ Found 'ranked_securities' with {len(securities)} items")
            if isinstance(securities[0], dict):
                print(f"  Sample keys: {list(securities[0].keys())}")
                if 'ticker' in securities[0]:
                    tickers_found.extend([s.get('ticker') for s in securities if s.get('ticker')])
                    print(f"  ✓ Found 'ticker' field")
                    print(f"  Sample tickers: {tickers_found[:5]}")
    
    # Method 2: results
    if 'results' in data:
        results = data['results']
        if results:
            print(f"✓ Found 'results' with {len(results)} items")
            if isinstance(results[0], dict):
                print(f"  Sample keys: {list(results[0].keys())}")
                if 'ticker' in results[0]:
                    tickers_found.extend([s.get('ticker') for s in results if s.get('ticker')])
                    print(f"  ✓ Found 'ticker' field")
    
    # Method 3: Direct list at top level
    for key, value in data.items():
        if isinstance(value, list) and value and isinstance(value[0], dict):
            if 'ticker' in value[0]:
                print(f"✓ Found tickers in '{key}' with {len(value)} items")
                sample_tickers = [item.get('ticker') for item in value[:5] if item.get('ticker')]
                print(f"  Sample tickers: {sample_tickers}")
                tickers_found.extend([item.get('ticker') for item in value if item.get('ticker')])
    
    print()
    print(f"Total unique tickers found: {len(set(tickers_found))}")
    
    if not tickers_found:
        print()
        print("❌ Could not find tickers in expected locations")
        print()
        print("Full file structure (first 50 lines):")
        print(json.dumps(data, indent=2)[:2000])
        print("...")
    
except Exception as e:
    print(f"❌ Error reading file: {e}")
    import traceback
    traceback.print_exc()

print()
print("="*70)
print("RECOMMENDATION")
print("="*70)
print()

if tickers_found:
    print("✓ Tickers successfully extracted!")
    print()
    print("Your checkpoint files are formatted correctly.")
    print("The price fetcher should work now.")
else:
    print("❌ Could not extract tickers from checkpoint files.")
    print()
    print("Please share the output above so we can fix the extraction logic.")
