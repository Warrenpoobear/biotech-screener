"""
Interactive price data fetcher with step-by-step guidance.

This script will:
1. Scan your checkpoint files to find all tickers
2. Determine the date range needed
3. Download historical prices from Yahoo Finance
4. Save to CSV for use in optimization

Run with: python optimization/fetch_prices_interactive.py
"""

import json
import csv
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
import sys
import time

print("="*70)
print("BIOTECH SCREENER - HISTORICAL PRICE DATA FETCHER")
print("="*70)
print()

# Check for yfinance
try:
    import yfinance as yf
    print("✓ yfinance is installed")
except ImportError:
    print("❌ yfinance is not installed")
    print()
    print("Please install it with:")
    print("  pip install yfinance")
    print()
    sys.exit(1)

# Configuration
CHECKPOINT_DIR = Path('checkpoints')
OUTPUT_FILE = Path('production_data/price_history.csv')

print()
print("Configuration:")
print(f"  Checkpoint directory: {CHECKPOINT_DIR}")
print(f"  Output file: {OUTPUT_FILE}")
print()

# Step 1: Find checkpoint files
print("-" * 70)
print("STEP 1: Scanning for checkpoint files...")
print("-" * 70)

if not CHECKPOINT_DIR.exists():
    print(f"❌ ERROR: Directory '{CHECKPOINT_DIR}' does not exist")
    print()
    print("This script expects screening results in the 'checkpoints/' directory.")
    print("Files should be named like: module_5_2024-01-15.json")
    print()
    sys.exit(1)

checkpoint_files = list(CHECKPOINT_DIR.glob('module_5_*.json'))

if not checkpoint_files:
    print(f"❌ ERROR: No checkpoint files found in '{CHECKPOINT_DIR}'")
    print()
    print("Looking for files matching pattern: module_5_YYYY-MM-DD.json")
    print()
    print("Do you have screening results saved? They should be in:")
    print(f"  {CHECKPOINT_DIR.absolute()}")
    print()
    sys.exit(1)

print(f"✓ Found {len(checkpoint_files)} checkpoint files")

# Extract dates
checkpoint_dates = []
for filepath in checkpoint_files:
    try:
        date_str = filepath.stem.replace('module_5_', '')
        date = datetime.strptime(date_str, '%Y-%m-%d')
        checkpoint_dates.append(date)
    except ValueError:
        continue

if not checkpoint_dates:
    print("❌ ERROR: Could not parse dates from checkpoint filenames")
    sys.exit(1)

min_date = min(checkpoint_dates)
max_date = max(checkpoint_dates)

print(f"  Date range: {min_date.date()} to {max_date.date()}")
print(f"  Span: {(max_date - min_date).days} days")
print()

# Step 2: Extract tickers
print("-" * 70)
print("STEP 2: Extracting tickers from checkpoint files...")
print("-" * 70)

all_tickers = set()

for i, filepath in enumerate(checkpoint_files, 1):
    try:
        with open(filepath) as f:
            data = json.load(f)
        
        # Try different possible structures
        securities = data.get('ranked_securities', data.get('results', []))
        
        file_tickers = set()
        for security in securities:
            ticker = security.get('ticker')
            if ticker:
                all_tickers.add(ticker)
                file_tickers.add(ticker)
        
        print(f"  [{i}/{len(checkpoint_files)}] {filepath.name}: {len(file_tickers)} tickers")
        
    except Exception as e:
        print(f"  [{i}/{len(checkpoint_files)}] {filepath.name}: ❌ Error: {e}")
        continue

if not all_tickers:
    print()
    print("❌ ERROR: No tickers found in checkpoint files")
    sys.exit(1)

all_tickers = sorted(all_tickers)

print()
print(f"✓ Found {len(all_tickers)} unique tickers")
print(f"  Sample: {', '.join(all_tickers[:10])}")
if len(all_tickers) > 10:
    print(f"  ... and {len(all_tickers) - 10} more")
print()

# Calculate date range for price fetching (extend by 30 days on each end)
fetch_start = min_date - timedelta(days=30)
fetch_end = max_date + timedelta(days=30)

print(f"Price data will be fetched from {fetch_start.date()} to {fetch_end.date()}")
print(f"  (Extended ±30 days to ensure forward return coverage)")
print()

# Confirm before proceeding
print("-" * 70)
print("READY TO FETCH PRICES")
print("-" * 70)
print()
print(f"About to download {len(all_tickers)} tickers × ~{(fetch_end - fetch_start).days} days")
print(f"Estimated time: {len(all_tickers) * 2 // 60} - {len(all_tickers) * 3 // 60} minutes")
print()

response = input("Proceed with download? (yes/no): ").strip().lower()

if response not in ['yes', 'y']:
    print()
    print("Download cancelled.")
    sys.exit(0)

# Step 3: Download prices
print()
print("-" * 70)
print("STEP 3: Downloading price data from Yahoo Finance...")
print("-" * 70)
print()

price_data = []
failed_tickers = []
success_count = 0
start_time = time.time()

for i, ticker in enumerate(all_tickers, 1):
    # Progress indicator
    elapsed = time.time() - start_time
    if i > 1:
        avg_time = elapsed / (i - 1)
        remaining = avg_time * (len(all_tickers) - i + 1)
        eta = f"{int(remaining // 60)}m {int(remaining % 60)}s"
    else:
        eta = "calculating..."
    
    print(f"[{i}/{len(all_tickers)}] {ticker:8s} (ETA: {eta})...", end='', flush=True)
    
    try:
        # Download with progress disabled (auto_adjust=False for backward compatibility)
        df = yf.download(
            ticker,
            start=fetch_start.strftime('%Y-%m-%d'),
            end=fetch_end.strftime('%Y-%m-%d'),
            progress=False,
            auto_adjust=False
        )
        
        if df.empty:
            print(" ❌ No data")
            failed_tickers.append(ticker)
            continue
        
        # Extract adjusted close prices
        # Handle both single-ticker and multi-ticker dataframes
        for date, row in df.iterrows():
            try:
                # Try to get Adj Close
                if 'Adj Close' in df.columns:
                    close_price = float(row['Adj Close'])
                elif ('Adj Close', ticker) in df.columns:
                    close_price = float(row[('Adj Close', ticker)])
                elif 'Close' in df.columns:
                    close_price = float(row['Close'])
                else:
                    # Fallback: use the first numeric column
                    close_price = float(row.iloc[-1])
                
                price_data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'ticker': ticker,
                    'close': close_price
                })
            except (KeyError, ValueError, IndexError):
                continue
        
        success_count += 1
        print(f" ✓ {len(df)} days")
        
    except Exception as e:
        print(f" ❌ Error: {str(e)[:30]}")
        failed_tickers.append(ticker)
        continue
    
    # Small delay to avoid rate limiting
    time.sleep(0.5)

print()
print("-" * 70)
print("DOWNLOAD COMPLETE")
print("-" * 70)
print()

total_time = time.time() - start_time
print(f"Total time: {int(total_time // 60)}m {int(total_time % 60)}s")
print(f"Success: {success_count}/{len(all_tickers)} tickers ({success_count/len(all_tickers)*100:.1f}%)")
print(f"Total price observations: {len(price_data):,}")
print()

if failed_tickers:
    print(f"Failed tickers ({len(failed_tickers)}):")
    for ticker in failed_tickers[:20]:
        print(f"  - {ticker}")
    if len(failed_tickers) > 20:
        print(f"  ... and {len(failed_tickers) - 20} more")
    print()

# Step 4: Save to CSV
print("-" * 70)
print("STEP 4: Saving price data...")
print("-" * 70)

if not price_data:
    print("❌ ERROR: No price data to save")
    sys.exit(1)

# Create output directory
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# Write CSV
with open(OUTPUT_FILE, 'w', newline='') as f:
    fieldnames = ['date', 'ticker', 'close']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(price_data)

print(f"✓ Saved {len(price_data):,} price observations to:")
print(f"  {OUTPUT_FILE.absolute()}")
print()

# Statistics
tickers_with_data = len(set(p['ticker'] for p in price_data))
dates_covered = len(set(p['date'] for p in price_data))

print("Price data statistics:")
print(f"  Tickers with data: {tickers_with_data}")
print(f"  Dates covered: {dates_covered}")
print(f"  Date range: {min(p['date'] for p in price_data)} to {max(p['date'] for p in price_data)}")
print()

# Next steps
print("="*70)
print("SUCCESS! Price data is ready.")
print("="*70)
print()
print("Next steps:")
print()
print("1. Extract training data:")
print("   python optimization\\extract_historical_data.py")
print()
print("2. Optimize weights:")
print("   python -m optimization.optimize_weights_scipy")
print()
print("3. Validate results:")
print("   python -m optimization.validate_weights")
print()
