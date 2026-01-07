#!/usr/bin/env python3
"""
process_all_etfs_final.py - Process All Three ETF Files

Handles:
1. XBI - Fix CSV format (has metadata header rows)
2. IBB - Already working (263 tickers)
3. NBI - Process uploaded Excel file (262 tickers)

Usage:
    python process_all_etfs_final.py
"""

import pandas as pd
import csv
from pathlib import Path
import json


def process_xbi_csv():
    """Fix XBI CSV - skip metadata header rows, find actual data"""
    print("\n📥 Processing XBI...")
    
    xbi_file = Path('etf_csvs/XBI_holdings.csv')
    
    if not xbi_file.exists():
        print(f"  ❌ XBI CSV not found: {xbi_file}")
        return []
    
    try:
        # Read all lines
        with open(xbi_file, 'r', encoding='utf-8-sig') as f:
            lines = f.readlines()
        
        # Find the row with "Ticker" or "Symbol" header
        header_idx = None
        for i, line in enumerate(lines):
            # Check if line contains ticker-like header
            if any(term in line.lower() for term in ['ticker', 'symbol', 'identifier', 'name']):
                # Make sure it's not just metadata
                if ',' in line and not line.startswith('Fund'):
                    header_idx = i
                    print(f"  → Found data header at line {i+1}: {line.strip()[:80]}")
                    break
        
        if header_idx is None:
            print(f"  ❌ Could not find data header")
            return []
        
        # Try to parse from header onwards
        with open(xbi_file, 'r', encoding='utf-8-sig') as f:
            for _ in range(header_idx):
                next(f)  # Skip metadata rows
            
            reader = csv.DictReader(f)
            
            # Find ticker column
            columns = [c.strip() for c in reader.fieldnames]
            print(f"  → Columns: {columns}")
            
            ticker_col = None
            for col in columns:
                col_lower = col.lower()
                if any(term in col_lower for term in ['ticker', 'symbol', 'identifier']):
                    ticker_col = col
                    break
            
            if not ticker_col:
                # Sometimes it's just called "Name" or is the first column
                ticker_col = columns[0]
                print(f"  → Using first column as ticker: '{ticker_col}'")
            
            # Extract tickers
            tickers = []
            for row in reader:
                ticker = row.get(ticker_col, '').strip()
                # Clean ticker
                if ticker and len(ticker) <= 6 and ticker.isalpha():
                    tickers.append(ticker.upper())
            
            print(f"  ✅ Extracted {len(tickers)} tickers")
            print(f"     Sample: {', '.join(tickers[:5])}")
            
            return tickers
    
    except Exception as e:
        print(f"  ❌ Processing failed: {e}")
        import traceback
        traceback.print_exc()
        return []


def process_ibb_csv():
    """Process IBB CSV (already working)"""
    print("\n📥 Processing IBB...")
    
    ibb_file = Path('etf_csvs/IBB_holdings.csv')
    
    if not ibb_file.exists():
        print(f"  ❌ IBB CSV not found: {ibb_file}")
        return []
    
    try:
        with open(ibb_file, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            
            # Find ticker column
            ticker_col = None
            for col in reader.fieldnames:
                if 'ticker' in col.lower() or 'symbol' in col.lower():
                    ticker_col = col
                    break
            
            if not ticker_col:
                print(f"  ❌ No ticker column found")
                return []
            
            tickers = []
            for row in reader:
                ticker = row.get(ticker_col, '').strip()
                if ticker and ticker not in ['', 'Cash', 'CASH']:
                    tickers.append(ticker.upper())
            
            print(f"  ✅ Loaded {len(tickers)} tickers")
            print(f"     Sample: {', '.join(tickers[:5])}")
            
            return tickers
    
    except Exception as e:
        print(f"  ❌ Processing failed: {e}")
        return []


def process_nbi_excel():
    """Process NBI Excel file (uploaded)"""
    print("\n📥 Processing NBI...")
    
    # Check in uploads directory first
    nbi_file = Path('/mnt/user-data/uploads/SODWeightings_20260107_NBI__1_.xlsx')
    
    if not nbi_file.exists():
        # Check in etf_csvs
        nbi_file = Path('etf_csvs/NBI_holdings.xlsx')
        if not nbi_file.exists():
            nbi_file = Path('etf_csvs/NBI_holdings.csv')
    
    if not nbi_file.exists():
        print(f"  ❌ NBI file not found")
        return []
    
    try:
        print(f"  → Reading from: {nbi_file}")
        
        if nbi_file.suffix == '.xlsx':
            df = pd.read_excel(nbi_file)
        else:
            df = pd.read_csv(nbi_file)
        
        # Find ticker column
        ticker_col = None
        for col in df.columns:
            col_lower = col.lower()
            if 'symbol' in col_lower or 'ticker' in col_lower:
                ticker_col = col
                break
        
        if not ticker_col:
            print(f"  ❌ No ticker column found")
            print(f"     Columns: {list(df.columns)}")
            return []
        
        print(f"  → Ticker column: '{ticker_col}'")
        
        tickers = df[ticker_col].dropna().astype(str).str.strip().str.upper().unique().tolist()
        
        # Filter out non-tickers
        tickers = [t for t in tickers if t and len(t) <= 6 and t.isalpha()]
        
        print(f"  ✅ Loaded {len(tickers)} tickers")
        print(f"     Sample: {', '.join(tickers[:5])}")
        
        return tickers
    
    except Exception as e:
        print(f"  ❌ Processing failed: {e}")
        import traceback
        traceback.print_exc()
        return []


def main():
    print("="*80)
    print("PROCESSING ALL ETF HOLDINGS")
    print("="*80)
    
    # Process all three ETFs
    xbi_tickers = process_xbi_csv()
    ibb_tickers = process_ibb_csv()
    nbi_tickers = process_nbi_excel()
    
    # Summary
    print("\n" + "="*80)
    print("PROCESSING SUMMARY")
    print("="*80)
    print(f"XBI: {len(xbi_tickers)} tickers")
    print(f"IBB: {len(ibb_tickers)} tickers")
    print(f"NBI: {len(nbi_tickers)} tickers")
    
    # Combine unique
    all_tickers = set(xbi_tickers) | set(ibb_tickers) | set(nbi_tickers)
    print(f"Total unique: {len(all_tickers)} tickers")
    
    # Save to JSON
    holdings = {
        'xbi': xbi_tickers,
        'ibb': ibb_tickers,
        'nbi': nbi_tickers
    }
    
    with open('etf_holdings_complete.json', 'w') as f:
        json.dump(holdings, f, indent=2, sort_keys=True)
    
    print(f"\n✅ Saved to: etf_holdings_complete.json")
    
    # Overlap analysis
    xbi_set = set(xbi_tickers)
    ibb_set = set(ibb_tickers)
    nbi_set = set(nbi_tickers)
    
    in_all_three = xbi_set & ibb_set & nbi_set
    
    print("\n" + "="*80)
    print("OVERLAP ANALYSIS")
    print("="*80)
    print(f"In all three: {len(in_all_three)} tickers")
    print(f"XBI only: {len(xbi_set - ibb_set - nbi_set)} tickers")
    print(f"IBB only: {len(ibb_set - xbi_set - nbi_set)} tickers")
    print(f"NBI only: {len(nbi_set - xbi_set - ibb_set)} tickers")
    
    # Sample
    print(f"\n📋 Sample of all tickers (first 20):")
    for i, ticker in enumerate(sorted(all_tickers)[:20], 1):
        in_xbi = '✓' if ticker in xbi_set else '✗'
        in_ibb = '✓' if ticker in ibb_set else '✗'
        in_nbi = '✓' if ticker in nbi_set else '✗'
        print(f"   {i:2d}. {ticker:6s}  XBI:{in_xbi}  IBB:{in_ibb}  NBI:{in_nbi}")
    
    if len(all_tickers) > 20:
        print(f"   ... and {len(all_tickers) - 20} more")
    
    # Next steps
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("Run: python add_etf_tickers_to_universe.py")
    print("="*80 + "\n")
    
    return 0 if len(all_tickers) >= 250 else 1


if __name__ == "__main__":
    exit(main())
