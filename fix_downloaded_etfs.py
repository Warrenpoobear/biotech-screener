#!/usr/bin/env python3
"""
fix_downloaded_etfs.py - Fix Downloaded ETF Files

Handles:
1. Convert XBI from Excel to CSV
2. Clean up IBB CSV (remove header rows)
3. Verify files are ready for import

Usage:
    python fix_downloaded_etfs.py
"""

from pathlib import Path
import csv


def convert_xbi_excel_to_csv():
    """Convert XBI Excel file to CSV"""
    print("\n📥 Converting XBI Excel to CSV...")
    
    xlsx_file = Path('etf_csvs/XBI_holdings.xlsx')
    csv_file = Path('etf_csvs/XBI_holdings.csv')
    
    if not xlsx_file.exists():
        print(f"  ⚠️  XBI Excel file not found: {xlsx_file}")
        return False
    
    try:
        # Try with pandas
        import pandas as pd
        print(f"  → Reading Excel file...")
        df = pd.read_excel(xlsx_file)
        
        # Save to CSV
        df.to_csv(csv_file, index=False)
        print(f"  ✅ Converted to CSV: {csv_file}")
        print(f"     Rows: {len(df)}, Columns: {len(df.columns)}")
        
        return True
    
    except ImportError:
        print(f"  ❌ pandas not installed")
        print(f"  → Installing pandas and openpyxl...")
        
        import subprocess
        try:
            subprocess.run(['pip', 'install', 'pandas', 'openpyxl'], check=True)
            print(f"  ✅ Installed pandas and openpyxl")
            
            # Try again
            import pandas as pd
            df = pd.read_excel(xlsx_file)
            df.to_csv(csv_file, index=False)
            print(f"  ✅ Converted to CSV: {csv_file}")
            return True
        except Exception as e:
            print(f"  ❌ Auto-install failed: {e}")
            print(f"\n  Manual fix:")
            print(f"  1. pip install pandas openpyxl")
            print(f"  2. Or open {xlsx_file} in Excel")
            print(f"  3. Save As → CSV: {csv_file}")
            return False
    
    except Exception as e:
        print(f"  ❌ Conversion failed: {e}")
        return False


def fix_ibb_csv():
    """Fix IBB CSV - skip metadata header rows"""
    print("\n📥 Fixing IBB CSV...")
    
    csv_file = Path('etf_csvs/IBB_holdings.csv')
    
    if not csv_file.exists():
        print(f"  ⚠️  IBB CSV not found: {csv_file}")
        return False
    
    try:
        # Read all lines
        with open(csv_file, 'r', encoding='utf-8-sig') as f:
            lines = f.readlines()
        
        print(f"  → File has {len(lines)} lines")
        
        # Find the header row (contains "Ticker" or "Symbol")
        header_row_idx = None
        for i, line in enumerate(lines):
            line_lower = line.lower()
            if 'ticker' in line_lower or 'symbol' in line_lower:
                header_row_idx = i
                print(f"  → Found header at line {i+1}")
                break
        
        if header_row_idx is None:
            print(f"  ❌ Could not find header row with 'Ticker' or 'Symbol'")
            print(f"  → First 10 lines:")
            for i, line in enumerate(lines[:10]):
                print(f"     {i+1}: {line.strip()[:80]}")
            return False
        
        # Write cleaned CSV (from header row onwards)
        cleaned_file = Path('etf_csvs/IBB_holdings_cleaned.csv')
        with open(cleaned_file, 'w', encoding='utf-8', newline='') as f:
            for line in lines[header_row_idx:]:
                f.write(line)
        
        # Replace original
        cleaned_file.replace(csv_file)
        
        data_rows = len(lines) - header_row_idx - 1
        print(f"  ✅ Cleaned CSV: {data_rows} data rows")
        
        return True
    
    except Exception as e:
        print(f"  ❌ Fix failed: {e}")
        return False


def check_csv_format(csv_file: Path, etf_name: str) -> bool:
    """Check if CSV has proper format"""
    print(f"\n🔍 Checking {etf_name} format...")
    
    if not csv_file.exists():
        print(f"  ❌ File not found: {csv_file}")
        return False
    
    try:
        with open(csv_file, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            columns = reader.fieldnames
            
            # Check for ticker column
            ticker_col = None
            for col in columns:
                col_lower = col.lower().strip()
                if any(term in col_lower for term in ['ticker', 'symbol', 'component']):
                    ticker_col = col
                    break
            
            if not ticker_col:
                print(f"  ❌ No ticker column found")
                print(f"     Columns: {columns}")
                return False
            
            # Count rows
            rows = list(reader)
            tickers = [row.get(ticker_col, '').strip() for row in rows if row.get(ticker_col, '').strip()]
            
            print(f"  ✅ Format OK")
            print(f"     Ticker column: '{ticker_col}'")
            print(f"     Tickers found: {len(tickers)}")
            print(f"     Sample: {', '.join(tickers[:5])}")
            
            return True
    
    except Exception as e:
        print(f"  ❌ Check failed: {e}")
        return False


def main():
    print("="*80)
    print("FIX DOWNLOADED ETF FILES")
    print("="*80)
    
    results = {}
    
    # Fix XBI (Excel → CSV)
    results['xbi'] = convert_xbi_excel_to_csv()
    
    # Fix IBB (remove header rows)
    results['ibb'] = fix_ibb_csv()
    
    # Check formats
    print("\n" + "="*80)
    print("VERIFICATION")
    print("="*80)
    
    xbi_ok = check_csv_format(Path('etf_csvs/XBI_holdings.csv'), 'XBI')
    ibb_ok = check_csv_format(Path('etf_csvs/IBB_holdings.csv'), 'IBB')
    nbi_ok = check_csv_format(Path('etf_csvs/NBI_holdings.csv'), 'NBI')
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"XBI: {'✅ Ready' if xbi_ok else '❌ Needs fix'}")
    print(f"IBB: {'✅ Ready' if ibb_ok else '❌ Needs fix'}")
    print(f"NBI: {'✅ Ready' if nbi_ok else '❌ Not downloaded'}")
    
    ready_count = sum([xbi_ok, ibb_ok, nbi_ok])
    
    # Next steps
    if not nbi_ok:
        print("\n" + "="*80)
        print("NBI MANUAL DOWNLOAD REQUIRED")
        print("="*80)
        print("\n1. Go to: https://indexes.nasdaqomx.com/Index/Weighting/NBI")
        print("2. Click 'Download' button (usually top right)")
        print("3. Save as: etf_csvs/NBI_holdings.csv")
        print("\n4. Then re-run: python fix_downloaded_etfs.py")
    
    if ready_count == 3:
        print("\n" + "="*80)
        print("ALL FILES READY!")
        print("="*80)
        print("\nNext steps:")
        print("  python import_etf_csvs.py")
        print("  python add_etf_tickers_to_universe.py")
        print("="*80)
        return 0
    elif ready_count >= 2:
        print(f"\n⚠️  {ready_count}/3 files ready")
        print(f"   Complete NBI download and you're good to go!")
        return 1
    else:
        print(f"\n❌ Only {ready_count}/3 files ready")
        print(f"   Review errors above")
        return 1


if __name__ == "__main__":
    exit(main())
