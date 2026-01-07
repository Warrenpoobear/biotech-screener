#!/usr/bin/env python3
"""
auto_download_etf_holdings.py - Automatically Download XBI, IBB, NBI Holdings

Attempts to automatically download ETF holdings CSVs without manual clicking.

Usage:
    python auto_download_etf_holdings.py
"""

import requests
from pathlib import Path
import time
from datetime import date


def download_ibb_holdings(output_dir: Path) -> bool:
    """
    Download IBB holdings from iShares direct API.
    
    iShares provides a direct CSV download URL that doesn't require JavaScript.
    """
    print("\n📥 Downloading IBB (iShares Biotechnology ETF)...")
    
    # iShares direct download URL
    # Format: https://www.ishares.com/us/products/[FUND_ID]/[FUND_NAME]/[MAGIC_NUMBER].ajax?fileType=csv&fileName=[TICKER]_holdings&dataType=fund
    url = "https://www.ishares.com/us/products/239699/ishares-biotechnology-etf/1467271812596.ajax"
    params = {
        'fileType': 'csv',
        'fileName': 'IBB_holdings',
        'dataType': 'fund'
    }
    
    try:
        print(f"  Fetching from iShares API...")
        response = requests.get(url, params=params, timeout=30)
        
        if response.status_code == 200 and len(response.content) > 1000:
            output_file = output_dir / "IBB_holdings.csv"
            with open(output_file, 'wb') as f:
                f.write(response.content)
            
            # Count lines to estimate holdings
            lines = response.text.count('\n')
            print(f"  ✅ Downloaded IBB: ~{lines-10} holdings")
            return True
        else:
            print(f"  ❌ Failed: HTTP {response.status_code}")
            return False
    
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def download_xbi_holdings(output_dir: Path) -> bool:
    """
    Attempt to download XBI holdings from SPDR.
    
    SPDR is trickier - they may require JavaScript or have anti-scraping.
    Try a few known URL patterns.
    """
    print("\n📥 Downloading XBI (SPDR S&P Biotech ETF)...")
    
    # Known SPDR URL patterns to try
    url_patterns = [
        # Pattern 1: Direct holdings download
        "https://www.ssga.com/us/en/individual/etfs/library-content/products/fund-data/etfs/us/holdings-daily-us-en-xbi.xlsx",
        # Pattern 2: Alternative URL
        "https://www.ssga.com/us/en/intermediary/etfs/library-content/products/fund-data/etfs/us/holdings-daily-us-en-xbi.xlsx",
        # Pattern 3: CSV format
        "https://www.ssga.com/us/en/individual/etfs/library-content/products/fund-data/etfs/us/holdings-daily-us-en-xbi.csv",
    ]
    
    for i, url in enumerate(url_patterns, 1):
        try:
            print(f"  Attempt {i}/{len(url_patterns)}...")
            response = requests.get(url, timeout=30, allow_redirects=True)
            
            if response.status_code == 200 and len(response.content) > 1000:
                # Determine file extension
                ext = '.xlsx' if 'xlsx' in url else '.csv'
                output_file = output_dir / f"XBI_holdings{ext}"
                
                with open(output_file, 'wb') as f:
                    f.write(response.content)
                
                print(f"  ✅ Downloaded XBI: {len(response.content):,} bytes")
                
                if ext == '.xlsx':
                    print(f"  ⚠️  Downloaded as Excel - converting to CSV...")
                    try:
                        import pandas as pd
                        df = pd.read_excel(output_file)
                        csv_file = output_dir / "XBI_holdings.csv"
                        df.to_csv(csv_file, index=False)
                        print(f"  ✅ Converted to CSV: {csv_file}")
                    except ImportError:
                        print(f"  ⚠️  Install pandas to auto-convert: pip install pandas openpyxl")
                        print(f"  ⚠️  Or manually open {output_file} in Excel and Save As CSV")
                
                return True
        
        except Exception as e:
            continue
    
    print(f"  ❌ All download attempts failed")
    return False


def download_nbi_holdings(output_dir: Path) -> bool:
    """
    Attempt to download NBI holdings from Nasdaq.
    
    Nasdaq index page may require JavaScript, but we can try direct API calls.
    """
    print("\n📥 Downloading NBI (Nasdaq Biotechnology Index)...")
    
    # Try Nasdaq API endpoints
    url_patterns = [
        # Pattern 1: Direct CSV download
        "https://indexes.nasdaqomx.com/Index/ExportWeightings/NBI",
        # Pattern 2: Alternative format
        "https://api.nasdaq.com/api/indexes/constituents/NBI",
    ]
    
    for i, url in enumerate(url_patterns, 1):
        try:
            print(f"  Attempt {i}/{len(url_patterns)}...")
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'text/csv,application/json'
            }
            
            response = requests.get(url, headers=headers, timeout=30)
            
            if response.status_code == 200 and len(response.content) > 1000:
                output_file = output_dir / "NBI_holdings.csv"
                
                # Handle JSON response
                if 'application/json' in response.headers.get('Content-Type', ''):
                    import json
                    data = response.json()
                    # Extract constituents (format varies by API)
                    # This is a placeholder - actual parsing depends on API response structure
                    print(f"  ⚠️  Got JSON response - may need manual parsing")
                    with open(output_dir / "NBI_holdings.json", 'w') as f:
                        json.dump(data, f, indent=2)
                    print(f"  ℹ️  Saved as JSON: {output_dir / 'NBI_holdings.json'}")
                    return False  # Needs manual conversion
                else:
                    # CSV response
                    with open(output_file, 'wb') as f:
                        f.write(response.content)
                    
                    lines = response.text.count('\n')
                    print(f"  ✅ Downloaded NBI: ~{lines-2} holdings")
                    return True
        
        except Exception as e:
            continue
    
    print(f"  ❌ All download attempts failed")
    return False


def download_all_etf_holdings():
    """Main function to download all ETF holdings"""
    
    print("="*80)
    print("AUTOMATIC ETF HOLDINGS DOWNLOADER")
    print("="*80)
    print(f"Date: {date.today()}")
    
    # Create output directory
    output_dir = Path('etf_csvs')
    output_dir.mkdir(exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Track successes
    results = {
        'ibb': False,
        'xbi': False,
        'nbi': False
    }
    
    # Download each ETF
    results['ibb'] = download_ibb_holdings(output_dir)
    time.sleep(1)  # Be nice to servers
    
    results['xbi'] = download_xbi_holdings(output_dir)
    time.sleep(1)
    
    results['nbi'] = download_nbi_holdings(output_dir)
    
    # Summary
    print("\n" + "="*80)
    print("DOWNLOAD SUMMARY")
    print("="*80)
    
    success_count = sum(results.values())
    
    print(f"IBB: {'✅ Success' if results['ibb'] else '❌ Failed'}")
    print(f"XBI: {'✅ Success' if results['xbi'] else '❌ Failed'}")
    print(f"NBI: {'✅ Success' if results['nbi'] else '❌ Failed'}")
    print(f"\nSuccessfully downloaded: {success_count}/3")
    
    # Manual instructions for failures
    if success_count < 3:
        print("\n" + "="*80)
        print("MANUAL DOWNLOAD REQUIRED")
        print("="*80)
        
        if not results['xbi']:
            print("\n❌ XBI Failed - Download manually:")
            print("   1. Go to: https://www.ssga.com/us/en/individual/etfs/funds/xbi")
            print("   2. Click 'Holdings' tab")
            print("   3. Click 'Download All Holdings'")
            print("   4. Save as: etf_csvs/XBI_holdings.csv")
        
        if not results['ibb']:
            print("\n❌ IBB Failed - Download manually:")
            print("   1. Go to: https://www.ishares.com/us/products/239699/")
            print("   2. Click 'Holdings' tab")
            print("   3. Click 'Download'")
            print("   4. Save as: etf_csvs/IBB_holdings.csv")
        
        if not results['nbi']:
            print("\n❌ NBI Failed - Download manually:")
            print("   1. Go to: https://indexes.nasdaqomx.com/Index/Weighting/NBI")
            print("   2. Click 'Download' button")
            print("   3. Save as: etf_csvs/NBI_holdings.csv")
    
    # Next steps
    if success_count > 0:
        print("\n" + "="*80)
        print("NEXT STEPS")
        print("="*80)
        
        if success_count == 3:
            print("✅ All downloads successful!")
            print("\nRun these commands:")
            print("  python import_etf_csvs.py")
            print("  python add_etf_tickers_to_universe.py")
        else:
            print(f"⚠️  {success_count}/3 downloads successful")
            print("\n1. Complete manual downloads (see above)")
            print("2. Then run:")
            print("   python import_etf_csvs.py")
            print("   python add_etf_tickers_to_universe.py")
    
    print("="*80 + "\n")
    
    return success_count


def main():
    try:
        success_count = download_all_etf_holdings()
        return 0 if success_count == 3 else 1
    except KeyboardInterrupt:
        print("\n\n❌ Download cancelled by user")
        return 1
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
