#!/usr/bin/env python3
"""
collect_all_data.py - Master Data Collection Script

Runs all data collection scripts in optimal order:
1. Market data (fast, ~5 min)
2. Clinical trials (medium, ~10-15 min)  
3. Financial data (slow, ~30-60 min due to SEC rate limits)

Usage:
    python collect_all_data.py
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime


def run_collection(script_name: str, description: str) -> bool:
    """Run a collection script and report results"""
    
    print("\n" + "="*80)
    print(f"STARTING: {description}")
    print("="*80)
    print(f"Script: {script_name}")
    print(f"Started: {datetime.now().strftime('%H:%M:%S')}")
    print("="*80 + "\n")
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            check=True
        )
        
        print(f"\n✅ {description} completed successfully")
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description} failed with error code {e.returncode}")
        return False
    
    except KeyboardInterrupt:
        print(f"\n⚠️  {description} cancelled by user")
        raise


def main():
    print("="*80)
    print("MASTER DATA COLLECTION SCRIPT")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nThis will collect ALL data for your 302-ticker universe:")
    print("  1. Market data (prices, volume, market cap) - ~5 min")
    print("  2. Clinical trials (ClinicalTrials.gov) - ~10-15 min")
    print("  3. Financial data (SEC EDGAR filings) - ~30-60 min")
    print("\n📊 Total estimated time: 45-90 minutes")
    print("\n💡 Tip: You can run this overnight or during lunch!")
    print("="*80)
    
    response = input("\nProceed with full data collection? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("❌ Cancelled")
        return 1
    
    start_time = datetime.now()
    results = []
    
    try:
        # Collection 1: Market Data (fastest)
        results.append(('Market Data', run_collection(
            'collect_market_data.py',
            'Market Data Collection (Yahoo Finance)'
        )))
        
        # Collection 2: Clinical Trials (medium)
        results.append(('Clinical Trials', run_collection(
            'collect_ctgov_data.py',
            'Clinical Trials Collection (ClinicalTrials.gov)'
        )))
        
        # Collection 3: Financial Data (slowest)
        results.append(('Financial Data', run_collection(
            'collect_financial_data.py',
            'Financial Data Collection (SEC EDGAR)'
        )))
    
    except KeyboardInterrupt:
        print("\n\n❌ Collection cancelled by user")
        return 1
    
    # Summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "="*80)
    print("COLLECTION COMPLETE")
    print("="*80)
    print(f"Started:  {start_time.strftime('%H:%M:%S')}")
    print(f"Finished: {end_time.strftime('%H:%M:%S')}")
    print(f"Duration: {duration.total_seconds()/60:.1f} minutes")
    print("\nResults:")
    
    for name, success in results:
        status = "✅ Success" if success else "❌ Failed"
        print(f"  {status}: {name}")
    
    success_count = sum(1 for _, s in results if s)
    print(f"\nOverall: {success_count}/{len(results)} collections successful")
    
    if success_count == len(results):
        print("\n" + "="*80)
        print("🎊 ALL DATA COLLECTED SUCCESSFULLY!")
        print("="*80)
        print("\n📁 Output files:")
        print("  • production_data/market_data.json")
        print("  • production_data/trial_records.json")
        print("  • production_data/financial_data.json")
        print("\n🚀 Next steps:")
        print("  1. Run screening:")
        print("     python run_screen.py --as-of-date 2026-01-07 --data-dir production_data")
        print("\n  2. Expected results:")
        print("     • Universe: 302 tickers")
        print("     • Module 1 filtered: 100-150 (proper filtering)")
        print("     • Active universe: 150-200 tickers ✅")
        print("     • Final ranked: 60-100 top picks")
        print("="*80 + "\n")
    else:
        print("\n⚠️  Some collections failed - review errors above")
        print("    You can re-run individual scripts to retry:")
        for name, success in results:
            if not success:
                print(f"    • python collect_{name.lower().replace(' ', '_')}.py")
    
    print("="*80 + "\n")
    
    return 0 if success_count == len(results) else 1


if __name__ == "__main__":
    try:
        exit(main())
    except KeyboardInterrupt:
        print("\n\n❌ Collection cancelled by user")
        exit(1)
