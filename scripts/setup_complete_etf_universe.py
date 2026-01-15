#!/usr/bin/env python3
"""
setup_complete_etf_universe.py - ONE COMMAND TO RULE THEM ALL

Automatically downloads XBI, IBB, NBI holdings, imports them, and adds to universe.

Usage:
    python setup_complete_etf_universe.py
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd: str, description: str) -> bool:
    """Run a command and report success/failure"""
    print(f"\n{'='*80}")
    print(f"STEP: {description}")
    print(f"{'='*80}")
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            capture_output=False,
            text=True
        )
        print(f"✅ Success: {description}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {description}")
        print(f"   Error: {e}")
        return False


def main():
    print("="*80)
    print("COMPLETE ETF UNIVERSE SETUP")
    print("ONE-COMMAND SOLUTION")
    print("="*80)
    print("\nThis will:")
    print("  1. Automatically download XBI, IBB, NBI holdings")
    print("  2. Import CSVs into JSON")
    print("  3. Add missing tickers to your universe")
    print("  4. Verify 100% coverage")
    print("\n" + "="*80)
    
    response = input("\nProceed? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("❌ Cancelled")
        return 1
    
    # Step 1: Auto-download holdings
    success_1 = run_command(
        "python auto_download_etf_holdings.py",
        "Auto-download ETF holdings"
    )
    
    # Check if we got at least some CSVs
    csv_dir = Path('etf_csvs')
    csv_files = list(csv_dir.glob('*.csv')) if csv_dir.exists() else []
    
    if len(csv_files) < 3:
        print(f"\n⚠️  Only {len(csv_files)}/3 ETFs downloaded automatically")
        print(f"   Some may require manual download (see instructions above)")
        
        response = input("\nContinue with available CSVs? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("❌ Cancelled - complete manual downloads and re-run")
            return 1
    
    # Step 2: Import CSVs
    success_2 = run_command(
        "python import_etf_csvs.py",
        "Import CSVs into JSON"
    )
    
    if not success_2:
        print("❌ Import failed - check CSV files")
        return 1
    
    # Step 3: Add to universe
    success_3 = run_command(
        "python add_etf_tickers_to_universe.py",
        "Add ETF tickers to universe"
    )
    
    if not success_3:
        print("❌ Failed to add tickers to universe")
        return 1
    
    # Step 4: Verify coverage
    success_4 = run_command(
        "python check_etf_coverage.py --universe production_data/universe.json",
        "Verify ETF coverage"
    )
    
    # Final summary
    print("\n" + "="*80)
    print("SETUP COMPLETE!")
    print("="*80)
    
    if success_1 and success_2 and success_3 and success_4:
        print("✅ All steps successful!")
        print("\n🎊 Your universe now has 100% ETF coverage!")
        print("\nNext steps:")
        print("  1. Collect data for new tickers:")
        print("     python collect_financial_data.py")
        print("     python collect_ctgov_data.py --output production_data/trial_records.json")
        print("  2. Re-run screening:")
        print("     python run_screen.py --as-of-date 2026-01-07 --data-dir production_data")
    else:
        print("⚠️  Some steps failed - review output above")
        return 1
    
    print("="*80 + "\n")
    return 0


if __name__ == "__main__":
    exit(main())
