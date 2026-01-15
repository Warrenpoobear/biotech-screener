#!/usr/bin/env python3
"""
verify_fixes.py

Check if the two fixes have been applied correctly.
"""

from pathlib import Path


def verify_run_screen():
    """Verify run_screen.py uses financial_records.json"""
    
    file_path = Path("run_screen.py")
    
    if not file_path.exists():
        print(f"❌ {file_path} not found!")
        return False
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check for new line
    if 'data_dir / "financial_records.json"' in content:
        print("✅ Fix 1: run_screen.py uses financial_records.json")
        return True
    elif 'data_dir / "financial.json"' in content:
        print("❌ Fix 1: run_screen.py still uses financial.json (not fixed)")
        return False
    else:
        print("⚠️  Fix 1: Could not find financial data loading line")
        return False


def verify_module_5():
    """Verify module_5_composite_with_defensive.py has top_n parameter"""
    
    file_path = Path("module_5_composite_with_defensive.py")
    
    if not file_path.exists():
        print(f"❌ {file_path} not found!")
        return False
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check for top_n parameter
    if 'top_n=60' in content or 'top_n = 60' in content:
        print("✅ Fix 2: module_5_composite_with_defensive.py has top_n=60")
        return True
    elif 'enrich_with_defensive_overlays' in content:
        print("❌ Fix 2: module_5_composite_with_defensive.py missing top_n parameter")
        return False
    else:
        print("⚠️  Fix 2: Could not find enrich_with_defensive_overlays call")
        return False


def main():
    print("="*80)
    print("VERIFYING FIXES")
    print("="*80)
    print()
    
    results = []
    
    results.append(verify_run_screen())
    results.append(verify_module_5())
    
    print()
    print("="*80)
    
    if all(results):
        print("✅ ALL FIXES VERIFIED!")
        print("="*80)
        print()
        print("Ready to run:")
        print("  python run_screen.py --as-of-date 2026-01-06 --data-dir production_data --output screening_FIXED.json")
        print()
        return 0
    else:
        print("❌ FIXES NOT APPLIED")
        print("="*80)
        print()
        print("Run this to apply automatically:")
        print("  python apply_fixes_automatically.py")
        print()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
