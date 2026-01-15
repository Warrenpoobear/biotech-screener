#!/usr/bin/env python3
"""
verify_all_patches.py - Verify all patches have been applied correctly
"""

from pathlib import Path


def verify_financial_path():
    """Check if run_screen.py uses financial_records.json"""
    file_path = Path("run_screen.py")
    if not file_path.exists():
        return False, "run_screen.py not found"
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    if 'financial_records.json' in content:
        return True, "Using financial_records.json"
    else:
        return False, "Still using financial.json"


def verify_topn():
    """Check if module_5 has top_n parameter"""
    file_path = Path("module_5_composite_with_defensive.py")
    if not file_path.exists():
        return False, "module_5_composite_with_defensive.py not found"
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    if 'top_n=' in content or 'top_n =' in content:
        return True, "Has top_n parameter"
    else:
        return False, "Missing top_n parameter"


def verify_module2_set():
    """Check if Module 2 gets set(active_tickers)"""
    file_path = Path("run_screen.py")
    if not file_path.exists():
        return False, "run_screen.py not found"
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    if 'active_tickers=set(active_tickers)' in content:
        return True, "Using set(active_tickers)"
    else:
        return False, "Not converting to set"


def main():
    print("="*80)
    print("VERIFYING PATCHES")
    print("="*80)
    print()
    
    checks = [
        ("Financial path", verify_financial_path),
        ("Top-N parameter", verify_topn),
        ("Module 2 Set conversion", verify_module2_set),
    ]
    
    results = []
    
    for name, check_func in checks:
        try:
            success, message = check_func()
            status = "✅" if success else "❌"
            print(f"{status} {name}: {message}")
            results.append(success)
        except Exception as e:
            print(f"❌ {name}: Error - {e}")
            results.append(False)
    
    print()
    print("="*80)
    
    if all(results):
        print("✅ ALL PATCHES VERIFIED")
        print("="*80)
        print()
        print("Ready to run:")
        print("  python run_screen.py --as-of-date 2026-01-06 --data-dir production_data --output screening_FINAL.json")
        print()
        return 0
    else:
        print("❌ SOME PATCHES MISSING")
        print("="*80)
        print()
        failed = [name for (name, _), success in zip(checks, results) if not success]
        print("Failed checks:")
        for name in failed:
            print(f"  • {name}")
        print()
        print("Run: python apply_all_patches.py")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
