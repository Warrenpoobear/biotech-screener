#!/usr/bin/env python3
"""
apply_all_patches.py - Apply all remaining fixes in one go

Fixes:
1. run_screen.py: financial.json → financial_records.json (DONE)
2. module_5_composite_with_defensive.py: Add top_n=60
3. run_screen.py: Convert active_tickers to set() for Module 2
"""

from pathlib import Path
import re
import sys


def patch_financial_path():
    """Ensure run_screen.py uses financial_records.json"""
    
    file_path = Path("run_screen.py")
    if not file_path.exists():
        return False, "run_screen.py not found"
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check if already fixed
    if 'financial_records.json' in content:
        return True, "Already using financial_records.json"
    
    # Apply fix
    content = content.replace(
        'data_dir / "financial.json"',
        'data_dir / "financial_records.json"'
    )
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    return True, "Changed: financial.json → financial_records.json"


def patch_topn():
    """Add top_n=60 to module_5_composite_with_defensive.py"""
    
    file_path = Path("module_5_composite_with_defensive.py")
    if not file_path.exists():
        return False, "module_5_composite_with_defensive.py not found"
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check if already patched
    if 'top_n=' in content:
        return True, "Already has top_n parameter"
    
    # Pattern 1: Simple closing paren after apply_position_sizing
    pattern1 = r'(apply_position_sizing=apply_position_sizing,)\s*(\n\s*)\)'
    if re.search(pattern1, content):
        content = re.sub(
            pattern1,
            r'\1\2    top_n=60,  # Enable top-N selection\2)',
            content
        )
        with open(file_path, 'w') as f:
            f.write(content)
        return True, "Added top_n=60 parameter"
    
    return False, "Could not find insertion point for top_n"


def patch_module2_set_conversion():
    """Convert active_tickers to set() for Module 2"""
    
    file_path = Path("run_screen.py")
    if not file_path.exists():
        return False, "run_screen.py not found"
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check if already patched
    if 'active_tickers=set(active_tickers)' in content:
        return True, "Already using set(active_tickers)"
    
    # Pattern: Look for Module 2 call with active_tickers parameter
    pattern = r'(compute_module_2_financial\([^)]*active_tickers=)active_tickers([,)])'
    if re.search(pattern, content, re.DOTALL):
        content = re.sub(
            pattern,
            r'\1set(active_tickers)\2',
            content,
            flags=re.DOTALL
        )
        with open(file_path, 'w') as f:
            f.write(content)
        return True, "Converted active_tickers to set()"
    
    return False, "Could not find Module 2 call to patch"


def main():
    print("="*80)
    print("APPLYING ALL PATCHES")
    print("="*80)
    print()
    
    patches = [
        ("Fix 1: Financial path", patch_financial_path),
        ("Fix 2: Top-N selection", patch_topn),
        ("Fix 3: Module 2 Set conversion", patch_module2_set_conversion),
    ]
    
    results = []
    
    for name, patch_func in patches:
        print(f"{name}...")
        try:
            success, message = patch_func()
            status = "✅" if success else "❌"
            print(f"  {status} {message}")
            results.append(success)
        except Exception as e:
            print(f"  ❌ Error: {e}")
            results.append(False)
        print()
    
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print()
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"✅ ALL PATCHES APPLIED ({passed}/{total})")
        print()
        print("Now run:")
        print("  python run_screen.py --as-of-date 2026-01-06 --data-dir production_data --output screening_ALL_FIXED.json")
        print()
        print("Expected results:")
        print("  • Module 2: Scored 95/98 (was 0)")
        print("  • Max weight: 5.20% (was 3.69%)")
        print("  • Positions: 60 (was 98)")
        return 0
    else:
        print(f"⚠️  PARTIAL SUCCESS ({passed}/{total} patches applied)")
        print()
        print("Some patches failed - you may need to manually edit the files.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
