#!/usr/bin/env python3
"""
patch_module2_add_diagnostic_counts.py

Add diagnostic_counts to Module 2's return value so run_screen.py can see it scored securities.
"""

from pathlib import Path
import re


def patch_module2():
    """Add diagnostic_counts to Module 2's return statement."""
    
    file_path = Path("module_2_financial.py")
    
    if not file_path.exists():
        print(f"❌ {file_path} not found!")
        return False
    
    print("="*80)
    print("PATCHING MODULE 2 TO ADD DIAGNOSTIC_COUNTS")
    print("="*80)
    print()
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check if already patched
    if '"diagnostic_counts"' in content:
        print("✅ Already patched - diagnostic_counts exists")
        return True
    
    # Find the return statement and add diagnostic_counts
    # Look for the return block at the end of compute_module_2_financial
    
    pattern = r'(return \{[\s\S]*?"coverage_stats": coverage_stats,[\s\S]*?"as_of_date": as_of_date,)\s*\}'
    
    replacement = r'''\1
        "diagnostic_counts": {
            "scored": len(securities),
            "missing": coverage_stats["total_active"] - len(securities),
        },
    }'''
    
    new_content, count = re.subn(pattern, replacement, content)
    
    if count == 0:
        print("❌ Could not find return statement to patch")
        print()
        print("Manual fix needed:")
        print()
        print("In module_2_financial.py, change the return statement from:")
        print()
        print("    return {")
        print('        "securities": securities,')
        print('        "coverage_stats": coverage_stats,')
        print('        "as_of_date": as_of_date,')
        print("    }")
        print()
        print("To:")
        print()
        print("    return {")
        print('        "securities": securities,')
        print('        "coverage_stats": coverage_stats,')
        print('        "as_of_date": as_of_date,')
        print('        "diagnostic_counts": {')
        print('            "scored": len(securities),')
        print('            "missing": coverage_stats["total_active"] - len(securities),')
        print('        },')
        print("    }")
        return False
    
    # Write back
    with open(file_path, 'w') as f:
        f.write(new_content)
    
    print("✅ Patched! Added diagnostic_counts to Module 2 return value")
    print()
    print("Module 2 will now report:")
    print('  - diagnostic_counts["scored"]: number of securities scored')
    print('  - diagnostic_counts["missing"]: number with no data')
    print()
    return True


if __name__ == "__main__":
    import sys
    
    success = patch_module2()
    
    print("="*80)
    print()
    
    if success:
        print("✅ Patch applied successfully!")
        print()
        print("Re-run screening:")
        print("  python run_screen.py --as-of-date 2026-01-06 --data-dir production_data --output screening_FINAL.json")
        print()
        print("Expected: Module 2 scored 97/98 (was 0)")
        sys.exit(0)
    else:
        print("❌ Patch failed - manual fix needed (see instructions above)")
        sys.exit(1)
