#!/usr/bin/env python3
"""
apply_fixes_automatically.py

Automatically applies both fixes:
1. Update run_screen.py: financial.json → financial_records.json
2. Update module_5_composite_with_defensive.py: add top_n=60
"""

import sys
from pathlib import Path


def fix_run_screen():
    """Fix run_screen.py to use financial_records.json"""
    
    file_path = Path("run_screen.py")
    
    if not file_path.exists():
        print(f"❌ {file_path} not found!")
        return False
    
    print(f"Fixing {file_path}...")
    
    # Read file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Make replacement
    old_line = 'data_dir / "financial.json"'
    new_line = 'data_dir / "financial_records.json"'
    
    if old_line not in content:
        print(f"  ⚠️  Line not found or already fixed")
        print(f"     Looking for: {old_line}")
        
        # Check if already fixed
        if new_line in content:
            print(f"  ✅ Already using financial_records.json")
            return True
        else:
            print(f"  ❌ Neither old nor new line found - manual fix needed")
            return False
    
    # Apply fix
    content = content.replace(old_line, new_line)
    
    # Write back
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"  ✅ Fixed! Changed: financial.json → financial_records.json")
    return True


def fix_module_5_composite():
    """Fix module_5_composite_with_defensive.py to add top_n=60"""
    
    file_path = Path("module_5_composite_with_defensive.py")
    
    if not file_path.exists():
        print(f"❌ {file_path} not found!")
        return False
    
    print(f"Fixing {file_path}...")
    
    # Read file
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Find the enrich_with_defensive_overlays call
    found_call = False
    already_has_topn = False
    insert_line = None
    
    for i, line in enumerate(lines):
        if 'enrich_with_defensive_overlays(' in line:
            found_call = True
            
        if found_call and 'top_n=' in line:
            already_has_topn = True
            print(f"  ✅ Already has top_n parameter (line {i+1})")
            return True
            
        if found_call and 'apply_position_sizing=True' in line:
            # Insert top_n after this line
            insert_line = i + 1
            break
    
    if not found_call:
        print(f"  ❌ Could not find enrich_with_defensive_overlays call")
        return False
    
    if already_has_topn:
        return True
    
    if insert_line is None:
        print(f"  ❌ Could not find apply_position_sizing=True line")
        return False
    
    # Insert top_n parameter
    indent = '    '  # Match existing indentation
    new_line = f'{indent}top_n=60,  # Enable top-N selection\n'
    lines.insert(insert_line, new_line)
    
    # Write back
    with open(file_path, 'w') as f:
        f.writelines(lines)
    
    print(f"  ✅ Fixed! Added top_n=60 at line {insert_line+1}")
    return True


def main():
    print("="*80)
    print("APPLYING FIXES AUTOMATICALLY")
    print("="*80)
    print()
    
    results = []
    
    # Fix 1: run_screen.py
    print("Fix 1: Module 2 (financial data)")
    results.append(fix_run_screen())
    print()
    
    # Fix 2: module_5_composite_with_defensive.py
    print("Fix 2: Top-N selection")
    results.append(fix_module_5_composite())
    print()
    
    # Summary
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print()
    
    if all(results):
        print("✅ All fixes applied successfully!")
        print()
        print("Now run:")
        print("  python run_screen.py --as-of-date 2026-01-06 --data-dir production_data --output screening_FIXED.json")
        print()
        print("Expected results:")
        print("  • Module 2: Scored 95/98 (was 0)")
        print("  • Max weight: 5.20% (was 3.69%)")
        print("  • Positions: 60 (was 98)")
        return 0
    else:
        print("⚠️  Some fixes failed - see errors above")
        print()
        print("You may need to manually edit the files.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
