#!/usr/bin/env python3
"""
fix_run_screen.py

Automatically fixes all diagnostic_counts issues in run_screen.py
Creates a backup before making changes.
"""

import os
import shutil
import re

def fix_run_screen():
    filename = "run_screen.py"
    backup = "run_screen.py.backup"
    
    # Create backup
    print(f"Creating backup: {backup}")
    shutil.copy2(filename, backup)
    
    # Read file
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("Applying fixes...")
    
    # Fix 1: Module 2 (lines 191-192)
    content = content.replace(
        """    print(f"  Scored: {m2_result['diagnostic_counts']['scored']}, "
          f"Missing: {m2_result['diagnostic_counts']['missing']}")""",
        """    diag = m2_result.get('diagnostic_counts', {})
    print(f"  Scored: {diag.get('scored', len(m2_result.get('scores', [])))}, "
          f"Missing: {diag.get('missing', 'N/A')}")"""
    )
    
    # Fix 2: Module 3 (lines 201-202)
    content = content.replace(
        """    print(f"  With catalyst: {m3_result['diagnostic_counts']['with_catalyst']}, "
          f"No catalyst: {m3_result['diagnostic_counts']['no_catalyst']}")""",
        """    diag = m3_result.get('diagnostic_counts', {})
    print(f"  With catalyst: {diag.get('with_catalyst', 'N/A')}, "
          f"No catalyst: {diag.get('no_catalyst', 'N/A')}")"""
    )
    
    # Fix 3: Module 4 (lines 211-213)
    content = content.replace(
        """    print(f"  Scored: {m4_result['diagnostic_counts']['scored']}, "
          f"Trials evaluated: {m4_result['diagnostic_counts']['total_trials']}, "
          f"PIT filtered: {m4_result['diagnostic_counts']['pit_filtered']}")""",
        """    diag = m4_result.get('diagnostic_counts', {})
    print(f"  Scored: {diag.get('scored', len(m4_result.get('scores', [])))}, "
          f"Trials evaluated: {diag.get('total_trials', 'N/A')}, "
          f"PIT filtered: {diag.get('pit_filtered', 'N/A')}")"""
    )
    
    # Fix 4: Module 5 (lines 228-229)
    content = content.replace(
        """    print(f"  Rankable: {m5_result['diagnostic_counts']['rankable']}, "
          f"Excluded: {m5_result['diagnostic_counts']['excluded']}")""",
        """    diag = m5_result.get('diagnostic_counts', {})
    print(f"  Rankable: {diag.get('rankable', len(m5_result.get('ranked_securities', [])))}, "
          f"Excluded: {diag.get('excluded', len(m5_result.get('excluded_securities', [])))}")"""
    )
    
    # Fix 5: Summary section - total_evaluated
    content = re.sub(
        r'"total_evaluated":\s*m1_result\["diagnostic_counts"\]\["total_input"\]',
        '"total_evaluated": m1_result.get("diagnostic_counts", {}).get("total_input", len(active_tickers))',
        content
    )
    
    # Fix 6: Summary section - excluded
    content = re.sub(
        r'"excluded":\s*m1_result\["diagnostic_counts"\]\["excluded"\]',
        '"excluded": m1_result.get("diagnostic_counts", {}).get("excluded", 0)',
        content
    )
    
    # Fix 7: Summary section - final_ranked
    content = re.sub(
        r'"final_ranked":\s*m5_result\["diagnostic_counts"\]\["rankable"\]',
        '"final_ranked": m5_result.get("diagnostic_counts", {}).get("rankable", len(m5_result.get("ranked_securities", [])))',
        content
    )
    
    # Write fixed content
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("\n✓ Fixes applied successfully!")
    print(f"✓ Backup saved as: {backup}")
    print("\nTo test the fixed file:")
    print("  python run_screen.py --as-of-date 2026-01-06 --data-dir production_data --output outputs/test.json")
    print("\nTo restore backup if needed:")
    print(f"  copy {backup} {filename}  (Windows)")
    print(f"  cp {backup} {filename}  (Linux/Mac)")

if __name__ == "__main__":
    try:
        fix_run_screen()
    except FileNotFoundError:
        print("ERROR: run_screen.py not found in current directory")
        print("Please run this script from the directory containing run_screen.py")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
