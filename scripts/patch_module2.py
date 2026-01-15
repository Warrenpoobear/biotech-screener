#!/usr/bin/env python3
"""
patch_module2.py - Diagnose and fix Module 2 scoring issue
"""

from pathlib import Path
import re


def diagnose_module2():
    """Diagnose why Module 2 scores 0."""
    
    print("DIAGNOSING MODULE 2...")
    print()
    
    # Check if financial_records.json exists
    fin_path = Path("production_data/financial_records.json")
    if not fin_path.exists():
        print("❌ production_data/financial_records.json not found!")
        return False
    
    import json
    with open(fin_path) as f:
        records = json.load(f)
    
    print(f"✅ financial_records.json exists: {len(records)} records")
    
    # Check sample record
    if records:
        sample = records[0]
        print(f"   Sample ticker: {sample.get('ticker')}")
        print(f"   Has market_cap: {sample.get('market_cap') is not None}")
        print(f"   Has cash: {sample.get('cash') is not None}")
    
    print()
    
    # Check Module 2 signature
    try:
        from module_2_financial import compute_module_2_financial
        import inspect
        sig = inspect.signature(compute_module_2_financial)
        params = sig.parameters
        
        print("Module 2 signature:")
        for name, param in params.items():
            print(f"  {name}: {param.annotation}")
        print()
        
        # Check if it expects Set
        active_param = params.get('active_tickers')
        if active_param and 'Set' in str(active_param.annotation):
            print("⚠️  Module 2 expects Set[str] for active_tickers")
            print("   But run_screen.py might be passing List[str]")
            return 'set_conversion_needed'
        
    except Exception as e:
        print(f"⚠️  Could not inspect Module 2: {e}")
    
    return True


def apply_module2_patch():
    """Fix run_screen.py to convert active_tickers to Set."""
    
    file_path = Path("run_screen.py")
    
    if not file_path.exists():
        print(f"❌ {file_path} not found!")
        return False
    
    print(f"Patching {file_path}...")
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Find Module 2 call and add set conversion
    patched = False
    for i, line in enumerate(lines):
        if 'compute_module_2_financial' in line and i > 0:
            # Look for active_tickers parameter in next few lines
            for j in range(i, min(i+5, len(lines))):
                if 'active_tickers=' in lines[j] and 'set(' not in lines[j]:
                    # Check if it's using a list
                    if 'active_tickers=active_tickers' in lines[j]:
                        # Convert to set
                        lines[j] = lines[j].replace(
                            'active_tickers=active_tickers',
                            'active_tickers=set(active_tickers)'
                        )
                        patched = True
                        print(f"  ✅ Line {j+1}: Added set() conversion")
                        break
            break
    
    if not patched:
        print("  ⚠️  Could not find pattern to patch")
        return False
    
    # Write back
    with open(file_path, 'w') as f:
        f.writelines(lines)
    
    print(f"✅ Patched! Converted active_tickers to set()")
    return True


if __name__ == "__main__":
    import sys
    
    print("="*80)
    print("MODULE 2 DIAGNOSTIC & PATCH")
    print("="*80)
    print()
    
    result = diagnose_module2()
    
    if result == 'set_conversion_needed':
        print()
        print("APPLYING PATCH...")
        print()
        
        if apply_module2_patch():
            print()
            print("✅ Patch applied successfully!")
            print()
            print("Re-run screening:")
            print("  python run_screen.py --as-of-date 2026-01-06 --data-dir production_data --output screening_M2_FIXED.json")
            print()
            print("Expected: Module 2 scored 95/98 (was 0)")
            sys.exit(0)
        else:
            print()
            print("❌ Patch failed")
            sys.exit(1)
    elif result:
        print("✅ Module 2 setup looks correct")
        print()
        print("If still scoring 0, the issue may be elsewhere.")
        print("Try running with verbose output to see what Module 2 receives.")
        sys.exit(0)
    else:
        print("❌ Module 2 diagnostic failed")
        sys.exit(1)
