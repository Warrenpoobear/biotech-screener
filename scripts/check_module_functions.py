#!/usr/bin/env python3
"""
Check what functions exist in each module and fix imports
"""

import re

print("="*80)
print("MODULE FUNCTION NAME CHECKER")
print("="*80)

modules = [
    ('module_2_financial.py', 'compute_module_2_financial'),
    ('module_4_clinical_dev.py', 'compute_module_4_clinical_dev'),
    ('module_5_composite.py', 'compute_module_5_composite'),
]

fixes_needed = {}

for module_file, expected_func in modules:
    try:
        with open(module_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find all function definitions
        functions = re.findall(r'^def (\w+)\(', content, re.MULTILINE)
        
        print(f"\n{module_file}:")
        print(f"  Expected: {expected_func}")
        
        if expected_func in functions:
            print(f"  ✅ Found!")
        else:
            print(f"  ❌ NOT FOUND")
            print(f"  Available functions:")
            for func in functions[:10]:  # Show first 10
                print(f"    - {func}")
            
            # Try to guess the right function
            likely = [f for f in functions if 'module' in f.lower() or 'run' in f.lower() or 'compute' in f.lower()]
            if likely:
                print(f"  Likely correct function: {likely[0]}")
                fixes_needed[expected_func] = (module_file, likely[0])
    
    except FileNotFoundError:
        print(f"\n{module_file}: ❌ FILE NOT FOUND")

if fixes_needed:
    print("\n" + "="*80)
    print("FIXES NEEDED IN run_screen.py")
    print("="*80)
    
    for expected, (module, actual) in fixes_needed.items():
        print(f"\nChange this:")
        print(f"  from {module[:-3]} import {expected}")
        print(f"To this:")
        print(f"  from {module[:-3]} import {actual} as {expected}")

print("\n" + "="*80)
