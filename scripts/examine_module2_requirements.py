#!/usr/bin/env python3
"""
examine_module2_requirements.py - See what Module 2 actually requires
"""

import inspect
import re


def examine_module2():
    """Look at Module 2 source to find requirements."""
    
    print("="*80)
    print("MODULE 2 REQUIREMENTS ANALYSIS")
    print("="*80)
    print()
    
    from module_2_financial import compute_module_2_financial
    
    # Get source code
    source = inspect.getsource(compute_module_2_financial)
    
    print("Looking for skip/continue conditions...")
    print()
    
    # Find lines with 'continue' (record skipping)
    lines = source.split('\n')
    for i, line in enumerate(lines):
        if 'continue' in line and not line.strip().startswith('#'):
            # Print context around continue statement
            context_start = max(0, i-3)
            context_end = min(len(lines), i+2)
            
            print(f"Skip condition at line {i+1}:")
            for j in range(context_start, context_end):
                prefix = ">>> " if j == i else "    "
                print(f"{prefix}{lines[j]}")
            print()
    
    print("="*80)
    print()
    
    # Look for required fields
    print("Looking for field requirements...")
    print()
    
    # Common patterns that indicate required fields
    patterns = [
        (r'if not (\w+):', 'Required (must be truthy)'),
        (r'if (\w+) is None:', 'Cannot be None'),
        (r'if (\w+) == None:', 'Cannot be None'),
        (r'if not rec\.get\([\'"](\w+)[\'"]\)', 'Required from record'),
    ]
    
    for pattern, description in patterns:
        matches = re.finditer(pattern, source)
        for match in matches:
            field = match.group(1)
            print(f"  {description}: {field}")
    
    print()
    print("="*80)
    print()
    
    # Show the actual scoring loop
    print("Module 2 scoring loop structure:")
    print()
    
    in_loop = False
    indent_level = 0
    
    for line in lines:
        stripped = line.lstrip()
        
        # Detect loop start
        if 'for' in stripped and 'financial_records' in stripped:
            in_loop = True
            indent_level = len(line) - len(stripped)
            print(line)
            continue
        
        if in_loop:
            current_indent = len(line) - len(line.lstrip())
            
            # Still in loop
            if current_indent > indent_level or line.strip() == '':
                # Show important lines
                if any(keyword in stripped for keyword in ['continue', 'if', 'get', 'score', 'append', 'return']):
                    print(line)
            else:
                # Loop ended
                break
    
    print()
    print("="*80)


if __name__ == "__main__":
    examine_module2()
