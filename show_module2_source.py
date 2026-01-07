#!/usr/bin/env python3
"""
show_module2_source.py - Display Module 2 source code
"""

import inspect


def show_module2_source():
    """Display Module 2 source code."""
    
    print("="*80)
    print("MODULE 2 SOURCE CODE")
    print("="*80)
    print()
    
    try:
        from module_2_financial import compute_module_2_financial
        
        source = inspect.getsource(compute_module_2_financial)
        
        # Print with line numbers
        for i, line in enumerate(source.split('\n'), 1):
            print(f"{i:3d} | {line}")
        
        print()
        print("="*80)
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    show_module2_source()
