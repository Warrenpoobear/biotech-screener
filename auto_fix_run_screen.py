#!/usr/bin/env python3
"""
auto_fix_run_screen.py - Automatically Fix EventType JSON Serialization

This script will patch your run_screen.py file to handle Enum serialization.

Usage:
    python auto_fix_run_screen.py
"""

import re
from pathlib import Path


def fix_run_screen():
    """Automatically patch run_screen.py"""
    
    script_path = Path('run_screen.py')
    
    if not script_path.exists():
        print("❌ run_screen.py not found in current directory")
        return False
    
    print("="*80)
    print("AUTO-FIX: run_screen.py - EventType JSON Serialization")
    print("="*80)
    
    # Read current file
    with open(script_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Backup original
    backup_path = Path('run_screen.py.backup')
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ Backed up to: {backup_path}")
    
    # Fix 1: Add enum import if not present
    if 'from enum import Enum' not in content:
        print("\n📝 Adding enum import...")
        
        # Find import section
        import_pattern = r'(from decimal import Decimal\s*\n)'
        if re.search(import_pattern, content):
            content = re.sub(
                import_pattern,
                r'\1from enum import Enum\n',
                content
            )
            print("   ✅ Added: from enum import Enum")
        else:
            # Alternative: add after datetime import
            import_pattern = r'(from datetime import date\s*\n)'
            if re.search(import_pattern, content):
                content = re.sub(
                    import_pattern,
                    r'\1from enum import Enum\n',
                    content
                )
                print("   ✅ Added: from enum import Enum")
            else:
                print("   ⚠️  Could not find import section - add manually")
    else:
        print("\n✅ Enum import already present")
    
    # Fix 2: Update CustomJSONEncoder
    print("\n📝 Updating CustomJSONEncoder...")
    
    # Pattern to find the CustomJSONEncoder class
    old_pattern = r'(class CustomJSONEncoder\(json\.JSONEncoder\):.*?def default\(self, obj\):.*?if isinstance\(obj, date\):.*?return obj\.isoformat\(\)\s*)(return super\(\)\.default\(obj\))'
    
    new_code = r'\1if isinstance(obj, Enum):  # Handle enums from Module 3A\n            return obj.value\n        \2'
    
    if re.search(old_pattern, content, re.DOTALL):
        # Check if enum handling already exists
        if 'isinstance(obj, Enum)' in content:
            print("   ✅ Enum handling already present")
        else:
            content = re.sub(old_pattern, new_code, content, flags=re.DOTALL)
            print("   ✅ Added enum handling to CustomJSONEncoder")
    else:
        print("   ⚠️  Could not find CustomJSONEncoder pattern")
        print("   → Manual update required (see instructions below)")
    
    # Write updated file
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"\n✅ Updated: {script_path}")
    
    # Show what was changed
    print("\n" + "="*80)
    print("CHANGES SUMMARY")
    print("="*80)
    print("1. Added import: from enum import Enum")
    print("2. Updated CustomJSONEncoder to handle Enum types")
    print("\nIf auto-fix failed, manually apply changes from EXACT_FIX_run_screen.py")
    print("="*80)
    
    return True


def main():
    print("\n⚠️  WARNING: This will modify run_screen.py")
    print("   A backup will be saved as run_screen.py.backup")
    
    response = input("\nProceed with auto-fix? (yes/no): ")
    
    if response.lower() not in ['yes', 'y']:
        print("❌ Cancelled")
        return 1
    
    success = fix_run_screen()
    
    if success:
        print("\n" + "="*80)
        print("✅ FIX COMPLETE!")
        print("="*80)
        print("\nNext step:")
        print("  python run_screen.py --as-of-date 2026-01-07 --data-dir production_data --output screening_complete.json")
        print("="*80 + "\n")
        return 0
    else:
        return 1


if __name__ == "__main__":
    exit(main())
