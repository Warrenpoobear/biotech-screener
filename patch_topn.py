#!/usr/bin/env python3
"""
patch_topn.py - Add top_n=60 to module_5_composite_with_defensive.py
"""

from pathlib import Path
import re


def apply_topn_patch():
    """Add top_n=60 parameter to enrich_with_defensive_overlays call."""
    
    file_path = Path("module_5_composite_with_defensive.py")
    
    if not file_path.exists():
        print(f"❌ {file_path} not found!")
        return False
    
    print(f"Patching {file_path}...")
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check if already patched
    if 'top_n=' in content:
        print("✅ Already patched - top_n parameter exists")
        return True
    
    # Find the enrich_with_defensive_overlays call and add top_n
    pattern = r'(enrich_with_defensive_overlays\([^)]*apply_position_sizing=apply_position_sizing,)\s*\)'
    replacement = r'\1\n    top_n=60,  # Enable top-N selection\n)'
    
    new_content, count = re.subn(pattern, replacement, content, flags=re.DOTALL)
    
    if count == 0:
        # Try alternate pattern
        pattern2 = r'(apply_position_sizing=apply_position_sizing,)\s*\n\s*\)'
        replacement2 = r'\1\n    top_n=60,  # Enable top-N selection\n    )'
        new_content, count = re.subn(pattern2, replacement2, content)
    
    if count == 0:
        print("❌ Could not find pattern to patch")
        print("Manual fix needed - add 'top_n=60,' after 'apply_position_sizing=apply_position_sizing,'")
        return False
    
    # Write back
    with open(file_path, 'w') as f:
        f.write(new_content)
    
    print(f"✅ Patched! Added top_n=60 parameter")
    return True


if __name__ == "__main__":
    import sys
    
    print("="*80)
    print("TOP-N PATCH")
    print("="*80)
    print()
    
    success = apply_topn_patch()
    
    print()
    if success:
        print("✅ Patch applied successfully!")
        print()
        print("Re-run screening:")
        print("  python run_screen.py --as-of-date 2026-01-06 --data-dir production_data --output screening_TOPN.json")
        print()
        print("Expected: Max weight 5.20%, 60 positions")
        sys.exit(0)
    else:
        print("❌ Patch failed - see error above")
        sys.exit(1)
