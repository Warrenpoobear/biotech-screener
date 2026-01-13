"""
Quick helper to update comparison data with new validation results.

Usage:
    python update_comparison.py 2023-Q2 \
        --spread 8.95 --alpha 4.21 --q1-return 8.64 --q1-alpha 12.45 \
        --hit-rate 38.5 --tickers 280
"""

import argparse
import re
from pathlib import Path


def update_enhanced_results(quarter: str, spread: float, alpha: float,
                            q1_return: float, q1_alpha: float,
                            hit_rate: float, tickers: int):
    """Add a new enhanced result to compare_validations.py."""
    
    script_path = Path('compare_validations.py')
    
    if not script_path.exists():
        script_path = Path('scripts/compare_validations.py')
    
    if not script_path.exists():
        print("❌ Could not find compare_validations.py")
        print("   Make sure you're in the correct directory")
        return False
    
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Extract screen_date from baseline results
    pattern = f"'{quarter}'.*?'screen_date': '([0-9-]+)'"
    match = re.search(pattern, content, re.DOTALL)
    
    if not match:
        print(f"❌ Could not find {quarter} in baseline results")
        print(f"   Available quarters: 2023-Q1, 2023-Q2, 2023-Q3, 2023-Q4, 2024-Q1, 2024-Q2, 2024-Q3")
        return False
    
    screen_date = match.group(1)
    
    # Create new entry
    new_entry = f"""        '{quarter}': {{
            'screen_date': '{screen_date}',
            'q1_q5_spread': {spread:.2f},
            'avg_alpha': {alpha:.2f},
            'q1_return': {q1_return:.2f},
            'q1_alpha': {q1_alpha:.2f},
            'hit_rate': {hit_rate:.2f},
            'tickers_tested': {tickers}
        }},"""
    
    # Find insertion point in enhanced results
    enhanced_section_start = content.find("def load_enhanced_results()")
    enhanced_return_start = content.find("return {", enhanced_section_start)
    
    # Check if this quarter already exists
    if f"'{quarter}':" in content[enhanced_return_start:]:
        # Replace existing entry
        # Find the start of this quarter's entry
        quarter_start = content.find(f"'{quarter}':", enhanced_return_start)
        # Find the end (next closing brace followed by comma)
        quarter_end = content.find("},", quarter_start) + 2
        
        # Replace
        content = content[:quarter_start] + new_entry.strip() + "\n" + content[quarter_end:]
        print(f"✅ Updated {quarter} in enhanced results")
    else:
        # Find where to insert (before closing brace of return dict)
        # Look for the last closing brace before "def main()"
        main_func_pos = content.find("def main()", enhanced_return_start)
        # Find last } before that
        insert_pos = content.rfind("}", enhanced_return_start, main_func_pos)
        
        # Check if this is the closing brace of the return dict
        # by looking backwards for "return {"
        return_brace_pos = content.rfind("return {", 0, insert_pos)
        
        if return_brace_pos > enhanced_section_start:
            # This is the right closing brace
            # Check if there are existing entries
            if content[insert_pos-1] == '{':
                # First entry (empty dict)
                content = content[:insert_pos] + "\n" + new_entry + "\n        " + content[insert_pos:]
            else:
                # Additional entry
                content = content[:insert_pos] + "\n" + new_entry + "\n        " + content[insert_pos:]
            
            print(f"✅ Added {quarter} to enhanced results")
        else:
            print(f"❌ Could not find insertion point")
            return False
    
    # Write back
    with open(script_path, 'w') as f:
        f.write(content)
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Update comparison data with new validation results'
    )
    parser.add_argument('quarter', help='Quarter ID (e.g., 2023-Q2)')
    parser.add_argument('--spread', type=float, required=True, help='Q1-Q5 spread')
    parser.add_argument('--alpha', type=float, required=True, help='Average alpha')
    parser.add_argument('--q1-return', type=float, required=True, help='Q1 return')
    parser.add_argument('--q1-alpha', type=float, required=True, help='Q1 alpha')
    parser.add_argument('--hit-rate', type=float, required=True, help='Hit rate')
    parser.add_argument('--tickers', type=int, required=True, help='Tickers tested')
    
    args = parser.parse_args()
    
    print(f"\nUpdating comparison data for {args.quarter}...")
    print(f"  Q1-Q5 Spread: {args.spread:+.2f}%")
    print(f"  Avg Alpha: {args.alpha:+.2f}%")
    print(f"  Q1 Return: {args.q1_return:+.2f}%")
    print(f"  Q1 Alpha: {args.q1_alpha:+.2f}%")
    print(f"  Hit Rate: {args.hit_rate:.2f}%")
    print(f"  Tickers: {args.tickers}")
    print()
    
    success = update_enhanced_results(
        quarter=args.quarter,
        spread=args.spread,
        alpha=args.alpha,
        q1_return=args.q1_return,
        q1_alpha=args.q1_alpha,
        hit_rate=args.hit_rate,
        tickers=args.tickers
    )
    
    if success:
        print(f"\n✅ Successfully updated {args.quarter}")
        print(f"\nRun to see updated comparison:")
        print(f"  python compare_validations.py")
        print(f"\nOr generate report:")
        print(f"  python compare_validations.py --output reports/validation_comparison.md")
    else:
        print(f"\n❌ Failed to update {args.quarter}")
        print(f"   Check error messages above")


if __name__ == '__main__':
    main()
