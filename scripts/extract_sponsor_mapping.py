#!/usr/bin/env python3
"""
extract_sponsor_mapping.py

Extracts ticker → sponsor mapping from existing trial_records.json
This mapping is needed for AACT refresh queries.
"""

import json
from pathlib import Path
from collections import defaultdict

def extract_sponsor_mapping(trial_records_path: str = 'trial_records.json') -> dict:
    """
    Extract unique ticker → sponsor(s) mapping from trial_records.json
    
    Returns:
        {
            'ARGX': ['argenx'],
            'NVAX': ['Novavax', 'Novavax Inc'],
            ...
        }
    """
    
    print(f"Loading {trial_records_path}...")
    with open(trial_records_path) as f:
        records = json.load(f)
    
    # Build mapping
    mapping = defaultdict(set)
    
    for record in records:
        ticker = record.get('ticker')
        sponsor = record.get('sponsor')
        
        if ticker and sponsor:
            mapping[ticker].add(sponsor)
    
    # Convert sets to sorted lists
    mapping_dict = {
        ticker: sorted(list(sponsors))
        for ticker, sponsors in sorted(mapping.items())
    }
    
    # Statistics
    total_tickers = len(mapping_dict)
    total_sponsors = sum(len(sponsors) for sponsors in mapping_dict.values())
    avg_sponsors = total_sponsors / total_tickers if total_tickers > 0 else 0
    
    print(f"\nExtracted mapping:")
    print(f"  Tickers: {total_tickers}")
    print(f"  Total sponsor names: {total_sponsors}")
    print(f"  Avg sponsors per ticker: {avg_sponsors:.1f}")
    
    # Show tickers with multiple sponsors
    multi_sponsor = {
        ticker: sponsors 
        for ticker, sponsors in mapping_dict.items() 
        if len(sponsors) > 1
    }
    
    if multi_sponsor:
        print(f"\nTickers with multiple sponsor names ({len(multi_sponsor)}):")
        for ticker, sponsors in sorted(multi_sponsor.items())[:10]:
            print(f"  {ticker}: {sponsors}")
        if len(multi_sponsor) > 10:
            print(f"  ... and {len(multi_sponsor) - 10} more")
    
    return mapping_dict


def save_mapping(mapping: dict, output_path: str = 'ticker_sponsor_mapping.json'):
    """Save mapping to JSON file"""
    with open(output_path, 'w') as f:
        json.dump(mapping, f, indent=2)
    print(f"\n✓ Saved mapping to {output_path}")


def save_as_python_dict(mapping: dict, output_path: str = 'ticker_sponsor_mapping.py'):
    """Save mapping as Python dict for easy import"""
    with open(output_path, 'w') as f:
        f.write('"""Ticker to sponsor mapping for AACT queries"""\n\n')
        f.write('TICKER_TO_SPONSORS = {\n')
        
        for ticker, sponsors in sorted(mapping.items()):
            sponsors_str = repr(sponsors)
            f.write(f"    '{ticker}': {sponsors_str},\n")
        
        f.write('}\n')
    
    print(f"✓ Saved mapping to {output_path}")


def validate_mapping(mapping: dict):
    """Check for potential issues in mapping"""
    issues = []
    
    # Check for empty sponsor lists
    empty = [ticker for ticker, sponsors in mapping.items() if not sponsors]
    if empty:
        issues.append(f"Tickers with no sponsors: {empty}")
    
    # Check for very long sponsor names (might be malformed)
    long_names = []
    for ticker, sponsors in mapping.items():
        for sponsor in sponsors:
            if len(sponsor) > 100:
                long_names.append((ticker, sponsor))
    
    if long_names:
        issues.append(f"Unusually long sponsor names: {long_names[:5]}")
    
    # Check for suspicious characters
    suspicious = []
    for ticker, sponsors in mapping.items():
        for sponsor in sponsors:
            if any(char in sponsor for char in ['|', '\n', '\r', '\t']):
                suspicious.append((ticker, sponsor))
    
    if suspicious:
        issues.append(f"Sponsors with suspicious characters: {suspicious[:5]}")
    
    if issues:
        print("\n⚠️  Validation warnings:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n✓ Validation passed - no issues found")
    
    return len(issues) == 0


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Extract ticker → sponsor mapping from trial_records.json'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='trial_records.json',
        help='Path to trial_records.json'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='ticker_sponsor_mapping.json',
        help='Output path for JSON mapping'
    )
    parser.add_argument(
        '--python',
        action='store_true',
        help='Also save as Python dict for easy import'
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("TICKER → SPONSOR MAPPING EXTRACTOR")
    print("="*80)
    
    # Extract mapping
    mapping = extract_sponsor_mapping(args.input)
    
    # Validate
    validate_mapping(mapping)
    
    # Save
    save_mapping(mapping, args.output)
    
    if args.python:
        python_path = args.output.replace('.json', '.py')
        save_as_python_dict(mapping, python_path)
    
    print("\n" + "="*80)
    print("✓ Extraction complete!")
    print("="*80)
    print("\nNext steps:")
    print("1. Review ticker_sponsor_mapping.json")
    print("2. Register for AACT access at https://aact.ctti-clinicaltrials.org/users/sign_up")
    print("3. Use this mapping in your AACT refresh script")
    print("="*80)
