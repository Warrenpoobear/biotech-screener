#!/usr/bin/env python3
"""
Wake Robin - Top 60 Securities Extractor
Extracts and formats the top 60 securities from screening results.
"""

import json
import sys
import csv
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any


def find_latest_results(directory: str = ".") -> Path:
    """Find the most recent results JSON file."""
    path = Path(directory)
    results_files = sorted(path.glob("results_*.json"), reverse=True)
    
    if results_files:
        return results_files[0]
    
    # Fall back to test files
    test_files = sorted(path.glob("test*.json"), reverse=True)
    if test_files:
        return test_files[0]
    
    raise FileNotFoundError("No results files found!")


def load_top_60(filepath: Path) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Load and return top 60 securities from results file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Get ranked securities
    ranked = data['module_5_composite']['ranked_securities']
    
    # Sort by composite rank
    ranked_sorted = sorted(ranked, key=lambda x: int(x.get('composite_rank', 999)))
    
    # Get top 60
    top_60 = ranked_sorted[:60]
    
    # Get metadata
    metadata = data.get('run_metadata', {})
    
    return top_60, metadata


def print_report(securities: List[Dict], metadata: Dict, filepath: Path):
    """Print formatted report to console."""
    print("=" * 80)
    print("WAKE ROBIN - TOP 60 SECURITIES REPORT")
    print(f"Source: {filepath.name}")
    print(f"Date: {metadata.get('as_of_date', 'Unknown')}")
    print("=" * 80)
    print()
    
    # Header
    print(f"{'Rank':<6}{'Ticker':<8}{'Score':<10}{'Weight':<10}{'Financial':<12}{'Clinical':<10}{'Catalyst':<10}")
    print("-" * 80)
    
    # Securities
    for sec in securities:
        rank = sec['composite_rank']
        ticker = sec['ticker']
        score = float(sec['composite_score'])
        weight = float(sec['position_weight']) * 100
        financial = float(sec.get('financial_normalized', 0))
        clinical = float(sec.get('clinical_dev_normalized', 0))
        catalyst = float(sec.get('catalyst_normalized', 50))
        
        print(f"{rank:<6}{ticker:<8}{score:<10.2f}{weight:>8.2f}%  {financial:>10.2f}  {clinical:>9.2f}  {catalyst:>9.2f}")
    
    # Summary
    print()
    print("=" * 80)
    total_weight = sum(float(s['position_weight']) for s in securities) * 100
    avg_score = sum(float(s['composite_score']) for s in securities) / len(securities)
    min_score = min(float(s['composite_score']) for s in securities)
    max_score = max(float(s['composite_score']) for s in securities)
    
    print(f"Total Weight: {total_weight:.2f}%")
    print(f"Positions: {len(securities)}")
    print(f"Score Range: {min_score:.2f} - {max_score:.2f}")
    print(f"Average Score: {avg_score:.2f}")
    print("=" * 80)


def save_to_text(securities: List[Dict], metadata: Dict, filepath: Path, output_file: Path):
    """Save report to text file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("WAKE ROBIN - TOP 60 SECURITIES REPORT\n")
        f.write(f"Source: {filepath.name}\n")
        f.write(f"Date: {metadata.get('as_of_date', 'Unknown')}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        # Header
        f.write(f"{'Rank':<6}{'Ticker':<8}{'Score':<10}{'Weight':<10}{'Financial':<12}{'Clinical':<10}{'Catalyst':<10}{'Stage':<8}\n")
        f.write("-" * 90 + "\n")
        
        # Securities
        for sec in securities:
            rank = sec['composite_rank']
            ticker = sec['ticker']
            score = float(sec['composite_score'])
            weight = float(sec['position_weight']) * 100
            financial = float(sec.get('financial_normalized', 0))
            clinical = float(sec.get('clinical_dev_normalized', 0))
            catalyst = float(sec.get('catalyst_normalized', 50))
            stage = sec.get('stage_bucket', 'unknown')
            
            f.write(f"{rank:<6}{ticker:<8}{score:<10.2f}{weight:>8.2f}%  {financial:>10.2f}  {clinical:>9.2f}  {catalyst:>9.2f}  {stage:<8}\n")
        
        # Summary
        f.write("\n" + "=" * 80 + "\n")
        total_weight = sum(float(s['position_weight']) for s in securities) * 100
        avg_score = sum(float(s['composite_score']) for s in securities) / len(securities)
        min_score = min(float(s['composite_score']) for s in securities)
        max_score = max(float(s['composite_score']) for s in securities)
        
        f.write(f"Total Weight: {total_weight:.2f}%\n")
        f.write(f"Positions: {len(securities)}\n")
        f.write(f"Score Range: {min_score:.2f} - {max_score:.2f}\n")
        f.write(f"Average Score: {avg_score:.2f}\n")
        f.write("=" * 80 + "\n")
    
    print(f"\nReport saved to: {output_file}")


def save_to_csv(securities: List[Dict], metadata: Dict, filepath: Path, output_file: Path):
    """Save report to CSV file."""
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([
            'Rank', 'Ticker', 'Composite_Score', 'Weight_Percent',
            'Financial_Score', 'Clinical_Score', 'Catalyst_Score',
            'Stage', 'Market_Cap_Bucket'
        ])
        
        # Securities
        for sec in securities:
            writer.writerow([
                sec['composite_rank'],
                sec['ticker'],
                f"{float(sec['composite_score']):.2f}",
                f"{float(sec['position_weight']) * 100:.2f}",
                f"{float(sec.get('financial_normalized', 0)):.2f}",
                f"{float(sec.get('clinical_dev_normalized', 0)):.2f}",
                f"{float(sec.get('catalyst_normalized', 50)):.2f}",
                sec.get('stage_bucket', 'unknown'),
                sec.get('market_cap_bucket', 'unknown')
            ])
    
    print(f"\nCSV saved to: {output_file}")


def main():
    """Main execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract top 60 securities from screening results')
    parser.add_argument('--file', '-f', type=str, help='Results JSON file (default: auto-detect latest)')
    parser.add_argument('--output', '-o', type=str, help='Output file (txt or csv)')
    parser.add_argument('--format', '-fmt', choices=['text', 'csv', 'both'], default='text',
                        help='Output format (default: text)')
    parser.add_argument('--quiet', '-q', action='store_true', help='Don\'t print to console')
    
    args = parser.parse_args()
    
    # Find results file
    if args.file:
        filepath = Path(args.file)
    else:
        try:
            filepath = find_latest_results()
            print(f"Auto-detected: {filepath.name}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            sys.exit(1)
    
    # Load data
    try:
        securities, metadata = load_top_60(filepath)
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)
    
    # Print to console
    if not args.quiet:
        print()
        print_report(securities, metadata, filepath)
    
    # Save to file
    if args.output:
        output_path = Path(args.output)
    else:
        # Auto-generate filename
        today = datetime.now().strftime('%Y-%m-%d')
        output_path = Path(f"top60_{today}.txt")
    
    if args.format in ['text', 'both']:
        txt_path = output_path.with_suffix('.txt')
        save_to_text(securities, metadata, filepath, txt_path)
    
    if args.format in ['csv', 'both']:
        csv_path = output_path.with_suffix('.csv')
        save_to_csv(securities, metadata, filepath, csv_path)


if __name__ == '__main__':
    main()
