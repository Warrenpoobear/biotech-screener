#!/usr/bin/env python3
"""
Transform collected clinical_data into Module 4's expected format.

Module 4 expects a flat list of trials with specific fields.
Our collection created nested structure per ticker.
This script flattens and transforms the data.
"""

import json
from pathlib import Path
from typing import List, Dict, Any


def transform_clinical_data_for_module4(universe_path: str) -> List[Dict[str, Any]]:
    """
    Transform universe clinical_data into Module 4's trial_records format.
    
    Input (in universe.json):
        {
          "ticker": "ARGX",
          "clinical_data": {
            "total_trials": 6,
            "top_trials": [
              {"nct_id": "...", "phase": "PHASE_3", "status": "COMPLETED", ...}
            ]
          }
        }
    
    Output (for Module 4):
        [
          {
            "ticker": "ARGX",
            "nct_id": "NCT04188379",
            "phase": "Phase 3",
            "status": "Completed",
            "randomized": False,  # Don't have this data
            "blinded": "",        # Don't have this data
            "primary_endpoint": "", # Don't have this data
          },
          ...
        ]
    """
    print("="*80)
    print("TRANSFORMING CLINICAL DATA FOR MODULE 4")
    print("="*80)
    print()
    
    with open(universe_path, 'r') as f:
        universe = json.load(f)
    
    print(f"Loaded {len(universe)} securities from {universe_path}")
    print()
    
    trial_records = []
    
    for security in universe:
        ticker = security.get('ticker', 'UNKNOWN')
        
        # Skip benchmark
        if ticker == '_XBI_BENCHMARK_':
            continue
        
        clinical_data = security.get('clinical_data', {})
        
        if not clinical_data:
            print(f"  ⚠️  {ticker}: No clinical_data field")
            continue
        
        # Get trials from top_trials field
        top_trials = clinical_data.get('top_trials', [])
        
        if not top_trials:
            total = clinical_data.get('total_trials', 0)
            if total > 0:
                print(f"  ⚠️  {ticker}: {total} trials but no top_trials data")
            continue
        
        # Transform each trial
        for trial in top_trials:
            # Normalize phase format for Module 4
            phase = trial.get('phase', 'N/A')
            if phase:
                # Convert "PHASE_3" → "Phase 3"
                phase = phase.replace('PHASE_', 'Phase ').replace('_', ' ')
            
            # Normalize status
            status = trial.get('status', '')
            if status:
                status = status.title()  # "COMPLETED" → "Completed"
            
            trial_record = {
                'ticker': ticker,
                'nct_id': trial.get('nct_id'),
                'phase': phase,
                'status': status,
                
                # Fields we didn't collect - use conservative defaults
                # Module 4 will use base scores when these are missing
                'randomized': False,  # Conservative: assume not randomized
                'blinded': '',        # Conservative: assume not blinded
                'primary_endpoint': '',  # Conservative: no endpoint bonus
                
                # PIT dates - use None to bypass PIT filtering
                # (We don't have historical data anyway)
                'last_update_posted': None,
                'source_date': None,
            }
            
            trial_records.append(trial_record)
        
        if top_trials:
            print(f"  ✓ {ticker}: {len(top_trials)} trials transformed")
    
    print()
    print(f"Total trial records created: {len(trial_records)}")
    print()
    
    return trial_records


def save_trial_records(trial_records: List[Dict], output_path: str):
    """Save trial records to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(trial_records, f, indent=2)
    
    print(f"✓ Saved {len(trial_records)} trial records to: {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Transform clinical data for Module 4")
    parser.add_argument("--universe", default="production_data/universe.json",
                       help="Input universe file")
    parser.add_argument("--output", default="production_data/trial_records.json",
                       help="Output trial records file")
    args = parser.parse_args()
    
    # Transform
    trial_records = transform_clinical_data_for_module4(args.universe)
    
    # Save
    save_trial_records(trial_records, args.output)
    
    # Summary
    print()
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print()
    
    tickers = set(t['ticker'] for t in trial_records)
    print(f"Tickers with trials: {len(tickers)}")
    print(f"Total trial records: {len(trial_records)}")
    print(f"Average trials per ticker: {len(trial_records)/len(tickers) if tickers else 0:.1f}")
    print()
    
    # Show phase distribution
    from collections import Counter
    phases = Counter(t['phase'] for t in trial_records)
    print("Phase distribution:")
    for phase, count in sorted(phases.items(), key=lambda x: -x[1]):
        print(f"  {phase}: {count}")
    
    print()
    print("="*80)
    print("NEXT STEPS")
    print("="*80)
    print()
    print("Now you need to pass this trial_records file to Module 4:")
    print()
    print("In your screening pipeline, load trial_records.json and pass to")
    print("compute_module_4_clinical_dev() as the first parameter.")
    print()
    print("Example:")
    print("  with open('production_data/trial_records.json') as f:")
    print("      trial_records = json.load(f)")
    print()
    print("  m4_output = compute_module_4_clinical_dev(")
    print("      trial_records=trial_records,  # ← Use this!")
    print("      active_tickers=active_tickers,")
    print("      as_of_date=as_of_date")
    print("  )")
    print()


if __name__ == "__main__":
    main()
