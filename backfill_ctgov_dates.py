#!/usr/bin/env python3
"""
backfill_ctgov_dates.py - Add missing date fields to trial_records.json

Queries CT.gov API v2 to backfill:
- last_update_posted
- primary_completion_date/type
- completion_date/type
- results_first_posted
"""

import json
import requests
import time
from pathlib import Path
from typing import Optional


def fetch_ctgov_dates(nct_id: str) -> Optional[dict]:
    """
    Fetch date fields from CT.gov API v2
    
    Returns dict with date fields or None if failed
    """
    url = f"https://clinicaltrials.gov/api/v2/studies/{nct_id}"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Navigate to the nested structure
        protocol = data.get("protocolSection", {})
        status_module = protocol.get("statusModule", {})
        results_section = data.get("resultsSection", {})
        
        # Extract date fields
        dates = {
            "last_update_posted": status_module.get("lastUpdatePostDateStruct", {}).get("date"),
            "primary_completion_date": status_module.get("primaryCompletionDateStruct", {}).get("date"),
            "primary_completion_type": status_module.get("primaryCompletionDateStruct", {}).get("type"),
            "completion_date": status_module.get("completionDateStruct", {}).get("date"),
            "completion_type": status_module.get("completionDateStruct", {}).get("type"),
            "results_first_posted": results_section.get("resultsFirstPostDateStruct", {}).get("date"),
        }
        
        return dates
        
    except Exception as e:
        print(f"  ⚠️  Failed to fetch {nct_id}: {e}")
        return None


def backfill_trial_records(
    input_path: Path = Path("production_data/trial_records.json"),
    output_path: Path = Path("production_data/trial_records_with_dates.json"),
    rate_limit_seconds: float = 1.0
):
    """
    Backfill date fields for all trials in trial_records.json
    
    Args:
        input_path: Path to existing trial_records.json
        output_path: Path to write enhanced records
        rate_limit_seconds: Delay between API calls (be nice to CT.gov!)
    """
    
    print("="*80)
    print("BACKFILLING CT.GOV DATE FIELDS")
    print("="*80)
    print()
    
    # Load existing records
    with open(input_path) as f:
        records = json.load(f)
    
    print(f"Loaded {len(records)} trial records")
    print()
    
    # Track stats
    stats = {
        'total': len(records),
        'success': 0,
        'failed': 0,
        'already_has_dates': 0,
        'missing_last_update_posted': 0
    }
    
    enhanced_records = []
    
    for i, record in enumerate(records, 1):
        nct_id = record.get('nct_id', 'UNKNOWN')
        ticker = record.get('ticker', 'UNKNOWN')
        
        print(f"[{i}/{len(records)}] {ticker} - {nct_id}...", end=' ')
        
        # Check if already has last_update_posted
        if record.get('last_update_posted'):
            print("✓ Already has dates")
            stats['already_has_dates'] += 1
            enhanced_records.append(record)
            continue
        
        # Fetch dates from CT.gov
        dates = fetch_ctgov_dates(nct_id)
        
        if dates:
            # Update record with new date fields
            record.update(dates)
            
            # Check critical field
            if dates.get('last_update_posted'):
                print(f"✅ Added dates (last_update: {dates['last_update_posted']})")
                stats['success'] += 1
            else:
                print("⚠️  Added dates but missing last_update_posted")
                stats['missing_last_update_posted'] += 1
            
            enhanced_records.append(record)
        else:
            print("❌ Failed to fetch")
            stats['failed'] += 1
            enhanced_records.append(record)  # Keep original
        
        # Rate limiting (be nice to CT.gov)
        if i < len(records):
            time.sleep(rate_limit_seconds)
    
    # Save enhanced records
    with open(output_path, 'w') as f:
        json.dump(enhanced_records, f, indent=2)
    
    print()
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print()
    print(f"Total records: {stats['total']}")
    print(f"Already had dates: {stats['already_has_dates']}")
    print(f"Successfully enhanced: {stats['success']}")
    print(f"Failed to fetch: {stats['failed']}")
    print(f"Missing last_update_posted: {stats['missing_last_update_posted']}")
    print()
    print(f"Output written to: {output_path}")
    print()
    
    # Validation
    coverage = (stats['success'] + stats['already_has_dates']) / stats['total']
    print(f"Date coverage: {coverage:.1%}")
    
    if coverage >= 0.95:
        print("✅ GOOD: ≥95% coverage achieved")
    else:
        print("⚠️  WARNING: <95% coverage - some trials missing dates")
    
    print()
    print("Next step:")
    print("  cp production_data/trial_records_with_dates.json production_data/trial_records.json")
    print()


if __name__ == "__main__":
    backfill_trial_records()
