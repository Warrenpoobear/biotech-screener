#!/usr/bin/env python3
"""
backfill_ctgov_dates.py - Add missing date fields to trial_records.json

Queries CT.gov API v2 to backfill:
- last_update_posted
- primary_completion_date/type
- completion_date/type
- results_first_posted

Usage:
    # Backfill missing dates
    python backfill_ctgov_dates.py

    # Verify date coverage without making changes
    python backfill_ctgov_dates.py --verify

    # Custom input/output paths
    python backfill_ctgov_dates.py --input data/trials.json --output data/trials_enhanced.json
"""

import argparse
import json
import requests
import time
from pathlib import Path
from typing import Optional, Dict, Any


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


def verify_trial_date_coverage(
    input_path: Path = Path("production_data/trial_records.json"),
) -> Dict[str, Any]:
    """
    Verify date coverage in trial_records.json without making changes.

    Returns:
        Dict with coverage statistics and warnings
    """
    print("="*80)
    print("VERIFYING CT.GOV DATE COVERAGE")
    print("="*80)
    print()

    # Load records
    try:
        with open(input_path) as f:
            records = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found: {input_path}")
        return {"error": f"File not found: {input_path}"}
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {input_path}: {e}")
        return {"error": f"Invalid JSON: {e}"}

    print(f"Loaded {len(records)} trial records from {input_path}")
    print()

    # Track date field coverage
    stats = {
        "total_trials": len(records),
        "has_last_update_posted": 0,
        "has_first_posted": 0,
        "has_primary_completion_date": 0,
        "has_completion_date": 0,
        "has_results_first_posted": 0,
        "has_any_pit_date": 0,  # Has at least one date usable for PIT filtering
        "missing_all_dates": 0,
    }

    # PIT date priority fields (matches module_4_clinical_dev.py)
    pit_priority_fields = [
        "first_posted",
        "last_update_posted",
        "results_first_posted",
        "source_date",
        "start_date",
    ]

    trials_missing_dates = []

    for record in records:
        nct_id = record.get("nct_id", "UNKNOWN")
        ticker = record.get("ticker", "UNKNOWN")

        # Check individual fields
        if record.get("last_update_posted"):
            stats["has_last_update_posted"] += 1
        if record.get("first_posted"):
            stats["has_first_posted"] += 1
        if record.get("primary_completion_date"):
            stats["has_primary_completion_date"] += 1
        if record.get("completion_date"):
            stats["has_completion_date"] += 1
        if record.get("results_first_posted"):
            stats["has_results_first_posted"] += 1

        # Check if any PIT-usable date exists
        has_pit_date = any(record.get(field) for field in pit_priority_fields)
        if has_pit_date:
            stats["has_any_pit_date"] += 1
        else:
            stats["missing_all_dates"] += 1
            trials_missing_dates.append({"nct_id": nct_id, "ticker": ticker})

    # Calculate coverage percentages
    total = stats["total_trials"]
    if total == 0:
        print("No trial records found.")
        return {"error": "No trial records", **stats}

    coverage = {
        "last_update_posted_pct": stats["has_last_update_posted"] / total * 100,
        "first_posted_pct": stats["has_first_posted"] / total * 100,
        "primary_completion_date_pct": stats["has_primary_completion_date"] / total * 100,
        "completion_date_pct": stats["has_completion_date"] / total * 100,
        "results_first_posted_pct": stats["has_results_first_posted"] / total * 100,
        "any_pit_date_pct": stats["has_any_pit_date"] / total * 100,
    }

    # Print results
    print("DATE FIELD COVERAGE")
    print("-"*40)
    print(f"last_update_posted:     {stats['has_last_update_posted']:4d}/{total} ({coverage['last_update_posted_pct']:5.1f}%)")
    print(f"first_posted:           {stats['has_first_posted']:4d}/{total} ({coverage['first_posted_pct']:5.1f}%)")
    print(f"primary_completion_date:{stats['has_primary_completion_date']:4d}/{total} ({coverage['primary_completion_date_pct']:5.1f}%)")
    print(f"completion_date:        {stats['has_completion_date']:4d}/{total} ({coverage['completion_date_pct']:5.1f}%)")
    print(f"results_first_posted:   {stats['has_results_first_posted']:4d}/{total} ({coverage['results_first_posted_pct']:5.1f}%)")
    print()
    print(f"Any PIT-usable date:    {stats['has_any_pit_date']:4d}/{total} ({coverage['any_pit_date_pct']:5.1f}%)")
    print(f"Missing ALL dates:      {stats['missing_all_dates']:4d}/{total}")
    print()

    # Check thresholds and provide recommendations
    warnings = []

    if coverage["any_pit_date_pct"] < 90:
        warnings.append(f"PIT date coverage ({coverage['any_pit_date_pct']:.1f}%) is below 90% threshold")

    if coverage["last_update_posted_pct"] < 80:
        warnings.append(f"last_update_posted coverage ({coverage['last_update_posted_pct']:.1f}%) is low - run backfill")

    # Print warnings
    if warnings:
        print("="*80)
        print("WARNINGS")
        print("="*80)
        for w in warnings:
            print(f"  WARNING: {w}")
        print()
        print("RECOMMENDATION: Run backfill to improve date coverage:")
        print(f"  python backfill_ctgov_dates.py --input {input_path}")
        print()
    else:
        print("="*80)
        print("COVERAGE OK")
        print("="*80)
        print("Date coverage meets threshold (>=90%). PIT filtering will work correctly.")
        print()

    # Show sample of trials missing dates
    if trials_missing_dates and len(trials_missing_dates) <= 20:
        print("Trials missing all PIT dates:")
        for t in trials_missing_dates[:20]:
            print(f"  {t['ticker']:8s} - {t['nct_id']}")
        if len(trials_missing_dates) > 20:
            print(f"  ... and {len(trials_missing_dates) - 20} more")
        print()

    return {
        "stats": stats,
        "coverage": coverage,
        "warnings": warnings,
        "trials_missing_dates": trials_missing_dates[:100],  # Limit for output
    }


def main():
    parser = argparse.ArgumentParser(
        description="Backfill or verify CT.gov date fields in trial_records.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Verify date coverage (no changes)
  python backfill_ctgov_dates.py --verify

  # Backfill missing dates
  python backfill_ctgov_dates.py

  # Custom paths
  python backfill_ctgov_dates.py --input data/trials.json --output data/trials_enhanced.json
        """
    )

    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify date coverage without backfilling"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("production_data/trial_records.json"),
        help="Input trial records file (default: production_data/trial_records.json)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("production_data/trial_records_with_dates.json"),
        help="Output file for enhanced records (default: production_data/trial_records_with_dates.json)"
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=1.0,
        help="Seconds between API calls (default: 1.0)"
    )

    args = parser.parse_args()

    if args.verify:
        # Verify mode - just check coverage
        result = verify_trial_date_coverage(args.input)
        if result.get("error"):
            return 1
        if result.get("warnings"):
            return 2  # Warnings but not errors
        return 0
    else:
        # Backfill mode
        backfill_trial_records(
            input_path=args.input,
            output_path=args.output,
            rate_limit_seconds=args.rate_limit
        )
        return 0


if __name__ == "__main__":
    exit(main())
