#!/usr/bin/env python3
"""
AACT Extract Validator

Quick validation of real AACT CSV extracts before running full snapshot generation.
Catches format/join/normalization issues in seconds.

Usage:
    python scripts/validate_aact_extract.py \
        --studies data/aact_snapshots/2024-01-29/studies.csv \
        --sponsors data/aact_snapshots/2024-01-29/sponsors.csv \
        --trial-map data/trial_mapping.csv

Or validate a whole snapshot folder:
    python scripts/validate_aact_extract.py \
        --snapshot-dir data/aact_snapshots/2024-01-29 \
        --trial-map data/trial_mapping.csv
"""

import argparse
import csv
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.providers.protocols import Phase, TrialStatus, PCDType


# Schema definitions
REQUIRED_STUDIES_COLUMNS = {
    "nct_id", "phase", "overall_status", 
    "primary_completion_date", "primary_completion_date_type",
    "last_update_posted_date", "study_type"
}

REQUIRED_SPONSORS_COLUMNS = {"nct_id", "name", "lead_or_collaborator"}

REQUIRED_MAPPING_COLUMNS = {"ticker", "nct_id"}


class ValidationResult:
    """Collects validation results."""
    
    def __init__(self, name: str):
        self.name = name
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.info: list[str] = []
        self.stats: dict[str, int] = {}
    
    def error(self, msg: str):
        self.errors.append(msg)
    
    def warn(self, msg: str):
        self.warnings.append(msg)
    
    def add_info(self, msg: str):
        self.info.append(msg)
    
    def set_stat(self, key: str, value: int):
        self.stats[key] = value
    
    @property
    def passed(self) -> bool:
        return len(self.errors) == 0
    
    def print_report(self):
        status = "✅ PASSED" if self.passed else "❌ FAILED"
        print(f"\n{'='*60}")
        print(f"{self.name}: {status}")
        print(f"{'='*60}")
        
        if self.stats:
            print("\nStatistics:")
            for key, value in sorted(self.stats.items()):
                print(f"  {key}: {value}")
        
        if self.info:
            print("\nInfo:")
            for msg in self.info:
                print(f"  ℹ️  {msg}")
        
        if self.warnings:
            print("\nWarnings:")
            for msg in self.warnings:
                print(f"  ⚠️  {msg}")
        
        if self.errors:
            print("\nErrors:")
            for msg in self.errors:
                print(f"  ❌ {msg}")


def validate_studies_csv(path: Path) -> ValidationResult:
    """Validate studies.csv format and content."""
    result = ValidationResult(f"Studies CSV: {path.name}")
    
    if not path.exists():
        result.error(f"File not found: {path}")
        return result
    
    # Track stats
    row_count = 0
    duplicates: dict[str, int] = {}
    phase_counts: Counter = Counter()
    status_counts: Counter = Counter()
    pcd_type_counts: Counter = Counter()
    parse_issues: list[str] = []
    
    # Normalization failures
    phase_unmapped: Counter = Counter()
    status_unmapped: Counter = Counter()
    
    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            
            # Schema check
            if reader.fieldnames is None:
                result.error("Empty file or no header row")
                return result
            
            actual_cols = set(reader.fieldnames)
            missing_cols = REQUIRED_STUDIES_COLUMNS - actual_cols
            extra_cols = actual_cols - REQUIRED_STUDIES_COLUMNS
            
            if missing_cols:
                result.error(f"Missing required columns: {sorted(missing_cols)}")
            
            if extra_cols:
                result.add_info(f"Extra columns (ignored): {sorted(extra_cols)}")
            
            # Process rows
            seen_nct_ids: dict[str, int] = {}
            
            for row_num, row in enumerate(reader, start=2):  # Row 2 is first data row
                row_count += 1
                nct_id = row.get("nct_id", "").strip().upper()
                
                # Check for duplicates
                if nct_id in seen_nct_ids:
                    if nct_id not in duplicates:
                        duplicates[nct_id] = 1
                    duplicates[nct_id] += 1
                seen_nct_ids[nct_id] = row_num
                
                # Phase normalization
                phase_raw = row.get("phase", "")
                phase = Phase.from_aact(phase_raw)
                phase_counts[phase.value] += 1
                if phase == Phase.UNKNOWN and phase_raw.strip():
                    phase_unmapped[phase_raw] += 1
                
                # Status normalization
                status_raw = row.get("overall_status", "")
                status = TrialStatus.from_aact(status_raw)
                status_counts[status.value] += 1
                if status == TrialStatus.UNKNOWN and status_raw.strip():
                    status_unmapped[status_raw] += 1
                
                # PCD type normalization
                pcd_type_raw = row.get("primary_completion_date_type", "")
                pcd_type = PCDType.from_aact(pcd_type_raw)
                pcd_type_counts[pcd_type.value] += 1
                
                # Date parsing checks
                pcd = row.get("primary_completion_date", "").strip()
                if pcd:
                    try:
                        date.fromisoformat(pcd)
                    except ValueError:
                        parse_issues.append(f"Row {row_num}: PCD '{pcd}' not YYYY-MM-DD")
                
                last_update = row.get("last_update_posted_date", "").strip()
                if last_update:
                    try:
                        date.fromisoformat(last_update)
                    except ValueError:
                        parse_issues.append(f"Row {row_num}: last_update '{last_update}' not YYYY-MM-DD")
    
    except csv.Error as e:
        result.error(f"CSV parsing error: {e}")
        return result
    except UnicodeDecodeError as e:
        result.error(f"Encoding error (expected UTF-8): {e}")
        return result
    
    # Report stats
    result.set_stat("total_rows", row_count)
    result.set_stat("unique_nct_ids", len(seen_nct_ids))
    result.set_stat("duplicate_nct_ids", len(duplicates))
    
    # Phase distribution
    result.add_info(f"Phase distribution: {dict(phase_counts)}")
    
    # Status distribution
    result.add_info(f"Status distribution: {dict(status_counts)}")
    
    # Warnings for unmapped values
    if phase_unmapped:
        result.warn(f"Unmapped phase values (→UNKNOWN): {dict(phase_unmapped.most_common(10))}")
    
    if status_unmapped:
        result.warn(f"Unmapped status values (→UNKNOWN): {dict(status_unmapped.most_common(10))}")
    
    # Duplicates
    if duplicates:
        top_dups = sorted(duplicates.items(), key=lambda x: -x[1])[:5]
        result.warn(f"Duplicate NCT IDs found: {len(duplicates)} unique IDs appear multiple times")
        result.warn(f"  Top duplicates: {top_dups}")
    
    # Parse issues
    if parse_issues:
        result.warn(f"Date parse issues: {len(parse_issues)} rows")
        for issue in parse_issues[:5]:
            result.warn(f"  {issue}")
        if len(parse_issues) > 5:
            result.warn(f"  ...and {len(parse_issues) - 5} more")
    
    return result


def validate_sponsors_csv(path: Path) -> ValidationResult:
    """Validate sponsors.csv format and content."""
    result = ValidationResult(f"Sponsors CSV: {path.name}")
    
    if not path.exists():
        result.error(f"File not found: {path}")
        return result
    
    row_count = 0
    lead_count = 0
    collab_count = 0
    other_count = 0
    lead_or_collab_values: Counter = Counter()
    nct_ids_with_lead: set[str] = set()
    
    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            
            # Schema check
            if reader.fieldnames is None:
                result.error("Empty file or no header row")
                return result
            
            actual_cols = set(reader.fieldnames)
            missing_cols = REQUIRED_SPONSORS_COLUMNS - actual_cols
            
            if missing_cols:
                result.error(f"Missing required columns: {sorted(missing_cols)}")
            
            # Process rows
            for row in reader:
                row_count += 1
                nct_id = row.get("nct_id", "").strip().upper()
                lead_or_collab = row.get("lead_or_collaborator", "").strip().upper()
                
                lead_or_collab_values[lead_or_collab] += 1
                
                if lead_or_collab == "LEAD":
                    lead_count += 1
                    nct_ids_with_lead.add(nct_id)
                elif lead_or_collab == "COLLABORATOR":
                    collab_count += 1
                else:
                    other_count += 1
    
    except csv.Error as e:
        result.error(f"CSV parsing error: {e}")
        return result
    
    result.set_stat("total_rows", row_count)
    result.set_stat("lead_sponsors", lead_count)
    result.set_stat("collaborators", collab_count)
    result.set_stat("unique_nct_ids_with_lead", len(nct_ids_with_lead))
    
    result.add_info(f"lead_or_collaborator values: {dict(lead_or_collab_values)}")
    
    if other_count > 0:
        result.warn(f"Found {other_count} rows with unexpected lead_or_collaborator value")
    
    return result


def validate_trial_mapping(path: Path) -> ValidationResult:
    """Validate trial_mapping.csv format."""
    result = ValidationResult(f"Trial Mapping: {path.name}")
    
    if not path.exists():
        result.error(f"File not found: {path}")
        return result
    
    row_count = 0
    tickers: set[str] = set()
    nct_ids: set[str] = set()
    
    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            
            if reader.fieldnames is None:
                result.error("Empty file or no header row")
                return result
            
            actual_cols = set(reader.fieldnames)
            missing_cols = REQUIRED_MAPPING_COLUMNS - actual_cols
            
            if missing_cols:
                result.error(f"Missing required columns: {sorted(missing_cols)}")
            
            # Check for recommended columns
            recommended = {"effective_start", "source", "mapping_confidence"}
            missing_recommended = recommended - actual_cols
            if missing_recommended:
                result.add_info(f"Optional columns not present: {sorted(missing_recommended)}")
            
            for row in reader:
                row_count += 1
                ticker = row.get("ticker", "").strip().upper()
                nct_id = row.get("nct_id", "").strip().upper()
                
                if ticker:
                    tickers.add(ticker)
                if nct_id:
                    nct_ids.add(nct_id)
    
    except csv.Error as e:
        result.error(f"CSV parsing error: {e}")
        return result
    
    result.set_stat("total_mappings", row_count)
    result.set_stat("unique_tickers", len(tickers))
    result.set_stat("unique_nct_ids", len(nct_ids))
    
    if len(tickers) > 0:
        avg_trials = row_count / len(tickers)
        result.add_info(f"Average trials per ticker: {avg_trials:.1f}")
    
    return result


def validate_joins(
    studies_path: Path, 
    sponsors_path: Path, 
    mapping_path: Optional[Path]
) -> ValidationResult:
    """Validate that joins between files will work."""
    result = ValidationResult("Join Validation")
    
    # Load NCT IDs from each file
    studies_nct_ids: set[str] = set()
    sponsors_nct_ids: set[str] = set()
    mapping_nct_ids: set[str] = set()
    
    if studies_path.exists():
        with open(studies_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                nct_id = row.get("nct_id", "").strip().upper()
                if nct_id:
                    studies_nct_ids.add(nct_id)
    
    if sponsors_path.exists():
        with open(sponsors_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("lead_or_collaborator", "").strip().upper() == "LEAD":
                    nct_id = row.get("nct_id", "").strip().upper()
                    if nct_id:
                        sponsors_nct_ids.add(nct_id)
    
    if mapping_path and mapping_path.exists():
        with open(mapping_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                nct_id = row.get("nct_id", "").strip().upper()
                if nct_id:
                    mapping_nct_ids.add(nct_id)
    
    # Check overlaps
    result.set_stat("studies_nct_ids", len(studies_nct_ids))
    result.set_stat("sponsors_nct_ids_lead", len(sponsors_nct_ids))
    
    # Studies ↔ Sponsors join
    studies_without_sponsor = studies_nct_ids - sponsors_nct_ids
    if studies_without_sponsor:
        pct = len(studies_without_sponsor) / len(studies_nct_ids) * 100
        result.add_info(f"Studies without lead sponsor: {len(studies_without_sponsor)} ({pct:.1f}%)")
        if pct > 10:
            result.warn(f"High percentage of studies missing lead sponsor")
    
    # Mapping ↔ Studies join
    if mapping_nct_ids:
        result.set_stat("mapping_nct_ids", len(mapping_nct_ids))
        
        mapping_in_studies = mapping_nct_ids & studies_nct_ids
        mapping_not_in_studies = mapping_nct_ids - studies_nct_ids
        
        result.set_stat("mapping_found_in_studies", len(mapping_in_studies))
        result.set_stat("mapping_not_in_studies", len(mapping_not_in_studies))
        
        coverage = len(mapping_in_studies) / len(mapping_nct_ids) * 100 if mapping_nct_ids else 0
        result.add_info(f"Mapping → Studies coverage: {coverage:.1f}%")
        
        if mapping_not_in_studies:
            result.warn(f"NCT IDs in mapping but not in studies: {len(mapping_not_in_studies)}")
            sample = sorted(mapping_not_in_studies)[:5]
            result.warn(f"  Sample: {sample}")
            if coverage < 50:
                result.error(f"Coverage too low ({coverage:.1f}%) - check snapshot date or mapping")
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Validate AACT CSV extracts before snapshot generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        help="Path to snapshot folder containing studies.csv and sponsors.csv",
    )
    parser.add_argument(
        "--studies",
        type=Path,
        help="Path to studies.csv (alternative to --snapshot-dir)",
    )
    parser.add_argument(
        "--sponsors",
        type=Path,
        help="Path to sponsors.csv (alternative to --snapshot-dir)",
    )
    parser.add_argument(
        "--trial-map",
        type=Path,
        help="Path to trial_mapping.csv",
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    if args.snapshot_dir:
        studies_path = args.snapshot_dir / "studies.csv"
        sponsors_path = args.snapshot_dir / "sponsors.csv"
    elif args.studies and args.sponsors:
        studies_path = args.studies
        sponsors_path = args.sponsors
    else:
        parser.error("Provide either --snapshot-dir or both --studies and --sponsors")
    
    print("\n" + "="*60)
    print("AACT EXTRACT VALIDATION")
    print("="*60)
    
    # Run validations
    results: list[ValidationResult] = []
    
    results.append(validate_studies_csv(studies_path))
    results.append(validate_sponsors_csv(sponsors_path))
    
    if args.trial_map:
        results.append(validate_trial_mapping(args.trial_map))
    
    results.append(validate_joins(studies_path, sponsors_path, args.trial_map))
    
    # Print reports
    for r in results:
        r.print_report()
    
    # Overall summary
    all_passed = all(r.passed for r in results)
    total_warnings = sum(len(r.warnings) for r in results)
    
    print("\n" + "="*60)
    print("OVERALL RESULT")
    print("="*60)
    
    if all_passed:
        print("✅ All validations PASSED")
        if total_warnings > 0:
            print(f"⚠️  {total_warnings} warning(s) - review recommended")
        print("\n→ Safe to run snapshot generation")
    else:
        print("❌ Some validations FAILED")
        print("\n→ Fix errors before running snapshot generation")
    
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
