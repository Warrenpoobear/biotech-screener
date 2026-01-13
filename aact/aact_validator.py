#!/usr/bin/env python3
"""
AACT Extract Validator for Wake Robin Biotech Alpha System
Run locally, paste output to Claude for analysis.

Usage:
    python aact_validator.py <path_to_aact_folder>
    
Example:
    python aact_validator.py C:\data\aact\20260105
    
The folder should contain the .txt files from an AACT daily snapshot.
"""

import sys
import os
from pathlib import Path
from collections import Counter
from datetime import datetime
import csv

# Increase field size limit for large AACT fields
csv.field_size_limit(10 * 1024 * 1024)  # 10MB

# =============================================================================
# EXPECTED SCHEMA (columns we need for Modules 3 & 4)
# =============================================================================

REQUIRED_TABLES = {
    'studies': [
        'nct_id', 'overall_status', 'phase', 'study_type',
        'start_date', 'completion_date', 'primary_completion_date',
        'enrollment', 'brief_title', 'official_title'
    ],
    'conditions': ['nct_id', 'name', 'downcase_name'],
    'interventions': ['nct_id', 'intervention_type', 'name'],
    'sponsors': ['nct_id', 'agency_class', 'lead_or_collaborator', 'name'],
    'calculated_values': ['nct_id', 'number_of_facilities', 'has_us_facility'],
}

OPTIONAL_TABLES = {
    'facilities': ['nct_id', 'country', 'city', 'state'],
    'outcome_counts': ['nct_id', 'count'],
    'design_outcomes': ['nct_id', 'outcome_type', 'measure'],
}

# Expected enum values
EXPECTED_PHASES = {
    'Phase 1', 'Phase 2', 'Phase 3', 'Phase 4',
    'Phase 1/Phase 2', 'Phase 2/Phase 3',
    'Early Phase 1', 'Not Applicable', 'N/A', '', None
}

EXPECTED_STATUSES = {
    'Recruiting', 'Active, not recruiting', 'Completed',
    'Terminated', 'Suspended', 'Withdrawn', 'Enrolling by invitation',
    'Not yet recruiting', 'Unknown status', 'No longer available',
    'Approved for marketing', 'Available', 'Temporarily not available',
    '', None
}

EXPECTED_INTERVENTION_TYPES = {
    'Drug', 'Biological', 'Device', 'Procedure', 'Radiation',
    'Behavioral', 'Genetic', 'Dietary Supplement', 'Combination Product',
    'Diagnostic Test', 'Other', '', None
}


def read_table(folder: Path, table_name: str) -> tuple[list[str], list[dict]]:
    """Read a pipe-delimited AACT table. Returns (columns, rows)."""
    filepath = folder / f"{table_name}.txt"
    if not filepath.exists():
        return [], []
    
    rows = []
    columns = []
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f, delimiter='|')
        columns = reader.fieldnames or []
        for row in reader:
            rows.append(row)
            if len(rows) >= 100000:  # Cap for memory safety
                break
    return columns, rows


def check_nulls(rows: list[dict], columns: list[str]) -> dict[str, float]:
    """Return null rate for each column."""
    if not rows:
        return {}
    null_counts = Counter()
    for row in rows:
        for col in columns:
            val = row.get(col, '')
            if val is None or val.strip() == '':
                null_counts[col] += 1
    return {col: null_counts[col] / len(rows) for col in columns}


def check_enum_values(rows: list[dict], column: str, expected: set) -> list[str]:
    """Return unexpected values found in column."""
    found = set()
    for row in rows:
        val = row.get(column, '')
        if val and val not in expected:
            found.add(val)
    return sorted(found)


def check_date_formats(rows: list[dict], date_columns: list[str]) -> dict[str, dict]:
    """Check date parsing patterns."""
    results = {}
    for col in date_columns:
        patterns = Counter()
        parse_failures = []
        for row in rows[:10000]:  # Sample
            val = row.get(col, '')
            if not val or val.strip() == '':
                patterns['NULL/EMPTY'] += 1
                continue
            
            # Try common formats
            parsed = False
            for fmt in ['%Y-%m-%d', '%B %d, %Y', '%B %Y', '%Y']:
                try:
                    datetime.strptime(val.strip(), fmt)
                    patterns[fmt] += 1
                    parsed = True
                    break
                except ValueError:
                    continue
            
            if not parsed:
                patterns['UNPARSEABLE'] += 1
                if len(parse_failures) < 5:
                    parse_failures.append(val[:50])
        
        results[col] = {
            'patterns': dict(patterns),
            'failures': parse_failures
        }
    return results


def filter_biotech_trials(studies: list[dict], conditions: list[dict]) -> set[str]:
    """Return NCT IDs that match biotech-relevant criteria."""
    # Get NCT IDs with drug/biological interventions would require interventions table
    # For now, just filter by phase (clinical trials)
    clinical_phases = {'Phase 1', 'Phase 2', 'Phase 3', 'Phase 1/Phase 2', 'Phase 2/Phase 3'}
    
    biotech_ncts = set()
    for row in studies:
        phase = row.get('phase', '')
        status = row.get('overall_status', '')
        study_type = row.get('study_type', '')
        
        # Interventional trials in clinical phases
        if phase in clinical_phases and study_type == 'Interventional':
            biotech_ncts.add(row.get('nct_id', ''))
    
    return biotech_ncts


def main(aact_folder: str):
    folder = Path(aact_folder)
    if not folder.exists():
        print(f"ERROR: Folder not found: {folder}")
        sys.exit(1)
    
    print("=" * 70)
    print("AACT EXTRACT VALIDATOR — Wake Robin Biotech Alpha System")
    print("=" * 70)
    print(f"Source folder: {folder}")
    print(f"Validation time: {datetime.now().isoformat()}")
    print()
    
    # Check which files exist
    print("## FILE INVENTORY")
    print()
    all_files = list(folder.glob("*.txt"))
    print(f"Total .txt files found: {len(all_files)}")
    
    for table in REQUIRED_TABLES:
        filepath = folder / f"{table}.txt"
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"  ✓ {table}.txt ({size_mb:.1f} MB)")
        else:
            print(f"  ✗ {table}.txt — MISSING (REQUIRED)")
    
    for table in OPTIONAL_TABLES:
        filepath = folder / f"{table}.txt"
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"  ○ {table}.txt ({size_mb:.1f} MB)")
    print()
    
    # Load and validate each required table
    tables_data = {}
    
    print("## SCHEMA VALIDATION")
    print()
    
    for table, required_cols in REQUIRED_TABLES.items():
        columns, rows = read_table(folder, table)
        tables_data[table] = {'columns': columns, 'rows': rows}
        
        if not columns:
            print(f"### {table}: SKIPPED (not found or empty)")
            continue
        
        print(f"### {table}")
        print(f"  Rows loaded: {len(rows):,}" + (" (capped at 100k)" if len(rows) >= 100000 else ""))
        print(f"  Columns found: {len(columns)}")
        
        # Check required columns
        missing = [c for c in required_cols if c not in columns]
        extra = [c for c in columns if c not in required_cols][:10]  # First 10 extra
        
        if missing:
            print(f"  ⚠ MISSING COLUMNS: {missing}")
        else:
            print(f"  ✓ All required columns present")
        
        if extra:
            print(f"  ℹ Extra columns (first 10): {extra}")
        print()
    
    # Null analysis for studies table
    print("## NULL RATE ANALYSIS (studies table)")
    print()
    
    if tables_data.get('studies', {}).get('rows'):
        studies = tables_data['studies']
        null_rates = check_nulls(studies['rows'], studies['columns'])
        
        # Sort by null rate descending, show columns with >5% nulls
        high_null = [(col, rate) for col, rate in null_rates.items() if rate > 0.05]
        high_null.sort(key=lambda x: -x[1])
        
        if high_null:
            print("Columns with >5% null rate:")
            for col, rate in high_null[:15]:
                print(f"  {col}: {rate*100:.1f}%")
        else:
            print("  All columns have <5% null rate ✓")
        
        # Critical columns
        critical = ['nct_id', 'overall_status', 'phase', 'study_type']
        print()
        print("Critical columns null check:")
        for col in critical:
            rate = null_rates.get(col, 0)
            status = "✓" if rate < 0.01 else "⚠"
            print(f"  {status} {col}: {rate*100:.2f}%")
    print()
    
    # Enum validation
    print("## ENUM VALUE VALIDATION")
    print()
    
    if tables_data.get('studies', {}).get('rows'):
        studies_rows = tables_data['studies']['rows']
        
        # Phase values
        unexpected_phases = check_enum_values(studies_rows, 'phase', EXPECTED_PHASES)
        if unexpected_phases:
            print(f"⚠ Unexpected PHASE values: {unexpected_phases[:10]}")
        else:
            print("✓ All phase values expected")
        
        # Status values
        unexpected_statuses = check_enum_values(studies_rows, 'overall_status', EXPECTED_STATUSES)
        if unexpected_statuses:
            print(f"⚠ Unexpected STATUS values: {unexpected_statuses[:10]}")
        else:
            print("✓ All status values expected")
    
    if tables_data.get('interventions', {}).get('rows'):
        int_rows = tables_data['interventions']['rows']
        unexpected_types = check_enum_values(int_rows, 'intervention_type', EXPECTED_INTERVENTION_TYPES)
        if unexpected_types:
            print(f"⚠ Unexpected INTERVENTION_TYPE values: {unexpected_types[:10]}")
        else:
            print("✓ All intervention_type values expected")
    print()
    
    # Date format analysis
    print("## DATE FORMAT ANALYSIS")
    print()
    
    if tables_data.get('studies', {}).get('rows'):
        date_results = check_date_formats(
            tables_data['studies']['rows'],
            ['start_date', 'completion_date', 'primary_completion_date']
        )
        
        for col, info in date_results.items():
            print(f"### {col}")
            for pattern, count in sorted(info['patterns'].items(), key=lambda x: -x[1]):
                print(f"  {pattern}: {count:,}")
            if info['failures']:
                print(f"  Sample failures: {info['failures']}")
            print()
    
    # Linkage check
    print("## CROSS-TABLE LINKAGE")
    print()
    
    if tables_data.get('studies', {}).get('rows'):
        studies_ncts = {r.get('nct_id') for r in tables_data['studies']['rows']}
        print(f"Unique NCT IDs in studies: {len(studies_ncts):,}")
        
        for table in ['conditions', 'interventions', 'sponsors', 'calculated_values']:
            if tables_data.get(table, {}).get('rows'):
                table_ncts = {r.get('nct_id') for r in tables_data[table]['rows']}
                orphans = table_ncts - studies_ncts
                missing = studies_ncts - table_ncts
                
                if len(orphans) > 0:
                    print(f"  ⚠ {table}: {len(orphans):,} NCT IDs not in studies")
                if len(missing) > len(studies_ncts) * 0.5:
                    print(f"  ⚠ {table}: {len(missing):,} studies have no {table} record")
                else:
                    print(f"  ✓ {table}: linkage OK")
    print()
    
    # Biotech filter test
    print("## BIOTECH FILTER EFFICACY")
    print()
    
    if tables_data.get('studies', {}).get('rows'):
        biotech_ncts = filter_biotech_trials(
            tables_data['studies']['rows'],
            tables_data.get('conditions', {}).get('rows', [])
        )
        total = len(tables_data['studies']['rows'])
        filtered = len(biotech_ncts)
        print(f"Total studies: {total:,}")
        print(f"After biotech filter (Phase 1-3 Interventional): {filtered:,}")
        print(f"Filter rate: {(1 - filtered/total)*100:.1f}% excluded")
        
        # Phase distribution of filtered
        phase_dist = Counter()
        for row in tables_data['studies']['rows']:
            if row.get('nct_id') in biotech_ncts:
                phase_dist[row.get('phase', 'UNKNOWN')] += 1
        
        print()
        print("Phase distribution (filtered set):")
        for phase, count in phase_dist.most_common():
            print(f"  {phase}: {count:,}")
    
    print()
    print("=" * 70)
    print("END OF VALIDATION REPORT")
    print("=" * 70)
    print()
    print("Copy everything above and paste to Claude for analysis.")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python aact_validator.py <path_to_aact_folder>")
        print("Example: python aact_validator.py C:\\data\\aact\\20260105")
        sys.exit(1)
    
    main(sys.argv[1])
