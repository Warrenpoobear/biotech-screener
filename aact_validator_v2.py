#!/usr/bin/env python3
"""
AACT Extract Validator v2 — With Normalization
Applies enum normalization to get accurate biotech filter counts.

Usage:
    python aact_validator_v2.py <path_to_aact_folder>
"""

import sys
import os
from pathlib import Path
from collections import Counter
from datetime import datetime
import csv

csv.field_size_limit(10 * 1024 * 1024)

# =============================================================================
# NORMALIZATION MAPS (from aact_normalize.py)
# =============================================================================

PHASE_MAP = {
    'PHASE1': 'Phase 1', 'PHASE2': 'Phase 2', 'PHASE3': 'Phase 3', 'PHASE4': 'Phase 4',
    'PHASE1/PHASE2': 'Phase 1/Phase 2', 'PHASE2/PHASE3': 'Phase 2/Phase 3',
    'EARLY_PHASE1': 'Early Phase 1', 'NA': 'Not Applicable',
    'Phase 1': 'Phase 1', 'Phase 2': 'Phase 2', 'Phase 3': 'Phase 3', 'Phase 4': 'Phase 4',
    'Phase 1/Phase 2': 'Phase 1/Phase 2', 'Phase 2/Phase 3': 'Phase 2/Phase 3',
    'Early Phase 1': 'Early Phase 1', 'Not Applicable': 'Not Applicable', 'N/A': 'Not Applicable',
}

STATUS_MAP = {
    'RECRUITING': 'Recruiting', 'ACTIVE_NOT_RECRUITING': 'Active, not recruiting',
    'COMPLETED': 'Completed', 'TERMINATED': 'Terminated', 'SUSPENDED': 'Suspended',
    'WITHDRAWN': 'Withdrawn', 'ENROLLING_BY_INVITATION': 'Enrolling by invitation',
    'NOT_YET_RECRUITING': 'Not yet recruiting', 'UNKNOWN_STATUS': 'Unknown status',
    'NO_LONGER_AVAILABLE': 'No longer available', 'APPROVED_FOR_MARKETING': 'Approved for marketing',
    'AVAILABLE': 'Available', 'TEMPORARILY_NOT_AVAILABLE': 'Temporarily not available',
    'Recruiting': 'Recruiting', 'Active, not recruiting': 'Active, not recruiting',
    'Completed': 'Completed', 'Terminated': 'Terminated', 'Suspended': 'Suspended',
    'Withdrawn': 'Withdrawn', 'Enrolling by invitation': 'Enrolling by invitation',
    'Not yet recruiting': 'Not yet recruiting', 'Unknown status': 'Unknown status',
}

STUDY_TYPE_MAP = {
    'INTERVENTIONAL': 'Interventional', 'OBSERVATIONAL': 'Observational',
    'EXPANDED_ACCESS': 'Expanded Access',
    'Interventional': 'Interventional', 'Observational': 'Observational',
    'Expanded Access': 'Expanded Access',
}

INTERVENTION_TYPE_MAP = {
    'DRUG': 'Drug', 'BIOLOGICAL': 'Biological', 'DEVICE': 'Device',
    'PROCEDURE': 'Procedure', 'RADIATION': 'Radiation', 'BEHAVIORAL': 'Behavioral',
    'GENETIC': 'Genetic', 'DIETARY_SUPPLEMENT': 'Dietary Supplement',
    'COMBINATION_PRODUCT': 'Combination Product', 'DIAGNOSTIC_TEST': 'Diagnostic Test',
    'OTHER': 'Other',
    'Drug': 'Drug', 'Biological': 'Biological', 'Device': 'Device',
}

CLINICAL_PHASES = {'Phase 1', 'Phase 2', 'Phase 3', 'Phase 1/Phase 2', 'Phase 2/Phase 3'}
ACTIVE_STATUSES = {'Recruiting', 'Active, not recruiting', 'Enrolling by invitation', 'Not yet recruiting'}
BIOTECH_INTERVENTIONS = {'Drug', 'Biological', 'Combination Product'}


def normalize(raw, mapping):
    if raw is None or raw.strip() == '':
        return None
    return mapping.get(raw.strip())


def read_table(folder, table_name, max_rows=100000):
    filepath = folder / f"{table_name}.txt"
    if not filepath.exists():
        return [], []
    rows = []
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f, delimiter='|')
        columns = reader.fieldnames or []
        for row in reader:
            rows.append(row)
            if len(rows) >= max_rows:
                break
    return columns, rows


def main(aact_folder):
    folder = Path(aact_folder)
    if not folder.exists():
        print(f"ERROR: Folder not found: {folder}")
        sys.exit(1)
    
    print("=" * 70)
    print("AACT VALIDATOR v2 — With Normalization")
    print("=" * 70)
    print(f"Source: {folder}")
    print(f"Time: {datetime.now().isoformat()}")
    print()
    
    # Load studies
    print("Loading studies...")
    _, studies = read_table(folder, 'studies')
    print(f"  Loaded: {len(studies):,} rows")
    
    # Load interventions (to filter by drug/biological)
    print("Loading interventions...")
    _, interventions = read_table(folder, 'interventions')
    print(f"  Loaded: {len(interventions):,} rows")
    
    # Build set of NCT IDs with biotech-relevant interventions
    print()
    print("## INTERVENTION TYPE DISTRIBUTION (normalized)")
    int_type_dist = Counter()
    biotech_ncts = set()
    
    for row in interventions:
        raw_type = row.get('intervention_type', '')
        norm_type = normalize(raw_type, INTERVENTION_TYPE_MAP)
        int_type_dist[norm_type or 'NULL'] += 1
        
        if norm_type in BIOTECH_INTERVENTIONS:
            biotech_ncts.add(row.get('nct_id'))
    
    for itype, count in int_type_dist.most_common():
        marker = "★" if itype in BIOTECH_INTERVENTIONS else " "
        print(f"  {marker} {itype}: {count:,}")
    
    print(f"\n  NCT IDs with Drug/Biological interventions: {len(biotech_ncts):,}")
    
    # Normalize and filter studies
    print()
    print("## STUDY FILTER ANALYSIS (normalized)")
    
    phase_dist = Counter()
    status_dist = Counter()
    study_type_dist = Counter()
    
    # Filters
    interventional_count = 0
    clinical_phase_count = 0
    active_count = 0
    full_filter_count = 0
    
    passing_studies = []
    
    for row in studies:
        nct_id = row.get('nct_id', '')
        phase = normalize(row.get('phase'), PHASE_MAP)
        status = normalize(row.get('overall_status'), STATUS_MAP)
        study_type = normalize(row.get('study_type'), STUDY_TYPE_MAP)
        
        phase_dist[phase or 'NULL'] += 1
        status_dist[status or 'NULL'] += 1
        study_type_dist[study_type or 'NULL'] += 1
        
        # Apply filters
        is_interventional = (study_type == 'Interventional')
        is_clinical = (phase in CLINICAL_PHASES)
        is_active = (status in ACTIVE_STATUSES)
        has_biotech_intervention = (nct_id in biotech_ncts)
        
        if is_interventional:
            interventional_count += 1
        if is_clinical:
            clinical_phase_count += 1
        if is_active:
            active_count += 1
        
        # Full biotech filter: Interventional + Clinical Phase + Active + Biotech Intervention
        if is_interventional and is_clinical and has_biotech_intervention:
            full_filter_count += 1
            passing_studies.append({
                'nct_id': nct_id,
                'phase': phase,
                'status': status,
                'title': row.get('brief_title', '')[:60]
            })
    
    print()
    print("### Phase Distribution (normalized)")
    for phase, count in phase_dist.most_common():
        marker = "★" if phase in CLINICAL_PHASES else " "
        print(f"  {marker} {phase}: {count:,}")
    
    print()
    print("### Status Distribution (normalized)")
    for status, count in status_dist.most_common(10):
        marker = "★" if status in ACTIVE_STATUSES else " "
        print(f"  {marker} {status}: {count:,}")
    
    print()
    print("### Study Type Distribution (normalized)")
    for stype, count in study_type_dist.most_common():
        print(f"    {stype}: {count:,}")
    
    print()
    print("## FILTER FUNNEL")
    print(f"  Total studies loaded:              {len(studies):>10,}")
    print(f"  → Interventional:                  {interventional_count:>10,}")
    print(f"  → Clinical Phase (1-3):            {clinical_phase_count:>10,}")
    print(f"  → With Drug/Bio intervention:      {len(biotech_ncts):>10,}")
    print(f"  ─────────────────────────────────────────────")
    print(f"  BIOTECH FILTER PASS:               {full_filter_count:>10,}")
    
    if passing_studies:
        print()
        print("## SAMPLE PASSING STUDIES (first 10)")
        for s in passing_studies[:10]:
            print(f"  {s['nct_id']} | {s['phase']:18} | {s['status']:25} | {s['title']}")
    
    print()
    print("=" * 70)
    print("END OF VALIDATION REPORT")
    print("=" * 70)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python aact_validator_v2.py <path_to_aact_folder>")
        sys.exit(1)
    main(sys.argv[1])
