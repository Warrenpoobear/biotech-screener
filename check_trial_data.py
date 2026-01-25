#!/usr/bin/env python3
"""
check_trial_data.py - Diagnostic for clinical scoring issue

Purpose: Figure out why clinical scores still have 42.53 clustering
"""

import json
from collections import Counter

print("="*80)
print("CLINICAL SCORING DIAGNOSTIC")
print("="*80)

# Load trial records
print("\n1. Loading trial records...")
with open('production_data/trial_records.json', encoding='utf-8') as f:
    trials = json.load(f)
print(f"   Loaded {len(trials)} trial records")

# Check field availability
print("\n2. Checking available fields:")
all_fields = set()
for trial in trials:
    all_fields.update(trial.keys())

field_coverage = {}
for field in sorted(all_fields):
    count = sum(1 for t in trials if t.get(field) is not None)
    pct = count / len(trials) * 100
    field_coverage[field] = (count, pct)
    print(f"   {field:25s}: {count:5d}/{len(trials)} ({pct:5.1f}%)")

# Check enrollment specifically
print("\n3. Enrollment field analysis:")
enrollments = [t.get('enrollment') for t in trials if t.get('enrollment') is not None and t.get('enrollment') > 0]
if enrollments:
    print(f"   Non-null & >0: {len(enrollments)}/{len(trials)} ({len(enrollments)/len(trials)*100:.1f}%)")
    print(f"   Range: {min(enrollments)} - {max(enrollments)}")
    print(f"   Average: {sum(enrollments)/len(enrollments):.0f}")
else:
    print(f"   NO ENROLLMENT DATA (all None or 0)")

# Check status field
print("\n4. Status field analysis:")
statuses = [t.get('status') for t in trials if t.get('status')]
if statuses:
    status_counts = Counter(statuses)
    print(f"   Unique statuses: {len(status_counts)}")
    for status, count in status_counts.most_common(10):
        print(f"     {status:30s}: {count:5d} ({count/len(trials)*100:.1f}%)")
else:
    print(f"   NO STATUS DATA")

# Check phase field
print("\n5. Phase field analysis:")
phases = [t.get('phase') for t in trials if t.get('phase')]
if phases:
    phase_counts = Counter(phases)
    print(f"   Unique phases: {len(phase_counts)}")
    for phase, count in phase_counts.most_common():
        print(f"     {phase:30s}: {count:5d} ({count/len(trials)*100:.1f}%)")
else:
    print(f"   NO PHASE DATA")

# Load clinical results to see actual scores
print("\n6. Checking actual clinical scores from results:")
try:
    with open('results_continuous.json', encoding='utf-8') as f:
        results = json.load(f)
    clinical_scores = [float(s.get('clinical_dev_normalized', 0))
                      for s in results['module_5_composite']['ranked_securities']]
    score_counts = Counter(clinical_scores)

    print(f"   Total unique scores: {len(score_counts)}/322")
    print(f"   Top 5 most common scores:")
    for score, count in score_counts.most_common(5):
        print(f"     {score:.2f}: {count} tickers ({count/322*100:.1f}%)")

    # THE SMOKING GUN: Check if bonuses were actually applied
    print("\n7. Checking if bonuses were applied:")
    module_4_results = results.get('module_4_clinical_dev', {})
    if 'scores' in module_4_results:
        sample_scores = module_4_results['scores'][:10]
        has_bonus_fields = any('phase_progress_bonus' in s for s in sample_scores)

        if has_bonus_fields:
            print("   Bonus fields present in output!")
            # Check if bonuses vary
            progress_bonuses = [float(s.get('phase_progress_bonus', 0)) for s in module_4_results['scores']]
            enrollment_bonuses = [float(s.get('enrollment_bonus', 0)) for s in module_4_results['scores']]

            progress_unique = len(set(progress_bonuses))
            enrollment_unique = len(set(enrollment_bonuses))

            print(f"   Phase progress bonuses: {progress_unique} unique values")
            print(f"     Range: {min(progress_bonuses):.2f} - {max(progress_bonuses):.2f}")
            print(f"     Most common: {Counter(progress_bonuses).most_common(3)}")

            print(f"   Enrollment bonuses: {enrollment_unique} unique values")
            print(f"     Range: {min(enrollment_bonuses):.2f} - {max(enrollment_bonuses):.2f}")
            print(f"     Most common: {Counter(enrollment_bonuses).most_common(3)}")

            if progress_unique == 1 and enrollment_unique == 1:
                print("\n   WARNING: All bonuses are identical!")
                print("   This means the bonuses aren't differentiating scores.")
        else:
            print("   NO BONUS FIELDS - Old module still being used!")
            print("   The fixed module wasn't deployed correctly.")
    else:
        print("   Can't find module_4 results in output")
except FileNotFoundError:
    print("   results_continuous.json not found - skipping score analysis")
except json.JSONDecodeError as e:
    print(f"   Failed to parse results_continuous.json: {e}")

print("\n" + "="*80)
print("DIAGNOSIS COMPLETE")
print("="*80)
