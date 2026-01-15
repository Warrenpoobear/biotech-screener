#!/usr/bin/env python3
"""
Quick script to re-rank existing screening results with the FIXED sort order.
(Lower composite_score = better = rank 1)
"""

import json
import csv
from pathlib import Path
from decimal import Decimal

def rerank_results(input_file: str, output_csv: str):
    """Load results, re-sort ascending (lower score = rank 1), save CSV."""

    print(f"Loading: {input_file}")
    with open(input_file, 'r') as f:
        data = json.load(f)

    ranked = data['module_5_composite']['ranked_securities']
    print(f"  Found {len(ranked)} securities")

    # RE-SORT: ASCENDING by composite_score (lower = better = rank 1)
    # This is the FIX - previously was descending (higher = rank 1)
    ranked.sort(key=lambda x: (Decimal(x["composite_score"]), x["ticker"]))

    # Re-assign ranks
    for i, rec in enumerate(ranked):
        rec["composite_rank"] = i + 1

    # Write CSV
    print(f"Writing: {output_csv}")
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Rank', 'Ticker', 'Composite_Score', 'Clinical_Score',
                         'Financial_Score', 'Catalyst_Score', 'Stage_Bucket'])

        for rec in ranked:
            writer.writerow([
                rec['composite_rank'],
                rec['ticker'],
                rec.get('composite_score', ''),
                rec.get('clinical_dev_normalized', ''),
                rec.get('financial_normalized', ''),
                rec.get('catalyst_normalized', ''),
                rec.get('stage_bucket', '')
            ])

    print(f"\nDone! Top 10 (now ranked correctly - lower score = better):")
    for rec in ranked[:10]:
        print(f"  {rec['composite_rank']:3d}. {rec['ticker']:6s}  Score: {rec['composite_score']}")

    print(f"\nBottom 5:")
    for rec in ranked[-5:]:
        print(f"  {rec['composite_rank']:3d}. {rec['ticker']:6s}  Score: {rec['composite_score']}")

if __name__ == "__main__":
    # Use latest results file
    input_file = "results_FINAL_COMPLETE.json"
    output_csv = "outputs/rankings_FIXED.csv"

    # Check if file exists
    if not Path(input_file).exists():
        # Try other results files
        for f in ["results_2026-01-12.json", "results_FULL_DATA.json", "results_custom.json"]:
            if Path(f).exists():
                input_file = f
                break

    rerank_results(input_file, output_csv)
