#!/usr/bin/env python3
"""Feature drift sanity check across PIT-lite archives.

Checks whether key clinical/catalyst columns vary plausibly across years
or look suspiciously "modern" (smoking gun for PIT-lite contamination).

For each archive date, computes cross-sectional statistics for:
  - clinical_optionality_pct_dev
  - catalyst fields (de_catalyst_days, catalyst_mode)
  - lead phase, program count, trial-derived fields
  - stage_bucket distribution

If these barely change across 2020→2026 or match current values too closely,
the PIT-lite panel is measuring future-conditioned features.
"""
from __future__ import annotations

import csv
import json
import statistics
import sys
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.research.anchor_replay import discover_archives, load_archive_rankings

ARCHIVE_DIR = PROJECT_ROOT / "data" / "archives"

# Columns to track
NUMERIC_COLS = [
    "clinical_optionality_pct_dev",
    "composite_score",
    "composite_rank",
    "alpha_cohort_pct",
    "alpha_cohort_raw",
    "clinical_score_v2",
    "clinical_score_v2_z",
    "clinical_alpha_z",
    "clinical_readout_days",
    "program_count",
    "readout_density_90",
    "lead_program_phase",
    "competitive_intensity_z",
]

CATEGORICAL_COLS = [
    "catalyst_mode",
    "stage_bucket",
    "archetype",
    "eligible",
]


def safe_float(v: Any) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def analyze_date(rows: List[Dict[str, str]]) -> Dict[str, Any]:
    """Compute cross-sectional stats for one archive date."""
    result: Dict[str, Any] = {"n_rows": len(rows)}

    for col in NUMERIC_COLS:
        vals = [safe_float(r.get(col)) for r in rows]
        vals = [v for v in vals if v is not None]
        if vals:
            result[f"{col}_mean"] = round(statistics.mean(vals), 4)
            result[f"{col}_std"] = round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0
            result[f"{col}_p25"] = round(sorted(vals)[len(vals) // 4], 4)
            result[f"{col}_p75"] = round(sorted(vals)[3 * len(vals) // 4], 4)
            result[f"{col}_n"] = len(vals)
        else:
            result[f"{col}_mean"] = None
            result[f"{col}_n"] = 0

    for col in CATEGORICAL_COLS:
        vals = [r.get(col, "") for r in rows]
        counts = Counter(v for v in vals if v)
        result[f"{col}_dist"] = dict(counts.most_common(10))

    return result


def main() -> None:
    archives = discover_archives(ARCHIVE_DIR, "2020-01-01", "2026-02-28", "monthly")
    if not archives:
        print("No archives found.")
        return

    print(f"Checking feature drift across {len(archives)} monthly archives ...\n")

    all_stats: Dict[str, Dict[str, Any]] = {}
    for date_str, tar_path in archives:
        rows = load_archive_rankings(tar_path, date_str)
        if not rows:
            continue
        stats = analyze_date(rows)
        all_stats[date_str] = stats

    # Report: year-by-year means for key columns
    years = sorted(set(d[:4] for d in all_stats))

    key_cols = [
        "clinical_optionality_pct_dev",
        "clinical_score_v2",
        "program_count",
        "readout_density_90",
        "lead_program_phase",
        "clinical_readout_days",
        "competitive_intensity_z",
    ]

    print("=" * 100)
    print("YEAR-BY-YEAR FEATURE MEANS (cross-sectional mean of each column)")
    print("=" * 100)

    header = f"{'Column':<35}"
    for y in years:
        header += f" {y:>8}"
    print(header)
    print("-" * len(header))

    for col in key_cols:
        line = f"{col:<35}"
        for y in years:
            year_dates = [d for d in all_stats if d[:4] == y]
            means = [all_stats[d].get(f"{col}_mean") for d in year_dates]
            means = [m for m in means if m is not None]
            if means:
                line += f" {statistics.mean(means):>8.3f}"
            else:
                line += f" {'—':>8}"
        print(line)

    # Report: catalyst_mode distribution by year
    print()
    print("=" * 100)
    print("CATALYST_MODE DISTRIBUTION BY YEAR")
    print("=" * 100)
    for y in years:
        year_dates = [d for d in all_stats if d[:4] == y]
        merged: Counter = Counter()
        for d in year_dates:
            dist = all_stats[d].get("catalyst_mode_dist", {})
            for k, v in dist.items():
                merged[k] += v
        total = sum(merged.values())
        pcts = {k: f"{v/total:.0%}" for k, v in merged.most_common()} if total else {}
        print(f"  {y}: {dict(merged.most_common())}  ({pcts})")

    # Report: stage_bucket distribution by year
    print()
    print("=" * 100)
    print("STAGE_BUCKET DISTRIBUTION BY YEAR")
    print("=" * 100)
    for y in years:
        year_dates = [d for d in all_stats if d[:4] == y]
        merged: Counter = Counter()
        for d in year_dates:
            dist = all_stats[d].get("stage_bucket_dist", {})
            for k, v in dist.items():
                merged[k] += v
        total = sum(merged.values())
        pcts = {k: f"{v/total:.0%}" for k, v in merged.most_common()} if total else {}
        print(f"  {y}: {pcts}")

    # Report: eligible count by year
    print()
    print("=" * 100)
    print("ELIGIBLE COUNT BY YEAR (mean per date)")
    print("=" * 100)
    for y in years:
        year_dates = [d for d in all_stats if d[:4] == y]
        elig_counts = []
        for d in year_dates:
            dist = all_stats[d].get("eligible_dist", {})
            elig_counts.append(dist.get("1", 0))
        mean_elig = statistics.mean(elig_counts) if elig_counts else 0
        print(f"  {y}: {mean_elig:.0f} eligible / "
              f"{statistics.mean(all_stats[d]['n_rows'] for d in year_dates):.0f} total")

    # Drift metric: correlation of clinical_optionality_pct_dev_mean with date index
    opt_means = []
    for d in sorted(all_stats.keys()):
        m = all_stats[d].get("clinical_optionality_pct_dev_mean")
        if m is not None:
            opt_means.append((d, m))

    if len(opt_means) >= 5:
        indices = list(range(len(opt_means)))
        vals = [m for _, m in opt_means]
        # Quick rank correlation
        from scripts.research.anchor_replay import _avg_ranks, spearman_ic
        r_idx = list(range(len(vals)))
        corr = spearman_ic(r_idx, vals)
        print(f"\n  Time trend in optionality mean: rank-corr = {corr:.4f}")
        print(f"    First date mean: {opt_means[0][1]:.4f} ({opt_means[0][0]})")
        print(f"    Last date mean:  {opt_means[-1][1]:.4f} ({opt_means[-1][0]})")

    # Write JSON
    out_dir = PROJECT_ROOT / "output" / "feature_drift"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "feature_drift.json", "w") as f:
        json.dump(all_stats, f, indent=2, default=str)
    print(f"\nOutput: {out_dir / 'feature_drift.json'}")


if __name__ == "__main__":
    main()
