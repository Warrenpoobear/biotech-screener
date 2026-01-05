"""
Cohort Census Diagnostic

Analyze cohort distribution to identify normalization fallback causes.
Outputs:
- Counts per {stage}_{mcap} cohort
- Counts per stage alone
- % unknown stage
- % using fallback normalization

Run this BEFORE fixing cohort geometry to understand the problem.
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional

import sys
sys.path.insert(0, "/home/claude/biotech_screener")

from module_1_universe import compute_module_1_universe
from module_2_financial import compute_module_2_financial
from module_3_catalyst import compute_module_3_catalyst
from module_4_clinical_dev import compute_module_4_clinical_dev
from module_5_composite import compute_module_5_composite, MIN_COHORT_SIZE

# Import sample data generator
from scripts.run_first_real_backtest import (
    BIOTECH_UNIVERSE,
    COMPANY_DATA,
    generate_monthly_dates,
    generate_sample_data,
)


def analyze_cohort_distribution(snapshots: List[Dict]) -> Dict[str, Any]:
    """
    Analyze cohort distribution across all snapshots.
    
    Returns detailed census for diagnosis.
    """
    # Aggregate stats
    all_cohort_counts = defaultdict(list)  # cohort_key -> [counts per date]
    all_stage_counts = defaultdict(list)   # stage -> [counts per date]
    all_mcap_counts = defaultdict(list)    # mcap -> [counts per date]
    
    date_details = []
    
    for snap in snapshots:
        as_of = snap["as_of_date"]
        cohort_stats = snap.get("cohort_stats", {})
        ranked = snap.get("ranked_securities", [])
        
        # Count by cohort
        cohort_counts = {}
        stage_counts = defaultdict(int)
        mcap_counts = defaultdict(int)
        fallback_counts = {"normal": 0, "stage_only": 0, "none": 0}
        
        for cohort_key, stats in cohort_stats.items():
            count = stats.get("count", 0)
            fallback = stats.get("normalization_fallback", "unknown")
            
            cohort_counts[cohort_key] = {
                "count": count,
                "fallback": fallback,
            }
            
            # Parse stage and mcap from key
            parts = cohort_key.split("_")
            if len(parts) >= 2:
                stage = parts[0]
                mcap = "_".join(parts[1:])
            else:
                stage = cohort_key
                mcap = "unknown"
            
            stage_counts[stage] += count
            mcap_counts[mcap] += count
            
            if fallback in fallback_counts:
                fallback_counts[fallback] += 1
        
        # Calculate totals
        total_securities = sum(stage_counts.values())
        unknown_stage_count = stage_counts.get("unknown", 0) + stage_counts.get("early", 0)  # early often contains unknowns
        
        # Track smallest cohorts
        smallest_cohorts = sorted(
            [(k, v["count"]) for k, v in cohort_counts.items()],
            key=lambda x: x[1]
        )[:5]
        
        date_detail = {
            "as_of_date": as_of,
            "total_securities": total_securities,
            "cohort_counts": cohort_counts,
            "stage_counts": dict(stage_counts),
            "mcap_counts": dict(mcap_counts),
            "fallback_counts": fallback_counts,
            "smallest_cohorts": smallest_cohorts,
            "unknown_stage_pct": stage_counts.get("unknown", 0) / total_securities if total_securities > 0 else 0,
        }
        date_details.append(date_detail)
        
        # Aggregate
        for cohort, stats in cohort_counts.items():
            all_cohort_counts[cohort].append(stats["count"])
        for stage, count in stage_counts.items():
            all_stage_counts[stage].append(count)
        for mcap, count in mcap_counts.items():
            all_mcap_counts[mcap].append(count)
    
    # Compute summary
    summary = {
        "n_dates": len(snapshots),
        "cohorts": {},
        "stages": {},
        "mcaps": {},
    }
    
    for cohort, counts in all_cohort_counts.items():
        summary["cohorts"][cohort] = {
            "min": min(counts),
            "max": max(counts),
            "mean": sum(counts) / len(counts),
            "below_threshold": sum(1 for c in counts if c < MIN_COHORT_SIZE),
        }
    
    for stage, counts in all_stage_counts.items():
        summary["stages"][stage] = {
            "min": min(counts),
            "max": max(counts),
            "mean": sum(counts) / len(counts),
        }
    
    for mcap, counts in all_mcap_counts.items():
        summary["mcaps"][mcap] = {
            "min": min(counts),
            "max": max(counts),
            "mean": sum(counts) / len(counts),
        }
    
    return {
        "summary": summary,
        "date_details": date_details,
    }


def print_cohort_census(census: Dict[str, Any]) -> None:
    """Print formatted cohort census."""
    summary = census["summary"]
    details = census["date_details"]
    
    print("\n" + "=" * 70)
    print("COHORT CENSUS DIAGNOSTIC")
    print("=" * 70)
    
    # Stage distribution
    print("\n1. STAGE DISTRIBUTION (averaged across dates)")
    print("-" * 50)
    print(f"  {'Stage':<15} {'Min':>6} {'Max':>6} {'Mean':>8}")
    print("  " + "-" * 40)
    
    total_mean = 0
    for stage, stats in sorted(summary["stages"].items()):
        total_mean += stats["mean"]
        print(f"  {stage:<15} {stats['min']:>6} {stats['max']:>6} {stats['mean']:>8.1f}")
    
    # Unknown rate
    unknown_mean = summary["stages"].get("unknown", {}).get("mean", 0)
    unknown_pct = unknown_mean / total_mean * 100 if total_mean > 0 else 0
    print(f"\n  → Unknown stage rate: {unknown_pct:.1f}%")
    
    # Mcap distribution
    print("\n2. MCAP DISTRIBUTION (averaged across dates)")
    print("-" * 50)
    print(f"  {'Mcap':<15} {'Min':>6} {'Max':>6} {'Mean':>8}")
    print("  " + "-" * 40)
    
    for mcap, stats in sorted(summary["mcaps"].items()):
        print(f"  {mcap:<15} {stats['min']:>6} {stats['max']:>6} {stats['mean']:>8.1f}")
    
    # Cohort distribution with fallback highlighting
    print("\n3. COHORT DISTRIBUTION (stage x mcap)")
    print("-" * 50)
    print(f"  {'Cohort':<25} {'Min':>5} {'Max':>5} {'Mean':>6} {'<5 dates':>10}")
    print("  " + "-" * 55)
    
    # Sort by min count (smallest first)
    sorted_cohorts = sorted(
        summary["cohorts"].items(),
        key=lambda x: x[1]["min"]
    )
    
    problem_cohorts = []
    for cohort, stats in sorted_cohorts:
        below = stats["below_threshold"]
        flag = "⚠" if below > 0 else " "
        print(f"  {flag} {cohort:<23} {stats['min']:>5} {stats['max']:>5} {stats['mean']:>6.1f} {below:>10}")
        if below > 0:
            problem_cohorts.append((cohort, stats))
    
    # Fallback summary
    print("\n4. FALLBACK SUMMARY (per date)")
    print("-" * 50)
    
    total_cohort_instances = 0
    total_fallbacks = 0
    
    for detail in details:
        fb = detail["fallback_counts"]
        total_cohort_instances += sum(fb.values())
        total_fallbacks += fb.get("stage_only", 0) + fb.get("none", 0)
    
    fallback_rate = total_fallbacks / total_cohort_instances if total_cohort_instances > 0 else 0
    print(f"  Total cohort instances: {total_cohort_instances}")
    print(f"  Using fallback:         {total_fallbacks}")
    print(f"  Fallback rate:          {fallback_rate*100:.1f}%")
    
    # Top 5 smallest cohorts
    print("\n5. SMALLEST COHORTS (need attention)")
    print("-" * 50)
    
    if problem_cohorts:
        for cohort, stats in problem_cohorts[:5]:
            print(f"  • {cohort}: min={stats['min']}, below threshold {stats['below_threshold']} dates")
    else:
        print("  None below threshold!")
    
    # Recommendation
    print("\n6. DIAGNOSIS")
    print("-" * 50)
    
    if unknown_pct > 10:
        print("  ⚠ HIGH UNKNOWN STAGE RATE")
        print("    → Fix stage assignment (use Module 4 phase, then Module 3)")
    
    if fallback_rate > 0.20:
        print("  ⚠ HIGH FALLBACK RATE")
        if len(problem_cohorts) > 0:
            # Check if problem is mcap granularity
            problem_mcaps = set()
            for cohort, _ in problem_cohorts:
                parts = cohort.split("_")
                if len(parts) >= 2:
                    problem_mcaps.add("_".join(parts[1:]))
            
            if "unknown" in problem_mcaps or "micro" in problem_mcaps:
                print("    → Consider normalizing by STAGE ONLY (not stage x mcap)")
                print("    → Or collapse mcap: micro + small → small_or_below")
    
    if fallback_rate <= 0.20 and unknown_pct <= 10:
        print("  ✓ Cohort geometry looks good!")
    
    print("\n" + "=" * 70)


def run_cohort_census():
    """Run cohort census on sample data."""
    print("\nGenerating snapshots for cohort census...")
    
    dates = generate_monthly_dates(2023, 2024)
    snapshots = []
    
    for i, d in enumerate(dates):
        data = generate_sample_data(BIOTECH_UNIVERSE, d, seed=100 + i)
        
        m1 = compute_module_1_universe(data["universe"], d, universe_tickers=BIOTECH_UNIVERSE)
        active_tickers = [s["ticker"] for s in m1["active_securities"]]
        
        m2 = compute_module_2_financial(data["financial"], active_tickers, d)
        m3 = compute_module_3_catalyst(data["trials"], active_tickers, d)
        m4 = compute_module_4_clinical_dev(data["trials"], active_tickers, d)
        m5 = compute_module_5_composite(m1, m2, m3, m4, d)
        
        snapshots.append(m5)
    
    print(f"  Created {len(snapshots)} snapshots")
    
    # Analyze
    census = analyze_cohort_distribution(snapshots)
    
    # Print
    print_cohort_census(census)
    
    # Save
    output_path = Path("/home/claude/biotech_screener/output/cohort_census.json")
    with open(output_path, "w") as f:
        json.dump(census, f, indent=2, default=str)
    print(f"\nFull census saved to: {output_path}")
    
    return census


if __name__ == "__main__":
    run_cohort_census()
