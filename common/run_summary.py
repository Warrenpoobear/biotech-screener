"""
Run Summary Artifact

Generates a single-glance quality gate summary for each backtest run.
Saved to: output/runs/{run_id}/run_summary.json

Includes:
- Validation results
- Coverage stats
- Fallback stats
- Delisting diagnostics
- Headline metrics (IC, spread, monotonicity)
- Hashes for reproducibility
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional

from common.run_manifest import DecimalEncoder, compute_results_hash


def compute_coverage_stats(
    snapshots: List[Dict],
    returns_by_date: Dict[str, Dict[str, Optional[str]]],
) -> Dict[str, Any]:
    """
    Compute coverage statistics across snapshots.
    
    Args:
        snapshots: Module 5 outputs
        returns_by_date: {as_of: {ticker: return_str or None}}
    
    Returns:
        Coverage stats dict
    """
    total_securities = 0
    rankable = 0
    with_returns = 0
    
    for snap in snapshots:
        as_of = snap["as_of_date"]
        ranked = snap.get("ranked_securities", [])
        excluded = snap.get("excluded_securities", [])
        
        total_securities += len(ranked) + len(excluded)
        rankable += len(ranked)
        
        # Count returns
        if as_of in returns_by_date:
            date_returns = returns_by_date[as_of]
            for sec in ranked:
                if date_returns.get(sec["ticker"]) is not None:
                    with_returns += 1
    
    return {
        "total_securities": total_securities,
        "rankable": rankable,
        "with_returns": with_returns,
        "rankable_rate": round(rankable / total_securities, 4) if total_securities > 0 else 0,
        "return_coverage_rate": round(with_returns / rankable, 4) if rankable > 0 else 0,
    }


def compute_fallback_stats(snapshots: List[Dict]) -> Dict[str, Any]:
    """
    Compute cohort fallback statistics.
    
    Computes fallback rate based on SECURITIES affected, not cohort instances.
    This gives a more meaningful measure of data quality.
    """
    MIN_STAGE_SIZE = 5
    
    total_securities = 0
    normal_securities = 0
    fallback_securities = 0
    
    cohort_sizes = {}
    stage_sizes = defaultdict(list)  # Track stage sizes per date
    fallback_reasons = defaultdict(int)
    
    for snap in snapshots:
        cohort_mode = snap.get("cohort_mode", "stage_mcap")
        date_stage_counts = defaultdict(int)
        
        for cohort, stats in snap.get("cohort_stats", {}).items():
            count = stats.get("count", 0)
            fallback = stats.get("normalization_fallback", "unknown")
            
            total_securities += count
            
            if fallback == "normal":
                normal_securities += count
            else:
                fallback_securities += count
                fallback_reasons[fallback] += count
            
            if cohort not in cohort_sizes:
                cohort_sizes[cohort] = []
            cohort_sizes[cohort].append(count)
            
            # Track stage sizes (cohort key is stage in stage_only mode)
            if cohort_mode == "stage_only":
                date_stage_counts[cohort] += count
            else:
                # Parse stage from cohort key
                parts = cohort.split("_")
                if parts:
                    date_stage_counts[parts[0]] += count
        
        # Record stage sizes for this date
        for stage, count in date_stage_counts.items():
            stage_sizes[stage].append(count)
    
    # Fallback rate: % of securities in fallback cohorts
    fallback_rate = fallback_securities / total_securities if total_securities > 0 else 0
    
    # Identify stages below minimum size
    stage_stats = {}
    stages_below_min = []
    
    for stage, counts in stage_sizes.items():
        min_count = min(counts) if counts else 0
        stage_stats[stage] = {
            "min": min_count,
            "max": max(counts) if counts else 0,
            "mean": round(mean(counts), 1) if counts else 0,
        }
        if min_count < MIN_STAGE_SIZE:
            stages_below_min.append(stage)
    
    return {
        "total_securities": total_securities,
        "normal_securities": normal_securities,
        "fallback_securities": fallback_securities,
        "fallback_reasons": dict(fallback_reasons),
        "fallback_rate": round(fallback_rate, 4),
        "cohort_size_ranges": {
            c: {"min": min(sizes), "max": max(sizes), "mean": round(sum(sizes)/len(sizes), 1)}
            for c, sizes in cohort_sizes.items()
        },
        # Stage-level diagnostics
        "stage_sizes": stage_stats,
        "min_stage_size_threshold": MIN_STAGE_SIZE,
        "stages_below_min": stages_below_min,
    }


def extract_headline_metrics(backtest_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract headline metrics from backtest result.
    """
    headlines = {}
    
    for horizon in ["63d", "126d", "252d"]:
        agg = backtest_result.get("aggregate_metrics", {}).get(horizon, {})
        if agg:
            headlines[horizon] = {
                "ic_mean": float(agg["ic_mean"]) if agg.get("ic_mean") else None,
                "ic_pos_frac": float(agg["ic_pos_frac"]) if agg.get("ic_pos_frac") else None,
                "bucket_spread_mean": float(agg["bucket_spread_mean"]) if agg.get("bucket_spread_mean") else None,
                "monotonicity_rate": float(agg["monotonicity_rate"]) if agg.get("monotonicity_rate") else None,
                "n_periods": agg.get("n_periods", 0),
            }
    
    return headlines


def generate_run_summary(
    run_id: str,
    config: Dict[str, Any],
    snapshots: List[Dict],
    backtest_result: Dict[str, Any],
    validation_results: Dict[str, bool],
    validation_details: Dict[str, Any],
    provider_diagnostics: Optional[Dict[str, Any]] = None,
    returns_by_date: Optional[Dict[str, Dict[str, Optional[str]]]] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Generate comprehensive run summary.
    
    Args:
        run_id: Unique run identifier
        config: Run configuration
        snapshots: Module 5 outputs
        backtest_result: Metrics suite output
        validation_results: Pass/fail for each validation
        validation_details: Detailed validation outputs
        provider_diagnostics: Returns provider diagnostics (delisting, etc.)
        returns_by_date: {as_of: {ticker: return}} for coverage calc
        output_dir: Where to save (default: output/runs/{run_id}/)
    
    Returns:
        Summary dict (also saved to disk)
    """
    if output_dir is None:
        output_dir = Path("output/runs") / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build summary
    summary = {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        
        # Validation results
        "validations": {
            "all_passed": all(validation_results.values()),
            "results": validation_results,
            "details": {
                k: {kk: vv for kk, vv in v.items() if kk != "passed"}
                for k, v in validation_details.items()
            },
        },
        
        # Coverage stats
        "coverage": compute_coverage_stats(snapshots, returns_by_date or {}),
        
        # Fallback stats
        "cohort_fallbacks": compute_fallback_stats(snapshots),
        
        # Delisting diagnostics
        "delisting": provider_diagnostics or {},
        
        # Headline metrics
        "headline_metrics": extract_headline_metrics(backtest_result),
        
        # Hashes
        "hashes": {
            "config": backtest_result.get("provenance", {}).get("config_hash"),
            "results": compute_results_hash(backtest_result),
        },
        
        # Quality gates
        "quality_gates": {},
    }
    
    # Evaluate quality gates
    coverage = summary["coverage"]
    fallbacks = summary["cohort_fallbacks"]
    
    summary["quality_gates"] = {
        "return_coverage_ok": coverage.get("return_coverage_rate", 0) >= 0.80,
        "fallback_rate_ok": fallbacks.get("fallback_rate", 1) <= 0.20,
        "validations_ok": summary["validations"]["all_passed"],
    }
    summary["quality_gates"]["all_gates_passed"] = all(summary["quality_gates"].values())
    
    # Stability gates (non-fatal, for measurement mode)
    # These are intentionally looser than eventual production targets
    summary["stability_gates"] = {
        "note": "Stability gates are informational during measurement mode",
        "rank_corr_threshold": 0.35,
        "churn_threshold": 0.65,
        # Will be populated when sanity metrics are run
        "rank_corr_ok": None,
        "churn_ok": None,
    }
    
    # Save to disk
    summary_file = output_dir / "run_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, cls=DecimalEncoder)
    
    return summary


def print_run_summary(summary: Dict[str, Any]) -> None:
    """Print formatted run summary to console."""
    print("\n" + "=" * 70)
    print(f"RUN SUMMARY: {summary['run_id']}")
    print("=" * 70)
    
    # Quality gates
    gates = summary.get("quality_gates", {})
    all_ok = gates.get("all_gates_passed", False)
    status = "✓ ALL GATES PASSED" if all_ok else "✗ GATES FAILED"
    print(f"\n{status}")
    print("-" * 40)
    for gate, passed in gates.items():
        if gate != "all_gates_passed":
            symbol = "✓" if passed else "✗"
            print(f"  {symbol} {gate}")
    
    # Validations
    print("\nValidations:")
    for name, passed in summary.get("validations", {}).get("results", {}).items():
        symbol = "✓" if passed else "✗"
        print(f"  {symbol} {name}")
    
    # Coverage
    cov = summary.get("coverage", {})
    print(f"\nCoverage:")
    print(f"  Rankable rate:        {cov.get('rankable_rate', 0)*100:.1f}%")
    print(f"  Return coverage:      {cov.get('return_coverage_rate', 0)*100:.1f}%")
    
    # Fallbacks
    fb = summary.get("cohort_fallbacks", {})
    print(f"\nCohort Normalization:")
    print(f"  Normal securities:    {fb.get('normal_securities', 0)}")
    print(f"  Fallback securities:  {fb.get('fallback_securities', 0)}")
    if fb.get("fallback_reasons"):
        for reason, count in fb["fallback_reasons"].items():
            print(f"    - {reason}: {count}")
    print(f"  Fallback rate:        {fb.get('fallback_rate', 0)*100:.1f}% (of securities)")
    
    # Stage size warnings
    if fb.get("stages_below_min"):
        print(f"\n  ⚠ Stages below min size ({fb.get('min_stage_size_threshold', 5)}):")
        for stage in fb["stages_below_min"]:
            stats = fb.get("stage_sizes", {}).get(stage, {})
            print(f"    - {stage}: min={stats.get('min', 'N/A')} (treat subgroup IC as informational only)")
    
    # Delisting
    delist = summary.get("delisting", {})
    if delist:
        print(f"\nDelisting / Missing Returns:")
        print(f"  Policy:                    {delist.get('delisting_policy', 'unknown')}")
        print(f"  Ticker not in data:        {delist.get('n_missing_ticker_not_in_data', 0)}")
        print(f"  Missing start price:       {delist.get('n_missing_start_price', 0)}")
        print(f"  Missing end price:         {delist.get('n_missing_end_price', 0)}")
        print(f"  Delisted before end:       {delist.get('n_missing_due_to_delist', 0)}")
        print(f"  Returns calculated:        {delist.get('n_returns_calculated', 0)}")
    
    # Headlines
    print(f"\nHeadline Metrics:")
    for horizon, metrics in summary.get("headline_metrics", {}).items():
        ic = metrics.get("ic_mean")
        spread = metrics.get("bucket_spread_mean")
        ic_str = f"{ic:.4f}" if ic else "N/A"
        spread_str = f"{spread*100:.1f}%" if spread else "N/A"
        print(f"  {horizon}: IC={ic_str}, Spread={spread_str}")
    
    # Decision
    print("\n" + "-" * 70)
    if all_ok:
        print("DECISION: Proceed with analysis")
    else:
        print("DECISION: Fix data/coverage issues before interpreting IC")
    print("=" * 70)
