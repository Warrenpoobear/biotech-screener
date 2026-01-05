"""
Research Sanity Metrics

Step 2 diagnostics to validate screener behavior before interpretation.

1. IC by Subgroup (stage buckets)
2. Turnover Proxy (rank stability)
3. Factor Stability (sub-score correlations)
4. Delisting Sensitivity (policy comparison)

Run AFTER quality gates pass, BEFORE interpreting IC/spreads.
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from statistics import mean, stdev, correlation
from typing import Any, Dict, List, Optional, Tuple

import sys
sys.path.insert(0, "/home/claude/biotech_screener")

from backtest.metrics import (
    compute_spearman_ic,
    compute_forward_windows,
    run_metrics_suite,
)
from backtest.returns_provider import CSVReturnsProvider
from backtest.sharadar_provider import (
    SharadarReturnsProvider,
    DELISTING_POLICY_CONSERVATIVE,
    DELISTING_POLICY_LAST_PRICE,
    DELISTING_POLICY_PENALTY,
)


# ============================================================================
# 1. IC BY SUBGROUP (Stage Buckets)
# ============================================================================

def compute_ic_by_stage(
    snapshots: List[Dict],
    provider,
    horizon: str = "63d",
) -> Dict[str, Any]:
    """
    Compute IC separately for each stage bucket.
    
    Returns:
        {
            "pooled": {"ic_mean": float, "n_obs": int},
            "early": {"ic_mean": float, "n_obs": int},
            "mid": {"ic_mean": float, "n_obs": int},
            "late": {"ic_mean": float, "n_obs": int},
            "stage_sizes": {"early": int, "mid": int, "late": int},
            "stages_below_min": [...],
            "interpretation": str,
        }
    """
    MIN_STAGE_SIZE = 5
    
    # Group securities by stage across all snapshots
    stage_data: Dict[str, List[Tuple[float, float]]] = defaultdict(list)  # stage -> [(score, return)]
    pooled_data: List[Tuple[float, float]] = []
    
    stage_sizes: Dict[str, List[int]] = defaultdict(list)  # stage -> [count per date]
    
    for snap in snapshots:
        as_of = snap["as_of_date"]
        windows = compute_forward_windows(as_of, [horizon])
        w = windows[horizon]
        
        # Count by stage for this date
        date_stage_counts: Dict[str, int] = defaultdict(int)
        
        for sec in snap.get("ranked_securities", []):
            ticker = sec["ticker"]
            score = float(sec.get("composite_score", 0))
            stage = sec.get("stage_bucket", "unknown")
            
            # Get return
            ret = provider(ticker, w["start"], w["end"])
            if ret is None:
                continue
            
            ret_float = float(ret)
            
            # Add to stage group
            stage_data[stage].append((score, ret_float))
            pooled_data.append((score, ret_float))
            date_stage_counts[stage] += 1
        
        # Track stage sizes per date
        for stage, count in date_stage_counts.items():
            stage_sizes[stage].append(count)
    
    # Compute IC for each group
    results = {}
    
    # Pooled IC
    if len(pooled_data) >= 5:
        scores, returns = zip(*pooled_data)
        ic = compute_spearman_ic(list(scores), list(returns))
        results["pooled"] = {
            "ic_mean": round(ic, 4) if ic else None,
            "n_obs": len(pooled_data),
        }
    else:
        results["pooled"] = {"ic_mean": None, "n_obs": len(pooled_data)}
    
    # Per-stage IC
    for stage in ["early", "mid", "late", "unknown"]:
        data = stage_data.get(stage, [])
        if len(data) >= 5:
            scores, returns = zip(*data)
            ic = compute_spearman_ic(list(scores), list(returns))
            results[stage] = {
                "ic_mean": round(ic, 4) if ic else None,
                "n_obs": len(data),
            }
        else:
            results[stage] = {"ic_mean": None, "n_obs": len(data)}
    
    # Stage size stats
    results["stage_sizes"] = {
        stage: {
            "min": min(counts) if counts else 0,
            "max": max(counts) if counts else 0,
            "mean": round(mean(counts), 1) if counts else 0,
        }
        for stage, counts in stage_sizes.items()
    }
    
    # Flag stages below minimum
    results["stages_below_min"] = [
        stage for stage, stats in results["stage_sizes"].items()
        if stats["min"] < MIN_STAGE_SIZE
    ]
    results["min_stage_size_threshold"] = MIN_STAGE_SIZE
    
    # Interpretation
    pooled_ic = results["pooled"]["ic_mean"]
    stage_ics = {
        s: results[s]["ic_mean"] 
        for s in ["early", "mid", "late"] 
        if results[s]["ic_mean"] is not None
    }
    
    if pooled_ic is not None and stage_ics:
        # Check if pooled sign matches 2/3 stages
        pooled_sign = 1 if pooled_ic > 0 else -1
        matching_signs = sum(1 for ic in stage_ics.values() if (ic > 0) == (pooled_ic > 0))
        
        if matching_signs >= len(stage_ics) * 2 / 3:
            results["interpretation"] = "CONSISTENT: Pooled IC sign matches majority of stages"
        else:
            results["interpretation"] = "WARNING: Pooled IC may be dominated by one stage"
    else:
        results["interpretation"] = "INSUFFICIENT: Not enough data for interpretation"
    
    return results


# ============================================================================
# 2. TURNOVER PROXY (Rank Stability)
# ============================================================================

def compute_rank_stability(
    snapshots: List[Dict],
    top_pct: float = 0.20,  # Top quintile
) -> Dict[str, Any]:
    """
    Compute month-to-month rank stability.
    
    Returns:
        {
            "rank_correlations": [float],  # Spearman(rank_t, rank_t-1)
            "rank_corr_mean": float,
            "rank_corr_std": float,
            "top_quintile_churn": [float],  # % entering/leaving top quintile
            "churn_mean": float,
            "interpretation": str,
        }
    """
    if len(snapshots) < 2:
        return {
            "rank_correlations": [],
            "rank_corr_mean": None,
            "top_quintile_churn": [],
            "churn_mean": None,
            "interpretation": "INSUFFICIENT: Need at least 2 snapshots",
        }
    
    # Sort snapshots by date
    sorted_snaps = sorted(snapshots, key=lambda x: x["as_of_date"])
    
    rank_correlations = []
    top_quintile_churn = []
    
    prev_ranks = None
    prev_top_set = None
    
    for snap in sorted_snaps:
        # Build current ranks
        current_ranks = {}
        ranked_secs = snap.get("ranked_securities", [])
        
        for sec in ranked_secs:
            ticker = sec["ticker"]
            rank = sec.get("composite_rank", 0)
            current_ranks[ticker] = rank
        
        # Compute top quintile set
        n_top = max(1, int(len(ranked_secs) * top_pct))
        top_tickers = sorted(current_ranks.keys(), key=lambda t: current_ranks[t])[:n_top]
        current_top_set = set(top_tickers)
        
        if prev_ranks is not None:
            # Compute rank correlation for overlapping tickers
            common_tickers = set(current_ranks.keys()) & set(prev_ranks.keys())
            
            if len(common_tickers) >= 5:
                curr_list = [current_ranks[t] for t in common_tickers]
                prev_list = [prev_ranks[t] for t in common_tickers]
                
                corr = compute_spearman_ic(curr_list, prev_list)
                if corr is not None:
                    rank_correlations.append(corr)
            
            # Compute top quintile churn
            if prev_top_set:
                stayed = len(current_top_set & prev_top_set)
                total = len(current_top_set | prev_top_set)
                if total > 0:
                    # Churn = % that changed (entered or left)
                    churn = 1 - (stayed / max(len(current_top_set), len(prev_top_set)))
                    top_quintile_churn.append(churn)
        
        prev_ranks = current_ranks
        prev_top_set = current_top_set
    
    # Aggregate
    results = {
        "rank_correlations": [round(c, 4) for c in rank_correlations],
        "rank_corr_mean": round(mean(rank_correlations), 4) if rank_correlations else None,
        "rank_corr_std": round(stdev(rank_correlations), 4) if len(rank_correlations) > 1 else None,
        "top_quintile_churn": [round(c, 4) for c in top_quintile_churn],
        "churn_mean": round(mean(top_quintile_churn), 4) if top_quintile_churn else None,
        "n_periods": len(rank_correlations),
    }
    
    # Interpretation
    rc_mean = results["rank_corr_mean"]
    churn_mean = results["churn_mean"]
    
    interpretations = []
    
    if rc_mean is not None:
        if rc_mean >= 0.5:
            interpretations.append(f"STABLE: Rank correlation {rc_mean:.2f} >= 0.5")
        elif rc_mean >= 0.3:
            interpretations.append(f"MODERATE: Rank correlation {rc_mean:.2f} in [0.3, 0.5)")
        else:
            interpretations.append(f"UNSTABLE: Rank correlation {rc_mean:.2f} < 0.3 (high noise)")
    
    if churn_mean is not None:
        if churn_mean <= 0.50:
            interpretations.append(f"LOW CHURN: Top-quintile turnover {churn_mean*100:.0f}% <= 50%")
        else:
            interpretations.append(f"HIGH CHURN: Top-quintile turnover {churn_mean*100:.0f}% > 50%")
    
    results["interpretation"] = " | ".join(interpretations) if interpretations else "INSUFFICIENT DATA"
    
    return results


# ============================================================================
# 3. FACTOR STABILITY (Sub-score Correlations)
# ============================================================================

def compute_factor_stability(
    snapshots: List[Dict],
) -> Dict[str, Any]:
    """
    Track correlation of each sub-score with composite over time.
    
    Returns:
        {
            "clinical_vs_composite": [float],  # per date
            "financial_vs_composite": [float],
            "catalyst_vs_composite": [float],
            "summary": {
                "clinical": {"mean": float, "std": float, "sign_flips": int},
                ...
            },
            "interpretation": str,
        }
    """
    factor_corrs = {
        "clinical": [],
        "financial": [],
        "catalyst": [],
    }
    
    for snap in snapshots:
        ranked_secs = snap.get("ranked_securities", [])
        
        if len(ranked_secs) < 5:
            continue
        
        # Extract scores
        composite = []
        clinical = []
        financial = []
        catalyst = []
        
        for sec in ranked_secs:
            composite.append(float(sec.get("composite_score", 0)))
            clinical.append(float(sec.get("clinical_dev_normalized", 0)))
            financial.append(float(sec.get("financial_normalized", 0)))
            catalyst.append(float(sec.get("catalyst_normalized", 0)))
        
        # Compute correlations
        try:
            if len(set(clinical)) > 1 and len(set(composite)) > 1:
                factor_corrs["clinical"].append(correlation(clinical, composite))
        except:
            pass
        
        try:
            if len(set(financial)) > 1 and len(set(composite)) > 1:
                factor_corrs["financial"].append(correlation(financial, composite))
        except:
            pass
        
        try:
            if len(set(catalyst)) > 1 and len(set(composite)) > 1:
                factor_corrs["catalyst"].append(correlation(catalyst, composite))
        except:
            pass
    
    # Summarize
    results = {
        "clinical_vs_composite": [round(c, 4) for c in factor_corrs["clinical"]],
        "financial_vs_composite": [round(c, 4) for c in factor_corrs["financial"]],
        "catalyst_vs_composite": [round(c, 4) for c in factor_corrs["catalyst"]],
        "summary": {},
    }
    
    for factor, corrs in factor_corrs.items():
        if corrs:
            sign_flips = sum(1 for i in range(1, len(corrs)) if (corrs[i] > 0) != (corrs[i-1] > 0))
            results["summary"][factor] = {
                "mean": round(mean(corrs), 4),
                "std": round(stdev(corrs), 4) if len(corrs) > 1 else 0,
                "min": round(min(corrs), 4),
                "max": round(max(corrs), 4),
                "sign_flips": sign_flips,
                "n_periods": len(corrs),
            }
        else:
            results["summary"][factor] = {"mean": None, "n_periods": 0}
    
    # Interpretation
    issues = []
    for factor, stats in results["summary"].items():
        if stats.get("mean") is not None:
            if stats["mean"] < 0:
                issues.append(f"{factor} negatively correlated with composite")
            if stats.get("sign_flips", 0) > len(factor_corrs.get(factor, [])) * 0.3:
                issues.append(f"{factor} sign flips frequently")
            if stats.get("std", 0) > 0.3:
                issues.append(f"{factor} correlation unstable (std={stats['std']:.2f})")
    
    if issues:
        results["interpretation"] = "WARNING: " + "; ".join(issues)
    else:
        results["interpretation"] = "STABLE: All factors positively and consistently contribute"
    
    return results


# ============================================================================
# 4. DELISTING SENSITIVITY
# ============================================================================

def compute_delisting_sensitivity(
    snapshots: List[Dict],
    prices_file: str,
    tickers: List[str],
    horizon: str = "63d",
) -> Dict[str, Any]:
    """
    Compare backtest results under different delisting policies.
    
    Returns:
        {
            "conservative": {"ic_mean": float, "coverage": float},
            "last_price": {"ic_mean": float, "coverage": float},
            "penalty": {"ic_mean": float, "coverage": float},
            "sensitivity": {
                "ic_range": float,  # max - min IC
                "coverage_range": float,
            },
            "interpretation": str,
        }
    """
    policies = {
        "conservative": DELISTING_POLICY_CONSERVATIVE,
        "last_price": DELISTING_POLICY_LAST_PRICE,
        "penalty": DELISTING_POLICY_PENALTY,
    }
    
    results = {}
    
    for name, policy in policies.items():
        # Create provider with this policy
        provider = SharadarReturnsProvider.from_csv(
            prices_file,
            ticker_filter=tickers,
            delisting_policy=policy,
        )
        
        # Compute metrics
        metrics_result = run_metrics_suite(
            snapshots, provider, f"delist_{name}", horizons=[horizon]
        )
        
        agg = metrics_result.get("aggregate_metrics", {}).get(horizon, {})
        diag = provider.get_diagnostics()
        
        # Calculate coverage
        total_requests = (
            diag.get("n_returns_calculated", 0) +
            diag.get("n_missing_ticker_not_in_data", 0) +
            diag.get("n_missing_start_price", 0) +
            diag.get("n_missing_end_price", 0) +
            diag.get("n_missing_due_to_delist", 0)
        )
        coverage = diag.get("n_returns_calculated", 0) / total_requests if total_requests > 0 else 0
        
        results[name] = {
            "ic_mean": float(agg["ic_mean"]) if agg.get("ic_mean") else None,
            "spread_mean": float(agg["bucket_spread_mean"]) if agg.get("bucket_spread_mean") else None,
            "coverage": round(coverage, 4),
            "n_delist": diag.get("n_missing_due_to_delist", 0),
        }
    
    # Compute sensitivity
    policy_names = ["conservative", "last_price", "penalty"]
    ics = [results[p]["ic_mean"] for p in policy_names if results[p]["ic_mean"] is not None]
    coverages = [results[p]["coverage"] for p in policy_names]
    
    results["sensitivity"] = {
        "ic_range": round(max(ics) - min(ics), 4) if len(ics) >= 2 else None,
        "coverage_range": round(max(coverages) - min(coverages), 4),
    }
    
    # Interpretation
    ic_range = results["sensitivity"]["ic_range"]
    
    if ic_range is None:
        results["interpretation"] = "INSUFFICIENT: Could not compare IC across policies"
    elif ic_range < 0.02:
        results["interpretation"] = "ROBUST: IC stable across delisting policies (range < 0.02)"
    elif ic_range < 0.05:
        results["interpretation"] = "MODERATE: IC varies somewhat with delisting policy"
    else:
        results["interpretation"] = "SENSITIVE: Conclusions may depend on delisting treatment"
    
    # Check for sign flips
    policy_results = [results[p] for p in ["conservative", "last_price", "penalty"]]
    signs = [1 if r["ic_mean"] and r["ic_mean"] > 0 else -1 for r in policy_results if r["ic_mean"] is not None]
    if len(set(signs)) > 1:
        results["interpretation"] += " | WARNING: IC sign flips between policies!"
    
    return results


# ============================================================================
# COMBINED SANITY CHECK
# ============================================================================

def run_sanity_metrics(
    snapshots: List[Dict],
    provider,
    prices_file: str,
    tickers: List[str],
    horizon: str = "63d",
) -> Dict[str, Any]:
    """
    Run all sanity metrics and produce combined report.
    """
    print("\n" + "=" * 70)
    print("RESEARCH SANITY METRICS")
    print("=" * 70)
    
    # 1. IC by Stage
    print("\n[1] IC by Stage Bucket...")
    ic_by_stage = compute_ic_by_stage(snapshots, provider, horizon)
    
    print(f"    Pooled:  IC={ic_by_stage['pooled']['ic_mean']}, n={ic_by_stage['pooled']['n_obs']}")
    for stage in ["early", "mid", "late"]:
        ic = ic_by_stage[stage]["ic_mean"]
        n = ic_by_stage[stage]["n_obs"]
        flag = " ⚠" if stage in ic_by_stage.get("stages_below_min", []) else ""
        print(f"    {stage.capitalize():<7} IC={ic}, n={n}{flag}")
    print(f"    → {ic_by_stage['interpretation']}")
    
    # 2. Rank Stability
    print("\n[2] Rank Stability (month-to-month)...")
    rank_stability = compute_rank_stability(snapshots)
    
    print(f"    Rank correlation: mean={rank_stability['rank_corr_mean']}, std={rank_stability['rank_corr_std']}")
    print(f"    Top-quintile churn: mean={rank_stability['churn_mean']}")
    print(f"    → {rank_stability['interpretation']}")
    
    # 3. Factor Stability
    print("\n[3] Factor Stability...")
    factor_stability = compute_factor_stability(snapshots)
    
    for factor, stats in factor_stability["summary"].items():
        if stats.get("mean") is not None:
            print(f"    {factor.capitalize():<10} vs composite: mean={stats['mean']:.3f}, std={stats['std']:.3f}, flips={stats['sign_flips']}")
    print(f"    → {factor_stability['interpretation']}")
    
    # 4. Delisting Sensitivity
    print("\n[4] Delisting Sensitivity...")
    delist_sensitivity = compute_delisting_sensitivity(snapshots, prices_file, tickers, horizon)
    
    for policy in ["conservative", "last_price", "penalty"]:
        r = delist_sensitivity[policy]
        print(f"    {policy:<12} IC={r['ic_mean']}, coverage={r['coverage']*100:.1f}%")
    print(f"    IC range: {delist_sensitivity['sensitivity']['ic_range']}")
    print(f"    → {delist_sensitivity['interpretation']}")
    
    # Combined verdict
    print("\n" + "=" * 70)
    print("SANITY CHECK VERDICT")
    print("=" * 70)
    
    issues = []
    
    # Check IC by stage
    if "WARNING" in ic_by_stage["interpretation"]:
        issues.append("IC dominated by one stage")
    if ic_by_stage.get("stages_below_min"):
        issues.append(f"Stages below min size: {ic_by_stage['stages_below_min']}")
    
    # Check rank stability
    if rank_stability["rank_corr_mean"] is not None and rank_stability["rank_corr_mean"] < 0.3:
        issues.append("Low rank stability (< 0.3)")
    if rank_stability["churn_mean"] is not None and rank_stability["churn_mean"] > 0.5:
        issues.append("High top-quintile churn (> 50%)")
    
    # Check factor stability
    if "WARNING" in factor_stability["interpretation"]:
        issues.append("Factor contribution unstable")
    
    # Check delisting sensitivity
    if "SENSITIVE" in delist_sensitivity["interpretation"] or "WARNING" in delist_sensitivity["interpretation"]:
        issues.append("Results sensitive to delisting policy")
    
    if issues:
        print("⚠ ISSUES FOUND:")
        for issue in issues:
            print(f"  • {issue}")
        verdict = "INVESTIGATE"
    else:
        print("✓ All sanity checks passed")
        verdict = "PROCEED"
    
    print(f"\nVERDICT: {verdict}")
    print("=" * 70)
    
    return {
        "ic_by_stage": ic_by_stage,
        "rank_stability": rank_stability,
        "factor_stability": factor_stability,
        "delisting_sensitivity": delist_sensitivity,
        "issues": issues,
        "verdict": verdict,
    }


if __name__ == "__main__":
    # Example usage
    from scripts.run_first_real_backtest import (
        BIOTECH_UNIVERSE,
        generate_monthly_dates,
        run_pipeline,
    )
    
    print("Generating snapshots...")
    dates = generate_monthly_dates(2023, 2024)
    snapshots = [run_pipeline(BIOTECH_UNIVERSE, d, seed=100+i) for i, d in enumerate(dates)]
    
    prices_file = "/home/claude/biotech_screener/data/daily_prices.csv"
    provider = SharadarReturnsProvider.from_csv(
        prices_file,
        ticker_filter=BIOTECH_UNIVERSE,
        delisting_policy=DELISTING_POLICY_CONSERVATIVE,
    )
    
    results = run_sanity_metrics(
        snapshots, provider, prices_file, BIOTECH_UNIVERSE, horizon="63d"
    )
    
    # Save results
    output_path = Path("/home/claude/biotech_screener/output/sanity_metrics.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")
