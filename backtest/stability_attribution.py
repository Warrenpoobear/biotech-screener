"""
Stability Attribution Panel

Decomposes month-to-month rank changes to identify WHY ranks moved.

For each consecutive month pair (t-1 → t):
- Δcomposite
- Δclinical_norm, Δfinancial_norm, Δcatalyst_norm
- Δuncertainty_penalty
- Δseverity_multiplier

Outputs:
- Top 20 movers by |Δcomposite|
- Aggregate attribution: fraction of total |Δcomposite| from each component
- Diagnosis: normalization geometry vs real signal vs missingness
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

import sys
sys.path.insert(0, "/home/claude/biotech_screener")


def compute_monthly_deltas(
    snap_prev: Dict[str, Any],
    snap_curr: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compute deltas between two consecutive snapshots.
    
    Returns:
        {
            "date_prev": str,
            "date_curr": str,
            "ticker_deltas": [{
                "ticker": str,
                "rank_prev": int,
                "rank_curr": int,
                "Δrank": int,
                "composite_prev": float,
                "composite_curr": float,
                "Δcomposite": float,
                "Δclinical_norm": float,
                "Δfinancial_norm": float,
                "Δcatalyst_norm": float,
                "Δuncertainty_penalty": float,
                "severity_prev": str,
                "severity_curr": str,
                "severity_changed": bool,
            }],
            "top_movers": [...],  # Top 20 by |Δcomposite|
            "aggregate_attribution": {
                "clinical_pct": float,
                "financial_pct": float,
                "catalyst_pct": float,
                "uncertainty_pct": float,
                "severity_pct": float,
            },
        }
    """
    date_prev = snap_prev["as_of_date"]
    date_curr = snap_curr["as_of_date"]
    
    # Index previous snapshot by ticker
    prev_by_ticker = {
        sec["ticker"]: sec for sec in snap_prev.get("ranked_securities", [])
    }
    
    # Compute deltas for each ticker in current snapshot
    ticker_deltas = []
    
    # For aggregate attribution
    total_abs_delta = 0.0
    component_contributions = {
        "clinical": 0.0,
        "financial": 0.0,
        "catalyst": 0.0,
        "uncertainty": 0.0,
        "severity": 0.0,
    }
    
    for sec in snap_curr.get("ranked_securities", []):
        ticker = sec["ticker"]
        
        if ticker not in prev_by_ticker:
            continue  # New ticker, skip
        
        prev = prev_by_ticker[ticker]
        
        # Extract values (handle string → float)
        def safe_float(val):
            if val is None:
                return 0.0
            try:
                return float(val)
            except (ValueError, TypeError):
                return 0.0
        
        composite_prev = safe_float(prev.get("composite_score"))
        composite_curr = safe_float(sec.get("composite_score"))
        
        clinical_prev = safe_float(prev.get("clinical_dev_normalized"))
        clinical_curr = safe_float(sec.get("clinical_dev_normalized"))
        
        financial_prev = safe_float(prev.get("financial_normalized"))
        financial_curr = safe_float(sec.get("financial_normalized"))
        
        catalyst_prev = safe_float(prev.get("catalyst_normalized"))
        catalyst_curr = safe_float(sec.get("catalyst_normalized"))
        
        uncertainty_prev = safe_float(prev.get("uncertainty_penalty"))
        uncertainty_curr = safe_float(sec.get("uncertainty_penalty"))
        
        severity_prev = prev.get("severity", "none")
        severity_curr = sec.get("severity", "none")
        
        rank_prev = prev.get("composite_rank", 0)
        rank_curr = sec.get("composite_rank", 0)
        
        # Compute deltas
        d_composite = composite_curr - composite_prev
        d_clinical = clinical_curr - clinical_prev
        d_financial = financial_curr - financial_prev
        d_catalyst = catalyst_curr - catalyst_prev
        d_uncertainty = uncertainty_curr - uncertainty_prev
        d_rank = rank_curr - rank_prev
        
        # Severity change contribution (approximate)
        # Severity multipliers: none=1.0, sev1=0.9, sev2=0.5, sev3=0
        sev_mult = {"none": 1.0, "sev1": 0.9, "sev2": 0.5, "sev3": 0.0}
        sev_prev_mult = sev_mult.get(severity_prev, 1.0)
        sev_curr_mult = sev_mult.get(severity_curr, 1.0)
        d_severity_mult = sev_curr_mult - sev_prev_mult
        
        ticker_deltas.append({
            "ticker": ticker,
            "rank_prev": rank_prev,
            "rank_curr": rank_curr,
            "Δrank": d_rank,
            "composite_prev": round(composite_prev, 2),
            "composite_curr": round(composite_curr, 2),
            "Δcomposite": round(d_composite, 4),
            "Δclinical_norm": round(d_clinical, 4),
            "Δfinancial_norm": round(d_financial, 4),
            "Δcatalyst_norm": round(d_catalyst, 4),
            "Δuncertainty_penalty": round(d_uncertainty, 4),
            "severity_prev": severity_prev,
            "severity_curr": severity_curr,
            "severity_changed": severity_prev != severity_curr,
            "Δseverity_mult": round(d_severity_mult, 2),
        })
        
        # Aggregate attribution (using absolute values)
        abs_d = abs(d_composite)
        total_abs_delta += abs_d
        
        # Approximate contribution of each component
        # Using weighted deltas (weights: clinical=0.4, financial=0.35, catalyst=0.25)
        if abs_d > 0.001:
            component_contributions["clinical"] += abs(d_clinical * 0.4)
            component_contributions["financial"] += abs(d_financial * 0.35)
            component_contributions["catalyst"] += abs(d_catalyst * 0.25)
            component_contributions["uncertainty"] += abs(d_uncertainty * composite_curr) if composite_curr > 0 else 0
            component_contributions["severity"] += abs(d_severity_mult * (composite_curr / sev_curr_mult)) if sev_curr_mult > 0 else 0
    
    # Compute percentages
    total_contrib = sum(component_contributions.values())
    aggregate_attribution = {}
    if total_contrib > 0:
        aggregate_attribution = {
            f"{k}_pct": round(v / total_contrib * 100, 1)
            for k, v in component_contributions.items()
        }
    else:
        aggregate_attribution = {f"{k}_pct": 0.0 for k in component_contributions}
    
    # Top movers by |Δcomposite|
    top_movers = sorted(ticker_deltas, key=lambda x: abs(x["Δcomposite"]), reverse=True)[:20]
    
    return {
        "date_prev": date_prev,
        "date_curr": date_curr,
        "n_common_tickers": len(ticker_deltas),
        "ticker_deltas": ticker_deltas,
        "top_movers": top_movers,
        "aggregate_attribution": aggregate_attribution,
        "total_abs_delta": round(total_abs_delta, 4),
    }


def compute_stability_attribution(
    snapshots: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Compute stability attribution across all consecutive month pairs.
    
    Returns comprehensive decomposition of why ranks changed.
    """
    if len(snapshots) < 2:
        return {
            "error": "Need at least 2 snapshots",
            "monthly_panels": [],
            "summary": {},
        }
    
    # Sort by date
    sorted_snaps = sorted(snapshots, key=lambda x: x["as_of_date"])
    
    monthly_panels = []
    
    # Aggregate across all months
    all_attributions = defaultdict(list)
    all_top_movers = []
    
    for i in range(1, len(sorted_snaps)):
        snap_prev = sorted_snaps[i - 1]
        snap_curr = sorted_snaps[i]
        
        panel = compute_monthly_deltas(snap_prev, snap_curr)
        monthly_panels.append(panel)
        
        # Collect attributions
        for k, v in panel["aggregate_attribution"].items():
            all_attributions[k].append(v)
        
        # Collect top movers
        for mover in panel["top_movers"][:5]:  # Top 5 per month
            mover["month"] = f"{panel['date_prev']} → {panel['date_curr']}"
            all_top_movers.append(mover)
    
    # Compute summary
    summary = {
        "n_periods": len(monthly_panels),
        "mean_attribution": {
            k: round(mean(v), 1) if v else 0
            for k, v in all_attributions.items()
        },
        "dominant_driver": None,
        "interpretation": "",
    }
    
    # Identify dominant driver
    if summary["mean_attribution"]:
        dominant = max(summary["mean_attribution"].items(), key=lambda x: x[1])
        summary["dominant_driver"] = dominant[0].replace("_pct", "")
        
        # Interpretation
        driver = summary["dominant_driver"]
        pct = dominant[1]
        
        if driver == "clinical":
            summary["interpretation"] = f"Clinical score drives {pct:.0f}% of rank changes (expected if pipeline data changes)"
        elif driver == "financial":
            summary["interpretation"] = f"Financial score drives {pct:.0f}% of rank changes (check for quarterly updates)"
        elif driver == "catalyst":
            summary["interpretation"] = f"Catalyst score drives {pct:.0f}% of rank changes (check trial status changes)"
        elif driver == "uncertainty":
            summary["interpretation"] = f"Uncertainty penalty drives {pct:.0f}% of rank changes (missingness volatility)"
        elif driver == "severity":
            summary["interpretation"] = f"Severity gates drive {pct:.0f}% of rank changes (flag toggling)"
    
    # Global top movers
    global_top_movers = sorted(all_top_movers, key=lambda x: abs(x["Δcomposite"]), reverse=True)[:20]
    
    return {
        "monthly_panels": monthly_panels,
        "summary": summary,
        "global_top_movers": global_top_movers,
    }


def diagnose_instability(
    attribution: Dict[str, Any],
    rank_stability: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Diagnose root cause of rank instability.
    
    Categories:
    - normalization_geometry: cohort edges / fallback switching
    - missingness_penalty: uncertainty penalty changes
    - severity_gates: severity flag toggling
    - real_signal: actual sub-score movement
    """
    diagnosis = {
        "primary_cause": None,
        "confidence": "low",
        "evidence": [],
        "recommendations": [],
    }
    
    summary = attribution.get("summary", {})
    mean_attr = summary.get("mean_attribution", {})
    
    rank_corr = rank_stability.get("rank_corr_mean")
    churn = rank_stability.get("churn_mean")
    
    # Check for uncertainty-driven instability
    if mean_attr.get("uncertainty_pct", 0) > 30:
        diagnosis["evidence"].append(f"Uncertainty penalty drives {mean_attr['uncertainty_pct']:.0f}% of changes")
        diagnosis["recommendations"].append("Check for data completeness; reduce missingness")
        if diagnosis["primary_cause"] is None:
            diagnosis["primary_cause"] = "missingness_penalty"
    
    # Check for severity-driven instability
    if mean_attr.get("severity_pct", 0) > 20:
        diagnosis["evidence"].append(f"Severity gates drive {mean_attr['severity_pct']:.0f}% of changes")
        diagnosis["recommendations"].append("Review severity flag criteria; may be too sensitive")
        if diagnosis["primary_cause"] is None:
            diagnosis["primary_cause"] = "severity_gates"
    
    # Check for normalization issues (look at monthly panels for cohort changes)
    # This would require cohort tracking which we can add later
    
    # If primary drivers are clinical/financial/catalyst, it's likely real signal
    clinical_pct = mean_attr.get("clinical_pct", 0)
    financial_pct = mean_attr.get("financial_pct", 0)
    catalyst_pct = mean_attr.get("catalyst_pct", 0)
    
    real_signal_pct = clinical_pct + financial_pct + catalyst_pct
    
    if real_signal_pct > 70 and rank_corr and rank_corr > 0.3:
        diagnosis["primary_cause"] = "real_signal"
        diagnosis["confidence"] = "medium"
        diagnosis["evidence"].append(f"Sub-scores drive {real_signal_pct:.0f}% of changes with moderate stability")
    elif real_signal_pct > 70 and (rank_corr is None or rank_corr < 0.3):
        diagnosis["primary_cause"] = "real_signal_or_noise"
        diagnosis["confidence"] = "low"
        diagnosis["evidence"].append(f"Sub-scores drive {real_signal_pct:.0f}% but stability is low - may be input noise")
        diagnosis["recommendations"].append("Verify input data is not randomized; check for real temporal autocorrelation")
    
    if diagnosis["primary_cause"] is None:
        diagnosis["primary_cause"] = "unknown"
        diagnosis["recommendations"].append("Collect more data for diagnosis")
    
    return diagnosis


def print_stability_attribution(
    attribution: Dict[str, Any],
    rank_stability: Optional[Dict[str, Any]] = None,
) -> None:
    """Print formatted stability attribution report."""
    print("\n" + "=" * 70)
    print("STABILITY ATTRIBUTION PANEL")
    print("=" * 70)
    
    summary = attribution.get("summary", {})
    
    print(f"\nPeriods analyzed: {summary.get('n_periods', 0)}")
    
    # Mean attribution
    print("\nAggregate Attribution (what drives rank changes):")
    print("-" * 50)
    mean_attr = summary.get("mean_attribution", {})
    for component in ["clinical", "financial", "catalyst", "uncertainty", "severity"]:
        pct = mean_attr.get(f"{component}_pct", 0)
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"  {component:<12} {bar} {pct:>5.1f}%")
    
    print(f"\nDominant driver: {summary.get('dominant_driver', 'unknown')}")
    print(f"Interpretation: {summary.get('interpretation', 'N/A')}")
    
    # Top movers
    print("\nGlobal Top 10 Movers (largest |Δcomposite|):")
    print("-" * 70)
    print(f"  {'Ticker':<8} {'Month':<25} {'Δcomp':>8} {'Δclin':>8} {'Δfin':>8} {'Δcat':>8}")
    print("  " + "-" * 65)
    
    for mover in attribution.get("global_top_movers", [])[:10]:
        print(f"  {mover['ticker']:<8} {mover.get('month', 'N/A'):<25} "
              f"{mover['Δcomposite']:>8.2f} {mover['Δclinical_norm']:>8.2f} "
              f"{mover['Δfinancial_norm']:>8.2f} {mover['Δcatalyst_norm']:>8.2f}")
    
    # Diagnosis
    if rank_stability:
        diagnosis = diagnose_instability(attribution, rank_stability)
        print("\nInstability Diagnosis:")
        print("-" * 50)
        print(f"  Primary cause: {diagnosis['primary_cause']} (confidence: {diagnosis['confidence']})")
        for ev in diagnosis["evidence"]:
            print(f"  • {ev}")
        if diagnosis["recommendations"]:
            print("\n  Recommendations:")
            for rec in diagnosis["recommendations"]:
                print(f"    → {rec}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    # Example usage
    from scripts.run_first_real_backtest import (
        BIOTECH_UNIVERSE,
        generate_monthly_dates,
        run_pipeline,
    )
    from backtest.sanity_metrics import compute_rank_stability
    
    print("Generating snapshots...")
    dates = generate_monthly_dates(2023, 2024)
    snapshots = [run_pipeline(BIOTECH_UNIVERSE, d, seed=100+i) for i, d in enumerate(dates)]
    
    print("Computing stability attribution...")
    attribution = compute_stability_attribution(snapshots)
    
    print("Computing rank stability...")
    rank_stability = compute_rank_stability(snapshots)
    
    print_stability_attribution(attribution, rank_stability)
    
    # Save results
    output_path = Path("/home/claude/biotech_screener/output/stability_attribution.json")
    with open(output_path, "w") as f:
        # Exclude full monthly panels for readability
        output = {
            "summary": attribution["summary"],
            "global_top_movers": attribution["global_top_movers"],
            "diagnosis": diagnose_instability(attribution, rank_stability),
        }
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")
