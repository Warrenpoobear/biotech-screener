#!/usr/bin/env python3
"""
Export screen results JSON to CSV format.

Usage:
    python export_screen_csv.py production_data/screen_2026-01-31.json
    python export_screen_csv.py production_data/screen_2026-01-31.json -o custom_output.csv
"""
import argparse
import csv
import json
import sys
from pathlib import Path


def export_screen_to_csv(json_path: Path, csv_path: Path = None) -> Path:
    """
    Export screen results JSON to comprehensive CSV.

    Args:
        json_path: Path to screen JSON file
        csv_path: Optional output path (defaults to same name with .csv extension)

    Returns:
        Path to created CSV file
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    ranked = data["module_5_composite"]["ranked_securities"]

    rows = []
    for r in ranked:
        sb = r.get("score_breakdown", {}) or {}
        coin = r.get("coinvest", {}) or {}
        mom = r.get("momentum_signal", {}) or {}
        val = r.get("valuation_signal", {}) or {}
        si = r.get("short_interest_signal", {}) or {}
        cat_decay = r.get("catalyst_decay", {}) or {}
        cat_eff = r.get("catalyst_effective", {}) or {}
        vol_adj = r.get("volatility_adjustment", {}) or {}
        sm = r.get("smart_money_signal", {}) or {}
        part = r.get("partnership_signal", {}) or {}
        fda = r.get("fda_designation_signal", {}) or {}
        pipe = r.get("pipeline_diversity_signal", {}) or {}
        comp = r.get("competitive_intensity_signal", {}) or {}
        surv = r.get("survivability_signal", {}) or {}
        enh = (sb.get("enhancements", {}) or {}).get("smart_money_reinforcement", {}) or {}

        rows.append({
            # Core ranking
            "rank": r.get("composite_rank"),
            "ticker": r.get("ticker"),
            "composite_score": r.get("composite_score"),
            "risk_adjusted_score": r.get("risk_adjusted_score"),
            "expected_excess_return": r.get("expected_excess_return"),
            "score_z": r.get("score_z"),

            # Classification
            "stage_bucket": r.get("stage_bucket"),
            "market_cap_bucket": r.get("market_cap_bucket"),
            "severity": r.get("severity"),

            # Component scores (from breakdown)
            "clinical_score": sb.get("clinical"),
            "financial_score": sb.get("financial"),
            "catalyst_score": sb.get("catalyst"),
            "pos_score": sb.get("pos"),
            "momentum_score": sb.get("momentum"),
            "valuation_score": sb.get("valuation"),
            "smart_money_score": sb.get("smart_money"),

            # Confidence
            "confidence_overall": r.get("confidence_overall"),
            "confidence_clinical": r.get("confidence_clinical"),
            "confidence_catalyst": r.get("confidence_catalyst"),

            # Conviction (Baker-style)
            "conviction_overlap": coin.get("conviction_overlap"),
            "tier1_conviction_overlap": coin.get("tier1_conviction_overlap"),
            "tier1_count": coin.get("tier1_count"),
            "coinvest_overlap_count": r.get("coinvest_overlap_count"),

            # Reinforcement
            "reinforcement_applied": enh.get("reinforcement_applied"),
            "reinforcement_type": enh.get("reinforcement_type"),
            "thesis_gate_blocked": enh.get("thesis_gate_blocked"),
            "skip_reason": enh.get("skip_reason"),

            # Catalyst
            "catalyst_window_days": cat_eff.get("days_to_nearest"),
            "nearest_catalyst_type": cat_eff.get("nearest_event_type"),
            "in_optimal_window": cat_decay.get("in_optimal_window"),

            # Momentum
            "alpha_60d": mom.get("alpha_60d"),
            "momentum_bucket": mom.get("bucket"),

            # Valuation
            "ev_trial_ratio": val.get("ev_trial_ratio"),
            "peer_count": val.get("peer_count"),

            # Short interest
            "si_pct": si.get("si_pct"),
            "si_score": si.get("score"),

            # Volatility
            "annualized_vol_pct": vol_adj.get("annualized_vol_pct"),
            "vol_bucket": vol_adj.get("vol_bucket"),

            # Partnership
            "partnership_count": part.get("partnership_count"),
            "partnership_strength": part.get("strength"),
            "has_top_tier_partner": part.get("has_top_tier"),

            # FDA designations
            "has_fda_designations": fda.get("has_designations"),
            "designation_count": fda.get("designation_count"),

            # Pipeline
            "pipeline_diversity_score": pipe.get("diversity_score"),
            "is_single_asset": pipe.get("is_single_asset"),

            # Competitive
            "competitive_intensity": comp.get("intensity_bucket"),

            # Survivability
            "cash_runway_months": surv.get("cash_runway_months"),
            "burn_trajectory": surv.get("burn_trajectory"),
        })

    if csv_path is None:
        csv_path = json_path.with_suffix(".csv")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    return csv_path, len(rows)


def main():
    parser = argparse.ArgumentParser(description="Export screen results to CSV")
    parser.add_argument("json_path", type=Path, help="Path to screen JSON file")
    parser.add_argument("-o", "--output", type=Path, help="Output CSV path (default: same name with .csv)")
    args = parser.parse_args()

    if not args.json_path.exists():
        print(f"Error: {args.json_path} not found", file=sys.stderr)
        sys.exit(1)

    csv_path, row_count = export_screen_to_csv(args.json_path, args.output)
    print(f"Exported {row_count} rows to {csv_path}")


if __name__ == "__main__":
    main()
