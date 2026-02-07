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
        ms = r.get("morningstar_signal", {}) or {}
        enh = (sb.get("enhancements", {}) or {}).get("smart_money_reinforcement", {}) or {}

        # Extract component scores from nested structure
        components = sb.get("components", []) or []
        comp_scores = {}
        for c in components:
            name = c.get("name")
            if name:
                comp_scores[name] = {
                    "raw": c.get("raw"),
                    "normalized": c.get("normalized"),
                    "contribution": c.get("contribution"),
                }

        surv_metrics = surv.get("metrics", {}) or {}

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

            # Component scores (from score_breakdown.components)
            "clinical_score": comp_scores.get("clinical", {}).get("normalized"),
            "financial_score": comp_scores.get("financial", {}).get("normalized"),
            "catalyst_score": comp_scores.get("catalyst", {}).get("normalized"),
            "pos_score": comp_scores.get("pos", {}).get("normalized"),
            "smart_money_score": comp_scores.get("smart_money", {}).get("normalized"),

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
            "days_to_catalyst": cat_decay.get("days_to_catalyst"),
            "in_optimal_window": cat_decay.get("in_optimal_window"),
            "catalyst_score_effective": cat_eff.get("catalyst_score_effective"),

            # Momentum (price-based)
            "alpha_60d": mom.get("alpha_60d"),
            "momentum_score": mom.get("momentum_score"),
            "momentum_window": mom.get("window_used"),

            # Valuation
            "valuation_method": val.get("method"),
            "mcap_per_asset": val.get("mcap_per_asset"),
            "ev_multiple": val.get("ev_multiple"),
            "peer_count": val.get("peer_count"),

            # Short interest
            "si_score": si.get("score"),
            "si_crowding_risk": si.get("crowding_risk"),
            "si_squeeze_potential": si.get("squeeze_potential"),

            # Volatility
            "annualized_vol_pct": vol_adj.get("annualized_vol_pct"),
            "vol_bucket": vol_adj.get("vol_bucket"),

            # Partnership
            "partnership_count": part.get("partnership_count"),
            "partnership_strength": part.get("partnership_strength"),
            "top_tier_partners": part.get("top_tier_partners"),

            # FDA designations
            "has_fda_designations": fda.get("has_designations"),
            "designation_types": ", ".join(fda.get("designation_types", [])) or None,

            # Pipeline
            "pipeline_diversity_score": pipe.get("diversity_score"),
            "pipeline_risk_profile": pipe.get("risk_profile"),
            "program_count": pipe.get("program_count"),

            # Competitive
            "competitive_position": comp.get("competitive_position"),
            "crowding_level": comp.get("crowding_level"),
            "competitor_count": comp.get("competitor_count"),

            # Survivability
            "cash_runway_months": surv_metrics.get("effective_runway_months"),
            "survivability_score": surv.get("score"),

            # Morningstar signal
            "ms_composite": ms.get("morningstar_score"),
            "ms_fv_discount": ms.get("fair_value_discount_score"),
            "ms_capital_efficiency": ms.get("capital_efficiency_score"),
            "ms_leverage_health": ms.get("leverage_health_score"),
            "ms_growth_quality": ms.get("growth_quality_score"),
            "ms_momentum": ms.get("momentum_score"),
            "ms_moat_quality": ms.get("moat_quality_score"),
            "ms_regime": ms.get("regime"),
            "ms_confidence": ms.get("confidence"),
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
