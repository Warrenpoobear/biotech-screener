#!/usr/bin/env python3
"""Export screening results to CSV with all columns."""
import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# Core columns (always first, in order)
CORE_COLUMNS = [
    "ticker", "composite_rank", "composite_score", "z_score",
    "expected_excess_return", "volatility", "drawdown", "cluster_id",
    "corr_xbi", "beta_xbi",  # Diversification proof columns
    "vol_blended", "vol_63d", "vol_252d",
    "max_drawdown_blended", "max_drawdown_252d", "max_drawdown_2y",
    "risk_data_state_vol", "risk_data_state_drawdown", "risk_data_state_beta",
    "confidence_risk",
    "defensive_multiplier", "defensive_bucket", "defensive_notes",
    "rank_driver",  # IC audit: alpha | defensive_boost | defensive_penalty | suppressed
    "fundamental_red_flag", "fundamental_red_flag_reasons",
    "severity", "stage_bucket", "market_cap_bucket", "rankable",
]

# Signal columns (flattened from nested dicts)
SIGNAL_COLUMNS = [
    "momentum_score", "momentum_window", "momentum_alpha",
    "catalyst_score", "catalyst_effective_score", "catalyst_proximity_score",
    "smart_money_score", "smart_money_overlap", "smart_money_tier1_holders",
    "short_interest_score", "short_interest_crowding", "short_interest_squeeze",
    "valuation_score", "valuation_method", "valuation_confidence",
    "valuation_raw_metric", "valuation_mcap_mm",
    "valuation_denom_trials_total",
    "valuation_denom_trials_p1", "valuation_denom_trials_p2",
    "valuation_denom_trials_p3", "valuation_denom_trials_reg",
    "valuation_phase_weighted_trials", "valuation_metric_pw",
    "valuation_pct_overall", "valuation_pct_in_size_bucket",
    "valuation_size_bucket",
    "valuation_applicable", "valuation_notes",
    "partnership_score", "partnership_strength", "partnership_top_partners",
    "fda_designation_score", "fda_designations",
    "competitive_intensity_score", "competitive_crowding_level",
    "pipeline_diversity_score", "pipeline_risk_profile",
]

# Confidence columns
CONFIDENCE_COLUMNS = [
    "confidence_overall", "confidence_financial", "confidence_clinical",
    "confidence_catalyst", "confidence_pos",
]


def _cheapness_percentile(value: float, all_values: List[float]) -> float:
    """Percentile where 1.0 = cheapest (lowest raw metric), 0.0 = most expensive.

    Uses midpoint method: pct = (count_above + 0.5 * count_equal) / n.
    Lower raw metric = cheaper = higher percentile.
    """
    n = len(all_values)
    if n == 0:
        return 0.0
    above = sum(1 for v in all_values if v > value)
    equal = sum(1 for v in all_values if v == value)
    return round((above + 0.5 * equal) / n, 4)


def _compute_valuation_percentiles(flat_records: List[Dict[str, Any]]) -> None:
    """Compute cross-sectional valuation percentiles in-place (ranking-neutral).

    Populates valuation_pct_overall and valuation_pct_in_size_bucket.
    Percentiles are computed within each valuation_method separately so that
    mcap_per_trial, ev_cfo, and ev_revenue are not mixed (different scales).
    """
    # Collect non-null raw metrics: (idx, method, bucket, raw_float)
    indexed: List[tuple] = []
    for i, rec in enumerate(flat_records):
        raw = rec.get("valuation_raw_metric")
        if raw is not None and raw != "":
            try:
                indexed.append((
                    i, float(raw),
                    rec.get("valuation_method", ""),
                    rec.get("valuation_size_bucket", ""),
                ))
            except (ValueError, TypeError):
                pass

    if not indexed:
        return

    # Group values by method (for pct_overall within method)
    method_values: Dict[str, List[float]] = {}
    for _, v, method, _ in indexed:
        method_values.setdefault(method, []).append(v)

    # Group values by (method, bucket) (for pct_in_size_bucket)
    method_bucket_values: Dict[tuple, List[float]] = {}
    for _, v, method, bucket in indexed:
        method_bucket_values.setdefault((method, bucket), []).append(v)

    # Assign percentiles
    for i, v, method, bucket in indexed:
        mv = method_values.get(method, [])
        if len(mv) >= 2:
            flat_records[i]["valuation_pct_overall"] = _cheapness_percentile(v, mv)
        mbv = method_bucket_values.get((method, bucket), [])
        if len(mbv) >= 2:
            flat_records[i]["valuation_pct_in_size_bucket"] = _cheapness_percentile(v, mbv)


def flatten_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten a ranked security record for CSV export."""
    flat = {}

    # Core columns
    for col in CORE_COLUMNS:
        val = rec.get(col)
        if col == "defensive_notes" and isinstance(val, list):
            flat[col] = "; ".join(val) if val else ""
        else:
            flat[col] = val

    # Defensive features (corr/beta/vol/drawdown for diversification proof)
    def_feat = rec.get("defensive_features") or {}
    flat["corr_xbi"] = def_feat.get("corr_xbi")
    flat["beta_xbi"] = def_feat.get("beta_xbi_60d")
    flat["volatility"] = def_feat.get("vol_60d")
    flat["drawdown"] = def_feat.get("drawdown")

    # V2 multi-horizon risk columns (with fallbacks from legacy fields)
    flat["vol_blended"] = def_feat.get("vol_blended") or def_feat.get("vol_60d")
    flat["vol_63d"] = def_feat.get("vol_63d")
    flat["vol_252d"] = def_feat.get("vol_252d")
    flat["max_drawdown_blended"] = def_feat.get("max_drawdown_blended") or def_feat.get("drawdown")
    flat["max_drawdown_252d"] = def_feat.get("max_drawdown_252d")
    flat["max_drawdown_2y"] = def_feat.get("max_drawdown_2y")

    # Derive risk_data_state from presence of corresponding metrics when not explicit
    flat["risk_data_state_vol"] = def_feat.get("risk_data_state_vol") or (
        "live" if def_feat.get("vol_blended") or def_feat.get("vol_60d") else "missing"
    )
    flat["risk_data_state_drawdown"] = def_feat.get("risk_data_state_drawdown") or (
        "live" if def_feat.get("max_drawdown_blended") or def_feat.get("drawdown") else "missing"
    )
    flat["risk_data_state_beta"] = def_feat.get("risk_data_state_beta") or (
        "live" if def_feat.get("beta_xbi_60d") else "missing"
    )

    # Derive confidence_risk from coverage when not explicit
    if def_feat.get("confidence_risk") is not None:
        flat["confidence_risk"] = def_feat.get("confidence_risk")
    else:
        state_fields = [flat["risk_data_state_vol"], flat["risk_data_state_drawdown"],
                        flat["risk_data_state_beta"]]
        live_count = sum(1 for s in state_fields if s == "live")
        # Continuous mapping: 0.35 + (live_count/3) * 0.55
        flat["confidence_risk"] = round(0.35 + (live_count / 3) * 0.55, 2)

    # Rank driver and red-flag fields
    flat["rank_driver"] = rec.get("rank_driver")
    flat["fundamental_red_flag"] = rec.get("fundamental_red_flag")
    red_flag_reasons = rec.get("fundamental_red_flag_reasons") or []
    flat["fundamental_red_flag_reasons"] = "; ".join(red_flag_reasons) if red_flag_reasons else ""

    # Momentum signal
    mom = rec.get("momentum_signal") or {}
    flat["momentum_score"] = mom.get("momentum_score")
    flat["momentum_window"] = mom.get("window_used")
    flat["momentum_alpha"] = mom.get("alpha_60d")

    # Catalyst signal
    cat_eff = rec.get("catalyst_effective") or {}
    flat["catalyst_score"] = cat_eff.get("catalyst_score_window")
    flat["catalyst_effective_score"] = cat_eff.get("catalyst_score_effective")
    flat["catalyst_proximity_score"] = cat_eff.get("catalyst_proximity_score")

    # Smart money signal
    sm = rec.get("smart_money_signal") or {}
    flat["smart_money_score"] = sm.get("score")
    flat["smart_money_overlap"] = sm.get("overlap_count")
    tier1 = sm.get("tier1_holders") or []
    flat["smart_money_tier1_holders"] = "; ".join(tier1) if tier1 else ""

    # Short interest signal
    si = rec.get("short_interest_signal") or {}
    flat["short_interest_score"] = si.get("score")
    flat["short_interest_crowding"] = si.get("crowding_risk")
    flat["short_interest_squeeze"] = si.get("squeeze_potential")

    # Valuation signal (PoS)
    val_sig = rec.get("valuation_signal") or {}
    flat["valuation_score"] = val_sig.get("valuation_score")
    flat["valuation_method"] = val_sig.get("method")
    flat["valuation_confidence"] = val_sig.get("confidence")
    # Valuation debug columns (percentiles computed cross-sectionally in export_to_csv)
    for col in [
        "valuation_raw_metric", "valuation_mcap_mm",
        "valuation_denom_trials_total",
        "valuation_denom_trials_p1", "valuation_denom_trials_p2",
        "valuation_denom_trials_p3", "valuation_denom_trials_reg",
        "valuation_phase_weighted_trials", "valuation_metric_pw",
        "valuation_size_bucket",
        "valuation_applicable", "valuation_notes",
    ]:
        flat[col] = val_sig.get(col)

    # Partnership signal
    part = rec.get("partnership_signal") or {}
    flat["partnership_score"] = part.get("partnership_score")
    flat["partnership_strength"] = part.get("partnership_strength")
    top_partners = part.get("top_partners") or []
    flat["partnership_top_partners"] = "; ".join(top_partners) if top_partners else ""

    # FDA designation signal
    fda = rec.get("fda_designation_signal") or {}
    flat["fda_designation_score"] = fda.get("designation_score")
    desigs = fda.get("designation_types") or []
    flat["fda_designations"] = "; ".join(desigs) if desigs else ""

    # Competitive intensity signal
    ci = rec.get("competitive_intensity_signal") or {}
    flat["competitive_intensity_score"] = ci.get("intensity_score")
    flat["competitive_crowding_level"] = ci.get("crowding_level")

    # Pipeline diversity signal
    pd = rec.get("pipeline_diversity_signal") or {}
    flat["pipeline_diversity_score"] = pd.get("diversity_score")
    flat["pipeline_risk_profile"] = pd.get("risk_profile")

    # Confidence columns
    for col in CONFIDENCE_COLUMNS:
        flat[col] = rec.get(col)

    return flat


def export_to_csv(results_path: Path, output_path: Path) -> int:
    """Export results JSON to CSV. Returns number of records exported."""
    with open(results_path) as f:
        data = json.load(f)

    ranked = data.get("module_5_composite", {}).get("ranked_securities", [])
    if not ranked:
        print("Error: No ranked securities found in results", file=sys.stderr)
        return 0

    # Sort by rank
    ranked = sorted(ranked, key=lambda x: x.get("composite_rank", 999))

    # Flatten all records
    flat_records = [flatten_record(r) for r in ranked]

    # Cross-sectional valuation percentiles (ranking-neutral, diagnostic only)
    # Lower raw metric = cheaper = higher percentile (1.0 = cheapest, 0.0 = most expensive)
    _compute_valuation_percentiles(flat_records)

    # Get all columns (core + signal + confidence)
    columns = CORE_COLUMNS + SIGNAL_COLUMNS + CONFIDENCE_COLUMNS

    # Null safety: convert any remaining None values to "" for clean CSV output
    for rec in flat_records:
        for col in columns:
            if rec.get(col) is None:
                rec[col] = ""

    # Write CSV
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(flat_records)

    return len(flat_records)


def main():
    parser = argparse.ArgumentParser(description="Export screening results to CSV")
    parser.add_argument("input", help="Results JSON file")
    parser.add_argument("-o", "--output", help="Output CSV file (default: <input>.csv)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: File not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output) if args.output else input_path.with_suffix(".csv")

    count = export_to_csv(input_path, output_path)
    if count > 0:
        print(f"Exported {count} securities to {output_path}")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
