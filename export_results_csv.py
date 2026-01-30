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
    "defensive_multiplier", "defensive_bucket", "defensive_notes",
    "severity", "stage_bucket", "market_cap_bucket", "rankable",
]

# Signal columns (flattened from nested dicts)
SIGNAL_COLUMNS = [
    "momentum_score", "momentum_window", "momentum_return",
    "catalyst_score", "catalyst_effective_score", "catalyst_event_count",
    "smart_money_score", "smart_money_overlap", "smart_money_tier1_holders",
    "short_interest_score", "short_interest_pct", "short_interest_days_to_cover",
    "pos_score", "pos_confidence",
    "partnership_score", "partnership_strength",
    "fda_designation_score", "fda_designations",
    "competitive_intensity_score", "competitive_intensity_level",
    "pipeline_diversity_score", "pipeline_diversity_status",
]

# Confidence columns
CONFIDENCE_COLUMNS = [
    "confidence_overall", "confidence_financial", "confidence_clinical",
    "confidence_catalyst", "confidence_pos",
]


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

    # Momentum signal
    mom = rec.get("momentum_signal") or {}
    flat["momentum_score"] = mom.get("momentum_score")
    flat["momentum_window"] = mom.get("window_used")
    flat["momentum_return"] = mom.get("return_value")

    # Catalyst signal
    cat_eff = rec.get("catalyst_effective") or {}
    flat["catalyst_score"] = cat_eff.get("catalyst_score")
    flat["catalyst_effective_score"] = cat_eff.get("effective_score")
    flat["catalyst_event_count"] = cat_eff.get("event_count")

    # Smart money signal
    sm = rec.get("smart_money_signal") or {}
    flat["smart_money_score"] = sm.get("smart_money_score")
    flat["smart_money_overlap"] = sm.get("overlap_count")
    tier1 = sm.get("tier1_holders") or []
    flat["smart_money_tier1_holders"] = "; ".join(tier1) if tier1 else ""

    # Short interest signal
    si = rec.get("short_interest_signal") or {}
    flat["short_interest_score"] = si.get("short_interest_score")
    flat["short_interest_pct"] = si.get("short_pct")
    flat["short_interest_days_to_cover"] = si.get("days_to_cover")

    # PoS (from valuation_signal)
    val_sig = rec.get("valuation_signal") or {}
    flat["pos_score"] = val_sig.get("pos_score")
    flat["pos_confidence"] = val_sig.get("pos_confidence")

    # Partnership signal
    part = rec.get("partnership_signal") or {}
    flat["partnership_score"] = part.get("partnership_score")
    flat["partnership_strength"] = part.get("partnership_strength")

    # FDA designation signal
    fda = rec.get("fda_designation_signal") or {}
    flat["fda_designation_score"] = fda.get("designation_score")
    desigs = fda.get("designations") or []
    flat["fda_designations"] = "; ".join(desigs) if desigs else ""

    # Competitive intensity signal
    ci = rec.get("competitive_intensity_signal") or {}
    flat["competitive_intensity_score"] = ci.get("intensity_score")
    flat["competitive_intensity_level"] = ci.get("intensity_level")

    # Pipeline diversity signal
    pd = rec.get("pipeline_diversity_signal") or {}
    flat["pipeline_diversity_score"] = pd.get("diversity_score")
    flat["pipeline_diversity_status"] = pd.get("diversity_status")

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

    # Get all columns (core + signal + confidence)
    columns = CORE_COLUMNS + SIGNAL_COLUMNS + CONFIDENCE_COLUMNS

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
