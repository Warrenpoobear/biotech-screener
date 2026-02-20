#!/usr/bin/env python3
"""Sweep institutional sort weight via replay comparison.

Iterates over candidate weights, patches the baseline ruleset with
``enable_institutional_sort_signal=True`` and the test weight, then
calls ``compare()`` from compare_rulesets_replay to measure top-20/60
overlap and rank churn.

Usage:
    python scripts/sweep_institutional_weight.py \
        --snapshot-dir data/snapshots/2026-02-20 \
        --baseline production_data/decision_rulesets/v1.5.0_coinvest_candidate.json \
        --weights "0.01,0.03,0.05,0.10,0.15,0.20,0.30,0.50"
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import List, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from decision_engine import DecisionRuleset
from scripts.compare_rulesets_replay import compare

DEFAULT_WEIGHTS = [0.01, 0.03, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]


def _load_and_patch(
    baseline_path: str,
    overrides: dict,
) -> DecisionRuleset:
    """Load a ruleset from JSON and override specific fields."""
    base = DecisionRuleset.from_json(baseline_path)
    return replace(base, **overrides)


def _backfill_rankings(rankings: pd.DataFrame) -> pd.DataFrame:
    """Backfill columns needed for institutional sort (cold-start safe)."""
    for col, default in [("inst_delta_z", 0.0), ("inst_delta_net", 0),
                         ("inst_delta_new", 0), ("inst_delta_exit", 0)]:
        if col not in rankings.columns:
            rankings[col] = default
    # Clinical z backfill (same logic as compare_rulesets_replay)
    if "clinical_score_z" not in rankings.columns:
        rankings["clinical_score_z"] = 0.0
    if "clinical_score_z_tier" not in rankings.columns:
        rankings["clinical_score_z_tier"] = 0.0
    return rankings


def sweep(
    snapshot_dir: str,
    baseline_path: str,
    weights: List[float],
) -> List[dict]:
    """Run comparison for each weight and return structured results."""
    snap_dir = Path(snapshot_dir)
    rankings_path = snap_dir / "rankings.csv"
    if not rankings_path.exists():
        print(f"ERROR: {rankings_path} not found", file=sys.stderr)
        return []

    rankings = pd.read_csv(rankings_path)
    rankings = _backfill_rankings(rankings)

    baseline_rs = DecisionRuleset.from_json(baseline_path)

    results = []
    for w in weights:
        candidate_rs = _load_and_patch(baseline_path, {
            "enable_institutional_sort_signal": True,
            "institutional_sort_weight": w,
            "institutional_positive_only": True,
        })

        result = compare(rankings, baseline_rs, candidate_rs)
        results.append({
            "weight": w,
            "top20_overlap": result["top20_overlap"],
            "top60_overlap": result["top60_overlap"],
            "names_changed_20": len(result["entrants_20"]),
            "names_changed_60": len(result["entrants_60"]),
            "entrants_20": result["entrants_20"],
            "exits_20": result["exits_20"],
            "mean_rank_churn_top60": result["mean_rank_churn_top60"],
            "max_rank_churn_top60": result["max_rank_churn_top60"],
            "baseline_id": result["baseline_id"],
            "candidate_id": result["candidate_id"],
        })
        print(f"  w={w:.2f}  top20={result['top20_overlap']:.1f}%  "
              f"top60={result['top60_overlap']:.1f}%  "
              f"names_chg_20={len(result['entrants_20'])}  "
              f"mean_churn={result['mean_rank_churn_top60']:.2f}")

    return results


def format_report(results: List[dict]) -> str:
    """Format sweep results as markdown table."""
    lines = [
        "# Institutional Sort Weight Sweep",
        "",
        "| Weight | Top-20 Overlap | Top-60 Overlap | Names Changed (20) | Mean Rank Churn | Max Rank Churn |",
        "|--------|---------------|----------------|-------------------|----------------|----------------|",
    ]
    for r in results:
        lines.append(
            f"| {r['weight']:.2f}   | {r['top20_overlap']:.1f}%"
            f"          | {r['top60_overlap']:.1f}%"
            f"          | {r['names_changed_20']}"
            f"                 | {r['mean_rank_churn_top60']:.2f}"
            f"           | {r['max_rank_churn_top60']}"
            f"              |"
        )

    if results:
        lines += [
            "",
            f"Baseline ruleset: `{results[0]['baseline_id']}`",
            "",
            "## Entrants/Exits Detail (Top-20)",
            "",
        ]
        for r in results:
            if r["entrants_20"] or r["exits_20"]:
                lines.append(f"**w={r['weight']:.2f}**: "
                             f"in={', '.join(r['entrants_20']) or 'none'}, "
                             f"out={', '.join(r['exits_20']) or 'none'}")

    lines.append("")
    return "\n".join(lines)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Sweep institutional sort weight")
    p.add_argument("--snapshot-dir", required=True,
                   help="Snapshot directory containing rankings.csv")
    p.add_argument("--baseline", required=True,
                   help="Path to baseline ruleset JSON")
    p.add_argument("--weights", default=None,
                   help="Comma-separated weights (default: 0.01,0.03,0.05,0.10,0.15,0.20,0.30,0.50)")
    p.add_argument("--output-dir", default="artifacts",
                   help="Output directory for reports (default: artifacts)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.weights:
        weights = [float(w.strip()) for w in args.weights.split(",")]
    else:
        weights = DEFAULT_WEIGHTS

    print(f"Sweeping {len(weights)} weights on {args.snapshot_dir}")
    print(f"Baseline: {args.baseline}")
    print()

    results = sweep(args.snapshot_dir, args.baseline, weights)
    if not results:
        return 1

    # Write outputs
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "institutional_weight_sweep.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nJSON: {json_path}")

    md_path = out_dir / "institutional_weight_sweep.md"
    report = format_report(results)
    md_path.write_text(report, encoding="utf-8")
    print(f"Report: {md_path}")
    print()
    print(report)

    return 0


if __name__ == "__main__":
    sys.exit(main())
