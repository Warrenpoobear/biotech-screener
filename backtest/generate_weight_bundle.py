"""
Generate a schema_version=2 weight bundle JSON from walk-forward OOS outputs.

Reads the per-date weights CSV from walkforward_oos_m3 and produces a bundle
containing global weights + per-regime weights, suitable for production use
with --module5-weights-mode blended.

Usage:
    python3 backtest/generate_weight_bundle.py \
        --weights-csv output/backtest_walkforward_m3_3regime_tuned/oos_weights_by_date.csv.gz \
        --blend-k 25 --min-fit-weeks 5 \
        --output data/module5_weights_m3_bundle.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def generate_bundle(
    weights_csv: str,
    blend_k: int = 25,
    min_fit_weeks: int = 5,
    output: str = "data/module5_weights_m3_bundle.json",
) -> dict:
    """Generate a schema v2 weight bundle from walkforward weight outputs."""
    wt = pd.read_csv(weights_csv)

    # Identify feature columns (everything except eval_date and fit_scope)
    meta_cols = {"eval_date", "fit_scope"}
    feature_cols = [c for c in wt.columns if c not in meta_cols]

    # Global weights: average of all GLOBAL rows (these are the most recent ridge fits)
    global_rows = wt[wt["fit_scope"] == "GLOBAL"]
    if len(global_rows) == 0:
        raise ValueError("No GLOBAL rows found in weights CSV")

    # Use the LAST global weight vector (most recent, uses most data)
    last_global = global_rows.sort_values("eval_date").iloc[-1]
    global_fw = {col: float(last_global[col]) for col in feature_cols}

    # Per-regime weights: use last weight vector per regime
    by_regime = {}
    regime_scopes = [s for s in wt["fit_scope"].unique() if s.startswith("REGIME_")]
    for scope in regime_scopes:
        regime_name = scope.replace("REGIME_", "")
        regime_rows = wt[wt["fit_scope"] == scope].sort_values("eval_date")
        n_weeks = len(regime_rows)
        last_regime = regime_rows.iloc[-1]
        regime_fw = {col: float(last_regime[col]) for col in feature_cols}
        by_regime[regime_name] = {
            "n_weeks": n_weeks,
            "feature_weights": regime_fw,
        }

    bundle = {
        "schema_version": 2,
        "blend": {
            "k": blend_k,
            "min_fit_weeks": min_fit_weeks,
        },
        "global": {
            "feature_weights": global_fw,
            "n_weeks": len(global_rows),
        },
        "by_regime": by_regime,
    }

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2, sort_keys=False)

    print(f"Written bundle to {output}")
    print(f"  Global weights: {global_fw}")
    print(f"  Regimes: {sorted(by_regime.keys())}")
    for r, data in sorted(by_regime.items()):
        print(f"    {r}: n_weeks={data['n_weeks']}, weights={data['feature_weights']}")

    return bundle


def main():
    parser = argparse.ArgumentParser(description="Generate schema v2 weight bundle")
    parser.add_argument("--weights-csv", required=True,
                        help="Path to oos_weights_by_date.csv.gz from walkforward")
    parser.add_argument("--blend-k", type=int, default=25)
    parser.add_argument("--min-fit-weeks", type=int, default=5)
    parser.add_argument("--output", default="data/module5_weights_m3_bundle.json")
    args = parser.parse_args()

    generate_bundle(
        weights_csv=args.weights_csv,
        blend_k=args.blend_k,
        min_fit_weeks=args.min_fit_weeks,
        output=args.output,
    )


if __name__ == "__main__":
    main()
