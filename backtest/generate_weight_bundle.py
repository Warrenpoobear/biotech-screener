"""
Generate a schema_version=2 weight bundle JSON from walk-forward OOS outputs.

Reads the per-date weights CSV from walkforward_oos_m3 and produces a bundle
containing global weights + per-regime weights, suitable for production use
with --module5-weights-mode blended.

Global weights come from the production-scale calibrated JSON (e.g.,
data/module5_weights_m3.json) so the composite score scale is preserved.
Regime weights from walkforward are rescaled to match the global L1 norm,
preserving regime *tilts* without changing the score magnitude.

Usage:
    python3 backtest/generate_weight_bundle.py \
        --weights-csv output/backtest_walkforward_m3_3regime_tuned/oos_weights_by_date.csv.gz \
        --global-weights data/module5_weights_m3.json \
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


def _rescale_to_l1_norm(
    weights: dict[str, float],
    target_l1: float,
) -> dict[str, float]:
    """Rescale weight vector so sum(|w|) == target_l1, preserving sign ratios."""
    current_l1 = sum(abs(v) for v in weights.values())
    if current_l1 < 1e-15:
        return weights
    scale = target_l1 / current_l1
    return {k: v * scale for k, v in weights.items()}


def generate_bundle(
    weights_csv: str,
    global_weights_json: str,
    blend_k: int = 25,
    min_fit_weeks: int = 5,
    output: str = "data/module5_weights_m3_bundle.json",
) -> dict:
    """Generate a schema v2 weight bundle from walkforward weight outputs.

    Parameters
    ----------
    weights_csv : path to oos_weights_by_date.csv.gz from walkforward
    global_weights_json : path to production-scale calibrated JSON
        (e.g., data/module5_weights_m3.json) used for the global block
        and as the L1 norm reference for rescaling regime weights
    """
    # Load production-scale global weights
    with open(global_weights_json, "r", encoding="utf-8") as f:
        prod_data = json.load(f)
    prod_fw = prod_data.get("feature_weights", {})
    if not prod_fw:
        raise ValueError(f"No feature_weights in {global_weights_json}")

    global_l1 = sum(abs(v) for v in prod_fw.values())
    print(f"Global weights from {global_weights_json}:")
    print(f"  {prod_fw}")
    print(f"  L1 norm: {global_l1:.6f}")

    # Load walkforward regime weights
    wt = pd.read_csv(weights_csv)
    meta_cols = {"eval_date", "fit_scope"}
    feature_cols = [c for c in wt.columns if c not in meta_cols]

    # Per-regime weights: use last weight vector per regime, rescaled
    by_regime = {}
    regime_scopes = [s for s in wt["fit_scope"].unique() if s.startswith("REGIME_")]
    for scope in sorted(regime_scopes):
        regime_name = scope.replace("REGIME_", "")
        regime_rows = wt[wt["fit_scope"] == scope].sort_values("eval_date")
        n_weeks = len(regime_rows)
        last_regime = regime_rows.iloc[-1]
        raw_fw = {col: float(last_regime[col]) for col in feature_cols}
        rescaled_fw = _rescale_to_l1_norm(raw_fw, global_l1)
        by_regime[regime_name] = {
            "n_weeks": n_weeks,
            "feature_weights": rescaled_fw,
        }

    bundle = {
        "schema_version": 2,
        "blend": {
            "k": blend_k,
            "min_fit_weeks": min_fit_weeks,
        },
        "global": {
            "feature_weights": prod_fw,
            "n_weeks": int(prod_data.get("n_weeks", 0)),
        },
        "by_regime": by_regime,
    }

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2, sort_keys=False)

    print(f"\nWritten bundle to {output}")
    print(f"  Global: {prod_fw}")
    print(f"  Regimes: {sorted(by_regime.keys())}")
    for r, data in sorted(by_regime.items()):
        raw_l1 = sum(abs(v) for v in wt[wt["fit_scope"] == f"REGIME_{r}"].sort_values("eval_date").iloc[-1][feature_cols])
        print(f"    {r}: n_weeks={data['n_weeks']}, "
              f"raw_L1={raw_l1:.6f} → rescaled_L1={global_l1:.6f}")
        for col in sorted(data["feature_weights"].keys()):
            print(f"      {col}: {data['feature_weights'][col]:.6f}")

    return bundle


def main():
    parser = argparse.ArgumentParser(description="Generate schema v2 weight bundle")
    parser.add_argument("--weights-csv", required=True,
                        help="Path to oos_weights_by_date.csv.gz from walkforward")
    parser.add_argument("--global-weights", required=True,
                        help="Path to production-scale calibrated JSON (e.g., data/module5_weights_m3.json)")
    parser.add_argument("--blend-k", type=int, default=25)
    parser.add_argument("--min-fit-weeks", type=int, default=5)
    parser.add_argument("--output", default="data/module5_weights_m3_bundle.json")
    args = parser.parse_args()

    generate_bundle(
        weights_csv=args.weights_csv,
        global_weights_json=args.global_weights,
        blend_k=args.blend_k,
        min_fit_weeks=args.min_fit_weeks,
        output=args.output,
    )


if __name__ == "__main__":
    main()
