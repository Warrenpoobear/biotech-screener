#!/usr/bin/env python3
"""Three-way coinvest mode comparison: default vs off vs contra.

RESEARCH ONLY — not for production use.

Inputs:  snapshot-root, price-csv
Outputs: coinvest_diagnosis.md (side-by-side IC/L-S/excess tables + recommendation)

Runs eval_forward_returns.evaluate() three times and writes a
coinvest_diagnosis.md with side-by-side metrics and a recommendation.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval_forward_returns import (
    DEFAULT_COST_BPS,
    DEFAULT_MIN_PRICE_COVERAGE,
    DEFAULT_TOP_K,
    EvalSummary,
    evaluate,
)

MODES = ["default", "off", "contra"]


def _fmt(v: Optional[float]) -> str:
    return f"{v:.4f}" if v is not None else "\u2014"


def _fmt_pct(v: Optional[float]) -> str:
    return f"{v:.2%}" if v is not None else "\u2014"


def run_comparison(
    snapshot_root: Path,
    price_csv: Path,
    horizons: List[int],
    top_k: int,
    cost_bps: float,
    out_dir: Path,
    *,
    universe_mode: str = "current",
    anchor_mode: str = "exact",
    benchmark: str = "none",
    min_price_coverage: float = DEFAULT_MIN_PRICE_COVERAGE,
    component_eval: bool = False,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> Dict[str, EvalSummary]:
    """Run evaluate() for each coinvest mode and return summaries."""
    results: Dict[str, EvalSummary] = {}

    for mode in MODES:
        mode_dir = out_dir / mode if component_eval else None
        if mode_dir:
            mode_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== Running mode: {mode} ===")
        summary, _, _ = evaluate(
            snapshot_root=snapshot_root,
            price_csv=price_csv,
            horizons=horizons,
            top_k=top_k,
            cost_bps=cost_bps,
            anchor_mode=anchor_mode,
            benchmark=benchmark,
            long_short_deciles=True,
            universe_mode=universe_mode,
            min_price_coverage=min_price_coverage,
            component_eval=component_eval,
            out_dir=mode_dir,
            coinvest_eval_mode=mode,
            date_from=date_from,
            date_to=date_to,
        )
        results[mode] = summary

    return results


def write_diagnosis(
    results: Dict[str, EvalSummary],
    horizons: List[int],
    out_dir: Path,
) -> Path:
    """Write coinvest_diagnosis.md with comparison and recommendation."""
    lines = [
        "# Coinvest Signal Diagnosis",
        "",
        "Three-way comparison: **default** (production), **off** (coinvest_adj=0), "
        "**contra** (flipped coinvest sign).",
        "",
    ]

    # Side-by-side table per horizon
    for h in horizons:
        lines.append(f"## Horizon {h}d")
        lines.append("")
        lines.append(
            "| Metric | default | off | contra |"
        )
        lines.append(
            "|--------|---------|-----|--------|"
        )

        metrics = [
            ("Mean IC", "mean_ic", _fmt),
            ("Median IC", "median_ic", _fmt),
            ("IC t-stat", "ic_t_stat", _fmt),
            ("Mean Gross", "mean_gross_return", _fmt_pct),
            ("Cum Gross", "cumulative_gross", _fmt_pct),
            ("Mean Excess", "mean_excess_return", _fmt_pct),
            ("Cum Excess", "cumulative_excess", _fmt_pct),
            ("Mean L/S", "mean_ls_return", _fmt_pct),
            ("Cum L/S", "cumulative_ls", _fmt_pct),
            ("Mean Net", "mean_net_return", _fmt_pct),
            ("Cum Net", "cumulative_net", _fmt_pct),
        ]

        for label, key, formatter in metrics:
            vals = []
            for mode in MODES:
                bh = results[mode].by_horizon.get(h, {})
                vals.append(formatter(bh.get(key)))
            lines.append(f"| {label} | {vals[0]} | {vals[1]} | {vals[2]} |")

        lines.append("")

    # Recommendation logic
    lines.append("## Recommendation")
    lines.append("")

    off_better_count = 0
    contra_better_count = 0
    total_horizons = len(horizons)

    for h in horizons:
        def _get_ic(mode: str) -> Optional[float]:
            return results[mode].by_horizon.get(h, {}).get("mean_ic")

        ic_default = _get_ic("default")
        ic_off = _get_ic("off")
        ic_contra = _get_ic("contra")

        if ic_default is not None and ic_off is not None:
            if ic_off > ic_default:
                off_better_count += 1
        if ic_off is not None and ic_contra is not None:
            if ic_contra > ic_off:
                contra_better_count += 1

    if off_better_count == total_horizons:
        if contra_better_count > total_horizons // 2:
            rec = "INVERT"
            explanation = (
                "Coinvest signal is consistently anti-predictive. "
                "Inverting (contra mode) outperforms disabling at majority of horizons. "
                "Recommend flipping coinvest_positive_only to False or using contra sort tilt."
            )
        else:
            rec = "DISABLE"
            explanation = (
                "Coinvest signal is consistently value-destructive. "
                "Removing it (off mode) improves IC at all horizons. "
                "Recommend setting coinvest_sort_weight=0 in the active ruleset."
            )
    elif off_better_count > total_horizons // 2:
        rec = "GATE"
        explanation = (
            "Coinvest signal is harmful at majority but not all horizons. "
            "Recommend disabling for now and re-evaluating with more data."
        )
    else:
        rec = "KEEP"
        explanation = (
            "Coinvest signal is not consistently harmful. "
            "Current configuration appears reasonable. Monitor via component attribution."
        )

    lines.append(f"**{rec}**: {explanation}")
    lines.append("")

    path = out_dir / "coinvest_diagnosis.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Three-way coinvest mode comparison",
    )
    parser.add_argument(
        "--snapshot-root", type=Path,
        default=PROJECT_ROOT / "data" / "snapshots_reranked",
    )
    parser.add_argument(
        "--price-csv", type=Path,
        default=PROJECT_ROOT / "production_data" / "price_history.csv",
    )
    parser.add_argument("--horizons", type=str, default="5,20,63")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--cost-bps", type=float, default=DEFAULT_COST_BPS)
    parser.add_argument(
        "--out-dir", type=Path,
        default=PROJECT_ROOT / "output" / "coinvest_diagnosis",
    )
    parser.add_argument(
        "--universe-mode",
        choices=["current", "price_available"],
        default="price_available",
    )
    parser.add_argument(
        "--anchor-mode",
        choices=["exact", "next_trading_day", "prev_trading_day"],
        default="exact",
    )
    parser.add_argument("--benchmark", default="none")
    parser.add_argument(
        "--min-price-coverage", type=float, default=DEFAULT_MIN_PRICE_COVERAGE,
    )
    parser.add_argument("--component-eval", action="store_true", default=False)
    parser.add_argument("--date-from", type=str, default=None)
    parser.add_argument("--date-to", type=str, default=None)
    args = parser.parse_args()

    horizons = [int(h.strip()) for h in args.horizons.split(",")]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    results = run_comparison(
        snapshot_root=args.snapshot_root,
        price_csv=args.price_csv,
        horizons=horizons,
        top_k=args.top_k,
        cost_bps=args.cost_bps,
        out_dir=args.out_dir,
        universe_mode=args.universe_mode,
        anchor_mode=args.anchor_mode,
        benchmark=args.benchmark,
        min_price_coverage=args.min_price_coverage,
        component_eval=args.component_eval,
        date_from=args.date_from,
        date_to=args.date_to,
    )

    path = write_diagnosis(results, horizons, args.out_dir)
    print(f"\nDiagnosis written to {path}")

    # Print summary
    for h in horizons:
        print(f"\n  {h}d:")
        for mode in MODES:
            bh = results[mode].by_horizon.get(h, {})
            ic = bh.get("mean_ic")
            ls = bh.get("mean_ls_return")
            print(f"    {mode:8s}: IC={_fmt(ic)}, L/S={_fmt_pct(ls)}")


if __name__ == "__main__":
    main()
