#!/usr/bin/env python3
"""Alpha modifier sweep: compare (mode, weight) combos on a PIT snapshot grid.

RESEARCH ONLY — not for production use.

For each (mode, weight) pair, re-ranks snapshots with a modified ruleset and
evaluates IC, top-K returns, and turnover vs the production baseline.

Examples:
    python scripts/research/sweep_alpha_modifier.py \
        --date-from 2026-01-15 --date-to 2026-02-20 \
        --horizons 5,20

    python scripts/research/sweep_alpha_modifier.py \
        --date-from 2026-01-15 --date-to 2026-02-20 \
        --modes within_tier --weights 0.02,0.05,0.10,0.15
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import statistics
import sys
import time
from dataclasses import replace as dc_replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from common.ranking_utils import backfill_columns, safe_float
from decision_engine import DecisionRuleset

# Reuse building blocks from the alpha experiment runner
from scripts.research.run_alpha_experiment import (
    compute_turnover,
    discover_snapshot_dates,
    evaluate_single_date,
    filter_date_grid,
    hydrate_missing_exposures,
    load_price_series,
    load_rankings,
    rerank_faithful,
    spearman_ic,
    top_k_portfolio_return,
)


# ---------------------------------------------------------------------------
# Sweep logic
# ---------------------------------------------------------------------------

def sweep_alpha_modifier(
    snapshot_root: Path,
    price_csv: Path,
    ruleset: DecisionRuleset,
    modes: List[str],
    weights: List[float],
    horizons: List[int],
    top_k: int,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    date_grid: str = "all",
    min_cols: int = 50,
) -> List[Dict[str, Any]]:
    """Run sweep and return list of result dicts."""

    snap_dates = discover_snapshot_dates(snapshot_root, date_from, date_to)
    snap_dates = filter_date_grid(snap_dates, date_grid)
    print(f"Dates: {len(snap_dates)}")

    # Load prices
    print("Loading prices ...")
    prices = load_price_series(price_csv)
    all_dates_set: set = set()
    for tp in prices.values():
        all_dates_set.update(tp.keys())
    sorted_dates = sorted(all_dates_set)
    print(f"  {len(prices)} tickers, {len(sorted_dates)} trading dates")

    # Build combos: baseline + each (mode, weight)
    combos: List[Tuple[str, str, float, DecisionRuleset]] = []
    combos.append(("baseline", "off", 0.0, ruleset))
    for m in modes:
        for w in weights:
            label = f"{m}_w{w:.2f}"
            rs = dc_replace(ruleset, alpha_modifier_mode=m, alpha_modifier_weight=w)
            combos.append((label, m, w, rs))

    # Pre-load and hydrate all snapshots once
    print("Loading snapshots ...")
    snapshot_data: List[Tuple[str, List[Dict[str, str]]]] = []
    for snap_date in snap_dates:
        snap_dir = snapshot_root / snap_date
        raw_rows = load_rankings(snap_dir)
        if not raw_rows or len(raw_rows[0]) < min_cols:
            continue
        hydrate_missing_exposures(raw_rows, prices, snap_date, sorted_dates)
        snapshot_data.append((snap_date, raw_rows))
    print(f"  {len(snapshot_data)} usable snapshots")

    # Run each combo
    results: List[Dict[str, Any]] = []

    for label, mode, weight, rs in combos:
        print(f"\n--- {label} ---")
        ic_by_h: Dict[int, List[float]] = {h: [] for h in horizons}
        ret_by_h: Dict[int, List[Optional[float]]] = {h: [] for h in horizons}
        prev_topk: List[str] = []
        turnovers: List[float] = []

        for snap_date, raw_rows in snapshot_data:
            rows = copy.deepcopy(raw_rows)
            backfill_columns(rows)

            # Re-rank with this combo's ruleset
            rows = rerank_faithful(rows, rs)

            metrics = evaluate_single_date(
                snap_date, rows, prices, sorted_dates,
                horizons, top_k, [],
            )

            if metrics.skipped:
                continue

            for h in horizons:
                ic = metrics.ics.get(h)
                if ic is not None:
                    ic_by_h[h].append(ic)
                ret = metrics.gross_returns.get(h)
                if ret is not None:
                    ret_by_h[h].append(ret)

            # Turnover
            curr_topk = metrics.top_k_tickers[:top_k]
            if prev_topk:
                t = compute_turnover(prev_topk, curr_topk)
                turnovers.append(t)
            prev_topk = curr_topk

        # Aggregate
        row: Dict[str, Any] = {
            "label": label,
            "mode": mode,
            "weight": weight,
        }
        for h in horizons:
            ics = ic_by_h[h]
            row[f"ic_{h}d"] = statistics.mean(ics) if ics else None
            row[f"hit_{h}d"] = (
                sum(1 for ic in ics if ic > 0) / len(ics) if ics else None
            )
            rets = [r for r in ret_by_h[h] if r is not None]
            row[f"ret_{h}d"] = statistics.mean(rets) if rets else None
            row[f"n_{h}d"] = len(ics)
        row["mean_turnover"] = statistics.mean(turnovers) if turnovers else None

        results.append(row)

        # Print summary
        for h in horizons:
            ic = row.get(f"ic_{h}d")
            hit = row.get(f"hit_{h}d")
            ic_s = f"{ic:.4f}" if ic is not None else "N/A"
            hit_s = f"{hit:.0%}" if hit is not None else "N/A"
            print(f"  {h:>3d}d: IC={ic_s}, hit={hit_s}")
        tv = row.get("mean_turnover")
        print(f"  Turnover: {tv:.3f}" if tv is not None else "  Turnover: N/A")

    return results


def format_results(results: List[Dict[str, Any]], horizons: List[int]) -> str:
    """Format results as a markdown table."""
    lines: List[str] = []
    lines.append("# Alpha Modifier Sweep Results")
    lines.append("")

    # Header
    cols = ["Label", "Mode", "Weight"]
    for h in horizons:
        cols.extend([f"IC({h}d)", f"Hit({h}d)", f"Ret({h}d)"])
    cols.append("Turnover")
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")

    # Baseline row
    base = results[0]

    for r in results:
        vals = [r["label"], r["mode"], f"{r['weight']:.2f}"]
        for h in horizons:
            ic = r.get(f"ic_{h}d")
            hit = r.get(f"hit_{h}d")
            ret = r.get(f"ret_{h}d")

            # Delta from baseline
            b_ic = base.get(f"ic_{h}d")
            if ic is not None and b_ic is not None:
                delta = ic - b_ic
                ic_s = f"{ic:.4f} ({delta:+.4f})"
            elif ic is not None:
                ic_s = f"{ic:.4f}"
            else:
                ic_s = "—"

            hit_s = f"{hit:.0%}" if hit is not None else "—"
            ret_s = f"{ret:+.4%}" if ret is not None else "—"
            vals.extend([ic_s, hit_s, ret_s])

        tv = r.get("mean_turnover")
        vals.append(f"{tv:.3f}" if tv is not None else "—")
        lines.append("| " + " | ".join(vals) + " |")

    lines.append("")

    # Recommendation
    lines.append("## Recommendation")
    lines.append("")
    # Find best non-baseline IC improvement that doesn't spike turnover
    best = None
    base_ic = base.get("ic_20d") or base.get("ic_5d")
    base_hit = base.get("hit_20d") or base.get("hit_5d")
    base_tv = base.get("mean_turnover", 0) or 0
    ref_h = 20 if base.get("ic_20d") is not None else 5

    for r in results[1:]:
        ic = r.get(f"ic_{ref_h}d")
        hit = r.get(f"hit_{ref_h}d")
        tv = r.get("mean_turnover", 0) or 0

        if ic is None or base_ic is None:
            continue
        if ic <= base_ic:
            continue
        if hit is not None and base_hit is not None and hit < base_hit:
            continue
        if base_tv > 0 and tv > base_tv * 1.20:
            continue  # turnover spike > 20%

        if best is None or r["weight"] < best["weight"]:
            best = r

    if best:
        ic_d = best.get(f"ic_{ref_h}d", 0) - (base_ic or 0)
        lines.append(
            f"**PROMOTE `{best['label']}`**: "
            f"ΔIC({ref_h}d) = {ic_d:+.4f}, "
            f"hit = {best.get(f'hit_{ref_h}d', 0):.0%}, "
            f"turnover = {best.get('mean_turnover', 0):.3f}"
        )
    else:
        lines.append("**KEEP BASELINE**: No alpha modifier setting improves IC + hit-rate without turnover spike.")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Alpha modifier sweep",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--snapshot-root", type=Path,
                        default=PROJECT_ROOT / "data" / "snapshots")
    parser.add_argument("--price-csv", type=Path,
                        default=PROJECT_ROOT / "production_data" / "price_history.csv")
    parser.add_argument("--ruleset", type=Path, default=None)
    parser.add_argument("--modes", type=str, default="tiebreak,within_tier",
                        help="Alpha modifier modes to sweep")
    parser.add_argument("--weights", type=str, default="0.02,0.05,0.10,0.15",
                        help="Alpha modifier weights to sweep")
    parser.add_argument("--horizons", type=str, default="5,20")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--date-from", type=str, default=None)
    parser.add_argument("--date-to", type=str, default=None)
    parser.add_argument("--date-grid", choices=["all", "monthly"], default="all")
    parser.add_argument("--min-cols", type=int, default=50)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    modes = [m.strip() for m in args.modes.split(",")]
    weights = [float(w.strip()) for w in args.weights.split(",")]
    horizons = [int(h.strip()) for h in args.horizons.split(",")]

    # Load ruleset
    if args.ruleset:
        ruleset = DecisionRuleset.from_json(str(args.ruleset))
    else:
        dates = sorted([
            p.name for p in args.snapshot_root.iterdir()
            if p.is_dir() and len(p.name) == 10 and p.name[4] == "-"
            and (p / "decision_ruleset.json").exists()
        ])
        if not dates:
            print("ERROR: No snapshot with decision_ruleset.json found")
            sys.exit(1)
        rs_path = args.snapshot_root / dates[-1] / "decision_ruleset.json"
        ruleset = DecisionRuleset.from_json(str(rs_path))
    print(f"Ruleset: {ruleset.ruleset_id}")
    print(f"Current alpha_modifier: mode={ruleset.alpha_modifier_mode}, "
          f"weight={ruleset.alpha_modifier_weight}")

    t0 = time.time()
    results = sweep_alpha_modifier(
        snapshot_root=args.snapshot_root,
        price_csv=args.price_csv,
        ruleset=ruleset,
        modes=modes,
        weights=weights,
        horizons=horizons,
        top_k=args.top_k,
        date_from=args.date_from,
        date_to=args.date_to,
        date_grid=args.date_grid,
        min_cols=args.min_cols,
    )
    elapsed = time.time() - t0
    print(f"\nElapsed: {elapsed:.1f}s")

    # Output
    out_dir = args.out or (PROJECT_ROOT / "output" / "alpha_modifier_sweep")
    out_dir.mkdir(parents=True, exist_ok=True)

    report = format_results(results, horizons)
    report_path = out_dir / "sweep_results.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nReport: {report_path}")

    with open(out_dir / "sweep_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Print formatted table
    print("\n" + report)


if __name__ == "__main__":
    main()
