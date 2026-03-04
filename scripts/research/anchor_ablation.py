#!/usr/bin/env python3
"""Anchor ablation: compare sort-anchor strategies on PIT snapshot grid.

RESEARCH ONLY — not for production use.

Tests whether alpha_cohort works better as a dominant anchor vs a modifier
on top of legacy fundamentals.

Configs:
  A: anchor=alpha_cohort (current production)
  B: anchor=legacy_composite (composite_score_attn), no alpha modifier
  C: anchor=legacy_composite, alpha as within_tier nudge
  D: anchor=legacy_composite, alpha as within_tier stronger nudge
  E: anchor=optionality, no alpha modifier
  F: anchor=optionality, alpha as within_tier nudge
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from common.ranking_utils import backfill_columns, safe_float
from decision_engine import DecisionRuleset, compute_actionable_sort_key

from scripts.research.run_alpha_experiment import (
    compute_turnover,
    discover_snapshot_dates,
    evaluate_single_date,
    filter_date_grid,
    hydrate_missing_exposures,
    load_price_series,
    load_rankings,
    spearman_ic,
    top_k_portfolio_return,
)


# ---------------------------------------------------------------------------
# Re-rank with custom anchor column
# ---------------------------------------------------------------------------

def rerank_with_anchor(
    rows: List[Dict[str, str]],
    ruleset: DecisionRuleset,
    anchor_col: str,
) -> List[Dict[str, str]]:
    """Re-sort rows using a specified column as the anchor percentile.

    For configs that use legacy composite or optionality as anchor, we
    inject the specified column's value into 'alpha_cohort_pct' (which
    the sort key reads as tiebreaker_pct when sort_anchor=alpha_cohort).
    This avoids modifying the production sort key.
    """
    backfill_columns(rows)

    # Pre-compute: inject anchor_col value as the tiebreaker_pct source
    if anchor_col != "alpha_cohort_pct":
        for r in rows:
            r["_ablation_anchor_pct"] = r.get(anchor_col, "0")
    else:
        for r in rows:
            r["_ablation_anchor_pct"] = r.get("alpha_cohort_pct", "0")

    rows.sort(key=lambda r: compute_actionable_sort_key(
        decision_fields=r,
        archetype=r.get("archetype", ""),
        optionality=safe_float(r.get("clinical_optionality_pct_dev")),
        composite_rank=r.get("composite_rank"),
        ticker=r.get("ticker", ""),
        catalyst_event_type=r.get("catalyst_event_type", ""),
        catalyst_source=r.get("catalyst_source", ""),
        ruleset=ruleset,
        tiebreaker_pct=safe_float(r.get("_ablation_anchor_pct")),
        alpha_raw=safe_float(r.get("alpha_cohort_raw")),
    ))

    rank = 1
    for r in rows:
        if r.get("eligible") == "1":
            r["actionable_rank"] = str(rank)
            rank += 1
        else:
            r["actionable_rank"] = ""

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Anchor ablation study")
    parser.add_argument("--snapshot-root", type=Path,
                        default=PROJECT_ROOT / "data" / "snapshots")
    parser.add_argument("--price-csv", type=Path,
                        default=PROJECT_ROOT / "production_data" / "price_history.csv")
    parser.add_argument("--date-from", type=str, default=None)
    parser.add_argument("--date-to", type=str, default=None)
    parser.add_argument("--horizons", type=str, default="5,20")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-cols", type=int, default=50)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    horizons = [int(h.strip()) for h in args.horizons.split(",")]

    # Discover dates
    snap_dates = discover_snapshot_dates(args.snapshot_root, args.date_from, args.date_to)
    print(f"Dates: {len(snap_dates)}")

    # Load prices
    print("Loading prices ...")
    prices = load_price_series(args.price_csv)
    all_dates_set: set = set()
    for tp in prices.values():
        all_dates_set.update(tp.keys())
    sorted_dates = sorted(all_dates_set)
    print(f"  {len(prices)} tickers, {len(sorted_dates)} trading dates")

    # Load base ruleset from latest snapshot
    dates_with_rs = sorted([
        p.name for p in args.snapshot_root.iterdir()
        if p.is_dir() and len(p.name) == 10 and p.name[4] == "-"
        and (p / "decision_ruleset.json").exists()
    ])
    rs_path = args.snapshot_root / dates_with_rs[-1] / "decision_ruleset.json"
    base_rs = DecisionRuleset.from_json(str(rs_path))
    print(f"Base ruleset: {base_rs.ruleset_id}")

    # Define ablation configs
    # Each: (label, sort_anchor, alpha_modifier_mode, alpha_modifier_weight, anchor_col)
    from dataclasses import replace as dc_replace

    configs: List[Tuple[str, DecisionRuleset, str]] = []

    # A: current production — alpha_cohort anchor
    rs_a = dc_replace(base_rs,
                       sort_anchor="alpha_cohort",
                       alpha_modifier_mode="within_tier",
                       alpha_modifier_weight=0.05)
    configs.append(("A_alpha_anchor", rs_a, "alpha_cohort_pct"))

    # B: legacy composite anchor, no alpha modifier
    rs_b = dc_replace(base_rs,
                       sort_anchor="alpha_cohort",  # trick: use alpha_cohort mode but inject legacy pct
                       alpha_modifier_mode="off",
                       alpha_modifier_weight=0.0)
    configs.append(("B_legacy_anchor", rs_b, "score_rank_pct_attn"))

    # C: legacy composite + alpha within_tier w=0.05
    rs_c = dc_replace(base_rs,
                       sort_anchor="alpha_cohort",
                       alpha_modifier_mode="within_tier",
                       alpha_modifier_weight=0.05)
    configs.append(("C_legacy+alpha_0.05", rs_c, "score_rank_pct_attn"))

    # D: legacy composite + alpha within_tier w=0.15
    rs_d = dc_replace(base_rs,
                       sort_anchor="alpha_cohort",
                       alpha_modifier_mode="within_tier",
                       alpha_modifier_weight=0.15)
    configs.append(("D_legacy+alpha_0.15", rs_d, "score_rank_pct_attn"))

    # E: optionality anchor, no alpha
    rs_e = dc_replace(base_rs,
                       sort_anchor="alpha_cohort",
                       alpha_modifier_mode="off",
                       alpha_modifier_weight=0.0)
    configs.append(("E_optionality_anchor", rs_e, "clinical_optionality_pct_dev"))

    # F: optionality + alpha w=0.05
    rs_f = dc_replace(base_rs,
                       sort_anchor="alpha_cohort",
                       alpha_modifier_mode="within_tier",
                       alpha_modifier_weight=0.05)
    configs.append(("F_optionality+alpha_0.05", rs_f, "clinical_optionality_pct_dev"))

    # Pre-load snapshots
    print("Loading snapshots ...")
    snapshot_data: List[Tuple[str, List[Dict[str, str]]]] = []
    for snap_date in snap_dates:
        snap_dir = args.snapshot_root / snap_date
        raw_rows = load_rankings(snap_dir)
        if not raw_rows or len(raw_rows[0]) < args.min_cols:
            continue
        hydrate_missing_exposures(raw_rows, prices, snap_date, sorted_dates)
        snapshot_data.append((snap_date, raw_rows))
    print(f"  {len(snapshot_data)} usable snapshots")

    # Run each config
    results: List[Dict[str, Any]] = []

    for label, rs, anchor_col in configs:
        print(f"\n--- {label} (anchor={anchor_col}) ---")
        ic_by_h: Dict[int, List[float]] = {h: [] for h in horizons}
        ret_by_h: Dict[int, List[Optional[float]]] = {h: [] for h in horizons}
        prev_topk: List[str] = []
        turnovers: List[float] = []

        for snap_date, raw_rows in snapshot_data:
            rows = copy.deepcopy(raw_rows)

            # Re-rank with this config's anchor
            rows = rerank_with_anchor(rows, rs, anchor_col)

            metrics = evaluate_single_date(
                snap_date, rows, prices, sorted_dates,
                horizons, args.top_k, [],
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

            curr_topk = metrics.top_k_tickers[:args.top_k]
            if prev_topk:
                t = compute_turnover(prev_topk, curr_topk)
                turnovers.append(t)
            prev_topk = curr_topk

        # Aggregate
        row: Dict[str, Any] = {"label": label, "anchor_col": anchor_col}
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

        for h in horizons:
            ic = row.get(f"ic_{h}d")
            hit = row.get(f"hit_{h}d")
            ic_s = f"{ic:.4f}" if ic is not None else "N/A"
            hit_s = f"{hit:.0%}" if hit is not None else "N/A"
            print(f"  {h:>3d}d: IC={ic_s}, hit={hit_s}")
        tv = row.get("mean_turnover")
        print(f"  Turnover: {tv:.3f}" if tv is not None else "  Turnover: N/A")

    # Format results
    print("\n" + "=" * 100)
    print("# Anchor Ablation Results")
    print()

    # Find reference (A)
    ref = results[0]

    header = f"{'Label':<28s} {'Anchor':<32s}"
    for h in horizons:
        header += f" {'IC('+str(h)+'d)':>12s} {'ΔIC':>8s} {'Hit':>6s}"
    header += f" {'Turnover':>9s}"
    print(header)
    print("-" * len(header))

    for r in results:
        line = f"{r['label']:<28s} {r['anchor_col']:<32s}"
        for h in horizons:
            ic = r.get(f"ic_{h}d")
            ref_ic = ref.get(f"ic_{h}d")
            hit = r.get(f"hit_{h}d")
            ic_s = f"{ic:.4f}" if ic is not None else "—"
            if ic is not None and ref_ic is not None and r != ref:
                delta = ic - ref_ic
                d_s = f"{delta:+.4f}"
            else:
                d_s = "—"
            hit_s = f"{hit:.0%}" if hit is not None else "—"
            line += f" {ic_s:>12s} {d_s:>8s} {hit_s:>6s}"
        tv = r.get("mean_turnover")
        line += f" {tv:>9.3f}" if tv is not None else f" {'—':>9s}"
        print(line)

    # Write JSON
    out_dir = args.out or (PROJECT_ROOT / "output" / "anchor_ablation")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nJSON: {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
