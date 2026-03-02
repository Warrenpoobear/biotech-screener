#!/usr/bin/env python3
"""Alpha ablation study on PIT-correct bundle snapshots.

Re-ranks each PIT snapshot with three configurations:
  1. Baseline:       production sort (alpha_cohort anchor + within_tier modifier)
  2. Alpha removed:  optionality anchor, alpha_modifier=off
  3. Alpha inverted: alpha_cohort anchor but alpha_raw and alpha_pct negated

For each, reports: IC, decile L/S spread, quintile monotonicity, turnover.

Usage:
    python scripts/research/alpha_ablation.py
    python scripts/research/alpha_ablation.py --snapshot-root data/snapshots_pit \
        --date-from 2024-01-31 --date-to 2025-12-31
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from dataclasses import replace as dc_replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from common.ranking_utils import backfill_columns, safe_float
from decision_engine import DecisionRuleset, compute_actionable_sort_key
from scripts.research.anchor_replay import (
    _avg_ranks,
    compute_forward_return,
    load_price_series,
    paired_stats,
    spearman_ic,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_snapshot_rankings(snap_dir: Path) -> List[Dict[str, str]]:
    """Load rankings.csv from a snapshot directory."""
    csv_path = snap_dir / "rankings.csv"
    if not csv_path.exists():
        return []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def discover_snapshots(
    root: Path,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> List[Tuple[str, Path]]:
    """Find snapshot date directories."""
    result = []
    for p in sorted(root.iterdir()):
        if not p.is_dir() or len(p.name) != 10 or p.name[4] != "-":
            continue
        d = p.name
        if date_from and d < date_from:
            continue
        if date_to and d > date_to:
            continue
        if (p / "rankings.csv").exists():
            result.append((d, p))
    return result


def rerank_snapshot(
    rows: List[Dict[str, str]],
    ruleset: DecisionRuleset,
    *,
    invert_alpha: bool = False,
) -> List[Dict[str, str]]:
    """Re-sort rows using compute_actionable_sort_key."""
    rows = copy.deepcopy(rows)
    backfill_columns(rows)

    def _get_tiebreaker(r: Dict[str, str]) -> float:
        if ruleset.sort_anchor == "alpha_cohort":
            v = safe_float(r.get("alpha_cohort_pct")) or 0.5
            return (1.0 - v) if invert_alpha else v
        return safe_float(r.get("clinical_optionality_pct_dev")) or 0.0

    def _get_alpha_raw(r: Dict[str, str]) -> Optional[float]:
        v = safe_float(r.get("alpha_cohort_raw"))
        if v is not None and invert_alpha:
            return -v
        return v

    rows.sort(key=lambda r: compute_actionable_sort_key(
        decision_fields=r,
        archetype=r.get("archetype", ""),
        optionality=safe_float(r.get("clinical_optionality_pct_dev")),
        composite_rank=r.get("composite_rank"),
        ticker=r.get("ticker", ""),
        catalyst_event_type=r.get("catalyst_event_type", ""),
        catalyst_source=r.get("catalyst_source", ""),
        ruleset=ruleset,
        tiebreaker_pct=_get_tiebreaker(r),
        alpha_raw=_get_alpha_raw(r),
    ))

    rank = 1
    for r in rows:
        if r.get("eligible") == "1":
            r["actionable_rank"] = str(rank)
            rank += 1
        else:
            r["actionable_rank"] = ""

    return rows


def eval_ranking(
    rows: List[Dict[str, str]],
    snap_date: str,
    prices: Dict[str, Dict[str, float]],
    sorted_dates: List[str],
    horizons: List[int],
    top_k: int,
) -> Dict[str, Any]:
    """Evaluate a ranking: IC, top-K return, decile L/S, quintile monotonicity."""
    eligible = [r for r in rows if r.get("eligible") == "1"]
    ranked = sorted(eligible, key=lambda r: int(r.get("actionable_rank") or 9999))
    tickers = [r.get("ticker", "") for r in ranked]

    # Resolve trade date (next trading day for PIT-safe execution)
    trade_date = None
    for d in sorted_dates:
        if d > snap_date:
            trade_date = d
            break
    if trade_date is None:
        return {"skipped": True}

    result: Dict[str, Any] = {
        "skipped": False,
        "n_eligible": len(eligible),
        "top_k": tickers[:top_k],
    }

    for h in horizons:
        fwd_rets: Dict[str, float] = {}
        for t in tickers:
            if t not in prices:
                continue
            ret = compute_forward_return(prices[t], sorted_dates, trade_date, h)
            if ret is not None:
                fwd_rets[t] = ret

        signal_tickers = [t for t in tickers if t in fwd_rets]
        n_sig = len(signal_tickers)

        if n_sig < 10:
            result[f"ic_{h}d"] = None
            continue

        # IC
        signal_vals = [-float(i + 1) for i, t in enumerate(tickers) if t in fwd_rets]
        return_vals = [fwd_rets[t] for t in signal_tickers]
        ic = spearman_ic(signal_vals, return_vals)
        result[f"ic_{h}d"] = ic
        result[f"n_{h}d"] = n_sig

        # Top-K return
        held = [t for t in tickers[:top_k] if t in fwd_rets]
        result[f"topk_ret_{h}d"] = statistics.mean(fwd_rets[t] for t in held) if held else None

        # Bottom-K return
        bottom = [t for t in tickers[-top_k:] if t in fwd_rets]
        result[f"botk_ret_{h}d"] = statistics.mean(fwd_rets[t] for t in bottom) if bottom else None

        # Decile L/S spread
        d_size = max(n_sig // 10, 1)
        top_d = signal_tickers[:d_size]
        bot_d = signal_tickers[-d_size:]
        top_d_ret = statistics.mean(fwd_rets[t] for t in top_d) if top_d else 0
        bot_d_ret = statistics.mean(fwd_rets[t] for t in bot_d) if bot_d else 0
        result[f"ls_{h}d"] = top_d_ret - bot_d_ret

        # Quintile monotonicity (if enough names)
        if n_sig >= 25:
            ordered_rets = [fwd_rets[t] for t in signal_tickers]
            bucket_size = n_sig // 5
            q_means = []
            for qi in range(5):
                start = qi * bucket_size
                end = start + bucket_size if qi < 4 else n_sig
                q_means.append(statistics.mean(ordered_rets[start:end]))
            result[f"spread_{h}d"] = q_means[0] - q_means[4]
            result[f"mono_{h}d"] = all(q_means[i] > q_means[i + 1] for i in range(4))
            result[f"q1_gt_q5_{h}d"] = q_means[0] > q_means[4]
            # Slope
            result[f"slope_{h}d"] = sum((i + 1 - 3) * q_means[i] for i in range(5)) / 10.0

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Alpha ablation on PIT snapshots")
    parser.add_argument("--snapshot-root", type=Path,
                        default=PROJECT_ROOT / "data" / "snapshots_pit")
    parser.add_argument("--price-csv", type=Path,
                        default=PROJECT_ROOT / "production_data" / "price_history.csv")
    parser.add_argument("--date-from", type=str, default="2020-01-01")
    parser.add_argument("--date-to", type=str, default="2026-02-28")
    parser.add_argument("--horizons", type=str, default="20,60")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    horizons = [int(h.strip()) for h in args.horizons.split(",")]

    # Discover snapshots
    snapshots = discover_snapshots(args.snapshot_root, args.date_from, args.date_to)
    print(f"Snapshots: {len(snapshots)} ({snapshots[0][0]} → {snapshots[-1][0]})")

    # Load prices
    print("Loading prices ...")
    prices = load_price_series(args.price_csv)
    all_dates_set: set = set()
    for tp in prices.values():
        all_dates_set.update(tp.keys())
    sorted_dates = sorted(all_dates_set)
    print(f"  {len(prices)} tickers, {len(sorted_dates)} trading dates")

    # Load base ruleset
    snap_dir = PROJECT_ROOT / "data" / "snapshots"
    dates_with_rs = sorted([
        p.name for p in snap_dir.iterdir()
        if p.is_dir() and len(p.name) == 10 and p.name[4] == "-"
        and (p / "decision_ruleset.json").exists()
    ]) if snap_dir.exists() else []
    if dates_with_rs:
        rs_path = snap_dir / dates_with_rs[-1] / "decision_ruleset.json"
        base_rs = DecisionRuleset.from_json(str(rs_path))
    else:
        base_rs = DecisionRuleset()
    print(f"Base ruleset: {base_rs.ruleset_id}")

    # Three configs
    configs = {
        "baseline": {
            "label": "Baseline (alpha anchor + modifier)",
            "ruleset": base_rs,
            "invert_alpha": False,
        },
        "alpha_removed": {
            "label": "Alpha removed (optionality anchor, modifier off)",
            "ruleset": dc_replace(base_rs,
                                  sort_anchor="optionality_pct",
                                  alpha_modifier_mode="off",
                                  alpha_modifier_weight=0.0),
            "invert_alpha": False,
        },
        "alpha_inverted": {
            "label": "Alpha inverted (negated pct + raw)",
            "ruleset": base_rs,
            "invert_alpha": True,
        },
    }

    print(f"\nConfigs: {list(configs.keys())}")
    print(f"Horizons: {horizons}, top-K: {args.top_k}")
    print()

    # Evaluate
    all_results: Dict[str, List[Dict[str, Any]]] = {k: [] for k in configs}
    prev_topk: Dict[str, List[str]] = {k: [] for k in configs}
    turnover_acc: Dict[str, List[float]] = {k: [] for k in configs}

    for date_str, snap_path in snapshots:
        raw_rows = load_snapshot_rankings(snap_path)
        if not raw_rows:
            print(f"  {date_str}: no rows — skip")
            continue

        for cfg_key, cfg in configs.items():
            rows = rerank_snapshot(
                raw_rows, cfg["ruleset"],
                invert_alpha=cfg["invert_alpha"],
            )
            metrics = eval_ranking(
                rows, date_str, prices, sorted_dates, horizons, args.top_k,
            )
            if metrics.get("skipped"):
                continue

            metrics["date"] = date_str
            all_results[cfg_key].append(metrics)

            # Turnover
            curr_topk = metrics.get("top_k", [])
            if prev_topk[cfg_key]:
                from scripts.research.anchor_replay import compute_turnover
                t = compute_turnover(prev_topk[cfg_key], curr_topk)
                turnover_acc[cfg_key].append(t)
            prev_topk[cfg_key] = curr_topk

        print(f"  {date_str}: {len(raw_rows)} tickers")

    # Common dates
    common_dates = set.intersection(
        *[set(m["date"] for m in v) for v in all_results.values()]
    )
    print(f"\nCommon dates with all configs: {len(common_dates)}")

    # Aggregate per config
    print("\n" + "=" * 130)
    print("# Alpha Ablation Results (PIT-correct snapshots)")
    print(f"# Dates: {len(common_dates)}, horizons: {horizons}, top_k: {args.top_k}")
    print()

    header = f"{'Config':<40s}"
    for h in horizons:
        header += (f" {'IC('+str(h)+'d)':>10s} {'Hit':>5s}"
                   f" {'TopK':>8s} {'BotK':>8s} {'L/S':>8s}"
                   f" {'Sprd':>8s} {'Mono%':>6s} {'Q1>5':>5s} {'Slope':>8s}")
    header += f" {'Turn':>6s}"
    print(header)
    print("-" * len(header))

    summary: Dict[str, Dict[str, Any]] = {}
    for cfg_key, cfg in configs.items():
        results = [m for m in all_results[cfg_key] if m["date"] in common_dates]
        row_data: Dict[str, Any] = {"label": cfg["label"]}

        line = f"{cfg['label']:<40s}"
        for h in horizons:
            ics = [m[f"ic_{h}d"] for m in results if m.get(f"ic_{h}d") is not None]
            topk_rets = [m[f"topk_ret_{h}d"] for m in results if m.get(f"topk_ret_{h}d") is not None]
            botk_rets = [m[f"botk_ret_{h}d"] for m in results if m.get(f"botk_ret_{h}d") is not None]
            ls_vals = [m[f"ls_{h}d"] for m in results if m.get(f"ls_{h}d") is not None]
            spreads = [m[f"spread_{h}d"] for m in results if m.get(f"spread_{h}d") is not None]
            monos = [m[f"mono_{h}d"] for m in results if m.get(f"mono_{h}d") is not None]
            q1g5s = [m[f"q1_gt_q5_{h}d"] for m in results if m.get(f"q1_gt_q5_{h}d") is not None]
            slopes = [m[f"slope_{h}d"] for m in results if m.get(f"slope_{h}d") is not None]

            ic_mean = statistics.mean(ics) if ics else None
            hit = sum(1 for x in ics if x > 0) / len(ics) if ics else None
            topk_mean = statistics.mean(topk_rets) if topk_rets else None
            botk_mean = statistics.mean(botk_rets) if botk_rets else None
            ls_mean = statistics.mean(ls_vals) if ls_vals else None
            sprd_mean = statistics.mean(spreads) if spreads else None
            mono_pct = sum(1 for m in monos if m) / len(monos) if monos else None
            q1g5_pct = sum(1 for q in q1g5s if q) / len(q1g5s) if q1g5s else None
            slope_mean = statistics.mean(slopes) if slopes else None

            row_data[f"ic_{h}d"] = ic_mean
            row_data[f"ls_{h}d"] = ls_mean
            row_data[f"spread_{h}d"] = sprd_mean
            row_data[f"mono_{h}d"] = mono_pct
            row_data[f"ic_vals_{h}d"] = ics
            row_data[f"spread_vals_{h}d"] = spreads

            line += f" {ic_mean:>+10.4f}" if ic_mean is not None else f" {'—':>10s}"
            line += f" {hit:>5.0%}" if hit is not None else f" {'—':>5s}"
            line += f" {topk_mean*100:>7.2f}%" if topk_mean is not None else f" {'—':>8s}"
            line += f" {botk_mean*100:>7.2f}%" if botk_mean is not None else f" {'—':>8s}"
            line += f" {ls_mean*100:>+7.2f}%" if ls_mean is not None else f" {'—':>8s}"
            line += f" {sprd_mean*100:>+7.2f}%" if sprd_mean is not None else f" {'—':>8s}"
            line += f" {mono_pct:>5.0%}" if mono_pct is not None else f" {'—':>6s}"
            line += f" {q1g5_pct:>5.0%}" if q1g5_pct is not None else f" {'—':>5s}"
            line += f" {slope_mean*100:>+7.3f}%" if slope_mean is not None else f" {'—':>8s}"

        tv = statistics.mean(turnover_acc[cfg_key]) if turnover_acc[cfg_key] else None
        row_data["turnover"] = tv
        line += f" {tv:>6.3f}" if tv is not None else f" {'—':>6s}"

        print(line)
        summary[cfg_key] = row_data

    # Paired stats: alpha_removed and alpha_inverted vs baseline
    print()
    print("Paired Statistics (vs Baseline):")
    print("-" * 80)
    baseline = summary["baseline"]
    for cfg_key in ["alpha_removed", "alpha_inverted"]:
        cand = summary[cfg_key]
        for h in horizons:
            base_ics = baseline.get(f"ic_vals_{h}d", [])
            cand_ics = cand.get(f"ic_vals_{h}d", [])
            if len(base_ics) >= 3 and len(cand_ics) >= 3:
                ps = paired_stats(base_ics, cand_ics)
                print(f"  {cfg_key} IC({h}d): Δ={ps['mean_delta']:+.4f}, "
                      f"t={ps['t_stat']:.2f}, p={ps['p_value']:.3f}, "
                      f"CI=[{ps['ci_lo_95']:+.4f}, {ps['ci_hi_95']:+.4f}]"
                      if ps['mean_delta'] is not None else
                      f"  {cfg_key} IC({h}d): insufficient data")

            base_sp = baseline.get(f"spread_vals_{h}d", [])
            cand_sp = cand.get(f"spread_vals_{h}d", [])
            if len(base_sp) >= 3 and len(cand_sp) >= 3:
                ps = paired_stats(base_sp, cand_sp)
                print(f"  {cfg_key} Spread({h}d): Δ={ps['mean_delta']:+.4f}, "
                      f"t={ps['t_stat']:.2f}, p={ps['p_value']:.3f}, "
                      f"CI=[{ps['ci_lo_95']:+.4f}, {ps['ci_hi_95']:+.4f}]"
                      if ps['mean_delta'] is not None else
                      f"  {cfg_key} Spread({h}d): insufficient data")
        print()

    # Write JSON
    out_dir = args.out or (PROJECT_ROOT / "output" / "alpha_ablation")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Strip non-serializable fields
    for k in summary:
        for field in list(summary[k].keys()):
            if field.endswith("_vals_"):
                del summary[k][field]

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nOutput: {out_dir}")


if __name__ == "__main__":
    main()
