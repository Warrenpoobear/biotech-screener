#!/usr/bin/env python3
"""EES threshold sweep — efficient frontier of Sharpe vs frequency.

Sweeps quality cutoff (10-40%) × trap cutoff (0-40%) and measures
portfolio Sharpe, mean return, hit rate, and coverage (trade frequency).

Identifies the optimal operating point on the Sharpe-vs-frequency frontier.
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from event_ev.expectation_error_model import ExpectationErrorModel
from scripts.eval_forward_returns import (
    compute_forward_return,
    discover_snapshot_dates,
    load_price_series,
    load_rankings,
    resolve_trade_date,
)

TOP_K = 30
HORIZON = 20  # primary decision horizon
model = ExpectationErrorModel()

QUALITY_CUTS = [0, 5, 10, 15, 20, 25, 30, 35, 40]
TRAP_CUTS = [0, 5, 10, 15, 20, 25, 30, 35, 40]


def _pct_threshold(vals: List[float], pct: int) -> float:
    """Value at the pct-th percentile (0-100)."""
    if pct == 0:
        return float("-inf")
    s = sorted(vals)
    idx = min(int(len(s) * pct / 100), len(s) - 1)
    return s[idx]


def run_sweep(
    snapshot_root: Path,
    price_csv: Path,
    date_from: str = "2022-03-18",
    date_to: str = "2026-03-31",
) -> Dict[str, Any]:
    snap_dates = discover_snapshot_dates(snapshot_root, date_from, date_to)
    prices = load_price_series(price_csv)

    all_dates_set: set = set()
    for tp in prices.values():
        all_dates_set.update(tp.keys())
    sorted_dates = sorted(all_dates_set)

    # {(q_cut, t_cut): [portfolio_returns_per_date]}
    cell_returns: Dict[tuple, List[float]] = defaultdict(list)
    cell_n_names: Dict[tuple, List[int]] = defaultdict(list)
    n_dates = 0

    for snap_date in snap_dates:
        trade_date = resolve_trade_date(sorted_dates, snap_date, "next_trading_day")
        if not trade_date:
            continue

        rankings = load_rankings(snapshot_root / snap_date)
        if not rankings:
            continue

        scores = model.score_batch(rankings, snap_date)
        if not scores:
            continue

        by_ticker = {s.ticker: s for s in scores}
        rank_map = {}
        for row in rankings:
            t = row.get("ticker", "")
            ar = row.get("actionable_rank", "")
            if t and ar and ar.strip().isdigit():
                rank_map[t] = int(ar)

        # Forward returns
        fwd: Dict[str, float] = {}
        for ticker in by_ticker:
            if ticker in prices:
                ret = compute_forward_return(prices[ticker], sorted_dates, trade_date, HORIZON)
                if ret is not None:
                    fwd[ticker] = ret

        eligible = [t for t in by_ticker if t in fwd and t in rank_map]
        if len(eligible) < TOP_K:
            continue

        n_dates += 1

        quality_vals = [by_ticker[t].quality_overlay_score for t in eligible]
        trap_vals = [by_ticker[t].trap_overlay_score for t in eligible]

        for q_cut in QUALITY_CUTS:
            q_thresh = _pct_threshold(quality_vals, q_cut)
            for t_cut in TRAP_CUTS:
                t_thresh = _pct_threshold(trap_vals, t_cut)

                passed = [
                    t
                    for t in eligible
                    if by_ticker[t].quality_overlay_score > q_thresh and by_ticker[t].trap_overlay_score > t_thresh
                ]

                top = sorted(passed, key=lambda t: rank_map[t])[:TOP_K]
                if len(top) >= 10:
                    port_ret = statistics.mean(fwd[t] for t in top)
                    cell_returns[(q_cut, t_cut)].append(port_ret)
                    cell_n_names[(q_cut, t_cut)].append(len(top))

    # Aggregate
    rows = []
    for (q_cut, t_cut), rets in sorted(cell_returns.items()):
        if not rets:
            continue
        m = statistics.mean(rets)
        s = statistics.stdev(rets) if len(rets) >= 2 else 0.001
        sharpe = m / s if s > 0 else 0
        hr = sum(1 for r in rets if r > 0) / len(rets)
        freq = len(rets) / n_dates if n_dates > 0 else 0
        avg_names = statistics.mean(cell_n_names[(q_cut, t_cut)])
        rows.append(
            {
                "quality_cut_pct": q_cut,
                "trap_cut_pct": t_cut,
                "mean_ret_pct": round(m * 100, 4),
                "std_ret_pct": round(s * 100, 4),
                "sharpe": round(sharpe, 3),
                "hit_rate": round(hr, 3),
                "frequency": round(freq, 3),
                "n_dates": len(rets),
                "avg_names": round(avg_names, 1),
            }
        )

    return {"n_dates_total": n_dates, "horizon": HORIZON, "top_k": TOP_K, "cells": rows}


def main() -> None:
    snapshot_root = PROJECT_ROOT / "data" / "snapshots"
    price_csv = PROJECT_ROOT / "production_data" / "price_history.csv"
    out_dir = PROJECT_ROOT / "output" / "ees_threshold_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running threshold sweep (quality {QUALITY_CUTS} × trap {TRAP_CUTS})...")
    print(f"Horizon: {HORIZON}d, Top-K: {TOP_K}")
    results = run_sweep(snapshot_root, price_csv)

    with open(out_dir / "sweep_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    cells = results["cells"]
    n = results["n_dates_total"]
    print(f"\nEvaluated {n} dates, {len(cells)} threshold combinations\n")

    # Print heatmap: quality (rows) × trap (cols) → Sharpe
    print(f"{'=' * 90}")
    print(f"  SHARPE HEATMAP — {HORIZON}d horizon, EW Top-{TOP_K}")
    print(f"{'=' * 90}")

    lookup = {(r["quality_cut_pct"], r["trap_cut_pct"]): r for r in cells}

    qt_header = "Q \\ T"
    print(f"{qt_header:>8s}", end="")
    for t_cut in TRAP_CUTS:
        print(f"  {t_cut:>3d}%  ", end="")
    print()
    print("-" * 90)

    for q_cut in QUALITY_CUTS:
        print(f"  {q_cut:>3d}%  ", end="")
        for t_cut in TRAP_CUTS:
            r = lookup.get((q_cut, t_cut))
            if r and r["sharpe"] is not None:
                sharpe = r["sharpe"]
                # Highlight best cells
                marker = "*" if sharpe >= 0.40 else " "
                print(f"  {sharpe:>4.2f}{marker} ", end="")
            else:
                print("     —   ", end="")
        print()

    # Print return heatmap
    print(f"\n{'=' * 90}")
    print(f"  MEAN RETURN (%) HEATMAP — {HORIZON}d horizon")
    print(f"{'=' * 90}")

    print(f"{qt_header:>8s}", end="")
    for t_cut in TRAP_CUTS:
        print(f"  {t_cut:>3d}%  ", end="")
    print()
    print("-" * 90)

    for q_cut in QUALITY_CUTS:
        print(f"  {q_cut:>3d}%  ", end="")
        for t_cut in TRAP_CUTS:
            r = lookup.get((q_cut, t_cut))
            if r and r["mean_ret_pct"] is not None:
                print(f"  {r['mean_ret_pct']:>+5.2f} ", end="")
            else:
                print("     —   ", end="")
        print()

    # Print frequency heatmap
    print(f"\n{'=' * 90}")
    print(f"  FREQUENCY (% of dates with enough names) — {HORIZON}d horizon")
    print(f"{'=' * 90}")

    print(f"{qt_header:>8s}", end="")
    for t_cut in TRAP_CUTS:
        print(f"  {t_cut:>3d}%  ", end="")
    print()
    print("-" * 90)

    for q_cut in QUALITY_CUTS:
        print(f"  {q_cut:>3d}%  ", end="")
        for t_cut in TRAP_CUTS:
            r = lookup.get((q_cut, t_cut))
            if r and r["frequency"] is not None:
                print(f"  {r['frequency']:>4.0%}  ", end="")
            else:
                print("     —   ", end="")
        print()

    # Efficient frontier: top 10 by Sharpe with frequency >= 50%
    print(f"\n{'=' * 90}")
    print("  EFFICIENT FRONTIER — Top 10 by Sharpe (frequency >= 50%)")
    print(f"{'=' * 90}")
    viable = [r for r in cells if r["frequency"] >= 0.50]
    viable.sort(key=lambda r: r["sharpe"], reverse=True)
    print(f"{'Q%':>4s} {'T%':>4s} {'Mean%':>8s} {'Sharpe':>8s} {'Hit%':>7s} {'Freq':>6s} {'AvgN':>6s}")
    print("-" * 50)
    for r in viable[:10]:
        print(
            f"{r['quality_cut_pct']:>3d}% {r['trap_cut_pct']:>3d}% "
            f"{r['mean_ret_pct']:>+7.3f}% {r['sharpe']:>+7.3f} "
            f"{r['hit_rate']:>6.0%} {r['frequency']:>5.0%} {r['avg_names']:>5.1f}"
        )

    # Also top 10 by Sharpe with no frequency constraint
    print(f"\n{'=' * 90}")
    print("  UNCONSTRAINED — Top 10 by Sharpe (any frequency)")
    print(f"{'=' * 90}")
    all_sorted = sorted(cells, key=lambda r: r["sharpe"], reverse=True)
    print(f"{'Q%':>4s} {'T%':>4s} {'Mean%':>8s} {'Sharpe':>8s} {'Hit%':>7s} {'Freq':>6s} {'AvgN':>6s}")
    print("-" * 50)
    for r in all_sorted[:10]:
        print(
            f"{r['quality_cut_pct']:>3d}% {r['trap_cut_pct']:>3d}% "
            f"{r['mean_ret_pct']:>+7.3f}% {r['sharpe']:>+7.3f} "
            f"{r['hit_rate']:>6.0%} {r['frequency']:>5.0%} {r['avg_names']:>5.1f}"
        )

    print(f"\nWritten: {out_dir / 'sweep_results.json'}")


if __name__ == "__main__":
    main()
