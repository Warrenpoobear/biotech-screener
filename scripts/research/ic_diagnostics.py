#!/usr/bin/env python3
"""IC diagnostic breakdowns: year-by-year, rolling, and residual-return IC.

Reuses the anchor_replay infrastructure but adds:
  1. Year-by-year IC(60d) for A_alpha and E_optionality
  2. Rolling 24-month IC(60d)
  3. IC on beta-adjusted residual returns:
       resid_ret = ret - beta_xbi_60d * xbi_ret

If residual IC is positive while raw IC is negative, the "alpha" is masked
by factor structure and the next step is risk-neutral construction.

Usage:
    python scripts/research/ic_diagnostics.py \
        --date-from 2020-01-01 --date-to 2026-02-28 \
        --date-grid monthly --horizons 20,60
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import math
import statistics
import sys
import tarfile
import time
from collections import defaultdict
from dataclasses import dataclass, replace as dc_replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from common.ranking_utils import backfill_columns, safe_float
from decision_engine import DecisionRuleset, compute_actionable_sort_key
from module_5_alpha_cohort import attach_alpha_scores
from scripts.build_alpha_cohort_table import backfill_clinical_z_tier
from scripts.build_alpha_cohort_table_oos import build_oos_table
from scripts.research.anchor_replay import (
    AnchorConfig,
    compute_forward_return,
    discover_archives,
    load_archive_rankings,
    load_price_series,
    paired_stats,
    rerank_rows,
    spearman_ic,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Beta + residual return computation
# ---------------------------------------------------------------------------

def compute_rolling_beta(
    ticker_prices: Dict[str, float],
    xbi_prices: Dict[str, float],
    sorted_dates: List[str],
    as_of_date: str,
    window: int = 60,
) -> Optional[float]:
    """Compute rolling OLS beta of ticker vs XBI over `window` trading days."""
    try:
        idx = sorted_dates.index(as_of_date)
    except ValueError:
        # Find nearest date <= as_of_date
        idx = None
        for i, d in enumerate(sorted_dates):
            if d <= as_of_date:
                idx = i
        if idx is None:
            return None

    # Need window+1 dates to get `window` returns
    start_idx = max(0, idx - window)
    if idx - start_idx < 30:  # MIN_OVERLAP_BARS
        return None

    rets_ticker: List[float] = []
    rets_xbi: List[float] = []
    for i in range(start_idx + 1, idx + 1):
        d_prev = sorted_dates[i - 1]
        d_curr = sorted_dates[i]
        p0 = ticker_prices.get(d_prev)
        p1 = ticker_prices.get(d_curr)
        x0 = xbi_prices.get(d_prev)
        x1 = xbi_prices.get(d_curr)
        if p0 and p1 and x0 and x1 and p0 > 0 and x0 > 0:
            rets_ticker.append((p1 - p0) / p0)
            rets_xbi.append((x1 - x0) / x0)

    if len(rets_ticker) < 30:
        return None

    # OLS: beta = cov(r, xbi) / var(xbi)
    mx = statistics.mean(rets_xbi)
    my = statistics.mean(rets_ticker)
    cov = sum((rets_xbi[i] - mx) * (rets_ticker[i] - my)
              for i in range(len(rets_ticker)))
    var_x = sum((rets_xbi[i] - mx) ** 2 for i in range(len(rets_xbi)))
    if var_x == 0:
        return None
    return cov / var_x


def compute_ic_for_date(
    rows: List[Dict[str, str]],
    snap_date: str,
    prices: Dict[str, Dict[str, float]],
    xbi_prices: Dict[str, float],
    sorted_dates: List[str],
    horizon: int,
    use_residual: bool = False,
) -> Optional[float]:
    """Compute Spearman IC for a single date, optionally on residual returns."""
    eligible = [r for r in rows if r.get("eligible") == "1"
                and r.get("actionable_rank", "").strip()]
    if not eligible:
        return None

    ranked = sorted(eligible, key=lambda r: int(r.get("actionable_rank") or 9999))
    tickers = [r.get("ticker", "") for r in ranked]

    # Resolve trade date
    trade_date = None
    if snap_date in sorted_dates:
        trade_date = snap_date
    else:
        for d in sorted_dates:
            if d <= snap_date:
                trade_date = d
            else:
                break
    if trade_date is None:
        return None

    # Forward returns
    fwd_rets: Dict[str, float] = {}
    for t in tickers:
        if t not in prices:
            continue
        ret = compute_forward_return(prices[t], sorted_dates, trade_date, horizon)
        if ret is None:
            continue

        if use_residual:
            # Compute XBI forward return
            xbi_ret = compute_forward_return(xbi_prices, sorted_dates, trade_date, horizon)
            if xbi_ret is None:
                continue
            # Compute rolling beta as of trade date
            beta = compute_rolling_beta(
                prices[t], xbi_prices, sorted_dates, trade_date, window=60,
            )
            if beta is None:
                beta = 1.0  # fallback
            resid = ret - beta * xbi_ret
            fwd_rets[t] = resid
        else:
            fwd_rets[t] = ret

    signal_tickers = [t for t in tickers if t in fwd_rets]
    if len(signal_tickers) < 5:
        return None

    signal_vals = [-float(i + 1) for i, t in enumerate(tickers) if t in fwd_rets]
    return_vals = [fwd_rets[t] for t in signal_tickers]
    return spearman_ic(signal_vals, return_vals)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="IC diagnostic breakdowns (year-by-year, rolling, residual)",
    )
    parser.add_argument("--archive-dir", type=Path,
                        default=PROJECT_ROOT / "data" / "archives")
    parser.add_argument("--price-csv", type=Path,
                        default=PROJECT_ROOT / "production_data" / "price_history.csv")
    parser.add_argument("--date-from", type=str, default="2020-01-31")
    parser.add_argument("--date-to", type=str, default="2026-02-28")
    parser.add_argument("--date-grid", type=str, default="monthly",
                        choices=["monthly", "weekly", "all"])
    parser.add_argument("--horizons", type=str, default="20,60")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--log-level", default="WARNING",
                        choices=["DEBUG", "INFO", "WARNING"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    horizons = [int(h.strip()) for h in args.horizons.split(",")]

    # Discover archives
    archives = discover_archives(
        args.archive_dir, args.date_from, args.date_to, args.date_grid,
    )
    if not archives:
        print("No archives found.")
        return
    print(f"Archives: {len(archives)} ({args.date_grid}, "
          f"{archives[0][0]} → {archives[-1][0]})")

    # Load prices
    print("Loading prices ...")
    prices = load_price_series(args.price_csv)
    xbi_prices = prices.get("XBI", {})
    if not xbi_prices:
        print("WARNING: No XBI prices found — residual IC will be skipped")
    all_dates_set: set = set()
    for tp in prices.values():
        all_dates_set.update(tp.keys())
    sorted_dates = sorted(all_dates_set)
    print(f"  {len(prices)} tickers, {len(sorted_dates)} trading dates, "
          f"XBI: {len(xbi_prices)} dates")

    # Load ruleset
    snap_dir = PROJECT_ROOT / "data" / "snapshots"
    dates_with_rs = sorted([
        p.name for p in snap_dir.iterdir()
        if p.is_dir() and len(p.name) == 10 and p.name[4] == "-"
        and (p / "decision_ruleset.json").exists()
    ]) if snap_dir.exists() else []
    base_rs = (DecisionRuleset.from_json(str(snap_dir / dates_with_rs[-1] / "decision_ruleset.json"))
               if dates_with_rs else DecisionRuleset())
    rs_alpha = dc_replace(base_rs, sort_anchor="alpha_cohort",
                          alpha_modifier_mode="off", alpha_modifier_weight=0.0)

    configs = [
        AnchorConfig("A_alpha", "alpha_pct"),
        AnchorConfig("E_optionality", "optionality_pct"),
    ]

    # Pre-build alpha tables
    print("\nPre-building PIT alpha tables ...")
    alpha_tables: Dict[str, Optional[dict]] = {}
    for date_str, tar_path in archives:
        table = build_oos_table(
            as_of_date=date_str, train_mode="trailing-6", horizon=84,
            min_train_dates=6, archive_dir=args.archive_dir,
            price_csv=args.price_csv,
        )
        alpha_tables[date_str] = table

    n_built = sum(1 for v in alpha_tables.values() if v is not None)
    print(f"  {n_built}/{len(alpha_tables)} tables built")

    # Evaluate per-date IC (raw + residual) for each config
    # Structure: {config_label: {date: {horizon: {raw_ic, resid_ic}}}}
    results: Dict[str, Dict[str, Dict[int, Dict[str, Optional[float]]]]] = {}

    print(f"\nEvaluating {len(configs)} configs × {len(archives)} dates ...")

    for cfg in configs:
        results[cfg.label] = {}

    for date_str, tar_path in archives:
        raw_rows = load_archive_rankings(tar_path, date_str)
        if not raw_rows:
            continue

        backfill_columns(raw_rows)
        backfill_clinical_z_tier(raw_rows)

        alpha_table = alpha_tables.get(date_str)
        alpha_built = alpha_table is not None
        if alpha_built:
            attach_alpha_scores(
                raw_rows, alpha_table,
                shrink_k=base_rs.alpha_cohort_shrink_k,
                clip_min=base_rs.alpha_cohort_clip_min,
                clip_max=base_rs.alpha_cohort_clip_max,
            )
        else:
            for r in raw_rows:
                r["alpha_cohort_raw"] = 0.0
                r["alpha_cohort_pct"] = 0.5
                r["alpha_cohort_key"] = "early|none|nonpos"

        for cfg in configs:
            # Skip alpha-dependent configs without alpha table
            if cfg.anchor_source in ("alpha_pct", "blend") and not alpha_built:
                continue

            rows = copy.deepcopy(raw_rows)
            rows = rerank_rows(rows, cfg, rs_alpha)

            date_ics: Dict[int, Dict[str, Optional[float]]] = {}
            for h in horizons:
                raw_ic = compute_ic_for_date(
                    rows, date_str, prices, xbi_prices, sorted_dates, h,
                    use_residual=False,
                )
                resid_ic = None
                if xbi_prices:
                    resid_ic = compute_ic_for_date(
                        rows, date_str, prices, xbi_prices, sorted_dates, h,
                        use_residual=True,
                    )
                date_ics[h] = {"raw": raw_ic, "resid": resid_ic}
            results[cfg.label][date_str] = date_ics

        print(f"  {date_str}: alpha={'YES' if alpha_built else 'NO'}")

    # ---- Year-by-year breakdown ----
    print("\n" + "=" * 100)
    print("YEAR-BY-YEAR IC(60d) BREAKDOWN")
    print("=" * 100)

    for h in horizons:
        print(f"\n--- Horizon: {h}d ---")
        # Collect years
        all_years = sorted(set(d[:4] for d in
                               set().union(*(results[c].keys() for c in results))))

        # Header
        header = f"{'Year':<8}"
        for cfg in configs:
            header += f" {'IC_raw(' + cfg.label + ')':>22} {'IC_resid':>12} {'n':>4}"
        print(header)
        print("-" * len(header))

        for year in all_years:
            line = f"{year:<8}"
            for cfg in configs:
                year_dates = [d for d in results[cfg.label] if d[:4] == year]
                raw_ics = [results[cfg.label][d][h]["raw"]
                           for d in year_dates if results[cfg.label][d][h]["raw"] is not None]
                resid_ics = [results[cfg.label][d][h]["resid"]
                             for d in year_dates if results[cfg.label][d][h].get("resid") is not None]

                raw_mean = statistics.mean(raw_ics) if raw_ics else None
                resid_mean = statistics.mean(resid_ics) if resid_ics else None

                raw_s = f"{raw_mean:+.4f}" if raw_mean is not None else "—"
                resid_s = f"{resid_mean:+.4f}" if resid_mean is not None else "—"
                n_s = str(len(raw_ics))
                line += f" {raw_s:>22} {resid_s:>12} {n_s:>4}"
            print(line)

        # Overall
        line = f"{'ALL':<8}"
        for cfg in configs:
            all_raw = [results[cfg.label][d][h]["raw"]
                       for d in results[cfg.label] if results[cfg.label][d][h]["raw"] is not None]
            all_resid = [results[cfg.label][d][h]["resid"]
                         for d in results[cfg.label] if results[cfg.label][d][h].get("resid") is not None]
            raw_mean = statistics.mean(all_raw) if all_raw else None
            resid_mean = statistics.mean(all_resid) if all_resid else None
            raw_s = f"{raw_mean:+.4f}" if raw_mean is not None else "—"
            resid_s = f"{resid_mean:+.4f}" if resid_mean is not None else "—"
            line += f" {raw_s:>22} {resid_s:>12} {str(len(all_raw)):>4}"
        print(line)

    # ---- Paired stats: raw vs residual ----
    print("\n" + "=" * 100)
    print("RAW vs RESIDUAL IC COMPARISON")
    print("=" * 100)
    for cfg in configs:
        for h in horizons:
            raw_vals = []
            resid_vals = []
            for d in sorted(results[cfg.label].keys()):
                r = results[cfg.label][d][h]["raw"]
                s = results[cfg.label][d][h].get("resid")
                if r is not None and s is not None:
                    raw_vals.append(r)
                    resid_vals.append(s)

            if len(raw_vals) >= 3:
                raw_mean = statistics.mean(raw_vals)
                resid_mean = statistics.mean(resid_vals)
                ps = paired_stats(raw_vals, resid_vals)
                print(f"  {cfg.label} IC({h}d): raw={raw_mean:+.4f}, "
                      f"resid={resid_mean:+.4f}, "
                      f"Δ={ps['mean_delta']:+.4f}, t={ps['t_stat']:.2f}, "
                      f"p={ps['p_value']:.3f} (n={ps['n']})")

    # ---- Write JSON ----
    out_dir = args.out or (PROJECT_ROOT / "output" / "ic_diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Flatten for JSON
    json_out: Dict[str, Any] = {}
    for cfg_label, date_data in results.items():
        json_out[cfg_label] = {}
        for d, horizon_data in sorted(date_data.items()):
            json_out[cfg_label][d] = {}
            for h, ic_data in horizon_data.items():
                json_out[cfg_label][d][str(h)] = ic_data

    with open(out_dir / "ic_diagnostics.json", "w") as f:
        json.dump(json_out, f, indent=2, default=str)

    # Year summary
    year_summary: Dict[str, Any] = {}
    for cfg in configs:
        year_summary[cfg.label] = {}
        for h in horizons:
            all_years = sorted(set(d[:4] for d in results[cfg.label]))
            for year in all_years:
                year_dates = [d for d in results[cfg.label] if d[:4] == year]
                raw_ics = [results[cfg.label][d][h]["raw"]
                           for d in year_dates if results[cfg.label][d][h]["raw"] is not None]
                resid_ics = [results[cfg.label][d][h]["resid"]
                             for d in year_dates if results[cfg.label][d][h].get("resid") is not None]
                year_summary[cfg.label][f"{year}_ic{h}d_raw"] = (
                    round(statistics.mean(raw_ics), 6) if raw_ics else None)
                year_summary[cfg.label][f"{year}_ic{h}d_resid"] = (
                    round(statistics.mean(resid_ics), 6) if resid_ics else None)
                year_summary[cfg.label][f"{year}_n{h}d"] = len(raw_ics)

    with open(out_dir / "year_summary.json", "w") as f:
        json.dump(year_summary, f, indent=2, default=str)

    print(f"\nOutput: {out_dir}")
    print(f"  ic_diagnostics.json, year_summary.json")


if __name__ == "__main__":
    main()
