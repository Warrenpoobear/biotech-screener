#!/usr/bin/env python3
"""Sanity-check: does the replay harness reproduce archive rankings?

For 3 spot-check dates across regimes:
  1. Compute IC60 from archive's existing actionable_rank (production proxy)
  2. Compute IC60 from anchor_replay's A_alpha re-ranking
  3. Compare top-20 tickers between (1) and (2)
  4. Compute IC60 with REVERSED ranking (bottom-20 as "top")

If A_alpha doesn't reproduce the archive ranking, the replay isn't
measuring production and must be fixed before interpreting results.
"""
from __future__ import annotations

import copy
import csv
import json
import math
import statistics
import sys
import tarfile
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from common.ranking_utils import backfill_columns, safe_float
from decision_engine import DecisionRuleset, compute_actionable_sort_key
from module_5_alpha_cohort import attach_alpha_scores
from scripts.build_alpha_cohort_table import backfill_clinical_z_tier
from scripts.build_alpha_cohort_table_oos import build_oos_table
from scripts.research.anchor_replay import (
    AnchorConfig,
    _avg_ranks,
    compute_forward_return,
    load_archive_rankings,
    load_price_series,
    rerank_rows,
    spearman_ic,
)

ARCHIVE_DIR = PROJECT_ROOT / "data" / "archives"
PRICE_CSV = PROJECT_ROOT / "production_data" / "price_history.csv"

# 3 dates spanning regimes
SPOT_CHECK_DATES = [
    "2020-07-31",   # COVID biotech boom
    "2022-09-30",   # Bear market / rate hikes
    "2025-06-30",   # Recent
]


def get_archive_top_k(rows: List[Dict[str, str]], k: int = 20) -> List[str]:
    """Get top-K tickers from archive's existing actionable_rank."""
    eligible = [r for r in rows if r.get("eligible") == "1"]
    ranked = sorted(eligible, key=lambda r: int(r.get("actionable_rank") or 9999))
    return [r.get("ticker", "") for r in ranked[:k]]


def compute_ic_from_ranks(
    rows: List[Dict[str, str]],
    prices: Dict[str, Dict[str, float]],
    sorted_dates: List[str],
    snap_date: str,
    horizon: int,
    rank_field: str = "actionable_rank",
    reverse: bool = False,
) -> Dict[str, Any]:
    """Compute IC using a specific rank field. If reverse=True, flip the signal."""
    eligible = [r for r in rows if r.get("eligible") == "1"
                and r.get(rank_field, "").strip()]
    if not eligible:
        return {"ic": None, "n": 0}

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
        return {"ic": None, "n": 0, "reason": "no trade date"}

    # Get forward returns
    tickers_ranked = sorted(eligible, key=lambda r: int(r.get(rank_field) or 9999))
    ticker_list = [r.get("ticker", "") for r in tickers_ranked]

    fwd_rets: Dict[str, float] = {}
    for t in ticker_list:
        if t not in prices:
            continue
        ret = compute_forward_return(prices[t], sorted_dates, trade_date, horizon)
        if ret is not None:
            fwd_rets[t] = ret

    signal_tickers = [t for t in ticker_list if t in fwd_rets]
    if len(signal_tickers) < 5:
        return {"ic": None, "n": len(signal_tickers), "reason": "too few"}

    # Signal = negative rank (rank 1 = best = highest signal)
    # If reverse, flip: rank 1 = worst = lowest signal
    signal_vals = []
    for i, t in enumerate(ticker_list):
        if t in fwd_rets:
            rank_val = float(i + 1)
            if reverse:
                signal_vals.append(rank_val)   # high rank = high signal (reversed)
            else:
                signal_vals.append(-rank_val)  # low rank = high signal (normal)

    return_vals = [fwd_rets[t] for t in signal_tickers]
    ic = spearman_ic(signal_vals, return_vals)

    # Top-20 gross return
    held = [t for t in ticker_list[:20] if t in fwd_rets]
    top20_ret = statistics.mean(fwd_rets[t] for t in held) if held else None

    return {
        "ic": round(ic, 6) if ic is not None else None,
        "n": len(signal_tickers),
        "top20_ret": round(top20_ret, 6) if top20_ret is not None else None,
        "n_held": len(held),
    }


def main() -> None:
    print("Loading prices ...")
    prices = load_price_series(PRICE_CSV)
    all_dates_set: set = set()
    for tp in prices.values():
        all_dates_set.update(tp.keys())
    sorted_dates = sorted(all_dates_set)
    print(f"  {len(prices)} tickers, {len(sorted_dates)} trading dates\n")

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

    from dataclasses import replace as dc_replace
    rs_alpha = dc_replace(base_rs,
                          sort_anchor="alpha_cohort",
                          alpha_modifier_mode="off",
                          alpha_modifier_weight=0.0)

    cfg_alpha = AnchorConfig("A_alpha", "alpha_pct")
    cfg_optionality = AnchorConfig("E_optionality", "optionality_pct")

    horizon = 60

    for date_str in SPOT_CHECK_DATES:
        tar_path = ARCHIVE_DIR / f"{date_str}.tar.gz"
        if not tar_path.exists():
            print(f"=== {date_str}: ARCHIVE NOT FOUND ===\n")
            continue

        print(f"{'=' * 80}")
        print(f"  DATE: {date_str}")
        print(f"{'=' * 80}")

        raw_rows = load_archive_rankings(tar_path, date_str)
        if not raw_rows:
            print(f"  No rankings — skip\n")
            continue

        n_eligible = sum(1 for r in raw_rows if r.get("eligible") == "1")
        archive_top20 = get_archive_top_k(raw_rows, 20)

        # 1) IC from archive's existing actionable_rank
        archive_ic = compute_ic_from_ranks(
            raw_rows, prices, sorted_dates, date_str, horizon,
            rank_field="actionable_rank",
        )

        # Build alpha table for this date
        table = build_oos_table(
            as_of_date=date_str,
            train_mode="trailing-6",
            horizon=84,
            min_train_dates=6,
            archive_dir=ARCHIVE_DIR,
            price_csv=PRICE_CSV,
        )

        # Prepare rows for re-ranking
        rerank_base = copy.deepcopy(raw_rows)
        backfill_columns(rerank_base)
        backfill_clinical_z_tier(rerank_base)
        if table is not None:
            attach_alpha_scores(
                rerank_base, table,
                shrink_k=base_rs.alpha_cohort_shrink_k,
                clip_min=base_rs.alpha_cohort_clip_min,
                clip_max=base_rs.alpha_cohort_clip_max,
            )
        else:
            for r in rerank_base:
                r["alpha_cohort_raw"] = 0.0
                r["alpha_cohort_pct"] = 0.5
                r["alpha_cohort_key"] = "early|none|nonpos"

        # 2a) Re-rank with A_alpha config
        rows_alpha = copy.deepcopy(rerank_base)
        rows_alpha = rerank_rows(rows_alpha, cfg_alpha, rs_alpha)
        alpha_top20 = [r.get("ticker", "") for r in
                       sorted([r for r in rows_alpha if r.get("eligible") == "1"],
                              key=lambda r: int(r.get("actionable_rank") or 9999))[:20]]

        alpha_ic = compute_ic_from_ranks(
            rows_alpha, prices, sorted_dates, date_str, horizon,
            rank_field="actionable_rank",
        )

        # 2b) Re-rank with E_optionality config
        rows_opt = copy.deepcopy(rerank_base)
        rows_opt = rerank_rows(rows_opt, cfg_optionality, rs_alpha)
        opt_top20 = [r.get("ticker", "") for r in
                     sorted([r for r in rows_opt if r.get("eligible") == "1"],
                            key=lambda r: int(r.get("actionable_rank") or 9999))[:20]]

        opt_ic = compute_ic_from_ranks(
            rows_opt, prices, sorted_dates, date_str, horizon,
            rank_field="actionable_rank",
        )

        # 3) Reversed ranking (bottom-20 as "top")
        reversed_ic = compute_ic_from_ranks(
            raw_rows, prices, sorted_dates, date_str, horizon,
            rank_field="actionable_rank",
            reverse=True,
        )

        # Report
        print(f"\n  Eligible tickers: {n_eligible}")
        print(f"  Alpha table built: {'YES' if table else 'NO'}")
        print()

        print(f"  Archive ranking (production proxy):")
        print(f"    IC(60d) = {archive_ic['ic']}  (n={archive_ic['n']})")
        print(f"    Top-20 ret = {archive_ic.get('top20_ret')}")
        print(f"    Top-20: {archive_top20}")
        print()

        print(f"  Replay A_alpha re-ranking:")
        print(f"    IC(60d) = {alpha_ic['ic']}  (n={alpha_ic['n']})")
        print(f"    Top-20 ret = {alpha_ic.get('top20_ret')}")
        print(f"    Top-20: {alpha_top20}")
        print()

        # Overlap
        overlap = set(archive_top20) & set(alpha_top20)
        print(f"  Archive vs A_alpha top-20 overlap: {len(overlap)}/20 "
              f"({len(overlap)/20:.0%})")
        if len(overlap) < 20:
            only_archive = set(archive_top20) - set(alpha_top20)
            only_alpha = set(alpha_top20) - set(archive_top20)
            print(f"    Only in archive: {sorted(only_archive)}")
            print(f"    Only in A_alpha: {sorted(only_alpha)}")
        print()

        print(f"  Replay E_optionality re-ranking:")
        print(f"    IC(60d) = {opt_ic['ic']}  (n={opt_ic['n']})")
        print(f"    Top-20 ret = {opt_ic.get('top20_ret')}")
        print(f"    Top-20: {opt_top20}")
        print()

        overlap_ae = set(alpha_top20) & set(opt_top20)
        print(f"  A_alpha vs E_optionality top-20 overlap: {len(overlap_ae)}/20")
        print()

        print(f"  REVERSED ranking (bottom-20 = 'top'):")
        print(f"    IC(60d) = {reversed_ic['ic']}  (n={reversed_ic['n']})")
        print(f"    (Expect sign flip if ranking has any predictive content)")
        print()


if __name__ == "__main__":
    main()
