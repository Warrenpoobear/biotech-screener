#!/usr/bin/env python3
"""2024 Inversion Attribution — diagnose why D-tier outperforms AB in 2024.

Reads the walkforward panel CSV and breaks AB vs CD performance by multiple
dimensions to identify the top factors correlated with the 2024 tier inversion.

Usage:
    python scripts/attribution_2024_inversion.py \
        --panel artifacts/walkforward_panel__baseline_full.csv \
        [--output-dir artifacts]
"""
from __future__ import annotations

import argparse
import statistics
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PANEL = PROJECT_ROOT / "artifacts" / "walkforward_panel__baseline_full.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts"


# ---------------------------------------------------------------------------
# Bucketing helpers
# ---------------------------------------------------------------------------
def _bucket_continuous(
    series: pd.Series, edges: List[float], labels: List[str]
) -> pd.Series:
    """Bucket a numeric series into labeled bins. NaN → 'missing'."""
    result = pd.cut(series, bins=[-np.inf] + edges + [np.inf], labels=labels)
    return result.astype(str).fillna("missing")


def _adv_buckets(adv: pd.Series) -> pd.Series:
    """ADV dollar buckets: micro (<1M), small (1-5M), mid (5-25M), large (>25M)."""
    return _bucket_continuous(
        adv,
        [1_000_000, 5_000_000, 25_000_000],
        ["micro_<1M", "small_1-5M", "mid_5-25M", "large_>25M"],
    )


def _cost_buckets(cost: pd.Series) -> pd.Series:
    """Cost BPS buckets aligned with haircut thresholds."""
    return _bucket_continuous(
        cost,
        [400, 1000, 2000],
        ["low_<400", "mid_400-1k", "high_1k-2k", "extreme_>2k"],
    )


def _dd_abs_margin_buckets(margin: pd.Series) -> pd.Series:
    """Drawdown absolute margin buckets (how far from -0.40 gate)."""
    return _bucket_continuous(
        margin,
        [-0.20, -0.10, -0.05, 0.0, 0.10],
        ["deep_<-20pp", "stressed_-20to-10pp", "near_-10to-5pp",
         "borderline_-5to0pp", "safe_0to10pp", "comfortable_>10pp"],
    )


def _dd_rel_margin_buckets(margin: pd.Series) -> pd.Series:
    """Drawdown relative margin buckets (vs XBI gate)."""
    return _bucket_continuous(
        margin,
        [-0.15, -0.05, 0.0, 0.10],
        ["deep_<-15pp", "stressed_-15to-5pp", "borderline_-5to0pp",
         "safe_0to10pp", "comfortable_>10pp"],
    )


def _optionality_buckets(opt: pd.Series) -> pd.Series:
    """Optionality score buckets."""
    return _bucket_continuous(
        opt,
        [0.30, 0.55, 0.60, 0.75],
        ["low_<0.30", "mid_0.30-0.55", "b_floor_0.55-0.60",
         "a_zone_0.60-0.75", "high_>0.75"],
    )


# ---------------------------------------------------------------------------
# Stats computation
# ---------------------------------------------------------------------------
def _group_stats(
    df: pd.DataFrame, ret_col: str = "fwd_ret_60d"
) -> Dict[str, Any]:
    """Compute stats for a group of rows."""
    vals = pd.to_numeric(df[ret_col], errors="coerce").dropna()
    n = len(vals)
    if n == 0:
        return {"n": 0, "mean": None, "median": None, "hit_pct": None,
                "p25": None, "p75": None}
    return {
        "n": n,
        "mean": round(float(vals.mean()), 2),
        "median": round(float(vals.median()), 2),
        "hit_pct": round(float((vals > 0).mean() * 100), 1),
        "p25": round(float(vals.quantile(0.25)), 2),
        "p75": round(float(vals.quantile(0.75)), 2),
    }


def _compute_separation(
    df: pd.DataFrame, ret_col: str = "fwd_ret_60d"
) -> Optional[float]:
    """AB-CD mean separation (eligible only, C-tier only for CD)."""
    elig = df[df["eligible"].astype(str).str.strip() == "1"]
    ret = pd.to_numeric(elig[ret_col], errors="coerce")
    ab = ret[elig["tier"].isin(["A", "B"])].dropna()
    c = ret[elig["tier"] == "C"].dropna()
    if len(ab) == 0 or len(c) == 0:
        return None
    return round(float(ab.mean() - c.mean()), 2)


# ---------------------------------------------------------------------------
# Dimension breakout
# ---------------------------------------------------------------------------
def breakout_by_dimension(
    df: pd.DataFrame,
    dim_col: str,
    year: str,
    ret_col: str = "fwd_ret_60d",
) -> List[Dict[str, Any]]:
    """Break AB vs D performance by a categorical dimension."""
    sub = df[df["as_of_date"].str.startswith(year)].copy()
    sub["_ret"] = pd.to_numeric(sub[ret_col], errors="coerce")

    results = []
    for val in sorted(sub[dim_col].dropna().unique()):
        mask = sub[dim_col] == val
        group = sub[mask]

        ab_rows = group[group["tier"].isin(["A", "B"])]
        c_rows = group[(group["tier"] == "C") &
                       (group["eligible"].astype(str).str.strip() == "1")]
        d_rows = group[group["tier"] == "D"]
        all_elig = group[group["eligible"].astype(str).str.strip() == "1"]

        ab_stats = _group_stats(ab_rows, ret_col)
        c_stats = _group_stats(c_rows, ret_col)
        d_stats = _group_stats(d_rows, ret_col)
        all_stats = _group_stats(group, ret_col)

        # AB-D spread
        ab_d_spread = None
        if ab_stats["mean"] is not None and d_stats["mean"] is not None:
            ab_d_spread = round(ab_stats["mean"] - d_stats["mean"], 2)

        # AB-C spread (eligible)
        ab_c_spread = None
        if ab_stats["mean"] is not None and c_stats["mean"] is not None:
            ab_c_spread = round(ab_stats["mean"] - c_stats["mean"], 2)

        results.append({
            "value": str(val),
            "n_total": len(group),
            "n_ab": ab_stats["n"],
            "n_c": c_stats["n"],
            "n_d": d_stats["n"],
            "ab_mean": ab_stats["mean"],
            "ab_median": ab_stats["median"],
            "ab_hit": ab_stats["hit_pct"],
            "c_mean": c_stats["mean"],
            "d_mean": d_stats["mean"],
            "d_median": d_stats["median"],
            "d_hit": d_stats["hit_pct"],
            "ab_d_spread": ab_d_spread,
            "ab_c_spread": ab_c_spread,
            "all_mean": all_stats["mean"],
        })

    return results


def rank_drivers(
    breakouts_2024: Dict[str, List[Dict]],
    breakouts_2025: Dict[str, List[Dict]],
) -> List[Dict[str, Any]]:
    """Rank dimensions by the range of AB-D spread across their values.

    A wide range means the dimension explains a lot of the AB vs D
    performance difference — some cells are very positive, others very
    negative, indicating causal structure.
    """
    drivers = []
    for dim_name in breakouts_2024:
        b_2024 = breakouts_2024[dim_name]
        b_2025 = breakouts_2025.get(dim_name, [])

        # 2024 spread range
        spreads_2024 = [
            r["ab_d_spread"] for r in b_2024
            if r["ab_d_spread"] is not None and r["n_ab"] >= 5
        ]
        spreads_2025 = [
            r["ab_d_spread"] for r in b_2025
            if r["ab_d_spread"] is not None and r["n_ab"] >= 5
        ]

        if len(spreads_2024) < 2:
            continue

        spread_range_2024 = max(spreads_2024) - min(spreads_2024)

        # Find worst cell (most negative AB-D spread)
        worst = min(b_2024, key=lambda r: r["ab_d_spread"] or 0)
        best = max(b_2024, key=lambda r: r["ab_d_spread"] or 0)

        spread_range_2025 = (
            max(spreads_2025) - min(spreads_2025) if len(spreads_2025) >= 2
            else None
        )

        drivers.append({
            "dimension": dim_name,
            "spread_range_2024": round(spread_range_2024, 2),
            "spread_range_2025": round(spread_range_2025, 2) if spread_range_2025 else None,
            "worst_cell_2024": worst["value"],
            "worst_spread_2024": worst["ab_d_spread"],
            "best_cell_2024": best["value"],
            "best_spread_2024": best["ab_d_spread"],
            "n_cells_with_data": len(spreads_2024),
        })

    # Sort by spread range (descending — widest range = most explanatory)
    drivers.sort(key=lambda d: d["spread_range_2024"], reverse=True)
    return drivers


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------
def generate_report(
    df: pd.DataFrame,
    breakouts_2024: Dict[str, List[Dict]],
    breakouts_2025: Dict[str, List[Dict]],
    drivers: List[Dict],
) -> str:
    lines = []
    today = date.today().isoformat()
    lines.append(f"# 2024 Inversion Attribution Report")
    lines.append(f"")
    lines.append(f"**Date**: {today}")
    lines.append(f"**Panel**: walkforward_panel__baseline_full.csv")
    lines.append(f"**Rows**: {len(df)} ({df['as_of_date'].nunique()} snapshots)")
    lines.append("")

    # Executive summary
    df_2024 = df[df["as_of_date"].str.startswith("2024")]
    df_2025 = df[df["as_of_date"].str.startswith("2025")]
    sep_2024 = _compute_separation(df_2024)
    sep_2025 = _compute_separation(df_2025)

    lines.append("## Executive Summary")
    lines.append("")
    lines.append(f"- 2024 AB-CD separation: **{sep_2024:+.2f}pp** (inverted)")
    lines.append(f"- 2025 AB-CD separation: **{sep_2025:+.2f}pp** (healthy)")
    lines.append("")

    # Overall tier performance by year
    lines.append("## Tier Performance by Year")
    lines.append("")
    for year_label, sub in [("2024", df_2024), ("2025", df_2025)]:
        lines.append(f"### {year_label}")
        lines.append("")
        lines.append(f"| Tier | N | Mean 60d | Median | Hit% | P25 | P75 |")
        lines.append(f"|------|---|----------|--------|------|-----|-----|")
        for t in ["A", "B", "C", "D"]:
            tdf = sub[sub["tier"] == t]
            s = _group_stats(tdf)
            n = s["n"]
            m = f"{s['mean']:+.2f}%" if s["mean"] is not None else "N/A"
            md = f"{s['median']:+.2f}%" if s["median"] is not None else "N/A"
            h = f"{s['hit_pct']:.0f}%" if s["hit_pct"] is not None else "N/A"
            p25 = f"{s['p25']:+.2f}%" if s["p25"] is not None else "N/A"
            p75 = f"{s['p75']:+.2f}%" if s["p75"] is not None else "N/A"
            lines.append(f"| {t} | {n} | {m} | {md} | {h} | {p25} | {p75} |")
        lines.append("")

    # Top 3 Drivers
    lines.append("## Top 3 Drivers of 2024 Inversion")
    lines.append("")
    lines.append("Ranked by spread range (max AB-D spread minus min AB-D spread across")
    lines.append("dimension values). Wider range = dimension explains more variance in")
    lines.append("the AB vs D performance gap.")
    lines.append("")

    for i, drv in enumerate(drivers[:3], 1):
        lines.append(f"### #{i}: `{drv['dimension']}`")
        lines.append("")
        sr_2025 = drv['spread_range_2025']
        suffix = f" (2025: {sr_2025:+.2f}pp)" if sr_2025 else ""
        lines.append(f"- Spread range (2024): **{drv['spread_range_2024']:+.2f}pp**{suffix}")
        lines.append(f"- Worst cell: `{drv['worst_cell_2024']}` → AB-D = {drv['worst_spread_2024']:+.2f}pp")
        lines.append(f"- Best cell: `{drv['best_cell_2024']}` → AB-D = {drv['best_spread_2024']:+.2f}pp")
        lines.append("")

    # Detailed dimension tables
    lines.append("---")
    lines.append("")
    lines.append("## Detailed Dimension Breakouts")
    lines.append("")

    all_dims = list(breakouts_2024.keys())
    for dim_name in all_dims:
        lines.append(f"### {dim_name}")
        lines.append("")

        for year_label, breakouts in [("2024", breakouts_2024), ("2025", breakouts_2025)]:
            data = breakouts.get(dim_name, [])
            if not data:
                continue

            lines.append(f"**{year_label}:**")
            lines.append("")
            lines.append(f"| Value | N(AB) | AB mean | AB hit | N(D) | D mean | D hit | AB-D spread |")
            lines.append(f"|-------|-------|---------|--------|------|--------|-------|-------------|")

            for row in sorted(data, key=lambda r: r["ab_d_spread"] or 0):
                v = row["value"]
                nab = row["n_ab"]
                ab_m = f"{row['ab_mean']:+.2f}%" if row["ab_mean"] is not None else "N/A"
                ab_h = f"{row['ab_hit']:.0f}%" if row["ab_hit"] is not None else "N/A"
                nd = row["n_d"]
                d_m = f"{row['d_mean']:+.2f}%" if row["d_mean"] is not None else "N/A"
                d_h = f"{row['d_hit']:.0f}%" if row["d_hit"] is not None else "N/A"
                sp = f"**{row['ab_d_spread']:+.2f}pp**" if row["ab_d_spread"] is not None else "N/A"
                lines.append(f"| {v} | {nab} | {ab_m} | {ab_h} | {nd} | {d_m} | {d_h} | {sp} |")
            lines.append("")

    # D-tier deep dive: what are the outperforming D names doing?
    lines.append("---")
    lines.append("")
    lines.append("## D-Tier Outperformers Profile (2024)")
    lines.append("")
    d_2024 = df_2024[df_2024["tier"] == "D"].copy()
    d_2024["_ret"] = pd.to_numeric(d_2024["fwd_ret_60d"], errors="coerce")
    d_top = d_2024[d_2024["_ret"] > d_2024["_ret"].quantile(0.75)]
    d_bottom = d_2024[d_2024["_ret"] < d_2024["_ret"].quantile(0.25)]

    lines.append(f"D-tier top quartile (P75+): n={len(d_top)}, mean={d_top['_ret'].mean():+.2f}%")
    lines.append(f"D-tier bottom quartile (P25-): n={len(d_bottom)}, mean={d_bottom['_ret'].mean():+.2f}%")
    lines.append("")

    # Profile: what characterizes top vs bottom D
    for col_name, bucket_fn, label in [
        ("band", None, "Size band"),
        ("catalyst_mode", None, "Catalyst mode"),
        ("mom_state", None, "Momentum state"),
        ("_adv_bucket", lambda d: _adv_buckets(pd.to_numeric(d["adv_dollars"], errors="coerce")), "ADV bucket"),
        ("_dd_abs_bucket", lambda d: _dd_abs_margin_buckets(pd.to_numeric(d["dd_abs_margin"], errors="coerce")), "DD abs margin"),
    ]:
        if bucket_fn:
            d_top[col_name] = bucket_fn(d_top)
            d_bottom[col_name] = bucket_fn(d_bottom)

        lines.append(f"**{label}:**")
        top_dist = d_top[col_name].value_counts(normalize=True).round(3)
        bot_dist = d_bottom[col_name].value_counts(normalize=True).round(3)
        all_vals = sorted(set(top_dist.index) | set(bot_dist.index))
        lines.append(f"| Value | Top Q (winners) | Bottom Q (losers) | Delta |")
        lines.append(f"|-------|-----------------|-------------------|-------|")
        for v in all_vals:
            tp = top_dist.get(v, 0)
            bp = bot_dist.get(v, 0)
            delta = tp - bp
            lines.append(f"| {v} | {tp*100:.1f}% | {bp*100:.1f}% | {delta*100:+.1f}pp |")
        lines.append("")

    # What to test next
    lines.append("---")
    lines.append("")
    lines.append("## What to Test Next")
    lines.append("")
    lines.append("Based on the attribution above, the recommended next step is a")
    lines.append("**membership-preserving weight modifier** targeting the consistently")
    lines.append("bad cell(s) identified in the top drivers. This stays in L3 (sizing)")
    lines.append("and does not change eligibility or tier assignments.")
    lines.append("")
    lines.append("Candidate modifiers to evaluate via walkforward:")
    lines.append("")

    if drivers:
        d1 = drivers[0]
        lines.append(f"1. **{d1['dimension']} penalty**: Weight *= penalty_mult for rows in the")
        lines.append(f"   `{d1['worst_cell_2024']}` bucket (worst AB-D spread: {d1['worst_spread_2024']:+.2f}pp).")
        lines.append(f"   Test with mult = 0.70 and 0.85 via walkforward panel replay.")
    if len(drivers) > 1:
        d2 = drivers[1]
        lines.append(f"2. **{d2['dimension']} penalty**: Same approach for `{d2['worst_cell_2024']}` bucket.")
    lines.append(f"3. **abs_shallow + rel_deep composite**: Weight penalty for rows near")
    lines.append(f"   abs gate but crushed vs XBI (sector-wide drawdown + idiosyncratic weakness).")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="2024 Inversion Attribution Report"
    )
    parser.add_argument(
        "--panel", type=str, default=str(DEFAULT_PANEL),
        help="Walkforward panel CSV path",
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory",
    )
    args = parser.parse_args()

    panel_path = Path(args.panel)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading panel: {panel_path}")
    df = pd.read_csv(panel_path)
    print(f"Rows: {len(df)}, Snapshots: {df['as_of_date'].nunique()}")

    # Add bucketed columns
    df["adv_bucket"] = _adv_buckets(pd.to_numeric(df["adv_dollars"], errors="coerce"))
    df["cost_bucket"] = _cost_buckets(pd.to_numeric(df["est_cost_bps"], errors="coerce"))
    df["dd_abs_margin_bucket"] = _dd_abs_margin_buckets(
        pd.to_numeric(df["dd_abs_margin"], errors="coerce")
    )
    df["dd_rel_margin_bucket"] = _dd_rel_margin_buckets(
        pd.to_numeric(df["dd_rel_margin"], errors="coerce")
    )
    df["optionality_bucket"] = _optionality_buckets(
        pd.to_numeric(df["optionality"], errors="coerce")
    )

    # Define dimensions to break by
    dimensions = [
        "band",
        "catalyst_mode",
        "catalyst_strength",
        "mom_state",
        "adv_bucket",
        "cost_bucket",
        "dd_abs_margin_bucket",
        "dd_rel_margin_bucket",
        "optionality_bucket",
    ]

    # Compute breakouts for 2024 and 2025
    print("Computing breakouts ...")
    breakouts_2024 = {}
    breakouts_2025 = {}
    for dim in dimensions:
        breakouts_2024[dim] = breakout_by_dimension(df, dim, "2024")
        breakouts_2025[dim] = breakout_by_dimension(df, dim, "2025")

    # Rank drivers
    print("Ranking drivers ...")
    drivers = rank_drivers(breakouts_2024, breakouts_2025)

    print("\nTop drivers by spread range:")
    for i, d in enumerate(drivers[:5], 1):
        print(f"  {i}. {d['dimension']}: range={d['spread_range_2024']:+.2f}pp "
              f"(worst: {d['worst_cell_2024']}={d['worst_spread_2024']:+.2f}pp)")

    # Generate report
    report = generate_report(df, breakouts_2024, breakouts_2025, drivers)
    today = date.today().isoformat()
    report_path = output_dir / f"2024_inversion_attribution_{today}.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"\nWrote: {report_path}")


if __name__ == "__main__":
    main()
