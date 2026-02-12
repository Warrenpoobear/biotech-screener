#!/usr/bin/env python3
"""
Eligibility Gate Diagnosis — 2024 D-Tier Outperformance Investigation

Reads the walkforward panel CSV (with ineligible_reasons + first_failed_gate columns),
diagnoses WHY D-tier outperforms B/C in 2024, and runs surgical ablations to
identify which gate(s) are excluding future outperformers.

Sections:
  1. Tier Performance by Year
  2. D-Tier Attribution (ineligible reasons breakdown)
  3. Surgical Ablations (disable gates, recompute tier, measure separation)
  4. Drawdown Margin Analysis (barely-ineligible distribution)

Usage:
    python scripts/diagnose_eligibility_gates.py [options]

    --panel PATH       Panel CSV (default: artifacts/walkforward_panel_diag.csv)
    --output-dir PATH  Output directory (default: artifacts/)
    --suffix TAG       Suffix for output filename
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.calibrate_ruleset_from_panel import retier_row

DEFAULT_PANEL = PROJECT_ROOT / "artifacts" / "walkforward_panel_diag.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts"
DEFAULT_PRICE_CSV = PROJECT_ROOT / "production_data" / "price_history.csv"

# Active ruleset parameters (68b2c45e)
A_FLOOR = 0.60
B_FLOOR = 0.30
CATALYST_NEAR = 120
CATALYST_MID = 180


# =============================================================================
# PANEL READER
# =============================================================================

def read_panel(path: Path) -> List[Dict[str, Any]]:
    """Read the panel CSV and parse numeric columns."""
    rows: List[Dict[str, Any]] = []
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        for raw in csv.DictReader(f):
            row: Dict[str, Any] = dict(raw)
            for col in ("optionality", "catalyst_days_raw",
                        "fwd_ret_20d", "fwd_ret_60d",
                        "fwd_max_dd_20d", "fwd_max_dd_60d", "weight",
                        "drawdown_abs", "drawdown_xbi", "drawdown_rel_xbi",
                        "dd_abs_margin", "dd_rel_margin"):
                v = row.get(col, "")
                if v == "" or v is None:
                    row[col] = None
                else:
                    try:
                        row[col] = float(v)
                    except (ValueError, TypeError):
                        row[col] = None
            rows.append(row)
    return rows


def year_of(row: Dict[str, Any]) -> str:
    return row["as_of_date"][:4]


# =============================================================================
# STATISTICS HELPERS
# =============================================================================

def summary_stats(values: List[float]) -> Dict[str, Any]:
    """Compute summary stats for a list of floats."""
    if not values:
        return {"count": 0, "mean": None, "median": None, "p25": None, "p75": None, "hit_pct": None}
    sv = sorted(values)
    n = len(sv)
    p25 = sv[int(n * 0.25)] if n > 1 else sv[0]
    p75 = sv[int(n * 0.75)] if n > 1 else sv[0]
    hit = sum(1 for v in values if v > 0)
    return {
        "count": n,
        "mean": round(mean(values), 2),
        "median": round(median(values), 2),
        "p25": round(p25, 2),
        "p75": round(p75, 2),
        "hit_pct": round(hit / n * 100, 1),
    }


def fmt(v, suffix="%", default="n/a"):
    """Format a numeric value for display."""
    if v is None:
        return default
    if suffix == "%":
        return f"{v:+.2f}%"
    return f"{v:.2f}{suffix}"


# =============================================================================
# SECTION 1: TIER PERFORMANCE BY YEAR
# =============================================================================

def section_tier_performance(rows: List[Dict[str, Any]]) -> str:
    """Tier performance breakdown by year."""
    lines = ["# Section 1: Tier Performance by Year", ""]

    for yr in sorted(set(year_of(r) for r in rows)):
        yr_rows = [r for r in rows if year_of(r) == yr]
        lines.append(f"## {yr}  ({len(yr_rows)} rows)")
        lines.append("")

        tier_data: Dict[str, List[float]] = defaultdict(list)
        for r in yr_rows:
            tier = r.get("tier", "?")
            ret = r.get("fwd_ret_60d")
            if ret is not None:
                tier_data[tier].append(ret)

        lines.append(f"{'Tier':<6} {'N':>5} {'Mean':>8} {'Med':>8} {'P25':>8} {'P75':>8} {'Hit%':>7}")
        lines.append("-" * 52)
        ab_rets, cd_rets = [], []
        for tier in ["A", "B", "C", "D"]:
            s = summary_stats(tier_data.get(tier, []))
            lines.append(
                f"{tier:<6} {s['count']:>5} "
                f"{fmt(s['mean']):>8} {fmt(s['median']):>8} "
                f"{fmt(s['p25']):>8} {fmt(s['p75']):>8} "
                f"{fmt(s['hit_pct'], suffix=''):>6}%"
            )
            if tier in ("A", "B"):
                ab_rets.extend(tier_data.get(tier, []))
            else:
                cd_rets.extend(tier_data.get(tier, []))

        ab_med = median(ab_rets) if ab_rets else None
        cd_med = median(cd_rets) if cd_rets else None
        ab_mean = mean(ab_rets) if ab_rets else None
        cd_mean = mean(cd_rets) if cd_rets else None
        sep_med = round(ab_med - cd_med, 2) if ab_med is not None and cd_med is not None else None
        sep_mean = round(ab_mean - cd_mean, 2) if ab_mean is not None and cd_mean is not None else None
        lines.append("")
        lines.append(f"AB-CD separation (median): {fmt(sep_med)}")
        lines.append(f"AB-CD separation (mean):   {fmt(sep_mean)}")
        lines.append("")

    return "\n".join(lines)


# =============================================================================
# SECTION 2: D-TIER ATTRIBUTION
# =============================================================================

def section_dtier_attribution(rows: List[Dict[str, Any]]) -> str:
    """D-tier breakdown by ineligible_reasons and first_failed_gate."""
    lines = ["# Section 2: D-Tier Attribution (Ineligible Reasons)", ""]

    d_rows = [r for r in rows if r.get("tier") == "D"]
    lines.append(f"Total D-tier rows: {len(d_rows)}")
    lines.append("")

    for yr in sorted(set(year_of(r) for r in rows)):
        yr_d = [r for r in d_rows if year_of(r) == yr]
        lines.append(f"## {yr} — D-Tier ({len(yr_d)} rows)")
        lines.append("")

        # Group by ineligible_reasons
        reason_groups: Dict[str, List[float]] = defaultdict(list)
        for r in yr_d:
            reason = r.get("ineligible_reasons", "") or "(empty)"
            ret = r.get("fwd_ret_60d")
            if ret is not None:
                reason_groups[reason].append(ret)

        lines.append(f"### By ineligible_reasons")
        lines.append(f"{'Reason':<35} {'N':>5} {'Mean':>8} {'Med':>8} {'P25':>8} {'P75':>8} {'Hit%':>7}")
        lines.append("-" * 80)
        for reason in sorted(reason_groups.keys(), key=lambda k: -len(reason_groups[k])):
            s = summary_stats(reason_groups[reason])
            lines.append(
                f"{reason:<35} {s['count']:>5} "
                f"{fmt(s['mean']):>8} {fmt(s['median']):>8} "
                f"{fmt(s['p25']):>8} {fmt(s['p75']):>8} "
                f"{fmt(s['hit_pct'], suffix=''):>6}%"
            )
        lines.append("")

        # Group by first_failed_gate
        gate_groups: Dict[str, List[float]] = defaultdict(list)
        for r in yr_d:
            gate = r.get("first_failed_gate", "") or "(empty)"
            ret = r.get("fwd_ret_60d")
            if ret is not None:
                gate_groups[gate].append(ret)

        lines.append(f"### By first_failed_gate")
        lines.append(f"{'Gate':<35} {'N':>5} {'Mean':>8} {'Med':>8} {'P25':>8} {'P75':>8} {'Hit%':>7}")
        lines.append("-" * 80)
        for gate in sorted(gate_groups.keys(), key=lambda k: -len(gate_groups[k])):
            s = summary_stats(gate_groups[gate])
            lines.append(
                f"{gate:<35} {s['count']:>5} "
                f"{fmt(s['mean']):>8} {fmt(s['median']):>8} "
                f"{fmt(s['p25']):>8} {fmt(s['p75']):>8} "
                f"{fmt(s['hit_pct'], suffix=''):>6}%"
            )
        lines.append("")

    return "\n".join(lines)


# =============================================================================
# SECTION 3: SURGICAL ABLATIONS
# =============================================================================

def compute_separation(rows: List[Dict[str, Any]], tier_col: str = "tier") -> Dict[str, Any]:
    """Compute AB-CD separation from rows with a tier column."""
    ab, cd = [], []
    tier_counts: Counter = Counter()
    for r in rows:
        tier = r.get(tier_col, "?")
        tier_counts[tier] += 1
        ret = r.get("fwd_ret_60d")
        if ret is None:
            continue
        if tier in ("A", "B"):
            ab.append(ret)
        elif tier in ("C", "D"):
            cd.append(ret)

    ab_mean = mean(ab) if ab else None
    cd_mean = mean(cd) if cd else None
    ab_med = median(ab) if ab else None
    cd_med = median(cd) if cd else None
    sep_mean = round(ab_mean - cd_mean, 2) if ab_mean is not None and cd_mean is not None else None
    sep_med = round(ab_med - cd_med, 2) if ab_med is not None and cd_med is not None else None

    return {
        "tier_counts": dict(tier_counts),
        "ab_mean": round(ab_mean, 2) if ab_mean is not None else None,
        "cd_mean": round(cd_mean, 2) if cd_mean is not None else None,
        "sep_mean": sep_mean,
        "sep_med": sep_med,
        "n_ab": len(ab),
        "n_cd": len(cd),
    }


def run_ablation(
    rows: List[Dict[str, Any]],
    label: str,
    eligible_override_fn,
) -> Dict[str, Any]:
    """Run an ablation: override eligible for matching rows, retier, measure separation.

    eligible_override_fn(row) -> bool: returns True if this row should be
    force-set to eligible=1 for the ablation.
    """
    ablated = []
    n_flipped = 0
    for r in rows:
        row = dict(r)  # shallow copy
        if eligible_override_fn(row):
            row["eligible"] = "1"
            n_flipped += 1
        # Retier using the row's (possibly overridden) eligible flag
        new_tier = retier_row(
            row,
            a_floor=A_FLOOR,
            b_floor=B_FLOOR,
            catalyst_near_days=CATALYST_NEAR,
            catalyst_mid_days=CATALYST_MID,
        )
        row["tier_ablated"] = new_tier
        ablated.append(row)

    sep = compute_separation(ablated, tier_col="tier_ablated")
    sep["label"] = label
    sep["n_flipped"] = n_flipped
    return sep


def section_ablations(rows: List[Dict[str, Any]]) -> str:
    """Run surgical ablations and report results."""
    lines = ["# Section 3: Surgical Ablations", ""]

    # Baseline
    baseline = compute_separation(rows)
    lines.append(f"Baseline tier distribution: {baseline['tier_counts']}")
    lines.append(f"Baseline AB-CD sep (mean): {fmt(baseline['sep_mean'])}")
    lines.append(f"Baseline AB-CD sep (med):  {fmt(baseline['sep_med'])}")
    lines.append("")

    for yr in sorted(set(year_of(r) for r in rows)):
        yr_rows = [r for r in rows if year_of(r) == yr]
        lines.append(f"## {yr}  ({len(yr_rows)} rows)")
        lines.append("")

        # Baseline for this year
        bl = compute_separation(yr_rows)
        lines.append(f"Baseline: {bl['tier_counts']}  sep_mean={fmt(bl['sep_mean'])}  sep_med={fmt(bl['sep_med'])}")
        lines.append("")

        # Ablation 1: Disable ALL gating
        abl1 = run_ablation(
            yr_rows,
            "disable_all_gates",
            lambda r: r.get("tier") == "D",
        )

        # Ablation 2: Disable drawdown gate only (rows where ONLY deep_drawdown)
        def _dd_only(r):
            if r.get("tier") != "D":
                return False
            reasons = r.get("ineligible_reasons", "")
            return reasons == "deep_drawdown"

        abl2 = run_ablation(yr_rows, "disable_drawdown_only", _dd_only)

        # Ablation 3: Disable each gate individually
        GATES = ["deep_drawdown", "sev3", "fundamental_red_flag", "adv_fail"]
        gate_ablations = []
        for gate in GATES:
            def _gate_match(r, g=gate):
                if r.get("tier") != "D":
                    return False
                ffg = r.get("first_failed_gate", "")
                return ffg == g
            abl = run_ablation(yr_rows, f"disable_{gate}", _gate_match)
            gate_ablations.append(abl)

        # Report
        lines.append(f"{'Ablation':<30} {'Flipped':>7} {'A':>4} {'B':>4} {'C':>4} {'D':>4} {'Sep(mean)':>10} {'Sep(med)':>10} {'Δmean':>8}")
        lines.append("-" * 96)

        for abl in [abl1, abl2] + gate_ablations:
            tc = abl["tier_counts"]
            delta = round(abl["sep_mean"] - bl["sep_mean"], 2) if abl["sep_mean"] is not None and bl["sep_mean"] is not None else None
            lines.append(
                f"{abl['label']:<30} {abl['n_flipped']:>7} "
                f"{tc.get('A', 0):>4} {tc.get('B', 0):>4} {tc.get('C', 0):>4} {tc.get('D', 0):>4} "
                f"{fmt(abl['sep_mean']):>10} {fmt(abl['sep_med']):>10} "
                f"{fmt(delta):>8}"
            )
        lines.append("")

    return "\n".join(lines)


# =============================================================================
# SECTION 4: DRAWDOWN MARGIN ANALYSIS
# =============================================================================

def section_drawdown_margins(rows: List[Dict[str, Any]]) -> str:
    """Analyze drawdown margins for D-tier rows."""
    lines = ["# Section 4: Drawdown Margin Analysis (D-Tier)", ""]

    d_rows = [r for r in rows if r.get("tier") == "D"]
    lines.append(f"Total D-tier rows: {len(d_rows)}")
    lines.append("")

    for yr in sorted(set(year_of(r) for r in rows)):
        yr_d = [r for r in d_rows if year_of(r) == yr]
        lines.append(f"## {yr} — D-Tier Drawdown Margins ({len(yr_d)} rows)")
        lines.append("")

        # Only look at rows where drawdown is a reason
        dd_rows = [r for r in yr_d if "deep_drawdown" in (r.get("ineligible_reasons", "") or "")]

        abs_margins = [r["dd_abs_margin"] for r in dd_rows if r.get("dd_abs_margin") is not None]
        rel_margins = [r["dd_rel_margin"] for r in dd_rows if r.get("dd_rel_margin") is not None]

        lines.append(f"Rows with deep_drawdown reason: {len(dd_rows)}")
        lines.append("")

        if abs_margins:
            # Margins are in raw decimal: -0.15 = 15pp below gate
            lines.append(f"dd_abs_margin distribution (N={len(abs_margins)}):")
            lines.append(f"  Mean: {mean(abs_margins)*100:.1f}pp  Median: {median(abs_margins)*100:.1f}pp")
            sv = sorted(abs_margins)
            p25 = sv[int(len(sv)*0.25)] if len(sv) > 1 else sv[0]
            p75 = sv[int(len(sv)*0.75)] if len(sv) > 1 else sv[0]
            lines.append(f"  P25: {p25*100:.1f}pp  P75: {p75*100:.1f}pp")
            # Barely ineligible: margin within 5pp of threshold (0.05 in decimal)
            barely = [m for m in abs_margins if m > -0.05]
            lines.append(f"  Barely ineligible (margin > -5pp): {len(barely)} / {len(abs_margins)} ({len(barely)/len(abs_margins)*100:.1f}%)")
            very_close = [m for m in abs_margins if m > -0.02]
            lines.append(f"  Very close (margin > -2pp): {len(very_close)} / {len(abs_margins)} ({len(very_close)/len(abs_margins)*100:.1f}%)")
            # Histogram
            buckets = [(0.0, -0.02, "0-2pp"), (-0.02, -0.05, "2-5pp"),
                       (-0.05, -0.10, "5-10pp"), (-0.10, -0.20, "10-20pp"),
                       (-0.20, -0.50, "20-50pp"), (-0.50, -999, ">50pp")]
            lines.append(f"  Histogram:")
            for lo, hi, label in buckets:
                count = sum(1 for m in abs_margins if m <= lo and m > hi)
                pct = count / len(abs_margins) * 100
                lines.append(f"    {label:>8}: {count:>5} ({pct:.1f}%)")
        else:
            lines.append("dd_abs_margin: no data")
        lines.append("")

        if rel_margins:
            lines.append(f"dd_rel_margin distribution (N={len(rel_margins)}):")
            lines.append(f"  Mean: {mean(rel_margins)*100:.1f}pp  Median: {median(rel_margins)*100:.1f}pp")
            barely = [m for m in rel_margins if m > -0.05]
            lines.append(f"  Barely ineligible (margin > -5pp): {len(barely)} / {len(rel_margins)} ({len(barely)/len(rel_margins)*100:.1f}%)")
        else:
            lines.append("dd_rel_margin: no data")
        lines.append("")

        # Performance split: barely-ineligible vs deep-ineligible drawdown rows
        if dd_rows:
            close_rows = [r for r in dd_rows if r.get("dd_abs_margin") is not None and r["dd_abs_margin"] > -0.10]
            deep_rows = [r for r in dd_rows if r.get("dd_abs_margin") is not None and r["dd_abs_margin"] <= -0.10]

            close_rets = [r["fwd_ret_60d"] for r in close_rows if r.get("fwd_ret_60d") is not None]
            deep_rets = [r["fwd_ret_60d"] for r in deep_rows if r.get("fwd_ret_60d") is not None]

            sc = summary_stats(close_rets)
            sd = summary_stats(deep_rets)
            lines.append(f"Performance split (deep_drawdown rows only):")
            lines.append(f"  Near gate  (margin > -10pp): N={sc['count']}, mean={fmt(sc['mean'])}, med={fmt(sc['median'])}, hit={fmt(sc['hit_pct'], suffix='')}")
            lines.append(f"  Far below  (margin <= -10pp): N={sd['count']}, mean={fmt(sd['mean'])}, med={fmt(sd['median'])}, hit={fmt(sd['hit_pct'], suffix='')}")
            lines.append("")

    return "\n".join(lines)


# =============================================================================
# SECTION 5: D-TIER REVERSAL FILTER (trail_ret_20d)
# =============================================================================

def load_price_index(price_csv: Path) -> Dict[str, Dict[str, float]]:
    """Load price_history.csv into {TICKER: {date_str: close}} index.

    Returns dates as strings (YYYY-MM-DD) for easy lookup.
    """
    prices: Dict[str, Dict[str, float]] = {}
    with open(price_csv, "r", newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            ticker = row.get("ticker", "").strip().upper()
            date_str = row.get("date", "").strip()[:10]
            close_str = row.get("close", "").strip()
            if not ticker or not date_str or not close_str:
                continue
            try:
                close = float(close_str)
            except (ValueError, TypeError):
                continue
            if ticker not in prices:
                prices[ticker] = {}
            prices[ticker][date_str] = close
    return prices


def compute_trailing_return(
    prices: Dict[str, Dict[str, float]],
    ticker: str,
    as_of_date: str,
    lookback_bars: int = 20,
    tolerance_days: int = 5,
) -> Optional[float]:
    """Compute trailing return from price history.

    Finds the close on as_of_date (±tolerance), then the close lookback_bars
    trading days earlier, returns (end/start - 1).
    """
    from datetime import date as _date, timedelta

    ticker_prices = prices.get(ticker.upper())
    if not ticker_prices:
        return None

    ref = _date.fromisoformat(as_of_date)

    # Find end price (nearest to as_of_date)
    end_price = None
    for delta in range(0, tolerance_days + 1):
        for d in [ref - timedelta(days=delta), ref + timedelta(days=delta)]:
            ds = d.isoformat()
            if ds in ticker_prices:
                end_price = ticker_prices[ds]
                ref = d  # anchor to actual date found
                break
        if end_price is not None:
            break
    if end_price is None or end_price <= 0:
        return None

    # Get sorted dates <= ref, pick the one lookback_bars back
    valid_dates = sorted(d for d in ticker_prices if d <= ref.isoformat())
    if len(valid_dates) < lookback_bars + 1:
        return None

    start_date = valid_dates[-(lookback_bars + 1)]
    start_price = ticker_prices[start_date]
    if start_price <= 0:
        return None

    return round((end_price / start_price) - 1.0, 6)


def enrich_panel_with_trailing_returns(
    rows: List[Dict[str, Any]],
    prices: Dict[str, Dict[str, float]],
    lookback_bars: int = 20,
) -> int:
    """Add trail_ret_20d to each panel row in-place. Returns count of populated rows."""
    n_filled = 0
    for r in rows:
        ticker = r.get("ticker", "")
        as_of = r.get("as_of_date", "")
        ret = compute_trailing_return(prices, ticker, as_of, lookback_bars)
        r["trail_ret_20d"] = ret
        if ret is not None:
            n_filled += 1
    return n_filled


def section_reversal_filter(rows: List[Dict[str, Any]]) -> str:
    """D-tier reversal filter: split by trail_ret_20d > 0 vs <= 0."""
    lines = ["# Section 5: D-Tier Reversal Filter (trail_ret_20d)", ""]

    d_rows = [r for r in rows if r.get("tier") == "D"]
    has_trail = [r for r in d_rows if r.get("trail_ret_20d") is not None]
    lines.append(f"D-tier rows with trail_ret_20d: {len(has_trail)} / {len(d_rows)}")
    lines.append("")

    for yr in sorted(set(year_of(r) for r in rows)):
        yr_d = [r for r in d_rows if year_of(r) == yr and r.get("trail_ret_20d") is not None]
        lines.append(f"## {yr} — D-Tier Reversal Split ({len(yr_d)} rows with trail_ret_20d)")
        lines.append("")

        # Split by sign
        up = [r for r in yr_d if r["trail_ret_20d"] > 0]
        down = [r for r in yr_d if r["trail_ret_20d"] <= 0]

        up_rets = [r["fwd_ret_60d"] for r in up if r.get("fwd_ret_60d") is not None]
        down_rets = [r["fwd_ret_60d"] for r in down if r.get("fwd_ret_60d") is not None]

        su = summary_stats(up_rets)
        sd = summary_stats(down_rets)

        lines.append(f"{'Group':<25} {'N':>5} {'Mean':>8} {'Med':>8} {'P25':>8} {'P75':>8} {'Hit%':>7}")
        lines.append("-" * 70)
        lines.append(
            f"{'trail_ret_20d > 0':<25} {su['count']:>5} "
            f"{fmt(su['mean']):>8} {fmt(su['median']):>8} "
            f"{fmt(su['p25']):>8} {fmt(su['p75']):>8} "
            f"{fmt(su['hit_pct'], suffix=''):>6}%"
        )
        lines.append(
            f"{'trail_ret_20d <= 0':<25} {sd['count']:>5} "
            f"{fmt(sd['mean']):>8} {fmt(sd['median']):>8} "
            f"{fmt(sd['p25']):>8} {fmt(sd['p75']):>8} "
            f"{fmt(sd['hit_pct'], suffix=''):>6}%"
        )
        delta_mean = round(su["mean"] - sd["mean"], 2) if su["mean"] is not None and sd["mean"] is not None else None
        delta_med = round(su["median"] - sd["median"], 2) if su["median"] is not None and sd["median"] is not None else None
        lines.append("")
        lines.append(f"Delta (up - down) mean: {fmt(delta_mean)}  median: {fmt(delta_med)}")
        lines.append("")

        # Finer granularity: quintiles of trail_ret_20d
        if yr_d:
            sorted_by_trail = sorted(yr_d, key=lambda r: r["trail_ret_20d"])
            n = len(sorted_by_trail)
            q_size = n // 5
            if q_size > 0:
                lines.append(f"### trail_ret_20d quintiles")
                lines.append(f"{'Quintile':<12} {'Range':>20} {'N':>5} {'Mean_60d':>9} {'Med_60d':>9} {'Hit%':>7}")
                lines.append("-" * 70)
                for qi in range(5):
                    start = qi * q_size
                    end = (qi + 1) * q_size if qi < 4 else n
                    q_rows = sorted_by_trail[start:end]
                    q_rets = [r["fwd_ret_60d"] for r in q_rows if r.get("fwd_ret_60d") is not None]
                    qs = summary_stats(q_rets)
                    lo = q_rows[0]["trail_ret_20d"]
                    hi = q_rows[-1]["trail_ret_20d"]
                    lines.append(
                        f"Q{qi+1:<11} {lo*100:>+8.1f}% to {hi*100:>+6.1f}% "
                        f"{qs['count']:>5} {fmt(qs['mean']):>9} {fmt(qs['median']):>9} "
                        f"{fmt(qs['hit_pct'], suffix=''):>6}%"
                    )
                lines.append("")

    return "\n".join(lines)


# =============================================================================
# SECTION 6: RELATIVE-DRAWDOWN RESCUE AUDIT
# =============================================================================

def section_relative_rescue(rows: List[Dict[str, Any]]) -> str:
    """Audit relative-drawdown rescue signal for D-tier rows.

    Three sub-analyses:
      6a) rescued_by_rel context (eligible rows rescued vs not)
      6b) 2×2 abs/rel margin grid for D-tier
      6c) Hypothetical rescue sweep: relax rel margin → retier → measure sep
    """
    lines = ["# Section 6: Relative-Drawdown Rescue Audit", ""]

    d_rows = [r for r in rows if r.get("tier") == "D"]
    elig_rows = [r for r in rows if str(r.get("eligible", "0")) == "1"]

    # ---- 6a: rescued_by_rel context (eligible rows) ----
    lines.append("## 6a: rescued_by_rel Performance (Eligible Rows)")
    lines.append("")
    lines.append("Note: rescued_by_rel is 0 for ALL D-tier rows (require_both gate mode).")
    lines.append("This section examines rescued eligible rows as context for the signal's value.")
    lines.append("")

    for yr in sorted(set(year_of(r) for r in rows)):
        yr_elig = [r for r in elig_rows if year_of(r) == yr]
        rescued = [r for r in yr_elig if r.get("rescued_by_rel") == "1"]
        not_rescued = [r for r in yr_elig if r.get("rescued_by_rel") != "1"]

        r_rets = [r["fwd_ret_60d"] for r in rescued if r.get("fwd_ret_60d") is not None]
        nr_rets = [r["fwd_ret_60d"] for r in not_rescued if r.get("fwd_ret_60d") is not None]

        sr = summary_stats(r_rets)
        snr = summary_stats(nr_rets)
        lines.append(f"### {yr}")
        lines.append(f"  Rescued (abs breach, rel saved):  N={sr['count']}, mean={fmt(sr['mean'])}, med={fmt(sr['median'])}, hit={fmt(sr['hit_pct'], suffix='')}")
        lines.append(f"  Not rescued (no abs breach):      N={snr['count']}, mean={fmt(snr['mean'])}, med={fmt(snr['median'])}, hit={fmt(snr['hit_pct'], suffix='')}")
        lines.append("")

    # ---- 6b: 2×2 abs/rel margin grid for D-tier ----
    lines.append("## 6b: 2×2 Abs/Rel Margin Grid (D-Tier Only)")
    lines.append("")
    lines.append("Splits D-tier rows at -10pp margin on each axis.")
    lines.append("'Shallow' = margin > -0.10 (closer to gate), 'Deep' = margin <= -0.10")
    lines.append("")

    ABS_SPLIT = -0.10
    REL_SPLIT = -0.10

    for yr in sorted(set(year_of(r) for r in rows)):
        yr_d = [r for r in d_rows if year_of(r) == yr
                and r.get("dd_abs_margin") is not None
                and r.get("dd_rel_margin") is not None]
        lines.append(f"### {yr} ({len(yr_d)} D-tier rows with both margins)")
        lines.append("")

        grid: Dict[str, List[float]] = {
            "abs_shallow+rel_shallow": [],
            "abs_shallow+rel_deep": [],
            "abs_deep+rel_shallow": [],
            "abs_deep+rel_deep": [],
        }

        for r in yr_d:
            abs_m = r["dd_abs_margin"]
            rel_m = r["dd_rel_margin"]
            ret = r.get("fwd_ret_60d")
            if ret is None:
                continue

            abs_label = "abs_shallow" if abs_m > ABS_SPLIT else "abs_deep"
            rel_label = "rel_shallow" if rel_m > REL_SPLIT else "rel_deep"
            grid[f"{abs_label}+{rel_label}"].append(ret)

        lines.append(f"{'Cell':<30} {'N':>5} {'Mean':>8} {'Med':>8} {'Hit%':>7}")
        lines.append("-" * 62)
        for cell in ["abs_shallow+rel_shallow", "abs_shallow+rel_deep",
                      "abs_deep+rel_shallow", "abs_deep+rel_deep"]:
            s = summary_stats(grid[cell])
            lines.append(
                f"{cell:<30} {s['count']:>5} "
                f"{fmt(s['mean']):>8} {fmt(s['median']):>8} "
                f"{fmt(s['hit_pct'], suffix=''):>6}%"
            )
        lines.append("")

        # Is rel_shallow better than rel_deep (holding abs constant)?
        for abs_level in ["abs_shallow", "abs_deep"]:
            s_vals = grid[f"{abs_level}+rel_shallow"]
            d_vals = grid[f"{abs_level}+rel_deep"]
            if s_vals and d_vals:
                delta_mean = round(mean(s_vals) - mean(d_vals), 2)
                delta_med = round(median(s_vals) - median(d_vals), 2)
                lines.append(
                    f"  {abs_level}: rel_shallow − rel_deep = "
                    f"Δmean={fmt(delta_mean)}, Δmed={fmt(delta_med)}"
                )
        lines.append("")

    # ---- 6c: Hypothetical rescue sweep ----
    lines.append("## 6c: Hypothetical Rescue Sweep")
    lines.append("")
    lines.append("What if D-tier rows with dd_rel_margin above threshold were rescued (set eligible=1)?")
    lines.append("Retiers using current ruleset params, measures AB-CD separation.")
    lines.append("")

    RESCUE_THRESHOLDS = [-0.02, -0.05, -0.10, -0.15, -0.20]

    for yr in sorted(set(year_of(r) for r in rows)):
        yr_rows = [r for r in rows if year_of(r) == yr]
        yr_d = [r for r in yr_rows if r.get("tier") == "D"]

        # Baseline
        bl = compute_separation(yr_rows)
        lines.append(f"### {yr} (baseline sep_mean={fmt(bl['sep_mean'])}, sep_med={fmt(bl['sep_med'])})")
        lines.append("")
        lines.append(f"{'Rescue threshold':<20} {'Rescued':>7} {'A':>4} {'B':>4} {'C':>4} {'D':>4} {'Sep(mean)':>10} {'Sep(med)':>10} {'Δmean':>8}")
        lines.append("-" * 88)

        for thresh in RESCUE_THRESHOLDS:
            def _rescue_fn(r, t=thresh):
                if r.get("tier") != "D":
                    return False
                rel_m = r.get("dd_rel_margin")
                if rel_m is None:
                    return False
                return rel_m > t

            abl = run_ablation(yr_rows, f"rel_margin>{thresh:.2f}", _rescue_fn)
            tc = abl["tier_counts"]
            delta = round(abl["sep_mean"] - bl["sep_mean"], 2) if abl["sep_mean"] is not None and bl["sep_mean"] is not None else None
            lines.append(
                f"dd_rel_margin > {thresh:>5.2f}  {abl['n_flipped']:>7} "
                f"{tc.get('A', 0):>4} {tc.get('B', 0):>4} {tc.get('C', 0):>4} {tc.get('D', 0):>4} "
                f"{fmt(abl['sep_mean']):>10} {fmt(abl['sep_med']):>10} "
                f"{fmt(delta):>8}"
            )

        # Also report performance of rescued rows at each threshold
        lines.append("")
        lines.append(f"{'Rescue threshold':<20} {'Rescued':>7} {'Rescued mean':>12} {'Rescued med':>12} {'Rescued hit%':>12}")
        lines.append("-" * 70)
        for thresh in RESCUE_THRESHOLDS:
            rescued_rets = [
                r["fwd_ret_60d"] for r in yr_d
                if r.get("dd_rel_margin") is not None
                and r["dd_rel_margin"] > thresh
                and r.get("fwd_ret_60d") is not None
            ]
            s = summary_stats(rescued_rets)
            lines.append(
                f"dd_rel_margin > {thresh:>5.2f}  {s['count']:>7} "
                f"{fmt(s['mean']):>12} {fmt(s['median']):>12} "
                f"{fmt(s['hit_pct'], suffix=''):>11}%"
            )
        lines.append("")

    return "\n".join(lines)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Eligibility Gate Diagnosis")
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--suffix", default="")
    parser.add_argument("--price-csv", type=Path, default=DEFAULT_PRICE_CSV,
                        help="Price history CSV for trailing return computation")
    args = parser.parse_args()

    print(f"Reading panel: {args.panel}")
    rows = read_panel(args.panel)
    print(f"  Total rows: {len(rows)}")
    print(f"  Date range: {min(r['as_of_date'] for r in rows)} to {max(r['as_of_date'] for r in rows)}")
    print(f"  Years: {sorted(set(year_of(r) for r in rows))}")

    # Check for required columns
    sample = rows[0] if rows else {}
    has_reasons = "ineligible_reasons" in sample
    has_gate = "first_failed_gate" in sample
    print(f"  ineligible_reasons column: {'YES' if has_reasons else 'MISSING'}")
    print(f"  first_failed_gate column:  {'YES' if has_gate else 'MISSING'}")
    if not has_reasons:
        print("WARNING: ineligible_reasons column missing — attribution will be limited")

    # Load price history for trailing return computation
    print(f"Loading prices: {args.price_csv}")
    prices = load_price_index(args.price_csv)
    print(f"  Tickers with prices: {len(prices)}")
    n_filled = enrich_panel_with_trailing_returns(rows, prices, lookback_bars=20)
    print(f"  trail_ret_20d populated: {n_filled} / {len(rows)} ({n_filled/len(rows)*100:.1f}%)")

    # Build report
    sections = []
    sections.append(section_tier_performance(rows))
    sections.append(section_dtier_attribution(rows))
    sections.append(section_ablations(rows))
    sections.append(section_drawdown_margins(rows))
    sections.append(section_reversal_filter(rows))
    sections.append(section_relative_rescue(rows))

    # Header
    today = datetime.now().strftime("%Y-%m-%d")
    header = [
        f"# Eligibility Gate Diagnosis — {today}",
        f"",
        f"Panel: {args.panel.name}",
        f"Rows: {len(rows)}",
        f"Date range: {min(r['as_of_date'] for r in rows)} to {max(r['as_of_date'] for r in rows)}",
        f"Ruleset params: a_floor={A_FLOOR}, b_floor={B_FLOOR}, catalyst_near={CATALYST_NEAR}d, catalyst_mid={CATALYST_MID}d",
        f"",
        f"---",
        f"",
    ]

    report = "\n".join(header) + "\n\n".join(sections)

    # Write
    suffix = f"_{args.suffix}" if args.suffix else ""
    out_path = args.output_dir / f"eligibility_gate_diagnosis{suffix}.md"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"\nReport written to: {out_path}")


if __name__ == "__main__":
    main()
