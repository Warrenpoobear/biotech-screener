#!/usr/bin/env python3
"""Promotion-grade anchor replay on monthly/weekly archive grid.

Tests A (alpha_cohort) vs E (optionality) vs E+blend variants on the full
archive history with matured forward returns.

Configs:
  A:            alpha_cohort_pct anchor (current production)
  E:            clinical_optionality_pct_dev anchor (no alpha)
  E_blend_0.02: (1-0.02)*opt_pct + 0.02*alpha_pct
  E_blend_0.05: (1-0.05)*opt_pct + 0.05*alpha_pct
  E_blend_0.10: (1-0.10)*opt_pct + 0.10*alpha_pct

Promotion criteria (pre-registered):
  ΔIC(60d)  ≥ +0.005 vs A
  ΔSpread(60d) ≥ +1.0% AND bootstrap 95% CI lower bound > 0
  Turnover not worse by > 0.05
  (Hit-rate, monotonicity reported but not binding)

Usage:
    python scripts/research/anchor_replay.py \
        --date-from 2020-01-31 --date-to 2026-02-28 \
        --date-grid monthly --horizons 20,60 --top-k 20

    python scripts/research/anchor_replay.py \
        --date-grid weekly --date-from 2024-01-01 --date-to 2025-12-31
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import math
import random
import statistics
import sys
import tarfile
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from common.ranking_utils import backfill_columns, safe_float
from decision_engine import DecisionRuleset, compute_actionable_sort_key
from module_5_alpha_cohort import attach_alpha_scores, compute_alpha_raw
from scripts.build_alpha_cohort_table import backfill_clinical_z_tier
from scripts.build_alpha_cohort_table_oos import build_oos_table

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Archive loading
# ---------------------------------------------------------------------------

def discover_archives(
    archive_dir: Path,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    date_grid: str = "monthly",
) -> List[Tuple[str, Path]]:
    """Find archives in date range, optionally filtering to monthly grid.

    date_grid:
      "all"     — use every archive in the range
      "monthly" — keep only one archive per calendar month (prefer last day)
      "weekly"  — keep only one archive per ISO week (prefer last day)
    """
    import re
    pattern = re.compile(r"(\d{4}-\d{2}-\d{2})\.tar\.gz$")
    all_archives: List[Tuple[str, Path]] = []
    for p in sorted(archive_dir.glob("*.tar.gz")):
        m = pattern.search(p.name)
        if not m:
            continue
        d = m.group(1)
        if date_from and d < date_from:
            continue
        if date_to and d > date_to:
            continue
        all_archives.append((d, p))
    all_archives.sort(key=lambda x: x[0])

    if date_grid == "all":
        return all_archives

    if date_grid == "monthly":
        # Keep last archive per YYYY-MM
        by_month: Dict[str, Tuple[str, Path]] = {}
        for d, p in all_archives:
            ym = d[:7]
            by_month[ym] = (d, p)  # last one wins
        return sorted(by_month.values(), key=lambda x: x[0])

    if date_grid == "weekly":
        # Keep last archive per ISO year-week
        from datetime import date as dt_date
        by_week: Dict[str, Tuple[str, Path]] = {}
        for d, p in all_archives:
            parts = d.split("-")
            iso = dt_date(int(parts[0]), int(parts[1]), int(parts[2])).isocalendar()
            wk = f"{iso[0]:04d}-W{iso[1]:02d}"
            by_week[wk] = (d, p)
        return sorted(by_week.values(), key=lambda x: x[0])

    return all_archives


def load_archive_rankings(tar_path: Path, date_str: str) -> List[Dict[str, str]]:
    """Extract rankings.csv from an archive tarball."""
    csv_name = f"{date_str}/rankings.csv"
    try:
        with tarfile.open(tar_path, "r:gz") as tf:
            member = tf.getmember(csv_name)
            f = tf.extractfile(member)
            if f is None:
                return []
            text = f.read().decode("utf-8")
            reader = csv.DictReader(text.splitlines())
            return list(reader)
    except (KeyError, tarfile.TarError):
        return []


# ---------------------------------------------------------------------------
# Price loading
# ---------------------------------------------------------------------------

def load_price_series(csv_path: Path) -> Dict[str, Dict[str, float]]:
    """Load price_history.csv -> {ticker: {date_str: close}}."""
    prices: Dict[str, Dict[str, float]] = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ticker = (row.get("ticker") or "").strip().upper()
            close_str = (row.get("close") or "").strip()
            date_str = (row.get("date") or "").strip()
            if not ticker or not close_str or not date_str:
                continue
            try:
                close = float(close_str)
            except (ValueError, TypeError):
                continue
            prices.setdefault(ticker, {})[date_str] = close
    return prices


def compute_forward_return(
    ticker_prices: Dict[str, float],
    sorted_dates: List[str],
    trade_date: str,
    horizon: int,
) -> Optional[float]:
    """Forward return over horizon trading days from trade_date."""
    try:
        idx = sorted_dates.index(trade_date)
    except ValueError:
        return None
    end_idx = idx + horizon
    if end_idx >= len(sorted_dates):
        return None
    end_date = sorted_dates[end_idx]
    p0 = ticker_prices.get(trade_date)
    p1 = ticker_prices.get(end_date)
    if p0 is None or p1 is None or p0 <= 0:
        return None
    return (p1 - p0) / p0


# ---------------------------------------------------------------------------
# IC + portfolio eval
# ---------------------------------------------------------------------------

def _avg_ranks(values: List[float]) -> List[float]:
    n = len(values)
    indexed = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n - 1 and values[indexed[j + 1]] == values[indexed[j]]:
            j += 1
        avg_r = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[indexed[k]] = avg_r
        i = j + 1
    return ranks


def spearman_ic(signal: List[float], returns: List[float]) -> Optional[float]:
    n = len(signal)
    if n < 5:
        return None
    rx = _avg_ranks(signal)
    ry = _avg_ranks(returns)
    mx = statistics.mean(rx)
    my = statistics.mean(ry)
    cov = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    sx = math.sqrt(sum((rx[i] - mx) ** 2 for i in range(n)))
    sy = math.sqrt(sum((ry[i] - my) ** 2 for i in range(n)))
    if sx == 0.0 or sy == 0.0:
        return None
    return cov / (sx * sy)


def compute_turnover(prev_set: List[str], curr_set: List[str]) -> float:
    if not prev_set and not curr_set:
        return 0.0
    s_prev = set(prev_set)
    s_curr = set(curr_set)
    diff = len(s_prev ^ s_curr)
    denom = max(len(s_prev), len(s_curr), 1)
    return 0.5 * diff / denom


# ---------------------------------------------------------------------------
# Paired statistics
# ---------------------------------------------------------------------------

def paired_stats(
    a_vals: List[float],
    b_vals: List[float],
    n_boot: int = 5000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compute paired difference statistics between two aligned series.

    Returns: mean_delta, std_delta, t_stat, p_value, ci_lo_95, ci_hi_95.
    """
    n = min(len(a_vals), len(b_vals))
    if n < 3:
        return {
            "n": n, "mean_delta": None, "std_delta": None,
            "t_stat": None, "p_value": None,
            "ci_lo_95": None, "ci_hi_95": None,
        }

    deltas = [b_vals[i] - a_vals[i] for i in range(n)]
    mean_d = statistics.mean(deltas)
    std_d = statistics.stdev(deltas) if n > 1 else 0.0
    se = std_d / math.sqrt(n) if n > 0 else 0.0
    t_stat = mean_d / se if se > 0 else 0.0

    # Two-tailed p-value approximation (t-distribution with n-1 df)
    # Using normal approximation for simplicity (accurate for n > 20)
    z = abs(t_stat)
    # Abramowitz-Stegun approximation for normal CDF tail
    p_val = 2.0 * _normal_sf(z)

    # Bootstrap 95% CI on mean delta
    rng = random.Random(seed)
    boot_means: List[float] = []
    for _ in range(n_boot):
        sample = [rng.choice(deltas) for _ in range(n)]
        boot_means.append(statistics.mean(sample))
    boot_means.sort()
    lo_idx = int(0.025 * n_boot)
    hi_idx = int(0.975 * n_boot)
    ci_lo = boot_means[lo_idx]
    ci_hi = boot_means[min(hi_idx, len(boot_means) - 1)]

    return {
        "n": n,
        "mean_delta": round(mean_d, 6),
        "std_delta": round(std_d, 6),
        "t_stat": round(t_stat, 4),
        "p_value": round(p_val, 4),
        "ci_lo_95": round(ci_lo, 6),
        "ci_hi_95": round(ci_hi, 6),
    }


def _normal_sf(z: float) -> float:
    """Survival function for standard normal (1 - CDF), Abramowitz-Stegun."""
    if z < 0:
        return 1.0 - _normal_sf(-z)
    t = 1.0 / (1.0 + 0.2316419 * z)
    d = 0.3989422804014327  # 1/sqrt(2*pi)
    poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937
           + t * (-1.821255978 + t * 1.330274429))))
    return d * math.exp(-0.5 * z * z) * poly


# ---------------------------------------------------------------------------
# Re-ranking
# ---------------------------------------------------------------------------

@dataclass
class AnchorConfig:
    label: str
    anchor_source: str  # "alpha_pct", "optionality_pct", "blend"
    blend_w: float = 0.0  # only for blend
    alpha_modifier_mode: str = "off"
    alpha_modifier_weight: float = 0.0


def compute_blend_pct(rows: List[Dict[str, str]], w: float) -> None:
    """Compute blended_pct = (1-w)*optionality_pct + w*alpha_pct for each row."""
    for r in rows:
        opt_pct = safe_float(r.get("clinical_optionality_pct_dev")) or 0.0
        alpha_pct = safe_float(r.get("alpha_cohort_pct")) or 0.0
        r["_blend_pct"] = str((1.0 - w) * opt_pct + w * alpha_pct)


def rerank_rows(
    rows: List[Dict[str, str]],
    config: AnchorConfig,
    ruleset: DecisionRuleset,
) -> List[Dict[str, str]]:
    """Re-sort rows using the given anchor config."""
    backfill_columns(rows)

    if config.anchor_source == "blend":
        compute_blend_pct(rows, config.blend_w)

    def _get_tiebreaker(r: Dict[str, str]) -> float:
        if config.anchor_source == "alpha_pct":
            return safe_float(r.get("alpha_cohort_pct")) or 0.0
        elif config.anchor_source == "optionality_pct":
            return safe_float(r.get("clinical_optionality_pct_dev")) or 0.0
        elif config.anchor_source == "blend":
            return safe_float(r.get("_blend_pct")) or 0.0
        return 0.0

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
# Date evaluation
# ---------------------------------------------------------------------------

@dataclass
class QuintileResult:
    q_means: List[float] = field(default_factory=list)  # [q1..q5]
    spread: Optional[float] = None   # q1 - q5
    universe_mean: Optional[float] = None
    q1_excess: Optional[float] = None
    q5_excess: Optional[float] = None
    n_ranked: int = 0
    monotone: bool = False            # Q1>Q2>Q3>Q4>Q5
    q1_gt_q5: bool = False            # Q1>Q5 (weak monotonicity)
    slope: Optional[float] = None     # linear fit: quintile index vs return


@dataclass
class DateResult:
    date: str = ""
    n_eligible: int = 0
    skipped: bool = False
    alpha_table_built: bool = False
    ics: Dict[int, Optional[float]] = field(default_factory=dict)
    returns: Dict[int, Optional[float]] = field(default_factory=dict)
    n_held: Dict[int, int] = field(default_factory=dict)
    top_k_tickers: List[str] = field(default_factory=list)
    quintiles: Dict[int, Optional[QuintileResult]] = field(default_factory=dict)


def evaluate_date(
    rows: List[Dict[str, str]],
    snap_date: str,
    prices: Dict[str, Dict[str, float]],
    sorted_dates: List[str],
    horizons: List[int],
    top_k: int,
) -> DateResult:
    """Evaluate a ranked snapshot: IC + top-K returns per horizon."""
    result = DateResult(date=snap_date)

    eligible_rows = [r for r in rows if r.get("eligible") == "1"]
    result.n_eligible = len(eligible_rows)
    if not eligible_rows:
        result.skipped = True
        return result

    ranked = sorted(eligible_rows, key=lambda r: int(r.get("actionable_rank") or 9999))
    tickers_ranked = [r.get("ticker", "") for r in ranked]

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
        result.skipped = True
        return result

    for h in horizons:
        fwd_rets: Dict[str, float] = {}
        for t in tickers_ranked:
            if t not in prices:
                continue
            ret = compute_forward_return(prices[t], sorted_dates, trade_date, h)
            if ret is not None:
                fwd_rets[t] = ret

        signal_tickers = [t for t in tickers_ranked if t in fwd_rets]
        if len(signal_tickers) < 5:
            result.ics[h] = None
            result.returns[h] = None
            result.n_held[h] = 0
            continue

        signal_vals = [-float(i + 1) for i, t in enumerate(tickers_ranked) if t in fwd_rets]
        return_vals = [fwd_rets[t] for t in signal_tickers]
        result.ics[h] = spearman_ic(signal_vals, return_vals)

        held = [t for t in tickers_ranked[:top_k] if t in fwd_rets]
        if held:
            result.returns[h] = statistics.mean(fwd_rets[t] for t in held)
            result.n_held[h] = len(held)
        else:
            result.returns[h] = None
            result.n_held[h] = 0

        if not result.top_k_tickers:
            result.top_k_tickers = held

        # Quintile spread
        if len(signal_tickers) >= 25:
            ordered_rets = [fwd_rets[t] for t in signal_tickers]
            n_q = len(ordered_rets)
            bucket_size = n_q // 5
            q_means = []
            for qi in range(5):
                start = qi * bucket_size
                end = start + bucket_size if qi < 4 else n_q
                q_means.append(statistics.mean(ordered_rets[start:end]))
            uni_mean = statistics.mean(ordered_rets)
            # Monotonicity: strict Q1>Q2>Q3>Q4>Q5
            mono = all(q_means[i] > q_means[i + 1] for i in range(4))
            # Slope: linear fit of quintile index (1..5) vs q_means
            # x_mean=3, x_var=2, slope = sum((xi-3)*yi) / 10
            slope = sum((i + 1 - 3) * q_means[i] for i in range(5)) / 10.0
            qr = QuintileResult(
                q_means=q_means,
                spread=q_means[0] - q_means[4],
                universe_mean=uni_mean,
                q1_excess=q_means[0] - uni_mean,
                q5_excess=q_means[4] - uni_mean,
                n_ranked=n_q,
                monotone=mono,
                q1_gt_q5=q_means[0] > q_means[4],
                slope=slope,
            )
            result.quintiles[h] = qr
        else:
            result.quintiles[h] = None

    return result


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def write_summary_md(
    out_path: Path,
    summary_rows: List[Dict[str, Any]],
    horizons: List[int],
    configs: List[AnchorConfig],
    config_results: Dict[str, Dict[str, Any]],
    common_dates: List[str],
    all_by_date: Dict[str, Dict[str, Dict[str, Any]]],
    args: argparse.Namespace,
) -> None:
    """Write human-readable summary.md report."""
    lines: List[str] = []
    lines.append("# Anchor Replay Results\n")
    lines.append(f"**Date grid**: {args.date_grid}")
    lines.append(f"**Range**: {args.date_from} → {args.date_to}")
    lines.append(f"**Common eval dates**: {len(common_dates)}")
    lines.append(f"**Horizons**: {horizons}")
    lines.append(f"**Top-K**: {args.top_k}")
    lines.append(f"**Configs**: {[c.label for c in configs]}")
    lines.append("")

    # Aggregate table
    lines.append("## Aggregate Metrics (common-date intersection)\n")
    h_parts = ["Config"]
    for h in horizons:
        h_parts += [f"IC({h}d)", f"ΔIC", f"Hit", f"n"]
    h_parts += ["Turnover"]
    for h in horizons:
        h_parts += [f"Ret({h}d)"]
    for h in horizons:
        h_parts += [f"Sprd({h}d)", f"ΔSprd", f"S>0", f"Mono%", f"Q1>5", f"Slope"]
    lines.append("| " + " | ".join(h_parts) + " |")
    lines.append("|" + "|".join(["---"] * len(h_parts)) + "|")

    ref_label = configs[0].label
    ref = config_results[ref_label]

    for sr in summary_rows:
        parts = [sr["label"]]
        for h in horizons:
            ic = sr.get(f"ic_{h}d")
            hit = sr.get(f"hit_{h}d")
            n = sr.get(f"n_{h}d", 0)
            ref_ic = summary_rows[0].get(f"ic_{h}d")

            ic_s = f"{ic:.4f}" if ic is not None else "—"
            if ic is not None and ref_ic is not None and sr["label"] != ref_label:
                d_s = f"{ic - ref_ic:+.4f}"
            else:
                d_s = "—"
            hit_s = f"{hit:.0%}" if hit is not None else "—"
            parts += [ic_s, d_s, hit_s, str(n)]

        tv = sr.get("mean_turnover")
        parts.append(f"{tv:.3f}" if tv is not None else "—")

        for h in horizons:
            ret = sr.get(f"ret_{h}d")
            parts.append(f"{ret*100:.2f}%" if ret is not None else "—")

        for h in horizons:
            sp = sr.get(f"spread_{h}d")
            sp_hit = sr.get(f"hit_spread_{h}d")
            ref_sp = summary_rows[0].get(f"spread_{h}d")
            sp_s = f"{sp*100:.2f}%" if sp is not None else "—"
            if sp is not None and ref_sp is not None and sr["label"] != ref_label:
                d_sp = f"{(sp - ref_sp)*100:+.2f}%"
            else:
                d_sp = "—"
            sp_hit_s = f"{sp_hit:.0%}" if sp_hit is not None else "—"
            mono = sr.get(f"mono_{h}d")
            q1g5 = sr.get(f"q1_gt_q5_{h}d")
            sl = sr.get(f"slope_{h}d")
            mono_s = f"{mono:.0%}" if mono is not None else "—"
            q1g5_s = f"{q1g5:.0%}" if q1g5 is not None else "—"
            sl_s = f"{sl*100:+.3f}%" if sl is not None else "—"
            parts += [sp_s, d_sp, sp_hit_s, mono_s, q1g5_s, sl_s]

        lines.append("| " + " | ".join(parts) + " |")

    # Paired stats section
    lines.append("")
    lines.append("## Paired Statistics (vs A_alpha)\n")

    for cfg in configs[1:]:
        lines.append(f"### {cfg.label}\n")
        for h in horizons:
            ref_ics = ref["ic_by_h"][h]
            cand_ics = config_results[cfg.label]["ic_by_h"][h]
            ps = paired_stats(ref_ics, cand_ics)
            lines.append(f"**IC({h}d)**: ΔIC={ps['mean_delta']}, "
                         f"std={ps['std_delta']}, t={ps['t_stat']}, "
                         f"p={ps['p_value']}, "
                         f"95%CI=[{ps['ci_lo_95']}, {ps['ci_hi_95']}]")

            ref_rets = ref["ret_by_h"][h]
            cand_rets = config_results[cfg.label]["ret_by_h"][h]
            ps_r = paired_stats(ref_rets, cand_rets)
            lines.append(f"**Ret({h}d)**: ΔRet={ps_r['mean_delta']}, "
                         f"std={ps_r['std_delta']}, t={ps_r['t_stat']}, "
                         f"p={ps_r['p_value']}, "
                         f"95%CI=[{ps_r['ci_lo_95']}, {ps_r['ci_hi_95']}]")

            ref_sp = ref["spread_by_h"][h]
            cand_sp = config_results[cfg.label]["spread_by_h"][h]
            if len(ref_sp) >= 3 and len(cand_sp) >= 3:
                ps_s = paired_stats(ref_sp, cand_sp)
                lines.append(f"**Spread({h}d)**: ΔSpread={ps_s['mean_delta']}, "
                             f"std={ps_s['std_delta']}, t={ps_s['t_stat']}, "
                             f"p={ps_s['p_value']}, "
                             f"95%CI=[{ps_s['ci_lo_95']}, {ps_s['ci_hi_95']}]")
        lines.append("")

    # Promotion check
    lines.append("## Promotion Gate (ΔIC + ΔSpread + CI + Turnover)\n")
    lines.append("Gates: ΔIC(60d) ≥ +0.005, ΔSpread(60d) ≥ +1.0% with CI_lo > 0, "
                 "ΔTurnover ≤ +0.05\n")
    ref_row = summary_rows[0]
    for sr in summary_rows[1:]:
        checks = []
        info = []
        for h in horizons:
            ref_ic = ref_row.get(f"ic_{h}d")
            cand_ic = sr.get(f"ic_{h}d")
            if cand_ic is not None and ref_ic is not None:
                d_ic = cand_ic - ref_ic
                checks.append(f"ΔIC({h}d)={d_ic:+.4f} {'PASS' if d_ic >= 0.005 else 'FAIL'}")

            # ΔSpread gate
            ref_sp_list = ref["spread_by_h"][h]
            cand_sp_list = config_results[sr["label"]]["spread_by_h"][h]
            if len(ref_sp_list) >= 3 and len(cand_sp_list) >= 3:
                ps_sp = paired_stats(ref_sp_list, cand_sp_list)
                d_spread = ps_sp["mean_delta"]
                ci_lo = ps_sp["ci_lo_95"]
                if d_spread is not None and ci_lo is not None:
                    sp_pass = d_spread >= 0.01 and ci_lo > 0
                    checks.append(
                        f"ΔSpread({h}d)={d_spread*100:+.2f}%, "
                        f"CI_lo={ci_lo*100:+.2f}% "
                        f"{'PASS' if sp_pass else 'FAIL'}"
                    )

            # Info: ΔHit, monotonicity
            ref_hit = ref_row.get(f"hit_{h}d")
            cand_hit = sr.get(f"hit_{h}d")
            if cand_hit is not None and ref_hit is not None:
                info.append(f"ΔHit({h}d)={cand_hit - ref_hit:+.0%}")
            mono = sr.get(f"mono_{h}d")
            q1g5 = sr.get(f"q1_gt_q5_{h}d")
            if mono is not None:
                info.append(f"Mono({h}d)={mono:.0%}, Q1>Q5={q1g5:.0%}")

        tv_ref = ref_row.get("mean_turnover") or 0
        tv_cand = sr.get("mean_turnover") or 0
        checks.append(f"ΔTurnover={tv_cand - tv_ref:+.3f} {'PASS' if (tv_cand - tv_ref) <= 0.05 else 'FAIL'}")

        status = "PROMOTE" if all("PASS" in c for c in checks) else "NO_PROMOTE"
        lines.append(f"**{sr['label']}**: {status}")
        for c in checks:
            lines.append(f"- {c}")
        if info:
            lines.append(f"- _(info: {', '.join(info)})_")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Promotion-grade anchor replay on monthly/weekly archives",
    )
    parser.add_argument("--archive-dir", type=Path,
                        default=PROJECT_ROOT / "data" / "archives")
    parser.add_argument("--price-csv", type=Path,
                        default=PROJECT_ROOT / "production_data" / "price_history.csv")
    parser.add_argument("--date-from", type=str, default="2020-01-31")
    parser.add_argument("--date-to", type=str, default="2026-02-28")
    parser.add_argument("--date-grid", type=str, default="monthly",
                        choices=["monthly", "weekly", "all"],
                        help="Date sampling grid (default: monthly)")
    parser.add_argument("--horizons", type=str, default="20,60")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-eval-dates", type=int, default=12,
                        help="Minimum eval dates with matured returns for reporting")
    parser.add_argument("--blend-weights", type=str, default="0.02,0.05,0.10")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    horizons = [int(h.strip()) for h in args.horizons.split(",")]
    blend_weights = [float(w.strip()) for w in args.blend_weights.split(",")]

    # ---- Discover archives ----
    archives = discover_archives(
        args.archive_dir, args.date_from, args.date_to, args.date_grid,
    )
    if not archives:
        print("No archives found in range.")
        return
    print(f"Archives: {len(archives)} ({args.date_grid} grid, "
          f"{archives[0][0]} → {archives[-1][0]})")

    # ---- Load prices ----
    print("Loading prices ...")
    prices = load_price_series(args.price_csv)
    all_dates_set: set = set()
    for tp in prices.values():
        all_dates_set.update(tp.keys())
    sorted_dates = sorted(all_dates_set)
    print(f"  {len(prices)} tickers, {len(sorted_dates)} trading dates")

    # ---- Build base ruleset ----
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

    from dataclasses import replace as dc_replace

    rs_alpha = dc_replace(base_rs,
                          sort_anchor="alpha_cohort",
                          alpha_modifier_mode="off",
                          alpha_modifier_weight=0.0)

    # ---- Define configs ----
    configs: List[AnchorConfig] = [
        AnchorConfig("A_alpha", "alpha_pct"),
        AnchorConfig("E_optionality", "optionality_pct"),
    ]
    for w in blend_weights:
        configs.append(AnchorConfig(f"E_blend_{w:.2f}", "blend", blend_w=w))

    print(f"Configs: {[c.label for c in configs]}")

    # ---- Pre-build PIT alpha tables ----
    print("\nPre-building PIT alpha tables ...")
    alpha_tables: Dict[str, Optional[dict]] = {}
    for date_str, tar_path in archives:
        t0 = time.time()
        table = build_oos_table(
            as_of_date=date_str,
            train_mode="trailing-6",
            horizon=84,
            min_train_dates=6,
            archive_dir=args.archive_dir,
            price_csv=args.price_csv,
        )
        dt = time.time() - t0
        if table is not None:
            populated = sum(1 for c in table.get("cells", {}).values() if c.get("n", 0) > 0)
            print(f"  {date_str}: {populated}/36 cells ({dt:.1f}s)")
        else:
            print(f"  {date_str}: NONE (insufficient training data) ({dt:.1f}s)")
        alpha_tables[date_str] = table

    # ---- Main evaluation loop ----
    print(f"\nEvaluating {len(configs)} configs × {len(archives)} dates ...")

    # Per-config accumulators: store per-date results
    config_date_results: Dict[str, Dict[str, DateResult]] = {
        cfg.label: {} for cfg in configs
    }

    config_results: Dict[str, Dict[str, Any]] = {}
    for cfg in configs:
        config_results[cfg.label] = {
            "ic_by_h": {h: [] for h in horizons},
            "ret_by_h": {h: [] for h in horizons},
            "prev_topk": [],
            "turnovers": [],
            "eval_dates": [],
        }

    # Track by-date output
    all_by_date: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for date_str, tar_path in archives:
        raw_rows = load_archive_rankings(tar_path, date_str)
        if not raw_rows:
            print(f"  {date_str}: no rankings — skip")
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

        date_entry: Dict[str, Dict[str, Any]] = {}

        for cfg in configs:
            rows = copy.deepcopy(raw_rows)
            rows = rerank_rows(rows, cfg, rs_alpha)
            metrics = evaluate_date(rows, date_str, prices, sorted_dates, horizons, args.top_k)

            metrics.alpha_table_built = alpha_built
            config_date_results[cfg.label][date_str] = metrics

            if metrics.skipped:
                continue

            acc = config_results[cfg.label]
            acc["eval_dates"].append(date_str)

            for h in horizons:
                ic = metrics.ics.get(h)
                if ic is not None:
                    acc["ic_by_h"][h].append(ic)
                ret = metrics.returns.get(h)
                if ret is not None:
                    acc["ret_by_h"][h].append(ret)

            curr_topk = metrics.top_k_tickers[:args.top_k]
            if acc["prev_topk"]:
                t = compute_turnover(acc["prev_topk"], curr_topk)
                acc["turnovers"].append(t)
            acc["prev_topk"] = curr_topk

            # by-date entry
            entry: Dict[str, Any] = {
                "n_eligible": metrics.n_eligible,
                "alpha_table_built": alpha_built,
                "top_k": metrics.top_k_tickers[:args.top_k],
            }
            for h in horizons:
                entry[f"ic_{h}d"] = metrics.ics.get(h)
                entry[f"ret_{h}d"] = metrics.returns.get(h)
                entry[f"n_held_{h}d"] = metrics.n_held.get(h, 0)
                qr = metrics.quintiles.get(h)
                if qr is not None:
                    entry[f"spread_{h}d"] = qr.spread
                    entry[f"q1_ret_{h}d"] = qr.q_means[0]
                    entry[f"q5_ret_{h}d"] = qr.q_means[4]
                    entry[f"q1_excess_{h}d"] = qr.q1_excess
                    entry[f"q5_excess_{h}d"] = qr.q5_excess
                    entry[f"monotone_{h}d"] = qr.monotone
                    entry[f"q1_gt_q5_{h}d"] = qr.q1_gt_q5
                    entry[f"slope_{h}d"] = qr.slope
            date_entry[cfg.label] = entry

        all_by_date[date_str] = date_entry

        print(f"  {date_str}: {len(raw_rows)} tickers, "
              f"alpha_table={'YES' if alpha_built else 'NO'}")

    # ---- Common-date intersection ----
    # Find dates where ALL configs have valid IC for all horizons.
    # Alpha-dependent configs (alpha_pct, blend) additionally require
    # alpha_table_built == True so A vs E comparisons are honest.
    _ALPHA_SOURCES = {"alpha_pct", "blend"}

    common_dates: List[str] = []
    for date_str, _ in archives:
        ok = True
        for cfg in configs:
            dr = config_date_results[cfg.label].get(date_str)
            if dr is None or dr.skipped:
                ok = False
                break
            if not all(dr.ics.get(h) is not None for h in horizons):
                ok = False
                break
            # Alpha-dependent configs must have a real alpha table
            if cfg.anchor_source in _ALPHA_SOURCES and not dr.alpha_table_built:
                ok = False
                break
        if ok:
            common_dates.append(date_str)

    print(f"\nCommon eval dates: {len(common_dates)} / {len(archives)}")

    # Rebuild accumulators on common-date intersection only
    for cfg in configs:
        acc = config_results[cfg.label]
        acc["ic_by_h"] = {h: [] for h in horizons}
        acc["ret_by_h"] = {h: [] for h in horizons}
        acc["spread_by_h"] = {h: [] for h in horizons}
        acc["slope_by_h"] = {h: [] for h in horizons}
        acc["monotone_by_h"] = {h: [] for h in horizons}
        acc["q1_gt_q5_by_h"] = {h: [] for h in horizons}
        acc["turnovers"] = []
        acc["eval_dates"] = []
        prev_topk: List[str] = []
        for d in common_dates:
            dr = config_date_results[cfg.label][d]
            acc["eval_dates"].append(d)
            for h in horizons:
                ic = dr.ics.get(h)
                if ic is not None:
                    acc["ic_by_h"][h].append(ic)
                ret = dr.returns.get(h)
                if ret is not None:
                    acc["ret_by_h"][h].append(ret)
                qr = dr.quintiles.get(h)
                if qr is not None and qr.spread is not None:
                    acc["spread_by_h"][h].append(qr.spread)
                    acc["monotone_by_h"][h].append(qr.monotone)
                    acc["q1_gt_q5_by_h"][h].append(qr.q1_gt_q5)
                    if qr.slope is not None:
                        acc["slope_by_h"][h].append(qr.slope)
            curr_topk = dr.top_k_tickers[:args.top_k]
            if prev_topk:
                acc["turnovers"].append(compute_turnover(prev_topk, curr_topk))
            prev_topk = curr_topk

    # ---- Aggregate + report ----
    print("\n" + "=" * 120)
    print("# Anchor Replay Results (common-date intersection)")
    print(f"# Grid: {args.date_grid}, dates: {len(common_dates)}, "
          f"horizons: {horizons}, top_k: {args.top_k}")
    print()

    header = f"{'Config':<22s}"
    for h in horizons:
        header += f" {'IC('+str(h)+'d)':>10s} {'ΔIC':>8s} {'Hit':>6s} {'n':>4s}"
    header += f" {'Turnover':>9s}"
    for h in horizons:
        header += f" {'Ret('+str(h)+'d)':>10s}"
    for h in horizons:
        header += f" {'Sprd('+str(h)+'d)':>11s} {'ΔSprd':>8s} {'S>0':>5s} {'Mono%':>6s} {'Q1>5':>5s} {'Slope':>8s}"
    print(header)
    print("-" * len(header))

    ref_label = configs[0].label
    ref = config_results[ref_label]

    summary_rows: List[Dict[str, Any]] = []

    for cfg in configs:
        acc = config_results[cfg.label]
        row: Dict[str, Any] = {"label": cfg.label}

        line = f"{cfg.label:<22s}"
        for h in horizons:
            ics = acc["ic_by_h"][h]
            ref_ics = ref["ic_by_h"][h]
            ic_mean = statistics.mean(ics) if ics else None
            ref_ic_mean = statistics.mean(ref_ics) if ref_ics else None
            hit = sum(1 for x in ics if x > 0) / len(ics) if ics else None

            ic_s = f"{ic_mean:.4f}" if ic_mean is not None else "—"
            if ic_mean is not None and ref_ic_mean is not None and cfg.label != ref_label:
                delta = ic_mean - ref_ic_mean
                d_s = f"{delta:+.4f}"
            else:
                d_s = "—"
            hit_s = f"{hit:.0%}" if hit is not None else "—"
            n_s = str(len(ics))

            line += f" {ic_s:>10s} {d_s:>8s} {hit_s:>6s} {n_s:>4s}"
            row[f"ic_{h}d"] = ic_mean
            row[f"hit_{h}d"] = hit
            row[f"n_{h}d"] = len(ics)

        tv = statistics.mean(acc["turnovers"]) if acc["turnovers"] else None
        row["mean_turnover"] = tv
        line += f" {tv:>9.3f}" if tv is not None else f" {'—':>9s}"

        for h in horizons:
            rets = acc["ret_by_h"][h]
            if rets:
                gross = statistics.mean(rets)
                row[f"ret_{h}d"] = gross
                line += f" {gross*100:>9.2f}%"
            else:
                line += f" {'—':>10s}"

        # Spread + monotonicity columns
        for h in horizons:
            spreads = acc["spread_by_h"][h]
            ref_spreads = ref["spread_by_h"][h]
            monos = acc["monotone_by_h"][h]
            q1g5s = acc["q1_gt_q5_by_h"][h]
            slopes = acc["slope_by_h"][h]
            if spreads:
                sp_mean = statistics.mean(spreads)
                sp_hit = sum(1 for s in spreads if s > 0) / len(spreads)
                mono_pct = sum(1 for m in monos if m) / len(monos) if monos else 0
                q1g5_pct = sum(1 for q in q1g5s if q) / len(q1g5s) if q1g5s else 0
                slope_mean = statistics.mean(slopes) if slopes else 0
                row[f"spread_{h}d"] = sp_mean
                row[f"hit_spread_{h}d"] = sp_hit
                row[f"mono_{h}d"] = mono_pct
                row[f"q1_gt_q5_{h}d"] = q1g5_pct
                row[f"slope_{h}d"] = slope_mean
                line += f" {sp_mean*100:>10.2f}%"
                if cfg.label != ref_label and ref_spreads:
                    ref_sp = statistics.mean(ref_spreads)
                    d_sp = sp_mean - ref_sp
                    line += f" {d_sp*100:>+7.2f}%"
                else:
                    line += f" {'—':>8s}"
                line += f" {sp_hit:>5.0%}"
                line += f" {mono_pct:>5.0%}"
                line += f" {q1g5_pct:>5.0%}"
                line += f" {slope_mean*100:>+7.3f}%"
            else:
                row[f"spread_{h}d"] = None
                row[f"hit_spread_{h}d"] = None
                row[f"mono_{h}d"] = None
                row[f"q1_gt_q5_{h}d"] = None
                row[f"slope_{h}d"] = None
                line += f" {'—':>11s} {'—':>8s} {'—':>5s} {'—':>6s} {'—':>5s} {'—':>8s}"

        print(line)
        summary_rows.append(row)

    # ---- Paired statistics ----
    print()
    print("Paired Statistics (vs A_alpha, common-date intersection):")
    print("-" * 80)
    for cfg in configs[1:]:
        for h in horizons:
            ref_ics = ref["ic_by_h"][h]
            cand_ics = config_results[cfg.label]["ic_by_h"][h]
            ps = paired_stats(ref_ics, cand_ics)
            print(f"  {cfg.label} IC({h}d): ΔIC={ps['mean_delta']:+.4f}, "
                  f"t={ps['t_stat']:.2f}, p={ps['p_value']:.3f}, "
                  f"95%CI=[{ps['ci_lo_95']:+.4f}, {ps['ci_hi_95']:+.4f}]"
                  if ps['mean_delta'] is not None else
                  f"  {cfg.label} IC({h}d): insufficient data")

    # Spread paired stats
    print()
    print("Paired Spread Statistics (vs A_alpha):")
    print("-" * 80)
    for cfg in configs[1:]:
        for h in horizons:
            ref_sp = ref["spread_by_h"][h]
            cand_sp = config_results[cfg.label]["spread_by_h"][h]
            if len(ref_sp) >= 3 and len(cand_sp) >= 3:
                ps = paired_stats(ref_sp, cand_sp)
                print(f"  {cfg.label} Spread({h}d): Δ={ps['mean_delta']:+.4f}, "
                      f"t={ps['t_stat']:.2f}, p={ps['p_value']:.3f}, "
                      f"95%CI=[{ps['ci_lo_95']:+.4f}, {ps['ci_hi_95']:+.4f}]"
                      if ps['mean_delta'] is not None else
                      f"  {cfg.label} Spread({h}d): insufficient data")
            else:
                print(f"  {cfg.label} Spread({h}d): insufficient data "
                      f"(ref={len(ref_sp)}, cand={len(cand_sp)})")

    # ---- Promotion check ----
    # Gates (all must PASS):
    #   1. ΔIC(60d) >= +0.005
    #   2. ΔSpread(60d) >= +1.0% AND bootstrap 95% CI lower bound > 0
    #   3. ΔTurnover <= +0.05
    # Reporting only (no gate): ΔHit, monotonicity
    print()
    print("Promotion Gate (ΔIC + ΔSpread + CI + Turnover):")
    print("-" * 80)
    ref_row = summary_rows[0]
    for sr in summary_rows[1:]:
        label = sr["label"]
        checks = []
        info = []  # reporting-only metrics

        for h in horizons:
            # Gate: ΔIC
            ref_ic = ref_row.get(f"ic_{h}d")
            cand_ic = sr.get(f"ic_{h}d")
            if cand_ic is not None and ref_ic is not None:
                d_ic = cand_ic - ref_ic
                ic_pass = d_ic >= 0.005
                checks.append(f"ΔIC({h}d)={d_ic:+.4f} {'PASS' if ic_pass else 'FAIL'}")

            # Gate: ΔSpread + CI
            ref_sp_list = ref["spread_by_h"][h]
            cand_sp_list = config_results[sr["label"]]["spread_by_h"][h]
            if len(ref_sp_list) >= 3 and len(cand_sp_list) >= 3:
                ps_sp = paired_stats(ref_sp_list, cand_sp_list)
                d_spread = ps_sp["mean_delta"]
                ci_lo = ps_sp["ci_lo_95"]
                if d_spread is not None and ci_lo is not None:
                    sp_pass = d_spread >= 0.01 and ci_lo > 0
                    checks.append(
                        f"ΔSpread({h}d)={d_spread*100:+.2f}%, "
                        f"CI_lo={ci_lo*100:+.2f}% "
                        f"{'PASS' if sp_pass else 'FAIL'}"
                    )

            # Info: ΔHit (reporting only)
            ref_hit = ref_row.get(f"hit_{h}d")
            cand_hit = sr.get(f"hit_{h}d")
            if cand_hit is not None and ref_hit is not None:
                d_hit = cand_hit - ref_hit
                info.append(f"ΔHit({h}d)={d_hit:+.0%}")

            # Info: monotonicity (reporting only)
            mono = sr.get(f"mono_{h}d")
            q1g5 = sr.get(f"q1_gt_q5_{h}d")
            if mono is not None:
                info.append(f"Mono({h}d)={mono:.0%}, Q1>Q5={q1g5:.0%}")

        # Gate: Turnover
        tv_ref = ref_row.get("mean_turnover") or 0
        tv_cand = sr.get("mean_turnover") or 0
        tv_pass = (tv_cand - tv_ref) <= 0.05
        checks.append(f"ΔTurnover={tv_cand - tv_ref:+.3f} {'PASS' if tv_pass else 'FAIL'}")

        status = "PROMOTE" if all("PASS" in c for c in checks) else "NO_PROMOTE"
        print(f"  {label}: {status}")
        for c in checks:
            print(f"    {c}")
        if info:
            print(f"    (info: {', '.join(info)})")

    # ---- Write outputs ----
    out_dir = args.out or (PROJECT_ROOT / "output" / "anchor_replay")
    out_dir.mkdir(parents=True, exist_ok=True)

    # results.json
    with open(out_dir / "results.json", "w") as f:
        json.dump(summary_rows, f, indent=2, default=str)

    # by_date.json
    by_date_out: Dict[str, Any] = {}
    for d in sorted(all_by_date.keys()):
        by_date_out[d] = all_by_date[d]
    with open(out_dir / "by_date.json", "w") as f:
        json.dump(by_date_out, f, indent=2, default=str)

    # paired_stats.json
    paired_out: Dict[str, Dict[str, Any]] = {}
    for cfg in configs[1:]:
        paired_out[cfg.label] = {}
        for h in horizons:
            ref_ics = ref["ic_by_h"][h]
            cand_ics = config_results[cfg.label]["ic_by_h"][h]
            paired_out[cfg.label][f"ic_{h}d"] = paired_stats(ref_ics, cand_ics)

            ref_rets = ref["ret_by_h"][h]
            cand_rets = config_results[cfg.label]["ret_by_h"][h]
            paired_out[cfg.label][f"ret_{h}d"] = paired_stats(ref_rets, cand_rets)

            ref_sp = ref["spread_by_h"][h]
            cand_sp = config_results[cfg.label]["spread_by_h"][h]
            if len(ref_sp) >= 3 and len(cand_sp) >= 3:
                paired_out[cfg.label][f"spread_{h}d"] = paired_stats(ref_sp, cand_sp)
    with open(out_dir / "paired_stats.json", "w") as f:
        json.dump(paired_out, f, indent=2, default=str)

    # summary.md
    write_summary_md(
        out_dir / "summary.md",
        summary_rows, horizons, configs, config_results,
        common_dates, all_by_date, args,
    )

    print(f"\nOutput: {out_dir}")
    print(f"  results.json, by_date.json, paired_stats.json, summary.md")


if __name__ == "__main__":
    main()
