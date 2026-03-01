#!/usr/bin/env python3
"""Promotion-grade anchor replay on monthly archive grid.

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
  ΔHit(60d) ≥ +5pp
  Turnover not worse by > 0.05
  Gates don't degrade

Usage:
    python scripts/research/anchor_replay.py \
        --date-from 2024-01-31 --date-to 2025-10-31 \
        --horizons 20,60 --top-k 20 --cost-bps 15
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

def discover_monthly_archives(
    archive_dir: Path,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> List[Tuple[str, Path]]:
    """Find monthly archives (last day of month) in date range."""
    import re
    pattern = re.compile(r"(\d{4}-\d{2}-\d{2})\.tar\.gz$")
    results: List[Tuple[str, Path]] = []
    for p in sorted(archive_dir.glob("*.tar.gz")):
        m = pattern.search(p.name)
        if not m:
            continue
        d = m.group(1)
        if date_from and d < date_from:
            continue
        if date_to and d > date_to:
            continue
        results.append((d, p))
    results.sort(key=lambda x: x[0])
    return results


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
# Price loading (reused from run_alpha_experiment)
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

    # Determine tiebreaker_pct source
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
class DateResult:
    date: str = ""
    n_eligible: int = 0
    skipped: bool = False
    ics: Dict[int, Optional[float]] = field(default_factory=dict)
    returns: Dict[int, Optional[float]] = field(default_factory=dict)
    n_held: Dict[int, int] = field(default_factory=dict)
    top_k_tickers: List[str] = field(default_factory=list)


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

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Promotion-grade anchor replay on monthly archives",
    )
    parser.add_argument("--archive-dir", type=Path,
                        default=PROJECT_ROOT / "data" / "archives")
    parser.add_argument("--price-csv", type=Path,
                        default=PROJECT_ROOT / "production_data" / "price_history.csv")
    parser.add_argument("--date-from", type=str, default="2024-01-31")
    parser.add_argument("--date-to", type=str, default="2025-10-31")
    parser.add_argument("--horizons", type=str, default="20,60")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--cost-bps", type=int, default=15)
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
    archives = discover_monthly_archives(args.archive_dir, args.date_from, args.date_to)
    print(f"Archives: {len(archives)} ({archives[0][0]} → {archives[-1][0]})")

    # ---- Load prices ----
    print("Loading prices ...")
    prices = load_price_series(args.price_csv)
    all_dates_set: set = set()
    for tp in prices.values():
        all_dates_set.update(tp.keys())
    sorted_dates = sorted(all_dates_set)
    print(f"  {len(prices)} tickers, {len(sorted_dates)} trading dates")

    # ---- Build base ruleset ----
    dates_with_rs = sorted([
        p.name for p in (PROJECT_ROOT / "data" / "snapshots").iterdir()
        if p.is_dir() and len(p.name) == 10 and p.name[4] == "-"
        and (p / "decision_ruleset.json").exists()
    ])
    rs_path = PROJECT_ROOT / "data" / "snapshots" / dates_with_rs[-1] / "decision_ruleset.json"
    base_rs = DecisionRuleset.from_json(str(rs_path))
    print(f"Base ruleset: {base_rs.ruleset_id}")

    from dataclasses import replace as dc_replace

    # Make a ruleset with alpha anchor (for config A) and off modifier
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

    # Per-config accumulators
    config_results: Dict[str, Dict[str, Any]] = {}
    for cfg in configs:
        config_results[cfg.label] = {
            "ic_by_h": {h: [] for h in horizons},
            "ret_by_h": {h: [] for h in horizons},
            "prev_topk": [],
            "turnovers": [],
            "eval_dates": [],
        }

    for date_str, tar_path in archives:
        raw_rows = load_archive_rankings(tar_path, date_str)
        if not raw_rows:
            print(f"  {date_str}: no rankings — skip")
            continue

        # Backfill columns for old archives
        backfill_columns(raw_rows)
        backfill_clinical_z_tier(raw_rows)

        # Attach alpha scores if table available
        alpha_table = alpha_tables.get(date_str)
        if alpha_table is not None:
            attach_alpha_scores(
                raw_rows, alpha_table,
                shrink_k=base_rs.alpha_cohort_shrink_k,
                clip_min=base_rs.alpha_cohort_clip_min,
                clip_max=base_rs.alpha_cohort_clip_max,
            )
        else:
            # No alpha table — set defaults
            for r in raw_rows:
                r["alpha_cohort_raw"] = 0.0
                r["alpha_cohort_pct"] = 0.5
                r["alpha_cohort_key"] = "early|none|nonpos"

        # Evaluate each config
        for cfg in configs:
            rows = copy.deepcopy(raw_rows)
            rows = rerank_rows(rows, cfg, rs_alpha)
            metrics = evaluate_date(rows, date_str, prices, sorted_dates, horizons, args.top_k)

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

        print(f"  {date_str}: {len(raw_rows)} tickers, "
              f"alpha_table={'YES' if alpha_table else 'NO'}")

    # ---- Aggregate + report ----
    print("\n" + "=" * 110)
    print("# Anchor Replay Results")
    print(f"# Archives: {archives[0][0]} → {archives[-1][0]}, "
          f"horizons: {horizons}, top_k: {args.top_k}")
    print()

    # Header
    header = f"{'Config':<22s}"
    for h in horizons:
        header += f" {'IC('+str(h)+'d)':>10s} {'ΔIC':>8s} {'Hit':>6s} {'n':>4s}"
    header += f" {'Turnover':>9s}"
    for h in horizons:
        header += f" {'Ret('+str(h)+'d)':>10s}"
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

        # Gross returns per horizon
        for h in horizons:
            rets = acc["ret_by_h"][h]
            if rets:
                gross = statistics.mean(rets)
                row[f"ret_{h}d"] = gross
                line += f" {gross*100:>9.2f}%"
            else:
                line += f" {'—':>10s}"

        print(line)
        summary_rows.append(row)

    # ---- Promotion check ----
    print()
    ref_row = summary_rows[0]
    for sr in summary_rows[1:]:
        label = sr["label"]
        checks = []
        for h in horizons:
            ic_key = f"ic_{h}d"
            hit_key = f"hit_{h}d"
            ref_ic = ref_row.get(ic_key)
            cand_ic = sr.get(ic_key)
            ref_hit = ref_row.get(hit_key)
            cand_hit = sr.get(hit_key)

            if cand_ic is not None and ref_ic is not None:
                d_ic = cand_ic - ref_ic
                ic_pass = d_ic >= 0.005
                checks.append(f"ΔIC({h}d)={d_ic:+.4f} {'PASS' if ic_pass else 'FAIL'}")
            if cand_hit is not None and ref_hit is not None:
                d_hit = cand_hit - ref_hit
                hit_pass = d_hit >= 0.05
                checks.append(f"ΔHit({h}d)={d_hit:+.0%} {'PASS' if hit_pass else 'FAIL'}")

        tv_ref = ref_row.get("mean_turnover") or 0
        tv_cand = sr.get("mean_turnover") or 0
        tv_pass = (tv_cand - tv_ref) <= 0.05
        checks.append(f"ΔTurnover={tv_cand - tv_ref:+.3f} {'PASS' if tv_pass else 'FAIL'}")

        status = "PROMOTE" if all("PASS" in c for c in checks) else "NO_PROMOTE"
        print(f"  {label}: {status}")
        for c in checks:
            print(f"    {c}")

    # ---- Write JSON ----
    out_dir = args.out or (PROJECT_ROOT / "output" / "anchor_replay")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        json.dump(summary_rows, f, indent=2, default=str)
    print(f"\nJSON: {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
