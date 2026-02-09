#!/usr/bin/env python3
"""
Policy Matrix Evaluation

Runs a compact matrix of policy configurations against cached archive data
to select the Phase-2 default decision policy.

Matrix axes:
  - Ruleset:     v1 (default), v1.2 (a_floor=0.55), v1.3 (soft drawdown + 0.55)
  - Tier filter: A only, A+B
  - Top-K:       10, 20, 30

= 3 × 2 × 3 = 18 runs

Archives and forward returns are loaded once and shared across all runs.

Usage:
    python run_policy_matrix.py [options]

    --output-dir PATH    Output directory (default: output/)
    --start DATE         Archive start date filter
    --end DATE           Archive end date filter
    --dry-run            Print matrix + archive count, exit
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from decision_engine import DecisionRuleset, DEFAULT_RULESET
from run_decision_ruleset_sweep import (
    ArchiveData,
    compute_snapshot_returns,
    init_providers,
    load_archive_data,
)
from run_rank_ic_backtest import (
    ARCHIVE_DIR,
    OUTPUT_DIR,
    _winsorize,
    compute_as_of_fence,
    compute_forward_returns,
    compute_residual_returns,
    discover_archives,
    estimate_xbi_betas,
    compute_xbi_regime,
)
from run_decision_strategy_backtest import (
    HORIZONS,
    MultiHorizonReturns,
    build_strategy_portfolio,
    build_composite_baseline,
    build_universe_baseline,
    evaluate_portfolio,
    compute_portfolio_turnover,
    compute_multi_horizon_returns,
    _percentile,
)


# =============================================================================
# MATRIX CONFIGURATION
# =============================================================================

RULESET_DIR = PROJECT_ROOT / "production_data" / "decision_rulesets"

RULESETS = {
    "v1_default": RULESET_DIR / "v1.json",
    "v1.2_a055": RULESET_DIR / "v1.2_candidate.json",
    "v1.3_soft_dd": RULESET_DIR / "v1.3_candidate.json",
}

TIER_FILTERS = {
    "A_only": ["A"],
    "A_B": ["A", "B"],
}

TOP_K_VALUES = [10, 20, 30]


# =============================================================================
# SINGLE POLICY EVALUATION
# =============================================================================

def evaluate_policy(
    archive_cache: Dict[str, ArchiveData],
    return_cache: Dict[str, MultiHorizonReturns],
    ruleset: DecisionRuleset,
    tier_filter: List[str],
    top_k: int,
    horizons: List[int] = HORIZONS,
) -> Dict[str, Any]:
    """Evaluate one policy config across all cached archives.

    Returns aggregated metrics for selection comparison.
    """
    dates = sorted(archive_cache.keys())
    prev_positions: List[Dict[str, Any]] = []

    # Per-snapshot accumulators
    strat_weighted_resid: Dict[int, List[float]] = {h: [] for h in horizons}
    strat_ew_resid: Dict[int, List[float]] = {h: [] for h in horizons}
    baseline_ew_resid: Dict[int, List[float]] = {h: [] for h in horizons}
    universe_ew_resid: Dict[int, List[float]] = {h: [] for h in horizons}
    hit_rates: Dict[int, List[float]] = {h: [] for h in horizons}
    left_tail_p5: Dict[int, List[float]] = {h: [] for h in horizons}
    left_tail_p10: Dict[int, List[float]] = {h: [] for h in horizons}
    left_tail_max_loss: Dict[int, List[float]] = {h: [] for h in horizons}
    position_turnovers: List[float] = []
    weight_turnovers: List[float] = []
    position_counts: List[int] = []
    all_tickers: Counter = Counter()
    regime_strat_resid_60: Dict[str, List[float]] = {}

    for date_str in dates:
        ad = archive_cache[date_str]
        mhr = return_cache[date_str]

        # Build portfolios
        strat = build_strategy_portfolio(ad, ruleset, tier_filter, top_k)
        baseline = build_composite_baseline(ad, top_k)
        universe = build_universe_baseline(ad, ruleset)

        # Evaluate
        strat_eval = evaluate_portfolio(strat, mhr, horizons)
        base_eval = evaluate_portfolio(baseline, mhr, horizons)
        univ_eval = evaluate_portfolio(universe, mhr, horizons)

        n = strat_eval.get("n_positions", 0)
        position_counts.append(n)

        for t in strat:
            all_tickers[t["ticker"]] += 1

        for h in horizons:
            v = strat_eval.get(f"weighted_resid_{h}d_pct")
            if v is not None:
                strat_weighted_resid[h].append(v)
            v = strat_eval.get(f"ew_resid_{h}d_pct")
            if v is not None:
                strat_ew_resid[h].append(v)
            v = base_eval.get(f"ew_resid_{h}d_pct")
            if v is not None:
                baseline_ew_resid[h].append(v)
            v = univ_eval.get(f"ew_resid_{h}d_pct")
            if v is not None:
                universe_ew_resid[h].append(v)
            v = strat_eval.get(f"hit_rate_{h}d_pct")
            if v is not None:
                hit_rates[h].append(v)
            v = strat_eval.get(f"p5_{h}d_pct")
            if v is not None:
                left_tail_p5[h].append(v)
            v = strat_eval.get(f"p10_{h}d_pct")
            if v is not None:
                left_tail_p10[h].append(v)
            v = strat_eval.get(f"max_loss_{h}d_pct")
            if v is not None:
                left_tail_max_loss[h].append(v)

        # Regime split
        regime = mhr.regime
        s60 = strat_eval.get("weighted_resid_60d_pct")
        if s60 is not None:
            regime_strat_resid_60.setdefault(regime, []).append(s60)

        # Turnover
        if prev_positions:
            to = compute_portfolio_turnover(prev_positions, strat)
            position_turnovers.append(to["position_turnover"])
            weight_turnovers.append(to["weight_turnover_pct"])
        prev_positions = strat

    # Aggregate
    result: Dict[str, Any] = {
        "n_snapshots": len(dates),
        "mean_n_positions": round(mean(position_counts), 1) if position_counts else 0,
        "unique_tickers": len(all_tickers),
    }

    for h in horizons:
        # Strategy metrics
        vals = strat_weighted_resid[h]
        if vals:
            result[f"strat_wt_resid_{h}d_mean"] = round(mean(vals), 2)
            result[f"strat_wt_resid_{h}d_median"] = round(median(vals), 2)
            result[f"strat_wt_resid_{h}d_winsor"] = round(mean(_winsorize(vals)), 2)
        else:
            result[f"strat_wt_resid_{h}d_mean"] = None
            result[f"strat_wt_resid_{h}d_median"] = None
            result[f"strat_wt_resid_{h}d_winsor"] = None

        # Baseline
        bvals = baseline_ew_resid[h]
        if bvals:
            result[f"baseline_ew_resid_{h}d_mean"] = round(mean(bvals), 2)
        else:
            result[f"baseline_ew_resid_{h}d_mean"] = None

        # Universe
        uvals = universe_ew_resid[h]
        if uvals:
            result[f"universe_ew_resid_{h}d_mean"] = round(mean(uvals), 2)
        else:
            result[f"universe_ew_resid_{h}d_mean"] = None

        # Spread
        if vals and bvals and len(vals) == len(bvals):
            spreads = [v - b for v, b in zip(vals, bvals)]
            result[f"spread_vs_topk_{h}d_mean"] = round(mean(spreads), 2)
            result[f"spread_vs_topk_{h}d_median"] = round(median(spreads), 2)
        elif vals and bvals:
            result[f"spread_vs_topk_{h}d_mean"] = round(
                (mean(vals) - mean(bvals)), 2)
            result[f"spread_vs_topk_{h}d_median"] = None
        else:
            result[f"spread_vs_topk_{h}d_mean"] = None
            result[f"spread_vs_topk_{h}d_median"] = None

        # Hit rate
        if hit_rates[h]:
            result[f"hit_rate_{h}d_mean"] = round(mean(hit_rates[h]), 1)
        else:
            result[f"hit_rate_{h}d_mean"] = None

        # Left tail
        if left_tail_p5[h]:
            result[f"p5_{h}d_mean"] = round(mean(left_tail_p5[h]), 2)
        else:
            result[f"p5_{h}d_mean"] = None
        if left_tail_p10[h]:
            result[f"p10_{h}d_mean"] = round(mean(left_tail_p10[h]), 2)
        else:
            result[f"p10_{h}d_mean"] = None
        if left_tail_max_loss[h]:
            result[f"max_loss_{h}d_mean"] = round(mean(left_tail_max_loss[h]), 2)
        else:
            result[f"max_loss_{h}d_mean"] = None

    # Turnover
    if position_turnovers:
        result["mean_pos_turnover"] = round(mean(position_turnovers), 3)
        result["mean_wt_turnover_pct"] = round(mean(weight_turnovers), 1)
    else:
        result["mean_pos_turnover"] = None
        result["mean_wt_turnover_pct"] = None

    # Regime split
    for regime, vals in sorted(regime_strat_resid_60.items()):
        result[f"regime_{regime}_n"] = len(vals)
        result[f"regime_{regime}_mean_60d"] = round(mean(vals), 2)

    # Concentration: top-3 tickers by frequency
    top3 = all_tickers.most_common(3)
    result["top3_tickers"] = ", ".join(
        f"{t}({c}/{len(dates)})" for t, c in top3
    )

    return result


# =============================================================================
# SCORING + SELECTION
# =============================================================================

def score_policy(row: Dict[str, Any]) -> Dict[str, Any]:
    """Compute a composite score for policy selection.

    Criteria (all equally weighted):
      1. 60d residual spread vs baseline (higher = better)
      2. Left tail P5 at 60d (less negative = better)
      3. Turnover (lower = better)
      4. Position count stability (penalize very low N)
    """
    spread_60 = row.get("spread_vs_topk_60d_mean")
    p5_60 = row.get("p5_60d_mean")
    turnover = row.get("mean_pos_turnover")
    n_pos = row.get("mean_n_positions", 0)

    # Normalize each to [0, 1] score (will rank-normalize across policies later)
    scores: Dict[str, Any] = {}
    scores["spread_60d_raw"] = spread_60 if spread_60 is not None else -999
    scores["p5_60d_raw"] = p5_60 if p5_60 is not None else -999
    scores["turnover_raw"] = -(turnover if turnover is not None else 999)  # lower is better
    scores["n_stability_raw"] = n_pos if n_pos >= 5 else 0  # penalize < 5 positions

    return scores


def rank_normalize(all_scores: List[Dict[str, Any]], keys: List[str]) -> List[float]:
    """Rank-normalize scores across policies and return composite."""
    n = len(all_scores)
    if n == 0:
        return []

    composites = [0.0] * n
    for key in keys:
        vals = [(i, s[key]) for i, s in enumerate(all_scores)]
        vals.sort(key=lambda x: x[1])  # ascending rank
        for rank, (idx, _) in enumerate(vals):
            composites[idx] += rank / max(n - 1, 1)  # normalize to [0, 1]

    # Average across criteria
    return [c / len(keys) for c in composites]


# =============================================================================
# OUTPUT
# =============================================================================

def write_matrix_csv(
    rows: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """Write the policy matrix comparison CSV."""
    if not rows:
        return

    fieldnames = [
        "rank", "config_label", "ruleset", "tier_filter", "top_k", "ruleset_id",
        "n_snapshots", "mean_n_positions", "unique_tickers",
        "strat_wt_resid_20d_mean", "strat_wt_resid_60d_mean",
        "strat_wt_resid_60d_winsor",
        "baseline_ew_resid_60d_mean",
        "spread_vs_topk_20d_mean", "spread_vs_topk_60d_mean",
        "hit_rate_20d_mean", "hit_rate_60d_mean",
        "p5_60d_mean", "p10_60d_mean", "max_loss_60d_mean",
        "mean_pos_turnover", "mean_wt_turnover_pct",
        "top3_tickers",
        "composite_score",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def generate_selection_report(
    ranked_rows: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """Generate narrative selection report."""
    lines: List[str] = []
    lines.append("=" * 72)
    lines.append("PHASE-2 POLICY SELECTION REPORT")
    lines.append("=" * 72)
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"Matrix: {len(ranked_rows)} configs evaluated")
    lines.append("")

    # Section 1: Selection table
    lines.append("1. SELECTION TABLE (ranked by composite score)")
    lines.append("-" * 72)
    hdr = (f"  {'#':>2} {'Config':<28} {'Resid60':>8} {'Spread':>7} "
           f"{'Hit%':>5} {'P5':>7} {'PTO':>5} {'N':>4} {'Score':>6}")
    lines.append(hdr)
    lines.append(f"  {'--':>2} {'-'*28} {'-'*8} {'-'*7} "
                 f"{'-'*5} {'-'*7} {'-'*5} {'-'*4} {'-'*6}")

    for row in ranked_rows:
        r60 = row.get("strat_wt_resid_60d_mean")
        sp = row.get("spread_vs_topk_60d_mean")
        hr = row.get("hit_rate_60d_mean")
        p5 = row.get("p5_60d_mean")
        pto = row.get("mean_pos_turnover")
        n = row.get("mean_n_positions", 0)
        cs = row.get("composite_score", 0)

        parts = [f"  {row.get('rank', ''):>2} {row.get('config_label', ''):<28}"]
        parts.append(f" {r60:>+8.2f}" if r60 is not None else f" {'n/a':>8}")
        parts.append(f" {sp:>+7.2f}" if sp is not None else f" {'n/a':>7}")
        parts.append(f" {hr:>5.1f}" if hr is not None else f" {'n/a':>5}")
        parts.append(f" {p5:>+7.1f}" if p5 is not None else f" {'n/a':>7}")
        parts.append(f" {pto:>5.1%}" if pto is not None else f" {'n/a':>5}")
        parts.append(f" {n:>4.0f}")
        parts.append(f" {cs:>6.3f}")
        lines.append("".join(parts))

    lines.append("")

    # Section 2: Winner analysis
    winner = ranked_rows[0] if ranked_rows else {}
    lines.append("2. RECOMMENDED PHASE-2 DEFAULT")
    lines.append("-" * 72)
    lines.append(f"  Config:      {winner.get('config_label', 'n/a')}")
    lines.append(f"  Ruleset:     {winner.get('ruleset', 'n/a')} "
                 f"(ID: {winner.get('ruleset_id', 'n/a')})")
    lines.append(f"  Tier filter: {winner.get('tier_filter', 'n/a')}")
    lines.append(f"  Top-K:       {winner.get('top_k', 'n/a')}")
    lines.append("")

    w60 = winner.get("strat_wt_resid_60d_mean")
    b60 = winner.get("baseline_ew_resid_60d_mean")
    sp60 = winner.get("spread_vs_topk_60d_mean")
    lines.append(f"  60d residual (strategy):  {w60:+.2f}%" if w60 is not None else "  60d residual: n/a")
    lines.append(f"  60d residual (baseline):  {b60:+.2f}%" if b60 is not None else "  60d baseline: n/a")
    lines.append(f"  60d spread vs baseline:   {sp60:+.2f}%" if sp60 is not None else "  60d spread: n/a")
    lines.append(f"  Hit rate 60d:             {winner.get('hit_rate_60d_mean', 'n/a')}")
    lines.append(f"  P5 60d:                   {winner.get('p5_60d_mean', 'n/a')}")
    lines.append(f"  P10 60d:                  {winner.get('p10_60d_mean', 'n/a')}")
    lines.append(f"  Max loss 60d:             {winner.get('max_loss_60d_mean', 'n/a')}")
    lines.append(f"  Mean positions:           {winner.get('mean_n_positions', 'n/a')}")
    lines.append(f"  Unique tickers:           {winner.get('unique_tickers', 'n/a')}")
    lines.append(f"  Position turnover:        {winner.get('mean_pos_turnover', 'n/a')}")
    lines.append(f"  Weight turnover:          {winner.get('mean_wt_turnover_pct', 'n/a')}%")
    lines.append(f"  Top-3 names:              {winner.get('top3_tickers', 'n/a')}")
    lines.append("")

    # Section 3: Runner-up comparison
    if len(ranked_rows) >= 2:
        lines.append("3. TOP-3 vs RUNNER-UPS")
        lines.append("-" * 72)
        for i, row in enumerate(ranked_rows[:5]):
            prefix = ">>>" if i == 0 else "   "
            sp = row.get("spread_vs_topk_60d_mean")
            p5 = row.get("p5_60d_mean")
            n = row.get("mean_n_positions", 0)
            pto = row.get("mean_pos_turnover")
            sp_str = f"{sp:+.2f}%" if sp is not None else "n/a"
            p5_str = f"{p5:+.1f}%" if p5 is not None else "n/a"
            pto_str = f"{pto:.1%}" if pto is not None else "n/a"
            lines.append(
                f"  {prefix} #{row.get('rank', '')} {row.get('config_label', ''):<28} "
                f"spread={sp_str:<8} P5={p5_str:<8} N={n:<5.0f} PTO={pto_str}"
            )
        lines.append("")

    # Section 4: Sensitivity analysis
    lines.append("4. SENSITIVITY ANALYSIS")
    lines.append("-" * 72)

    # Group by ruleset
    lines.append("  By ruleset (mean spread_60d across tier/topk combos):")
    by_ruleset: Dict[str, List[float]] = {}
    for row in ranked_rows:
        rs = row.get("ruleset", "")
        sp = row.get("spread_vs_topk_60d_mean")
        if sp is not None:
            by_ruleset.setdefault(rs, []).append(sp)
    for rs, vals in sorted(by_ruleset.items()):
        lines.append(f"    {rs:<20} mean_spread={mean(vals):+.2f}%  (n={len(vals)})")

    # Group by tier filter
    lines.append("  By tier filter:")
    by_tier: Dict[str, List[float]] = {}
    for row in ranked_rows:
        tf = row.get("tier_filter", "")
        sp = row.get("spread_vs_topk_60d_mean")
        if sp is not None:
            by_tier.setdefault(tf, []).append(sp)
    for tf, vals in sorted(by_tier.items()):
        lines.append(f"    {tf:<20} mean_spread={mean(vals):+.2f}%  (n={len(vals)})")

    # Group by top-K
    lines.append("  By top-K:")
    by_topk: Dict[int, List[float]] = {}
    for row in ranked_rows:
        tk = row.get("top_k", 0)
        sp = row.get("spread_vs_topk_60d_mean")
        if sp is not None:
            by_topk.setdefault(tk, []).append(sp)
    for tk, vals in sorted(by_topk.items()):
        lines.append(f"    K={tk:<17} mean_spread={mean(vals):+.2f}%  (n={len(vals)})")

    lines.append("")

    # Section 5: Caveats
    lines.append("5. CAVEATS")
    lines.append("-" * 72)
    lines.append("  - Composite score ranks on: spread_60d, P5_60d, turnover, N_stability.")
    lines.append("  - Equal weighting across criteria; no subjective preference applied.")
    lines.append("  - Small sample sizes: 22 usable archives (60d fence).")
    lines.append("  - Baseline is equal-weight top-K by composite rank (same K).")
    lines.append("  - No transaction costs, capacity, or rebalance timing modeled.")
    lines.append("  - Selection should be confirmed with forward walkthrough, not just backtest.")
    lines.append("")
    lines.append("=" * 72)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Policy Matrix Evaluation for Phase-2 Default Selection"
    )
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load rulesets
    rulesets: Dict[str, DecisionRuleset] = {}
    for name, path in RULESETS.items():
        rulesets[name] = DecisionRuleset.from_json(str(path))
        print(f"Loaded ruleset {name}: id={rulesets[name].ruleset_id}")

    # Build matrix
    matrix: List[Dict[str, Any]] = []
    for rs_name in RULESETS:
        for tf_name, tf_vals in TIER_FILTERS.items():
            for top_k in TOP_K_VALUES:
                matrix.append({
                    "ruleset_name": rs_name,
                    "ruleset": rulesets[rs_name],
                    "tier_filter_name": tf_name,
                    "tier_filter": tf_vals,
                    "top_k": top_k,
                    "config_label": f"{rs_name}/{tf_name}/K{top_k}",
                })

    print(f"\nPolicy Matrix: {len(matrix)} configurations")
    for m in matrix:
        print(f"  {m['config_label']}")

    # Discover archives
    archives = discover_archives(ARCHIVE_DIR, start=args.start, end=args.end)
    print(f"\nArchives: {len(archives)} found")

    if args.dry_run:
        print("\n[DRY RUN] Exiting.")
        return

    if not archives:
        print("ERROR: No archives found.")
        sys.exit(1)

    # Initialize providers
    print("\nInitializing providers ...")
    chained, _ms, csv_provider = init_providers()

    # Compute as-of fence
    snapshot_dates = [d for d, _ in archives]
    last_date = csv_provider.get_last_date()
    max_horizon = max(HORIZONS)
    usable_dates, fence_skipped = compute_as_of_fence(
        snapshot_dates, last_date, max_horizon
    )
    usable_set = set(usable_dates)
    print(f"Usable archives: {len(usable_dates)} "
          f"(skipped {len(fence_skipped)} past fence)")

    # Phase 1: Load archives and compute returns ONCE
    print("\nPhase 1: Loading archives + computing returns ...")
    archive_cache: Dict[str, ArchiveData] = {}
    return_cache: Dict[str, MultiHorizonReturns] = {}

    for i, (date_str, tar_path) in enumerate(archives):
        if date_str not in usable_set:
            continue
        print(f"  [{i+1}/{len(archives)}] {date_str} ... ", end="", flush=True)
        ad = load_archive_data(tar_path, date_str)
        archive_cache[date_str] = ad

        mhr = compute_multi_horizon_returns(
            chained, csv_provider, ad.tickers, date_str, HORIZONS
        )
        return_cache[date_str] = mhr
        print(f"ok (regime={mhr.regime})")

    print(f"  Cached {len(archive_cache)} archives, "
          f"{len(return_cache)} return sets")

    # Phase 2: Evaluate each policy
    print(f"\nPhase 2: Evaluating {len(matrix)} policies ...")
    results: List[Dict[str, Any]] = []

    for i, m in enumerate(matrix):
        label = m["config_label"]
        print(f"  [{i+1}/{len(matrix)}] {label} ... ", end="", flush=True)

        metrics = evaluate_policy(
            archive_cache, return_cache,
            m["ruleset"], m["tier_filter"], m["top_k"], HORIZONS,
        )

        row = {
            "config_label": label,
            "ruleset": m["ruleset_name"],
            "tier_filter": m["tier_filter_name"],
            "top_k": m["top_k"],
            "ruleset_id": m["ruleset"].ruleset_id,
            **metrics,
        }
        results.append(row)

        sp60 = metrics.get("spread_vs_topk_60d_mean")
        sp_str = f"{sp60:+.2f}%" if sp60 is not None else "n/a"
        print(f"spread_60d={sp_str}, N={metrics.get('mean_n_positions', 0)}")

    # Phase 3: Score and rank
    print("\nPhase 3: Scoring and ranking ...")
    all_scores = [score_policy(r) for r in results]
    score_keys = ["spread_60d_raw", "p5_60d_raw", "turnover_raw", "n_stability_raw"]
    composites = rank_normalize(all_scores, score_keys)

    for r, c in zip(results, composites):
        r["composite_score"] = round(c, 4)

    # Sort by composite (higher = better)
    results.sort(key=lambda x: x["composite_score"], reverse=True)
    for i, r in enumerate(results, 1):
        r["rank"] = i

    # Output
    csv_path = output_dir / "policy_matrix.csv"
    write_matrix_csv(results, csv_path)
    print(f"\n  Wrote {csv_path}")

    json_path = output_dir / "policy_matrix_details.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "generated": datetime.now().isoformat(timespec="seconds"),
            "n_configs": len(results),
            "n_snapshots": len(archive_cache),
            "results": results,
        }, f, indent=2, default=str)
    print(f"  Wrote {json_path}")

    report_path = output_dir / "policy_selection_report.txt"
    generate_selection_report(results, report_path)
    print(f"  Wrote {report_path}")

    # Print headline
    print("\n" + "=" * 60)
    print("POLICY MATRIX RESULTS")
    print("=" * 60)
    print(f"  {'#':>2} {'Config':<30} {'Spread60':>9} {'P5_60':>7} "
          f"{'Hit60':>6} {'PTO':>6} {'N':>4}")
    print(f"  {'--':>2} {'-'*30} {'-'*9} {'-'*7} {'-'*6} {'-'*6} {'-'*4}")
    for r in results[:10]:
        sp = r.get("spread_vs_topk_60d_mean")
        p5 = r.get("p5_60d_mean")
        hr = r.get("hit_rate_60d_mean")
        pto = r.get("mean_pos_turnover")
        n = r.get("mean_n_positions", 0)
        parts = [f"  {r['rank']:>2} {r['config_label']:<30}"]
        parts.append(f" {sp:>+9.2f}%" if sp is not None else f" {'n/a':>10}")
        parts.append(f" {p5:>+7.1f}" if p5 is not None else f" {'n/a':>7}")
        parts.append(f" {hr:>6.1f}" if hr is not None else f" {'n/a':>6}")
        parts.append(f" {pto:>6.1%}" if pto is not None else f" {'n/a':>6}")
        parts.append(f" {n:>4.0f}")
        print("".join(parts))

    winner = results[0]
    print(f"\n  RECOMMENDED: {winner['config_label']}")
    print(f"  Ruleset:     {winner['ruleset']} (ID: {winner['ruleset_id']})")
    sp = winner.get("spread_vs_topk_60d_mean")
    print(f"  Spread 60d:  {sp:+.2f}%" if sp is not None else "  Spread 60d: n/a")
    print("=" * 60)


if __name__ == "__main__":
    main()
