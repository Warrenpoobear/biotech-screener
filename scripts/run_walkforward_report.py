#!/usr/bin/env python3
"""
Walk-Forward Validation Report

Orchestrates the walk-forward panel export, computes tier/band separation
metrics, stability measures, and coverage summaries. Produces:
  - artifacts/walkforward_panel.csv
  - artifacts/walkforward_report.json
  - artifacts/walkforward_report.md

Usage:
    python scripts/run_walkforward_report.py [options]

    --ruleset PATH       DecisionRuleset JSON (default: built-in default)
    --tier-filter A,B    Comma-separated tiers for portfolio (default: A,B)
    --top-k 20           Max portfolio positions (default: 20)
    --output-dir PATH    Output directory (default: artifacts/)
    --date-min DATE      Archive start date filter (YYYY-MM-DD)
    --date-max DATE      Archive end date filter (YYYY-MM-DD)
    --suffix TAG         Suffix for output files (e.g., '2025' → walkforward_panel_2025.csv)
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
from typing import Any, Dict, List, Optional

# Add project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from decision_engine import DecisionRuleset, DEFAULT_RULESET
from run_decision_strategy_backtest import (
    HORIZONS,
    PANEL_COLUMNS,
    run_strategy_backtest,
    write_panel_csv,
)
from run_decision_ruleset_sweep import init_providers
from run_rank_ic_backtest import ARCHIVE_DIR, discover_archives
from outcome_provenance import build_provenance
from decision_engine_codes import registry_fingerprint

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts"


# =============================================================================
# METRICS
# =============================================================================

def compute_tier_separation(
    panel_rows: List[Dict[str, Any]],
    return_col: str = "fwd_ret_60d",
    dd_col: str = "fwd_max_dd_60d",
) -> List[Dict[str, Any]]:
    """Compute per-tier metrics from the panel.

    Returns a list of dicts, one per tier (A/B/C/D), with:
    count, mean_ret, median_ret, hit_rate, mean_max_dd, median_max_dd
    """
    by_tier: Dict[str, List[Dict[str, Any]]] = {}
    for row in panel_rows:
        tier = row.get("tier", "")
        if tier:
            by_tier.setdefault(tier, []).append(row)

    results = []
    for tier in ["A", "B", "C", "D"]:
        rows = by_tier.get(tier, [])
        rets = []
        dds = []
        for r in rows:
            v = r.get(return_col)
            if v != "" and v is not None:
                try:
                    rets.append(float(v))
                except (ValueError, TypeError):
                    pass
            dd_v = r.get(dd_col)
            if dd_v != "" and dd_v is not None:
                try:
                    dds.append(float(dd_v))
                except (ValueError, TypeError):
                    pass

        entry: Dict[str, Any] = {
            "tier": tier,
            "count": len(rows),
            "with_returns": len(rets),
        }
        if rets:
            entry["mean_ret"] = round(mean(rets), 2)
            entry["median_ret"] = round(median(rets), 2)
            entry["hit_rate"] = round(sum(1 for r in rets if r > 0) / len(rets) * 100, 1)
        else:
            entry["mean_ret"] = None
            entry["median_ret"] = None
            entry["hit_rate"] = None

        if dds:
            entry["mean_max_dd"] = round(mean(dds), 2)
            entry["median_max_dd"] = round(median(dds), 2)
        else:
            entry["mean_max_dd"] = None
            entry["median_max_dd"] = None

        results.append(entry)

    return results


def compute_band_separation(
    panel_rows: List[Dict[str, Any]],
    return_col: str = "fwd_ret_60d",
    dd_col: str = "fwd_max_dd_60d",
) -> List[Dict[str, Any]]:
    """Compute per-band metrics from the panel. Same structure as tier separation."""
    by_band: Dict[str, List[Dict[str, Any]]] = {}
    for row in panel_rows:
        band = row.get("band", "")
        if band:
            by_band.setdefault(band, []).append(row)

    results = []
    for band in ["L", "M", "S", "XS"]:
        rows = by_band.get(band, [])
        rets = []
        dds = []
        for r in rows:
            v = r.get(return_col)
            if v != "" and v is not None:
                try:
                    rets.append(float(v))
                except (ValueError, TypeError):
                    pass
            dd_v = r.get(dd_col)
            if dd_v != "" and dd_v is not None:
                try:
                    dds.append(float(dd_v))
                except (ValueError, TypeError):
                    pass

        entry: Dict[str, Any] = {
            "band": band,
            "count": len(rows),
            "with_returns": len(rets),
        }
        if rets:
            entry["mean_ret"] = round(mean(rets), 2)
            entry["median_ret"] = round(median(rets), 2)
            entry["hit_rate"] = round(sum(1 for r in rets if r > 0) / len(rets) * 100, 1)
        else:
            entry["mean_ret"] = None
            entry["median_ret"] = None
            entry["hit_rate"] = None

        if dds:
            entry["mean_max_dd"] = round(mean(dds), 2)
            entry["median_max_dd"] = round(median(dds), 2)
        else:
            entry["mean_max_dd"] = None
            entry["median_max_dd"] = None

        results.append(entry)

    return results


def compute_strength_distribution(
    panel_rows: List[Dict[str, Any]],
    return_col: str = "fwd_ret_60d",
    dd_col: str = "fwd_max_dd_60d",
) -> List[Dict[str, Any]]:
    """Compute per-catalyst-strength-band metrics from the panel.

    Returns a list of dicts, one per strength band (near/mid/far/missing),
    with: count, mean_ret, median_ret, hit_rate, mean_max_dd.
    """
    by_strength: Dict[str, List[Dict[str, Any]]] = {}
    for row in panel_rows:
        strength = row.get("catalyst_strength", "")
        if strength:
            by_strength.setdefault(strength, []).append(row)

    results = []
    for band in ["near", "mid", "far", "missing"]:
        rows = by_strength.get(band, [])
        rets = []
        dds = []
        for r in rows:
            v = r.get(return_col)
            if v != "" and v is not None:
                try:
                    rets.append(float(v))
                except (ValueError, TypeError):
                    pass
            dd_v = r.get(dd_col)
            if dd_v != "" and dd_v is not None:
                try:
                    dds.append(float(dd_v))
                except (ValueError, TypeError):
                    pass

        entry: Dict[str, Any] = {
            "strength": band,
            "count": len(rows),
            "with_returns": len(rets),
        }
        if rets:
            entry["mean_ret"] = round(mean(rets), 2)
            entry["median_ret"] = round(median(rets), 2)
            entry["hit_rate"] = round(sum(1 for r in rets if r > 0) / len(rets) * 100, 1)
        else:
            entry["mean_ret"] = None
            entry["median_ret"] = None
            entry["hit_rate"] = None

        if dds:
            entry["mean_max_dd"] = round(mean(dds), 2)
            entry["median_max_dd"] = round(median(dds), 2)
        else:
            entry["mean_max_dd"] = None
            entry["median_max_dd"] = None

        results.append(entry)

    return results


def compute_eligible_comparison(
    panel_rows: List[Dict[str, Any]],
    return_col: str = "fwd_ret_60d",
) -> Dict[str, Any]:
    """Compare eligible vs ineligible forward returns."""
    eligible_rets: List[float] = []
    ineligible_rets: List[float] = []
    for row in panel_rows:
        v = row.get(return_col)
        if v == "" or v is None:
            continue
        try:
            ret = float(v)
        except (ValueError, TypeError):
            continue
        if str(row.get("eligible", "0")) == "1":
            eligible_rets.append(ret)
        else:
            ineligible_rets.append(ret)

    result: Dict[str, Any] = {}
    for label, rets in [("eligible", eligible_rets), ("ineligible", ineligible_rets)]:
        if rets:
            result[f"{label}_count"] = len(rets)
            result[f"{label}_mean_ret"] = round(mean(rets), 2)
            result[f"{label}_median_ret"] = round(median(rets), 2)
            result[f"{label}_hit_rate"] = round(
                sum(1 for r in rets if r > 0) / len(rets) * 100, 1
            )
        else:
            result[f"{label}_count"] = 0
            result[f"{label}_mean_ret"] = None
            result[f"{label}_median_ret"] = None
            result[f"{label}_hit_rate"] = None

    return result


def compute_stability_metrics(
    panel_rows: List[Dict[str, Any]],
    top_n: int = 25,
) -> Dict[str, Any]:
    """Compute top-N ticker overlap (Jaccard) between consecutive dates.

    Also computes mean position turnover (symmetric diff / union).
    """
    # Group by date
    by_date: Dict[str, List[str]] = {}
    for row in panel_rows:
        d = row.get("as_of_date", "")
        # Only count eligible tickers with weight > 0 (portfolio members)
        try:
            w = float(row.get("weight", 0))
        except (ValueError, TypeError):
            w = 0.0
        if w > 0:
            by_date.setdefault(d, []).append(row.get("ticker", ""))

    sorted_dates = sorted(by_date.keys())
    if len(sorted_dates) < 2:
        return {"mean_top_n_overlap": None, "mean_turnover": None, "n_transitions": 0}

    overlaps: List[float] = []
    turnovers: List[float] = []
    for i in range(1, len(sorted_dates)):
        prev_set = set(by_date[sorted_dates[i - 1]][:top_n])
        curr_set = set(by_date[sorted_dates[i]][:top_n])
        union = prev_set | curr_set
        if union:
            jaccard = len(prev_set & curr_set) / len(union)
            overlaps.append(jaccard)
            turnover = len(prev_set.symmetric_difference(curr_set)) / len(union)
            turnovers.append(turnover)

    return {
        "mean_top_n_overlap": round(mean(overlaps) * 100, 1) if overlaps else None,
        "mean_turnover": round(mean(turnovers) * 100, 1) if turnovers else None,
        "n_transitions": len(overlaps),
    }


def compute_cost_summary(
    panel_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute cost statistics across portfolio rows (weight > 0).

    Returns dict with median/mean/p95 cost bps, participation stats,
    low-ADV count, and a capacity note.
    """
    costs: List[float] = []
    participations: List[float] = []
    low_adv_count = 0

    for row in panel_rows:
        try:
            w = float(row.get("weight", 0))
        except (ValueError, TypeError):
            w = 0.0
        if w <= 0:
            continue

        try:
            cost = float(row.get("est_cost_bps", 0))
        except (ValueError, TypeError):
            cost = 0.0
        costs.append(cost)

        try:
            part = float(row.get("participation_pct", 0))
        except (ValueError, TypeError):
            part = 0.0
        participations.append(part)

        try:
            adv = float(row.get("adv_dollars", 0))
        except (ValueError, TypeError):
            adv = 0.0
        if adv < 500_000:
            low_adv_count += 1

    if not costs:
        return {
            "n_portfolio_rows": 0,
            "median_cost_bps": None,
            "mean_cost_bps": None,
            "p95_cost_bps": None,
            "median_participation_pct": None,
            "p95_participation_pct": None,
            "low_adv_count": 0,
            "capacity_note": "No portfolio rows with cost data",
        }

    sorted_costs = sorted(costs)
    sorted_parts = sorted(participations)
    p95_idx = int(len(sorted_costs) * 0.95)

    med_cost = round(median(sorted_costs), 1)
    return {
        "n_portfolio_rows": len(costs),
        "median_cost_bps": med_cost,
        "mean_cost_bps": round(mean(costs), 1),
        "p95_cost_bps": round(sorted_costs[min(p95_idx, len(sorted_costs) - 1)], 1),
        "median_participation_pct": round(median(sorted_parts), 2),
        "p95_participation_pct": round(sorted_parts[min(p95_idx, len(sorted_parts) - 1)], 2),
        "low_adv_count": low_adv_count,
        "capacity_note": f"At $50M AUM: median {med_cost} bps round-trip",
    }


def compute_coverage_summary(
    panel_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Summarize data coverage in the panel."""
    total = len(panel_rows)
    with_ret = sum(1 for r in panel_rows if r.get("fwd_ret_60d") not in ("", None))
    with_dd = sum(1 for r in panel_rows if r.get("fwd_max_dd_60d") not in ("", None))

    # Missing reason breakdown
    reasons: Counter = Counter()
    for r in panel_rows:
        reason = r.get("fwd_dd_missing_reason", "")
        if reason:
            reasons[reason] += 1

    return {
        "total_rows": total,
        "with_fwd_returns": with_ret,
        "with_fwd_returns_pct": round(with_ret / total * 100, 1) if total else 0,
        "with_fwd_max_dd": with_dd,
        "with_fwd_max_dd_pct": round(with_dd / total * 100, 1) if total else 0,
        "missing_reason_breakdown": dict(reasons.most_common()),
    }


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_walkforward_report_md(
    tier_sep: List[Dict[str, Any]],
    band_sep: List[Dict[str, Any]],
    eligible_cmp: Dict[str, Any],
    stability: Dict[str, Any],
    coverage: Dict[str, Any],
    config: Dict[str, Any],
    strength_dist: Optional[List[Dict[str, Any]]] = None,
    tier_sep_net: Optional[List[Dict[str, Any]]] = None,
    cost_summary: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate the markdown report string."""
    lines: List[str] = []
    lines.append("# Walk-Forward Validation Report")
    lines.append("")
    lines.append(f"Generated: {config.get('generated', '')}")
    lines.append(
        f"Ruleset: {config.get('ruleset_id', 'default')} | "
        f"Tier filter: {config.get('tier_filter', [])} | "
        f"Top-K: {config.get('top_k', 20)} | "
        f"Snapshots: {config.get('n_snapshots', 0)} | "
        f"Date range: {config.get('date_range', 'n/a')}"
    )
    lines.append("")

    # Tier separation
    lines.append("## Tier Separation (60d forward returns)")
    lines.append("")
    lines.append("| Tier | N | Mean % | Median % | Hit Rate % | Mean Max-DD % |")
    lines.append("|------|---|--------|----------|------------|---------------|")
    for t in tier_sep:
        n = t["count"]
        m = f"{t['mean_ret']:+.2f}" if t["mean_ret"] is not None else "n/a"
        md = f"{t['median_ret']:+.2f}" if t["median_ret"] is not None else "n/a"
        hr = f"{t['hit_rate']:.1f}" if t["hit_rate"] is not None else "n/a"
        dd = f"{t['mean_max_dd']:.2f}" if t["mean_max_dd"] is not None else "n/a"
        lines.append(f"| {t['tier']} | {n} | {m} | {md} | {hr} | {dd} |")
    lines.append("")

    # Band separation
    lines.append("## Band Separation (60d forward returns)")
    lines.append("")
    lines.append("| Band | N | Mean % | Median % | Hit Rate % | Mean Max-DD % |")
    lines.append("|------|---|--------|----------|------------|---------------|")
    for b in band_sep:
        n = b["count"]
        m = f"{b['mean_ret']:+.2f}" if b["mean_ret"] is not None else "n/a"
        md = f"{b['median_ret']:+.2f}" if b["median_ret"] is not None else "n/a"
        hr = f"{b['hit_rate']:.1f}" if b["hit_rate"] is not None else "n/a"
        dd = f"{b['mean_max_dd']:.2f}" if b["mean_max_dd"] is not None else "n/a"
        lines.append(f"| {b['band']} | {n} | {m} | {md} | {hr} | {dd} |")
    lines.append("")

    # Catalyst strength distribution
    if strength_dist:
        lines.append("## Catalyst Strength Distribution (60d forward returns)")
        lines.append("")
        lines.append("| Strength | N | Mean % | Median % | Hit Rate % | Mean Max-DD % |")
        lines.append("|----------|---|--------|----------|------------|---------------|")
        for s in strength_dist:
            n = s["count"]
            m = f"{s['mean_ret']:+.2f}" if s["mean_ret"] is not None else "n/a"
            md = f"{s['median_ret']:+.2f}" if s["median_ret"] is not None else "n/a"
            hr = f"{s['hit_rate']:.1f}" if s["hit_rate"] is not None else "n/a"
            dd = f"{s['mean_max_dd']:.2f}" if s["mean_max_dd"] is not None else "n/a"
            lines.append(f"| {s['strength']} | {n} | {m} | {md} | {hr} | {dd} |")
        lines.append("")

    # Cost analysis (gross vs net)
    if tier_sep_net and cost_summary:
        lines.append("## Cost Analysis")
        lines.append("")

        # Gross vs net tier separation table
        net_by_tier = {t["tier"]: t for t in tier_sep_net}
        lines.append("### Gross vs Net Tier Separation (60d forward returns)")
        lines.append("")
        lines.append("| Tier | N | Gross Mean % | Net Mean % | Gross Hit % | Net Hit % |")
        lines.append("|------|---|-------------|-----------|------------|----------|")
        for t in tier_sep:
            tier = t["tier"]
            nt = net_by_tier.get(tier, {})
            gm = f"{t['mean_ret']:+.2f}" if t["mean_ret"] is not None else "n/a"
            nm = f"{nt.get('mean_ret', None):+.2f}" if nt.get("mean_ret") is not None else "n/a"
            gh = f"{t['hit_rate']:.1f}" if t["hit_rate"] is not None else "n/a"
            nh = f"{nt.get('hit_rate', None):.1f}" if nt.get("hit_rate") is not None else "n/a"
            lines.append(f"| {tier} | {t['count']} | {gm} | {nm} | {gh} | {nh} |")
        lines.append("")

        # Cost distribution stats
        lines.append("### Cost Distribution (portfolio positions)")
        lines.append("")
        lines.append(f"- Portfolio rows: {cost_summary.get('n_portfolio_rows', 0)}")
        med = cost_summary.get("median_cost_bps")
        lines.append(f"- Median round-trip cost: {med} bps" if med is not None else "- Median round-trip cost: n/a")
        mn = cost_summary.get("mean_cost_bps")
        lines.append(f"- Mean round-trip cost: {mn} bps" if mn is not None else "- Mean round-trip cost: n/a")
        p95 = cost_summary.get("p95_cost_bps")
        lines.append(f"- P95 round-trip cost: {p95} bps" if p95 is not None else "- P95 round-trip cost: n/a")
        med_p = cost_summary.get("median_participation_pct")
        lines.append(f"- Median participation: {med_p}%" if med_p is not None else "- Median participation: n/a")
        p95_p = cost_summary.get("p95_participation_pct")
        lines.append(f"- P95 participation: {p95_p}%" if p95_p is not None else "- P95 participation: n/a")
        lines.append(f"- Low-ADV positions (< $500K): {cost_summary.get('low_adv_count', 0)}")
        lines.append(f"- {cost_summary.get('capacity_note', '')}")
        lines.append("")

    # Eligible vs ineligible
    lines.append("## Eligible vs Ineligible (60d forward returns)")
    lines.append("")
    lines.append("| Group | N | Mean % | Median % | Hit Rate % |")
    lines.append("|-------|---|--------|----------|------------|")
    for group in ["eligible", "ineligible"]:
        n = eligible_cmp.get(f"{group}_count", 0)
        m = eligible_cmp.get(f"{group}_mean_ret")
        md = eligible_cmp.get(f"{group}_median_ret")
        hr = eligible_cmp.get(f"{group}_hit_rate")
        m_s = f"{m:+.2f}" if m is not None else "n/a"
        md_s = f"{md:+.2f}" if md is not None else "n/a"
        hr_s = f"{hr:.1f}" if hr is not None else "n/a"
        lines.append(f"| {group.capitalize()} | {n} | {m_s} | {md_s} | {hr_s} |")
    lines.append("")

    # Stability
    lines.append("## Stability")
    lines.append("")
    overlap = stability.get("mean_top_n_overlap")
    turnover = stability.get("mean_turnover")
    lines.append(f"- Mean top-25 overlap (Jaccard): "
                 f"{overlap:.1f}%" if overlap is not None else "- Mean top-25 overlap: n/a")
    lines.append(f"- Mean position turnover: "
                 f"{turnover:.1f}%" if turnover is not None else "- Mean position turnover: n/a")
    lines.append(f"- Date transitions: {stability.get('n_transitions', 0)}")
    lines.append("")

    # Coverage
    lines.append("## Coverage")
    lines.append("")
    lines.append(
        f"- Panel rows: {coverage['total_rows']} | "
        f"With fwd returns: {coverage['with_fwd_returns']} "
        f"({coverage['with_fwd_returns_pct']}%) | "
        f"With max-DD: {coverage['with_fwd_max_dd']} "
        f"({coverage['with_fwd_max_dd_pct']}%)"
    )
    reasons = coverage.get("missing_reason_breakdown", {})
    if reasons:
        lines.append("- Missing reason breakdown:")
        for reason, count in reasons.items():
            lines.append(f"  - {reason}: {count}")
    lines.append("")

    return "\n".join(lines) + "\n"


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Walk-Forward Validation Report"
    )
    parser.add_argument(
        "--ruleset", type=str, default=None,
        help="DecisionRuleset JSON path (default: built-in default)",
    )
    parser.add_argument(
        "--tier-filter", type=str, default="A,B",
        help="Comma-separated tiers for portfolio (default: A,B)",
    )
    parser.add_argument(
        "--top-k", type=int, default=20,
        help="Max portfolio positions (default: 20)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory (default: artifacts/)",
    )
    parser.add_argument(
        "--date-min", "--start", type=str, default=None, dest="date_min",
        help="Archive start date filter (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--date-max", "--end", type=str, default=None, dest="date_max",
        help="Archive end date filter (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--suffix", type=str, default=None,
        help="Suffix for output files (e.g., '2025' → walkforward_panel_2025.csv)",
    )
    parser.add_argument(
        "--archive-dir", type=str, default=None,
        help=f"Archive directory (default: {ARCHIVE_DIR})",
    )
    args = parser.parse_args()

    # Load ruleset
    if args.ruleset:
        ruleset = DecisionRuleset.from_json(args.ruleset)
        print(f"Loaded ruleset from {args.ruleset}")
    else:
        ruleset = DEFAULT_RULESET
        print(f"Using default ruleset (ID={ruleset.ruleset_id})")

    tier_filter = [t.strip().upper() for t in args.tier_filter.split(",")]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover archives
    arch_dir = Path(args.archive_dir) if args.archive_dir else ARCHIVE_DIR
    archives = discover_archives(arch_dir, start=args.date_min, end=args.date_max)
    if args.date_min or args.date_max:
        print(f"Date filter: {args.date_min or '*'} to {args.date_max or '*'}")
    print(f"Archives: {len(archives)} found")

    if not archives:
        print("ERROR: No archives found.")
        sys.exit(1)

    # Initialize providers
    print("Initializing returns providers ...")
    chained, _ms_provider, csv_provider = init_providers()

    # Run strategy backtest with panel emission
    print("Running walk-forward backtest with panel emission ...")
    results, panel_rows = run_strategy_backtest(
        archives=archives,
        chained=chained,
        csv_provider=csv_provider,
        ruleset=ruleset,
        tier_filter=tier_filter,
        top_k=args.top_k,
        horizons=HORIZONS,
        emit_panel=True,
    )

    if not results:
        print("ERROR: No usable snapshots produced results.")
        sys.exit(1)

    print(f"\nPanel: {len(panel_rows)} rows across {len(results)} snapshots")

    # File naming
    sfx = f"_{args.suffix}" if args.suffix else ""

    # Write panel CSV
    panel_path = output_dir / f"walkforward_panel{sfx}.csv"
    write_panel_csv(panel_rows, panel_path)
    print(f"Wrote {panel_path}")

    # Compute metrics
    print("Computing metrics ...")
    tier_sep = compute_tier_separation(panel_rows)
    band_sep = compute_band_separation(panel_rows)
    strength_dist = compute_strength_distribution(panel_rows)
    eligible_cmp = compute_eligible_comparison(panel_rows)
    stability = compute_stability_metrics(panel_rows)
    coverage = compute_coverage_summary(panel_rows)
    tier_sep_net = compute_tier_separation(panel_rows, return_col="fwd_ret_60d_net")
    cost_summ = compute_cost_summary(panel_rows)

    config = {
        "ruleset_id": ruleset.ruleset_id,
        "tier_filter": tier_filter,
        "top_k": args.top_k,
        "n_snapshots": len(results),
        "date_range": f"{results[0].date_str} to {results[-1].date_str}" if results else "n/a",
        "date_min": args.date_min,
        "date_max": args.date_max,
        "generated": datetime.now().isoformat(timespec="seconds"),
    }

    # Provenance
    provenance = build_provenance(
        panel_path=panel_path,
        ruleset_id=ruleset.ruleset_id,
        explanation_registry_fingerprint=registry_fingerprint(),
    )

    # Write JSON
    report_json = {
        "config": config,
        "provenance": provenance,
        "tier_separation": tier_sep,
        "tier_separation_net": tier_sep_net,
        "band_separation": band_sep,
        "strength_distribution": strength_dist,
        "eligible_comparison": eligible_cmp,
        "stability": stability,
        "coverage": coverage,
        "cost_summary": cost_summ,
    }
    json_path = output_dir / f"walkforward_report{sfx}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_json, f, indent=2, default=str)
    print(f"Wrote {json_path}")

    # Write markdown
    md_content = generate_walkforward_report_md(
        tier_sep, band_sep, eligible_cmp, stability, coverage, config,
        strength_dist=strength_dist,
        tier_sep_net=tier_sep_net,
        cost_summary=cost_summ,
    )
    md_path = output_dir / f"walkforward_report{sfx}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Wrote {md_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("TIER SEPARATION (60d)")
    print("=" * 60)
    for t in tier_sep:
        m = f"{t['mean_ret']:+.2f}%" if t["mean_ret"] is not None else "n/a"
        hr = f"{t['hit_rate']:.1f}%" if t["hit_rate"] is not None else "n/a"
        dd = f"{t['mean_max_dd']:.2f}%" if t["mean_max_dd"] is not None else "n/a"
        print(f"  {t['tier']}: n={t['count']}, mean={m}, hit_rate={hr}, max_dd={dd}")
    print("=" * 60)


if __name__ == "__main__":
    main()
