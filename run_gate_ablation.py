#!/usr/bin/env python3
"""
Gate Ablation + D-tier Audit Report

Quantifies each eligibility gate's contribution to D-tier assignments,
measures whether excluded names are genuinely dangerous (left-tail returns),
and models the impact of softening/removing individual gates on tier
distribution and A-vs-C separation.

Decision Engine v1.1.0 has 4 eligibility gates, but only 2 are active:
  - deep_drawdown: parameterized at drawdown_gate=-0.40 in DecisionRuleset
  - sev3: hardcoded check severity == "SEV3" in _compute_eligibility()
  - fundamental_red_flag: INERT (always False in backfill_decision_columns)
  - adv_fail: INERT (no attn_flags/flags data in archives)

Usage:
    python run_gate_ablation.py [options]

    --output-dir PATH     Output directory (default: output/)
    --start DATE          Start date filter for archives
    --end DATE            End date filter for archives
    --dry-run             Print archive count + scenario names, don't run
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

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from backfill_decision_columns import (
    _load_catalyst_map,
    _load_decision_inputs,
    _load_defensive_map,
    _load_holdings_map,
    build_rec,
)
from decision_engine import DecisionRuleset, compute_decision_fields
from run_decision_ruleset_sweep import (
    ArchiveData,
    ForwardReturnData,
    compute_snapshot_returns,
    init_providers,
    load_archive_data,
)
from run_rank_ic_backtest import (
    ARCHIVE_DIR,
    OUTPUT_DIR,
    _winsorize,
    compute_as_of_fence,
    discover_archives,
)


# =============================================================================
# CONSTANTS
# =============================================================================

KNOWN_GATES = ["fundamental_red_flag", "sev3", "deep_drawdown", "adv_fail"]

ABLATION_SCENARIOS = [
    "baseline",
    "no_drawdown",
    "no_sev3",
    "no_gates",
]


# =============================================================================
# HELPERS
# =============================================================================

def _percentile(vals: List[float], pct: float) -> float:
    """Compute the given percentile from a sorted list. pct in [0, 100]."""
    if not vals:
        return 0.0
    s = sorted(vals)
    k = (len(s) - 1) * pct / 100.0
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] + frac * (s[hi] - s[lo])


def _safe_div(num: float, denom: float) -> float:
    """Safe division returning 0.0 on zero denom."""
    return num / denom if denom else 0.0


# =============================================================================
# PHASE 1: GATE FREQUENCY
# =============================================================================

def count_gate_frequencies(
    archive_cache: Dict[str, ArchiveData],
    ruleset: DecisionRuleset,
) -> Dict[str, Any]:
    """Count per-gate hit rates across all archives for dev-stage tickers.

    Returns dict with per_date and pooled summaries, plus inert_gates list.
    """
    per_date: Dict[str, Dict[str, Any]] = {}
    pooled_gate_counts: Counter = Counter()
    pooled_overlap_counts: Counter = Counter()
    pooled_n_dev = 0
    pooled_n_ineligible = 0

    for date_str in sorted(archive_cache.keys()):
        ad = archive_cache[date_str]
        gate_counts: Counter = Counter()
        overlap_counts: Counter = Counter()
        n_dev = 0
        n_ineligible = 0

        for ticker in ad.tickers:
            archetype = ad.archetypes.get(ticker, "")
            if archetype != "drug_developer":
                continue
            n_dev += 1

            rec = ad.recs.get(ticker)
            if rec is None:
                continue
            opt = ad.optionalities.get(ticker)

            fields = compute_decision_fields(rec, archetype, opt, ruleset=ruleset)
            reasons_str = fields.get("ineligible_reasons", "")
            if not reasons_str:
                continue

            n_ineligible += 1
            reasons = reasons_str.split("|")

            for r in reasons:
                gate_counts[r] += 1

            # Track overlaps (pairs of gates)
            if len(reasons) > 1:
                for i in range(len(reasons)):
                    for j in range(i + 1, len(reasons)):
                        pair = "+".join(sorted([reasons[i], reasons[j]]))
                        overlap_counts[pair] += 1

        per_date[date_str] = {
            "n_dev": n_dev,
            "n_ineligible": n_ineligible,
            "gates": {
                g: {"n": gate_counts.get(g, 0),
                    "pct": round(_safe_div(gate_counts.get(g, 0), n_dev) * 100, 1)}
                for g in KNOWN_GATES
            },
            "overlap": {
                k: {"n": v, "pct": round(_safe_div(v, n_dev) * 100, 1)}
                for k, v in overlap_counts.items()
            },
        }

        pooled_n_dev += n_dev
        pooled_n_ineligible += n_ineligible
        pooled_gate_counts += gate_counts
        pooled_overlap_counts += overlap_counts

    # Identify inert gates (zero hits pooled)
    inert_gates = [g for g in KNOWN_GATES if pooled_gate_counts.get(g, 0) == 0]

    pooled = {
        "n_dev": pooled_n_dev,
        "n_ineligible": pooled_n_ineligible,
        "gates": {
            g: {"n": pooled_gate_counts.get(g, 0),
                "pct": round(_safe_div(pooled_gate_counts.get(g, 0), pooled_n_dev) * 100, 1)}
            for g in KNOWN_GATES
        },
        "overlap": {
            k: {"n": v, "pct": round(_safe_div(v, pooled_n_dev) * 100, 1)}
            for k, v in pooled_overlap_counts.items()
        },
    }

    return {
        "per_date": per_date,
        "pooled": pooled,
        "inert_gates": inert_gates,
    }


# =============================================================================
# PHASE 2: D vs ABC RETURN STATS
# =============================================================================

def compute_tier_return_stats(
    archive_cache: Dict[str, ArchiveData],
    return_cache: Dict[str, ForwardReturnData],
    ruleset: DecisionRuleset,
) -> Dict[str, Any]:
    """Pool all dev tickers across snapshots, group by tier, compute return stats.

    Returns per-tier stats with left-tail analysis for D-tier.
    """
    tier_raw: Dict[str, List[float]] = {}
    tier_resid: Dict[str, List[float]] = {}
    tier_names: Dict[str, List[Tuple[str, str]]] = {}  # tier -> [(ticker, date)]

    for date_str in sorted(archive_cache.keys()):
        ad = archive_cache[date_str]
        fd = return_cache.get(date_str)
        if fd is None:
            continue

        for ticker in ad.tickers:
            archetype = ad.archetypes.get(ticker, "")
            if archetype != "drug_developer":
                continue
            if ticker not in fd.raw_returns:
                continue

            rec = ad.recs.get(ticker)
            if rec is None:
                continue
            opt = ad.optionalities.get(ticker)

            fields = compute_decision_fields(rec, archetype, opt, ruleset=ruleset)
            tier = fields.get("tier_dev", "")
            if not tier:
                continue

            raw_ret = fd.raw_returns[ticker]
            resid_ret = fd.residual_returns.get(ticker, raw_ret)

            tier_raw.setdefault(tier, []).append(raw_ret)
            tier_resid.setdefault(tier, []).append(resid_ret)
            tier_names.setdefault(tier, []).append((ticker, date_str))

    result: Dict[str, Any] = {}
    for tier in ["A", "B", "C", "D"]:
        raw_vals = tier_raw.get(tier, [])
        resid_vals = tier_resid.get(tier, [])
        n = len(raw_vals)
        if n == 0:
            result[tier] = {"n": 0}
            continue

        raw_pcts = [v * 100 for v in raw_vals]
        resid_pcts = [v * 100 for v in resid_vals]

        stats: Dict[str, Any] = {
            "n": n,
            "mean_raw_pct": round(mean(raw_pcts), 2),
            "mean_resid_pct": round(mean(resid_pcts), 2),
            "median_resid_pct": round(median(resid_pcts), 2),
            "winsor_resid_pct": round(mean(_winsorize(resid_pcts)), 2),
            "hit_pct": round(_safe_div(sum(1 for v in raw_vals if v > 0), n) * 100, 1),
            "resid_hit_pct": round(_safe_div(sum(1 for v in resid_vals if v > 0), n) * 100, 1),
        }

        # Left-tail stats (particularly interesting for D-tier)
        stats["p5_resid_pct"] = round(_percentile(resid_pcts, 5), 2)
        stats["p10_resid_pct"] = round(_percentile(resid_pcts, 10), 2)
        stats["max_loss_pct"] = round(min(raw_pcts), 2) if raw_pcts else 0.0

        result[tier] = stats

    return result


# =============================================================================
# PHASE 3: ABLATION SCENARIOS
# =============================================================================

def _make_scenario_ruleset(scenario: str) -> DecisionRuleset:
    """Return the DecisionRuleset for the given ablation scenario."""
    if scenario in ("no_drawdown", "no_gates"):
        return DecisionRuleset(drawdown_gate=-999.0)
    return DecisionRuleset()


def _patch_rec_for_scenario(rec: Dict[str, Any], scenario: str) -> Dict[str, Any]:
    """Return a potentially patched copy of rec for the given scenario.

    For sev3 ablation: clear severity so the SEV3 gate won't fire.
    """
    if scenario in ("no_sev3", "no_gates"):
        patched = dict(rec)
        patched["severity"] = ""
        return patched
    return rec


def _compute_tier_distribution(tier_assignments: Dict[str, str]) -> Dict[str, Any]:
    """Compute tier distribution summary from {ticker: tier} mapping."""
    counts: Counter = Counter(tier_assignments.values())
    n = len(tier_assignments)
    return {
        tier: {
            "n": counts.get(tier, 0),
            "pct": round(_safe_div(counts.get(tier, 0), n) * 100, 1),
        }
        for tier in ["A", "B", "C", "D"]
    }


def find_promoted_names(
    baseline_tiers: Dict[str, str],
    ablated_tiers: Dict[str, str],
    forward_data: ForwardReturnData,
    date_str: str,
) -> List[Dict[str, Any]]:
    """Find tickers promoted from D-tier in baseline to a higher tier in ablated.

    Returns list of dicts with ticker, date, old_tier, new_tier, raw/resid returns.
    """
    promoted = []
    for ticker, base_tier in baseline_tiers.items():
        if base_tier != "D":
            continue
        new_tier = ablated_tiers.get(ticker, "D")
        if new_tier == "D":
            continue

        raw_ret = forward_data.raw_returns.get(ticker)
        resid_ret = forward_data.residual_returns.get(ticker)

        promoted.append({
            "ticker": ticker,
            "date": date_str,
            "old_tier": "D",
            "new_tier": new_tier,
            "raw_return_pct": round(raw_ret * 100, 2) if raw_ret is not None else None,
            "resid_return_pct": round(resid_ret * 100, 2) if resid_ret is not None else None,
        })

    return promoted


def run_ablation_scenario(
    scenario: str,
    archive_cache: Dict[str, ArchiveData],
    return_cache: Dict[str, ForwardReturnData],
    baseline_tiers_by_date: Dict[str, Dict[str, str]],
) -> Dict[str, Any]:
    """Run a single ablation scenario across all archives.

    Returns: tier_distribution, delta_ac stats, D-share, promoted names + stats.
    """
    ruleset = _make_scenario_ruleset(scenario)

    all_a_resids: List[float] = []
    all_c_resids: List[float] = []
    all_promoted: List[Dict[str, Any]] = []
    tier_counts_pooled: Counter = Counter()
    n_dev_total = 0

    per_date: Dict[str, Dict[str, Any]] = {}

    for date_str in sorted(archive_cache.keys()):
        ad = archive_cache[date_str]
        fd = return_cache.get(date_str)
        if fd is None:
            continue

        tier_assignments: Dict[str, str] = {}
        a_resids: List[float] = []
        c_resids: List[float] = []
        n_dev = 0

        for ticker in ad.tickers:
            archetype = ad.archetypes.get(ticker, "")
            if archetype != "drug_developer":
                continue
            n_dev += 1

            rec = ad.recs.get(ticker)
            if rec is None:
                continue

            patched_rec = _patch_rec_for_scenario(rec, scenario)
            opt = ad.optionalities.get(ticker)

            fields = compute_decision_fields(
                patched_rec, archetype, opt, ruleset=ruleset
            )
            tier = fields.get("tier_dev", "")
            if not tier:
                continue

            tier_assignments[ticker] = tier
            tier_counts_pooled[tier] += 1

            if ticker in fd.raw_returns:
                resid_ret = fd.residual_returns.get(
                    ticker, fd.raw_returns[ticker]
                )
                if tier == "A":
                    a_resids.append(resid_ret * 100)
                elif tier == "C":
                    c_resids.append(resid_ret * 100)

        n_dev_total += n_dev
        all_a_resids.extend(a_resids)
        all_c_resids.extend(c_resids)

        # Find promoted names (vs baseline)
        baseline_tiers = baseline_tiers_by_date.get(date_str, {})
        if baseline_tiers and fd is not None:
            promoted = find_promoted_names(
                baseline_tiers, tier_assignments, fd, date_str
            )
            all_promoted.extend(promoted)

        per_date[date_str] = {
            "n_dev": n_dev,
            "tier_distribution": _compute_tier_distribution(tier_assignments),
            "a_resid_mean": round(mean(a_resids), 2) if a_resids else None,
            "c_resid_mean": round(mean(c_resids), 2) if c_resids else None,
            "n_promoted": len(promoted) if baseline_tiers else 0,
        }

    # Pooled stats
    n_total_tier = sum(tier_counts_pooled.values())
    d_share = round(
        _safe_div(tier_counts_pooled.get("D", 0), n_total_tier) * 100, 1
    )

    a_resid_mean = round(mean(_winsorize(all_a_resids)), 2) if all_a_resids else 0.0
    c_resid_mean = round(mean(_winsorize(all_c_resids)), 2) if all_c_resids else 0.0
    delta_ac = round(a_resid_mean - c_resid_mean, 2)

    # Promoted names stats
    promoted_raw = [
        p["raw_return_pct"] for p in all_promoted
        if p["raw_return_pct"] is not None
    ]
    promoted_resid = [
        p["resid_return_pct"] for p in all_promoted
        if p["resid_return_pct"] is not None
    ]
    promoted_new_tiers: Counter = Counter(p["new_tier"] for p in all_promoted)

    promoted_stats: Dict[str, Any] = {
        "n": len(all_promoted),
        "new_tier_distribution": dict(promoted_new_tiers),
    }
    if promoted_resid:
        promoted_stats.update({
            "mean_resid_pct": round(mean(promoted_resid), 2),
            "median_resid_pct": round(median(promoted_resid), 2),
            "winsor_resid_pct": round(mean(_winsorize(promoted_resid)), 2),
            "hit_pct": round(
                _safe_div(sum(1 for v in promoted_raw if v is not None and v > 0),
                          len(promoted_raw)) * 100, 1
            ),
            "p5_resid_pct": round(_percentile(promoted_resid, 5), 2),
            "p10_resid_pct": round(_percentile(promoted_resid, 10), 2),
        })

    return {
        "scenario": scenario,
        "ruleset_id": ruleset.ruleset_id,
        "tier_distribution": {
            tier: {
                "n": tier_counts_pooled.get(tier, 0),
                "pct": round(
                    _safe_div(tier_counts_pooled.get(tier, 0), n_total_tier) * 100, 1
                ),
            }
            for tier in ["A", "B", "C", "D"]
        },
        "d_share_pct": d_share,
        "a_resid_mean_pct": a_resid_mean,
        "c_resid_mean_pct": c_resid_mean,
        "delta_ac_resid_pct": delta_ac,
        "n_promoted": len(all_promoted),
        "promoted_stats": promoted_stats,
        "promoted_names": all_promoted,
        "per_date": per_date,
    }


def build_ablation_scenarios() -> List[str]:
    """Return list of scenario names."""
    return list(ABLATION_SCENARIOS)


# =============================================================================
# PHASE 4: REPORT
# =============================================================================

def generate_ablation_report(
    gate_freq: Dict[str, Any],
    tier_stats: Dict[str, Any],
    scenario_results: Dict[str, Dict[str, Any]],
    output_path: Path,
) -> None:
    """Write human-readable gate ablation report."""
    lines: List[str] = []

    lines.append("=" * 72)
    lines.append("GATE ABLATION + D-TIER AUDIT REPORT")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("=" * 72)
    lines.append("")

    # -------------------------------------------------------------------------
    # Section 1: Gate frequency breakdown
    # -------------------------------------------------------------------------
    lines.append("1. GATE FREQUENCY BREAKDOWN")
    lines.append("-" * 40)

    pooled = gate_freq["pooled"]
    lines.append(f"  Total dev-stage ticker-snapshots: {pooled['n_dev']}")
    lines.append(f"  Total ineligible: {pooled['n_ineligible']} "
                 f"({_safe_div(pooled['n_ineligible'], pooled['n_dev']) * 100:.1f}%)")
    lines.append("")

    # Sort gates by hit count descending
    gate_items = sorted(
        pooled["gates"].items(),
        key=lambda x: x[1]["n"],
        reverse=True,
    )
    lines.append(f"  {'Gate':<25} {'Hits':>6} {'% of dev':>9}")
    lines.append(f"  {'-'*25} {'-'*6} {'-'*9}")
    for gate, stats in gate_items:
        marker = " (INERT)" if gate in gate_freq["inert_gates"] else ""
        lines.append(f"  {gate:<25} {stats['n']:>6} {stats['pct']:>8.1f}%{marker}")

    if pooled["overlap"]:
        lines.append("")
        lines.append("  Overlaps (both gates fire):")
        for pair, stats in sorted(pooled["overlap"].items(),
                                   key=lambda x: x[1]["n"], reverse=True):
            lines.append(f"    {pair:<30} {stats['n']:>4} ({stats['pct']:.1f}%)")

    lines.append("")
    lines.append(f"  Inert gates (never fire): {', '.join(gate_freq['inert_gates'])}")
    lines.append("")

    # -------------------------------------------------------------------------
    # Section 2: D vs ABC return comparison
    # -------------------------------------------------------------------------
    lines.append("2. D-TIER vs ABC RETURN COMPARISON (60d residual)")
    lines.append("-" * 40)

    hdr = (f"  {'Tier':>4} {'N':>6} {'Mean%':>7} {'Med%':>7} {'Win%':>7} "
           f"{'Hit%':>6} {'P5%':>7} {'P10%':>7} {'MaxLoss%':>9}")
    lines.append(hdr)
    lines.append(f"  {'-'*4} {'-'*6} {'-'*7} {'-'*7} {'-'*7} {'-'*6} {'-'*7} {'-'*7} {'-'*9}")

    for tier in ["A", "B", "C", "D"]:
        ts = tier_stats.get(tier, {})
        n = ts.get("n", 0)
        if n == 0:
            lines.append(f"  {tier:>4} {0:>6}    ---     ---     ---    ---     ---     ---       ---")
            continue
        lines.append(
            f"  {tier:>4} {n:>6} "
            f"{ts.get('mean_resid_pct', 0):>7.2f} "
            f"{ts.get('median_resid_pct', 0):>7.2f} "
            f"{ts.get('winsor_resid_pct', 0):>7.2f} "
            f"{ts.get('hit_pct', 0):>6.1f} "
            f"{ts.get('p5_resid_pct', 0):>7.2f} "
            f"{ts.get('p10_resid_pct', 0):>7.2f} "
            f"{ts.get('max_loss_pct', 0):>9.2f}"
        )

    lines.append("")

    # Interpret D-tier danger
    d_stats = tier_stats.get("D", {})
    c_stats = tier_stats.get("C", {})
    if d_stats.get("n", 0) > 0 and c_stats.get("n", 0) > 0:
        d_mean = d_stats.get("winsor_resid_pct", 0)
        c_mean = c_stats.get("winsor_resid_pct", 0)
        if d_mean >= c_mean:
            lines.append("  FINDING: D-tier names are NOT genuinely worse than C-tier on average.")
            lines.append(f"  D winsorized residual ({d_mean:+.2f}%) >= C ({c_mean:+.2f}%).")
            lines.append("  The gates may be over-penalizing viable names.")
        else:
            lines.append(f"  FINDING: D-tier names underperform C-tier on average.")
            lines.append(f"  D winsorized residual ({d_mean:+.2f}%) < C ({c_mean:+.2f}%).")
            lines.append("  Gates appear to be removing genuinely weaker names.")

        d_p5 = d_stats.get("p5_resid_pct", 0)
        c_p5 = c_stats.get("p5_resid_pct", 0)
        if d_p5 < c_p5:
            lines.append(f"  D-tier has heavier left tail: P5={d_p5:.1f}% vs C P5={c_p5:.1f}%.")
        else:
            lines.append(f"  D-tier left tail is comparable or better: P5={d_p5:.1f}% vs C P5={c_p5:.1f}%.")
    lines.append("")

    # -------------------------------------------------------------------------
    # Section 3: Ablation results table
    # -------------------------------------------------------------------------
    lines.append("3. ABLATION SCENARIO RESULTS")
    lines.append("-" * 40)

    hdr2 = (f"  {'Scenario':<15} {'D_share%':>8} {'A_res%':>7} {'C_res%':>7} "
            f"{'dAC%':>6} {'Promoted':>8} {'Prom_hit%':>9}")
    lines.append(hdr2)
    lines.append(f"  {'-'*15} {'-'*8} {'-'*7} {'-'*7} {'-'*6} {'-'*8} {'-'*9}")

    for name in ABLATION_SCENARIOS:
        sr = scenario_results.get(name, {})
        prom_stats = sr.get("promoted_stats", {})
        prom_hit = prom_stats.get("hit_pct", "---")
        prom_hit_str = f"{prom_hit:>9.1f}" if isinstance(prom_hit, (int, float)) else f"{'---':>9}"
        lines.append(
            f"  {name:<15} "
            f"{sr.get('d_share_pct', 0):>8.1f} "
            f"{sr.get('a_resid_mean_pct', 0):>7.2f} "
            f"{sr.get('c_resid_mean_pct', 0):>7.2f} "
            f"{sr.get('delta_ac_resid_pct', 0):>6.2f} "
            f"{sr.get('n_promoted', 0):>8} "
            f"{prom_hit_str}"
        )
    lines.append("")

    # -------------------------------------------------------------------------
    # Section 4: Promoted names analysis
    # -------------------------------------------------------------------------
    lines.append("4. PROMOTED NAMES ANALYSIS (D -> A/B/C)")
    lines.append("-" * 40)

    for name in ABLATION_SCENARIOS:
        if name == "baseline":
            continue
        sr = scenario_results.get(name, {})
        promoted = sr.get("promoted_names", [])
        prom_stats = sr.get("promoted_stats", {})
        if not promoted:
            lines.append(f"  [{name}] No promotions.")
            lines.append("")
            continue

        lines.append(f"  [{name}] {len(promoted)} promoted name-snapshots:")
        new_tiers = prom_stats.get("new_tier_distribution", {})
        tier_str = ", ".join(f"{t}={n}" for t, n in sorted(new_tiers.items()))
        lines.append(f"    New tier distribution: {tier_str}")

        if "mean_resid_pct" in prom_stats:
            lines.append(f"    Mean residual: {prom_stats['mean_resid_pct']:+.2f}%")
            lines.append(f"    Median residual: {prom_stats['median_resid_pct']:+.2f}%")
            lines.append(f"    Winsorized residual: {prom_stats['winsor_resid_pct']:+.2f}%")
            lines.append(f"    Hit rate: {prom_stats.get('hit_pct', 0):.1f}%")
            lines.append(f"    P5 residual: {prom_stats.get('p5_resid_pct', 0):+.2f}%")

        # Show a sample of promoted names (top 10 by resid return)
        with_returns = [p for p in promoted if p["resid_return_pct"] is not None]
        with_returns.sort(key=lambda x: x["resid_return_pct"], reverse=True)
        sample = with_returns[:10]
        if sample:
            lines.append(f"    Top promoted (by residual):")
            for p in sample:
                lines.append(
                    f"      {p['ticker']:<6} {p['date']}  "
                    f"D->{p['new_tier']}  "
                    f"resid={p['resid_return_pct']:+.1f}%"
                )
        lines.append("")

    # -------------------------------------------------------------------------
    # Section 5: Concrete proposal
    # -------------------------------------------------------------------------
    lines.append("5. PROPOSAL")
    lines.append("-" * 40)

    baseline_sr = scenario_results.get("baseline", {})
    no_dd_sr = scenario_results.get("no_drawdown", {})
    no_sev3_sr = scenario_results.get("no_sev3", {})
    no_gates_sr = scenario_results.get("no_gates", {})

    # Determine which gate removal is more impactful
    dd_promoted = no_dd_sr.get("n_promoted", 0)
    sev3_promoted = no_sev3_sr.get("n_promoted", 0)
    dd_prom_stats = no_dd_sr.get("promoted_stats", {})
    sev3_prom_stats = no_sev3_sr.get("promoted_stats", {})

    dd_prom_mean = dd_prom_stats.get("winsor_resid_pct", 0)
    sev3_prom_mean = sev3_prom_stats.get("winsor_resid_pct", 0)

    baseline_delta = baseline_sr.get("delta_ac_resid_pct", 0)
    no_dd_delta = no_dd_sr.get("delta_ac_resid_pct", 0)
    no_sev3_delta = no_sev3_sr.get("delta_ac_resid_pct", 0)
    no_gates_delta = no_gates_sr.get("delta_ac_resid_pct", 0)

    lines.append(f"  Baseline Δ(A-C): {baseline_delta:+.2f}%")
    lines.append(f"  No-drawdown Δ(A-C): {no_dd_delta:+.2f}% "
                 f"(change: {no_dd_delta - baseline_delta:+.2f}%)")
    lines.append(f"  No-SEV3 Δ(A-C): {no_sev3_delta:+.2f}% "
                 f"(change: {no_sev3_delta - baseline_delta:+.2f}%)")
    lines.append(f"  No-gates Δ(A-C): {no_gates_delta:+.2f}% "
                 f"(change: {no_gates_delta - baseline_delta:+.2f}%)")
    lines.append("")

    # Decision logic for proposal
    if dd_promoted > 0 and dd_prom_mean > 0:
        lines.append("  DRAWDOWN GATE:")
        lines.append(f"    Removing promotes {dd_promoted} name-snapshots with "
                     f"positive avg residual ({dd_prom_mean:+.2f}%).")
        lines.append(f"    Consider: soften drawdown_gate from -40% to -50%,")
        lines.append(f"    or convert to a sizing penalty (M->S) instead of exclusion.")
    elif dd_promoted > 0:
        lines.append("  DRAWDOWN GATE:")
        lines.append(f"    Removing promotes {dd_promoted} name-snapshots but avg "
                     f"residual is negative ({dd_prom_mean:+.2f}%). Gate is effective.")
        lines.append(f"    Recommendation: keep as-is.")
    else:
        lines.append("  DRAWDOWN GATE:")
        lines.append(f"    No promotions from removing — gate has minimal practical impact.")

    lines.append("")

    if sev3_promoted > 0 and sev3_prom_mean > 0:
        lines.append("  SEV3 GATE:")
        lines.append(f"    Removing promotes {sev3_promoted} name-snapshots with "
                     f"positive avg residual ({sev3_prom_mean:+.2f}%).")
        lines.append(f"    Consider: convert to risk_flag + sizing penalty instead")
        lines.append(f"    of hard exclusion.")
    elif sev3_promoted > 0:
        lines.append("  SEV3 GATE:")
        lines.append(f"    Removing promotes {sev3_promoted} name-snapshots but avg "
                     f"residual is negative ({sev3_prom_mean:+.2f}%). Gate is effective.")
        lines.append(f"    Recommendation: keep as-is.")
    else:
        lines.append("  SEV3 GATE:")
        lines.append(f"    No promotions from removing — gate has minimal practical impact.")

    lines.append("")
    lines.append("  INERT GATES:")
    lines.append(f"    {', '.join(gate_freq['inert_gates'])} never fire in the archive history.")
    lines.append(f"    These can be safely removed from the engine code or documented")
    lines.append(f"    as placeholders for future data sources.")
    lines.append("")

    # Overall recommendation
    best_scenario = max(
        [("no_drawdown", no_dd_delta), ("no_sev3", no_sev3_delta),
         ("no_gates", no_gates_delta), ("baseline", baseline_delta)],
        key=lambda x: x[1],
    )
    lines.append(f"  OVERALL: Best Δ(A-C) under '{best_scenario[0]}' "
                 f"({best_scenario[1]:+.2f}%).")
    if best_scenario[0] == "baseline":
        lines.append("  Current gates are optimal for A-vs-C separation.")
    else:
        lines.append(f"  Consider adopting '{best_scenario[0]}' scenario if promoted names")
        lines.append(f"  have acceptable risk profiles.")

    lines.append("")
    lines.append("6. CAVEATS")
    lines.append("-" * 40)
    lines.append("  - Small sample sizes per tier (especially A and D).")
    lines.append("  - Gates interact: removing one may shift names between B/C, not just D->C.")
    lines.append("  - Forward returns use HS793+CSV fallback (price return, no dividends).")
    lines.append("  - fundamental_red_flag and adv_fail are inert; this analysis only covers")
    lines.append("    deep_drawdown and sev3.")
    lines.append("  - Results may differ with v1.2 candidate ruleset (a_floor=0.55).")
    lines.append("")
    lines.append("=" * 72)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Gate Ablation + D-tier Audit Report"
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(OUTPUT_DIR),
        help="Output directory (default: output/)",
    )
    parser.add_argument(
        "--start", type=str, default=None,
        help="Start date filter for archives (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end", type=str, default=None,
        help="End date filter for archives (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print archive count + scenario names, don't run",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    default_ruleset = DecisionRuleset()

    # Discover archives
    archives = discover_archives(ARCHIVE_DIR, start=args.start, end=args.end)
    print(f"Archives found: {len(archives)}")
    if archives:
        print(f"  Date range: {archives[0][0]} to {archives[-1][0]}")
    print(f"Ablation scenarios: {', '.join(ABLATION_SCENARIOS)}")
    print()

    if args.dry_run:
        print(f"DRY RUN — would process {len(archives)} archives "
              f"x {len(ABLATION_SCENARIOS)} scenarios")
        return

    if not archives:
        print("ERROR: No archives found.")
        sys.exit(1)

    # Initialize providers
    print("Loading returns providers...", flush=True)
    chained, ms_provider, csv_provider = init_providers()
    last_date = chained.get_last_date()
    print(f"  Returns data ends: {last_date.isoformat() if last_date else 'NONE'}")

    # Determine usable archives
    snapshot_dates = [d for d, _ in archives]
    usable_dates, fence_skipped = compute_as_of_fence(
        snapshot_dates, last_date, 60
    )
    print(f"  Usable snapshots: {len(usable_dates)} / {len(archives)}")
    if fence_skipped:
        for sk in fence_skipped:
            print(f"    SKIP {sk['snapshot_date']}: {sk['skipped_reason']}")
    print()

    if not usable_dates:
        print("ERROR: No usable snapshots (all lack 60d forward return window).")
        sys.exit(1)

    usable_archives = [(d, p) for d, p in archives if d in usable_dates]

    # =================================================================
    # Load archives + compute forward returns (once)
    # =================================================================
    print(f"Loading {len(usable_archives)} archives + computing returns...")
    archive_cache: Dict[str, ArchiveData] = {}
    return_cache: Dict[str, ForwardReturnData] = {}

    for i, (date_str, tar_path) in enumerate(usable_archives):
        print(f"  [{i+1}/{len(usable_archives)}] {date_str} ...", end=" ", flush=True)
        try:
            ad = load_archive_data(tar_path, date_str)
            archive_cache[date_str] = ad

            fd = compute_snapshot_returns(
                chained, csv_provider, ad.tickers, date_str
            )
            return_cache[date_str] = fd
            print(f"OK ({len(ad.tickers)} tickers, "
                  f"{len(fd.raw_returns)} with returns)")
        except Exception as e:
            print(f"ERROR: {e}")
    print()

    # =================================================================
    # Phase 1: Gate frequency
    # =================================================================
    print("Phase 1: Counting gate frequencies...", flush=True)
    gate_freq = count_gate_frequencies(archive_cache, default_ruleset)
    pooled = gate_freq["pooled"]
    print(f"  Dev ticker-snapshots: {pooled['n_dev']}")
    print(f"  Ineligible: {pooled['n_ineligible']} "
          f"({_safe_div(pooled['n_ineligible'], pooled['n_dev']) * 100:.1f}%)")
    for g in KNOWN_GATES:
        gs = pooled["gates"][g]
        inert = " (INERT)" if g in gate_freq["inert_gates"] else ""
        print(f"    {g}: {gs['n']} hits ({gs['pct']}%){inert}")
    print()

    # =================================================================
    # Phase 2: D vs ABC return stats
    # =================================================================
    print("Phase 2: Computing tier return stats...", flush=True)
    tier_stats = compute_tier_return_stats(
        archive_cache, return_cache, default_ruleset
    )
    for tier in ["A", "B", "C", "D"]:
        ts = tier_stats.get(tier, {})
        n = ts.get("n", 0)
        if n > 0:
            print(f"  {tier}: n={n}, winsor_resid={ts.get('winsor_resid_pct', 0):+.2f}%, "
                  f"hit={ts.get('hit_pct', 0):.1f}%, "
                  f"P5={ts.get('p5_resid_pct', 0):+.2f}%")
        else:
            print(f"  {tier}: n=0")
    print()

    # =================================================================
    # Phase 3: Ablation scenarios
    # =================================================================
    print("Phase 3: Running ablation scenarios...", flush=True)

    # First run baseline to get tier assignments
    print("  [baseline] ...", end=" ", flush=True)
    baseline_result = run_ablation_scenario(
        "baseline", archive_cache, return_cache, {}
    )
    # Build baseline tiers by date for promoted-name detection
    baseline_tiers_by_date: Dict[str, Dict[str, str]] = {}
    for date_str in sorted(archive_cache.keys()):
        ad = archive_cache[date_str]
        tiers: Dict[str, str] = {}
        for ticker in ad.tickers:
            archetype = ad.archetypes.get(ticker, "")
            if archetype != "drug_developer":
                continue
            rec = ad.recs.get(ticker)
            if rec is None:
                continue
            opt = ad.optionalities.get(ticker)
            fields = compute_decision_fields(rec, archetype, opt, ruleset=default_ruleset)
            tier = fields.get("tier_dev", "")
            if tier:
                tiers[ticker] = tier
        baseline_tiers_by_date[date_str] = tiers
    print(f"D={baseline_result['d_share_pct']}%, "
          f"Δ(A-C)={baseline_result['delta_ac_resid_pct']:+.2f}%")

    scenario_results: Dict[str, Dict[str, Any]] = {"baseline": baseline_result}

    for scenario in ABLATION_SCENARIOS:
        if scenario == "baseline":
            continue
        print(f"  [{scenario}] ...", end=" ", flush=True)
        sr = run_ablation_scenario(
            scenario, archive_cache, return_cache, baseline_tiers_by_date
        )
        scenario_results[scenario] = sr
        print(f"D={sr['d_share_pct']}%, "
              f"Δ(A-C)={sr['delta_ac_resid_pct']:+.2f}%, "
              f"promoted={sr['n_promoted']}")
    print()

    # =================================================================
    # Phase 4: Write outputs
    # =================================================================
    print("Phase 4: Writing outputs...", flush=True)

    # CSV summary — one row per scenario
    csv_path = output_dir / "gate_ablation_summary.csv"
    csv_columns = [
        "scenario", "ruleset_id", "d_share_pct",
        "a_resid_mean_pct", "c_resid_mean_pct", "delta_ac_resid_pct",
        "n_promoted", "promoted_mean_resid_pct", "promoted_hit_pct",
        "a_n", "b_n", "c_n", "d_n",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns, extrasaction="ignore")
        writer.writeheader()
        for name in ABLATION_SCENARIOS:
            sr = scenario_results[name]
            prom_stats = sr.get("promoted_stats", {})
            row = {
                "scenario": sr["scenario"],
                "ruleset_id": sr["ruleset_id"],
                "d_share_pct": sr["d_share_pct"],
                "a_resid_mean_pct": sr["a_resid_mean_pct"],
                "c_resid_mean_pct": sr["c_resid_mean_pct"],
                "delta_ac_resid_pct": sr["delta_ac_resid_pct"],
                "n_promoted": sr["n_promoted"],
                "promoted_mean_resid_pct": prom_stats.get("winsor_resid_pct", ""),
                "promoted_hit_pct": prom_stats.get("hit_pct", ""),
                "a_n": sr["tier_distribution"]["A"]["n"],
                "b_n": sr["tier_distribution"]["B"]["n"],
                "c_n": sr["tier_distribution"]["C"]["n"],
                "d_n": sr["tier_distribution"]["D"]["n"],
            }
            writer.writerow(row)
    print(f"  Wrote {csv_path}")

    # JSON details — full audit trail
    json_path = output_dir / "gate_ablation_details.json"
    json_output = {
        "metadata": {
            "generated": datetime.now().isoformat(),
            "n_archives": len(usable_archives),
            "date_range": [usable_archives[0][0], usable_archives[-1][0]],
            "default_ruleset_id": default_ruleset.ruleset_id,
        },
        "gate_frequency": gate_freq,
        "tier_return_stats": tier_stats,
        "scenarios": {
            name: {
                k: v for k, v in sr.items()
                if k != "per_date"  # keep JSON size manageable
            }
            for name, sr in scenario_results.items()
        },
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_output, f, indent=2, default=str)
        f.write("\n")
    print(f"  Wrote {json_path}")

    # Text report
    report_path = output_dir / "gate_ablation_report.txt"
    generate_ablation_report(gate_freq, tier_stats, scenario_results, report_path)
    print(f"  Wrote {report_path}")

    # Print summary table
    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"{'Scenario':<15} {'D_share%':>8} {'Δ(A-C)%':>8} {'Promoted':>8} {'Prom_res%':>9}")
    print("-" * 50)
    for name in ABLATION_SCENARIOS:
        sr = scenario_results[name]
        prom_stats = sr.get("promoted_stats", {})
        prom_res = prom_stats.get("winsor_resid_pct")
        prom_str = f"{prom_res:+.2f}" if prom_res is not None else "---"
        print(f"{name:<15} {sr['d_share_pct']:>8.1f} "
              f"{sr['delta_ac_resid_pct']:>8.2f} "
              f"{sr['n_promoted']:>8} "
              f"{prom_str:>9}")
    print()


if __name__ == "__main__":
    main()
