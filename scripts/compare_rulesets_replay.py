#!/usr/bin/env python3
"""Compare two DecisionRulesets by re-sorting an existing snapshot.

Loads rankings.csv from a snapshot, re-applies compute_actionable_sort_key
with baseline vs candidate rulesets, and writes a comparison report.

Usage:
    python3 scripts/compare_rulesets_replay.py \
        --as-of-date 2026-02-16 \
        --baseline-ruleset production_data/decision_rulesets/v1.3.2_candidate.json \
        --candidate-ruleset production_data/decision_rulesets/v1.3.3_missing_sort_only_candidate.json \
        --snapshot-dir data/snapshots/2026-02-16
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from decision_engine import (
    DecisionRuleset,
    compute_actionable_sort_key,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TOP_K = 20
TOP_60 = 60


def _sort_key_for_row(row: pd.Series, ruleset: DecisionRuleset) -> tuple:
    """Build sort key from a rankings.csv row."""
    fields = {k: ("" if pd.isna(v) else v) for k, v in row.to_dict().items()}
    return compute_actionable_sort_key(
        decision_fields=fields,
        archetype=str(row.get("archetype", "")),
        optionality=_safe_float(row.get("clinical_optionality_pct_dev")),
        composite_rank=_safe_int(row.get("composite_rank")),
        ticker=str(row.get("ticker", "")),
        catalyst_event_type=str(row.get("catalyst_event_type", "")),
        catalyst_source=str(row.get("catalyst_source", "")),
        ruleset=ruleset,
    )


def _safe_float(v) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _safe_int(v) -> Optional[int]:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def rank_portfolio(df: pd.DataFrame, ruleset: DecisionRuleset, top_k: int = TOP_K) -> pd.DataFrame:
    """Re-sort rankings and return the top-K portfolio."""
    # Filter to eligible dev-stage only (mirrors decision engine)
    mask = (
        (df.get("archetype", pd.Series(dtype=str)) == "drug_developer")
        & (df.get("eligible", pd.Series(dtype=object)).astype(str).isin(["1", "1.0", "True"]))
    )
    dev_eligible = df[mask].copy()
    if dev_eligible.empty:
        return dev_eligible

    # Filter to tier_filter (A, B by default)
    tier_filter = set(ruleset.tier_filter) if hasattr(ruleset, "tier_filter") else {"A", "B"}
    dev_eligible = dev_eligible[dev_eligible["tier_dev"].isin(tier_filter)]

    # Compute sort keys
    dev_eligible["_sort_key"] = dev_eligible.apply(
        lambda r: _sort_key_for_row(r, ruleset), axis=1
    )
    dev_eligible = dev_eligible.sort_values("_sort_key").reset_index(drop=True)
    dev_eligible["_rank"] = range(1, len(dev_eligible) + 1)
    return dev_eligible


def overlap_pct(set_a: set, set_b: set) -> float:
    """Jaccard overlap as percentage."""
    if not set_a and not set_b:
        return 100.0
    union = set_a | set_b
    if not union:
        return 100.0
    return round(100 * len(set_a & set_b) / len(union), 1)


def compare(
    rankings: pd.DataFrame,
    baseline_rs: DecisionRuleset,
    candidate_rs: DecisionRuleset,
) -> dict:
    """Run comparison and return structured results."""
    base_sorted = rank_portfolio(rankings, baseline_rs)
    cand_sorted = rank_portfolio(rankings, candidate_rs)

    base_top20 = set(base_sorted.head(TOP_K)["ticker"])
    cand_top20 = set(cand_sorted.head(TOP_K)["ticker"])
    base_top60 = set(base_sorted.head(TOP_60)["ticker"])
    cand_top60 = set(cand_sorted.head(TOP_60)["ticker"])

    entrants_20 = sorted(cand_top20 - base_top20)
    exits_20 = sorted(base_top20 - cand_top20)
    entrants_60 = sorted(cand_top60 - base_top60)
    exits_60 = sorted(base_top60 - cand_top60)

    # Missingness in portfolio
    def _missing_count(df, n):
        top = df.head(n)
        mc = top.get("missing_components", pd.Series(dtype=str))
        return int(mc.apply(lambda x: pd.notna(x) and bool(str(x).strip())).sum())

    # Rank churn: mean absolute rank change for common tickers
    common = base_top60 & cand_top60
    rank_changes = []
    if common:
        base_ranks = dict(zip(base_sorted["ticker"], base_sorted["_rank"]))
        cand_ranks = dict(zip(cand_sorted["ticker"], cand_sorted["_rank"]))
        for t in common:
            if t in base_ranks and t in cand_ranks:
                rank_changes.append(abs(base_ranks[t] - cand_ranks[t]))

    # Tier distribution comparison (top-20)
    base_tiers = base_sorted.head(TOP_K)["tier_dev"].value_counts().to_dict()
    cand_tiers = cand_sorted.head(TOP_K)["tier_dev"].value_counts().to_dict()

    return {
        "baseline_id": baseline_rs.ruleset_id,
        "candidate_id": candidate_rs.ruleset_id,
        "top20_overlap": overlap_pct(base_top20, cand_top20),
        "top60_overlap": overlap_pct(base_top60, cand_top60),
        "entrants_20": entrants_20,
        "exits_20": exits_20,
        "entrants_60": entrants_60,
        "exits_60": exits_60,
        "base_top20_missing": _missing_count(base_sorted, TOP_K),
        "cand_top20_missing": _missing_count(cand_sorted, TOP_K),
        "base_top60_missing": _missing_count(base_sorted, TOP_60),
        "cand_top60_missing": _missing_count(cand_sorted, TOP_60),
        "mean_rank_churn_top60": round(sum(rank_changes) / len(rank_changes), 2) if rank_changes else 0.0,
        "max_rank_churn_top60": max(rank_changes) if rank_changes else 0,
        "base_tier_dist_20": base_tiers,
        "cand_tier_dist_20": cand_tiers,
        "base_eligible_count": len(rank_portfolio(rankings, baseline_rs)),
        "cand_eligible_count": len(rank_portfolio(rankings, candidate_rs)),
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def format_report(result: dict, date: str) -> str:
    """Format comparison as markdown."""
    lines = [
        f"# Ruleset Comparison — {date}",
        "",
        f"Baseline: `{result['baseline_id']}` vs Candidate: `{result['candidate_id']}`",
        "",
        "## Overlap",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Top-20 overlap | {result['top20_overlap']}% |",
        f"| Top-60 overlap | {result['top60_overlap']}% |",
        f"| Mean rank churn (top-60 common) | {result['mean_rank_churn_top60']} |",
        f"| Max rank churn (top-60 common) | {result['max_rank_churn_top60']} |",
        "",
        "## Top-20 Changes",
        f"Entrants: {', '.join(result['entrants_20']) or 'none'}  ",
        f"Exits: {', '.join(result['exits_20']) or 'none'}",
        "",
        "## Top-60 Changes",
        f"Entrants: {', '.join(result['entrants_60']) or 'none'}  ",
        f"Exits: {', '.join(result['exits_60']) or 'none'}",
        "",
        "## Missingness in Portfolio",
        f"| Scope | Baseline | Candidate |",
        f"|-------|----------|-----------|",
        f"| Top-20 | {result['base_top20_missing']} | {result['cand_top20_missing']} |",
        f"| Top-60 | {result['base_top60_missing']} | {result['cand_top60_missing']} |",
        "",
        "## Tier Distribution (Top-20)",
        f"| Tier | Baseline | Candidate |",
        f"|------|----------|-----------|",
    ]
    all_tiers = sorted(set(list(result["base_tier_dist_20"]) + list(result["cand_tier_dist_20"])))
    for t in all_tiers:
        lines.append(f"| {t} | {result['base_tier_dist_20'].get(t, 0)} | {result['cand_tier_dist_20'].get(t, 0)} |")

    lines.append("")
    lines.append(f"Eligible pool: baseline={result['base_eligible_count']}, candidate={result['cand_eligible_count']}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Compare two rulesets on an existing snapshot")
    p.add_argument("--as-of-date", required=True, help="Snapshot date (YYYY-MM-DD)")
    p.add_argument("--baseline-ruleset", required=True, help="Path to baseline ruleset JSON")
    p.add_argument("--candidate-ruleset", required=True, help="Path to candidate ruleset JSON")
    p.add_argument("--snapshot-dir", required=True, help="Snapshot directory containing rankings.csv")
    p.add_argument("--output", default=None, help="Output path for report (default: <snapshot-dir>/ruleset_compare.md)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    snap_dir = Path(args.snapshot_dir)
    rankings_path = snap_dir / "rankings.csv"
    if not rankings_path.exists():
        print(f"ERROR: {rankings_path} not found", file=sys.stderr)
        return 1

    rankings = pd.read_csv(rankings_path)
    baseline_rs = DecisionRuleset.from_json(args.baseline_ruleset)
    candidate_rs = DecisionRuleset.from_json(args.candidate_ruleset)

    result = compare(rankings, baseline_rs, candidate_rs)
    report = format_report(result, args.as_of_date)

    out_path = Path(args.output) if args.output else snap_dir / "ruleset_compare.md"
    out_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nWritten to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
