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
    _compute_tier_commercial,
    compute_actionable_sort_key,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TOP_K = 20
TOP_60 = 60

import numpy as np

_CQ_WEIGHTS = {"financial": 0.45, "valuation": 0.35, "momentum": 0.20}


def _compute_commercial_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Add commercial_quality_pct column (percentile within commercial cohort).

    Mirrors the computation in run_screen.py.  Operates in-place on a copy.
    """
    df = df.copy()
    df["commercial_quality_pct"] = None

    comm_mask = df["archetype"].str.startswith("commercial_", na=False)
    comm = df.loc[
        comm_mask
        & df["financial_score"].notna()
        & df["valuation_score"].notna()
    ].copy()

    if comm.empty:
        return df

    comm["_cq"] = (
        _CQ_WEIGHTS["financial"] * comm["financial_score"].astype(float)
        + _CQ_WEIGHTS["valuation"] * comm["valuation_score"].astype(float)
        + _CQ_WEIGHTS["momentum"] * comm["momentum_score"].fillna(0).astype(float)
    )
    comm = comm.sort_values(["_cq", "ticker"])
    n = len(comm)
    for rank_i, idx in enumerate(comm.index, start=1):
        p = max(0.001, min(0.999, (rank_i - 0.5) / n))
        df.at[idx, "commercial_quality_pct"] = round(p, 6)

    return df


def _recompute_tier_any(row: pd.Series, ruleset: DecisionRuleset) -> str:
    """Recompute tier_commercial and return tier_any for a row."""
    archetype = str(row.get("archetype", ""))
    eligible = str(row.get("eligible", "0")) in ("1", "1.0", "True")
    cq_pct = _safe_float(row.get("commercial_quality_pct"))
    cat_in_window = str(row.get("catalyst_in_window", "")) if pd.notna(row.get("catalyst_in_window")) else ""
    cat_days = row.get("catalyst_days", "")
    if pd.isna(cat_days):
        cat_days = ""

    tier_commercial, _ = _compute_tier_commercial(
        archetype=archetype,
        eligible=eligible,
        quality_pct=cq_pct,
        catalyst_in_window=cat_in_window,
        catalyst_days=cat_days,
        ruleset=ruleset,
    )
    tier_dev = str(row.get("tier_dev", "")) if pd.notna(row.get("tier_dev")) else ""
    return tier_dev or tier_commercial


def _sort_key_for_row(row: pd.Series, ruleset: DecisionRuleset) -> tuple:
    """Build sort key from a rankings.csv row."""
    fields = {k: ("" if pd.isna(v) else v) for k, v in row.to_dict().items()}
    archetype = str(row.get("archetype", ""))
    if archetype.startswith("commercial_"):
        tb_pct = _safe_float(row.get("commercial_quality_pct"))
    else:
        tb_pct = _safe_float(row.get("clinical_optionality_pct_dev"))
    return compute_actionable_sort_key(
        decision_fields=fields,
        archetype=archetype,
        optionality=_safe_float(row.get("clinical_optionality_pct_dev")),
        composite_rank=_safe_int(row.get("composite_rank")),
        ticker=str(row.get("ticker", "")),
        catalyst_event_type=str(row.get("catalyst_event_type", "")),
        catalyst_source=str(row.get("catalyst_source", "")),
        ruleset=ruleset,
        tiebreaker_pct=tb_pct,
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
    """Re-sort rankings and return the top-K portfolio.

    In dev_first mode: filter to drug_developer + tier_dev in tier_filter.
    In tier_first mode: recompute tier_any, filter all archetypes by tier_any.
    """
    tier_filter = set(ruleset.tier_filter) if hasattr(ruleset, "tier_filter") else {"A", "B"}
    eligible_mask = df.get("eligible", pd.Series(dtype=object)).astype(str).isin(["1", "1.0", "True"])

    if ruleset.tiering_priority_mode == "tier_first":
        # Recompute commercial quality + tier_any for all eligible rows
        pool = df[eligible_mask].copy()
        pool = _compute_commercial_quality(pool)
        pool["tier_any"] = pool.apply(lambda r: _recompute_tier_any(r, ruleset), axis=1)
        pool = pool[pool["tier_any"].isin(tier_filter)]
    else:
        # dev_first: only drug developers, filter on tier_dev
        dev_mask = df.get("archetype", pd.Series(dtype=str)) == "drug_developer"
        pool = df[eligible_mask & dev_mask].copy()
        pool = pool[pool["tier_dev"].isin(tier_filter)]

    if pool.empty:
        return pool

    # Compute sort keys
    pool["_sort_key"] = pool.apply(
        lambda r: _sort_key_for_row(r, ruleset), axis=1
    )
    pool = pool.sort_values("_sort_key").reset_index(drop=True)
    pool["_rank"] = range(1, len(pool) + 1)
    return pool


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

    # Tier distribution comparison (top-20) — use tier_any if available, else tier_dev
    def _tier_col(df):
        if "tier_any" in df.columns and df["tier_any"].notna().any():
            return "tier_any"
        return "tier_dev"

    base_tiers = base_sorted.head(TOP_K)[_tier_col(base_sorted)].value_counts().to_dict()
    cand_tiers = cand_sorted.head(TOP_K)[_tier_col(cand_sorted)].value_counts().to_dict()

    # Archetype distribution in candidate portfolio (for tier_first analysis)
    cand_arch_20 = cand_sorted.head(TOP_K)["archetype"].value_counts().to_dict() if not cand_sorted.empty else {}
    cand_arch_60 = cand_sorted.head(TOP_60)["archetype"].value_counts().to_dict() if not cand_sorted.empty else {}
    base_arch_20 = base_sorted.head(TOP_K)["archetype"].value_counts().to_dict() if not base_sorted.empty else {}

    # Commercial names entering candidate portfolio (tier_first specific)
    cand_commercial_20 = sorted(
        cand_sorted.head(TOP_K).loc[
            cand_sorted.head(TOP_K)["archetype"] != "drug_developer", "ticker"
        ].tolist()
    ) if not cand_sorted.empty else []

    return {
        "baseline_id": baseline_rs.ruleset_id,
        "candidate_id": candidate_rs.ruleset_id,
        "baseline_mode": baseline_rs.tiering_priority_mode,
        "candidate_mode": candidate_rs.tiering_priority_mode,
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
        "base_eligible_count": len(base_sorted),
        "cand_eligible_count": len(cand_sorted),
        "base_arch_dist_20": base_arch_20,
        "cand_arch_dist_20": cand_arch_20,
        "cand_arch_dist_60": cand_arch_60,
        "cand_commercial_20": cand_commercial_20,
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def format_report(result: dict, date: str) -> str:
    """Format comparison as markdown."""
    base_mode = result.get("baseline_mode", "dev_first")
    cand_mode = result.get("candidate_mode", "dev_first")
    lines = [
        f"# Ruleset Comparison — {date}",
        "",
        f"Baseline: `{result['baseline_id']}` ({base_mode}) vs Candidate: `{result['candidate_id']}` ({cand_mode})",
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

    # Archetype distribution (shows commercial names entering in tier_first)
    cand_arch = result.get("cand_arch_dist_20", {})
    base_arch = result.get("base_arch_dist_20", {})
    if cand_arch and (len(cand_arch) > 1 or len(base_arch) > 1):
        lines += [
            "",
            "## Archetype Distribution (Top-20)",
            "| Archetype | Baseline | Candidate |",
            "|-----------|----------|-----------|",
        ]
        all_archs = sorted(set(list(base_arch) + list(cand_arch)))
        for a in all_archs:
            lines.append(f"| {a} | {base_arch.get(a, 0)} | {cand_arch.get(a, 0)} |")

    comm_names = result.get("cand_commercial_20", [])
    if comm_names:
        lines += [
            "",
            f"Commercial names in candidate top-20: {', '.join(comm_names)}",
        ]

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

    # Backfill clinical z-scores if not in snapshot (pre-feature snapshots)
    # Each archetype cohort gets its own z-score (never mixed).
    _CLINICAL_COHORTS = {
        "drug_developer": "dev",
        "commercial_biotech": "comm_biotech",
        "commercial_pharma": "comm_pharma",
    }
    dev_mask = rankings["archetype"] == "drug_developer"
    if "clinical_score_z" not in rankings.columns and "clinical_score" in rankings.columns:
        for arch_name, cohort_label in _CLINICAL_COHORTS.items():
            mask = rankings["archetype"] == arch_name
            scores = pd.to_numeric(rankings.loc[mask, "clinical_score"], errors="coerce")
            valid = scores.dropna()
            if len(valid) > 1:
                mean, std = valid.mean(), valid.std(ddof=0)
                if std > 0:
                    rankings.loc[valid.index, "clinical_score_z"] = ((valid - mean) / std).round(4)
            print(f"[INFO] Backfilled clinical_score_z for {len(valid)} {arch_name} (ddof=0)")

    if "clinical_score_z_tier" not in rankings.columns and "clinical_score" in rankings.columns:
        # Drug developers: group by tier_dev
        n_filled = 0
        for tier, grp in rankings.loc[dev_mask].groupby("tier_dev"):
            if not tier:
                continue
            scores = pd.to_numeric(grp["clinical_score"], errors="coerce")
            valid = scores.dropna()
            if len(valid) < 2:
                rankings.loc[valid.index, "clinical_score_z_tier"] = 0.0
                n_filled += len(valid)
                continue
            mean, std = valid.mean(), valid.std(ddof=0)
            if std > 0:
                rankings.loc[valid.index, "clinical_score_z_tier"] = ((valid - mean) / std).round(4)
            else:
                rankings.loc[valid.index, "clinical_score_z_tier"] = 0.0
            n_filled += len(valid)
        print(f"[INFO] Backfilled clinical_score_z_tier for {n_filled} drug_developers (ddof=0)")

        # Commercial cohorts: group by tier_commercial (if column exists)
        _COMM_ARCHETYPES = ("commercial_biotech", "commercial_pharma")
        has_tier_comm = "tier_commercial" in rankings.columns
        n_comm_filled = 0
        for arch_name in _COMM_ARCHETYPES:
            comm_mask = rankings["archetype"] == arch_name
            if not has_tier_comm:
                # Older snapshots: no tier_commercial → set 0.0
                comm_cs = pd.to_numeric(rankings.loc[comm_mask, "clinical_score"], errors="coerce")
                valid = comm_cs.dropna()
                rankings.loc[valid.index, "clinical_score_z_tier"] = 0.0
                n_comm_filled += len(valid)
                continue
            for tier, grp in rankings.loc[comm_mask].groupby("tier_commercial"):
                if not tier:
                    scores = pd.to_numeric(grp["clinical_score"], errors="coerce")
                    rankings.loc[scores.dropna().index, "clinical_score_z_tier"] = 0.0
                    n_comm_filled += len(scores.dropna())
                    continue
                scores = pd.to_numeric(grp["clinical_score"], errors="coerce")
                valid = scores.dropna()
                if len(valid) < 2:
                    rankings.loc[valid.index, "clinical_score_z_tier"] = 0.0
                    n_comm_filled += len(valid)
                    continue
                mean, std = valid.mean(), valid.std(ddof=0)
                if std > 0:
                    rankings.loc[valid.index, "clinical_score_z_tier"] = ((valid - mean) / std).round(4)
                else:
                    rankings.loc[valid.index, "clinical_score_z_tier"] = 0.0
                n_comm_filled += len(valid)
        if n_comm_filled:
            print(f"[INFO] Backfilled clinical_score_z_tier for {n_comm_filled} commercial tickers (ddof=0)")

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
