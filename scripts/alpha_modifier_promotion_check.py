#!/usr/bin/env python3
"""Alpha modifier promotion checklist: A/B rerank over historical snapshots.

For each date, loads rankings.csv and reranks with both the **candidate**
ruleset (alpha modifier ON) and the **baseline** (same ruleset but
alpha_modifier_mode forced to "off").  Compares the two orderings and
prints a one-page promotion checklist with gate verdicts.

Gate criteria (all must hold for PROMOTE verdict):
  - top60_overlap >= 0.90  (every date)
  - mean top60_overlap >= 0.93  (aggregate)
  - max rank shift <= 30
  - no tier-A count regression > 2  (modified top60 loses > 2 tier-A vs baseline)

Usage:
    python scripts/alpha_modifier_promotion_check.py \\
        --candidate production_data/decision_rulesets/v1.6.0_alpha_modifier_candidate.json \\
        --date-from 2026-02-17 --date-to 2026-02-25
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, fields as dc_fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from common.ranking_utils import backfill_columns, safe_float as _safe_float
from decision_engine import DecisionRuleset, compute_actionable_sort_key


# ---------------------------------------------------------------------------
# Constants / gate thresholds
# ---------------------------------------------------------------------------
GATE_TOP60_OVERLAP_MIN = 0.90        # per-date floor
GATE_MEAN_TOP60_OVERLAP_MIN = 0.93   # aggregate floor
GATE_MAX_RANK_SHIFT = 30             # per-date cap
GATE_TIER_A_REGRESSION_MAX = 2       # max loss of tier-A in modified top60

MIN_COLS_FOR_SCHEMA = 70  # skip snapshots with old schema


# ---------------------------------------------------------------------------
# Core: rerank and compare
# ---------------------------------------------------------------------------

def _rerank_with_ruleset(
    rows: List[Dict[str, str]],
    ruleset: DecisionRuleset,
) -> Dict[str, int]:
    """Sort rows by *ruleset* and return {ticker: rank} for eligible rows."""
    backfill_columns(rows)
    rows.sort(key=lambda r: compute_actionable_sort_key(
        decision_fields=r,
        archetype=r.get("archetype", ""),
        optionality=_safe_float(r.get("clinical_optionality_pct_dev")),
        composite_rank=r.get("composite_rank"),
        ticker=r.get("ticker", ""),
        catalyst_event_type=r.get("catalyst_event_type", ""),
        catalyst_source=r.get("catalyst_source", ""),
        ruleset=ruleset,
        tiebreaker_pct=(
            _safe_float(r.get("alpha_cohort_pct"))
            if ruleset.sort_anchor == "alpha_cohort"
            else (_safe_float(r.get("commercial_quality_pct"))
                  if r.get("archetype", "").startswith("commercial_")
                  else _safe_float(r.get("clinical_optionality_pct_dev")))
        ),
        alpha_raw=_safe_float(r.get("alpha_cohort_raw")),
    ))
    ranks: Dict[str, int] = {}
    rank = 0
    for r in rows:
        if r.get("eligible") == "1":
            rank += 1
            ranks[r["ticker"]] = rank
    return ranks


def _build_baseline_ruleset(candidate: DecisionRuleset) -> DecisionRuleset:
    """Copy *candidate* with alpha_modifier forced off."""
    kwargs = {f.name: getattr(candidate, f.name) for f in dc_fields(candidate)}
    kwargs["alpha_modifier_mode"] = "off"
    kwargs["alpha_modifier_weight"] = 0.0
    return DecisionRuleset(**kwargs)


@dataclass
class DateABResult:
    """Per-date A/B comparison result."""
    date: str
    n_eligible: int
    n_alpha_nonzero: int
    top60_overlap: float
    mean_abs_shift: float
    max_shift: int
    pct_changed: float
    tier_a_modified: int
    tier_a_baseline: int
    tier_a_delta: int  # modified - baseline (negative = regression)
    tier_b_modified: int
    tier_b_baseline: int
    flagged: bool = False
    flag_reasons: str = ""


def rerank_and_compare(
    rows: List[Dict[str, str]],
    candidate: DecisionRuleset,
    baseline: DecisionRuleset,
    snap_date: str,
) -> DateABResult:
    """Rerank *rows* with both rulesets and compute A/B metrics."""
    mod_rows = deepcopy(rows)
    base_rows = deepcopy(rows)

    mod_ranks = _rerank_with_ruleset(mod_rows, candidate)
    base_ranks = _rerank_with_ruleset(base_rows, baseline)

    common = set(mod_ranks) & set(base_ranks)
    n_eligible = len(common)

    # Alpha coverage
    n_alpha_nonzero = sum(
        1 for r in rows
        if r.get("eligible") == "1"
        and (_safe_float(r.get("alpha_cohort_raw")) or 0.0) != 0.0
    )

    # Rank shifts
    shifts = [abs(mod_ranks[t] - base_ranks[t]) for t in common]
    n_changed = sum(1 for s in shifts if s > 0)
    mean_shift = statistics.mean(shifts) if shifts else 0.0
    max_shift = max(shifts) if shifts else 0

    # Top-60 overlap (Jaccard)
    mod_top60 = {t for t, rk in mod_ranks.items() if rk <= 60}
    base_top60 = {t for t, rk in base_ranks.items() if rk <= 60}
    if mod_top60 or base_top60:
        overlap = len(mod_top60 & base_top60) / len(mod_top60 | base_top60)
    else:
        overlap = 1.0

    # Tier distribution in top-60
    # Build ticker → tier_dev map from original rows
    tier_map = {r["ticker"]: r.get("tier_dev", "") for r in rows if r.get("ticker")}
    mod_tiers = Counter(tier_map.get(t, "") for t in mod_top60)
    base_tiers = Counter(tier_map.get(t, "") for t in base_top60)

    tier_a_mod = mod_tiers.get("A", 0)
    tier_a_base = base_tiers.get("A", 0)
    tier_b_mod = mod_tiers.get("B", 0)
    tier_b_base = base_tiers.get("B", 0)

    # Gate checks
    flags = []
    if overlap < GATE_TOP60_OVERLAP_MIN:
        flags.append(f"overlap={overlap:.3f}<{GATE_TOP60_OVERLAP_MIN}")
    if max_shift > GATE_MAX_RANK_SHIFT:
        flags.append(f"max_shift={max_shift}>{GATE_MAX_RANK_SHIFT}")
    tier_a_delta = tier_a_mod - tier_a_base
    if tier_a_delta < -GATE_TIER_A_REGRESSION_MAX:
        flags.append(f"tier_A_delta={tier_a_delta}")

    return DateABResult(
        date=snap_date,
        n_eligible=n_eligible,
        n_alpha_nonzero=n_alpha_nonzero,
        top60_overlap=round(overlap, 4),
        mean_abs_shift=round(mean_shift, 2),
        max_shift=max_shift,
        pct_changed=round(n_changed / max(n_eligible, 1) * 100, 1),
        tier_a_modified=tier_a_mod,
        tier_a_baseline=tier_a_base,
        tier_a_delta=tier_a_delta,
        tier_b_modified=tier_b_mod,
        tier_b_baseline=tier_b_base,
        flagged=bool(flags),
        flag_reasons="; ".join(flags),
    )


# ---------------------------------------------------------------------------
# Aggregate + gate verdict
# ---------------------------------------------------------------------------

@dataclass
class PromotionVerdict:
    """Aggregate gate verdict across all dates."""
    n_dates: int
    n_flagged: int
    mean_overlap: float
    min_overlap: float
    max_max_shift: int
    mean_pct_changed: float
    promote: bool
    reasons: List[str]


def compute_verdict(results: List[DateABResult]) -> PromotionVerdict:
    """Aggregate per-date results into a promotion verdict."""
    if not results:
        return PromotionVerdict(
            n_dates=0, n_flagged=0, mean_overlap=0.0, min_overlap=0.0,
            max_max_shift=0, mean_pct_changed=0.0, promote=False,
            reasons=["no dates evaluated"],
        )

    overlaps = [r.top60_overlap for r in results]
    mean_overlap = statistics.mean(overlaps)
    min_overlap = min(overlaps)
    max_max_shift = max(r.max_shift for r in results)
    mean_pct = statistics.mean(r.pct_changed for r in results)
    n_flagged = sum(1 for r in results if r.flagged)

    reasons = []
    if min_overlap < GATE_TOP60_OVERLAP_MIN:
        reasons.append(f"min_overlap={min_overlap:.3f} < {GATE_TOP60_OVERLAP_MIN}")
    if mean_overlap < GATE_MEAN_TOP60_OVERLAP_MIN:
        reasons.append(f"mean_overlap={mean_overlap:.3f} < {GATE_MEAN_TOP60_OVERLAP_MIN}")
    if max_max_shift > GATE_MAX_RANK_SHIFT:
        reasons.append(f"max_shift={max_max_shift} > {GATE_MAX_RANK_SHIFT}")
    if n_flagged > 0:
        reasons.append(f"{n_flagged}/{len(results)} dates flagged")

    return PromotionVerdict(
        n_dates=len(results),
        n_flagged=n_flagged,
        mean_overlap=round(mean_overlap, 4),
        min_overlap=round(min_overlap, 4),
        max_max_shift=max_max_shift,
        mean_pct_changed=round(mean_pct, 1),
        promote=len(reasons) == 0,
        reasons=reasons,
    )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def format_checklist(
    results: List[DateABResult],
    verdict: PromotionVerdict,
    candidate: DecisionRuleset,
) -> str:
    """Format a one-page promotion checklist as text/markdown."""
    lines = [
        "# Alpha Modifier Promotion Checklist",
        "",
        f"Candidate ruleset: {candidate.ruleset_id}",
        f"  alpha_modifier_mode = {candidate.alpha_modifier_mode}",
        f"  alpha_modifier_weight = {candidate.alpha_modifier_weight}",
        f"  alpha_train_mode = {candidate.alpha_train_mode}",
        f"  alpha_train_horizon = {candidate.alpha_train_horizon}",
        "",
        "## Per-Date Results",
        "",
        "| Date       | Elig | Alpha | Overlap | Mean Shift | Max Shift | %Changed | A(mod) | A(base) | deltaA | Flag |",
        "|------------|------|-------|---------|------------|-----------|----------|--------|---------|--------|------|",
    ]
    for r in results:
        flag = r.flag_reasons if r.flagged else ""
        lines.append(
            f"| {r.date} | {r.n_eligible:3d} | {r.n_alpha_nonzero:3d} "
            f"| {r.top60_overlap:.3f}  | {r.mean_abs_shift:5.1f}      "
            f"| {r.max_shift:3d}       | {r.pct_changed:5.1f}%   "
            f"| {r.tier_a_modified:2d}     | {r.tier_a_baseline:2d}      "
            f"| {r.tier_a_delta:+3d}    | {flag} |"
        )

    lines += [
        "",
        "## Aggregate",
        "",
        f"| Metric            | Value  | Gate       |",
        f"|-------------------|--------|------------|",
        f"| Dates evaluated   | {verdict.n_dates:6d} |            |",
        f"| Mean overlap      | {verdict.mean_overlap:.4f} | >= {GATE_MEAN_TOP60_OVERLAP_MIN:.2f}   |",
        f"| Min overlap       | {verdict.min_overlap:.4f} | >= {GATE_TOP60_OVERLAP_MIN:.2f}   |",
        f"| Max rank shift    | {verdict.max_max_shift:6d} | <= {GATE_MAX_RANK_SHIFT:3d}      |",
        f"| Mean %changed     | {verdict.mean_pct_changed:5.1f}% |            |",
        f"| Dates flagged     | {verdict.n_flagged:6d} | = 0        |",
        "",
    ]

    if verdict.promote:
        lines.append("## VERDICT: PROMOTE")
        lines.append("")
        lines.append("All gates passed. Safe to promote candidate to production.")
    else:
        lines.append("## VERDICT: DO NOT PROMOTE")
        lines.append("")
        for reason in verdict.reasons:
            lines.append(f"  - {reason}")

    lines.append("")
    return "\n".join(lines)


def write_results_json(
    results: List[DateABResult],
    verdict: PromotionVerdict,
    out_path: Path,
) -> None:
    """Write machine-readable results JSON."""
    from dataclasses import asdict
    data = {
        "per_date": [asdict(r) for r in results],
        "verdict": asdict(verdict),
        "gates": {
            "top60_overlap_min": GATE_TOP60_OVERLAP_MIN,
            "mean_top60_overlap_min": GATE_MEAN_TOP60_OVERLAP_MIN,
            "max_rank_shift": GATE_MAX_RANK_SHIFT,
            "tier_a_regression_max": GATE_TIER_A_REGRESSION_MAX,
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
        f.write("\n")


# ---------------------------------------------------------------------------
# Discovery + main loop
# ---------------------------------------------------------------------------

def discover_eligible_dates(
    snapshot_root: Path,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    min_cols: int = MIN_COLS_FOR_SCHEMA,
) -> List[str]:
    """Find snapshot dates with new-schema rankings that have alpha_cohort_raw."""
    dates = []
    if not snapshot_root.exists():
        return dates
    for d in sorted(snapshot_root.iterdir()):
        if not d.is_dir() or len(d.name) != 10 or d.name[4] != "-":
            continue
        csv_path = d / "rankings.csv"
        if not csv_path.exists():
            continue
        name = d.name
        if date_from and name < date_from:
            continue
        if date_to and name > date_to:
            continue
        # Quick schema check: read header
        with open(csv_path, newline="", encoding="utf-8") as f:
            header = f.readline().strip()
        cols = header.split(",")
        if len(cols) < min_cols:
            continue
        if "alpha_cohort_raw" not in cols:
            continue
        dates.append(name)
    return dates


def run_promotion_check(
    snapshot_root: Path,
    candidate: DecisionRuleset,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    min_cols: int = MIN_COLS_FOR_SCHEMA,
) -> Tuple[List[DateABResult], PromotionVerdict]:
    """Run the full promotion check across eligible dates."""
    baseline = _build_baseline_ruleset(candidate)
    dates = discover_eligible_dates(snapshot_root, date_from, date_to, min_cols=min_cols)

    results: List[DateABResult] = []
    for snap_date in dates:
        csv_path = snapshot_root / snap_date / "rankings.csv"
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        result = rerank_and_compare(rows, candidate, baseline, snap_date)
        results.append(result)

    verdict = compute_verdict(results)
    return results, verdict


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Alpha modifier A/B promotion checklist",
    )
    parser.add_argument(
        "--candidate", type=Path, required=True,
        help="Path to candidate ruleset JSON",
    )
    parser.add_argument(
        "--snapshot-root", type=Path,
        default=PROJECT_ROOT / "data" / "snapshots",
    )
    parser.add_argument("--date-from", type=str, default=None)
    parser.add_argument("--date-to", type=str, default=None)
    parser.add_argument(
        "--out-dir", type=Path,
        default=PROJECT_ROOT / "output" / "alpha_modifier_promotion",
    )
    args = parser.parse_args()

    candidate = DecisionRuleset.from_json(str(args.candidate))
    print(f"Candidate: {candidate.ruleset_id}  "
          f"mode={candidate.alpha_modifier_mode}  "
          f"weight={candidate.alpha_modifier_weight}")

    results, verdict = run_promotion_check(
        snapshot_root=args.snapshot_root,
        candidate=candidate,
        date_from=args.date_from,
        date_to=args.date_to,
    )

    if not results:
        print("No eligible snapshots found (need alpha_cohort_raw column)")
        return 1

    checklist = format_checklist(results, verdict, candidate)
    print(checklist)

    # Write outputs
    args.out_dir.mkdir(parents=True, exist_ok=True)
    md_path = args.out_dir / "promotion_checklist.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(checklist)
    json_path = args.out_dir / "promotion_results.json"
    write_results_json(results, verdict, json_path)

    print(f"\nWrote: {md_path}")
    print(f"Wrote: {json_path}")

    return 0 if verdict.promote else 2


if __name__ == "__main__":
    sys.exit(main())
