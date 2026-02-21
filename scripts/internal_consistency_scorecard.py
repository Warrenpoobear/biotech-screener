#!/usr/bin/env python3
"""Internal consistency scorecard for snapshot quality monitoring.

Checks: missingness rates, NaN hotspots, tie density, rank/tier invariants,
duplicate tickers, eligibility consistency.

Outputs: scorecard_{date}.json + scorecard_{date}.md
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SCHEMA_VERSION = "internal_consistency_scorecard.v1"

# Columns that must be present and non-empty for every eligible ticker
REQUIRED_COLUMNS = [
    "ticker", "actionable_rank", "tier_dev", "eligible",
    "composite_rank", "composite_score", "archetype",
]

# Numeric columns to check for NaN hotspots
NUMERIC_COLUMNS = [
    "actionable_rank", "composite_score", "score_rank_pct",
    "clinical_optionality_pct_dev", "alpha_cohort_pct",
    "de_drawdown", "de_rsi_14d", "de_beta_xbi_60d", "de_alpha_60d",
]

# Rank columns that should have zero or near-zero ties
RANK_COLUMNS = ["actionable_rank", "composite_rank"]

# Maximum acceptable missingness fraction before WARN
DEFAULT_WARN_MISSINGNESS = 0.05
DEFAULT_WARN_TIE_PCT = 0.02


# ---------------------------------------------------------------------------
# Check functions
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    name: str
    status: str  # "PASS" | "WARN"
    detail: str = ""
    value: Any = None
    threshold: Any = None


def check_duplicate_tickers(rows: List[Dict[str, str]]) -> CheckResult:
    """Detect duplicate ticker entries."""
    tickers = [r.get("ticker", "").strip() for r in rows]
    counts = Counter(tickers)
    dupes = {t: c for t, c in counts.items() if c > 1 and t}
    if dupes:
        sample = ", ".join(f"{t}({c})" for t, c in sorted(dupes.items())[:5])
        return CheckResult(
            name="duplicate_tickers", status="WARN",
            detail=f"{len(dupes)} duplicate tickers: {sample}",
            value=len(dupes),
        )
    return CheckResult(
        name="duplicate_tickers", status="PASS",
        detail=f"0 duplicates in {len(tickers)} rows",
    )


def check_missingness(
    rows: List[Dict[str, str]],
    warn_threshold: float = DEFAULT_WARN_MISSINGNESS,
) -> Tuple[CheckResult, Dict[str, float]]:
    """Check missingness rate across all columns."""
    if not rows:
        return CheckResult(
            name="missingness", status="WARN",
            detail="No rows to check",
        ), {}

    n = len(rows)
    all_cols = list(rows[0].keys()) if rows else []
    miss_rates: Dict[str, float] = {}

    for col in all_cols:
        missing = sum(1 for r in rows if not (r.get(col) or "").strip()
                      or (r.get(col) or "").strip().lower() == "nan")
        miss_rates[col] = missing / n

    high_miss = {col: rate for col, rate in miss_rates.items()
                 if rate > warn_threshold}

    if high_miss:
        sample = ", ".join(f"{c}={r:.1%}" for c, r in
                           sorted(high_miss.items(), key=lambda x: -x[1])[:5])
        return CheckResult(
            name="missingness", status="WARN",
            detail=f"{len(high_miss)} columns above {warn_threshold:.0%}: {sample}",
            value=len(high_miss), threshold=warn_threshold,
        ), miss_rates

    return CheckResult(
        name="missingness", status="PASS",
        detail=f"All {len(all_cols)} columns below {warn_threshold:.0%} missingness",
        threshold=warn_threshold,
    ), miss_rates


def check_nan_hotspots(rows: List[Dict[str, str]]) -> CheckResult:
    """Check numeric columns for NaN values specifically."""
    if not rows:
        return CheckResult(name="nan_hotspots", status="PASS", detail="No rows")

    n = len(rows)
    hotspots: Dict[str, int] = {}
    for col in NUMERIC_COLUMNS:
        if col not in rows[0]:
            continue
        nan_count = sum(1 for r in rows
                        if (r.get(col) or "").strip().lower() in ("nan", "none", ""))
        if nan_count > 0:
            hotspots[col] = nan_count

    if hotspots:
        sample = ", ".join(f"{c}={cnt}/{n}" for c, cnt in
                           sorted(hotspots.items(), key=lambda x: -x[1])[:5])
        return CheckResult(
            name="nan_hotspots", status="WARN" if any(v > n * 0.10 for v in hotspots.values()) else "PASS",
            detail=f"NaN found in {len(hotspots)} columns: {sample}",
            value=hotspots,
        )

    return CheckResult(
        name="nan_hotspots", status="PASS",
        detail=f"No NaN in {len(NUMERIC_COLUMNS)} numeric columns",
    )


def check_tie_density(
    rows: List[Dict[str, str]],
    warn_threshold: float = DEFAULT_WARN_TIE_PCT,
) -> CheckResult:
    """Check rank columns for excessive ties."""
    if not rows:
        return CheckResult(name="tie_density", status="PASS", detail="No rows")

    n = len(rows)
    problems: List[str] = []

    for col in RANK_COLUMNS:
        if col not in rows[0]:
            continue
        values = []
        for r in rows:
            v = (r.get(col) or "").strip()
            if v and v.lower() not in ("nan", "none"):
                values.append(v)
        if not values:
            continue
        unique = len(set(values))
        tie_pct = 1.0 - unique / len(values) if values else 0.0
        if tie_pct > warn_threshold:
            problems.append(f"{col}: {tie_pct:.1%} ties ({unique}/{len(values)} unique)")

    if problems:
        return CheckResult(
            name="tie_density", status="WARN",
            detail="; ".join(problems),
            threshold=warn_threshold,
        )

    return CheckResult(
        name="tie_density", status="PASS",
        detail=f"Tie density OK across {len(RANK_COLUMNS)} rank columns",
        threshold=warn_threshold,
    )


def check_rank_invariants(rows: List[Dict[str, str]]) -> CheckResult:
    """Check rank ordering invariants:
    - actionable_rank should be 1..N for eligible tickers
    - No gaps or out-of-range values
    """
    eligible_rows = [r for r in rows if r.get("eligible", "0") == "1"]
    if not eligible_rows:
        return CheckResult(
            name="rank_invariants", status="PASS",
            detail="No eligible tickers to check",
        )

    ranks = []
    for r in eligible_rows:
        try:
            ranks.append(int(r["actionable_rank"]))
        except (ValueError, KeyError):
            pass

    if not ranks:
        return CheckResult(
            name="rank_invariants", status="WARN",
            detail="Could not parse actionable_rank for eligible tickers",
        )

    n = len(ranks)
    sorted_ranks = sorted(ranks)
    expected = list(range(1, n + 1))

    if sorted_ranks != expected:
        # Find specific issues
        missing = set(expected) - set(sorted_ranks)
        extra = set(sorted_ranks) - set(expected)
        issues = []
        if missing:
            issues.append(f"missing ranks: {sorted(missing)[:5]}")
        if extra:
            issues.append(f"extra ranks: {sorted(extra)[:5]}")
        dupes = [r for r in sorted_ranks if sorted_ranks.count(r) > 1]
        if dupes:
            issues.append(f"duplicate ranks: {sorted(set(dupes))[:5]}")
        return CheckResult(
            name="rank_invariants", status="WARN",
            detail=f"Rank sequence broken ({n} eligible): {'; '.join(issues)}",
        )

    return CheckResult(
        name="rank_invariants", status="PASS",
        detail=f"Ranks 1..{n} contiguous for {n} eligible tickers",
    )


def check_eligibility_tier_consistency(rows: List[Dict[str, str]]) -> CheckResult:
    """Eligible tickers with tier A or B should have valid tier fields."""
    problems: List[str] = []

    for r in rows:
        ticker = r.get("ticker", "?")
        eligible = r.get("eligible", "0")
        tier_dev = (r.get("tier_dev") or "").strip()
        tier_any = (r.get("tier_any") or "").strip()

        # Ineligible but has a tier
        if eligible != "1" and tier_dev in ("A", "B"):
            problems.append(f"{ticker}: ineligible but tier_dev={tier_dev}")

        # Eligible with tier A/B should have tier_any
        if eligible == "1" and tier_dev in ("A", "B") and not tier_any:
            problems.append(f"{ticker}: tier_dev={tier_dev} but tier_any empty")

    if problems:
        sample = "; ".join(problems[:5])
        return CheckResult(
            name="eligibility_tier_consistency", status="WARN",
            detail=f"{len(problems)} issues: {sample}",
            value=len(problems),
        )

    return CheckResult(
        name="eligibility_tier_consistency", status="PASS",
        detail="Eligibility/tier consistent",
    )


def check_required_columns(rows: List[Dict[str, str]]) -> CheckResult:
    """Check that all required columns exist in the CSV."""
    if not rows:
        return CheckResult(
            name="required_columns", status="WARN",
            detail="No rows to check",
        )
    present = set(rows[0].keys())
    missing = [c for c in REQUIRED_COLUMNS if c not in present]
    if missing:
        return CheckResult(
            name="required_columns", status="WARN",
            detail=f"Missing columns: {', '.join(missing)}",
            value=missing,
        )
    return CheckResult(
        name="required_columns", status="PASS",
        detail=f"All {len(REQUIRED_COLUMNS)} required columns present",
    )


# ---------------------------------------------------------------------------
# Scorecard orchestrator
# ---------------------------------------------------------------------------

@dataclass
class Scorecard:
    schema: str = SCHEMA_VERSION
    snapshot_date: str = ""
    n_rows: int = 0
    n_eligible: int = 0
    verdict: str = "PASS"
    checks: List[Dict[str, Any]] = field(default_factory=list)
    missingness_detail: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def run_scorecard(
    snapshot_dir: Path,
    warn_missingness: float = DEFAULT_WARN_MISSINGNESS,
    warn_tie_pct: float = DEFAULT_WARN_TIE_PCT,
) -> Scorecard:
    """Run all consistency checks on a snapshot and produce a scorecard."""
    csv_path = snapshot_dir / "rankings.csv"
    snap_date = snapshot_dir.name

    rows: List[Dict[str, str]] = []
    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

    sc = Scorecard(snapshot_date=snap_date, n_rows=len(rows))
    sc.n_eligible = sum(1 for r in rows if r.get("eligible", "0") == "1")

    # Run checks
    check_results: List[CheckResult] = []

    check_results.append(check_required_columns(rows))
    check_results.append(check_duplicate_tickers(rows))

    miss_check, miss_detail = check_missingness(rows, warn_missingness)
    check_results.append(miss_check)
    sc.missingness_detail = {k: round(v, 4) for k, v in miss_detail.items()
                            if v > 0}

    check_results.append(check_nan_hotspots(rows))
    check_results.append(check_tie_density(rows, warn_tie_pct))
    check_results.append(check_rank_invariants(rows))
    check_results.append(check_eligibility_tier_consistency(rows))

    sc.checks = [asdict(cr) for cr in check_results]

    # Overall verdict
    if any(cr.status == "WARN" for cr in check_results):
        sc.verdict = "WARN"

    return sc


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_scorecard_json(sc: Scorecard, out_dir: Path) -> Path:
    path = out_dir / f"scorecard_{sc.snapshot_date}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sc.to_dict(), f, indent=2, default=str)
    return path


def write_scorecard_md(sc: Scorecard, out_dir: Path) -> Path:
    path = out_dir / f"scorecard_{sc.snapshot_date}.md"
    lines = [
        "# Internal Consistency Scorecard",
        "",
        f"- **Snapshot**: {sc.snapshot_date}",
        f"- **Rows**: {sc.n_rows}",
        f"- **Eligible**: {sc.n_eligible}",
        f"- **Verdict**: {sc.verdict}",
        "",
        "## Checks",
        "",
        "| Check | Status | Detail |",
        "|-------|--------|--------|",
    ]
    for chk in sc.checks:
        lines.append(f"| {chk['name']} | {chk['status']} | {chk['detail']} |")

    if sc.missingness_detail:
        lines.extend(["", "## Missingness Detail (columns with missing values)", ""])
        for col, rate in sorted(sc.missingness_detail.items(), key=lambda x: -x[1])[:20]:
            lines.append(f"- `{col}`: {rate:.1%}")

    lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Internal consistency scorecard for a snapshot",
    )
    parser.add_argument(
        "--snapshot-dir", type=Path, required=True,
        help="Path to a single snapshot date directory",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help="Output directory (default: same as snapshot dir)",
    )
    parser.add_argument(
        "--warn-missingness", type=float, default=DEFAULT_WARN_MISSINGNESS,
    )
    parser.add_argument(
        "--warn-tie-pct", type=float, default=DEFAULT_WARN_TIE_PCT,
    )
    args = parser.parse_args()

    if not args.snapshot_dir.exists():
        print(f"ERROR: Snapshot dir not found: {args.snapshot_dir}")
        sys.exit(1)

    sc = run_scorecard(
        args.snapshot_dir,
        warn_missingness=args.warn_missingness,
        warn_tie_pct=args.warn_tie_pct,
    )

    out_dir = args.out_dir or args.snapshot_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    write_scorecard_json(sc, out_dir)
    write_scorecard_md(sc, out_dir)

    print(f"Scorecard: {sc.verdict}")
    for chk in sc.checks:
        print(f"  [{chk['status']}] {chk['name']}: {chk['detail']}")
    print(f"Output → {out_dir}")


if __name__ == "__main__":
    main()
