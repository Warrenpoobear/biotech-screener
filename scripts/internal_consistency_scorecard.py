#!/usr/bin/env python3
"""Internal consistency scorecard for snapshot quality monitoring.

Checks: missingness rates (N/A-aware), NaN hotspots, tie density,
rank/tier invariants, duplicate tickers, eligibility consistency.

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

SCHEMA_VERSION = "internal_consistency_scorecard.v2"

# Columns that must be present and non-empty for every eligible ticker
REQUIRED_COLUMNS = [
    "ticker", "actionable_rank", "tier_dev", "eligible",
    "composite_rank", "composite_score", "archetype",
]

# Numeric columns to check for NaN hotspots (only should-be-present fields)
NUMERIC_COLUMNS = [
    "composite_score", "score_rank_pct", "alpha_cohort_pct",
    "de_drawdown", "de_rsi_14d", "de_beta_xbi_60d", "de_alpha_60d",
    "clinical_score_z_tier", "coinvest_score_z", "inst_delta_z",
    "catalyst_decay_w",
]

# Rank columns that should have zero or near-zero ties
RANK_COLUMNS = ["actionable_rank", "composite_rank"]

# Maximum acceptable missingness fraction before WARN
DEFAULT_WARN_MISSINGNESS = 0.05
DEFAULT_WARN_TIE_PCT = 0.02

# ---------------------------------------------------------------------------
# N/A field classification: fields that are expected to be empty for
# certain subsets of rows (not data quality failures).
# Each rule is a callable(row) -> bool; True means the field is N/A
# (expected missing) for that row.
# ---------------------------------------------------------------------------


def _is_flag_true(v) -> bool:
    """Robust boolean parsing for flag columns from CSV."""
    return str(v).strip() in ("1", "1.0", "True", "true")


NA_FIELD_RULES: Dict[str, Any] = {
    "commercial_quality_pct": lambda r: not _is_flag_true(r.get("has_commercial_quality", "")),
    "commercial_quality": lambda r: not _is_flag_true(r.get("has_commercial_quality", "")),
    "clinical_optionality_pct_dev": lambda r: not _is_flag_true(r.get("has_clinical_optionality_dev", "")),
    "clinical_rank_pct_dev": lambda r: not _is_flag_true(r.get("has_clinical_optionality_dev", "")),
    "actionable_rank": lambda r: not _is_flag_true(r.get("eligible", "")),
    "target_weight_pct": lambda r: not _is_flag_true(r.get("eligible", "")),
    "catalyst_days": lambda r: (r.get("catalyst_mode") or "").strip() in ("no_upcoming", "missing", ""),
    "coinvest_filing_age_days": lambda r: (r.get("coinvest_recency_state") or "").strip() == "",
}


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


def _is_missing(val: str) -> bool:
    """Return True if a cell value is missing/empty/NaN."""
    s = (val or "").strip()
    return not s or s.lower() in ("nan", "none")


def check_missingness(
    rows: List[Dict[str, str]],
    warn_threshold: float = DEFAULT_WARN_MISSINGNESS,
) -> Tuple[CheckResult, Dict[str, float], Dict[str, float]]:
    """Check missingness rate across all columns (N/A-aware).

    Returns (CheckResult, total_miss_rates, real_miss_rates).
    real_miss_rates excludes expected N/A rows per NA_FIELD_RULES.
    """
    if not rows:
        return CheckResult(
            name="missingness", status="WARN",
            detail="No rows to check",
        ), {}, {}

    n = len(rows)
    all_cols = list(rows[0].keys()) if rows else []
    total_miss_rates: Dict[str, float] = {}
    real_miss_rates: Dict[str, float] = {}

    for col in all_cols:
        total_missing = 0
        real_missing = 0
        na_rule = NA_FIELD_RULES.get(col)

        for r in rows:
            val = r.get(col, "")
            if _is_missing(val):
                total_missing += 1
                # If there's an N/A rule and it says this row is expected N/A,
                # don't count it as real missingness
                if na_rule and na_rule(r):
                    pass  # expected N/A
                else:
                    real_missing += 1

        total_miss_rates[col] = total_missing / n
        real_miss_rates[col] = real_missing / n

    # WARN only on real (unexpected) missingness
    high_miss = {col: rate for col, rate in real_miss_rates.items()
                 if rate > warn_threshold}

    if high_miss:
        sample = ", ".join(f"{c}={r:.1%}" for c, r in
                           sorted(high_miss.items(), key=lambda x: -x[1])[:5])
        return CheckResult(
            name="missingness", status="WARN",
            detail=f"{len(high_miss)} columns above {warn_threshold:.0%} (real): {sample}",
            value=len(high_miss), threshold=warn_threshold,
        ), total_miss_rates, real_miss_rates

    return CheckResult(
        name="missingness", status="PASS",
        detail=f"All {len(all_cols)} columns below {warn_threshold:.0%} real missingness",
        threshold=warn_threshold,
    ), total_miss_rates, real_miss_rates


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
# Breakdown helpers (for markdown output)
# ---------------------------------------------------------------------------

def _build_archetype_breakdown(rows: List[Dict[str, str]], miss_cols: List[str]) -> List[str]:
    """Build real-missingness breakdown by archetype (count + rate)."""
    if not rows or not miss_cols:
        return []
    archetypes = sorted(set(r.get("archetype", "?") for r in rows))
    lines = ["", "## Real Missingness by Archetype", "",
             "| Archetype | N | " + " | ".join(miss_cols) + " |",
             "|-----------|---|" + "|".join("---" for _ in miss_cols) + "|"]
    for arch in archetypes:
        arch_rows = [r for r in rows if r.get("archetype", "?") == arch]
        n_arch = len(arch_rows)
        if n_arch == 0:
            continue
        cells = []
        for col in miss_cols:
            na_rule = NA_FIELD_RULES.get(col)
            cnt = sum(1 for r in arch_rows
                      if _is_missing(r.get(col, ""))
                      and not (na_rule and na_rule(r)))
            pct = cnt / n_arch
            cells.append(f"{cnt} ({pct:.0%})")
        lines.append(f"| {arch} | {n_arch} | " + " | ".join(cells) + " |")
    return lines


def _build_tier_breakdown(rows: List[Dict[str, str]], miss_cols: List[str]) -> List[str]:
    """Build real-missingness breakdown by tier_dev (count + rate)."""
    if not rows or not miss_cols:
        return []
    tiers = sorted(set((r.get("tier_dev") or "").strip() or "(none)" for r in rows))
    lines = ["", "## Real Missingness by Tier", "",
             "| Tier | N | " + " | ".join(miss_cols) + " |",
             "|------|---|" + "|".join("---" for _ in miss_cols) + "|"]
    for tier in tiers:
        tier_rows = [r for r in rows
                     if ((r.get("tier_dev") or "").strip() or "(none)") == tier]
        n_tier = len(tier_rows)
        if n_tier == 0:
            continue
        cells = []
        for col in miss_cols:
            na_rule = NA_FIELD_RULES.get(col)
            cnt = sum(1 for r in tier_rows
                      if _is_missing(r.get(col, ""))
                      and not (na_rule and na_rule(r)))
            pct = cnt / n_tier
            cells.append(f"{cnt} ({pct:.0%})")
        lines.append(f"| {tier} | {n_tier} | " + " | ".join(cells) + " |")
    return lines


def _build_catalyst_mode_breakdown(rows: List[Dict[str, str]]) -> List[str]:
    """Catalyst missingness breakdown by catalyst_mode."""
    if not rows:
        return []
    modes = Counter((r.get("catalyst_mode") or "").strip() or "(empty)" for r in rows)
    if not modes:
        return []
    lines = ["", "## Catalyst Days Missingness by Mode", "",
             "| catalyst_mode | N | catalyst_days missing |",
             "|---------------|---|----------------------|"]
    for mode, n_mode in sorted(modes.items(), key=lambda x: -x[1]):
        mode_rows = [r for r in rows
                     if ((r.get("catalyst_mode") or "").strip() or "(empty)") == mode]
        miss = sum(1 for r in mode_rows if _is_missing(r.get("catalyst_days", "")))
        lines.append(f"| {mode} | {n_mode} | {miss} |")
    return lines


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
    real_missingness_detail: Dict[str, float] = field(default_factory=dict)
    rows: List[Dict[str, str]] = field(default_factory=list, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("rows", None)  # don't serialize raw rows
        return d


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
    sc.rows = rows

    # Run checks
    check_results: List[CheckResult] = []

    check_results.append(check_required_columns(rows))
    check_results.append(check_duplicate_tickers(rows))

    miss_check, total_miss, real_miss = check_missingness(rows, warn_missingness)
    check_results.append(miss_check)
    sc.missingness_detail = {k: round(v, 4) for k, v in total_miss.items()
                            if v > 0}
    sc.real_missingness_detail = {k: round(v, 4) for k, v in real_miss.items()
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
        lines.append("| Column | Total Miss | Real Miss | Expected N/A |")
        lines.append("|--------|-----------|-----------|--------------|")
        for col, total_rate in sorted(sc.missingness_detail.items(), key=lambda x: -x[1])[:20]:
            real_rate = sc.real_missingness_detail.get(col, 0.0)
            na_rate = total_rate - real_rate
            lines.append(f"| `{col}` | {total_rate:.1%} | {real_rate:.1%} | {na_rate:.1%} |")

    # Breakdown tables (only if we have rows)
    rows = sc.rows
    if rows:
        # Columns with real (unexpected) missingness > 5%
        real_cols = sorted(sc.real_missingness_detail.items(), key=lambda x: -x[1])
        miss_cols = [c for c, r in real_cols if r > 0.05][:8]
        if miss_cols:
            lines.extend(_build_archetype_breakdown(rows, miss_cols))
            lines.extend(_build_tier_breakdown(rows, miss_cols))
        lines.extend(_build_catalyst_mode_breakdown(rows))

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
