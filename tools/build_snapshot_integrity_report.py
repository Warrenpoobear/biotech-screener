#!/usr/bin/env python3
"""build_snapshot_integrity_report.py — Read-only structural validator.

Validates the per-snapshot rankings.csv for invariants that must hold
regardless of model behavior. Catches silent breakage early — duplicate
ranks, gaps in rank space, missing required columns, blank tickers, off-size
v2 cohort, etc. Also surfaces run provenance (dirty tree / commit / screen
exit) from run_manifest.json so a non-reproducible run is visible rather than
silently passing every structural check.

Diagnostic only. Does NOT modify scoring, selectors, ranking, eligibility,
or portfolio construction.

Outputs under data/snapshots/{date}/:
    snapshot_integrity_report.json
    snapshot_integrity_report.md

Usage:
    python tools/build_snapshot_integrity_report.py
    python tools/build_snapshot_integrity_report.py --as-of-date 2026-04-27
    python tools/build_snapshot_integrity_report.py --strict   # exit 1 on FAIL
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOTS_DIR = REPO_ROOT / "data" / "snapshots"
SCHEMA_VERSION = "snapshot_integrity_report.v1"
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

EXPECTED_V2_COHORT_SIZE = 60
TOP_N = 30

REQUIRED_COLUMNS = [
    "ticker",
    "company_name",
    "actionable_rank",
    "tier_any",
    "stage_bucket",
    "catalyst_days",
    "eligible",
    "ranker_v2_score",
    "selector_score",
    "final_score",
    "composite_score",
    "decision_engine_version",
    "decision_engine_ruleset_id",
]

SEVERITY_ORDER = ["PASS", "INFO", "WARN", "FAIL"]


def _max_severity(a: str, b: str) -> str:
    return a if SEVERITY_ORDER.index(a) >= SEVERITY_ORDER.index(b) else b


def _si(val: Any) -> Optional[int]:
    if val is None:
        return None
    s = str(val).strip()
    if not s or s.lower() == "nan":
        return None
    try:
        return int(float(s))
    except (TypeError, ValueError):
        return None


def _truthy(val: Any) -> bool:
    return str(val or "").strip().lower() in ("1", "true", "yes")


def find_latest_date(snapshots_dir: Path) -> Optional[str]:
    if not snapshots_dir.exists():
        return None
    dates = [
        p.name
        for p in snapshots_dir.iterdir()
        if p.is_dir() and DATE_RE.match(p.name) and (p / "rankings.csv").exists()
    ]
    return max(dates) if dates else None


def load_rankings(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = list(reader.fieldnames or [])
        rows = list(reader)
    return cols, rows


def check_columns(cols: list[str]) -> dict[str, Any]:
    missing = [c for c in REQUIRED_COLUMNS if c not in cols]
    return {
        "name": "required_columns_present",
        "severity": "FAIL" if missing else "PASS",
        "n_required": len(REQUIRED_COLUMNS),
        "missing": missing,
        "n_total_columns": len(cols),
    }


def check_ticker_uniqueness(rows: list[dict[str, str]]) -> dict[str, Any]:
    seen: dict[str, int] = {}
    blanks = 0
    for r in rows:
        t = (r.get("ticker") or "").strip()
        if not t:
            blanks += 1
            continue
        seen[t] = seen.get(t, 0) + 1
    duplicates = {t: n for t, n in seen.items() if n > 1}
    sev = "PASS"
    if blanks or duplicates:
        sev = "FAIL"
    return {
        "name": "ticker_uniqueness",
        "severity": sev,
        "n_rows": len(rows),
        "n_unique": len(seen),
        "n_blank": blanks,
        "duplicates": duplicates,
    }


def check_company_names(rows: list[dict[str, str]]) -> dict[str, Any]:
    blanks: list[str] = []
    for r in rows:
        if not (r.get("company_name") or "").strip():
            blanks.append(r.get("ticker", "?"))
    return {
        "name": "company_name_populated",
        "severity": "WARN" if blanks else "PASS",
        "n_blank": len(blanks),
        "tickers_with_blank": blanks[:20],
    }


def check_rank_space(rows: list[dict[str, str]]) -> dict[str, Any]:
    rank_to_tickers: dict[int, list[str]] = {}
    eligible_with_no_rank: list[str] = []
    ineligible_with_rank: list[str] = []
    for r in rows:
        tk = r.get("ticker", "")
        rk = _si(r.get("actionable_rank"))
        eligible = _truthy(r.get("eligible"))
        if rk is None:
            if eligible:
                eligible_with_no_rank.append(tk)
            continue
        rank_to_tickers.setdefault(rk, []).append(tk)
        if not eligible:
            ineligible_with_rank.append(tk)

    duplicates = {rk: tks for rk, tks in rank_to_tickers.items() if len(tks) > 1}
    ranks = sorted(rank_to_tickers.keys())
    gaps: list[int] = []
    if ranks:
        expected = set(range(min(ranks), max(ranks) + 1))
        gaps = sorted(expected - set(ranks))

    starts_at_one = bool(ranks) and ranks[0] == 1
    severity = "PASS"
    if duplicates or gaps or not starts_at_one:
        severity = "FAIL"
    if eligible_with_no_rank or ineligible_with_rank:
        severity = _max_severity(severity, "WARN")

    return {
        "name": "rank_space_integrity",
        "severity": severity,
        "n_ranked": len(ranks),
        "min_rank": ranks[0] if ranks else None,
        "max_rank": ranks[-1] if ranks else None,
        "starts_at_one": starts_at_one,
        "duplicate_ranks": duplicates,
        "missing_ranks": gaps,
        "n_eligible_without_rank": len(eligible_with_no_rank),
        "eligible_without_rank": eligible_with_no_rank[:20],
        "n_ineligible_with_rank": len(ineligible_with_rank),
        "ineligible_with_rank": ineligible_with_rank[:20],
    }


def check_top_n(rows: list[dict[str, str]], n: int) -> dict[str, Any]:
    in_top_n = []
    for r in rows:
        rk = _si(r.get("actionable_rank"))
        if rk is not None and rk <= n:
            in_top_n.append(r.get("ticker", "?"))
    severity = "PASS" if len(in_top_n) == n else "FAIL"
    return {
        "name": f"top_{n}_size",
        "severity": severity,
        "expected": n,
        "actual": len(in_top_n),
        "tickers": sorted(in_top_n),
    }


def check_v2_cohort(rows: list[dict[str, str]]) -> dict[str, Any]:
    cohort = [r for r in rows if str(r.get("ranker_v2_score") or "").strip()]
    severity = "PASS" if len(cohort) == EXPECTED_V2_COHORT_SIZE else "FAIL"
    return {
        "name": "v2_cohort_size",
        "severity": severity,
        "expected": EXPECTED_V2_COHORT_SIZE,
        "actual": len(cohort),
    }


def check_eligible_count(rows: list[dict[str, str]]) -> dict[str, Any]:
    eligible = sum(1 for r in rows if _truthy(r.get("eligible")))
    severity = "PASS" if eligible > 0 else "FAIL"
    return {
        "name": "eligible_count_positive",
        "severity": severity,
        "n_eligible": eligible,
        "n_total": len(rows),
        "eligible_pct": round(100.0 * eligible / max(len(rows), 1), 2),
    }


def check_decision_engine_consistency(rows: list[dict[str, str]]) -> dict[str, Any]:
    versions = {r.get("decision_engine_version", "") for r in rows if (r.get("decision_engine_version") or "").strip()}
    rulesets = {
        r.get("decision_engine_ruleset_id", "") for r in rows if (r.get("decision_engine_ruleset_id") or "").strip()
    }
    severity = "PASS"
    if len(versions) > 1 or len(rulesets) > 1:
        severity = "FAIL"
    elif not versions or not rulesets:
        severity = "WARN"
    return {
        "name": "decision_engine_consistency",
        "severity": severity,
        "versions": sorted(versions),
        "rulesets": sorted(rulesets),
    }


def check_provenance(snapshot_dir: Path) -> dict[str, Any]:
    """Surface the code-provenance state of the run that produced this snapshot.

    Reads run_manifest.json (written by run_daily_production). Closes the gap
    where the integrity report was blind to *how* a snapshot was produced: a run
    on a dirty tree / mismatched commit / non-zero screen exit passes every
    structural check yet is non-reproducible (see the 2026-07-15 provenance
    flag and its blast-radius diff).

    Severity is deliberately calibrated to be non-blocking, because dirty-tree
    and non-zero screen_exit runs are currently tolerated by design
    (run_daily_production logs a non-zero screen exit as a warning and
    continues):
      - PASS : clean run, or no manifest to assess
      - INFO : dirty working tree and/or non-zero screen_exit_code
               (surfaced, keeps report `ok` True)
      - WARN : commit_sha does not match an explicitly pinned reference
               (env BIOTECH_PRODUCTION_PINNED_SHA) — a real provenance mismatch

    INFO (not WARN) for dirty/exit means this never flips `ok` to False on the
    ~norm of dirty runs, so it does not disturb any consumer that reads `ok`.
    """
    manifest_path = snapshot_dir / "run_manifest.json"
    if not manifest_path.exists():
        return {
            "name": "run_provenance",
            "severity": "PASS",
            "manifest_present": False,
            "note": "no run_manifest.json — provenance not assessable",
        }
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        return {
            "name": "run_provenance",
            "severity": "INFO",
            "manifest_present": True,
            "note": f"run_manifest.json unreadable: {exc}",
        }

    git = manifest.get("git") or {}
    commit_sha = git.get("commit_sha")
    dirty = bool(git.get("dirty"))
    screen_exit = manifest.get("screen_exit_code")

    pinned = os.environ.get("BIOTECH_PRODUCTION_PINNED_SHA") or None
    # short/full sha tolerant match (pin may be an 8-char short sha)
    pin_mismatch = bool(
        pinned
        and commit_sha
        and not str(commit_sha).startswith(str(pinned))
        and not str(pinned).startswith(str(commit_sha))
    )

    severity = "PASS"
    reasons: list[str] = []
    if dirty:
        severity = _max_severity(severity, "INFO")
        reasons.append("dirty_working_tree")
    if screen_exit not in (0, None):
        severity = _max_severity(severity, "INFO")
        reasons.append(f"screen_exit_code={screen_exit}")
    if pin_mismatch:
        severity = _max_severity(severity, "WARN")
        reasons.append(f"commit_sha={str(commit_sha)[:10]} != pinned={pinned}")

    return {
        "name": "run_provenance",
        "severity": severity,
        "manifest_present": True,
        "commit_sha": commit_sha,
        "dirty": dirty,
        "screen_exit_code": screen_exit,
        "pinned_sha": pinned,
        "reasons": reasons,
    }


def build_integrity_report(snapshot_dir: Path) -> dict[str, Any]:
    rankings_path = snapshot_dir / "rankings.csv"
    if not rankings_path.exists():
        return {
            "schema_version": SCHEMA_VERSION,
            "as_of_date": snapshot_dir.name,
            "ok": False,
            "overall_severity": "FAIL",
            "checks": [],
            "error": f"missing rankings.csv at {rankings_path}",
        }

    cols, rows = load_rankings(rankings_path)
    checks = [
        check_columns(cols),
        check_ticker_uniqueness(rows),
        check_company_names(rows),
        check_rank_space(rows),
        check_top_n(rows, TOP_N),
        check_v2_cohort(rows),
        check_eligible_count(rows),
        check_decision_engine_consistency(rows),
        check_provenance(snapshot_dir),
    ]
    overall = "PASS"
    for c in checks:
        overall = _max_severity(overall, c["severity"])

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "as_of_date": snapshot_dir.name,
        "rankings_csv": str(rankings_path),
        "n_columns": len(cols),
        "n_rows": len(rows),
        "ok": overall in ("PASS", "INFO"),
        "overall_severity": overall,
        "checks": checks,
    }


def render_md(report: dict[str, Any]) -> str:
    lines = [
        f"# Snapshot integrity report — {report.get('as_of_date')}",
        "",
        f"- **overall**: `{report.get('overall_severity')}`",
        f"- **rows**: {report.get('n_rows')}    **columns**: {report.get('n_columns')}",
        f"- **rankings.csv**: `{report.get('rankings_csv')}`",
        "",
    ]
    if "error" in report:
        lines.append(f"_Error: {report['error']}_")
        return "\n".join(lines)

    lines.append("## Checks")
    lines.append("")
    lines.append("| check | severity | detail |")
    lines.append("|---|---|---|")
    for c in report["checks"]:
        detail = {k: v for k, v in c.items() if k not in ("name", "severity")}
        # Truncate long lists for display
        for k, v in list(detail.items()):
            if isinstance(v, list) and len(v) > 5:
                detail[k] = v[:5] + [f"…+{len(v)-5}"]
        lines.append(f"| {c['name']} | **{c['severity']}** | {detail} |")
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of-date")
    parser.add_argument("--snapshots-dir", default=str(SNAPSHOTS_DIR))
    parser.add_argument("--strict", action="store_true", help="exit 1 on FAIL")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    snapshots_dir = Path(args.snapshots_dir)
    as_of = args.as_of_date or find_latest_date(snapshots_dir)
    if not as_of:
        print(f"ERROR: no snapshots found under {snapshots_dir}")
        return 2

    snapshot_dir = snapshots_dir / as_of
    report = build_integrity_report(snapshot_dir)

    snapshot_dir.mkdir(parents=True, exist_ok=True)
    json_path = snapshot_dir / "snapshot_integrity_report.json"
    md_path = snapshot_dir / "snapshot_integrity_report.md"
    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    md_path.write_text(render_md(report), encoding="utf-8")

    if not args.quiet:
        n_fail = sum(1 for c in report.get("checks", []) if c["severity"] == "FAIL")
        n_warn = sum(1 for c in report.get("checks", []) if c["severity"] == "WARN")
        print(f"snapshot_integrity {as_of}: overall={report['overall_severity']} " f"FAIL={n_fail} WARN={n_warn}")
        print(f"  json: {json_path}")
        print(f"  md:   {md_path}")
        if n_fail:
            failed = [c["name"] for c in report["checks"] if c["severity"] == "FAIL"]
            print(f"  failed checks: {', '.join(failed)}")

    if args.strict and report.get("overall_severity") == "FAIL":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
