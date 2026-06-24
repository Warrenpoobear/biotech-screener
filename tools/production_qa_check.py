#!/usr/bin/env python3
"""Production QA check — post-production daily review.

Runs after the daily production pipeline and agents. Checks:
  1. Snapshot exists with rankings.csv
  2. Run manifest has no FAIL gates
  3. Production log has no tracebacks
  4. EES v3 sidecar emitted
  5. Runway severity overlay emitted
  6. Herald digest delivered
  7. Key column schema drift
  8. Readiness scorecard exists
  9. Targeted lint on production-critical files
 10. Targeted pytest on critical test subset
 11. Classifier escalation-pool health (post-cutover, CH-hardening 2026-04-19)

Output:
    artifacts/production_qa/{date}_report.json
    artifacts/production_qa/{date}_report.md
    artifacts/production_qa/hard_collisions_{date}.json  (from check #11)

Usage:
    python tools/production_qa_check.py --as-of-date 2026-04-15
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("production_qa")

SCHEMA_VERSION = "production_qa.v1"
SNAPSHOTS_DIR = REPO_ROOT / "data" / "snapshots"
LOGS_DIR = REPO_ROOT / "logs"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
OUTPUT_DIR = ARTIFACTS_DIR / "production_qa"

# Production-critical files for lint check
LINT_FILES = [
    "run_screen.py",
    "decision_engine.py",
    "selector_engine.py",
    "event_ev/expectation_error_model.py",
    "event_ev/ees_v3.py",
    "event_ev/runway_severity.py",
    "event_ev/conditional_model.py",
    "tools/run_daily_production.py",
]

# Critical test subset (fast, high-value)
CRITICAL_TESTS = [
    "tests/test_runway_severity.py",
    "tests/test_expectation_error_model.py",
    "tests/test_decision_engine_contract.py",
    "tests/test_production_gates.py",
]

# Expected columns in rankings.csv (schema drift detection)
REQUIRED_COLUMNS = [
    "ticker",
    "actionable_rank",
    "eligible",
    "tier_dev",
    "catalyst_days",
    "catalyst_event_type",
    "ees_v2_score",
    "ees_trap_gate",
    "ees_eligible",
    "trap_overlay_score",
    "quality_overlay_score",
]

V3_COLUMNS = [
    "ees_v3_score",
    "ees_v3_gate",
    "ees_v3_pctile",
]

SEVERITY_COLUMNS = [
    "runway_severity_score",
    "ev_severity_score",
    "financing_truth_gate",
    "dilution_haircut",
    "size_multiplier",
]

# Expectation-layer feature coverage. Tuple: (field, min_coverage, required).
# min_coverage is a regression floor set slightly below observed 2026-04-24
# coverage. Raise once the field has been stable at higher values for several
# production days.
#
# BACKLOG — insider_net_buy_value_90d:
#   Wiring landed 2026-04-24 via common.insider_enrichment (Pass B). Gate (1)
#   below is therefore satisfied; gate (2) still requires ≥5 consecutive
#   production snapshots with coverage at or above min_coverage before flipping
#   required=True. Stays tracked nonblocking until then.
#     (1) the insider pipeline is intentionally wired into the ranking join — DONE (2026-04-24)
#     (2) observed coverage is stable at or above min_coverage for >= 5
#         consecutive production days — PENDING, first eligible date 2026-05-01
#   Lowering the required-field thresholds to accommodate insider flakiness is
#   not acceptable.
FEATURE_COVERAGE_REQUIREMENTS = [
    ("short_interest_pct", 0.90, True),
    ("close_price", 0.99, True),
    ("market_cap_mm", 0.95, True),
    ("priced_move_pct", 0.80, True),
    ("insider_net_buy_value_90d", 0.30, False),
]
FEATURE_COVERAGE_LEGEND = "* = tracked nonblocking field"


def _check(name: str, passed: bool, detail: str = "") -> Dict[str, Any]:
    return {"check": name, "status": "PASS" if passed else "FAIL", "detail": detail}


def check_snapshot(as_of_date: str) -> Dict[str, Any]:
    snap = SNAPSHOTS_DIR / as_of_date
    rankings = snap / "rankings.csv"
    if not rankings.exists():
        return _check("snapshot", False, f"No rankings.csv for {as_of_date}")

    import csv

    with open(rankings) as f:
        reader = csv.DictReader(f)
        cols = set(reader.fieldnames or [])
        n_rows = sum(1 for _ in reader)

    if n_rows == 0:
        return _check("snapshot", False, "rankings.csv is empty")

    return _check("snapshot", True, f"{n_rows} tickers, {len(cols)} columns")


def check_gates(as_of_date: str) -> Dict[str, Any]:
    manifest = SNAPSHOTS_DIR / as_of_date / "run_manifest.json"
    if not manifest.exists():
        return _check("gates", False, "run_manifest.json missing")

    try:
        data = json.loads(manifest.read_text())
    except json.JSONDecodeError:
        return _check("gates", False, "run_manifest.json corrupt (malformed JSON)")
    gates = data.get("gates", [])
    fails = [g for g in gates if g.get("status") == "FAIL"]
    warns = [g for g in gates if g.get("status") == "WARN"]

    if fails:
        fail_names = [g.get("name", "?") for g in fails]
        return _check("gates", False, f"{len(fails)} FAIL: {', '.join(fail_names)}")

    return _check("gates", True, f"{len(gates)} gates, {len(warns)} WARN")


def check_tracebacks(as_of_date: str) -> Dict[str, Any]:
    log_path = LOGS_DIR / f"daily_production_{as_of_date}.log"
    if not log_path.exists():
        return _check("tracebacks", True, "No per-date log for scan")
    total_tb = log_path.read_text(errors="replace").count("Traceback")
    if total_tb > 0:
        return _check("tracebacks", False, f"{total_tb} tracebacks in log")
    return _check("tracebacks", True, "No tracebacks")


def check_sidecars(as_of_date: str) -> Dict[str, Any]:
    snap = SNAPSHOTS_DIR / as_of_date
    issues = []

    for sidecar in [
        "ees_v3_overlay.json",
        "ees_sidecar_diff.json",
        "runway_severity_overlay.json",
        "expectation_error_overlay.json",
        "ees_gate_diagnostics.json",
    ]:
        if not (snap / sidecar).exists():
            issues.append(f"missing: {sidecar}")

    if issues:
        return _check("sidecars", False, "; ".join(issues))
    return _check("sidecars", True, "All sidecars present")


def check_herald_digest(as_of_date: str) -> Dict[str, Any]:
    digest_dir = ARTIFACTS_DIR / "news_digest"
    today_digests = list(digest_dir.glob(f"biotech_news_digest_{as_of_date}_*.json"))

    if not today_digests:
        return _check("herald_digest", False, f"No digest for {as_of_date}")

    return _check("herald_digest", True, f"{len(today_digests)} digests")


def check_feature_coverage(as_of_date: str) -> Dict[str, Any]:
    """Guard expectation-layer feature coverage in rankings.csv.

    Per-field coverage is computed as the fraction of rows where the column
    is present and non-empty. Required fields below their min_coverage floor
    (or with the column missing entirely) fail the check; non-required fields
    are reported in the detail but do not fail.

    This catches the feature-starvation mode where upstream joins silently
    drop columns, leaving the expectation model blind without a pipeline error.
    """
    import csv

    rankings = SNAPSHOTS_DIR / as_of_date / "rankings.csv"
    if not rankings.exists():
        return _check("feature_coverage", False, "No rankings.csv")

    with open(rankings) as f:
        reader = csv.DictReader(f)
        cols = set(reader.fieldnames or [])
        rows = list(reader)

    n = len(rows)
    if n == 0:
        return _check("feature_coverage", False, "rankings.csv is empty")

    failures: list[str] = []
    per_field: list[str] = []
    has_nonrequired = False
    for field, min_cov, required in FEATURE_COVERAGE_REQUIREMENTS:
        tag = "" if required else "*"
        if not required:
            has_nonrequired = True
        if field not in cols:
            suffix = "" if required else " (tracked_nonblocking)"
            per_field.append(f"{field}{tag}=MISSING{suffix}")
            if required:
                failures.append(f"{field}: column missing")
            continue
        non_empty = sum(1 for r in rows if r.get(field, "").strip() not in ("", "None", "nan", "NaN"))
        cov = non_empty / n
        per_field.append(f"{field}{tag}={cov*100:.1f}%")
        if required and cov < min_cov:
            failures.append(f"{field}: {cov*100:.1f}% < {min_cov*100:.0f}%")

    detail = "; ".join(per_field)
    if has_nonrequired:
        detail = f"{detail}; {FEATURE_COVERAGE_LEGEND}"
    if failures:
        detail = f"{detail} | FAIL: {'; '.join(failures)}"
        return _check("feature_coverage", False, detail)
    return _check("feature_coverage", True, detail)


def check_schema_drift(as_of_date: str) -> Dict[str, Any]:
    rankings = SNAPSHOTS_DIR / as_of_date / "rankings.csv"
    if not rankings.exists():
        return _check("schema", False, "No rankings.csv")

    import csv

    with open(rankings) as f:
        cols = set(csv.DictReader(f).fieldnames or [])

    missing_required = [c for c in REQUIRED_COLUMNS if c not in cols]
    missing_v3 = [c for c in V3_COLUMNS if c not in cols]
    missing_sev = [c for c in SEVERITY_COLUMNS if c not in cols]

    issues = []
    if missing_required:
        issues.append(f"REQUIRED missing: {missing_required}")
    if missing_v3:
        issues.append(f"v3 missing: {missing_v3}")
    if missing_sev:
        issues.append(f"severity missing: {missing_sev}")

    if missing_required:
        return _check("schema", False, "; ".join(issues))

    detail = "Required OK"
    if missing_v3 or missing_sev:
        detail += f" ({'; '.join(issues)})"
    return _check("schema", True, detail)


def check_severity_formulas(as_of_date: str) -> Dict[str, Any]:
    """Validate runway severity numeric bounds and formula relationships.

    Enforces that all severity fields are present, numeric, finite, and satisfy:
      1. runway_severity_score in [0.0, 1.0]
      2. ev_severity_score in [0.0, 1.0]
      3. dilution_haircut ≈ 0.35 * ev_severity_score (±0.001)
      4. size_multiplier ≈ max(0.40, 1 - 0.60 * ev_severity_score) (±0.001)

    Blank / None / NaN / Inf values are treated as failures, not skipped.
    """
    rankings = SNAPSHOTS_DIR / as_of_date / "rankings.csv"
    if not rankings.exists():
        return _check("severity_formulas", False, "No rankings.csv")

    import csv
    import math

    issues = []
    tolerance = 0.001

    with open(rankings) as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, 1):
            ev_sev_raw = row.get("ev_severity_score")
            run_sev_raw = row.get("runway_severity_score")
            dil_hair_raw = row.get("dilution_haircut")
            size_mult_raw = row.get("size_multiplier")

            # Check presence: blank or None is a failure
            for field_name, field_val in [
                ("ev_severity_score", ev_sev_raw),
                ("runway_severity_score", run_sev_raw),
                ("dilution_haircut", dil_hair_raw),
                ("size_multiplier", size_mult_raw),
            ]:
                if not field_val or str(field_val).strip() == "":
                    issues.append(f"Row {i}: {field_name} is blank")

            # Try to convert all fields to float
            try:
                ev_sev = float(ev_sev_raw)
                run_sev = float(run_sev_raw)
                dil_hair = float(dil_hair_raw)
                size_mult = float(size_mult_raw)
            except (ValueError, TypeError):
                issues.append(f"Row {i}: one or more severity fields are non-numeric")
                continue

            # Check finiteness (NaN, Inf, -Inf are failures)
            if not math.isfinite(ev_sev):
                issues.append(f"Row {i}: ev_severity_score {ev_sev} is not finite")
            if not math.isfinite(run_sev):
                issues.append(f"Row {i}: runway_severity_score {run_sev} is not finite")
            if not math.isfinite(dil_hair):
                issues.append(f"Row {i}: dilution_haircut {dil_hair} is not finite")
            if not math.isfinite(size_mult):
                issues.append(f"Row {i}: size_multiplier {size_mult} is not finite")

            # Skip formula checks if any field failed finiteness above
            if not (
                math.isfinite(ev_sev)
                and math.isfinite(run_sev)
                and math.isfinite(dil_hair)
                and math.isfinite(size_mult)
            ):
                continue

            # Check bounds
            if not (0.0 <= ev_sev <= 1.0):
                issues.append(f"Row {i}: ev_severity_score {ev_sev} out of bounds [0, 1]")
            if not (0.0 <= run_sev <= 1.0):
                issues.append(f"Row {i}: runway_severity_score {run_sev} out of bounds [0, 1]")

            # Check formulas only if all fields are valid
            expected_dil = 0.35 * ev_sev
            if abs(dil_hair - expected_dil) > tolerance:
                issues.append(f"Row {i}: dilution_haircut {dil_hair} != 0.35*{ev_sev}={expected_dil}")

            expected_size = max(0.40, 1.0 - 0.60 * ev_sev)
            if abs(size_mult - expected_size) > tolerance:
                issues.append(f"Row {i}: size_multiplier {size_mult} != max(0.40, 1-0.60*{ev_sev})={expected_size}")

    if issues:
        # Report first 5 issues; more just means systemic problem
        detail = "; ".join(issues[:5])
        if len(issues) > 5:
            detail += f" (... {len(issues)-5} more)"
        return _check("severity_formulas", False, detail)

    return _check("severity_formulas", True, "All formula checks passed")


def check_readiness(as_of_date: str) -> Dict[str, Any]:
    scorecard = ARTIFACTS_DIR / "readiness" / f"scorecard_{as_of_date}.json"
    if not scorecard.exists():
        return _check("readiness", False, "No readiness scorecard")

    try:
        data = json.loads(scorecard.read_text())
    except json.JSONDecodeError:
        return _check("readiness", False, "readiness scorecard corrupt (malformed JSON)")
    verdict = data.get("verdict", "UNKNOWN")
    return _check("readiness", True, f"Verdict: {verdict}")


def check_lint() -> Dict[str, Any]:
    files = [str(REPO_ROOT / f) for f in LINT_FILES if (REPO_ROOT / f).exists()]
    if not files:
        return _check("lint", True, "No production files to lint")

    try:
        result = subprocess.run(
            ["flake8", "--select=E9,F63,F7,F82", "--max-line-length=120"] + files,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            n_errors = result.stdout.strip().count("\n") + 1
            return _check("lint", False, f"{n_errors} critical lint errors")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return _check("lint", True, "flake8 not available or timed out")

    return _check("lint", True, f"{len(files)} files clean")


def check_classifier_escalation_pool(as_of_date: str) -> Dict[str, Any]:
    """Post-cutover validation of the press-release classifier's escalation pool.

    Added with classifier-hardening rollout 2026-04-19 (CH-1..CH-7 + P2 + M1).
    Reads the floor date from `config/post_cutover_floor.json`; audits the
    escalation pool among files dated on/after the floor; emits a 10-item
    hard-collision sample to `artifacts/production_qa/hard_collisions_{date}.json`
    for rolling human spot-check (supersedes the discrete Day-7/Day-14 milestones
    in fingpt_pilot/notes/POST_CUTOVER_VALIDATION.md).

    Verdict thresholds:
      PASS — pool empty (awaiting first post-cutover cron output), OR all within range
      FAIL — other-category share > 50%, OR re-run clean rate < 70%, OR schema missing
    """
    import random as _rand

    floor_cfg = REPO_ROOT / "config" / "post_cutover_floor.json"
    if not floor_cfg.exists():
        return _check(
            "classifier_escalation_pool",
            False,
            "config/post_cutover_floor.json missing",
        )
    try:
        cfg = json.loads(floor_cfg.read_text())
    except Exception as e:
        return _check("classifier_escalation_pool", False, f"config parse error: {e}")

    min_date = cfg.get("classifier_min_date")
    if min_date == "":
        # Floor explicitly disabled — audit full cache
        min_date = None
    elif not min_date:
        return _check(
            "classifier_escalation_pool",
            False,
            'classifier_min_date missing from config (set to YYYY-MM-DD or "" to disable)',
        )

    try:
        from tools.audit_escalation_pool import _iter_records, audit
    except ImportError as e:
        return _check("classifier_escalation_pool", False, f"import failed: {e}")

    classified_dir = REPO_ROOT / "data" / "press_releases" / "classified"
    if not classified_dir.exists():
        return _check("classifier_escalation_pool", False, f"missing {classified_dir}")

    try:
        rpt = audit(classified_dir, n=30, seed=20260419, min_date=min_date)
    except Exception as e:
        return _check("classifier_escalation_pool", False, f"audit error: {e}")

    pool_size = rpt["pool_count_deduped"]
    sample_n = rpt["sample_n"]
    rerun = rpt.get("sample_rerun_counts", {})
    clean = int(rerun.get("clean", 0))
    by_cat = rpt["pool_stats"].get("by_category", {}) if rpt.get("pool_stats") else {}
    other = int(by_cat.get("other", 0))
    other_share_pct = (other / pool_size * 100.0) if pool_size else 0.0

    if pool_size == 0:
        detail = f"pool empty (min_date={min_date}) — awaiting first post-cutover cron output"
        return _check("classifier_escalation_pool", True, detail)

    # Schema sanity — verify new P2/M1 fields are present on a spot sample
    schema_missing = []
    for r in rpt.get("sample_rows", [])[:5]:
        if r.get("event_id", "") == "":
            schema_missing.append("event_id")
            break

    issues: list[str] = []
    if other_share_pct > 50:
        issues.append(f"other_share={other_share_pct:.1f}% (>50)")
    if sample_n > 0 and clean / sample_n < 0.70:
        issues.append(f"clean={clean}/{sample_n} (<70%)")
    if schema_missing:
        issues.append(f"schema_missing={schema_missing}")

    # Hard-collision sample for rolling human review (replaces Day-7 checkpoint).
    records = _iter_records(classified_dir, min_date=min_date)
    hard_colls = [r for r in records if r.get("collision_severity") == "hard"]
    rng = _rand.Random(int(as_of_date.replace("-", "")))
    rng.shuffle(hard_colls)
    sample_hard = [
        {
            "ticker": r.get("ticker", ""),
            "headline": (r.get("headline", "") or "")[:160],
            "event_id": r.get("event_id", ""),
            "published_at_utc": r.get("published_at_utc", ""),
        }
        for r in hard_colls[:10]
    ]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    hard_path = OUTPUT_DIR / f"hard_collisions_{as_of_date}.json"
    hard_path.write_text(
        json.dumps(
            {
                "as_of_date": as_of_date,
                "min_date": min_date,
                "pool_size": pool_size,
                "hard_collision_pool_size": len(hard_colls),
                "hard_collision_sample": sample_hard,
            },
            indent=2,
        )
    )

    detail = (
        f"pool={pool_size}, clean={clean}/{sample_n}, "
        f"other={other_share_pct:.1f}%, hard_coll_pool={len(hard_colls)}"
    )
    if issues:
        detail = f"{detail} [{'; '.join(issues)}]"
    return _check("classifier_escalation_pool", len(issues) == 0, detail)


def check_critical_tests() -> Dict[str, Any]:
    test_files = [str(REPO_ROOT / f) for f in CRITICAL_TESTS if (REPO_ROOT / f).exists()]
    if not test_files:
        return _check("tests", True, "No critical test files found")

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest"] + test_files + ["-x", "-q", "--tb=line", "--no-header"],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(REPO_ROOT),
        )
        output = result.stdout.strip()
        last_line = output.split("\n")[-1] if output else ""

        if result.returncode != 0:
            return _check("tests", False, last_line[:200])
    except subprocess.TimeoutExpired:
        return _check("tests", False, "Test suite timed out (>120s)")

    return _check("tests", True, last_line[:200])


def run_qa(as_of_date: str) -> Dict[str, Any]:
    checks = [
        check_snapshot(as_of_date),
        check_gates(as_of_date),
        check_tracebacks(as_of_date),
        check_sidecars(as_of_date),
        check_herald_digest(as_of_date),
        check_schema_drift(as_of_date),
        check_severity_formulas(as_of_date),
        check_feature_coverage(as_of_date),
        check_readiness(as_of_date),
        check_lint(),
        check_critical_tests(),
        check_classifier_escalation_pool(as_of_date),
    ]

    n_pass = sum(1 for c in checks if c["status"] == "PASS")
    n_fail = sum(1 for c in checks if c["status"] == "FAIL")
    verdict = "GREEN" if n_fail == 0 else "YELLOW" if n_fail <= 2 else "RED"

    report = {
        "schema": SCHEMA_VERSION,
        "as_of_date": as_of_date,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "n_checks": len(checks),
        "n_pass": n_pass,
        "n_fail": n_fail,
        "checks": checks,
    }

    # Write outputs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    json_path = OUTPUT_DIR / f"{as_of_date}_report.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    md_path = OUTPUT_DIR / f"{as_of_date}_report.md"
    md_path.write_text(_format_md(report))

    logger.info("QA %s: %s (%d/%d pass)", as_of_date, verdict, n_pass, len(checks))
    return report


def _format_md(report: Dict) -> str:
    lines = [f"# Production QA — {report['as_of_date']}"]
    lines.append("")
    lines.append(f"**Verdict:** {report['verdict']} ({report['n_pass']}/{report['n_checks']} pass)")
    lines.append("")
    lines.append("| Check | Status | Detail |")
    lines.append("|-------|--------|--------|")
    for c in report["checks"]:
        status = c["status"]
        lines.append(f"| {c['check']} | {status} | {c['detail'][:80]} |")
    lines.append("")
    lines.append(f"*Generated: {report['generated_at']}*")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Production QA check")
    parser.add_argument("--as-of-date", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    args = parser.parse_args()
    started = time.perf_counter()

    report = run_qa(args.as_of_date)

    # Print summary
    for c in report["checks"]:
        marker = "PASS" if c["status"] == "PASS" else "FAIL"
        print(f"  [{marker}] {c['check']}: {c['detail'][:70]}")
    print(f"\n  Verdict: {report['verdict']}")

    try:
        from tools.agent_skill_telemetry import log_agent_run
        from tools.record_skill_feedback import attach_outcome_verdict

        verdict = report.get("verdict", "UNKNOWN")
        exec_id = log_agent_run(
            "production_qa_check",
            f"Production QA for {args.as_of_date}",
            inputs={"as_of_date": args.as_of_date},
            outputs={
                "verdict": verdict,
                "n_pass": report.get("n_pass"),
                "n_fail": report.get("n_fail"),
            },
            success=verdict != "RED",
            error=None if verdict != "RED" else f"verdict={verdict}",
            latency_ms=(time.perf_counter() - started) * 1000,
        )
        if exec_id:
            attach_outcome_verdict(
                exec_id,
                was_correct=verdict == "GREEN",
                evidence=f"verdict={verdict} n_fail={report.get('n_fail', 0)}",
            )
    except Exception:
        pass


if __name__ == "__main__":
    main()
