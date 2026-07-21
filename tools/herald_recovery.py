#!/usr/bin/env python3
"""Herald pipeline recovery — deterministic fetch → dedupe → classify → digest.

Operator-host tool for F-2026-005 recovery. Does not modify scoring or rulesets.
Re-runs herald_health_check at the end and logs telemetry.

Usage:
    python3 tools/herald_recovery.py --as-of-date 2026-06-24
    python3 tools/herald_recovery.py --as-of-date 2026-06-24 --dry-run
    python3 tools/herald_recovery.py --as-of-date 2026-06-24 --full --digest
    python3 tools/herald_health_check.py --as-of-date 2026-06-24 --recover
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

PYTHON = sys.executable

STEP_TIMEOUTS = {
    "fetch": 1500,
    "dedupe": 120,
    "classify": 300,
    "digest": 180,
}


def _artifact_paths(as_of: str) -> dict[str, Path]:
    pr = REPO / "data" / "press_releases"
    return {
        "releases": pr / f"releases_{as_of}.jsonl",
        "deduped": pr / "deduped" / f"deduped_{as_of}.jsonl",
        "classified": pr / "classified" / f"classified_{as_of}.jsonl",
    }


def plan_recovery_steps(
    as_of: str,
    *,
    full: bool = False,
    include_digest: bool = False,
    pre_report: dict[str, Any] | None = None,
) -> list[str]:
    """Return ordered recovery step names for the given date."""
    if full:
        steps = ["fetch", "dedupe", "classify"]
        if include_digest:
            steps.append("digest")
        return steps

    paths = _artifact_paths(as_of)
    steps: list[str] = []

    if pre_report and not pre_report.get("herald_done"):
        if not paths["releases"].exists():
            steps.append("fetch")
        if not paths["deduped"].exists():
            if "fetch" not in steps:
                steps.append("fetch")
            steps.append("dedupe")
        steps.append("classify")
        if include_digest:
            steps.append("digest")
        return steps

    if not paths["classified"].exists():
        if not paths["releases"].exists():
            steps.append("fetch")
        if not paths["deduped"].exists():
            if "fetch" not in steps:
                steps.append("fetch")
            steps.append("dedupe")
        steps.append("classify")
    if include_digest:
        steps.append("digest")
    return steps


def _step_command(step: str, as_of: str) -> list[str]:
    if step == "fetch":
        return [PYTHON, str(REPO / "tools/fetch_company_press_releases.py"), "--as-of-date", as_of]
    if step == "dedupe":
        rel = f"data/press_releases/releases_{as_of}.jsonl"
        return [PYTHON, str(REPO / "tools/dedupe_press_releases.py"), "--input", rel]
    if step == "classify":
        rel = f"data/press_releases/deduped/deduped_{as_of}.jsonl"
        return [PYTHON, str(REPO / "tools/classify_press_releases.py"), "--input", rel]
    if step == "digest":
        return [
            PYTHON,
            str(REPO / "scripts/build_news_digest.py"),
            "--window",
            "evening",
            "--as-of-date",
            as_of,
        ]
    raise ValueError(f"unknown step: {step}")


def run_step(step: str, as_of: str, *, dry_run: bool = False) -> tuple[int, str]:
    cmd = _step_command(step, as_of)
    cmd_str = " ".join(cmd)
    if dry_run:
        return 0, f"DRY_RUN: {cmd_str}"

    timeout = STEP_TIMEOUTS.get(step, 300)
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return 124, f"TIMEOUT after {timeout}s: {cmd_str}"
    except OSError as exc:
        return 1, f"OS_ERROR: {exc}"

    tail = (proc.stdout or proc.stderr or "").strip().split("\n")[-5:]
    detail = f"exit={proc.returncode} {' | '.join(tail)}" if tail else f"exit={proc.returncode}"
    return proc.returncode, detail


def run_recovery(
    as_of: str,
    *,
    dry_run: bool = False,
    full: bool = False,
    include_digest: bool = False,
    pre_report: dict[str, Any] | None = None,
) -> int:
    """Execute recovery steps and return herald_health_check exit code."""
    from tools.herald_health_check import _exit_code, run_check

    started = time.perf_counter()
    steps = plan_recovery_steps(
        as_of,
        full=full,
        include_digest=include_digest,
        pre_report=pre_report,
    )

    if not steps:
        print(f"[herald_recovery] {as_of}: herald_done — no steps needed")
        report = pre_report or run_check(date.fromisoformat(as_of))
        return _exit_code(report["verdict"])

    results: dict[str, Any] = {"as_of_date": as_of, "steps": steps, "step_results": {}}
    print(f"[herald_recovery] {as_of}: running {steps} (dry_run={dry_run})")

    for step in steps:
        if step == "dedupe" and not _artifact_paths(as_of)["releases"].exists() and not dry_run:
            print(f"[herald_recovery] skip dedupe — releases missing for {as_of}")
            results["step_results"][step] = {"skipped": True, "reason": "no releases file"}
            continue
        if step == "classify" and not _artifact_paths(as_of)["deduped"].exists() and not dry_run:
            print(f"[herald_recovery] skip classify — deduped missing for {as_of}")
            results["step_results"][step] = {"skipped": True, "reason": "no deduped file"}
            continue

        rc, detail = run_step(step, as_of, dry_run=dry_run)
        results["step_results"][step] = {"exit_code": rc, "detail": detail}
        print(f"[herald_recovery] {step}: {detail}")
        if rc not in (0, 124) and step in ("fetch", "classify"):
            print(f"[herald_recovery] abort after {step} failure")
            break

    if dry_run:
        return 0

    final_report = run_check(date.fromisoformat(as_of))
    final_rc = _exit_code(final_report["verdict"])
    print(
        f"[herald_recovery] final health: {final_report['verdict']} "
        f"(done={final_report['herald_done']}, exit={final_rc})"
    )

    try:
        from tools.agent_skill_telemetry import log_agent_run
        from tools.record_skill_feedback import attach_outcome_verdict

        exec_id = log_agent_run(
            "herald_recovery",
            f"Herald recovery for {as_of}",
            inputs={"as_of_date": as_of, "steps": steps, "full": full},
            outputs={
                "final_verdict": final_report.get("verdict"),
                "herald_done": final_report.get("herald_done"),
                "step_results": results["step_results"],
            },
            success=final_rc == 0,
            error=None if final_rc == 0 else final_report.get("verdict"),
            latency_ms=(time.perf_counter() - started) * 1000,
        )
        if exec_id:
            attach_outcome_verdict(
                exec_id,
                was_correct=final_report.get("verdict") == "HEALTHY" and final_report.get("herald_done"),
                evidence=f"post_recovery verdict={final_report.get('verdict')} done={final_report.get('herald_done')}",
            )
    except Exception:
        pass

    return final_rc


def main() -> int:
    ap = argparse.ArgumentParser(description="Herald pipeline recovery (operator host)")
    ap.add_argument("--as-of-date", default=date.today().isoformat())
    ap.add_argument("--dry-run", action="store_true", help="Print steps only, do not execute")
    ap.add_argument("--full", action="store_true", help="Run fetch→dedupe→classify regardless of partial artifacts")
    ap.add_argument("--digest", action="store_true", help="Include evening digest build")
    args = ap.parse_args()

    pre_report = None
    if not args.full and not args.dry_run:
        from tools.herald_health_check import run_check

        pre_report = run_check(date.fromisoformat(args.as_of_date))
        if pre_report["verdict"] == "HEALTHY" and pre_report["herald_done"]:
            print(f"[herald_recovery] {args.as_of_date}: already HEALTHY")
            return 0

    return run_recovery(
        args.as_of_date,
        dry_run=args.dry_run,
        full=args.full,
        include_digest=args.digest,
        pre_report=pre_report,
    )


if __name__ == "__main__":
    raise SystemExit(main())
