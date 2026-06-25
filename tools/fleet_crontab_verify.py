#!/usr/bin/env python3
"""Read-only verification: live operator crontab vs fleet install reference.

Compares active crontab lines against required jobs from
`install_agent_fleet_crontab.sh`. Returns SKIP when crontab is unavailable
(cloud VMs) so CI wiring audits are not blocked.

Usage:
    python3 tools/fleet_crontab_verify.py
    python3 tools/fleet_crontab_verify.py --json
    python3 tools/fleet_crontab_verify.py --write
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "artifacts" / "fleet_ops"

# Needles must match tools/install_agent_fleet_crontab.sh active lines.
FLEET_CRON_JOBS: list[dict[str, Any]] = [
    {
        "job": "herald_health",
        "needle": "herald_health_check.py",
        "schedule": "40 14 * * 1-5",
    },
    {
        "job": "intraday_mover",
        "needle": "cron_intraday_mover.sh",
        "schedule": "weekday poll + digest",
    },
    {
        "job": "weekly_skills_review",
        "needle": "cron_weekly_skills_review.sh",
        "schedule": "30 19 * * 5",
    },
    {
        "job": "data_auditor",
        "needle": "cron_data_auditor.sh",
        "schedule": "30 17 * * 1-5",
    },
    {
        "job": "heartbeat_checks",
        "needle": "agent_heartbeat_checks.py",
        "schedule": "30 17 * * 1-5",
    },
    {
        "job": "hermes_knowledge",
        "needle": "build_hermes_knowledge_layer.py",
        "schedule": "0 18 * * 1-5",
    },
    {
        "job": "hermes_contradiction",
        "needle": "hermes-contradiction-detector/run_job.py",
        "schedule": "5 18 * * 1-5",
    },
    {
        "job": "evening_catchup",
        "needle": "cron_evening_catchup.sh",
        "schedule": "0 22 * * 1-5",
    },
    {
        "job": "watchdog_reboot",
        "needle": "cron_watchdog.sh",
        "schedule": "@reboot",
        "matcher": lambda line: "@reboot" in line and "cron_watchdog.sh" in line,
    },
    {
        "job": "watchdog_noon",
        "needle": "cron_watchdog.sh",
        "schedule": "30 12 * * 1-5",
        "matcher": lambda line: "30 12" in line and "cron_watchdog.sh" in line,
    },
]


def _line_matches(line: str, job: dict[str, Any]) -> bool:
    matcher: Callable[[str], bool] | None = job.get("matcher")
    if matcher is not None:
        return matcher(line)
    return job["needle"] in line


def read_active_crontab_lines() -> tuple[str, list[str]]:
    """Return (availability, active_lines). availability: OPERATOR_HOST | UNAVAILABLE."""
    if shutil.which("crontab") is None:
        return "UNAVAILABLE", []
    try:
        proc = subprocess.run(
            ["crontab", "-l"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return "UNAVAILABLE", []

    if proc.returncode != 0:
        return "OPERATOR_HOST", []

    active: list[str] = []
    for line in proc.stdout.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith(("SHELL=", "PATH=", "MAILTO=", "CRON_TZ=")):
            continue
        active.append(stripped)
    return "OPERATOR_HOST", active


def verify_crontab() -> dict[str, Any]:
    availability, active_lines = read_active_crontab_lines()
    checks: list[dict[str, Any]] = []

    if availability == "UNAVAILABLE":
        return {
            "schema": "fleet_crontab_verify.v1",
            "generated_at": datetime.now().isoformat(),
            "availability": availability,
            "overall": "SKIP",
            "pass_count": 0,
            "fail_count": 0,
            "skip_count": len(FLEET_CRON_JOBS),
            "active_job_count": 0,
            "checks": [
                {
                    "check": "crontab_available",
                    "status": "SKIP",
                    "detail": "crontab binary unavailable (cloud VM)",
                }
            ],
            "install_reference": "bash tools/install_agent_fleet_crontab.sh",
        }

    for job in FLEET_CRON_JOBS:
        matched = any(_line_matches(line, job) for line in active_lines)
        checks.append(
            {
                "check": "fleet_cron_job",
                "job": job["job"],
                "needle": job["needle"],
                "schedule": job["schedule"],
                "status": "PASS" if matched else "FAIL",
            }
        )

    fail = sum(1 for c in checks if c["status"] == "FAIL")
    pass_n = sum(1 for c in checks if c["status"] == "PASS")
    overall = "PASS" if fail == 0 else "FAIL"

    return {
        "schema": "fleet_crontab_verify.v1",
        "generated_at": datetime.now().isoformat(),
        "availability": availability,
        "overall": overall,
        "pass_count": pass_n,
        "fail_count": fail,
        "skip_count": 0,
        "active_job_count": len(active_lines),
        "checks": checks,
        "install_reference": "bash tools/install_agent_fleet_crontab.sh",
    }


def _print_human(report: dict[str, Any]) -> None:
    print("=" * 72)
    print(f"FLEET CRONTAB VERIFY — overall={report['overall']} availability={report['availability']}")
    print(f"  active lines: {report.get('active_job_count', 0)}")
    print("=" * 72)
    if report["overall"] == "SKIP":
        print("  crontab unavailable — re-run on WSL operator host after `crontab -e` install")
        print(f"  reference: {report['install_reference']}")
        print("=" * 72)
        return
    for item in report["checks"]:
        if item.get("status") == "FAIL":
            print(f"  FAIL  {item.get('job')} ({item.get('schedule')})")
    if report["fail_count"] == 0:
        print("  All fleet cron jobs present in live crontab.")
    else:
        print(f"  Missing {report['fail_count']} job(s). Paste block from:")
        print(f"    {report['install_reference']}")
    print("=" * 72)


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify live crontab against fleet install reference")
    ap.add_argument("--as-of-date", help="YYYY-MM-DD for artifact filename (default: today)")
    ap.add_argument("--json", action="store_true", help="Print JSON only")
    ap.add_argument("--write", action="store_true", help="Write artifacts/fleet_ops/{date}_crontab_verify.json")
    args = ap.parse_args()

    ds = args.as_of_date or date.today().isoformat()
    report = verify_crontab()
    report["as_of_date"] = ds

    if args.write:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUT_DIR / f"{ds}_crontab_verify.json"
        out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if not args.json:
            print(f"Wrote {out_path}")

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_human(report)

    if report["overall"] == "FAIL":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
