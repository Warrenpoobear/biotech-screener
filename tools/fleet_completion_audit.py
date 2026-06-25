#!/usr/bin/env python3
"""Read-only audit: agent fleet deterministic wiring completeness.

Validates production cron scripts retired run_agent_direct paths, install
crontab reference coverage, evening-catchup builder map, and key tool files.

Usage:
    python3 tools/fleet_completion_audit.py
    python3 tools/fleet_completion_audit.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# Active production cron wrappers (excludes one-shot / historical scripts).
PRODUCTION_CRON_SCRIPTS = [
    "tools/cron_evening_catchup.sh",
    "tools/cron_watchdog.sh",
    "tools/cron_weekly_skills_review.sh",
    "tools/cron_data_auditor.sh",
    "tools/cron_daily_production.sh",
    "tools/cron_data_refresh.sh",
    "tools/cron_data_extras.sh",
    "tools/cron_bellringer.sh",
    "tools/cron_intraday_mover.sh",
]

INSTALL_REQUIRED_STRINGS = [
    "agent_heartbeat_checks.py",
    "cron_watchdog.sh",
    "cron_evening_catchup.sh",
    "cron_weekly_skills_review.sh",
    "fleet_ops_status.py",
    "herald_health_check.py",
]

EVENING_CATCHUP_BUILDERS = [
    "build_ops_digest.py",
    "ruleset_health_monitor.py",
    "build_catalyst_delta.py",
    "build_price_action_watch.py",
    "build_hermes_knowledge_layer.py",
    "hermes-contradiction-detector/run_job.py",
    "agents/ops_supervisor/supervisor.py",
    "agent_supervisor_sentinel.py",
    "fleet_ops_status.py",
]

KEY_TOOL_FILES = [
    "tools/agent_heartbeat_checks.py",
    "tools/fleet_ops_status.py",
    "tools/herald_health_check.py",
    "tools/herald_recovery.py",
    "tools/pattern_to_skillpatch.py",
    "tools/skills_loop_review.py",
    "agents/ops_supervisor/supervisor.py",
    "docs/governance/RULE_12_PROMOTION_CHECKLIST.md",
]


def _active_lines(text: str) -> str:
    return "\n".join(
        line for line in text.splitlines() if line.strip() and not line.strip().startswith("#")
    )


def check_cron_llm_free() -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for rel in PRODUCTION_CRON_SCRIPTS:
        path = REPO / rel
        if not path.is_file():
            findings.append({"check": "cron_exists", "path": rel, "status": "FAIL", "detail": "missing"})
            continue
        active = _active_lines(path.read_text(encoding="utf-8"))
        if "run_agent_direct" in active:
            findings.append(
                {
                    "check": "cron_llm_free",
                    "path": rel,
                    "status": "FAIL",
                    "detail": "run_agent_direct in active cron lines",
                }
            )
        else:
            findings.append({"check": "cron_llm_free", "path": rel, "status": "PASS"})
    return findings


def check_install_crontab() -> list[dict[str, Any]]:
    path = REPO / "tools/install_agent_fleet_crontab.sh"
    if not path.is_file():
        return [{"check": "install_crontab", "status": "FAIL", "detail": "install script missing"}]
    text = path.read_text(encoding="utf-8")
    findings: list[dict[str, Any]] = []
    for needle in INSTALL_REQUIRED_STRINGS:
        if needle in text:
            findings.append({"check": "install_crontab", "needle": needle, "status": "PASS"})
        else:
            findings.append(
                {"check": "install_crontab", "needle": needle, "status": "FAIL", "detail": "not in install script"}
            )
    return findings


def check_evening_catchup_builders() -> list[dict[str, Any]]:
    path = REPO / "tools/cron_evening_catchup.sh"
    if not path.is_file():
        return [{"check": "evening_catchup", "status": "FAIL", "detail": "script missing"}]
    text = path.read_text(encoding="utf-8")
    findings: list[dict[str, Any]] = []
    for needle in EVENING_CATCHUP_BUILDERS:
        if needle in text:
            findings.append({"check": "evening_catchup_builder", "needle": needle, "status": "PASS"})
        else:
            findings.append(
                {
                    "check": "evening_catchup_builder",
                    "needle": needle,
                    "status": "FAIL",
                    "detail": "not referenced in evening catchup",
                }
            )
    return findings


def check_key_tools() -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for rel in KEY_TOOL_FILES:
        path = REPO / rel
        if path.is_file():
            findings.append({"check": "key_tool", "path": rel, "status": "PASS"})
        else:
            findings.append({"check": "key_tool", "path": rel, "status": "FAIL", "detail": "missing"})
    return findings


def check_heartbeat_llm_gated() -> dict[str, Any]:
    path = REPO / "tools/agent_heartbeat_checks.py"
    if not path.is_file():
        return {"check": "heartbeat_llm_gated", "status": "FAIL", "detail": "missing"}
    text = path.read_text(encoding="utf-8")
    if "HEARTBEAT_LLM_ESCALATE" not in text:
        return {"check": "heartbeat_llm_gated", "status": "FAIL", "detail": "env gate not found"}
    if "_heartbeat_llm_escalate_enabled" not in text:
        return {"check": "heartbeat_llm_gated", "status": "FAIL", "detail": "gate helper missing"}
    return {"check": "heartbeat_llm_gated", "status": "PASS"}


def check_supervisor_escalation_json() -> dict[str, Any]:
    path = REPO / "agents/ops_supervisor/supervisor.py"
    if not path.is_file():
        return {"check": "supervisor_escalation_json", "status": "FAIL", "detail": "missing"}
    text = path.read_text(encoding="utf-8")
    if "parse_heartbeat_escalation_json" not in text or "load_heartbeat_anomalies" not in text:
        return {"check": "supervisor_escalation_json", "status": "FAIL", "detail": "escalation loader missing"}
    return {"check": "supervisor_escalation_json", "status": "PASS"}


def build_audit() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    checks.extend(check_cron_llm_free())
    checks.extend(check_install_crontab())
    checks.extend(check_evening_catchup_builders())
    checks.extend(check_key_tools())
    checks.append(check_heartbeat_llm_gated())
    checks.append(check_supervisor_escalation_json())

    fail = sum(1 for c in checks if c.get("status") == "FAIL")
    pass_n = sum(1 for c in checks if c.get("status") == "PASS")
    overall = "PASS" if fail == 0 else "FAIL"

    return {
        "schema": "fleet_completion_audit.v1",
        "generated_at": datetime.now().isoformat(),
        "overall": overall,
        "pass_count": pass_n,
        "fail_count": fail,
        "checks": checks,
        "operator_commands": {
            "fleet_ops": "python3 tools/fleet_ops_status.py --write",
            "install_crontab": "bash tools/install_agent_fleet_crontab.sh",
            "rule_12": "docs/governance/RULE_12_PROMOTION_CHECKLIST.md",
        },
    }


def _print_human(report: dict[str, Any]) -> None:
    print("=" * 72)
    print(f"FLEET COMPLETION AUDIT — overall={report['overall']}")
    print(f"  PASS: {report['pass_count']}  FAIL: {report['fail_count']}")
    print("=" * 72)
    for item in report["checks"]:
        if item.get("status") != "FAIL":
            continue
        label = item.get("path") or item.get("needle") or item.get("check")
        detail = item.get("detail", "")
        print(f"  FAIL  {label}  {detail}")
    if report["fail_count"] == 0:
        print("  All checks passed — deterministic fleet wiring complete.")
    print("=" * 72)


def main() -> int:
    ap = argparse.ArgumentParser(description="Agent fleet completion audit (read-only)")
    ap.add_argument("--json", action="store_true", help="Print JSON only")
    args = ap.parse_args()

    report = build_audit()
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_human(report)
    return 1 if report["overall"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
