#!/usr/bin/env python3
"""Read-only audit: agent fleet deterministic wiring completeness.

Validates production cron scripts retired run_agent_direct paths, install
crontab reference coverage, evening-catchup builder map, and key tool files.

Usage:
    python3 tools/fleet_completion_audit.py
    python3 tools/fleet_completion_audit.py --json
    python3 tools/fleet_completion_audit.py --write
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "artifacts" / "fleet_ops"

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
    "fleet_completion_audit.py",
    "fleet_crontab_verify.py",
]

KEY_TOOL_FILES = [
    "tools/agent_heartbeat_checks.py",
    "tools/fleet_ops_status.py",
    "tools/fleet_completion_audit.py",
    "tools/fleet_crontab_verify.py",
    "tools/run_fleet_operator_checklist.sh",
    "tools/run_fleet_host_onboarding.sh",
    "tools/run_operator_host_setup.sh",
    "tools/run_research_host_battery.sh",
    "tools/run_forward_evidence_package.sh",
    "tools/forward_evidence_package.py",
    "tools/path_c_window_close_decision.py",
    "tools/path_a_timing_gates.py",
    "tools/run_path_a_shadow.sh",
    "production_data/portfolio_policy_path_a_shadow.json",
    "docs/governance/PATH_A_PORTFOLIO_TIMING_GATES_SPEC_106_2026_06_25.md",
    "docs/governance/FREEZE_LIFT_FORWARD_EVIDENCE_PACKAGE_2026_06_25.md",
    "tools/herald_health_check.py",
    "tools/herald_recovery.py",
    "tools/pattern_to_skillpatch.py",
    "tools/skills_loop_review.py",
    "agents/ops_supervisor/supervisor.py",
    "docs/governance/RULE_12_PROMOTION_CHECKLIST.md",
]

# Specialized heartbeat checks that are not separate registry agents.
ORPHAN_SPECIALIZED_CHECKS = {"ic_memory_hygiene"}

WATCHDOG_HERALD_RECOVERY_STRINGS = [
    "artifacts/herald/health_check_${TODAY}.json",
    "herald_health_check.py",
    "--recover",
    "recovery_done_${TODAY}.complete",
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


def check_registry_heartbeat_coverage() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Every active supervised agent must have specialized check, generic paths, or skip reason."""
    from tools.agent_heartbeat_checks import (
        SPECIALIZED_CHECKS,
        TERMINAL_UNSUPERVISED_AGENTS,
        heartbeat_skip_reason,
    )

    registry_path = REPO / "agents" / "AGENT_REGISTRY.json"
    if not registry_path.is_file():
        return [{"check": "registry_coverage", "status": "FAIL", "detail": "AGENT_REGISTRY.json missing"}], {}

    try:
        registry = json.loads(registry_path.read_text(encoding="utf-8")).get("agents", {})
    except json.JSONDecodeError:
        return [{"check": "registry_coverage", "status": "FAIL", "detail": "registry malformed"}], {}

    findings: list[dict[str, Any]] = []
    summary = {
        "active_supervised": 0,
        "specialized": 0,
        "generic_fallback": 0,
        "on_demand_skip": 0,
        "terminal_unsupervised": 0,
        "opted_out_unsupervised": 0,
    }

    for name, entry in sorted(registry.items()):
        if entry.get("status") != "active":
            continue
        if not entry.get("supervised_by_orchestrator", True):
            if name in TERMINAL_UNSUPERVISED_AGENTS:
                summary["terminal_unsupervised"] += 1
                findings.append(
                    {"check": "registry_coverage", "agent": name, "status": "PASS", "mode": "terminal_unsupervised"}
                )
            else:
                summary["opted_out_unsupervised"] += 1
                findings.append(
                    {"check": "registry_coverage", "agent": name, "status": "PASS", "mode": "opted_out_unsupervised"}
                )
            continue

        summary["active_supervised"] += 1
        skip_reason = heartbeat_skip_reason(name, entry)
        if skip_reason:
            summary["on_demand_skip"] += 1
            findings.append(
                {"check": "registry_coverage", "agent": name, "status": "PASS", "mode": "on_demand_skip"}
            )
            continue
        if name in SPECIALIZED_CHECKS:
            summary["specialized"] += 1
            findings.append(
                {"check": "registry_coverage", "agent": name, "status": "PASS", "mode": "specialized"}
            )
        elif entry.get("artifact_paths"):
            summary["generic_fallback"] += 1
            findings.append(
                {"check": "registry_coverage", "agent": name, "status": "PASS", "mode": "generic_fallback"}
            )
        else:
            findings.append(
                {
                    "check": "registry_coverage",
                    "agent": name,
                    "status": "FAIL",
                    "detail": "no specialized check or artifact_paths",
                }
            )

    for key in sorted(SPECIALIZED_CHECKS):
        if key in registry or key in ORPHAN_SPECIALIZED_CHECKS:
            continue
        findings.append(
            {
                "check": "registry_orphan_check",
                "agent": key,
                "status": "WARN",
                "detail": "SPECIALIZED_CHECKS key not in registry",
            }
        )

    return findings, summary


def check_deprecated_merged_into() -> list[dict[str, Any]]:
    registry_path = REPO / "agents" / "AGENT_REGISTRY.json"
    if not registry_path.is_file():
        return [{"check": "deprecated_merged_into", "status": "FAIL", "detail": "registry missing"}]
    try:
        registry = json.loads(registry_path.read_text(encoding="utf-8")).get("agents", {})
    except json.JSONDecodeError:
        return [{"check": "deprecated_merged_into", "status": "FAIL", "detail": "registry malformed"}]

    findings: list[dict[str, Any]] = []
    for name, entry in sorted(registry.items()):
        if entry.get("status") != "deprecated":
            continue
        if entry.get("merged_into"):
            findings.append({"check": "deprecated_merged_into", "agent": name, "status": "PASS"})
        else:
            findings.append(
                {
                    "check": "deprecated_merged_into",
                    "agent": name,
                    "status": "FAIL",
                    "detail": "deprecated without merged_into",
                }
            )
    return findings


def check_watchdog_herald_recovery() -> list[dict[str, Any]]:
    path = REPO / "tools" / "cron_watchdog.sh"
    if not path.is_file():
        return [{"check": "watchdog_herald_recovery", "status": "FAIL", "detail": "watchdog missing"}]
    text = path.read_text(encoding="utf-8")
    findings: list[dict[str, Any]] = []
    for needle in WATCHDOG_HERALD_RECOVERY_STRINGS:
        if needle in text:
            findings.append({"check": "watchdog_herald_recovery", "needle": needle, "status": "PASS"})
        else:
            findings.append(
                {
                    "check": "watchdog_herald_recovery",
                    "needle": needle,
                    "status": "FAIL",
                    "detail": "not in cron_watchdog.sh",
                }
            )
    return findings


def check_live_crontab() -> dict[str, Any]:
    from tools.fleet_crontab_verify import verify_crontab

    report = verify_crontab()
    overall = report.get("overall", "SKIP")
    if overall == "SKIP":
        return {
            "check": "live_crontab",
            "status": "SKIP",
            "detail": report.get("availability", "unavailable"),
        }
    return {
        "check": "live_crontab",
        "status": overall,
        "pass_count": report.get("pass_count"),
        "fail_count": report.get("fail_count"),
    }


def build_audit() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    checks.extend(check_cron_llm_free())
    checks.extend(check_install_crontab())
    checks.extend(check_evening_catchup_builders())
    checks.extend(check_key_tools())
    checks.append(check_heartbeat_llm_gated())
    checks.append(check_supervisor_escalation_json())
    registry_checks, registry_summary = check_registry_heartbeat_coverage()
    checks.extend(registry_checks)
    checks.extend(check_deprecated_merged_into())
    checks.extend(check_watchdog_herald_recovery())
    checks.append(check_live_crontab())

    fail = sum(1 for c in checks if c.get("status") == "FAIL")
    pass_n = sum(1 for c in checks if c.get("status") == "PASS")
    overall = "PASS" if fail == 0 else "FAIL"

    return {
        "schema": "fleet_completion_audit.v1",
        "generated_at": datetime.now().isoformat(),
        "overall": overall,
        "pass_count": pass_n,
        "fail_count": fail,
        "registry_coverage": registry_summary,
        "checks": checks,
        "operator_commands": {
            "fleet_ops": "python3 tools/fleet_ops_status.py --write",
            "install_crontab": "bash tools/install_agent_fleet_crontab.sh",
            "host_checklist": "bash tools/run_fleet_operator_checklist.sh",
            "crontab_verify": "python3 tools/fleet_crontab_verify.py",
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
    ap.add_argument("--as-of-date", help="YYYY-MM-DD for artifact filename (default: today)")
    ap.add_argument("--json", action="store_true", help="Print JSON only")
    ap.add_argument("--write", action="store_true", help="Write artifacts/fleet_ops/{date}_completion_audit.json")
    args = ap.parse_args()

    ds = args.as_of_date or date.today().isoformat()
    report = build_audit()
    report["as_of_date"] = ds

    if args.write:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUT_DIR / f"{ds}_completion_audit.json"
        out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if not args.json:
            print(f"Wrote {out_path}")

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_human(report)
    return 1 if report["overall"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
