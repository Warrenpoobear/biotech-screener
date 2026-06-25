#!/usr/bin/env python3
"""One-shot operator status for agent fleet health.

Read-only. Summarizes herald pipeline, heartbeat receipt, stalled loops,
and crontab install hints for WSL host triage.

Usage:
    python3 tools/fleet_ops_status.py
    python3 tools/fleet_ops_status.py --as-of-date 2026-06-24
    python3 tools/fleet_ops_status.py --json
    python3 tools/fleet_ops_status.py --write
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "artifacts" / "fleet_ops"

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

STALLED_LOOP_ACTIONS: dict[str, dict[str, str]] = {
    "F-2026-005": {
        "area": "Herald Digest",
        "symptom": "Herald classified JSONL stale or dark pipeline on host",
        "action": "bash tools/herald_recovery.sh && python3 tools/herald_health_check.py --recover",
    },
    "F-2026-006": {
        "area": "GitHub CI",
        "symptom": "Actions budget exhausted; merge gates unverified",
        "action": "Restore GitHub Actions budget; confirm tests workflow green on main",
    },
}

def _stalled_loops_status() -> list[dict[str, str]]:
    from tools.skills_loop_review import stalled_loop_entries

    rows: list[dict[str, str]] = []
    for entry in stalled_loop_entries():
        fid = entry.get("id", "")
        hints = STALLED_LOOP_ACTIONS.get(fid, {})
        rows.append(
            {
                "id": fid,
                "area": hints.get("area") or entry.get("system", ""),
                "status": entry.get("status", "UNKNOWN"),
                "symptom": hints.get("symptom", ""),
                "action": hints.get("action", "See .learnings/memory.md stalled-loop table"),
            }
        )
    if rows:
        return rows
    # Fallback when memory.md table unavailable (e.g. cloud clone)
    return [
        {"id": fid, "status": "OPEN", **meta}
        for fid, meta in STALLED_LOOP_ACTIONS.items()
    ]


def _selfimprove_gates() -> dict[str, Any]:
    from tools.skills_loop_review import selfimprove_gates_status

    return selfimprove_gates_status()


CRONTAB_HINTS: list[dict[str, str]] = [
    {"job": "herald_health", "schedule": "40 14 * * 1-5", "log": "logs/herald_health.log"},
    {"job": "agent_heartbeat_checks", "schedule": "30 17 * * 1-5", "log": "logs/heartbeat_checks.log"},
    {"job": "cron_data_auditor", "schedule": "30 17 * * 1-5", "log": "logs/data_auditor.log"},
    {"job": "cron_evening_catchup", "schedule": "0 22 * * 1-5", "log": "logs/evening_catchup.log"},
    {"job": "cron_watchdog", "schedule": "@reboot + weekday safety net", "log": "logs/watchdog.log"},
]

VERDICT_RE = re.compile(r"Verdict:\s*(\w+)")


def _log_has_date(log_path: Path, ds: str) -> bool:
    if not log_path.is_file():
        return False
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return ds in text


def _heartbeat_status(ds: str) -> dict[str, Any]:
    heartbeat_dir = REPO / "artifacts" / "heartbeat"
    receipt = heartbeat_dir / f"{ds}_receipt.md"
    escalation = heartbeat_dir / f"{ds}_escalation.json"
    anomalies = heartbeat_dir / f"{ds}_anomalies.md"
    log_ok = _log_has_date(REPO / "logs" / "heartbeat_checks.log", ds)

    verdict: str | None = None
    if receipt.is_file():
        match = VERDICT_RE.search(receipt.read_text(encoding="utf-8", errors="replace"))
        if match:
            verdict = match.group(1)

    escalation_mode: str | None = None
    if escalation.is_file():
        try:
            payload = json.loads(escalation.read_text(encoding="utf-8"))
            escalation_mode = payload.get("mode")
        except (json.JSONDecodeError, OSError):
            escalation_mode = "corrupt"

    return {
        "receipt_exists": receipt.is_file(),
        "verdict": verdict,
        "anomalies_md_exists": anomalies.is_file(),
        "escalation_json_exists": escalation.is_file(),
        "escalation_mode": escalation_mode,
        "log_has_today": log_ok,
    }


def _snapshot_status(ds: str) -> dict[str, Any]:
    snap = REPO / "data" / "snapshots" / ds / "rankings.csv"
    return {"exists": snap.is_file(), "path": str(snap.relative_to(REPO))}


def _fleet_jobs_status(ds: str) -> dict[str, bool]:
    return {
        "herald_health_log": _log_has_date(REPO / "logs" / "herald_health.log", ds),
        "evening_catchup_log": _log_has_date(REPO / "logs" / "evening_catchup.log", ds),
        "watchdog_log": _log_has_date(REPO / "logs" / "watchdog.log", ds),
        "data_auditor_log": _log_has_date(REPO / "logs" / "data_auditor.log", ds),
    }


def _completion_audit_status(ds: str) -> dict[str, Any]:
    path = REPO / "artifacts" / "fleet_ops" / f"{ds}_completion_audit.json"
    if not path.is_file():
        return {"exists": False, "overall": None, "fail_count": None}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"exists": True, "overall": "corrupt", "fail_count": None}
    return {
        "exists": True,
        "overall": payload.get("overall"),
        "pass_count": payload.get("pass_count"),
        "fail_count": payload.get("fail_count"),
        "registry_coverage": payload.get("registry_coverage"),
    }


def build_status(as_of: date | None = None) -> dict[str, Any]:
    as_of = as_of or date.today()
    ds = as_of.isoformat()

    from tools.herald_health_check import run_check

    herald = run_check(as_of)

    heartbeat = _heartbeat_status(ds)
    snapshot = _snapshot_status(ds)
    jobs = _fleet_jobs_status(ds)

    stalled = _stalled_loops_status()
    gates = _selfimprove_gates()
    blockers = [loop["id"] for loop in stalled if loop.get("status", "").upper() == "OPEN"]
    overall = "HEALTHY"
    if herald["verdict"] == "FAIL" or heartbeat.get("verdict") == "RED":
        overall = "FAIL"
    elif herald["verdict"] in ("WARN",) or heartbeat.get("verdict") in ("YELLOW", "RED"):
        overall = "WARN"
    elif blockers or not gates.get("selfimprove_gates_met_allowed", True):
        overall = "WARN"

    return {
        "schema": "fleet_ops_status.v1",
        "as_of_date": ds,
        "generated_at": datetime.now().isoformat(),
        "overall": overall,
        "herald": {
            "verdict": herald["verdict"],
            "herald_done": herald["herald_done"],
            "latest_classified_date": herald.get("latest_classified_date"),
            "source_age_days": herald.get("source_age_days"),
            "issues": herald.get("issues", []),
        },
        "heartbeat": heartbeat,
        "snapshot": snapshot,
        "fleet_jobs": jobs,
        "completion_audit": _completion_audit_status(ds),
        "stalled_loops": stalled,
        "selfimprove_gates": gates,
        "crontab_install": "bash tools/install_agent_fleet_crontab.sh",
        "crontab_hints": CRONTAB_HINTS,
        "rule_12_checklist": "docs/governance/RULE_12_PROMOTION_CHECKLIST.md",
        "fleet_completion_audit": "python3 tools/fleet_completion_audit.py",
    }


def _print_human(report: dict[str, Any]) -> None:
    ds = report["as_of_date"]
    print("=" * 72)
    print(f"FLEET OPS STATUS — {ds} — overall={report['overall']}")
    print("=" * 72)

    herald = report["herald"]
    print("\nHerald")
    print(f"  verdict: {herald['verdict']}  done={herald['herald_done']}")
    if herald.get("latest_classified_date"):
        print(
            f"  latest classified: {herald['latest_classified_date']} "
            f"({herald.get('source_age_days')}d ago)"
        )
    for issue in herald.get("issues", [])[:5]:
        print(f"  - {issue}")

    hb = report["heartbeat"]
    print("\nHeartbeat")
    print(f"  receipt: {'yes' if hb['receipt_exists'] else 'no'}  verdict={hb.get('verdict')}")
    print(f"  escalation: {hb.get('escalation_mode') or 'none'}")
    print(f"  log today: {'yes' if hb['log_has_today'] else 'no'}")

    snap = report["snapshot"]
    print("\nProduction snapshot")
    print(f"  rankings.csv: {'yes' if snap['exists'] else 'no'}")

    print("\nFleet job logs (date present)")
    for name, ok in report["fleet_jobs"].items():
        print(f"  {name}: {'yes' if ok else 'no'}")

    audit = report.get("completion_audit") or {}
    print("\nCompletion audit")
    if audit.get("exists"):
        print(f"  overall={audit.get('overall')}  fail_count={audit.get('fail_count')}")
        reg = audit.get("registry_coverage") or {}
        if reg:
            print(
                f"  registry: supervised={reg.get('active_supervised')} "
                f"specialized={reg.get('specialized')} generic={reg.get('generic_fallback')}"
            )
    else:
        print("  no artifact — run: python3 tools/fleet_completion_audit.py --write")

    print("\nStalled loops (operator confirm before close)")
    for loop in report["stalled_loops"]:
        print(f"  {loop['id']} [{loop['status']}] {loop['area']}: {loop['action']}")

    print("\nCrontab")
    print(f"  install reference: {report['crontab_install']}")
    for hint in report["crontab_hints"]:
        print(f"  {hint['job']}: {hint['schedule']} -> {hint['log']}")

    print("\nSelf-improve gates")
    gates = report.get("selfimprove_gates") or {}
    print(f"  {gates.get('message', 'n/a')}")

    print("\nRule 12 gate: close F-2026-005/006 before SELFIMPROVE_GATES_MET=1")
    print(f"  checklist: {report['rule_12_checklist']}")
    print("=" * 72)


def main() -> int:
    ap = argparse.ArgumentParser(description="Agent fleet operator status (read-only)")
    ap.add_argument("--as-of-date", help="YYYY-MM-DD (default: today)")
    ap.add_argument("--json", action="store_true", help="Print JSON only")
    ap.add_argument("--write", action="store_true", help="Write artifacts/fleet_ops/{date}_status.json")
    ap.add_argument("--no-telemetry", action="store_true", help="Skip agent_skill_telemetry logging")
    args = ap.parse_args()

    as_of = date.fromisoformat(args.as_of_date) if args.as_of_date else None
    started = time.perf_counter()
    report = build_status(as_of)

    if args.write:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUT_DIR / f"{report['as_of_date']}_status.json"
        out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if not args.json:
            print(f"Wrote {out_path}")

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_human(report)

    if not args.no_telemetry:
        try:
            from tools.agent_skill_telemetry import log_agent_run
            from tools.record_skill_feedback import attach_outcome_verdict

            overall = report["overall"]
            exec_id = log_agent_run(
                "fleet_ops_status",
                f"Fleet ops status for {report['as_of_date']}",
                inputs={"as_of_date": report["as_of_date"]},
                outputs={"overall": overall, "herald_verdict": report["herald"]["verdict"]},
                success=overall != "FAIL",
                error=None if overall != "FAIL" else overall,
                latency_ms=(time.perf_counter() - started) * 1000,
            )
            if exec_id:
                attach_outcome_verdict(
                    exec_id,
                    was_correct=overall == "HEALTHY",
                    evidence=f"overall={overall} herald={report['herald']['verdict']}",
                )
        except Exception:
            pass

    if report["overall"] == "FAIL":
        return 2
    if report["overall"] == "WARN":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
