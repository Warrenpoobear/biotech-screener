#!/usr/bin/env python3
"""Generate weekly/monthly self-improvement digest for operator review.

Combines learnings audit, trim candidates, efficacy overdue, telemetry, and
skill-patch drafts into one operator-facing artifact. Advisory only.

Usage:
    python3 tools/weekly_skills_digest.py
    python3 tools/weekly_skills_digest.py --date 2026-06-24
    python3 tools/weekly_skills_digest.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.audit_learnings import build_report
from tools.skills_loop_review import (
    format_loop_review_sections,
    stalled_loop_entries,
    trim_candidates,
)

LOGS_DIR = REPO / "artifacts" / "skills_learning"
DRAFTS_DIR = REPO / "artifacts" / "skill_patch_drafts"
OUT_DIR = LOGS_DIR

MIN_EXECUTIONS_FOR_EVAL = 5
MIN_SUCCESS_RATE_WARN = 0.80


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").strip().split("\n"):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def _telemetry_summary(month_str: str, environment: str = "prod") -> dict[str, dict]:
    log_path = LOGS_DIR / f"execution_log_{environment}_{month_str}.jsonl"
    legacy_path = LOGS_DIR / f"execution_log_{month_str}.jsonl"
    executions = _load_jsonl(log_path) or _load_jsonl(legacy_path)

    stats: dict[str, dict] = defaultdict(
        lambda: {"executions": 0, "successes": 0, "failures": 0, "total_latency": 0.0}
    )
    for row in executions:
        skill = row.get("skill_name", "unknown")
        s = stats[skill]
        s["executions"] += 1
        if row.get("outcome", {}).get("success"):
            s["successes"] += 1
        else:
            s["failures"] += 1
        s["total_latency"] += row.get("metrics", {}).get("latency_ms", 0)
    return dict(stats)


def _list_draft_files() -> list[Path]:
    if not DRAFTS_DIR.exists():
        return []
    return sorted(DRAFTS_DIR.glob("skill_patch_drafts_*.md"), reverse=True)


def _registry_coverage_lines(as_of: date) -> list[str]:
    """Registry heartbeat coverage from completion audit artifact."""
    audit_path = REPO / "artifacts" / "fleet_ops" / f"{as_of.isoformat()}_completion_audit.json"
    if not audit_path.is_file():
        return []
    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return [f"- Registry coverage: artifact corrupt (`{audit_path.relative_to(REPO)}`)"]
    reg = audit.get("registry_coverage") or {}
    if not reg:
        return []
    parts = [
        f"active_supervised={reg.get('active_supervised', '?')}",
        f"specialized={reg.get('specialized', '?')}",
        f"generic_fallback={reg.get('generic_fallback', '?')}",
        f"on_demand_skip={reg.get('on_demand_skip', '?')}",
    ]
    overall = audit.get("overall", "?")
    return [f"- Completion audit: **{overall}** — registry ({', '.join(parts)})"]


def _fleet_ops_section(as_of: date) -> list[str]:
    """Include fleet_ops artifact when present (written by evening/weekly cron)."""
    path = REPO / "artifacts" / "fleet_ops" / f"{as_of.isoformat()}_status.json"
    lines = ["## Fleet ops status", ""]
    if not path.is_file():
        lines.append("No artifact. Run: `python3 tools/fleet_completion_audit.py --write` then `fleet_ops_status.py --write`")
        lines.append("")
        lines.extend(_registry_coverage_lines(as_of))
        if lines[-1] != "":
            lines.append("")
        return lines
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        lines.append(f"Artifact corrupt: `{path.relative_to(REPO)}`")
        lines.append("")
        return lines
    lines.append(f"- Overall: **{report.get('overall', '?')}**")
    herald = report.get("herald") or {}
    lines.append(f"- Herald: {herald.get('verdict', '?')} (done={herald.get('herald_done')})")
    hb = report.get("heartbeat") or {}
    lines.append(
        f"- Heartbeat receipt: {'yes' if hb.get('receipt_exists') else 'no'} "
        f"verdict={hb.get('verdict')} escalation={hb.get('escalation_mode') or 'none'}"
    )
    gates = report.get("selfimprove_gates") or {}
    if gates.get("message"):
        lines.append(f"- Rule 12: {gates['message']}")
    audit = report.get("completion_audit") or {}
    if audit.get("exists") and audit.get("overall"):
        lines.append(
            f"- Completion audit (embedded): **{audit.get('overall')}** "
            f"pass={audit.get('pass_count')} fail={audit.get('fail_count')}"
        )
    lines.extend(_registry_coverage_lines(as_of))
    lines.append("")
    return lines


def generate_digest(as_of: date | None = None, *, dry_run: bool = False) -> Path:
    as_of = as_of or date.today()
    month_str = as_of.strftime("%Y-%m")
    stamp = as_of.isoformat()
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    report = build_report()
    telemetry = _telemetry_summary(month_str)
    drafts = _list_draft_files()
    trim = trim_candidates(as_of=as_of)
    stalled = stalled_loop_entries()

    lines = [
        f"# Skills Loop Review Digest — {stamp}",
        "",
        f"Generated: {generated}",
        "",
        "**Advisory only.** Review and act manually. No auto-routing, auto-merge, or auto-delete.",
        "",
        "## Summary",
        "",
        f"- LRN entries: {report.lrn_total}",
        f"- Promotion candidates (pending, rec ≥3): {len(report.promotion_candidates)}",
        f"- Skill-patch candidates (Skill-Path, rec ≥2, skill lane): {len(report.skill_candidates)}",
        f"- Spec-lane blocked: {len(report.spec_lane_blocked)}",
        f"- Trim candidates (0 loads / 30d): {len(trim)}",
        f"- Stalled-loop OPEN: {sum(1 for s in stalled if s.get('status', '').upper() == 'OPEN')}",
        f"- Skills with telemetry this month: {len(telemetry)}",
        f"- Open skill-patch drafts: {len(drafts)}",
        "",
    ]

    lines.extend(_fleet_ops_section(as_of))
    lines.extend(format_loop_review_sections(as_of=as_of))

    lines.extend(["## Promotion candidates", ""])
    if not report.promotion_candidates:
        lines.append("None.")
    else:
        for c in report.promotion_candidates:
            lines.append(
                f"- `{c['pattern_key']}` (rec={c['total_recurrence']}): "
                f"{', '.join(c['pending_lrns'])} → {c['action']}"
            )
    lines.append("")

    lines.extend(["## Skill-patch candidates", ""])
    if not report.skill_candidates:
        lines.append("None.")
    else:
        for c in report.skill_candidates[:20]:
            lane = c.get("promotion_lane", "skill")
            lines.append(f"- {c['lrn_id']} → `{c['skill_path']}` (lane={lane}, status={c['status']})")
    lines.append("")

    lines.extend(["## Spec-lane blocked", ""])
    if not report.spec_lane_blocked:
        lines.append("None.")
    else:
        for c in report.spec_lane_blocked:
            lines.append(f"- {c['lrn_id']}: `{c['pattern_key']}` — route to Spec, not skill patch")
    lines.append("")

    lines.extend(["## Telemetry (current month, prod)", ""])
    if not telemetry:
        lines.append("No execution logs yet. Wire `log_skill()` from agent sessions.")
    else:
        for skill, s in sorted(telemetry.items(), key=lambda x: x[1]["executions"], reverse=True)[:15]:
            execs = s["executions"]
            rate = s["successes"] / max(1, execs)
            avg_lat = s["total_latency"] / max(1, execs)
            flag = ""
            if execs >= MIN_EXECUTIONS_FOR_EVAL and rate < MIN_SUCCESS_RATE_WARN:
                flag = " ⚠️ low success rate"
            lines.append(f"- **{skill}**: {execs} execs, {rate:.0%} success, {avg_lat:.0f}ms avg{flag}")
    lines.append("")

    lines.extend(["## Open skill-patch drafts", ""])
    if not drafts:
        lines.append("None. Run: `SELFIMPROVE_GATES_MET=1 python3 tools/pattern_to_skillpatch.py`")
    else:
        for p in drafts[:5]:
            lines.append(f"- `{p.relative_to(REPO)}`")
    lines.append("")

    lines.extend(
        [
            "## Operator checklist",
            "",
            "1. `python3 tools/audit_learnings.py`",
            "2. Review trim list — demote/archive unused skills (30d rule)",
            "3. Review contradiction / spec-lane blocks before any skill merge",
            "4. `SELFIMPROVE_GATES_MET=1 python3 tools/pattern_to_skillpatch.py` → review drafts",
            "5. Close stalled-loop verdicts (F-2026-005/006) before efficacy checks",
            "6. Append harvest_log verification 14d post-merge (Rule 12)",
            "",
            "## Commands",
            "",
            "```bash",
            "python3 tools/weekly_skills_digest.py",
            "python3 tools/skills_telemetry_monthly_report.py",
            "```",
            "",
        ]
    )

    out_path = OUT_DIR / f"loop_review_digest_{stamp}.md"
    text = "\n".join(lines)
    if dry_run:
        print(text)
        return out_path
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    print(f"Wrote {out_path}")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Skills loop review digest")
    parser.add_argument("--date", help="YYYY-MM-DD (default: today)")
    parser.add_argument("--dry-run", action="store_true", help="Print to stdout only")
    args = parser.parse_args()
    as_of = date.fromisoformat(args.date) if args.date else None
    generate_digest(as_of, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
