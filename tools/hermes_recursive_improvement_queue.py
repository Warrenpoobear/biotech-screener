#!/usr/bin/env python3
"""Hermes recursive self-improvement queue builder.

Read-only, advisory-only bridge from Hermes skills telemetry into a governed
operator review queue. This tool never edits skills, changes routing, or marks
automation as human approval.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOGS_DIR = REPO_ROOT / "artifacts" / "skills_learning"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "skills_learning"

MIN_EXECUTIONS_FOR_ACTION = 5
MIN_FEEDBACK_FOR_ACTION = 3
FAILURE_RATE_THRESHOLD = 0.20
LATENCY_THRESHOLD_MS = 5000.0
COST_THRESHOLD_USD = 0.01

QUEUE_SCHEMA_VERSION = "1.0"

CLASSIFICATIONS = {
    "COLLECT_FEEDBACK": "Gather more operator feedback before changing skill behavior.",
    "INVESTIGATE_FAILURES": "Review repeated failures; do not auto-disable or reroute.",
    "REVIEW_LATENCY": "Inspect slow execution path for prompt/tooling improvements.",
    "REVIEW_COST": "Inspect high-cost usage for cheaper equivalent workflow.",
    "PROMOTION_CANDIDATE": "Candidate for human-reviewed skill/docs improvement.",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSONL records, skipping malformed lines."""
    if not path.exists():
        return []

    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def _execution_log_path(logs_dir: Path, environment: str, month: str) -> Path:
    return logs_dir / f"execution_log_{environment}_{month}.jsonl"


def _feedback_log_path(logs_dir: Path, environment: str, month: str) -> Path:
    return logs_dir / f"feedback_log_{environment}_{month}.jsonl"


def _safe_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _skill_stats(executions: list[dict[str, Any]], feedback: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    exec_id_to_skill: dict[str, str] = {}
    stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "executions": 0,
            "successes": 0,
            "failures": 0,
            "feedback_count": 0,
            "helpful_feedback": 0,
            "unhelpful_feedback": 0,
            "missing_feedback": 0,
            "total_latency_ms": 0.0,
            "total_cost_usd": 0.0,
            "top_errors": defaultdict(int),
        }
    )

    for record in executions:
        skill = str(record.get("skill_name") or "unknown")
        exec_id = str(record.get("execution_id") or "")
        if exec_id:
            exec_id_to_skill[exec_id] = skill

        item = stats[skill]
        item["executions"] += 1
        outcome = record.get("outcome") or {}
        success = bool(outcome.get("success"))
        if success:
            item["successes"] += 1
        else:
            item["failures"] += 1
            error = str(outcome.get("error") or "unknown")
            item["top_errors"][error] += 1

        metrics = record.get("metrics") or {}
        item["total_latency_ms"] += _safe_float(metrics.get("latency_ms"))
        item["total_cost_usd"] += _safe_float(metrics.get("cost_usd"))

    for record in feedback:
        exec_id = str(record.get("execution_id") or "")
        skill = exec_id_to_skill.get(exec_id, str(record.get("skill_name") or "unknown"))
        item = stats[skill]
        verdict = str(record.get("verdict") or "").lower()
        item["feedback_count"] += 1
        if verdict == "helpful":
            item["helpful_feedback"] += 1
        elif verdict == "unhelpful":
            item["unhelpful_feedback"] += 1
        elif verdict == "missing":
            item["missing_feedback"] += 1

    normalized: dict[str, dict[str, Any]] = {}
    for skill, item in sorted(stats.items()):
        executions_count = int(item["executions"])
        failures = int(item["failures"])
        avg_latency_ms = item["total_latency_ms"] / max(1, executions_count)
        avg_cost_usd = item["total_cost_usd"] / max(1, executions_count)
        normalized[skill] = {
            "executions": executions_count,
            "successes": int(item["successes"]),
            "failures": failures,
            "failure_rate": round(failures / max(1, executions_count), 4),
            "feedback_count": int(item["feedback_count"]),
            "helpful_feedback": int(item["helpful_feedback"]),
            "unhelpful_feedback": int(item["unhelpful_feedback"]),
            "missing_feedback": int(item["missing_feedback"]),
            "avg_latency_ms": round(avg_latency_ms, 2),
            "avg_cost_usd": round(avg_cost_usd, 6),
            "top_errors": dict(sorted(item["top_errors"].items(), key=lambda kv: (-kv[1], kv[0]))[:3]),
        }
    return normalized


def _queue_entry(skill: str, classification: str, priority: str, evidence: list[str]) -> dict[str, Any]:
    return {
        "skill_name": skill,
        "classification": classification,
        "priority": priority,
        "recommendation": CLASSIFICATIONS[classification],
        "evidence": evidence,
        "allowed_action": "operator_review_only",
        "requires_human_review": True,
        "automation_approval": False,
        "production_deployment_approved": False,
    }


def build_queue(
    *,
    environment: str,
    month: str,
    as_of_date: str,
    logs_dir: Path = DEFAULT_LOGS_DIR,
) -> dict[str, Any]:
    """Build a deterministic advisory queue from Hermes skill logs."""
    executions = load_jsonl(_execution_log_path(logs_dir, environment, month))
    feedback = load_jsonl(_feedback_log_path(logs_dir, environment, month))
    stats = _skill_stats(executions, feedback)

    entries: list[dict[str, Any]] = []
    for skill, item in stats.items():
        if item["executions"] < MIN_EXECUTIONS_FOR_ACTION:
            entries.append(
                _queue_entry(
                    skill,
                    "COLLECT_FEEDBACK",
                    "LOW",
                    [f"executions={item['executions']}/{MIN_EXECUTIONS_FOR_ACTION}"],
                )
            )
            continue

        if item["feedback_count"] < MIN_FEEDBACK_FOR_ACTION:
            entries.append(
                _queue_entry(
                    skill,
                    "COLLECT_FEEDBACK",
                    "MEDIUM",
                    [f"feedback={item['feedback_count']}/{MIN_FEEDBACK_FOR_ACTION}"],
                )
            )

        if item["failure_rate"] > FAILURE_RATE_THRESHOLD:
            entries.append(
                _queue_entry(
                    skill,
                    "INVESTIGATE_FAILURES",
                    "HIGH",
                    [
                        f"failure_rate={item['failure_rate']}",
                        f"failures={item['failures']}/{item['executions']}",
                    ],
                )
            )

        if item["avg_latency_ms"] > LATENCY_THRESHOLD_MS:
            entries.append(
                _queue_entry(
                    skill,
                    "REVIEW_LATENCY",
                    "MEDIUM",
                    [f"avg_latency_ms={item['avg_latency_ms']}"],
                )
            )

        if item["avg_cost_usd"] > COST_THRESHOLD_USD:
            entries.append(
                _queue_entry(
                    skill,
                    "REVIEW_COST",
                    "MEDIUM",
                    [f"avg_cost_usd={item['avg_cost_usd']}"],
                )
            )

        if (
            item["failure_rate"] == 0
            and item["feedback_count"] >= MIN_FEEDBACK_FOR_ACTION
            and item["unhelpful_feedback"] == 0
            and item["missing_feedback"] == 0
        ):
            entries.append(
                _queue_entry(
                    skill,
                    "PROMOTION_CANDIDATE",
                    "LOW",
                    [
                        f"executions={item['executions']}",
                        f"helpful_feedback={item['helpful_feedback']}/{item['feedback_count']}",
                    ],
                )
            )

    priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    entries = sorted(
        entries,
        key=lambda entry: (
            priority_order.get(str(entry["priority"]), 9),
            str(entry["skill_name"]),
            str(entry["classification"]),
        ),
    )

    return {
        "artifact_type": "hermes_recursive_self_improvement_queue",
        "schema_version": QUEUE_SCHEMA_VERSION,
        "as_of_date": as_of_date,
        "generated_at": f"{as_of_date}T00:00:00Z",
        "environment": environment,
        "month": month,
        "source_logs": {
            "execution_log": str(_execution_log_path(logs_dir, environment, month)),
            "feedback_log": str(_feedback_log_path(logs_dir, environment, month)),
        },
        "summary": {
            "execution_count": len(executions),
            "feedback_count": len(feedback),
            "skill_count": len(stats),
            "queue_count": len(entries),
            "high_priority_count": sum(1 for entry in entries if entry["priority"] == "HIGH"),
        },
        "skills": stats,
        "queue": entries,
        "governance": {
            "read_only_diagnostic": True,
            "recursive_self_improvement": True,
            "advisory_only": True,
            "automation_approval": False,
            "production_deployment_approved": False,
            "skill_file_mutation": False,
            "routing_change": False,
        },
        "next_steps": [
            "Review HIGH priority queue entries first.",
            "Collect missing feedback before changing skill docs or routing.",
            "Apply any skill/doc patch in a separate human-reviewed change.",
            "Run tools/audit_hermes_skills.py after skill mirror changes.",
        ],
    }


def write_queue_artifacts(queue: dict[str, Any], output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Path]:
    """Write queue JSON and Markdown artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    env = str(queue["environment"])
    month = str(queue["month"])
    json_path = output_dir / f"recursive_improvement_queue_{env}_{month}.json"
    md_path = output_dir / f"recursive_improvement_queue_{env}_{month}.md"

    json_path.write_text(json.dumps(queue, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(queue), encoding="utf-8", newline="\n")
    return {"json": json_path, "markdown": md_path}


def render_markdown(queue: dict[str, Any]) -> str:
    """Render the queue as operator-readable Markdown."""
    summary = queue["summary"]
    lines = [
        f"# Hermes Recursive Self-Improvement Queue - {queue['month']} ({str(queue['environment']).upper()})",
        "",
        f"Generated: {queue['generated_at']}",
        "",
        "## Governance",
        "",
        "- Read-only diagnostic: True",
        "- Advisory only: True",
        "- Automation approval: False",
        "- Production deployment approved: False",
        "- Skill file mutation: False",
        "- Routing change: False",
        "",
        "## Summary",
        "",
        f"- Executions: {summary['execution_count']}",
        f"- Feedback records: {summary['feedback_count']}",
        f"- Skills observed: {summary['skill_count']}",
        f"- Queue entries: {summary['queue_count']}",
        f"- High priority entries: {summary['high_priority_count']}",
        "",
        "## Queue",
        "",
    ]

    if not queue["queue"]:
        lines.append("(No queue entries.)")
        lines.append("")
    else:
        for entry in queue["queue"]:
            lines.append(f"### {entry['priority']} - {entry['skill_name']} - {entry['classification']}")
            lines.append(f"- Recommendation: {entry['recommendation']}")
            lines.append(f"- Allowed action: {entry['allowed_action']}")
            lines.append("- Evidence:")
            for evidence in entry["evidence"]:
                lines.append(f"  - {evidence}")
            lines.append("")

    lines.extend(["## Next Steps", ""])
    for step in queue["next_steps"]:
        lines.append(f"- {step}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Hermes recursive self-improvement queue")
    parser.add_argument("--environment", default="prod", choices=["prod", "test"])
    parser.add_argument("--month", required=True, help="Log month in YYYY-MM format")
    parser.add_argument("--as-of-date", required=True, help="Deterministic artifact date YYYY-MM-DD")
    parser.add_argument("--logs-dir", type=Path, default=DEFAULT_LOGS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    queue = build_queue(
        environment=args.environment,
        month=args.month,
        as_of_date=args.as_of_date,
        logs_dir=args.logs_dir,
    )
    paths = write_queue_artifacts(queue, args.output_dir)
    print(f"Wrote {paths['json']}")
    print(f"Wrote {paths['markdown']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
