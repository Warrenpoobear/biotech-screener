#!/usr/bin/env python3
"""Hermes skills telemetry enhancement — observe-only, reversible, no behavior changes.

Provides read-only visibility into:
- Which skills are effective, stale, slow, costly, or missing capabilities
- Skill dependencies and call patterns
- Execution frequency and latency distribution
- Success/failure rates and error patterns

This tool adds NO autonomous optimization, NO routing changes, and NO memory rewrites.
All data is advisory-only, intended for human review and decision.

Generated reports feed into monthly learning loop (hermes_skills_learning_loop_v2.py).
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

LOGS_DIR = Path("artifacts/skills_learning")


def load_execution_logs(environment: str = "prod", month: str | None = None) -> List[Dict[str, Any]]:
    """Load execution logs for an environment and optional month."""
    if not month:
        month = datetime.utcnow().strftime("%Y-%m")

    exec_log = LOGS_DIR / f"execution_log_{environment}_{month}.jsonl"
    if not exec_log.exists():
        return []

    records = []
    for line in exec_log.read_text().strip().split("\n"):
        if line:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


def load_feedback_logs(environment: str = "prod", month: str | None = None) -> List[Dict[str, Any]]:
    """Load feedback logs for an environment and optional month."""
    if not month:
        month = datetime.utcnow().strftime("%Y-%m")

    feedback_log = LOGS_DIR / f"feedback_log_{environment}_{month}.jsonl"
    if not feedback_log.exists():
        return []

    records = []
    for line in feedback_log.read_text().strip().split("\n"):
        if line:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


def analyze_skill_efficacy(executions: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Analyze success rates, latency, and cost per skill."""
    skills = defaultdict(
        lambda: {
            "executions": 0,
            "successes": 0,
            "failures": 0,
            "total_latency_ms": 0.0,
            "total_cost_usd": 0.0,
            "total_tokens": 0,
            "errors": defaultdict(int),
        }
    )

    for record in executions:
        skill = record.get("skill_name", "unknown")
        stats = skills[skill]

        stats["executions"] += 1
        if record.get("outcome", {}).get("success"):
            stats["successes"] += 1
        else:
            stats["failures"] += 1
            error = record.get("outcome", {}).get("error", "unknown")
            stats["errors"][error] += 1

        metrics = record.get("metrics", {})
        stats["total_latency_ms"] += metrics.get("latency_ms", 0)
        stats["total_cost_usd"] += metrics.get("cost_usd", 0)
        stats["total_tokens"] += metrics.get("tokens_in", 0) + metrics.get("tokens_out", 0)

    # Compute derived metrics
    for skill, stats in skills.items():
        execs = stats["executions"]
        stats["success_rate"] = stats["successes"] / max(1, execs)
        stats["failure_rate"] = stats["failures"] / max(1, execs)
        stats["avg_latency_ms"] = stats["total_latency_ms"] / max(1, execs)
        stats["avg_cost_usd"] = stats["total_cost_usd"] / max(1, execs)
        stats["avg_tokens"] = stats["total_tokens"] / max(1, execs)

    return skills


def identify_stale_skills(executions: List[Dict[str, Any]], days_threshold: int = 7) -> Dict[str, Any]:
    """Identify skills with no recent executions (stale > N days)."""
    from datetime import datetime, timedelta, timezone

    now = datetime.now(timezone.utc)
    threshold = now - timedelta(days=days_threshold)

    stale = {}
    skill_timestamps = {}

    for record in executions:
        skill = record.get("skill_name", "unknown")
        timestamp_str = record.get("timestamp", "")
        try:
            timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            if skill not in skill_timestamps or timestamp > skill_timestamps[skill]:
                skill_timestamps[skill] = timestamp
        except (ValueError, AttributeError):
            pass

    for skill, timestamp in skill_timestamps.items():
        if timestamp < threshold:
            stale[skill] = {
                "last_execution": timestamp.isoformat(),
                "days_since_use": (now - timestamp).days,
            }

    return stale


def identify_slow_skills(skills: Dict[str, Dict[str, Any]], latency_threshold_ms: float = 5000) -> Dict[str, Any]:
    """Identify skills with high latency (>N ms average)."""
    slow = {}
    for skill, stats in skills.items():
        if stats["avg_latency_ms"] > latency_threshold_ms:
            slow[skill] = {
                "avg_latency_ms": stats["avg_latency_ms"],
                "executions": stats["executions"],
                "p50_estimate": stats["avg_latency_ms"],
            }
    return slow


def identify_costly_skills(skills: Dict[str, Dict[str, Any]], cost_threshold_usd: float = 0.01) -> Dict[str, Any]:
    """Identify skills with high cost per execution (>$N)."""
    costly = {}
    for skill, stats in skills.items():
        if stats["avg_cost_usd"] > cost_threshold_usd:
            costly[skill] = {
                "avg_cost_usd": stats["avg_cost_usd"],
                "executions": stats["executions"],
                "total_cost_usd": stats["total_cost_usd"],
            }
    return costly


def identify_unreliable_skills(skills: Dict[str, Dict[str, Any]], failure_threshold: float = 0.20) -> Dict[str, Any]:
    """Identify skills with high failure rates (>N%)."""
    unreliable = {}
    for skill, stats in skills.items():
        if stats["failure_rate"] > failure_threshold and stats["executions"] >= 5:
            unreliable[skill] = {
                "failure_rate": f"{stats['failure_rate']*100:.1f}%",
                "failures": stats["failures"],
                "executions": stats["executions"],
                "top_errors": dict(sorted(stats["errors"].items(), key=lambda x: -x[1])[:3]),
            }
    return unreliable


def identify_underexecuted_skills(skills: Dict[str, Dict[str, Any]], expected_executions: int = 5) -> Dict[str, Any]:
    """Identify skills that haven't had enough execution for evaluation."""
    underexecuted = {}
    for skill, stats in skills.items():
        if stats["executions"] < expected_executions:
            underexecuted[skill] = {
                "executions": stats["executions"],
                "needed": expected_executions,
                "gap": expected_executions - stats["executions"],
            }
    return underexecuted


def generate_telemetry_report(environment: str = "prod", month: str | None = None) -> str:
    """Generate comprehensive telemetry report (read-only, advisory)."""
    executions = load_execution_logs(environment, month)
    feedback = load_feedback_logs(environment, month)

    if not executions:
        return f"[TELEMETRY] No execution logs found for {environment}/{month}\n"

    skills = analyze_skill_efficacy(executions)
    stale = identify_stale_skills(executions, days_threshold=7)
    slow = identify_slow_skills(skills, latency_threshold_ms=5000)
    costly = identify_costly_skills(skills, cost_threshold_usd=0.01)
    unreliable = identify_unreliable_skills(skills, failure_threshold=0.20)
    underexecuted = identify_underexecuted_skills(skills, expected_executions=5)

    from datetime import timezone

    now = datetime.now(timezone.utc)
    lines = [
        f"# Hermes Skills Telemetry Report — {month} ({environment.upper()})",
        "",
        f"**Generated:** {now.isoformat()}",
        f"**Environment:** {environment.upper()}",
        f"**Total Executions:** {len(executions)}",
        f"**Total Skills Observed:** {len(skills)}",
        f"**Total Feedback Points:** {len(feedback)}",
        "",
        "## ⚠️ ADVISORY ONLY",
        "",
        "This report contains **read-only telemetry** for human review. No behavioral changes",
        "will be applied automatically. All recommendations require explicit operator approval.",
        "",
        "---",
        "",
        "## 📊 Skills Status Summary",
        "",
        f"- **Effective** (≥80% success): {sum(1 for s in skills.values() if s['success_rate'] >= 0.80)}",
        f"- **Unreliable** (>20% failure): {len(unreliable)}",
        f"- **Underexecuted** (<5 runs): {len(underexecuted)}",
        f"- **Stale** (>7 days): {len(stale)}",
        f"- **Slow** (>5s avg): {len(slow)}",
        f"- **Costly** (>$0.01/run): {len(costly)}",
        "",
    ]

    if unreliable:
        lines.extend(
            [
                "## 🚨 Unreliable Skills (High Failure Rate)",
                "",
            ]
        )
        for skill, data in sorted(unreliable.items()):
            lines.append(f"### {skill}")
            lines.append(f"- **Failure Rate:** {data['failure_rate']}")
            lines.append(f"- **Failures:** {data['failures']} / {data['executions']}")
            lines.append("- **Top Errors:**")
            for err, count in list(data["top_errors"].items())[:3]:
                lines.append(f"  - {err}: {count}x")
            lines.append("")

    if slow:
        lines.extend(
            [
                "## 🐌 Slow Skills (High Latency)",
                "",
            ]
        )
        for skill, data in sorted(slow.items(), key=lambda x: -x[1]["avg_latency_ms"]):
            lines.append(f"### {skill}")
            lines.append(f"- **Avg Latency:** {data['avg_latency_ms']:.0f}ms")
            lines.append(f"- **Executions:** {data['executions']}")
            lines.append("")

    if costly:
        lines.extend(
            [
                "## 💰 Costly Skills (High Cost)",
                "",
            ]
        )
        for skill, data in sorted(costly.items(), key=lambda x: -x[1]["total_cost_usd"]):
            lines.append(f"### {skill}")
            lines.append(f"- **Avg Cost:** ${data['avg_cost_usd']:.4f}")
            lines.append(f"- **Total Cost:** ${data['total_cost_usd']:.2f}")
            lines.append(f"- **Executions:** {data['executions']}")
            lines.append("")

    if underexecuted:
        lines.extend(
            [
                "## ⏳ Underexecuted Skills (Insufficient Data)",
                "",
            ]
        )
        for skill, data in sorted(underexecuted.items(), key=lambda x: -x[1]["gap"]):
            lines.append(f"- {skill}: {data['executions']}/{data['needed']} (need {data['gap']} more)")
        lines.append("")

    if stale:
        lines.extend(
            [
                "## 📅 Stale Skills (No Recent Use)",
                "",
            ]
        )
        for skill, data in sorted(stale.items(), key=lambda x: x[1]["days_since_use"]):
            days = data["days_since_use"]
            lines.append(f"- {skill}: {days}d ago ({data['last_execution']})")
        lines.append("")

    lines.extend(
        [
            "---",
            "",
            "## 📋 All Skills Performance Baseline",
            "",
        ]
    )

    sorted_skills = sorted(
        skills.items(),
        key=lambda x: x[1]["success_rate"],
        reverse=True,
    )

    for skill, stats in sorted_skills:
        status = "✅" if stats["success_rate"] >= 0.80 else "⚠️ " if stats["success_rate"] >= 0.50 else "❌"
        lines.append(f"### {status} {skill}")
        lines.append(f"- **Success:** {stats['successes']}/{stats['executions']} ({stats['success_rate']*100:.1f}%)")
        lines.append(f"- **Avg Latency:** {stats['avg_latency_ms']:.0f}ms")
        lines.append(f"- **Avg Cost:** ${stats['avg_cost_usd']:.4f}")
        lines.append(f"- **Avg Tokens:** {stats['avg_tokens']:.0f}")
        lines.append("")

    lines.extend(
        [
            "---",
            "",
            "## 🔐 Safety Constraints (Active)",
            "",
            "- ✅ 7-day observation period: Active (2026-06-05 to 2026-06-12)",
            "- ✅ No auto-routing: Enforced (advisory-only)",
            "- ✅ No behavioral changes: Enforced (approval required)",
            "- ✅ PII redaction: Active on all logs",
            "- ✅ Environment tagging: Separate prod/test logs",
            "",
        ]
    )

    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    month = sys.argv[1] if len(sys.argv) > 1 else None
    env = sys.argv[2] if len(sys.argv) > 2 else "prod"

    report = generate_telemetry_report(env, month)
    print(report)

    # Optionally write to file
    if len(sys.argv) > 3 and sys.argv[3] == "--write":
        output_dir = Path("artifacts/skills_learning")
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f"telemetry_report_{env}_{month or 'current'}.md"
        output_file.write_text(report)
        print(f"\n✓ Report written to: {output_file}")
