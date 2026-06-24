#!/usr/bin/env python3
"""Hermes skills learning loop v2: safe, conservative recommendations.

Safety constraints:
- Minimum 5 executions before evaluating a skill
- Minimum 3 feedback points before calling skill "good" or "bad"
- Advisory-only recommendations (no auto-routing)
- One-week observation period before any behavioral changes
- Separate production from test logs
"""

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Safety thresholds
MIN_EXECUTIONS_FOR_EVAL = 5
MIN_FEEDBACK_FOR_JUDGMENT = 3
MIN_SUCCESS_RATE_FOR_GOOD = 0.80
MAX_SUCCESS_RATE_FOR_POOR = 0.50
OBSERVATION_PERIOD_DAYS = 7


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load JSONL file."""
    if not path.exists():
        return []
    records = []
    for line in path.read_text().strip().split("\n"):
        if line:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


def generate_monthly_report(month_str: str = None, environment: str = "prod") -> Path:
    """Aggregate logs and generate monthly skills learning report.

    Args:
        month_str: YYYY-MM format (default: current month)
        environment: "prod" for production, "test" for test data

    Returns:
        Path to generated report
    """
    if not month_str:
        month_str = datetime.utcnow().strftime("%Y-%m")

    logs_dir = Path("artifacts/skills_learning")
    logs_dir.mkdir(parents=True, exist_ok=True)

    exec_log = logs_dir / f"execution_log_{environment}_{month_str}.jsonl"
    feedback_log = logs_dir / f"feedback_log_{environment}_{month_str}.jsonl"

    # Load logs
    executions = load_jsonl(exec_log)
    feedback = load_jsonl(feedback_log)

    # Build execution_id → skill_name mapping and aggregate by skill
    exec_id_to_skill: Dict[str, str] = {}
    skill_stats: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "executions": 0,
            "successes": 0,
            "failures": 0,
            "total_latency": 0.0,
            "total_cost": 0.0,
            "total_tokens": 0,
            "feedback": [],
            "errors": [],
            "evaluable": False,  # True if >= MIN_EXECUTIONS_FOR_EVAL
            "judgment_ready": False,  # True if >= MIN_FEEDBACK_FOR_JUDGMENT
        }
    )

    for exec_record in executions:
        skill = exec_record.get("skill_name", "unknown")
        exec_id = exec_record.get("execution_id", "")
        if exec_id:
            exec_id_to_skill[exec_id] = skill

        stats = skill_stats[skill]

        stats["executions"] += 1
        if exec_record.get("outcome", {}).get("success"):
            stats["successes"] += 1
        else:
            stats["failures"] += 1
            error = exec_record.get("outcome", {}).get("error")
            if error:
                stats["errors"].append(error)

        metrics = exec_record.get("metrics", {})
        stats["total_latency"] += metrics.get("latency_ms", 0)
        stats["total_cost"] += metrics.get("cost_usd", 0)
        stats["total_tokens"] += metrics.get("tokens_in", 0) + metrics.get("tokens_out", 0)

    # Join feedback with executions by execution_id
    for fb in feedback:
        exec_id = fb.get("execution_id", "")
        skill = exec_id_to_skill.get(exec_id, "unknown")
        if skill in skill_stats:
            skill_stats[skill]["feedback"].append(fb)

    # Mark skills as evaluable/judgment-ready
    for skill, stats in skill_stats.items():
        if stats["executions"] >= MIN_EXECUTIONS_FOR_EVAL:
            stats["evaluable"] = True
        if len(stats["feedback"]) >= MIN_FEEDBACK_FOR_JUDGMENT:
            stats["judgment_ready"] = True

    # Generate report
    report_lines = [
        f"# Monthly Skills Learning Report — {month_str} ({environment.upper()})",
        "",
        f"Generated: {datetime.utcnow().isoformat()}Z",
        "",
        f"**Safety Status:** Conservative, advisory-only. No auto-routing until {OBSERVATION_PERIOD_DAYS}+ day observation.",
        "",
        "## Summary",
        "",
        f"- Total executions: {len(executions)}",
        f"- Total feedback records: {len(feedback)}",
        f"- Skills monitored: {len(skill_stats)}",
        f"- Evaluable skills (≥{MIN_EXECUTIONS_FOR_EVAL} execs): {sum(1 for s in skill_stats.values() if s['evaluable'])}",
        f"- Judgment-ready skills (≥{MIN_FEEDBACK_FOR_JUDGMENT} feedback): {sum(1 for s in skill_stats.values() if s['judgment_ready'])}",
        "",
        "## Skills by Efficacy (Success Rate)",
        "",
    ]

    # Sort by success rate
    sorted_skills = sorted(
        skill_stats.items(),
        key=lambda x: x[1]["successes"] / max(1, x[1]["executions"]),
        reverse=True,
    )

    for skill_name, stats in sorted_skills:
        execs = stats["executions"]
        success_rate = 100 * stats["successes"] / max(1, execs)
        avg_latency = stats["total_latency"] / max(1, execs)
        avg_cost = stats["total_cost"] / max(1, execs)
        avg_tokens = stats["total_tokens"] / max(1, execs)

        # Evaluation status
        eval_status = ""
        if not stats["evaluable"]:
            eval_status = f" ⚠️ [INSUFFICIENT_DATA: {execs}/{MIN_EXECUTIONS_FOR_EVAL} execs]"
        elif not stats["judgment_ready"]:
            eval_status = f" ⚠️ [LOW_FEEDBACK: {len(stats['feedback'])}/{MIN_FEEDBACK_FOR_JUDGMENT}]"

        report_lines.append(f"### {skill_name}{eval_status}")
        report_lines.append(f"- **Executions:** {execs}")
        report_lines.append(f"- **Success rate:** {success_rate:.1f}%")
        report_lines.append(f"- **Avg latency:** {avg_latency:.0f}ms")
        report_lines.append(f"- **Avg cost:** ${avg_cost:.4f}")
        report_lines.append(f"- **Avg tokens:** {avg_tokens:.0f}")
        report_lines.append("")

        if stats["feedback"]:
            helpful = sum(1 for f in stats["feedback"] if f.get("verdict") == "helpful")
            unhelpful = sum(1 for f in stats["feedback"] if f.get("verdict") == "unhelpful")
            missing = sum(1 for f in stats["feedback"] if f.get("verdict") == "missing")
            report_lines.append("**Feedback:**")
            report_lines.append(f"- Helpful: {helpful}")
            report_lines.append(f"- Unhelpful: {unhelpful}")
            report_lines.append(f"- Missing features: {missing}")

            if unhelpful > 0 or missing > 0:
                report_lines.append("")
                report_lines.append("**Notes:**")
                for fb in stats["feedback"]:
                    if fb.get("notes"):
                        report_lines.append(f"- {fb['notes']}")

        report_lines.append("")

    # Add learning insights (advisory only)
    note_text = (
        f"**Note:** Recommendations below are ADVISORY ONLY. No behavioral changes will be applied until:\n"
        f"  • {OBSERVATION_PERIOD_DAYS}+ days of production observation\n"
        f"  • ≥{MIN_EXECUTIONS_FOR_EVAL} executions per skill\n"
        f"  • ≥{MIN_FEEDBACK_FOR_JUDGMENT} feedback points per skill"
    )
    report_lines.extend(["## Learning Insights (Advisory)", "", note_text, ""])

    # Find most used skills (evaluable only)
    evaluable_skills = [(s, st) for s, st in skill_stats.items() if st["evaluable"]]
    if evaluable_skills:
        most_used = sorted(evaluable_skills, key=lambda x: x[1]["executions"], reverse=True)[:5]
        report_lines.append("### Most Used Skills (Evaluable)")
        for skill, stats in most_used:
            report_lines.append(f"- **{skill}:** {stats['executions']} executions")
        report_lines.append("")

    # Find most expensive skills (evaluable only)
    if evaluable_skills:
        most_expensive = sorted(evaluable_skills, key=lambda x: x[1]["total_cost"], reverse=True)[:5]
        report_lines.append("### Most Expensive Skills (Evaluable)")
        for skill, stats in most_expensive:
            avg_cost = stats["total_cost"] / max(1, stats["executions"])
            report_lines.append(f"- **{skill}:** ${stats['total_cost']:.2f} total (${avg_cost:.4f} avg)")
        report_lines.append("")

    # Find slowest skills (evaluable only)
    if evaluable_skills:
        slowest = sorted(evaluable_skills, key=lambda x: x[1]["total_latency"], reverse=True)[:5]
        report_lines.append("### Slowest Skills (Evaluable)")
        for skill, stats in slowest:
            avg_latency = stats["total_latency"] / max(1, stats["executions"])
            report_lines.append(f"- **{skill}:** {avg_latency:.0f}ms avg latency")
        report_lines.append("")

    # Skills needing feedback (evaluable but not judgment-ready)
    needs_feedback = [(s, st) for s, st in skill_stats.items() if st["evaluable"] and not st["judgment_ready"]]
    if needs_feedback:
        report_lines.append(
            f"### Skills Needing Feedback ({len(needs_feedback)} evaluable, <{MIN_FEEDBACK_FOR_JUDGMENT} feedback)"
        )
        for skill, stats in needs_feedback:
            report_lines.append(f"- **{skill}:** {len(stats['feedback'])}/{MIN_FEEDBACK_FOR_JUDGMENT} feedback points")
        report_lines.append("")

    # Recommendations
    needs_feedback_count = len([s for s in skill_stats.values() if not s["judgment_ready"]])
    report_lines.extend(
        [
            "## Recommendations (Advisory, No Auto-Apply)",
            "",
            f"1. **Collect feedback** on {needs_feedback_count} skills until ≥{MIN_FEEDBACK_FOR_JUDGMENT} points each",
            f"2. **Observe** for {OBSERVATION_PERIOD_DAYS}+ days before routing changes",
            "3. **Review** slow skills for optimization opportunities (latency, cost)",
            "4. **Document** all feedback to inform future auto-routing decisions",
            "5. **Operator approval required** before any behavioral changes",
            "",
        ]
    )

    # Loop review: trim list, efficacy overdue, stalled PENDINGs
    try:
        from tools.skills_loop_review import format_loop_review_sections

        report_lines.extend(
            format_loop_review_sections(environment=environment, logs_dir=logs_dir)
        )
    except ImportError:
        pass

    # Write report
    report_text = "\n".join(report_lines)
    report_path = logs_dir / f"monthly_report_{environment}_{month_str}.md"
    report_path.write_text(report_text)

    return report_path


if __name__ == "__main__":
    import sys

    month = sys.argv[1] if len(sys.argv) > 1 else None
    env = sys.argv[2] if len(sys.argv) > 2 else "prod"
    report_path = generate_monthly_report(month, env)
    print(f"✓ Report generated: {report_path}")
