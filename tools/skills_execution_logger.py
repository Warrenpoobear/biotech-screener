#!/usr/bin/env python3
"""Skill execution logging for Hermes learning framework.

Logs all skill invocations with metrics, outcomes, and feedback hooks.
Enables recursive self-improvement through performance tracking and feedback learning.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class SkillExecutionLogger:
    """Log skill executions to JSONL for learning and optimization."""

    def __init__(self, logs_dir: Path = Path("artifacts/skills_learning")):
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def log_execution(
        self,
        skill_name: str,
        task_context: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        latency_ms: float,
        tokens_in: int,
        tokens_out: int,
        cost_usd: float,
        success: bool,
        error: Optional[str] = None,
    ) -> str:
        """Log a skill execution. Returns execution_id for later feedback."""
        exec_id = str(uuid.uuid4())[:8]

        record = {
            "execution_id": exec_id,
            "skill_name": skill_name,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "task_context": task_context,
            "inputs": inputs,
            "outputs": outputs,
            "metrics": {
                "latency_ms": latency_ms,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "cost_usd": cost_usd,
            },
            "outcome": {
                "success": success,
                "error": error,
                "user_feedback": None,
            },
        }

        # Append to current month's log
        month_str = datetime.utcnow().strftime("%Y-%m")
        log_file = self.logs_dir / f"execution_log_{month_str}.jsonl"

        try:
            with open(log_file, "a") as f:
                f.write(json.dumps(record) + "\n")
        except OSError as e:
            print(f"Warning: Could not log skill execution: {e}")

        return exec_id

    def record_feedback(
        self,
        execution_id: str,
        verdict: str,  # "helpful" | "unhelpful" | "missing"
        notes: str = "",
    ) -> None:
        """Record feedback on a previous execution."""
        feedback = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "execution_id": execution_id,
            "verdict": verdict,
            "notes": notes,
        }

        month_str = datetime.utcnow().strftime("%Y-%m")
        log_file = self.logs_dir / f"feedback_log_{month_str}.jsonl"

        try:
            with open(log_file, "a") as f:
                f.write(json.dumps(feedback) + "\n")
        except OSError as e:
            print(f"Warning: Could not record feedback: {e}")


# Global instance
_skill_logger = None


def get_logger() -> SkillExecutionLogger:
    """Get or create global logger instance."""
    global _skill_logger
    if _skill_logger is None:
        _skill_logger = SkillExecutionLogger()
    return _skill_logger


def log_skill(
    skill_name: str,
    task_context: str,
    inputs: Dict[str, Any],
    outputs: Dict[str, Any],
    latency_ms: float,
    tokens_in: int = 0,
    tokens_out: int = 0,
    cost_usd: float = 0.0,
    success: bool = True,
    error: Optional[str] = None,
) -> str:
    """Convenience function to log a skill execution.

    Returns execution_id for later feedback recording.
    """
    return get_logger().log_execution(
        skill_name=skill_name,
        task_context=task_context,
        inputs=inputs,
        outputs=outputs,
        latency_ms=latency_ms,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost_usd=cost_usd,
        success=success,
        error=error,
    )


def record_feedback(execution_id: str, verdict: str, notes: str = "") -> None:
    """Record feedback on a skill execution. Gated when called from automation."""
    import os

    if os.getenv("SELFIMPROVE_IMMEDIATE_VERDICT") != "1":
        return
    get_logger().record_feedback(execution_id, verdict, notes)


if __name__ == "__main__":
    # Quick test
    logger = SkillExecutionLogger()
    exec_id = logger.log_execution(
        skill_name="test-skill",
        task_context="testing logging framework",
        inputs={"test": True},
        outputs={"result": "success"},
        latency_ms=123.4,
        tokens_in=100,
        tokens_out=50,
        cost_usd=0.001,
        success=True,
    )
    print(f"Logged execution: {exec_id}")

    logger.record_feedback(exec_id, "helpful", notes="Works as expected")
    print("Recorded feedback")
