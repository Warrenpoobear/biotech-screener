#!/usr/bin/env python3
"""Skill execution logging for Hermes learning framework.

Logs all skill invocations with metrics, outcomes, and feedback hooks.
Enables recursive self-improvement through performance tracking and feedback learning.

Log files are environment-tagged for prod/test separation:
  artifacts/skills_learning/execution_log_{env}_{YYYY-MM}.jsonl
  artifacts/skills_learning/feedback_log_{env}_{YYYY-MM}.jsonl

Set SKILLS_TELEMETRY_ENV=prod|test (default: prod).
Feedback via record_feedback() is gated behind SELFIMPROVE_IMMEDIATE_VERDICT=1.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def telemetry_environment() -> str:
    """Return prod or test from SKILLS_TELEMETRY_ENV."""
    env = os.getenv("SKILLS_TELEMETRY_ENV", "prod").strip().lower()
    return env if env in ("prod", "test") else "prod"


class SkillExecutionLogger:
    """Log skill executions to JSONL for learning and optimization."""

    def __init__(self, logs_dir: Path = Path("artifacts/skills_learning"), environment: str | None = None):
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.environment = environment or telemetry_environment()

    def _execution_log_path(self, month_str: str) -> Path:
        return self.logs_dir / f"execution_log_{self.environment}_{month_str}.jsonl"

    def _feedback_log_path(self, month_str: str) -> Path:
        return self.logs_dir / f"feedback_log_{self.environment}_{month_str}.jsonl"

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
        month_str = datetime.utcnow().strftime("%Y-%m")

        record = {
            "execution_id": exec_id,
            "skill_name": skill_name,
            "environment": self.environment,
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

        log_file = self._execution_log_path(month_str)
        try:
            with open(log_file, "a", encoding="utf-8") as f:
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
        month_str = datetime.utcnow().strftime("%Y-%m")
        feedback = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "execution_id": execution_id,
            "environment": self.environment,
            "verdict": verdict,
            "notes": notes,
        }

        log_file = self._feedback_log_path(month_str)
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(feedback) + "\n")
        except OSError as e:
            print(f"Warning: Could not record feedback: {e}")


_skill_logger: SkillExecutionLogger | None = None


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
    """Convenience function to log a skill execution."""
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
    if os.getenv("SELFIMPROVE_IMMEDIATE_VERDICT") != "1":
        return
    get_logger().record_feedback(execution_id, verdict, notes)


if __name__ == "__main__":
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
    print(f"Logged execution: {exec_id} -> {logger._execution_log_path(datetime.utcnow().strftime('%Y-%m'))}")

    logger.record_feedback(exec_id, "helpful", notes="Works as expected")
    print("Recorded feedback")
