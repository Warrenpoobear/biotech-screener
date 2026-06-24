"""Thin telemetry wrapper for deterministic agent/builder runs.

Non-blocking: logging failures never break production paths.
"""

from __future__ import annotations

import time
from typing import Any

from tools.skills_logger_v2 import log_skill  # noqa: E402


def log_agent_run(
    agent: str,
    task_context: str,
    *,
    inputs: dict[str, Any] | None = None,
    outputs: dict[str, Any] | None = None,
    success: bool = True,
    error: str | None = None,
    latency_ms: float | None = None,
    environment: str = "prod",
) -> str | None:
    """Log a deterministic agent or builder execution. Returns execution_id or None."""
    try:
        return log_skill(
            skill_name=agent,
            task_context=task_context,
            inputs=inputs or {},
            outputs=outputs or {},
            latency_ms=latency_ms if latency_ms is not None else 0.0,
            success=success,
            error=error,
            environment=environment,
        )
    except Exception:
        return None


class AgentRunTimer:
    """Context manager for timed agent/builder telemetry."""

    def __init__(self, agent: str, task_context: str, **kwargs: Any) -> None:
        self.agent = agent
        self.task_context = task_context
        self.kwargs = kwargs
        self._start = 0.0
        self.execution_id: str | None = None

    def __enter__(self) -> AgentRunTimer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, _tb) -> bool:
        latency_ms = (time.perf_counter() - self._start) * 1000
        self.execution_id = log_agent_run(
            self.agent,
            self.task_context,
            latency_ms=latency_ms,
            success=exc is None,
            error=str(exc)[:500] if exc else None,
            **self.kwargs,
        )
        return False


def log_hermes_job_exit(job_name: str, exit_code: int, started_perf: float) -> None:
    """Log Hermes Lane A job completion (non-blocking)."""
    log_agent_run(
        job_name,
        f"Hermes job {job_name}",
        outputs={"exit_code": exit_code},
        success=exit_code == 0,
        latency_ms=(time.perf_counter() - started_perf) * 1000,
    )
