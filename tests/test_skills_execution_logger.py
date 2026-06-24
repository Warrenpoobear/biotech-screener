"""Tests for tools/skills_execution_logger.py environment tagging."""

from __future__ import annotations

import json
import os
from pathlib import Path

from tools.skills_execution_logger import SkillExecutionLogger, record_feedback, telemetry_environment


def test_telemetry_environment_default():
    os.environ.pop("SKILLS_TELEMETRY_ENV", None)
    assert telemetry_environment() == "prod"


def test_telemetry_environment_test(monkeypatch):
    monkeypatch.setenv("SKILLS_TELEMETRY_ENV", "test")
    assert telemetry_environment() == "test"


def test_log_execution_writes_env_tagged_file(tmp_path, monkeypatch):
    monkeypatch.setenv("SKILLS_TELEMETRY_ENV", "test")
    logger = SkillExecutionLogger(logs_dir=tmp_path, environment="test")
    exec_id = logger.log_execution(
        skill_name="screener_ops",
        task_context="unit test",
        inputs={},
        outputs={},
        latency_ms=10.0,
        tokens_in=0,
        tokens_out=0,
        cost_usd=0.0,
        success=True,
    )
    assert exec_id
    logs = list(tmp_path.glob("execution_log_test_*.jsonl"))
    assert len(logs) == 1
    row = json.loads(logs[0].read_text(encoding="utf-8").strip())
    assert row["environment"] == "test"
    assert row["skill_name"] == "screener_ops"


def test_record_feedback_gated_without_verdict_flag(tmp_path, monkeypatch):
    monkeypatch.delenv("SELFIMPROVE_IMMEDIATE_VERDICT", raising=False)
    logger = SkillExecutionLogger(logs_dir=tmp_path, environment="prod")
    exec_id = logger.log_execution(
        skill_name="x",
        task_context="t",
        inputs={},
        outputs={},
        latency_ms=1.0,
        tokens_in=0,
        tokens_out=0,
        cost_usd=0.0,
        success=True,
    )
    record_feedback(exec_id, "helpful")
    assert not list(tmp_path.glob("feedback_log_*.jsonl"))


def test_record_feedback_writes_when_gate_open(tmp_path, monkeypatch):
    monkeypatch.setenv("SELFIMPROVE_IMMEDIATE_VERDICT", "1")
    monkeypatch.setenv("SKILLS_TELEMETRY_ENV", "prod")
    # Reset global logger so it picks up tmp_path via get_logger fresh instance
    import tools.skills_execution_logger as mod

    mod._skill_logger = SkillExecutionLogger(logs_dir=tmp_path, environment="prod")
    exec_id = mod.log_skill(
        skill_name="y",
        task_context="t",
        inputs={},
        outputs={},
        latency_ms=1.0,
    )
    mod.record_feedback(exec_id, "helpful", notes="ok")
    feedback_files = list(tmp_path.glob("feedback_log_prod_*.jsonl"))
    assert feedback_files
    row = json.loads(feedback_files[0].read_text(encoding="utf-8").strip())
    assert row["verdict"] == "helpful"
    assert row["execution_id"] == exec_id
