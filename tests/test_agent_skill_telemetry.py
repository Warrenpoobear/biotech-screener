"""Tests for tools/agent_skill_telemetry.py."""

from __future__ import annotations

from tools.agent_skill_telemetry import log_agent_run, log_hermes_job_exit


def test_log_agent_run_returns_execution_id(monkeypatch):
    calls: list[dict] = []

    def _fake_log_skill(**kwargs):
        calls.append(kwargs)
        return "exec-test-1"

    import tools.agent_skill_telemetry as mod

    monkeypatch.setattr(mod, "log_skill", _fake_log_skill)
    exec_id = log_agent_run("ops_supervisor", "test", outputs={"ok": True})
    assert exec_id == "exec-test-1"
    assert calls[0]["skill_name"] == "ops_supervisor"


def test_log_agent_run_swallows_errors(monkeypatch):
    import tools.agent_skill_telemetry as mod

    def _boom(**_kwargs):
        raise RuntimeError("log failed")

    monkeypatch.setattr(mod, "log_skill", _boom)
    assert log_agent_run("x", "y") is None


def test_log_hermes_job_exit(monkeypatch):
    calls: list[dict] = []

    def _fake_log_skill(**kwargs):
        calls.append(kwargs)
        return "hermes-1"

    import tools.agent_skill_telemetry as mod

    monkeypatch.setattr(mod, "log_skill", _fake_log_skill)
    log_hermes_job_exit("hermes-held-spec-ledger", 0, 0.0)
    assert calls[0]["skill_name"] == "hermes-held-spec-ledger"
    assert calls[0]["success"] is True
