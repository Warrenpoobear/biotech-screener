"""Static checks for tools/cron_evening_catchup.sh coverage."""

from __future__ import annotations

from pathlib import Path

CATCHUP = Path(__file__).resolve().parent.parent / "tools" / "cron_evening_catchup.sh"


def test_evening_catchup_has_no_llm_heartbeat_agents():
    text = CATCHUP.read_text(encoding="utf-8")
    assert "run_agent_direct.py" not in text
    assert "run_agent " not in "\n".join(
        line for line in text.splitlines() if line.strip() and not line.strip().startswith("#")
    )


def test_evening_catchup_includes_hermes_knowledge_layer():
    text = CATCHUP.read_text(encoding="utf-8")
    assert "hermes_knowledge" in text
    assert "artifacts/ops/knowledge_layer/latest_state.json" in text
    assert "build_hermes_knowledge_layer.py" in text


def test_evening_catchup_includes_hermes_contradiction_detector():
    text = CATCHUP.read_text(encoding="utf-8")
    assert "hermes_contradiction" in text
    assert "hermes-contradiction-detector/run_job.py" in text


def test_evening_catchup_includes_herald_and_ops_supervisor():
    text = CATCHUP.read_text(encoding="utf-8")
    assert "herald_health" in text
    assert "ops_supervisor" in text
    assert "supervisor_sentinel" in text


def test_evening_catchup_uses_builder_for_catalyst_delta_and_price_action():
    text = CATCHUP.read_text(encoding="utf-8")
    assert "build_catalyst_delta.py" in text
    assert "build_price_action_watch.py" in text


def test_evening_catchup_uses_deterministic_ops_and_sentinel():
    text = CATCHUP.read_text(encoding="utf-8")
    assert "build_ops_digest.py" in text
    assert "ruleset_health_monitor.py" in text
    assert "catalyst_resolution_tracker.py" in text
    assert "build_crt_options_join.py" in text
    assert "run_postmortem.py" in text


def test_evening_catchup_writes_fleet_ops_status():
    text = CATCHUP.read_text(encoding="utf-8")
    assert "fleet_ops_status.py" in text
    assert "artifacts/fleet_ops/" in text
    assert "--write" in text
    assert "fleet_completion_audit.py" in text


def test_evening_catchup_runs_audit_before_fleet_ops():
    text = CATCHUP.read_text(encoding="utf-8")
    audit_pos = text.find("fleet_completion_audit.py")
    fleet_pos = text.find("fleet_ops_status.py")
    assert audit_pos != -1 and fleet_pos != -1
    assert audit_pos < fleet_pos
