"""Static checks for tools/cron_evening_catchup.sh coverage."""

from __future__ import annotations

from pathlib import Path

CATCHUP = Path(__file__).resolve().parent.parent / "tools" / "cron_evening_catchup.sh"


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
