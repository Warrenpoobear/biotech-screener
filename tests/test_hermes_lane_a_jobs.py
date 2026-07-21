"""Contract tests for deterministic Lane A Hermes job entry points."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _load_run_job(agent_name: str):
    path = REPO / "agents" / agent_name / "run_job.py"
    spec = importlib.util.spec_from_file_location(f"{agent_name}_run_job", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_held_spec_job_reads_knowledge_layer_items_schema(tmp_path, monkeypatch):
    mod = _load_run_job("hermes-held-spec-ledger")
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)

    ledger_dir = tmp_path / "artifacts" / "ops" / "held_spec_ledger"
    ledger_dir.mkdir(parents=True)
    (ledger_dir / "latest.json").write_text(
        json.dumps(
            {
                "items": [
                    {"id": "spec_087_b1b", "status": "AWAITING_FIRST_FIRE"},
                    {"id": "spec_087_b2", "status": "HELD"},
                    {"id": "bioshort_watch_llm", "status": "HELD_SUPPRESSED"},
                ]
            }
        ),
        encoding="utf-8",
    )

    events = []

    def fake_send_operator_event(**kwargs):
        events.append(kwargs)
        return True

    monkeypatch.setattr(mod, "send_operator_event", fake_send_operator_event)

    assert mod.main() == 0
    assert len(events) == 1
    assert events[0]["event_type"] == "held_spec_ledger"
    assert events[0]["extra"] == {
        "total_specs": 3,
        "approved": 0,
        "blocked": 2,
        "waiting_clearance": 1,
    }


def test_first_fire_job_routes_builder_fail_eval_as_failure(tmp_path, monkeypatch):
    mod = _load_run_job("hermes-first-fire-validator")
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)

    ledger_dir = tmp_path / "artifacts" / "ops" / "first_fire_ledger"
    ledger_dir.mkdir(parents=True)
    (ledger_dir / "latest.json").write_text(
        json.dumps(
            {
                "jobs": [
                    {
                        "job": "biotech_hedge_report",
                        "eval": "FAIL_ARTIFACT_MISSING_PAST_DEADLINE",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    events = []

    def fake_send_operator_event(**kwargs):
        events.append(kwargs)
        return True

    monkeypatch.setattr(mod, "send_operator_event", fake_send_operator_event)

    assert mod.main() == 1
    assert len(events) == 1
    assert events[0]["event_type"] == "first_fire_fail"
    assert events[0]["extra"] == {
        "failed_jobs": ["biotech_hedge_report"],
        "fail_count": 1,
    }
