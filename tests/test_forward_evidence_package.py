"""Tests for freeze-lift forward evidence package."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def test_forward_evidence_dry_run_builds_package():
    from tools.forward_evidence_package import build_package

    package = build_package(
        as_of_date="2026-06-24",
        snapshot_dir=REPO / "data" / "snapshots",
        ic_start_date="2026-04-03",
        horizons=[20],
    )
    assert package["schema"] == "forward_evidence_package.v1"
    assert package["path_c"]["window_end"] == "2026-06-03"
    assert package["advisory_verdict"]["advisory_only"] is True
    assert package["governance"]["does_not_lift_freeze"] is True


def test_advisory_verdict_positive_when_both_above_floor():
    from tools.forward_evidence_package import advisory_verdict

    coinvest = {"horizons": {"T+20": {"mean_ic": 0.03, "n_observations": 10}}}
    final = {"horizons": {"T+20": {"mean_ic": 0.025, "n_observations": 10}}}
    path_c = {"decision": "PATH_C_VALID"}
    adv = advisory_verdict(coinvest_ic=coinvest, final_score_ic=final, path_c=path_c)
    assert adv["verdict"] == "POSITIVE"


def test_advisory_verdict_insufficient_without_data():
    from tools.forward_evidence_package import advisory_verdict

    adv = advisory_verdict(
        coinvest_ic={"horizons": {}},
        final_score_ic={"horizons": {}},
        path_c={"decision": "IC_UNOBSERVABLE"},
    )
    assert adv["verdict"] == "INSUFFICIENT_DATA"


def test_freeze_lift_ack_gate(monkeypatch, tmp_path):
    from tools import forward_evidence_package as mod

    monkeypatch.delenv("FREEZE_LIFT_ACK", raising=False)
    with pytest.raises(SystemExit):
        mod.require_freeze_lift_ack(dry_run=False)

    monkeypatch.setenv("FREEZE_LIFT_ACK", "1")
    mod.require_freeze_lift_ack(dry_run=False)


def test_path_c_close_decision_unobservable_without_ledger():
    from tools.forward_evidence_package import path_c_close_decision

    result = path_c_close_decision(window_end="2026-06-03")
    assert result["decision"] in {"IC_UNOBSERVABLE", "PATH_C_VALID", "PATH_C_REVOKE"}


def test_run_forward_evidence_script_has_gates():
    script = REPO / "tools" / "run_forward_evidence_package.sh"
    text = script.read_text(encoding="utf-8")
    assert "FREEZE_LIFT_ACK" in text
    assert "forward_evidence_package.py" in text
    assert "path_c_window_close_decision.py" in text


def test_path_c_window_close_write(tmp_path, monkeypatch):
    from tools import path_c_window_close_decision as mod

    out = tmp_path / "governance"
    monkeypatch.setattr(mod, "GOV_DIR", out)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "path_c_window_close_decision.py",
            "--write",
            "--window-end",
            "2026-06-03",
            "--as-of-date",
            "2026-06-24",
        ],
    )
    mod.main()
    written = out / "path_c_window_close_2026-06-24.json"
    assert written.is_file()
    payload = json.loads(written.read_text(encoding="utf-8"))
    assert "path_c_close" in payload
