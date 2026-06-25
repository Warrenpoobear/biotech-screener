"""CI contract: deterministic fleet wiring audit must PASS on the repo."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def test_fleet_completion_audit_passes_on_repo():
    from tools.fleet_completion_audit import build_audit

    report = build_audit()
    assert report["overall"] == "PASS", report.get("checks", [])
    assert report["fail_count"] == 0
