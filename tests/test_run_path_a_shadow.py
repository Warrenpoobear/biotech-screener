"""Static checks for tools/run_path_a_shadow.sh."""

from __future__ import annotations

from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "tools" / "run_path_a_shadow.sh"


def test_path_a_shadow_uses_shadow_policy_and_out_dir():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "portfolio_policy_path_a_shadow.json" in text
    assert "live_shadow_portfolio.py" in text
    assert "artifacts/live_shadow_path_a" in text
    assert "path_a_manifest.json" in text


def test_path_a_shadow_does_not_touch_production_policy():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "portfolio_policy.json" not in text.replace("portfolio_policy_path_a_shadow.json", "")
