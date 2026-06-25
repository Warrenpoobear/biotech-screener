"""Static checks for tools/cron_daily_production.sh Class P safety."""

from __future__ import annotations

from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "tools" / "cron_daily_production.sh"


def test_daily_production_cds_before_catchup():
    text = SCRIPT.read_text(encoding="utf-8")
    catchup_pos = text.find('if [ "${1:-}" = "--catch-up" ]')
    cd_pos = text.find('cd "${REPO_ROOT}"')
    assert catchup_pos != -1 and cd_pos != -1
    assert cd_pos < catchup_pos


def test_daily_production_catchup_uses_contract_check():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "check_output_contract.py" in text
    assert "MAX_CATCHUP_DAYS" in text
