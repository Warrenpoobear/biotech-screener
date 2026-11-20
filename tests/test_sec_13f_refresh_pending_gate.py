"""Gate: newer 13F filings waiting on a coinvest rebuild FAIL the run when most managers are ahead."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from run_daily_production import GATE_ALLOWLIST, check_sec_13f_refresh_pending  # noqa: E402

AS_OF = "2026-11-17"


def _setup(tmp_path: Path, in_use: str | None, periods: list[str], selected: bool = True):
    signals = tmp_path / "coinvest_signals.json"
    payload = {"tickers": {}}
    if in_use is not None:
        payload["period_of_report"] = in_use
    signals.write_text(json.dumps(payload), encoding="utf-8")
    cache = tmp_path / "PIT"
    (cache / AS_OF).mkdir(parents=True)
    managers = [
        {"manager_cik": f"{i:010d}", "selected": selected, "period_of_report": p} for i, p in enumerate(periods)
    ]
    (cache / AS_OF / "index.json").write_text(json.dumps({"managers": managers}), encoding="utf-8")
    return {"cache_base_dir": cache, "coinvest_path": signals}


def test_gate_name_is_allowlisted():
    assert "sec_13f_refresh_pending" in GATE_ALLOWLIST


def test_fails_when_most_managers_have_a_newer_quarter(tmp_path):
    kw = _setup(tmp_path, "2026-06-30", ["2026-09-30"] * 20 + ["2026-06-30"] * 10)
    gate = check_sec_13f_refresh_pending(AS_OF, **kw)
    assert gate.status == "FAIL"
    assert "20/30" in gate.detail and "2026-09-30" in gate.detail
    assert gate.value["managers_with_newer_filing"] == 20


def test_passes_while_only_a_few_have_filed(tmp_path):
    # 2026-11-13: 3 of 30 roster managers had filed Q3.
    kw = _setup(tmp_path, "2026-06-30", ["2026-09-30"] * 3 + ["2026-06-30"] * 27)
    gate = check_sec_13f_refresh_pending(AS_OF, **kw)
    assert gate.status == "PASS" and "3/30" in gate.detail


def test_passes_when_coinvest_is_current(tmp_path):
    kw = _setup(tmp_path, "2026-09-30", ["2026-09-30"] * 30)
    assert check_sec_13f_refresh_pending(AS_OF, **kw).status == "PASS"


def test_unselected_managers_are_ignored(tmp_path):
    kw = _setup(tmp_path, "2026-06-30", ["2026-09-30"] * 30, selected=False)
    assert check_sec_13f_refresh_pending(AS_OF, **kw).status == "PASS"


def test_warns_when_period_in_use_is_unknown(tmp_path):
    kw = _setup(tmp_path, None, ["2026-09-30"] * 30)
    assert check_sec_13f_refresh_pending(AS_OF, **kw).status == "WARN"
    kw["coinvest_path"] = tmp_path / "absent.json"
    assert check_sec_13f_refresh_pending(AS_OF, **kw).status == "WARN"


def test_missing_cache_index_is_left_to_the_cache_gate(tmp_path):
    kw = _setup(tmp_path, "2026-06-30", ["2026-09-30"] * 30)
    assert check_sec_13f_refresh_pending("2026-11-18", **kw).status == "PASS"


def test_fails_when_every_manager_is_newer(tmp_path):
    kw = _setup(tmp_path, "2026-06-30", ["2027-12-31"] * 30)
    assert check_sec_13f_refresh_pending(AS_OF, **kw).status == "FAIL"
