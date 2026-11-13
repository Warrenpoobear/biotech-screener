"""Ticker-change tool: one pass over production_data, atomic, refuses ticker reuse.

Classification: UNIVERSE_MAINTENANCE_TOOLING_NO_MODEL_CHANGE
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.apply_ticker_change import TickerChangeError, apply_plan, main, plan_ticker_change

REASON = "symbol change confirmed by vendor corporate-actions feed"


def _make(tmp_path: Path) -> Path:
    prod = tmp_path / "production_data"
    (prod / "pit_financials").mkdir(parents=True)
    (prod / "universe.json").write_text(
        '[\n  {\n    "ticker": "ACHV",\n    "name": "Achieve",\n    "cik": "0000949858"\n  },\n'
        '  {\n    "ticker": "ACHVX",\n    "name": "Other"\n  }\n]\n'
    )
    (prod / "trial_records.json").write_text(
        '[{"ticker": "ACHV", "nct_id": "NCT1"}, {"ticker":"ACHV","nct_id":"NCT2"}, {"ticker": "ZZZ", "nct_id": "NCT3"}]'
    )
    (prod / "ipo_dates.json").write_text('{"tickers": {"ACHV": {"first_price_date": "2024-11-04"}, "ZZZ": {}}}')
    (prod / "quarterly_burn_history.json").write_text('{"ACHV": {"ticker": "ACHV", "q": []}}')
    (prod / "price_history.csv").write_text(
        "date,ticker,close,open,high,low,volume\n2026-11-10,ACHV,6.4,0,0,0,10\n2026-11-10,ZZZ,1,0,0,0,5\n"
    )
    (prod / "price_history_split_adj.csv").write_text("ticker,date,close\nACHV,2026-11-10,6.400000\n")
    (prod / "pit_financials" / "ACHV.json").write_text('{"ticker": "ACHV", "facts": {}}')
    (prod / "corporate_actions.json").write_text(
        '{\n  "_schema": "corporate_actions.v1",\n  "actions": [\n    {\n      "ticker": "ZZZ",\n'
        '      "action": "delisting",\n      "effective_date": "2025-01-01"\n    }\n  ]\n}\n'
    )
    (prod / "universe_audit_log.jsonl").write_text('{"action": "add", "ticker": "ACHV"}\n')
    return prod


def _plan(prod: Path, old="ACHV", new="QUIT"):
    return plan_ticker_change(prod, old, new, "2026-11-10", REASON, "tester", "2026-11-13T00:00:00Z")


def test_every_file_is_relabelled_in_one_pass(tmp_path):
    prod = _make(tmp_path)
    writes, renames, counts = _plan(prod)
    apply_plan(writes, renames)

    universe = json.loads((prod / "universe.json").read_text())
    assert [e["ticker"] for e in universe] == ["QUIT", "ACHVX"]  # a longer symbol with the same prefix is untouched
    assert universe[0]["cik"] == "0000949858"
    trials = json.loads((prod / "trial_records.json").read_text())
    assert [t["ticker"] for t in trials] == ["QUIT", "QUIT", "ZZZ"]
    assert "QUIT" in json.loads((prod / "ipo_dates.json").read_text())["tickers"]
    burn = json.loads((prod / "quarterly_burn_history.json").read_text())
    assert list(burn) == ["QUIT"] and burn["QUIT"]["ticker"] == "QUIT"
    assert "2026-11-10,QUIT,6.4" in (prod / "price_history.csv").read_text()
    assert "2026-11-10,ZZZ,1" in (prod / "price_history.csv").read_text()
    assert (prod / "price_history_split_adj.csv").read_text().splitlines()[1] == "QUIT,2026-11-10,6.400000"
    assert not (prod / "pit_financials" / "ACHV.json").exists()
    assert json.loads((prod / "pit_financials" / "QUIT.json").read_text())["ticker"] == "QUIT"
    assert counts["trial_records.json"] == 2 and counts["price_history.csv"] == 1
    assert not list(prod.rglob("*.ticker_change.tmp"))


def test_registry_and_audit_log_record_the_change(tmp_path):
    prod = _make(tmp_path)
    apply_plan(*_plan(prod)[:2])
    actions = json.loads((prod / "corporate_actions.json").read_text())["actions"]
    assert actions[0]["ticker"] == "ZZZ"  # existing entries untouched
    assert actions[-1] == {
        "ticker": "QUIT",
        "action": "ticker_change",
        "effective_date": "2026-11-10",
        "old_ticker": "ACHV",
        "new_ticker": "QUIT",
        "notes": REASON,
    }
    log = [json.loads(line) for line in (prod / "universe_audit_log.jsonl").read_text().splitlines()]
    assert log[0] == {"action": "add", "ticker": "ACHV"}
    assert log[-1]["action"] == "ticker_change" and log[-1]["old_ticker"] == "ACHV" and log[-1]["operator"] == "tester"


def test_formatting_of_untouched_content_is_preserved(tmp_path):
    prod = _make(tmp_path)
    before = (prod / "universe.json").read_text()
    apply_plan(*_plan(prod)[:2])
    assert (prod / "universe.json").read_text() == before.replace('"ticker": "ACHV"', '"ticker": "QUIT"')


def test_dry_run_writes_nothing(tmp_path, capsys):
    prod = _make(tmp_path)
    snapshot = {p: p.read_bytes() for p in prod.rglob("*") if p.is_file()}
    rc = main(["--old", "ACHV", "--new", "QUIT", "--effective-date", "2026-11-10", "--reason", REASON,
               "--production-dir", str(prod)])  # fmt: skip
    assert rc == 0 and "Dry run" in capsys.readouterr().out
    assert {p: p.read_bytes() for p in prod.rglob("*") if p.is_file()} == snapshot


def test_ticker_reuse_is_refused(tmp_path):
    prod = _make(tmp_path)
    (prod / "trial_records.json").write_text('[{"ticker": "ACHV"}, {"ticker": "QUIT"}]')
    with pytest.raises(TickerChangeError, match="already appears"):
        _plan(prod)


def test_new_symbol_already_in_price_history_is_refused(tmp_path):
    prod = _make(tmp_path)
    with open(prod / "price_history.csv", "a") as f:
        f.write("2020-01-02,QUIT,3,0,0,0,1\n")
    with pytest.raises(TickerChangeError, match="price_history.csv"):
        _plan(prod)


def test_unknown_old_symbol_is_refused(tmp_path):
    prod = _make(tmp_path)
    with pytest.raises(TickerChangeError, match="not in universe.json"):
        _plan(prod, old="NOPE")


def test_apply_refuses_while_a_production_run_is_in_progress(tmp_path, capsys):
    import subprocess
    import sys

    prod = _make(tmp_path)
    (tmp_path / "logs").mkdir()
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        (tmp_path / "logs" / ".daily_production.lock").write_text(str(child.pid))
        rc = main(["--old", "ACHV", "--new", "QUIT", "--effective-date", "2026-11-10", "--reason", REASON,
                   "--production-dir", str(prod), "--apply"])  # fmt: skip
    finally:
        child.kill()
        child.wait()
    assert rc == 3 and "in progress" in capsys.readouterr().out
    assert '"ACHV"' in (prod / "universe.json").read_text()


def test_leftover_lock_file_does_not_block_apply(tmp_path, capsys):
    prod = _make(tmp_path)
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / ".daily_production_py.lock").write_text("")
    rc = main(["--old", "ACHV", "--new", "QUIT", "--effective-date", "2026-11-10", "--reason", REASON,
               "--production-dir", str(prod), "--apply"])  # fmt: skip
    assert rc == 0 and "Applied" in capsys.readouterr().out
    assert '"QUIT"' in (prod / "universe.json").read_text()


def test_second_run_is_refused_not_double_applied(tmp_path):
    prod = _make(tmp_path)
    apply_plan(*_plan(prod)[:2])
    with pytest.raises(TickerChangeError):
        _plan(prod)
