"""Daily fin_records stage must keep financial_records.json's ticker list current.

run_screen.py reads production_data/financial_records.json, and its PIT override
(pit_financials/) only replaces records for tickers ALREADY in that file. The
13:30 cron_data_extras.sh `fin_records` stage runs collect_financial_data.py,
which writes financial_data.json, not financial_records.json; only the weekly
universe refresh ran tools/merge_financial_data_to_records.py. So a ticker that
gained financials mid-week got no PIT financials until the weekend.

Also: the merge tool wrote financial_records.json in place. production_data
files are hard-linked into staging dirs, so an in-place write mutates every
linked copy. It must replace the file atomically.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import stat
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "tools" / "cron_data_extras.sh"


def _stage_function(name: str) -> str:
    text = SCRIPT.read_text(encoding="utf-8")
    m = re.search(rf"^{name}\(\) \{{.*?^\}}", text, re.S | re.M)
    assert m, f"{name} not found in {SCRIPT}"
    return m.group(0)


def test_fin_records_stage_runs_collector_then_merge(tmp_path):
    calls = tmp_path / "calls.txt"
    fake = tmp_path / "fakepy"
    fake.write_text(f'#!/usr/bin/env bash\necho "$*" >> {calls}\nexit 0\n', encoding="utf-8")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    body = _stage_function("stage_fin_records")
    script = f'set -euo pipefail\nPYTHON="{fake}"\nlog() {{ echo "LOG: $*"; }}\n{body}\nstage_fin_records\n'
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    lines = calls.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("collect_financial_data.py")
    assert any(line.startswith("tools/merge_financial_data_to_records.py") for line in lines[1:]), lines
    # the log must name the file the collector actually writes
    assert "financial_data.json" in out.stdout


def _load_merge(monkeypatch, fin_dat: Path, fin_rec: Path):
    spec = importlib.util.spec_from_file_location("merge_fin", REPO / "tools" / "merge_financial_data_to_records.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "FIN_DAT", fin_dat)
    monkeypatch.setattr(mod, "FIN_REC", fin_rec)
    return mod


def test_merge_replaces_records_atomically_not_through_hard_links(tmp_path, monkeypatch):
    fin_dat = tmp_path / "financial_data.json"
    fin_rec = tmp_path / "financial_records.json"
    fin_dat.write_text(json.dumps([{"ticker": "OLD", "Cash": 1.0}, {"ticker": "NEW", "Cash": 5e7, "CFO": -1e6}]))
    original = json.dumps([{"ticker": "OLD", "Cash": 1.0}])
    fin_rec.write_text(original)
    staged = tmp_path / "staging_2026-11-11_financial_records.json"
    os.link(fin_rec, staged)  # as production_data files are linked into staging dirs

    mod = _load_merge(monkeypatch, fin_dat, fin_rec)
    assert mod.main() == 0

    assert [r["ticker"] for r in json.loads(fin_rec.read_text())] == ["OLD", "NEW"]
    assert staged.read_text() == original  # linked copy untouched
