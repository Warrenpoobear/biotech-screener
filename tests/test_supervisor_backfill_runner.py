"""tools/supervisor_backfill_runner.sh must terminate when today's artifact exists.

The date loop was `while [ "$CURRENT_DATE" != "$TODAY" ]`, starting from
LAST_DATE + 1 day. When today's supervisor artifact already existed,
START_DATE was tomorrow and the loop never reached TODAY: the 2026-11-10 18:45
cron run spun for ~16 hours until it was killed.
"""

from __future__ import annotations

import shutil
import subprocess
from datetime import date, timedelta
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "tools" / "supervisor_backfill_runner.sh"

STUB = """import sys, pathlib
args = sys.argv[1:]
as_of = args[args.index("--as-of") + 1]
out = pathlib.Path(__file__).resolve().parents[{up}] / "artifacts" / "ops_supervisor"
out.mkdir(parents=True, exist_ok=True)
(out / f"{{as_of}}_{kind}.json").write_text("{{}}")
"""


def _layout(tmp_path: Path) -> Path:
    (tmp_path / "tools").mkdir()
    shutil.copy(SCRIPT, tmp_path / "tools" / SCRIPT.name)
    (tmp_path / "agents" / "ops_supervisor").mkdir(parents=True)
    (tmp_path / "agents" / "ops_supervisor" / "supervisor.py").write_text(STUB.format(up=2, kind="supervisor"))
    (tmp_path / "tools" / "agent_supervisor_sentinel.py").write_text(STUB.format(up=1, kind="sentinel"))
    (tmp_path / "logs").mkdir()
    (tmp_path / "artifacts" / "ops_supervisor").mkdir(parents=True)
    return tmp_path


def _run(root: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(root / "tools" / SCRIPT.name)], capture_output=True, text=True, timeout=30)


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_terminates_when_todays_artifact_already_exists(tmp_path):
    """THE regression: START_DATE = tomorrow. Used to loop forever."""
    root = _layout(tmp_path)
    today = date.today().isoformat()
    (root / "artifacts" / "ops_supervisor" / f"{today}_supervisor.json").write_text("{}")

    proc = _run(root)  # TimeoutExpired here == the infinite loop is back
    assert proc.returncode == 0, proc.stderr
    assert "No missing weekdays detected" in proc.stdout


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_backfills_missing_weekdays_then_today(tmp_path):
    root = _layout(tmp_path)
    today = date.today()
    last = today - timedelta(days=7)
    (root / "artifacts" / "ops_supervisor" / f"{last.isoformat()}_supervisor.json").write_text("{}")

    proc = _run(root)
    assert proc.returncode == 0, proc.stderr
    made = {p.name for p in (root / "artifacts" / "ops_supervisor").glob("*_supervisor.json")}
    expected = {
        f"{(last + timedelta(days=i)).isoformat()}_supervisor.json"
        for i in range(1, 7)
        if (last + timedelta(days=i)).isoweekday() <= 5
    }
    assert expected <= made
    assert f"{today.isoformat()}_supervisor.json" in made
