"""A leftover lock file is not a production run in progress."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from common.production_lock import PYTHON_LOCK, SHELL_LOCK, production_run_active


def _root(tmp_path: Path) -> Path:
    (tmp_path / "logs").mkdir()
    return tmp_path


def test_no_lock_files_means_no_run(tmp_path):
    assert production_run_active(_root(tmp_path)) is None


def test_leftover_python_lock_file_is_not_a_run(tmp_path):
    root = _root(tmp_path)
    (root / PYTHON_LOCK).write_text("")  # run_daily_production.py never removes this file
    assert production_run_active(root) is None


def test_held_python_lock_is_a_run(tmp_path):
    fcntl = pytest.importorskip("fcntl")
    root = _root(tmp_path)
    lock = root / PYTHON_LOCK
    fd = os.open(str(lock), os.O_RDWR | os.O_CREAT)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # flock is per open file description, so probe from another process.
        probe = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; from pathlib import Path; from common.production_lock import production_run_active;"
                "print(production_run_active(Path(sys.argv[1])))",
                str(root),
            ],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parents[1]),
        )
        assert probe.stdout.strip() == PYTHON_LOCK, probe.stderr
    finally:
        os.close(fd)
    assert production_run_active(root) is None  # released


def test_shell_lock_with_dead_pid_is_stale(tmp_path):
    root = _root(tmp_path)
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    (root / SHELL_LOCK).write_text(f"{child.pid}\n")
    assert production_run_active(root) is None


def test_shell_lock_with_live_pid_is_a_run(tmp_path):
    root = _root(tmp_path)
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        (root / SHELL_LOCK).write_text(f"{child.pid}\n")
        assert production_run_active(root) == SHELL_LOCK
    finally:
        child.kill()
        child.wait()


def test_shell_lock_with_garbage_is_ignored(tmp_path):
    root = _root(tmp_path)
    (root / SHELL_LOCK).write_text("not-a-pid")
    assert production_run_active(root) is None
