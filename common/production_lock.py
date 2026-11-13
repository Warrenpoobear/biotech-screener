"""Is a daily production run in progress in this checkout?

Two locks guard the daily run, and neither can be read by "does the file exist":

* ``logs/.daily_production.lock`` (``tools/cron_daily_production.sh``) holds the
  wrapper's PID and is removed on exit. A leftover file with a dead PID is stale.
* ``logs/.daily_production_py.lock`` (``tools/run_daily_production.py``) is held with
  ``flock`` and the file is never removed, so it exists permanently after the first run.

Read-only: the flock probe takes the lock without blocking and releases it at once.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

SHELL_LOCK = "logs/.daily_production.lock"
PYTHON_LOCK = "logs/.daily_production_py.lock"


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _flock_held(path: Path) -> bool:
    try:
        import fcntl
    except ImportError:  # no flock on this platform: the shell lock is the only signal
        return False
    try:
        fd = os.open(str(path), os.O_RDWR)
    except OSError:
        return False
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError):
            return True
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)


def production_run_active(repo_root: Path) -> Optional[str]:
    """Relative path of the lock a live run holds, or None when no run is in progress."""
    root = Path(repo_root)
    shell = root / SHELL_LOCK
    if shell.is_file():
        try:
            pid = int(shell.read_text(encoding="utf-8").strip() or "0")
        except (OSError, ValueError):
            pid = 0
        if pid > 0 and _pid_alive(pid):
            return SHELL_LOCK
    py_lock = root / PYTHON_LOCK
    if py_lock.is_file() and _flock_held(py_lock):
        return PYTHON_LOCK
    return None
