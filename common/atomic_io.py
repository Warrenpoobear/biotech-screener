"""Atomic file replacement for production_data/.

Files under ``production_data/`` are hard-linked into staged PIT input
directories (``data/staging/*``). Writing one in place (``open(p, "w")``,
``Path.write_text``, append mode) writes THROUGH every hard link and silently
rewrites inputs that past runs were pinned to. Replacing the file with
``os.replace`` instead swaps in a new inode: the live path gets the new
content and every staged copy keeps the bytes it was captured with.

Use these helpers for any write to a production_data file.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Union

PathLike = Union[str, Path]


def atomic_write_text(path: PathLike, text: str, encoding: str = "utf-8") -> None:
    """Write ``text`` to ``path`` by temp-file + ``os.replace`` in the same dir.

    Preserves the existing file's permission bits. Never writes through hard
    links. The temp file is removed if anything fails before the rename.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = path.stat().st_mode & 0o777 if path.exists() else None
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        if mode is not None:
            os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def atomic_write_json(
    path: PathLike, obj: Any, *, indent: int = 2, ensure_ascii: bool = True, trailing_newline: bool = False
) -> None:
    """``json.dumps`` ``obj`` and write it with :func:`atomic_write_text`."""
    text = json.dumps(obj, indent=indent, ensure_ascii=ensure_ascii)
    if trailing_newline:
        text += "\n"
    atomic_write_text(path, text)


def atomic_append_line(path: PathLike, line: str, encoding: str = "utf-8") -> None:
    """Append one line to a (possibly hard-linked) log file without writing through.

    Reads the current content, appends, and atomically replaces. Fine for the
    small audit logs in production_data; not meant for high-volume logs.
    """
    path = Path(path)
    existing = path.read_text(encoding=encoding) if path.exists() else ""
    if existing and not existing.endswith("\n"):
        existing += "\n"
    if not line.endswith("\n"):
        line += "\n"
    atomic_write_text(path, existing + line, encoding=encoding)
