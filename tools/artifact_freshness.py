"""Shared artifact freshness helpers for agent heartbeat checks.

Prefers YYYY-MM-DD embedded in artifact filenames over file mtime so git
checkout or copy does not mask true content staleness.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path

ISO_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def parse_dates_in_name(path: Path) -> list[date]:
    dates: list[date] = []
    for m in ISO_DATE_RE.finditer(path.name):
        try:
            dates.append(date.fromisoformat(m.group(1)))
        except ValueError:
            continue
    return dates


def _iter_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    if not root.is_dir():
        return []
    return [f for f in root.rglob("*") if f.is_file()]


def newest_content_date_under(root: Path) -> tuple[date | None, Path | None]:
    """Return newest YYYY-MM-DD found in any filename under root."""
    best: date | None = None
    best_path: Path | None = None
    for f in _iter_files(root):
        for d in parse_dates_in_name(f):
            if best is None or d > best:
                best, best_path = d, f
    return best, best_path


def newest_artifact_freshness(
    repo_root: Path,
    rel_paths: list[str],
) -> tuple[date | None, Path | None, str]:
    """Determine newest artifact date across declared paths.

    Prefers YYYY-MM-DD in filenames; falls back to file mtime when no
    embedded dates exist. Returns (newest_date, sample_path, method) where
    method is ``content_date``, ``mtime``, or ``none``.
    """
    best_content: date | None = None
    best_content_path: Path | None = None
    best_mtime: float | None = None
    best_mtime_path: Path | None = None

    for rel in rel_paths:
        p = repo_root / rel
        for f in _iter_files(p):
            dates = parse_dates_in_name(f)
            if dates:
                d = max(dates)
                if best_content is None or d > best_content:
                    best_content, best_content_path = d, f
            else:
                try:
                    mt = f.stat().st_mtime
                except OSError:
                    continue
                if best_mtime is None or mt > best_mtime:
                    best_mtime, best_mtime_path = mt, f

    if best_content is not None:
        return best_content, best_content_path, "content_date"

    if best_mtime_path is not None and best_mtime is not None:
        return datetime.fromtimestamp(best_mtime).date(), best_mtime_path, "mtime"

    return None, None, "none"


def age_days(as_of: date, artifact_date: date) -> int:
    return (as_of - artifact_date).days


def format_stale_source(latest: date, as_of: date, threshold: int) -> str:
    age = age_days(as_of, latest)
    return (
        f"STALE_SOURCE: latest artifact {latest.isoformat()} "
        f"({age}d ago; threshold {threshold}d)"
    )
