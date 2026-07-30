#!/usr/bin/env python3
"""Out-of-tree durable mirror for the forward-validation capture ledger.

Classification: FORWARD_VALIDATION_EVIDENCE_DURABILITY / NO_MODEL_CHANGE

Spec 115 Phase 2a. On 2026-07-23 a mandate-eligible capture was written
successfully — ``quality=PASS``, TRUTH_CARD.md on disk — and then destroyed by a
git working-tree revert of the tracked ledger. It went unnoticed for five days.
Two of the ten trading days 2026-07-15..2026-07-28 lost a window to git
mechanics, against a 52-window gate standing at n=4.

Design
------
``artifacts/forward_validation/captures.jsonl`` remains the **published audit
record**: it stays tracked, so the evidence keeps a reviewable history. Every
capture is additionally appended to a mirror that lives **outside the git
working tree**, where `checkout`, `restore`, `stash` and `reset` cannot reach it.

That makes Mode B *recoverable* rather than merely prevented, which is stronger
than relocating the ledger outright — relocation would protect durability but
give up the git-based audit trail.

Guarantees and non-guarantees
-----------------------------
* Best-effort: a mirror failure must never fail a production capture. Every
  entry point swallows I/O errors.
* Append-only. Nothing here rewrites or reorders the mirror.
* ``restore_missing`` only ever re-appends rows the mirror already holds. It
  cannot fabricate evidence, and it does not change ``capture_mode`` or
  eligibility — a restored row is the original row.
* Not protected against ``git clean -fdx`` of the repo (the mirror is outside
  it) nor against deletion of the mirror directory itself. Back that path up.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent

# Deliberately outside the repo. Honours XDG_DATA_HOME so the location is
# configurable without touching production code.
_MIRROR_ENV_VAR = "FV_MIRROR_PATH"
_MIRROR_SUBDIR = Path("biotech-screener") / "forward_validation"
_MIRROR_FILENAME = "captures.mirror.jsonl"


def default_mirror_path() -> Path:
    """Absolute path to the durable mirror, outside the git working tree."""
    override = os.environ.get(_MIRROR_ENV_VAR)
    if override:
        return Path(override).expanduser().resolve()
    base = os.environ.get("XDG_DATA_HOME")
    root = Path(base).expanduser() if base else Path.home() / ".local" / "share"
    return (root / _MIRROR_SUBDIR / _MIRROR_FILENAME).resolve()


def _read_records(path: Path) -> List[Dict[str, Any]]:
    """Parse a JSONL ledger, skipping unparseable lines. Missing file -> []."""
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError as exc:  # pragma: no cover - defensive
        log.warning("fv_durable_mirror: unreadable ledger %s: %s", path, exc)
        return []
    return out


def dates_in(path: Path) -> List[str]:
    """Capture dates present in a ledger, in file order."""
    return [r["date"] for r in _read_records(path) if isinstance(r.get("date"), str)]


def mirror_append(record: Dict[str, Any], mirror: Optional[Path] = None) -> bool:
    """Append one capture to the durable mirror. Best-effort; never raises.

    Returns True when the row was written.
    """
    target = mirror or default_mirror_path()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n")
        return True
    except (OSError, TypeError, ValueError) as exc:
        # A durability aid must not be able to break the thing it protects.
        log.warning("fv_durable_mirror: could not mirror capture to %s: %s", target, exc)
        return False


def missing_from_tracked(tracked: Path, mirror: Optional[Path] = None) -> List[str]:
    """Capture dates the mirror holds but the tracked ledger has lost.

    A tracked ledger that is *ahead* of the mirror is not reported — that is not
    evidence loss (e.g. the mirror was introduced after some captures existed).
    """
    mirror_path = mirror or default_mirror_path()
    mirror_dates = dates_in(mirror_path)
    if not mirror_dates:
        return []
    tracked_dates = set(dates_in(tracked))
    seen: set = set()
    lost: List[str] = []
    for d in mirror_dates:
        if d not in tracked_dates and d not in seen:
            seen.add(d)
            lost.append(d)
    return sorted(lost)


def restore_missing(tracked: Path, mirror: Optional[Path] = None) -> List[str]:
    """Re-append mirrored captures the tracked ledger has lost.

    Returns the dates restored, in date order. Idempotent. Only rows already
    present in the mirror are written — this cannot fabricate evidence.
    """
    mirror_path = mirror or default_mirror_path()
    lost = missing_from_tracked(tracked=tracked, mirror=mirror_path)
    if not lost:
        return []

    wanted = set(lost)
    by_date: Dict[str, Dict[str, Any]] = {}
    for rec in _read_records(mirror_path):
        d = rec.get("date")
        if isinstance(d, str) and d in wanted and d not in by_date:
            by_date[d] = rec

    restored: List[str] = []
    try:
        tracked.parent.mkdir(parents=True, exist_ok=True)
        with open(tracked, "a", encoding="utf-8") as f:
            for d in sorted(by_date):
                f.write(json.dumps(by_date[d], separators=(",", ":")) + "\n")
                restored.append(d)
    except OSError as exc:
        log.error("fv_durable_mirror: could not restore into %s: %s", tracked, exc)
        return []
    return restored
