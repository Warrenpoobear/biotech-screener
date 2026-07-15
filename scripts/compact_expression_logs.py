#!/usr/bin/env python3
"""One-off compaction of Spec 062 expression logs (issue #495).

Pre-#495, every pipeline re-run re-appended the full day's decision and
attribution records, so both logs hold each (ticker, node_id) once per
re-run (x8-x13 observed). This script collapses them keep-last per key
(a resolved attribution record wins over pending), writing a timestamped
.bak backup next to each log before rewriting.

Usage:
    python scripts/compact_expression_logs.py            # compact both logs
    python scripts/compact_expression_logs.py --dry-run  # report only
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from event_ev.expression_attribution import dedup_keep_last  # noqa: E402

LOGS = [
    REPO_ROOT / "data" / "expression_decision_log.jsonl",
    REPO_ROOT / "data" / "expression_attribution_log.jsonl",
]


def compact(path: Path, dry_run: bool) -> None:
    if not path.exists():
        print(f"{path.name}: missing, skipped")
        return

    raw_lines = []
    records = []
    malformed = 0
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            raw_lines.append(s)
            try:
                records.append(json.loads(s))
            except json.JSONDecodeError:
                malformed += 1

    deduped = dedup_keep_last(records)
    removed = len(records) - len(deduped)
    print(
        f"{path.name}: {len(records)} records -> {len(deduped)} "
        f"({removed} duplicates removed, {malformed} malformed lines preserved)"
    )
    if dry_run or removed == 0:
        return

    backup = path.with_name(f"{path.name}.pre_dedup_{date.today().isoformat()}.bak")
    shutil.copy2(path, backup)
    print(f"  backup: {backup.name}")

    tmp = path.with_name(path.name + ".compact.tmp")
    with open(tmp, "w") as f:
        # Malformed lines are preserved verbatim ahead of the deduped records;
        # loaders skip them either way.
        for s in raw_lines:
            try:
                json.loads(s)
            except json.JSONDecodeError:
                f.write(s + "\n")
        for rec in deduped:
            f.write(json.dumps(rec, default=str) + "\n")
    tmp.replace(path)
    print(f"  rewritten: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="report only, change nothing")
    args = parser.parse_args()
    for log in LOGS:
        compact(log, args.dry_run)


if __name__ == "__main__":
    main()
