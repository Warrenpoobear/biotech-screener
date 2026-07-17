#!/usr/bin/env python3
"""Snapshot production_data input files before each daily run.

Archives key reference files (and the forward-shadow evidence ledgers) to
data/pit_archives/YYYY-MM-DD/ with a SHA-256 manifest so that the exact
inputs for any historical date can be recovered. The evidence ledgers are
snapshotted here because they are being untracked from git (B2) to stop
dirtying the working tree on every run.

Usage:
    python3 scripts/archive_production_inputs.py --as-of-date 2026-04-02
    python3 scripts/archive_production_inputs.py --as-of-date 2026-04-02 --data-dir production_data
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import date, datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
_DEFAULT_DATA_DIR = SCRIPT_DIR / "production_data"
_DEFAULT_ARCHIVE_ROOT = SCRIPT_DIR / "data" / "pit_archives"

# Static reference files archived every run (sourced from --data-dir).
_STATIC_FILES = [
    "universe.json",
    "financial_records.json",
    "market_data.json",
    "price_history.csv",
    "trial_records.json",
]

# Forward-shadow evidence ledgers, as (archive_name, repo-relative source).
# These are append-only book-of-record ledgers (forward-validation mandate
# SM-20260629-001; inst_delta / cross_signal shadow audits). They are being
# untracked from git (B2) to stop dirtying the tree every run, so snapshotting
# them here preserves point-in-time content. Sourced from the repo root rather
# than --data-dir; archive names are namespaced to keep them flat and unambiguous.
_EVIDENCE_FILES = [
    ("forward_validation_captures.jsonl", "artifacts/forward_validation/captures.jsonl"),
    ("forward_validation_fills.jsonl", "artifacts/forward_validation/fills.jsonl"),
    ("inst_delta_forward_shadow_checkpoints.jsonl", "artifacts/audit/inst_delta_forward_shadow/checkpoints.jsonl"),
    ("cross_signal_forward_shadow_buckets.jsonl", "artifacts/audit/cross_signal_forward_shadow/buckets.jsonl"),
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_files(data_dir: Path, date_str: str) -> list[tuple[str, Path]]:
    """Return (archive_name, source_path) pairs for files to archive."""
    pairs: list[tuple[str, Path]] = []
    for name in _STATIC_FILES:
        src = data_dir / name
        if src.exists():
            pairs.append((name, src))
        else:
            print(f"[ARCHIVE] WARN: {name} not found in {data_dir}", file=sys.stderr)

    # Dated catalyst file
    cat_name = f"catalyst_events_{date_str}.json"
    cat_src = data_dir / cat_name
    if cat_src.exists():
        pairs.append((cat_name, cat_src))
    else:
        print(f"[ARCHIVE] WARN: {cat_name} not found in {data_dir}", file=sys.stderr)

    # Forward-shadow evidence ledgers (repo-relative, outside --data-dir).
    for arch_name, relpath in _EVIDENCE_FILES:
        src = SCRIPT_DIR / relpath
        if src.exists():
            pairs.append((arch_name, src))
        else:
            print(f"[ARCHIVE] WARN: {relpath} not found", file=sys.stderr)

    return pairs


def _load_manifest(manifest_path: Path) -> dict | None:
    if not manifest_path.exists():
        return None
    with open(manifest_path) as f:
        return json.load(f)


def archive(date_str: str, data_dir: Path, archive_root: Path) -> int:
    """Archive input files. Returns 0 on success."""
    dest_dir = archive_root / date_str
    manifest_path = dest_dir / "manifest.json"

    files = _resolve_files(data_dir, date_str)
    if not files:
        print("[ARCHIVE] No files to archive.", file=sys.stderr)
        return 1

    # Build current hashes from source.
    current: dict[str, dict] = {}
    for name, src in files:
        current[name] = {
            "sha256": _sha256(src),
            "size_bytes": src.stat().st_size,
        }

    # Idempotency check: skip if manifest matches.
    existing = _load_manifest(manifest_path)
    if existing is not None:
        if existing.get("files") == current:
            print(f"[ARCHIVE] SKIP {date_str} — archive exists with matching hashes")
            return 0

    # Write archive.
    dest_dir.mkdir(parents=True, exist_ok=True)
    for name, src in files:
        shutil.copy2(src, dest_dir / name)
        print(f"[ARCHIVE] {date_str}  {name} ({current[name]['size_bytes']:,} bytes)")

    manifest = {
        "as_of_date": date_str,
        "archived_at": datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "files": current,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    print(f"[ARCHIVE] {date_str}  manifest.json written ({len(current)} files)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Archive production input files.")
    p.add_argument(
        "--as-of-date",
        default=date.today().isoformat(),
        help="Date to archive (default: today)",
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=_DEFAULT_DATA_DIR,
        help="Production data directory",
    )
    p.add_argument(
        "--archive-dir",
        type=Path,
        default=_DEFAULT_ARCHIVE_ROOT,
        help="Archive root directory",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return archive(args.as_of_date, args.data_dir, args.archive_dir)


if __name__ == "__main__":
    raise SystemExit(main())
