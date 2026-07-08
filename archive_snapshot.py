#!/usr/bin/env python3
"""
PIT Snapshot Archival Script

Creates self-contained archives of point-in-time screening snapshots
bundled with the raw input files and code provenance metadata.

Archives are gzip tarballs containing:
  {date}/manifest.json      — provenance metadata with SHA-256 hashes
  {date}/rankings.csv       — ranked securities with component scores
  {date}/metadata.json      — run metadata from the screening pipeline
  {date}/inputs/             — copies of the actual input files used

Usage:
  python archive_snapshot.py                          # Archive today's snapshot
  python archive_snapshot.py --date 2026-02-07        # Archive specific date
  python archive_snapshot.py --verify data/archives/2026-02-07.tar.gz
  python archive_snapshot.py --list

Can be called programmatically:
  from archive_snapshot import create_archive
  create_archive("2026-02-07", snapshot_dir, data_dir, archive_dir)
"""

import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# 1. Core Input File Registry
# ---------------------------------------------------------------------------

CORE_INPUTS = [
    "universe.json",
    "financial_records.json",
    "trial_records.json",
    "market_data.json",
]

OPTIONAL_INPUTS = [
    "holdings_detailed.json",
    "short_interest.json",
    "market_snapshot.json",
    "adr_data.json",
    "fda_designations.json",
    "pdufa_dates.json",
    "partnerships.json",
    "corporate_catalysts.json",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def find_latest_catalyst_file(data_dir: Path, snapshot_date: str) -> Optional[str]:
    """Find the catalyst_events_{date}.json closest to (but not after) snapshot_date.

    Falls back to the nearest file after snapshot_date if none exist before.
    Ignores vnext variants.
    """
    pattern = re.compile(r"^catalyst_events_(\d{4}-\d{2}-\d{2})\.json$")
    candidates: List[str] = []
    for entry in data_dir.iterdir():
        m = pattern.match(entry.name)
        if m:
            candidates.append(m.group(1))
    if not candidates:
        return None
    candidates.sort()
    # Prefer the latest date <= snapshot_date
    before = [d for d in candidates if d <= snapshot_date]
    if before:
        chosen = before[-1]
    else:
        chosen = candidates[0]  # all are after snapshot_date
    return f"catalyst_events_{chosen}.json"


def get_git_info(repo_dir: Path) -> Dict[str, Any]:
    """Capture git branch, commit SHA, and dirty status."""
    info: Dict[str, Any] = {"branch": None, "commit_sha": None, "dirty": None}
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            cwd=repo_dir,
            timeout=5,
        )
        if result.returncode == 0:
            info["branch"] = result.stdout.strip()

        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=repo_dir,
            timeout=5,
        )
        if result.returncode == 0:
            info["commit_sha"] = result.stdout.strip()

        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            cwd=repo_dir,
            timeout=5,
        )
        if result.returncode == 0:
            info["dirty"] = len(result.stdout.strip()) > 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return info


# ---------------------------------------------------------------------------
# 2. Manifest Builder
# ---------------------------------------------------------------------------


def build_manifest(
    snapshot_date: str,
    snapshot_dir: Path,
    data_dir: Path,
    input_files: Dict[str, Path],
    archive_size: int = 0,
) -> Dict[str, Any]:
    """Build the provenance manifest for the archive."""
    repo_dir = Path(__file__).resolve().parent

    # Read version and scoring model from snapshot metadata
    meta_path = snapshot_dir / "metadata.json"
    with open(meta_path) as f:
        snapshot_meta = json.load(f)

    git_info = get_git_info(repo_dir)

    # Hash each input file
    file_hashes: Dict[str, Dict[str, Any]] = {}
    for name, path in sorted(input_files.items()):
        file_hashes[name] = {
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    manifest = {
        "snapshot_date": snapshot_date,
        "archived_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "code_version": snapshot_meta.get("version", "unknown"),
        "scoring_model": snapshot_meta.get("scoring_model", "unknown"),
        "git": git_info,
        "input_files": file_hashes,
        "ticker_count": snapshot_meta.get("ticker_count", 0),
        "archive_size_bytes": archive_size,
    }
    return manifest


# ---------------------------------------------------------------------------
# 3. Archive Creator
# ---------------------------------------------------------------------------


def _collect_input_files(
    data_dir: Path,
    snapshot_date: str,
) -> Tuple[Dict[str, Path], List[str]]:
    """Resolve input files, returning (found, warnings)."""
    found: Dict[str, Path] = {}
    warnings: List[str] = []

    for name in CORE_INPUTS:
        p = data_dir / name
        if not p.exists():
            raise FileNotFoundError(f"Core input file missing: {p}")
        found[name] = p

    for name in OPTIONAL_INPUTS:
        p = data_dir / name
        if p.exists():
            found[name] = p
        else:
            warnings.append(f"Optional input not found, skipping: {name}")

    catalyst = find_latest_catalyst_file(data_dir, snapshot_date)
    if catalyst:
        found[catalyst] = data_dir / catalyst
    else:
        warnings.append("No catalyst_events file found")

    return found, warnings


def create_archive(
    snapshot_date: str,
    snapshot_dir: Path,
    data_dir: Path,
    archive_dir: Path,
) -> Path:
    """Create a self-contained gzip tarball for the given snapshot date.

    Returns the path to the created archive.
    """
    snapshot_path = snapshot_dir / snapshot_date
    rankings = snapshot_path / "rankings.csv"
    metadata = snapshot_path / "metadata.json"

    if not rankings.exists() or not metadata.exists():
        raise FileNotFoundError(f"Snapshot incomplete — need rankings.csv and metadata.json in {snapshot_path}")

    input_files, warnings = _collect_input_files(data_dir, snapshot_date)
    for w in warnings:
        print(f"  WARNING: {w}", file=sys.stderr)

    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / f"{snapshot_date}.tar.gz"

    if archive_path.exists():
        print(f"  WARNING: Archive already exists at {archive_path} — skipping to preserve original", file=sys.stderr)
        return archive_path

    # Build manifest (archive_size filled in after creation)
    manifest = build_manifest(snapshot_date, snapshot_path, data_dir, input_files)

    # Write manifest to a temp file, then bundle everything
    with tempfile.TemporaryDirectory() as tmp:
        manifest_path = Path(tmp) / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
            f.write("\n")

        with tarfile.open(archive_path, "w:gz") as tar:
            prefix = snapshot_date

            tar.add(str(manifest_path), arcname=f"{prefix}/manifest.json")
            tar.add(str(rankings), arcname=f"{prefix}/rankings.csv")
            tar.add(str(metadata), arcname=f"{prefix}/metadata.json")

            for name, path in sorted(input_files.items()):
                tar.add(str(path), arcname=f"{prefix}/inputs/{name}")

    # Update manifest with final archive size and rewrite inside tarball
    final_size = archive_path.stat().st_size
    manifest["archive_size_bytes"] = final_size

    with tempfile.TemporaryDirectory() as tmp:
        manifest_path = Path(tmp) / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
            f.write("\n")

        # Rewrite the archive with the updated manifest
        old_archive = archive_path.with_suffix(".tar.gz.tmp")
        archive_path.rename(old_archive)

        with tarfile.open(old_archive, "r:gz") as old_tar, tarfile.open(archive_path, "w:gz") as new_tar:
            prefix = snapshot_date
            new_tar.add(str(manifest_path), arcname=f"{prefix}/manifest.json")
            for member in old_tar.getmembers():
                if member.name == f"{prefix}/manifest.json":
                    continue
                new_tar.addfile(member, old_tar.extractfile(member))

        old_archive.unlink()

    final_size = archive_path.stat().st_size
    print(f"  Archive created: {archive_path} ({final_size:,} bytes)")
    print(f"  Input files: {len(input_files)}")
    print(f"  Ticker count: {manifest['ticker_count']}")
    return archive_path


# ---------------------------------------------------------------------------
# 4. Verify Mode
# ---------------------------------------------------------------------------


def verify_archive(archive_path: Path) -> bool:
    """Verify integrity of an archive by recomputing SHA-256 hashes.

    Returns True if all hashes match, False otherwise.
    """
    if not archive_path.exists():
        print(f"  ERROR: Archive not found: {archive_path}", file=sys.stderr)
        return False

    with tarfile.open(archive_path, "r:gz") as tar:
        # Safety: reject path traversal / absolute paths in member names
        for m in tar.getmembers():
            p = Path(m.name)
            if p.is_absolute() or ".." in p.parts:
                raise ValueError(f"Unsafe path in archive: {m.name}")

        # Find the manifest
        manifest_member = None
        for member in tar.getmembers():
            if member.name.endswith("/manifest.json"):
                manifest_member = member
                break
        if manifest_member is None:
            print("  ERROR: No manifest.json found in archive", file=sys.stderr)
            return False

        manifest = json.load(tar.extractfile(manifest_member))
        prefix = manifest["snapshot_date"]
        expected_hashes = manifest.get("input_files", {})

        ok = True
        checked = 0

        for filename, info in sorted(expected_hashes.items()):
            arcname = f"{prefix}/inputs/{filename}"
            try:
                member = tar.getmember(arcname)
            except KeyError:
                print(f"  FAIL: {filename} — missing from archive")
                ok = False
                continue

            h = hashlib.sha256()
            fobj = tar.extractfile(member)
            for chunk in iter(lambda: fobj.read(1 << 16), b""):
                h.update(chunk)
            actual = h.hexdigest()

            if actual != info["sha256"]:
                print(f"  FAIL: {filename} — hash mismatch")
                print(f"         expected: {info['sha256']}")
                print(f"         actual:   {actual}")
                ok = False
            else:
                checked += 1

        if ok:
            print(f"  OK: All {checked} input files verified")
        return ok


# ---------------------------------------------------------------------------
# 5. List Mode
# ---------------------------------------------------------------------------


def list_archives(archive_dir: Path) -> None:
    """List all archives with date, size, and ticker count."""
    if not archive_dir.exists():
        print("  No archives directory found.")
        return

    archives = sorted(archive_dir.glob("*.tar.gz"))
    if not archives:
        print("  No archives found.")
        return

    print(f"  {'Date':<14} {'Size':>12} {'Tickers':>8}  {'Model'}")
    print(f"  {'─' * 14} {'─' * 12} {'─' * 8}  {'─' * 30}")

    for path in archives:
        size = path.stat().st_size
        date_str = path.name.replace(".tar.gz", "")  # e.g. "2026-02-07"
        ticker_count = "?"
        model = "?"
        try:
            with tarfile.open(path, "r:gz") as tar:
                for member in tar.getmembers():
                    if member.name.endswith("/manifest.json"):
                        manifest = json.load(tar.extractfile(member))
                        ticker_count = str(manifest.get("ticker_count", "?"))
                        model = manifest.get("scoring_model", "?")
                        break
        except Exception:
            pass

        size_str = _human_size(size)
        print(f"  {date_str:<14} {size_str:>12} {ticker_count:>8}  {model}")


def _human_size(n: float) -> str:
    """Format byte count as human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:,.0f} {unit}" if unit == "B" else f"{n:,.1f} {unit}"
        n /= 1024
    return f"{n:,.1f} TB"


# ---------------------------------------------------------------------------
# 6. CLI Interface
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Archive PIT screening snapshots with provenance metadata.",
    )
    parser.add_argument(
        "--date",
        default=datetime.now().strftime("%Y-%m-%d"),
        help="Snapshot date to archive (YYYY-MM-DD). Default: today",
    )
    parser.add_argument(
        "--verify",
        metavar="ARCHIVE",
        help="Verify integrity of an existing archive",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_archives",
        help="List all existing archives",
    )
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=Path("data/snapshots"),
        help="Snapshot directory (default: data/snapshots)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("production_data"),
        help="Production data directory (default: production_data)",
    )
    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=Path("data/archives"),
        help="Archive output directory (default: data/archives)",
    )

    args = parser.parse_args()

    if args.verify:
        print(f"Verifying archive: {args.verify}")
        ok = verify_archive(Path(args.verify))
        return 0 if ok else 1

    if args.list_archives:
        list_archives(args.archive_dir)
        return 0

    # Archive mode
    print(f"Archiving snapshot: {args.date}")
    try:
        create_archive(args.date, args.snapshot_dir, args.data_dir, args.archive_dir)
    except FileNotFoundError as e:
        print(f"  ERROR: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
