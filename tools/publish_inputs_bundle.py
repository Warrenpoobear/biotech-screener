#!/usr/bin/env python3
"""Publish Phase-2 inputs bundle as a GitHub Release asset.

Creates a lean tarball containing only the gitignored files that CI needs
to run the screen (currently just market_data.json), validates freshness,
and publishes as a GitHub Release.

Usage:
    # Publish for today (auto-detects date from market_data.json)
    python tools/publish_inputs_bundle.py

    # Publish for a specific date
    python tools/publish_inputs_bundle.py --as-of-date 2026-02-19

    # Also update the 'inputs-latest' floating tag
    python tools/publish_inputs_bundle.py --update-latest

    # Dry-run: build tarball without publishing
    python tools/publish_inputs_bundle.py --dry-run

Release naming:
    Tag:   inputs-{YYYY-MM-DD}
    Asset: phase2-inputs-{YYYY-MM-DD}.tar.gz

The workflow's "Hydrate inputs" step downloads these bundles.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tarfile
import tempfile
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "production_data"

# Files to include in the bundle (relative to production_data/)
BUNDLE_FILES = [
    "market_data.json",
]


def validate_market_data(data_dir: Path, as_of_date: str) -> tuple[bool, str]:
    """Validate market_data.json exists and is fresh enough to publish."""
    mkt_path = data_dir / "market_data.json"
    if not mkt_path.exists():
        return False, f"market_data.json not found in {data_dir}"

    try:
        with open(mkt_path) as f:
            records = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        return False, f"Cannot read market_data.json: {e}"

    if not isinstance(records, list) or len(records) == 0:
        return False, "market_data.json is empty or not a list"

    # Check collected_at freshness
    collected_dates = [
        r.get("collected_at") for r in records
        if isinstance(r, dict) and r.get("collected_at")
    ]
    if not collected_dates:
        return False, "No collected_at field found in market_data.json"

    collected_at = max(collected_dates)
    try:
        collected = date.fromisoformat(collected_at)
        as_of = date.fromisoformat(as_of_date)
    except ValueError as e:
        return False, f"Bad date format: {e}"

    age_days = (as_of - collected).days
    if age_days > 3:
        return False, (
            f"market_data.json is {age_days}d stale "
            f"(collected={collected_at}, as_of={as_of_date}). "
            f"Run collect_market_data.py first."
        )

    # Count records
    n_records = len(records)
    n_with_price = sum(1 for r in records if isinstance(r, dict) and r.get("price"))

    return True, (
        f"OK: {n_records} records, {n_with_price} with price, "
        f"collected={collected_at}, age={age_days}d"
    )


def build_tarball(data_dir: Path, as_of_date: str, output_path: Path) -> Path:
    """Build a tarball of BUNDLE_FILES from data_dir."""
    with tarfile.open(output_path, "w:gz") as tar:
        for filename in BUNDLE_FILES:
            filepath = data_dir / filename
            if filepath.exists():
                tar.add(str(filepath), arcname=filename)
                print(f"  + {filename} ({filepath.stat().st_size:,} bytes)")
            else:
                print(f"  SKIP {filename} (not found)")

    print(f"  Bundle: {output_path} ({output_path.stat().st_size:,} bytes)")
    return output_path


def gh_release_exists(tag: str) -> bool:
    """Check if a GitHub release with the given tag exists."""
    result = subprocess.run(
        ["gh", "release", "view", tag],
        capture_output=True, text=True,
    )
    return result.returncode == 0


def publish_release(
    tarball: Path,
    as_of_date: str,
    *,
    update_latest: bool = False,
) -> None:
    """Publish tarball as a GitHub Release."""
    tag = f"inputs-{as_of_date}"
    title = f"Phase2 inputs bundle {as_of_date}"

    if gh_release_exists(tag):
        print(f"  Release {tag} already exists — deleting and recreating")
        subprocess.run(
            ["gh", "release", "delete", tag, "--yes", "--cleanup-tag"],
            capture_output=True, text=True,
        )

    cmd = [
        "gh", "release", "create", tag,
        str(tarball),
        "--title", title,
        "--notes", f"Phase2 inputs bundle for {as_of_date}\n\nContents: {', '.join(BUNDLE_FILES)}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  FAILED: {result.stderr.strip()}")
        sys.exit(1)
    print(f"  Published: {tag}")
    if result.stdout.strip():
        print(f"  URL: {result.stdout.strip()}")

    if update_latest:
        latest_tag = "inputs-latest"
        latest_tarball = tarball.parent / "phase2-inputs-latest.tar.gz"

        # Copy tarball with the latest name
        import shutil
        shutil.copy2(str(tarball), str(latest_tarball))

        if gh_release_exists(latest_tag):
            subprocess.run(
                ["gh", "release", "delete", latest_tag, "--yes", "--cleanup-tag"],
                capture_output=True, text=True,
            )

        cmd = [
            "gh", "release", "create", latest_tag,
            str(latest_tarball),
            "--title", f"Phase2 inputs (latest: {as_of_date})",
            "--notes", f"Floating tag — always points to the most recent inputs bundle.\n\nCurrently: {as_of_date}",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  WARNING: Failed to update inputs-latest: {result.stderr.strip()}")
        else:
            print(f"  Updated: inputs-latest → {as_of_date}")


def detect_date(data_dir: Path) -> str:
    """Auto-detect as_of_date from market_data.json's collected_at."""
    mkt_path = data_dir / "market_data.json"
    if mkt_path.exists():
        try:
            with open(mkt_path) as f:
                records = json.load(f)
            collected_dates = [
                r.get("collected_at") for r in records
                if isinstance(r, dict) and r.get("collected_at")
            ]
            if collected_dates:
                return max(collected_dates)
        except (json.JSONDecodeError, OSError):
            pass
    return date.today().isoformat()


def main():
    parser = argparse.ArgumentParser(
        description="Publish Phase-2 inputs bundle as a GitHub Release",
    )
    parser.add_argument(
        "--as-of-date",
        help="Bundle date (YYYY-MM-DD). Auto-detected from market_data.json if omitted.",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=DATA_DIR,
        help="Path to production_data/",
    )
    parser.add_argument(
        "--update-latest", action="store_true",
        help="Also update the inputs-latest floating release",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Build tarball but don't publish",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    as_of_date = args.as_of_date or detect_date(data_dir)

    print(f"{'='*60}")
    print(f"PUBLISH INPUTS BUNDLE — {as_of_date}")
    print(f"{'='*60}")

    # Validate
    print("\nValidating market_data.json ...")
    ok, detail = validate_market_data(data_dir, as_of_date)
    print(f"  {detail}")
    if not ok:
        print("\nABORTED: validation failed")
        sys.exit(1)

    # Build tarball
    print("\nBuilding tarball ...")
    with tempfile.TemporaryDirectory() as tmpdir:
        tarball_name = f"phase2-inputs-{as_of_date}.tar.gz"
        tarball_path = Path(tmpdir) / tarball_name
        build_tarball(data_dir, as_of_date, tarball_path)

        if args.dry_run:
            print(f"\nDRY RUN — tarball built but not published")
            print(f"  {tarball_path}")
            return

        # Publish
        print("\nPublishing to GitHub ...")
        publish_release(
            tarball_path, as_of_date,
            update_latest=args.update_latest,
        )

    print(f"\n{'='*60}")
    print(f"DONE — inputs-{as_of_date} published")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
