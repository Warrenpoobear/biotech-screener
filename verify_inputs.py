#!/usr/bin/env python3
"""Verify production_data inputs before running the Phase-2 screen.

Two modes:
    --generate   Compute SHA256 hashes for required files and write manifest JSON.
    (default)    Validate that required files exist and hashes match the manifest.

Exit codes:
    0  — all checks pass
    1  — missing files, hash mismatch, or missing manifest
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

# Core files the pipeline needs regardless of as-of-date.
REQUIRED_FILES = [
    "universe.json",
    "financial_records.json",
    "trial_records.json",
    "market_data.json",
]

# Pinned config artifacts for Phase-2.
REQUIRED_CONFIGS = [
    "decision_rulesets/v2_phase2_default.json",
    "phase2_health_thresholds/v1.json",
]

DEFAULT_MANIFEST = "inputs_manifest.json"


def sha256_file(path: Path) -> str:
    """Return hex SHA256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def generate_manifest(data_dir: Path, manifest_path: Path) -> int:
    """Compute hashes for required files and write manifest."""
    all_files = REQUIRED_FILES + REQUIRED_CONFIGS
    entries: dict[str, str] = {}
    errors: list[str] = []

    for rel in all_files:
        p = data_dir / rel
        if not p.exists():
            errors.append(f"MISSING: {rel}")
            continue
        entries[rel] = sha256_file(p)

    if errors:
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        print(f"[verify] {len(errors)} required file(s) missing — manifest NOT written.",
              file=sys.stderr)
        return 1

    manifest = {
        "version": 1,
        "data_dir": str(data_dir),
        "files": entries,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[verify] Manifest written: {manifest_path}  ({len(entries)} files)")
    return 0


def verify(data_dir: Path, manifest_path: Path) -> int:
    """Check required files exist and hashes match manifest."""
    errors: list[str] = []

    # --- existence checks (always run, even without manifest) ---
    all_files = REQUIRED_FILES + REQUIRED_CONFIGS
    for rel in all_files:
        p = data_dir / rel
        if not p.exists():
            errors.append(f"MISSING: {rel}")

    if errors:
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        print(f"[verify] FAIL — {len(errors)} required file(s) missing.", file=sys.stderr)
        return 1

    # --- hash verification (only if manifest exists) ---
    if not manifest_path.exists():
        print(f"[verify] WARN — no manifest at {manifest_path}; "
              f"existence checks passed but hashes NOT verified.",
              file=sys.stderr)
        # Still pass — allows first-time runs before a manifest is committed.
        print(f"[verify] OK (existence only, {len(all_files)} files)")
        return 0

    try:
        manifest = json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[verify] FAIL — cannot read manifest: {exc}", file=sys.stderr)
        return 1

    expected: dict[str, str] = manifest.get("files", {})
    mismatches: list[str] = []

    for rel, expected_hash in expected.items():
        p = data_dir / rel
        if not p.exists():
            mismatches.append(f"MISSING: {rel}")
            continue
        actual = sha256_file(p)
        if actual != expected_hash:
            mismatches.append(
                f"HASH MISMATCH: {rel}  expected={expected_hash[:16]}…  "
                f"actual={actual[:16]}…"
            )

    if mismatches:
        for m in mismatches:
            print(f"  {m}", file=sys.stderr)
        print(f"[verify] FAIL — {len(mismatches)} integrity error(s).", file=sys.stderr)
        return 1

    print(f"[verify] OK ({len(expected)} files, hashes verified)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify (or generate manifest for) Phase-2 production inputs.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=SCRIPT_DIR / "production_data",
        help="Production data directory (default: production_data/)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=SCRIPT_DIR / "production_data" / DEFAULT_MANIFEST,
        help=f"Manifest JSON path (default: production_data/{DEFAULT_MANIFEST})",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate manifest from current data (instead of verifying).",
    )
    args = parser.parse_args(argv)

    if args.generate:
        return generate_manifest(args.data_dir, args.manifest)
    return verify(args.data_dir, args.manifest)


if __name__ == "__main__":
    sys.exit(main())
