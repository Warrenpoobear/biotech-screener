#!/usr/bin/env python3
"""Promote a candidate ruleset to active in the manifest.

Validates changelog entry exists and is not still [DRAFT], flips statuses
(candidate -> active, old active -> retired), and writes the updated manifest.

CLI:
  python scripts/promote_ruleset.py d3cdf5c8
  python scripts/promote_ruleset.py d3cdf5c8 --dry-run
  python scripts/promote_ruleset.py d3cdf5c8 --force   # skip changelog validation
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RULESETS_DIR = PROJECT_ROOT / "production_data" / "decision_rulesets"
MANIFEST_PATH = RULESETS_DIR / "manifest.json"
CHANGELOG_PATH = PROJECT_ROOT / "RULESET_CHANGELOG.md"


def _load_manifest() -> Dict[str, Any]:
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_manifest(manifest: Dict[str, Any]) -> None:
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")


def _find_by_id_and_status(manifest: Dict[str, Any], ruleset_id: str, status: str):
    """Find manifest entry by ID and status."""
    for entry in manifest["rulesets"]:
        if entry["id"] == ruleset_id and entry["status"] == status:
            return entry
    return None


def _find_active(manifest: Dict[str, Any]):
    for entry in manifest["rulesets"]:
        if entry["status"] == "active":
            return entry
    return None


def _validate_changelog(ruleset_id: str) -> tuple[bool, str]:
    """Check that a non-DRAFT changelog entry exists for this ruleset ID.

    Returns (ok, message).
    """
    if not CHANGELOG_PATH.exists():
        return False, f"RULESET_CHANGELOG.md not found."

    text = CHANGELOG_PATH.read_text(encoding="utf-8")

    if ruleset_id not in text:
        return False, f"No changelog entry references ruleset ID '{ruleset_id}'."

    # Check for DRAFT marker on lines containing this ID
    for line in text.splitlines():
        if ruleset_id in line and "[DRAFT]" in line:
            return False, (
                f"Changelog entry for '{ruleset_id}' still has [DRAFT] marker.\n"
                f"Edit the entry to remove [DRAFT] and fill in the description before promoting."
            )

    return True, "Changelog entry found and finalized."


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Promote a candidate ruleset to active in the manifest."
    )
    parser.add_argument(
        "ruleset_id",
        help="The 8-char ruleset ID to promote.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would change without writing.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Skip changelog validation (for CI/automation).",
    )
    parser.add_argument(
        "--rollback", action="store_true",
        help="Rollback: promote a retired ruleset back to active.",
    )
    args = parser.parse_args()

    rid = args.ruleset_id

    if not MANIFEST_PATH.exists():
        print("ERROR: manifest.json not found.", file=sys.stderr)
        return 1

    manifest = _load_manifest()

    # Check if this ID is already active
    active = _find_active(manifest)
    if active and active["id"] == rid:
        print(f"Ruleset '{rid}' is already active ({active['file']}). Nothing to do.")
        return 0

    # Find target entry (candidate or retired depending on --rollback)
    if args.rollback:
        if not args.force:
            print(
                "ERROR: --rollback requires --force (emergency rollback skips changelog).",
                file=sys.stderr,
            )
            return 1
        target = _find_by_id_and_status(manifest, rid, "retired")
        if not target:
            for entry in manifest["rulesets"]:
                if entry["id"] == rid:
                    print(
                        f"ERROR: Ruleset '{rid}' has status '{entry['status']}', "
                        f"not 'retired'.",
                        file=sys.stderr,
                    )
                    return 1
            print(f"ERROR: No ruleset with ID '{rid}' found in manifest.", file=sys.stderr)
            return 1
        source_status = "retired"
        updated_by = "promote_ruleset.py --rollback"
    else:
        target = _find_by_id_and_status(manifest, rid, "candidate")
        if not target:
            for entry in manifest["rulesets"]:
                if entry["id"] == rid:
                    print(
                        f"ERROR: Ruleset '{rid}' exists but has status '{entry['status']}', "
                        f"not 'candidate'.",
                        file=sys.stderr,
                    )
                    return 1
            print(f"ERROR: No ruleset with ID '{rid}' found in manifest.", file=sys.stderr)
            return 1
        source_status = "candidate"
        updated_by = "promote_ruleset.py"

    # Validate changelog (unless --force)
    if not args.force:
        ok, msg = _validate_changelog(rid)
        if not ok:
            print(f"ERROR: Changelog validation failed.\n  {msg}", file=sys.stderr)
            print("Use --force to skip changelog validation.", file=sys.stderr)
            return 1
        print(f"Changelog: {msg}")

    # Print summary
    print()
    print(f"Promote: {rid} ({target['file']})")
    if active:
        print(f"Retire:  {active['id']} ({active['file']})")
    print()

    if args.dry_run:
        print("[dry-run] Would update manifest:")
        print(f"  {target['file']}: {source_status} -> active")
        if active:
            print(f"  {active['file']}: active -> retired")
        return 0

    # Flip statuses
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    target["status"] = "active"
    target["updated_by"] = updated_by
    target["updated_at"] = now
    if active:
        active["status"] = "retired"
        active["updated_by"] = updated_by
        active["updated_at"] = now

    _save_manifest(manifest)
    print(f"Updated {MANIFEST_PATH}")
    print(f"  {target['file']}: now active")
    if active:
        print(f"  {active['file']}: now retired")

    return 0


if __name__ == "__main__":
    sys.exit(main())
