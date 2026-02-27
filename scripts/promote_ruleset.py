#!/usr/bin/env python3
"""Promote a candidate ruleset to active in the manifest.

Validates changelog entry, requires a passing gate artifact (or --force),
updates manifest + pinned IDs + default path, and writes a promotion receipt.

CLI:
  python scripts/promote_ruleset.py 0c1129f6 \
      --gate-summary /tmp/ruleset_gate_out/.../ruleset_eval.json
  python scripts/promote_ruleset.py 0c1129f6 --dry-run
  python scripts/promote_ruleset.py 0c1129f6 --force   # bypass gate + changelog
  python scripts/promote_ruleset.py 0c1129f6 --rollback --force
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RULESETS_DIR = PROJECT_ROOT / "production_data" / "decision_rulesets"
MANIFEST_PATH = RULESETS_DIR / "manifest.json"
CHANGELOG_PATH = PROJECT_ROOT / "RULESET_CHANGELOG.md"
RECEIPTS_DIR = PROJECT_ROOT / "artifacts" / "promotions"

# Files containing pinned ruleset IDs
PINNED_FILES = [
    PROJECT_ROOT / "run_screen.py",
    PROJECT_ROOT / "run_phase2_snapshot_delta.py",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_manifest(path: Path = MANIFEST_PATH) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_manifest(manifest: Dict[str, Any], path: Path = MANIFEST_PATH) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")


def _find_by_id_and_status(manifest: Dict[str, Any], ruleset_id: str, status: str):
    for entry in manifest["rulesets"]:
        if entry["id"] == ruleset_id and entry["status"] == status:
            return entry
    return None


def _find_active(manifest: Dict[str, Any]):
    for entry in manifest["rulesets"]:
        if entry["status"] == "active":
            return entry
    return None


def _compute_ruleset_id(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:8]


def _git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=str(PROJECT_ROOT),
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_changelog(
    ruleset_id: str, changelog_path: Path = CHANGELOG_PATH
) -> Tuple[bool, str]:
    if not changelog_path.exists():
        return False, "RULESET_CHANGELOG.md not found."
    text = changelog_path.read_text(encoding="utf-8")
    if ruleset_id not in text:
        return False, f"No changelog entry references ruleset ID '{ruleset_id}'."
    for line in text.splitlines():
        if ruleset_id in line and "[DRAFT]" in line:
            return False, (
                f"Changelog entry for '{ruleset_id}' still has [DRAFT] marker.\n"
                f"Edit the entry to remove [DRAFT] before promoting."
            )
    return True, "Changelog entry found and finalized."


def validate_gate(
    gate_path: Path,
    expected_candidate_id: str,
) -> Tuple[bool, str, Dict[str, Any]]:
    """Validate a gate summary JSON. Returns (ok, reason, gate_data)."""
    if not gate_path.exists():
        return False, f"Gate summary not found: {gate_path}", {}

    try:
        data = json.loads(gate_path.read_text(encoding="utf-8"))
    except Exception as e:
        return False, f"Cannot parse gate summary: {e}", {}

    gate = data.get("gate") or {}
    verdict = gate.get("verdict", "")
    n_evaluated = data.get("n_evaluated", 0)

    # Check candidate ID matches
    cand_id = data.get("candidate", {}).get("id", "")
    if cand_id != expected_candidate_id:
        return False, (
            f"Gate candidate ID '{cand_id}' does not match "
            f"expected '{expected_candidate_id}'"
        ), data

    # Check verdict
    if verdict not in ("PASS", "WARN", "FAIL"):
        return False, f"Gate verdict '{verdict}' is not recognized", data

    if verdict == "FAIL":
        reasons = gate.get("fail_reasons", [])
        return False, f"Gate verdict is FAIL: {'; '.join(reasons)}", data

    if verdict == "WARN":
        reasons = gate.get("warn_reasons", [])
        return False, f"Gate verdict is WARN: {'; '.join(reasons)}", data

    # Check evaluated dates
    if n_evaluated < 1:
        return False, f"Gate evaluated 0 dates (need >= 1)", data

    return True, "Gate PASS", data


# ---------------------------------------------------------------------------
# Pin updates
# ---------------------------------------------------------------------------


def update_pinned_ids(
    old_id: str,
    new_id: str,
    pinned_files: Optional[List[Path]] = None,
    dry_run: bool = False,
) -> List[str]:
    """Replace PHASE2_PINNED_RULESET_ID in pinned files. Returns list of updated paths."""
    files = pinned_files or PINNED_FILES
    updated = []
    pattern = re.compile(
        r'(PHASE2_PINNED_RULESET_ID\s*=\s*")[^"]*(")'
    )
    for fpath in files:
        if not fpath.exists():
            continue
        text = fpath.read_text(encoding="utf-8")
        new_text, n = pattern.subn(rf'\g<1>{new_id}\g<2>', text)
        if n > 0 and new_text != text:
            if not dry_run:
                fpath.write_text(new_text, encoding="utf-8")
            updated.append(str(fpath))
    return updated


def update_default_path(
    new_filename: str,
    run_screen_path: Optional[Path] = None,
    dry_run: bool = False,
) -> bool:
    """Update PHASE2_DEFAULT_RULESET_PATH in run_screen.py."""
    fpath = run_screen_path or (PROJECT_ROOT / "run_screen.py")
    if not fpath.exists():
        return False
    text = fpath.read_text(encoding="utf-8")
    # Match the filename in the path construction
    pattern = re.compile(
        r'(/ "production_data" / "decision_rulesets" / ")[^"]*(")'
    )
    new_text, n = pattern.subn(rf'\g<1>{new_filename}\g<2>', text)
    if n > 0 and new_text != text:
        if not dry_run:
            fpath.write_text(new_text, encoding="utf-8")
        return True
    return False


# ---------------------------------------------------------------------------
# Receipt
# ---------------------------------------------------------------------------


def write_receipt(
    old_active_id: str,
    new_active_id: str,
    candidate_path: str,
    promoted_file: str,
    gate_data: Dict[str, Any],
    gate_summary_path: str,
    gate_run_url: str,
    forced: bool,
    receipts_dir: Optional[Path] = None,
) -> Path:
    """Write a promotion receipt JSON. Returns the receipt path."""
    if receipts_dir is None:
        receipts_dir = RECEIPTS_DIR
    receipts_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    receipt_path = receipts_dir / f"promotion_{today}_{new_active_id}.json"

    gate_summary = {}
    if gate_data:
        gate_info = gate_data.get("gate") or {}
        ts = gate_data.get("temporal_stability") or {}
        perf = gate_data.get("performance") or {}
        gate_summary = {
            "verdict": gate_info.get("verdict"),
            "evaluated_dates": gate_data.get("n_evaluated", 0),
            "skipped_dates": gate_data.get("n_skipped", 0),
            "mean_top60_overlap": ts.get("mean_top60_overlap"),
            "max_rank_shift": (ts.get("max_rank_shift") or {}).get("shift"),
            "mean_turnover": (perf.get("candidate") or {}).get("mean_turnover"),
        }

    receipt = {
        "schema": "promote_receipt.v1",
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_sha": _git_sha(),
        "old_active_id": old_active_id,
        "new_active_id": new_active_id,
        "candidate_path": candidate_path,
        "promoted_file": promoted_file,
        "gate_summary_path": gate_summary_path,
        "gate_run_url": gate_run_url,
        "gate": gate_summary,
        "forced": forced,
    }

    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return receipt_path


# ---------------------------------------------------------------------------
# Gate artifact auto-fetch
# ---------------------------------------------------------------------------


def _extract_from_artifact(artifact_path: Path, work_dir: Path) -> Path:
    """Extract ruleset_eval.json from a downloaded artifact zip."""
    if not artifact_path.exists():
        raise FileNotFoundError(f"Artifact zip not found: {artifact_path}")
    with zipfile.ZipFile(artifact_path, "r") as zf:
        candidates = [n for n in zf.namelist() if n.endswith("ruleset_eval.json")]
        if not candidates:
            raise FileNotFoundError(
                "No ruleset_eval.json found inside artifact zip."
            )
        # Pick the shortest path match
        best = min(candidates, key=len)
        zf.extract(best, work_dir)
        return work_dir / best


def _fetch_from_run(run_id: str, work_dir: Path) -> Path:
    """Download snapshot artifact from a GitHub Actions run and find ruleset_eval.json."""
    try:
        subprocess.run(
            ["gh", "run", "download", run_id, "-D", str(work_dir)],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(PROJECT_ROOT),
        )
    except FileNotFoundError:
        raise RuntimeError("'gh' CLI not found. Install GitHub CLI to use --gate-run-id.")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"gh run download failed: {e.stderr.strip()}")

    candidates = list(work_dir.rglob("ruleset_eval.json"))
    if not candidates:
        raise FileNotFoundError(
            f"No ruleset_eval.json found in artifacts from run {run_id}."
        )
    return min(candidates, key=lambda p: len(p.parts))


def _resolve_gate_summary(
    gate_summary: Optional[Path],
    gate_run_id: Optional[str],
    gate_artifact: Optional[Path],
) -> Optional[Path]:
    """Resolve the gate summary path from one of three sources.

    Priority: --gate-summary > --gate-artifact > --gate-run-id.
    Returns the resolved Path, or None if no source provided.
    """
    if gate_summary:
        return gate_summary

    if gate_artifact:
        work_dir = Path(tempfile.mkdtemp(prefix="promote_artifact_"))
        return _extract_from_artifact(gate_artifact, work_dir)

    if gate_run_id:
        work_dir = Path(tempfile.mkdtemp(prefix="promote_run_"))
        return _fetch_from_run(gate_run_id, work_dir)

    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Promote a candidate ruleset to active in the manifest."
    )
    parser.add_argument(
        "ruleset_id",
        help="The 8-char ruleset ID to promote.",
    )
    parser.add_argument(
        "--gate-summary", type=Path,
        help="Path to ruleset_eval.json from the evaluator (required unless --force).",
    )
    parser.add_argument(
        "--gate-run-id", default="",
        help="Fetch snapshot artifact from GitHub Actions run and extract ruleset_eval.json.",
    )
    parser.add_argument(
        "--gate-artifact", type=Path, default=None,
        help="Path to a pre-downloaded artifact zip containing ruleset_eval.json.",
    )
    parser.add_argument(
        "--gate-run-url", default="",
        help="URL of the CI gate run (stored in receipt).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would change without writing.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Bypass gate + changelog validation.",
    )
    parser.add_argument(
        "--commit", action="store_true",
        help="Create a git commit with all touched files.",
    )
    parser.add_argument(
        "--rollback", action="store_true",
        help="Rollback: promote a retired ruleset back to active.",
    )
    parser.add_argument(
        "--manifest", type=Path, default=MANIFEST_PATH,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--changelog", type=Path, default=CHANGELOG_PATH,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv)

    # Resolve gate summary from multiple possible sources
    source_count = sum(
        bool(x) for x in [args.gate_summary, args.gate_run_id, args.gate_artifact]
    )
    if source_count > 1:
        print(
            "ERROR: Specify at most one of --gate-summary, --gate-run-id, --gate-artifact.",
            file=sys.stderr,
        )
        return 1

    try:
        resolved = _resolve_gate_summary(
            args.gate_summary, args.gate_run_id or None, args.gate_artifact
        )
    except (FileNotFoundError, RuntimeError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    if resolved is not None:
        args.gate_summary = resolved

    rid = args.ruleset_id
    manifest_path = args.manifest
    changelog_path = args.changelog

    if not manifest_path.exists():
        print("ERROR: manifest.json not found.", file=sys.stderr)
        return 1

    manifest = _load_manifest(manifest_path)

    # Check if already active
    active = _find_active(manifest)
    if active and active["id"] == rid:
        print(f"Ruleset '{rid}' is already active ({active['file']}). Nothing to do.")
        return 0

    old_active_id = active["id"] if active else ""

    # Find target entry
    if args.rollback:
        if not args.force:
            print(
                "ERROR: --rollback requires --force (emergency rollback).",
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
        updated_by = "promote_ruleset.py --rollback"
    else:
        target = _find_by_id_and_status(manifest, rid, "candidate")
        if not target:
            for entry in manifest["rulesets"]:
                if entry["id"] == rid:
                    print(
                        f"ERROR: Ruleset '{rid}' exists but has status "
                        f"'{entry['status']}', not 'candidate'.",
                        file=sys.stderr,
                    )
                    return 1
            print(f"ERROR: No ruleset with ID '{rid}' found in manifest.", file=sys.stderr)
            return 1
        updated_by = "promote_ruleset.py"

    # --- Gate validation ---
    gate_data: Dict[str, Any] = {}
    if not args.force:
        if not args.gate_summary:
            print(
                "ERROR: --gate-summary is required (or use --force to bypass).",
                file=sys.stderr,
            )
            return 1

        gate_ok, gate_msg, gate_data = validate_gate(args.gate_summary, rid)
        if not gate_ok:
            print(f"ERROR: Gate validation failed.\n  {gate_msg}", file=sys.stderr)
            print("Use --force to bypass gate validation.", file=sys.stderr)
            return 1
        print(f"Gate: {gate_msg}")
    else:
        print("WARNING: --force specified, skipping gate validation.", file=sys.stderr)
        if args.gate_summary and args.gate_summary.exists():
            try:
                gate_data = json.loads(args.gate_summary.read_text(encoding="utf-8"))
            except Exception:
                pass

    # --- Changelog validation ---
    if not args.force:
        ok, msg = _validate_changelog(rid, changelog_path)
        if not ok:
            print(f"ERROR: Changelog validation failed.\n  {msg}", file=sys.stderr)
            print("Use --force to skip changelog validation.", file=sys.stderr)
            return 1
        print(f"Changelog: {msg}")

    # --- Verify ruleset file ID matches ---
    candidate_file = RULESETS_DIR / target["file"]
    if candidate_file.exists():
        computed_id = _compute_ruleset_id(candidate_file)
        if computed_id != rid:
            print(
                f"ERROR: File {target['file']} computes to ID '{computed_id}', "
                f"expected '{rid}'.",
                file=sys.stderr,
            )
            return 1

    # Print summary
    print()
    print(f"Promote: {rid} ({target['file']})")
    if active:
        print(f"Retire:  {active['id']} ({active['file']})")
    print()

    if args.dry_run:
        print("[dry-run] Would:")
        print(f"  manifest: {target['file']}: candidate -> active")
        if active:
            print(f"  manifest: {active['file']}: active -> retired")
        print(f"  pin IDs: PHASE2_PINNED_RULESET_ID = \"{rid}\"")
        print(f"  default path: {target['file']}")
        print(f"  receipt: artifacts/promotions/promotion_*_{rid}.json")
        return 0

    # --- Flip statuses ---
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    target["status"] = "active"
    target["updated_by"] = updated_by
    target["updated_at"] = now
    if active:
        active["status"] = "retired"
        active["updated_by"] = updated_by
        active["updated_at"] = now

    _save_manifest(manifest, manifest_path)
    print(f"Updated {manifest_path}")

    # --- Update pinned IDs ---
    if old_active_id:
        updated_files = update_pinned_ids(old_active_id, rid)
        for f in updated_files:
            print(f"  Updated pin: {f}")
    else:
        updated_files = update_pinned_ids("", rid)

    # --- Update default path ---
    path_updated = update_default_path(target["file"])
    if path_updated:
        print(f"  Updated default path: {target['file']}")

    # --- Write receipt ---
    receipt_path = write_receipt(
        old_active_id=old_active_id,
        new_active_id=rid,
        candidate_path=str(candidate_file),
        promoted_file=target["file"],
        gate_data=gate_data,
        gate_summary_path=str(args.gate_summary) if args.gate_summary else "",
        gate_run_url=args.gate_run_url,
        forced=args.force,
    )
    print(f"  Receipt: {receipt_path}")

    # --- Git commit ---
    if args.commit:
        files_to_stage = [str(manifest_path)]
        files_to_stage.extend(updated_files)
        if path_updated:
            files_to_stage.append(str(PROJECT_ROOT / "run_screen.py"))
        if changelog_path.exists():
            files_to_stage.append(str(changelog_path))

        prefix = "FORCE " if args.force else ""
        verdict = (gate_data.get("gate") or {}).get("verdict", "bypass")
        msg = f"feat: {prefix}promote ruleset {rid} to active (gate {verdict})"

        try:
            subprocess.run(
                ["git", "add"] + files_to_stage,
                check=True, cwd=str(PROJECT_ROOT),
            )
            subprocess.run(
                ["git", "commit", "-m", msg],
                check=True, cwd=str(PROJECT_ROOT),
            )
            print(f"\n  Committed: {msg}")
        except subprocess.CalledProcessError as e:
            print(f"WARNING: git commit failed: {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
