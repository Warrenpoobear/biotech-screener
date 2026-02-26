#!/usr/bin/env python3
"""Replay a bundle and verify invariants against expected outputs.

Extracts a replay bundle .tgz, optionally re-runs run_screen.py, and
verifies that key invariants (ruleset ID, top-K tickers, alpha table hash,
metadata fingerprint) match the expected values stored in the bundle.

Usage:
    # Verify invariants only (no re-run):
    python3 tools/replay_bundle.py --bundle artifacts/replay_bundle_2026-02-26.tgz

    # Full replay + verification:
    python3 tools/replay_bundle.py --bundle artifacts/replay_bundle_2026-02-26.tgz --run

Exit codes:
    0 = all invariants match (or --run succeeded + matched)
    1 = invariant mismatch
    2 = execution error
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

VERSION = "1.0.0"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _stable_json(obj: Any) -> bytes:
    return (json.dumps(obj, sort_keys=True, indent=2) + "\n").encode("utf-8")


def verify_bundle_integrity(bundle_dir: Path) -> List[str]:
    """Verify SHA-256 hashes of all included files against manifest.

    Returns list of error messages (empty = all good).
    """
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.exists():
        return ["manifest.json not found in bundle"]

    manifest = _read_json(manifest_path)
    errors = []

    for entry in manifest.get("included_files", []):
        relpath = entry["relpath"]
        expected_sha = entry["sha256"]
        file_path = bundle_dir / relpath

        if not file_path.exists():
            errors.append(f"Missing file: {relpath}")
            continue

        actual_sha = _sha256(file_path)
        if actual_sha != expected_sha:
            errors.append(
                f"SHA mismatch for {relpath}: "
                f"expected {expected_sha[:16]}..., got {actual_sha[:16]}..."
            )

    return errors


def verify_invariants(
    bundle_dir: Path,
    replay_snapshot_dir: Optional[Path] = None,
    top_k: int = 60,
) -> Dict[str, Any]:
    """Verify replay invariants from the bundle.

    If replay_snapshot_dir is provided, compares actual replay output
    against expected values. Otherwise, just validates the bundle's
    internal consistency.

    Returns dict with:
        passed: bool
        checks: list of {name, passed, expected, actual, detail}
    """
    manifest = _read_json(bundle_dir / "manifest.json")
    invariants = manifest.get("replay_invariants", {})

    checks = []

    # 1) Schema version check
    schema = manifest.get("schema", "")
    checks.append({
        "name": "schema_version",
        "passed": schema == "replay_bundle.v1",
        "expected": "replay_bundle.v1",
        "actual": schema,
    })

    # 2) Ruleset ID present
    expected_ruleset_id = invariants.get("expected_ruleset_id", "")
    checks.append({
        "name": "ruleset_id_present",
        "passed": bool(expected_ruleset_id),
        "expected": "non-empty",
        "actual": expected_ruleset_id or "(empty)",
    })

    # 3) Top-K tickers file exists and is valid
    topk_path = bundle_dir / "replay" / "expected" / "topk_tickers.json"
    if topk_path.exists():
        topk_data = _read_json(topk_path)
        tickers = topk_data.get("tickers", [])
        checks.append({
            "name": "topk_tickers_present",
            "passed": len(tickers) > 0,
            "expected": f">0 tickers (top_k={topk_data.get('top_k', '?')})",
            "actual": f"{len(tickers)} tickers",
        })
    else:
        checks.append({
            "name": "topk_tickers_present",
            "passed": False,
            "expected": "topk_tickers.json exists",
            "actual": "missing",
        })

    # 4) Alpha table SHA (if expected)
    expected_alpha_sha = invariants.get("expected_alpha_table_sha256", "")
    alpha_path = bundle_dir / "inputs" / "alpha_table.json"
    if expected_alpha_sha and alpha_path.exists():
        actual_alpha_sha = _sha256(alpha_path)
        checks.append({
            "name": "alpha_table_sha256",
            "passed": actual_alpha_sha == expected_alpha_sha,
            "expected": expected_alpha_sha[:16] + "...",
            "actual": actual_alpha_sha[:16] + "...",
        })

    # 5) Metadata fingerprint (if expected)
    expected_fp = invariants.get("expected_metadata_fingerprint_sha256", "")
    fp_path = bundle_dir / "replay" / "expected" / "metadata_fingerprint.json"
    if expected_fp and fp_path.exists():
        fp_data = _read_json(fp_path)
        stored_fp = fp_data.get("fingerprint_sha256", "")
        checks.append({
            "name": "metadata_fingerprint_consistent",
            "passed": stored_fp == expected_fp,
            "expected": expected_fp[:16] + "...",
            "actual": stored_fp[:16] + "...",
        })

    # 6) File integrity
    integrity_errors = verify_bundle_integrity(bundle_dir)
    checks.append({
        "name": "file_integrity",
        "passed": len(integrity_errors) == 0,
        "expected": "all files intact",
        "actual": f"{len(integrity_errors)} errors" if integrity_errors else "all intact",
        "detail": integrity_errors[:5] if integrity_errors else None,
    })

    # 7) If replay_snapshot_dir provided, compare actual vs expected
    if replay_snapshot_dir is not None:
        replay_meta_path = replay_snapshot_dir / "metadata.json"
        if replay_meta_path.exists():
            replay_meta = _read_json(replay_meta_path)

            # Check ruleset ID
            replay_ruleset_id = replay_meta.get("clinical_sort_telemetry", {}).get("ruleset_id", "")
            checks.append({
                "name": "replay_ruleset_id",
                "passed": replay_ruleset_id == expected_ruleset_id,
                "expected": expected_ruleset_id,
                "actual": replay_ruleset_id,
            })

            # Check metadata fingerprint
            from tools.make_replay_bundle import FINGERPRINT_KEYS, _sha256_bytes

            fp_subset = {k: replay_meta[k] for k in FINGERPRINT_KEYS if k in replay_meta}
            actual_fp = hashlib.sha256(_stable_json(fp_subset)).hexdigest()
            checks.append({
                "name": "replay_metadata_fingerprint",
                "passed": actual_fp == expected_fp,
                "expected": expected_fp[:16] + "...",
                "actual": actual_fp[:16] + "...",
            })

    all_passed = all(c["passed"] for c in checks)
    return {"passed": all_passed, "checks": checks}


def extract_bundle(tgz_path: Path, workdir: Path) -> Path:
    """Extract a replay bundle .tgz to workdir.

    Returns path to the extracted replay_bundle_v1/ directory.
    """
    with tarfile.open(str(tgz_path), "r:gz") as tar:
        tar.extractall(path=str(workdir))

    bundle_dir = workdir / "replay_bundle_v1"
    if not bundle_dir.is_dir():
        # Check if files are at workdir root
        if (workdir / "manifest.json").exists():
            return workdir
        raise FileNotFoundError(
            f"Expected replay_bundle_v1/ directory in {workdir}, "
            f"found: {[p.name for p in workdir.iterdir()]}"
        )
    return bundle_dir


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Replay a bundle and verify invariants."
    )
    parser.add_argument("--bundle", required=True, type=Path, help="Path to .tgz bundle")
    parser.add_argument(
        "--workdir",
        type=Path,
        default=None,
        help="Working directory for extraction (default: temp dir)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=60,
        help="Top-K for invariant verification (default: 60)",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Actually re-run run_screen.py (not yet implemented)",
    )
    args = parser.parse_args(argv)

    if not args.bundle.exists():
        print(f"ERROR: Bundle not found: {args.bundle}", file=sys.stderr)
        return 2

    use_tempdir = args.workdir is None
    workdir = args.workdir or Path(tempfile.mkdtemp(prefix="replay_"))

    try:
        workdir.mkdir(parents=True, exist_ok=True)
        print(f"Extracting bundle to {workdir}...")
        bundle_dir = extract_bundle(args.bundle, workdir)

        manifest = _read_json(bundle_dir / "manifest.json")
        print(f"  Schema: {manifest.get('schema')}")
        print(f"  Date: {manifest.get('as_of_date')}")
        print(f"  Git SHA: {manifest.get('git_sha', 'unknown')[:12]}")
        print(f"  Files: {len(manifest.get('included_files', []))}")

        if args.run:
            print("\n--run is not yet implemented. Verifying invariants only.\n")

        print("Verifying invariants...")
        result = verify_invariants(bundle_dir, top_k=args.top_k)

        for check in result["checks"]:
            status = "PASS" if check["passed"] else "FAIL"
            print(f"  [{status}] {check['name']}: expected={check.get('expected', '?')}, actual={check.get('actual', '?')}")
            if check.get("detail"):
                for d in check["detail"]:
                    print(f"         {d}")

        if result["passed"]:
            print("\nAll invariants verified.")
            return 0
        else:
            failed = [c for c in result["checks"] if not c["passed"]]
            print(f"\n{len(failed)} invariant(s) failed.")
            return 1

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    finally:
        if use_tempdir and workdir.exists():
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
