#!/usr/bin/env python3
"""Package a deterministic replay bundle for a Phase-2 snapshot.

A replay bundle captures every input needed to reproduce a run_screen run
for a given as_of_date, plus expected invariants for verification.

Bundle structure (replay_bundle_v1/):
    manifest.json
    ruleset.json
    inputs/
        universe.json
        alpha_table.json
        alpha_table.artifact.json     (optional)
        cache_refresh.json            (optional)
        cache_health.json             (optional)
        prices/price_history.csv
        ctgov/trial_records_<date>.json
        sec_8k/8k_catalysts_<date>_<hash>.json
    replay/
        run_args.json
        expected/
            topk_tickers.json
            metadata_fingerprint.json

Usage:
    python3 tools/make_replay_bundle.py --as-of-date 2026-02-26
    python3 tools/make_replay_bundle.py --as-of-date 2026-02-26 --out /tmp/bundle.tgz
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

VERSION = "1.0.0"
SCHEMA_VERSION = "replay_bundle.v1"

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Stable subset of metadata keys used for fingerprint verification.
# Chosen for determinism: they don't depend on wall-clock or absolute paths.
FINGERPRINT_KEYS = [
    "as_of_date",
    "decision_mode",
    "ranking_mode",
    "version",
    "ticker_count",
    "source_type",
    "scoring_model",
    "total_evaluated",
    "active_universe",
    "clinical_excluded",
    "clinical_exempted",
    "archetype_distribution",
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_sha() -> str:
    """Return HEAD git SHA or 'unknown'."""
    try:
        import subprocess

        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=str(_PROJECT_ROOT),
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _stable_json(obj: Any) -> bytes:
    """Deterministic JSON bytes (sorted keys, 2-space indent, trailing newline)."""
    return (json.dumps(obj, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _extract_topk_from_rankings(rankings_csv: Path, top_k: int) -> List[str]:
    """Read rankings.csv and return first top_k tickers by actionable_rank."""
    rows = []
    with open(rankings_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rank_str = row.get("actionable_rank", "")
            if not rank_str:
                continue
            try:
                rank = int(rank_str)
            except (ValueError, TypeError):
                continue
            rows.append((rank, row["ticker"]))
    rows.sort(key=lambda t: t[0])
    return [ticker for _, ticker in rows[:top_k]]


def _find_sec_8k_cache(cache_dir: Path, as_of_date: str) -> Optional[Path]:
    """Find the 8K catalyst cache file for as_of_date."""
    pattern = f"8k_catalysts_{as_of_date}_*.json"
    matches = sorted(cache_dir.glob(pattern))
    return matches[0] if matches else None


def _find_ctgov_cache(cache_dir: Path, as_of_date: str) -> Optional[Path]:
    """Find the CTGov trial records cache file for as_of_date."""
    candidate = cache_dir / f"trial_records_{as_of_date}.json"
    return candidate if candidate.exists() else None


def _find_alpha_table(metadata: Dict, project_root: Path) -> Optional[Path]:
    """Resolve the alpha table path from metadata telemetry."""
    amt = metadata.get("alpha_modifier_telemetry", {})
    table_path_str = amt.get("table_path", "")
    if not table_path_str:
        return None

    # Could be absolute or relative
    candidate = Path(table_path_str)
    if candidate.is_absolute() and candidate.exists():
        return candidate

    # Try relative to project root
    candidate = project_root / table_path_str
    if candidate.exists():
        return candidate

    return None


def make_replay_bundle(
    as_of_date: str,
    snapshot_dir: Path,
    out_path: Path,
    top_k: int = 60,
    project_root: Optional[Path] = None,
) -> Path:
    """Build a replay bundle .tgz for the given as_of_date.

    Returns the path to the created .tgz file.
    Raises FileNotFoundError if required snapshot files are missing.
    """
    root = project_root or _PROJECT_ROOT
    snap_dir = snapshot_dir / as_of_date

    # --- Validate snapshot exists ---
    if not snap_dir.is_dir():
        raise FileNotFoundError(f"Snapshot directory not found: {snap_dir}")

    metadata_path = snap_dir / "metadata.json"
    rankings_path = snap_dir / "rankings.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {snap_dir}")
    if not rankings_path.exists():
        raise FileNotFoundError(f"rankings.csv not found in {snap_dir}")

    metadata = _read_json(metadata_path)

    # --- Collect files to bundle ---
    # Map of: archive_relpath -> (source_path | None, raw_bytes | None)
    # Exactly one of source_path / raw_bytes must be set.
    files: Dict[str, Tuple[Optional[Path], Optional[bytes]]] = {}

    # 1) Ruleset
    ruleset_path = snap_dir / "decision_ruleset.json"
    if ruleset_path.exists():
        files["ruleset.json"] = (ruleset_path, None)
    else:
        raise FileNotFoundError(f"decision_ruleset.json not found in {snap_dir}")

    # 2) Universe
    universe_path = root / "production_data" / "universe.json"
    if universe_path.exists():
        files["inputs/universe.json"] = (universe_path, None)

    # 3) Alpha table
    alpha_table_path = _find_alpha_table(metadata, root)
    if alpha_table_path and alpha_table_path.exists():
        files["inputs/alpha_table.json"] = (alpha_table_path, None)
        # Check for artifact marker
        marker_path = alpha_table_path.with_suffix(".artifact.json")
        if not marker_path.exists():
            marker_path = Path(str(alpha_table_path) + ".artifact.json")
        if marker_path.exists():
            files["inputs/alpha_table.artifact.json"] = (marker_path, None)

    # 4) Cache refresh sidecar
    cache_refresh_path = root / "cache" / f"cache_refresh_{as_of_date}.json"
    if cache_refresh_path.exists():
        files["inputs/cache_refresh.json"] = (cache_refresh_path, None)

    # 5) Cache health from snapshot
    cache_health_path = snap_dir / "cache_health.json"
    if cache_health_path.exists():
        files["inputs/cache_health.json"] = (cache_health_path, None)

    # 6) Price history
    price_path = root / "production_data" / "price_history.csv"
    if price_path.exists():
        files["inputs/prices/price_history.csv"] = (price_path, None)

    # 7) CTGov cache
    ctgov_dir = root / "cache" / "ctgov"
    ctgov_file = _find_ctgov_cache(ctgov_dir, as_of_date)
    if ctgov_file:
        files[f"inputs/ctgov/{ctgov_file.name}"] = (ctgov_file, None)

    # 8) SEC 8-K cache
    sec_dir = root / "cache" / "sec" / "8k_catalysts"
    sec_file = _find_sec_8k_cache(sec_dir, as_of_date)
    if sec_file:
        files[f"inputs/sec_8k/{sec_file.name}"] = (sec_file, None)

    # --- Compute expected invariants ---
    topk_tickers = _extract_topk_from_rankings(rankings_path, top_k)
    ruleset_data = _read_json(ruleset_path)
    expected_ruleset_id = ruleset_data.get("id", metadata.get("clinical_sort_telemetry", {}).get("ruleset_id", ""))

    alpha_table_sha = ""
    if alpha_table_path and alpha_table_path.exists():
        alpha_table_sha = _sha256(alpha_table_path)

    fingerprint_subset = {k: metadata[k] for k in FINGERPRINT_KEYS if k in metadata}
    fingerprint_bytes = _stable_json(fingerprint_subset)
    metadata_fingerprint = _sha256_bytes(fingerprint_bytes)

    # --- Write replay args ---
    run_args = {
        "as_of_date": as_of_date,
        "decision_mode": metadata.get("decision_mode", "phase2"),
        "ruleset": "ruleset.json",
        "snapshot_dir": "REPLAY_OUTPUT",
        "top_k": top_k,
    }
    files["replay/run_args.json"] = (None, _stable_json(run_args))

    # --- Write expected invariants ---
    topk_data = {
        "top_k": top_k,
        "tickers": topk_tickers,
    }
    files["replay/expected/topk_tickers.json"] = (None, _stable_json(topk_data))

    fingerprint_data = {
        "fingerprint_sha256": metadata_fingerprint,
        "keys": FINGERPRINT_KEYS,
        "values": fingerprint_subset,
    }
    files["replay/expected/metadata_fingerprint.json"] = (None, _stable_json(fingerprint_data))

    # --- Build manifest ---
    included_files: List[Dict[str, Any]] = []
    for relpath in sorted(files.keys()):
        src_path, raw = files[relpath]
        if src_path is not None:
            sha = _sha256(src_path)
            size = src_path.stat().st_size
        else:
            assert raw is not None
            sha = _sha256_bytes(raw)
            size = len(raw)
        included_files.append({
            "relpath": relpath,
            "sha256": sha,
            "bytes": size,
        })

    # Extract key telemetry for manifest
    amt = metadata.get("alpha_modifier_telemetry", {})
    cache_health_status = metadata.get("cache_health_overall_status", "unknown")
    cache_degraded = metadata.get("cache_degraded_run", False)
    refresh_had_rejections = metadata.get("cache_refresh_had_rejections", False)
    rejected_sources = metadata.get("cache_refresh_rejected_sources", [])

    manifest = {
        "schema": SCHEMA_VERSION,
        "version": VERSION,
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_sha": _git_sha(),
        "as_of_date": as_of_date,
        "decision_mode": metadata.get("decision_mode", "phase2"),
        "ruleset_id": expected_ruleset_id,
        "ruleset_path_original": amt.get("table_path", ""),
        "snapshot_source_dir": str(snap_dir),
        "included_files": included_files,
        "key_inputs": {
            "alpha_table_source_tag": amt.get("alpha_table_source", "unknown"),
            "cache_health_overall_status": cache_health_status,
            "cache_degraded_run": cache_degraded,
            "cache_refresh_had_rejections": refresh_had_rejections,
            "cache_refresh_rejected_sources": rejected_sources,
        },
        "replay_invariants": {
            "expected_topk": topk_tickers,
            "expected_ruleset_id": expected_ruleset_id,
            "expected_alpha_table_sha256": alpha_table_sha,
            "expected_metadata_fingerprint_sha256": metadata_fingerprint,
        },
    }
    files["manifest.json"] = (None, _stable_json(manifest))

    # Recompute included_files with manifest itself
    manifest_bytes = files["manifest.json"][1]

    # --- Write .tgz ---
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(str(out_path), "w:gz") as tar:
        for relpath in sorted(files.keys()):
            src_path, raw = files[relpath]
            arcname = f"replay_bundle_v1/{relpath}"

            if src_path is not None:
                tar.add(str(src_path), arcname=arcname)
            else:
                assert raw is not None
                info = tarfile.TarInfo(name=arcname)
                info.size = len(raw)
                info.mtime = int(time.time())
                tar.addfile(info, io.BytesIO(raw))

    return out_path


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Package a replay bundle for a Phase-2 snapshot."
    )
    parser.add_argument("--as-of-date", required=True, help="YYYY-MM-DD")
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=_PROJECT_ROOT / "data" / "snapshots",
        help="Root of snapshot directories (default: data/snapshots)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output .tgz path (default: artifacts/replay_bundle_<date>.tgz)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=60,
        help="Number of top tickers to capture as invariant (default: 60)",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Override project root (default: auto-detect)",
    )
    args = parser.parse_args(argv)

    out_path = args.out
    if out_path is None:
        out_path = _PROJECT_ROOT / "artifacts" / f"replay_bundle_{args.as_of_date}.tgz"

    try:
        result = make_replay_bundle(
            as_of_date=args.as_of_date,
            snapshot_dir=args.snapshot_dir,
            out_path=out_path,
            top_k=args.top_k,
            project_root=args.project_root,
        )
        print(f"Bundle created: {result}")
        print(f"  Size: {result.stat().st_size:,} bytes")

        # Quick summary
        with tarfile.open(str(result), "r:gz") as tar:
            members = tar.getnames()
        print(f"  Files: {len(members)}")
        for m in sorted(members):
            print(f"    {m}")
        return 0

    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
