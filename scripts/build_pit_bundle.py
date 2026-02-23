#!/usr/bin/env python3
"""Build PIT feature bundles combining clinical, catalyst, and coinvest.

Orchestrates all three PIT feature builders into per-date bundle directories
with a manifest for integrity verification.

Bundle Structure:
    data/bundles/PIT/{as_of_date}/
      manifest.json
      clinical_features.json
      catalyst_events.json
      coinvest_features.json

Usage:
    # Single date
    python3 scripts/build_pit_bundle.py \\
        --as-of-date 2026-01-15 \\
        --bundle-root data/bundles/PIT \\
        --universe production_data/universe.json

    # Batch: all dates with 13F caches
    python3 scripts/build_pit_bundle.py \\
        --batch \\
        --bundle-root data/bundles/PIT \\
        --universe production_data/universe.json

    # Validate existing bundle
    python3 scripts/build_pit_bundle.py \\
        --validate --bundle-root data/bundles/PIT/2026-01-15
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

VERSION = "1.0.0"
SCHEMA_VERSION = "pit_bundle_manifest.v1"

_DATE_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_PROJECT_ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> Optional[Any]:
    """Graceful JSON load; returns None on any error."""
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError, TypeError):
        return None


def _load_universe(path: Path) -> Set[str]:
    """Load universe tickers from JSON (list-of-dicts or list-of-strings)."""
    data = _load_json(path)
    if data is None:
        return set()
    tickers: Set[str] = set()
    if isinstance(data, list):
        for item in data:
            if isinstance(item, str):
                tickers.add(item.upper())
            elif isinstance(item, dict):
                tk = item.get("ticker", "")
                if tk:
                    tickers.add(tk.upper())
    elif isinstance(data, dict):
        tickers = {k.upper() for k in data.keys()}
    tickers.discard("")
    return tickers


def _sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Bundle building
# ---------------------------------------------------------------------------

def build_single_bundle(
    as_of_date: str,
    bundle_root: Path,
    universe_tickers: Set[str],
    trial_records_path: Optional[Path] = None,
    cache_13f_root: Optional[Path] = None,
    skip_clinical: bool = False,
    skip_catalyst: bool = False,
    skip_coinvest: bool = False,
    ledger_config: Optional[Any] = None,
    pit_mode: str = "strict",
) -> dict:
    """Build one date's PIT feature bundle.

    Args:
        as_of_date: ISO date string.
        bundle_root: Root directory for bundles (e.g. data/bundles/PIT).
        universe_tickers: Set of ticker strings.
        trial_records_path: Path to trial_records.json.
        cache_13f_root: Root for 13F caches.
        skip_clinical: Skip clinical features.
        skip_catalyst: Skip catalyst features.
        skip_coinvest: Skip coinvest features.
        ledger_config: LedgerConfig for catalyst builder.
        pit_mode: "strict" or "degrade". In strict mode, missing components
            are recorded as errors. In degrade, they are skipped gracefully.

    Returns:
        Manifest dict.
    """
    t0 = time.monotonic()
    bundle_dir = bundle_root / as_of_date
    bundle_dir.mkdir(parents=True, exist_ok=True)

    components: Dict[str, dict] = {}
    pit_violations: Dict[str, int] = {}

    # 1. Clinical features
    if not skip_clinical:
        try:
            from scripts.build_clinical_features_pit import (
                build_clinical_features,
                validate_clinical_features_schema,
            )
            if trial_records_path is None:
                trial_records_path = _PROJECT_ROOT / "production_data" / "trial_records.json"
            trial_data = _load_json(trial_records_path)
            if isinstance(trial_data, list):
                clin = build_clinical_features(
                    as_of_date=as_of_date,
                    trial_records=trial_data,
                    universe_tickers=universe_tickers,
                )
                clin["trial_records_source"] = str(trial_records_path)
                clin["provenance"]["trial_records_file"] = trial_records_path.name

                out_path = bundle_dir / "clinical_features.json"
                with open(out_path, "w") as f:
                    json.dump(clin, f, indent=2)

                components["clinical_features"] = {
                    "file": "clinical_features.json",
                    "schema_version": clin["schema_version"],
                    "sha256": _sha256_file(out_path),
                    "coverage_pct": clin["signal_coverage_pct"],
                }

                # PIT validation on the trial input
                try:
                    from scripts.validate_pit_inputs import validate_ctgov_pit
                    vr = validate_ctgov_pit(
                        trial_data, as_of_date, pit_mode,
                        max_examples=999_999,
                    )
                    pit_violations["ctgov_first_posted"] = vr["first_posted_violations"]
                except Exception:
                    pass
            else:
                print(f"  SKIP clinical: trial_records not a list", file=sys.stderr)
        except Exception as e:
            print(f"  SKIP clinical: {e}", file=sys.stderr)

    # 2. Catalyst events
    if not skip_catalyst:
        try:
            from scripts.build_catalyst_events_pit import (
                build_catalyst_events,
                validate_catalyst_events_schema,
            )
            cat = build_catalyst_events(
                as_of_date=as_of_date,
                universe_tickers=universe_tickers,
                config=ledger_config,
            )

            out_path = bundle_dir / "catalyst_events.json"
            with open(out_path, "w") as f:
                json.dump(cat, f, indent=2)

            components["catalyst_events"] = {
                "file": "catalyst_events.json",
                "schema_version": cat["schema_version"],
                "sha256": _sha256_file(out_path),
                "coverage_pct": cat["catalyst_coverage_pct"],
            }

            # PIT validation on catalyst events
            try:
                from scripts.validate_pit_inputs import validate_catalyst_pit
                vr = validate_catalyst_pit(cat, as_of_date, pit_mode)
                pit_violations["catalyst_missing_disclosed_at"] = vr["missing_disclosed_at"]
                pit_violations["catalyst_future_disclosed_at"] = vr["future_disclosed_at"]
            except Exception:
                pass
        except Exception as e:
            print(f"  SKIP catalyst: {e}", file=sys.stderr)

    # 3. Coinvest features
    if not skip_coinvest:
        try:
            from scripts.build_coinvest_features_from_13f import (
                build_coinvest_features,
                validate_coinvest_features_schema,
                _load_cusip_map,
                _pad_cik,
            )
            from elite_managers import get_all_managers

            if cache_13f_root is None:
                cache_13f_root = _PROJECT_ROOT / "data" / "caches" / "sec_13f" / "PIT"

            cache_date_dir = cache_13f_root / as_of_date
            if cache_date_dir.exists():
                # Build CIK -> tier/name maps from elite_managers
                manager_tiers: Dict[str, int] = {}
                manager_names: Dict[str, str] = {}
                for m in get_all_managers():
                    padded = _pad_cik(m["cik"])
                    manager_tiers[padded] = m.get("tier", 0)
                    manager_names[padded] = m.get("short_name", m.get("name", padded))

                cusip_map = _load_cusip_map(
                    _PROJECT_ROOT / "production_data" / "cusip_static_map.json"
                )

                coinv = build_coinvest_features(
                    as_of_date=as_of_date,
                    cache_root=cache_13f_root,
                    universe_tickers=universe_tickers,
                    manager_tiers=manager_tiers,
                    manager_names=manager_names,
                    cusip_map=cusip_map,
                )

                if coinv is not None:
                    out_path = bundle_dir / "coinvest_features.json"
                    with open(out_path, "w") as f:
                        json.dump(coinv, f, indent=2)

                    components["coinvest_features"] = {
                        "file": "coinvest_features.json",
                        "schema_version": coinv["schema_version"],
                        "sha256": _sha256_file(out_path),
                        "coverage_pct": coinv["signal_coverage_pct"],
                    }
                else:
                    print(f"  SKIP coinvest: build returned None", file=sys.stderr)
            else:
                print(f"  SKIP coinvest: no cache at {cache_date_dir}", file=sys.stderr)
        except Exception as e:
            print(f"  SKIP coinvest: {e}", file=sys.stderr)

    elapsed = time.monotonic() - t0

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "version": VERSION,
        "as_of_date": as_of_date,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "bundle_root": str(bundle_root),
        "pit_mode": pit_mode,
        "components": components,
        "pit_violations": pit_violations,
        "universe_size": len(universe_tickers),
        "build_duration_seconds": round(elapsed, 2),
    }

    manifest_path = bundle_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def validate_bundle(bundle_dir: Path) -> Tuple[bool, str]:
    """Verify manifest hashes and schemas for a bundle directory.

    Returns (True, "OK") on success, (False, reason) on failure.
    """
    manifest_path = bundle_dir / "manifest.json"
    manifest = _load_json(manifest_path)
    if manifest is None:
        return False, f"cannot load manifest from {manifest_path}"

    if manifest.get("schema_version") != SCHEMA_VERSION:
        return False, (
            f"expected manifest schema {SCHEMA_VERSION}, "
            f"got {manifest.get('schema_version')}"
        )

    components = manifest.get("components", {})
    if not components:
        return False, "no components in manifest"

    for name, info in components.items():
        fpath = bundle_dir / info["file"]
        if not fpath.exists():
            return False, f"missing component file: {fpath}"
        actual_hash = _sha256_file(fpath)
        if actual_hash != info["sha256"]:
            return False, (
                f"hash mismatch for {name}: expected {info['sha256']}, "
                f"got {actual_hash}"
            )

        # Validate component schema
        data = _load_json(fpath)
        if data is None:
            return False, f"cannot parse {fpath}"
        if data.get("schema_version") != info["schema_version"]:
            return False, (
                f"schema mismatch for {name}: expected {info['schema_version']}, "
                f"got {data.get('schema_version')}"
            )

    return True, "OK"


def discover_buildable_dates(
    cache_13f_root: Optional[Path] = None,
) -> List[str]:
    """Find dates with at least one cache available.

    Scans 13F cache dirs for date-named subdirectories.
    """
    if cache_13f_root is None:
        cache_13f_root = _PROJECT_ROOT / "data" / "caches" / "sec_13f" / "PIT"

    dates: List[str] = []
    if cache_13f_root.exists():
        for d in sorted(cache_13f_root.iterdir()):
            if d.is_dir() and _DATE_DIR_RE.match(d.name):
                dates.append(d.name)
    return dates


def build_batch_bundles(
    date_list: List[str],
    bundle_root: Path,
    universe_tickers: Set[str],
    **kwargs,
) -> List[dict]:
    """Build bundles for multiple dates."""
    results = []
    for as_of_date in date_list:
        print(f"\n--- Building bundle for {as_of_date} ---")
        try:
            manifest = build_single_bundle(
                as_of_date=as_of_date,
                bundle_root=bundle_root,
                universe_tickers=universe_tickers,
                **kwargs,
            )
            n_components = len(manifest["components"])
            if n_components > 0:
                results.append(manifest)
                print(f"  {n_components} components, "
                      f"{manifest['build_duration_seconds']}s")
            else:
                print(f"  SKIP: no components built")
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build PIT feature bundles",
    )
    parser.add_argument(
        "--as-of-date", default=None, help="Single date (ISO format)",
    )
    parser.add_argument(
        "--batch", action="store_true",
        help="Build bundles for all discoverable dates",
    )
    parser.add_argument(
        "--bundle-root", default="data/bundles/PIT",
        help="Bundle output root directory",
    )
    parser.add_argument(
        "--universe", default="production_data/universe.json",
        help="Universe JSON path",
    )
    parser.add_argument(
        "--trial-records", default=None,
        help="Path to trial_records.json (default: production_data/trial_records.json)",
    )
    parser.add_argument(
        "--13f-cache-root", dest="cache_13f_root", default=None,
        help="13F cache root (default: data/caches/sec_13f/PIT)",
    )
    parser.add_argument(
        "--skip-clinical", action="store_true",
        help="Skip clinical features",
    )
    parser.add_argument(
        "--skip-catalyst", action="store_true",
        help="Skip catalyst features",
    )
    parser.add_argument(
        "--skip-coinvest", action="store_true",
        help="Skip coinvest features",
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Validate an existing bundle instead of building",
    )
    parser.add_argument(
        "--pit-mode", default="strict",
        choices=["strict", "degrade"],
        help="PIT mode: strict (default) or degrade",
    )
    args = parser.parse_args(argv)

    # Import project modules (add project root to path).
    sys.path.insert(0, str(_PROJECT_ROOT))

    # Resolve paths
    bundle_root = Path(args.bundle_root)
    if not bundle_root.is_absolute():
        bundle_root = _PROJECT_ROOT / bundle_root

    uni_path = Path(args.universe)
    if not uni_path.is_absolute():
        uni_path = _PROJECT_ROOT / uni_path

    trial_path = None
    if args.trial_records:
        trial_path = Path(args.trial_records)
        if not trial_path.is_absolute():
            trial_path = _PROJECT_ROOT / trial_path

    cache_root = None
    if args.cache_13f_root:
        cache_root = Path(args.cache_13f_root)
        if not cache_root.is_absolute():
            cache_root = _PROJECT_ROOT / cache_root

    # Validate mode
    if args.validate:
        ok, reason = validate_bundle(bundle_root)
        if ok:
            print(f"PASS: {bundle_root}")
            return 0
        else:
            print(f"FAIL: {reason}")
            return 1

    # Build mode
    universe = _load_universe(uni_path)
    if not universe:
        print(f"ERROR: empty universe from {uni_path}", file=sys.stderr)
        return 1

    print(f"Universe: {len(universe)} tickers")

    if args.batch:
        dates = discover_buildable_dates(cache_root)
        if not dates:
            print("No buildable dates found", file=sys.stderr)
            return 1
        print(f"Discovered {len(dates)} dates")
        results = build_batch_bundles(
            dates, bundle_root, universe,
            trial_records_path=trial_path,
            cache_13f_root=cache_root,
            skip_clinical=args.skip_clinical,
            skip_catalyst=args.skip_catalyst,
            skip_coinvest=args.skip_coinvest,
            pit_mode=args.pit_mode,
        )
        print(f"\nBuilt {len(results)} bundles")
        return 0

    elif args.as_of_date:
        manifest = build_single_bundle(
            as_of_date=args.as_of_date,
            bundle_root=bundle_root,
            universe_tickers=universe,
            trial_records_path=trial_path,
            cache_13f_root=cache_root,
            skip_clinical=args.skip_clinical,
            skip_catalyst=args.skip_catalyst,
            skip_coinvest=args.skip_coinvest,
            pit_mode=args.pit_mode,
        )
        n = len(manifest["components"])
        print(f"\nBundle: {n} components, "
              f"{manifest['build_duration_seconds']}s")
        for name, info in manifest["components"].items():
            print(f"  {name}: {info['coverage_pct']}% coverage")
        return 0

    else:
        print("ERROR: specify --as-of-date or --batch", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
