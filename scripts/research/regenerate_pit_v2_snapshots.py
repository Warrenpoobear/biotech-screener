#!/usr/bin/env python3
"""Regenerate historical snapshots with PIT financials (pseudo-PIT v2).

Runs run_screen.py for each monthly snapshot date with --pit-mode degrade,
writing to data/snapshots_pit_v2/. Only runs dates not already present
in the output directory (idempotent).

Since 2026-11-13 three protections are on by default:

* --pit-market-data: each date gets its own market_data.json built from prices and
  SEC share counts available on that date (common/pit_market_data.py). Without it
  run_screen.py reads today's market_data.json, so every historical date sees today's
  market cap, price, volume and volatility. Histories generated before this date
  (snapshots_pit_v2 and earlier) carry that look-ahead.
* --isolate-side-outputs: Module 3 side outputs (catalyst_events, run_log,
  ctgov_state) go to <out-dir>/_side_outputs/<date>/ instead of the data dir.
* --protect-live-state: files a historical run would otherwise rewrite in the
  checkout (data/state, the expression logs, readiness history, "latest" pointers)
  are restored after each date, and the run refuses to start while a daily
  production run holds its lock.

Usage:
    python scripts/research/regenerate_pit_v2_snapshots.py
    python scripts/research/regenerate_pit_v2_snapshots.py --start 2023-01-01
    python scripts/research/regenerate_pit_v2_snapshots.py --dry-run
    python scripts/research/regenerate_pit_v2_snapshots.py --max-dates 5
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SNAPSHOTS_DIR = PROJECT_ROOT / "data" / "snapshots"
PIT_V2_DIR = PROJECT_ROOT / "data" / "snapshots_pit_v2"
PROD_DATA = PROJECT_ROOT / "production_data"
BUNDLES_DIR = PROJECT_ROOT / "data" / "bundles" / "PIT"
PIT_13F_CACHE = PROJECT_ROOT / "data" / "caches" / "sec_13f" / "PIT"
STAGING_ROOT = PROJECT_ROOT / "data" / "staging" / "pit_regen"
LOG_PATH = PROJECT_ROOT / "output" / "pit" / "regeneration_log.json"
SIDE_OUTPUTS_DIRNAME = "_side_outputs"

# Files and directories that run_screen.py rewrites relative to the checkout on every
# run, including historical ones. A regeneration must leave them as it found them.
PROTECTED_LIVE_STATE: tuple[str, ...] = (
    "data/state",
    "data/expression_decision_log.jsonl",
    "data/expression_attribution_log.jsonl",
    "artifacts/readiness/history.jsonl",
    "artifacts/bioshort_watch/latest_status.json",
    "artifacts/policy_shadow/tier_weighted/history.jsonl",
)
PRODUCTION_LOCKS: tuple[str, ...] = ("logs/.daily_production.lock", "logs/.daily_production_py.lock")

# Minimum 13F manager coverage required to consider a date PIT-stageable.
# Below this, the cache isn't representative enough to improve over the
# current-holdings fallback — label as contaminated instead.
PIT_STAGE_MIN_COVERAGE_PCT = 50.0

# Nearest-prior lookback window for non-quarter-end monthly dates.
# Covers up to one quarter of lag (roughly 95 days).
PIT_STAGE_NEAREST_PRIOR_DAYS = 95


def get_monthly_dates(start: str = "2020-01-01") -> list[str]:
    """Get one snapshot date per month (last available per month)."""
    by_month: dict[str, str] = {}
    for d in sorted(SNAPSHOTS_DIR.iterdir()):
        if not d.is_dir() or not (d / "rankings.csv").exists():
            continue
        name = d.name
        if "__" in name:  # skip staging dirs like 2026-04-02__pre_*
            continue
        if name < start:
            continue
        by_month[name[:7]] = name
    return sorted(by_month.values())


def already_done(date_str: str) -> bool:
    """Check if PIT v2 snapshot already exists for this date."""
    return (PIT_V2_DIR / date_str / "rankings.csv").exists()


def _resolve_data_dir(date_str: str) -> tuple[Path, str]:
    """Resolve the best data directory for a given date.

    Prefers archived PIT inputs from the production snapshot if available,
    falls back to current production_data/.
    Returns (data_dir, source_tag).
    """
    archived = SNAPSHOTS_DIR / date_str / "inputs"
    # The archive is usable only if run_screen.py's preflight would pass on it.
    # Archives written from 2026-04-20 hold universe.json but no financial_records.json.
    if archived.exists() and all((archived / name).exists() for name in ("universe.json", "financial_records.json")):
        return archived, "archived"
    return PROD_DATA, "current"


def _resolve_institutional_source(date_str: str, data_dir: Path) -> str:
    """Report where institutional (13F) features will come from for this date.

    Diagnostic only — does not change behavior. Return values:
      - "pit_13f_staged" : data_dir is a staging dir with a PIT-derived coinvest_signals.json overlay
      - "archived"       : {data_dir}/coinvest_signals.json or holdings_detailed.json is the archived snapshot input
      - "bundle"         : a PIT bundle exists for this date (Option A)
      - "bundle_nearby"  : a bundle exists within 95 days prior (Option B with lag)
      - "contaminated"   : will fall back to current production_data/holdings_detailed.json (pseudo-PIT)
    """
    # If data_dir is a Option B-lite staging dir, prefer that label.
    # Note: Phase 5 distinguishes exact vs prior via data_source, not here.
    try:
        if data_dir.is_relative_to(STAGING_ROOT):
            return "pit_13f_staged"
    except AttributeError:
        if str(data_dir).startswith(str(STAGING_ROOT)):
            return "pit_13f_staged"
    if (data_dir / "coinvest_signals.json").exists() or (data_dir / "holdings_detailed.json").exists():
        if data_dir != PROD_DATA:
            return "archived"
    if (BUNDLES_DIR / date_str / "manifest.json").exists():
        return "bundle"
    # Look for a prior bundle within 95 days
    try:
        from datetime import date, timedelta

        target = date.fromisoformat(date_str)
        for delta in range(1, 96):
            candidate = (target - timedelta(days=delta)).isoformat()
            if (BUNDLES_DIR / candidate / "manifest.json").exists():
                return "bundle_nearby"
    except (ValueError, OSError):
        pass
    return "contaminated"


def _bundle_dir_for(date_str: str) -> Path | None:
    """Return the bundle dir for an exact date match, else None."""
    d = BUNDLES_DIR / date_str
    if (d / "manifest.json").exists():
        return d
    return None


def _pit_13f_cache_coverage_at(cache_dir: Path) -> float | None:
    """Return coverage_pct of the PIT 13F cache at a specific dir, or None if invalid."""
    idx = cache_dir / "index.json"
    if not idx.exists():
        return None
    try:
        data = json.load(open(idx))
    except (json.JSONDecodeError, OSError):
        return None
    if data.get("schema_version") != "sec_13f_pit_index.v1":
        return None
    return float(data.get("coverage_pct", 0) or 0)


def _pit_13f_cache_coverage(date_str: str) -> float | None:
    """Return coverage_pct for an exact-match cache date, or None if missing."""
    return _pit_13f_cache_coverage_at(PIT_13F_CACHE / date_str)


def production_lock_held(root: Path | None = None) -> str | None:
    """Name of a daily-production lock file that currently exists, else None."""
    base = root or PROJECT_ROOT
    for rel in PRODUCTION_LOCKS:
        if (base / rel).exists():
            return rel
    return None


def _file_digest(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class LiveStateGuard:
    """Restore protected checkout files to their pre-run content.

    On entry every protected file is copied aside. On exit a file whose content
    changed is put back atomically, and a file that did not exist before is removed.
    `restored` lists what had to be repaired.
    """

    def __init__(self, root: Path | None = None, protected: tuple[str, ...] = PROTECTED_LIVE_STATE):
        self.root = root or PROJECT_ROOT
        self.protected = protected
        self.restored: list[str] = []
        self._backup_dir: Path | None = None
        self._before: dict[str, str] = {}

    def _files(self) -> list[Path]:
        out: list[Path] = []
        for rel in self.protected:
            target = self.root / rel
            if target.is_dir():
                out.extend(sorted(p for p in target.rglob("*") if p.is_file()))
            elif target.is_file():
                out.append(target)
        return out

    def __enter__(self) -> "LiveStateGuard":
        import shutil
        import tempfile

        self._backup_dir = Path(tempfile.mkdtemp(prefix="pit_regen_live_state_"))
        for path in self._files():
            rel = str(path.relative_to(self.root))
            dest = self._backup_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
            self._before[rel] = _file_digest(path)
        return self

    def __exit__(self, *exc) -> None:
        import os
        import shutil

        assert self._backup_dir is not None
        try:
            for path in self._files():
                rel = str(path.relative_to(self.root))
                if rel not in self._before:
                    path.unlink()  # created by the historical run
                    self.restored.append(rel)
                elif _file_digest(path) != self._before[rel]:
                    tmp = path.with_name(path.name + ".pit_regen_restore")
                    shutil.copy2(self._backup_dir / rel, tmp)
                    os.replace(tmp, path)
                    self.restored.append(rel)
            for rel in self._before:
                path = self.root / rel
                if not path.exists():  # deleted by the historical run
                    path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(self._backup_dir / rel, path)
                    self.restored.append(rel)
        finally:
            shutil.rmtree(self._backup_dir, ignore_errors=True)


def _shadow_tree(src: Path, dst: Path) -> None:
    """Hardlink `src` into `dst` (recursing into directories); never overwrites."""
    import os
    import shutil

    if dst.exists() or dst.is_symlink():
        return
    if src.is_dir():
        dst.mkdir(parents=True, exist_ok=True)
        for child in src.iterdir():
            _shadow_tree(child, dst / child.name)
        return
    try:
        os.link(str(src), str(dst))
    except OSError:
        if src.is_file() and src.stat().st_size < 50 * 1024 * 1024:
            shutil.copy2(str(src), str(dst))


_MARKET_BUILDERS: dict[str, object] = {}


def stage_pit_market_data(date_str: str, base_data_dir: Path) -> tuple[Path, dict]:
    """Write a point-in-time market_data.json into this date's staging dir.

    Built from the current production_data (prices, SEC share-count facts), never from
    an archived per-date inputs dir: those hold no price history. Returns the staging
    dir and the count of market-cap sources used.
    """
    sys.path.insert(0, str(PROJECT_ROOT))
    from common.pit_market_data import PitMarketDataBuilder

    key = str(PROD_DATA)
    if key not in _MARKET_BUILDERS:
        _MARKET_BUILDERS[key] = PitMarketDataBuilder(PROD_DATA)
    staging = STAGING_ROOT / date_str
    counts = _MARKET_BUILDERS[key].write(date_str, staging / "market_data.json")  # type: ignore[attr-defined]
    return staging, counts


def stage_data_dir_with_pit_institutional(
    date_str: str,
    base_data_dir: Path,
) -> tuple[Path, str] | tuple[None, str]:
    """Materialize a per-date staging data_dir with PIT-correct institutional features.

    Reuses `scripts/build_coinvest_features_from_13f.py` (no new conversion logic).
    Output staging dir contains:
      - coinvest_signals.json : UNWRAPPED dict of per-ticker features from the PIT cache
      - hardlinks to every file in base_data_dir (so run_screen.py sees everything else)

    Resolution policy:
      1. Exact-match PIT cache for date_str -> stage with tag "pit_13f_staged_exact"
      2. Nearest-prior cache within PIT_STAGE_NEAREST_PRIOR_DAYS -> stage with
         tag "pit_13f_staged_prior"
      3. No usable cache -> skip with tag "skipped_no_cache"
      4. Resolved cache below PIT_STAGE_MIN_COVERAGE_PCT -> skip with
         tag "skipped_low_coverage" (gate applied to the RESOLVED source
         cache, not the target date)

    Returns (staging_path, source_tag). If staging was skipped, returns (None, source_tag).
    """
    # Resolve the PIT cache source (exact or nearest-prior, backward-only)
    sys.path.insert(0, str(PROJECT_ROOT))
    from common.pit_cache import resolve_pit_cache_dir

    resolved_dir, source_tag = resolve_pit_cache_dir(PIT_13F_CACHE, date_str, PIT_STAGE_NEAREST_PRIOR_DAYS)
    if resolved_dir is None:
        return None, "skipped_no_cache"

    # Apply the coverage gate to the RESOLVED cache date (not the target date).
    cov = _pit_13f_cache_coverage_at(resolved_dir)
    if cov is None or cov < PIT_STAGE_MIN_COVERAGE_PCT:
        return None, "skipped_low_coverage"

    stage_kind = "pit_13f_staged_exact" if source_tag == "exact" else "pit_13f_staged_prior"

    staging = STAGING_ROOT / date_str
    staging.mkdir(parents=True, exist_ok=True)
    out_path = staging / "coinvest_signals.json"
    builder_raw = staging / "_builder_coinvest_features.json"

    # Idempotent: skip builder if output already exists
    if not out_path.exists():
        universe_src = base_data_dir / "universe.json"
        if not universe_src.exists():
            universe_src = PROD_DATA / "universe.json"
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "build_coinvest_features_from_13f.py"),
            "--as-of-date",
            date_str,
            "--cache-root",
            str(PIT_13F_CACHE),
            "--out",
            str(builder_raw),
            "--universe",
            str(universe_src),
            "--cusip-map",
            str(PROD_DATA / "cusip_static_map.json"),
            "--nearest-prior-days",
            str(PIT_STAGE_NEAREST_PRIOR_DAYS),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0 or not builder_raw.exists():
            return None, "skipped_builder_error"

        try:
            with open(builder_raw) as f:
                wrapped = json.load(f)
        except (json.JSONDecodeError, OSError):
            return None, "skipped_builder_error"

        tickers = wrapped.get("tickers", wrapped if isinstance(wrapped, dict) else {})
        with open(out_path, "w") as f:
            json.dump(tickers, f)

    # Shadow everything else from base_data_dir using hardlinks (run_screen.py's
    # security layer rejects symlinks). Fall back to recursive hardlink for
    # directories, and to file copy only as a last resort for small files.
    import os
    import shutil

    def _shadow(src: Path, dst: Path) -> None:
        if dst.exists() or dst.is_symlink():
            return
        if src.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
            for child in src.iterdir():
                _shadow(child, dst / child.name)
            return
        try:
            os.link(str(src), str(dst))
        except OSError:
            # Cross-filesystem or permission issue: copy small files only.
            if src.is_file() and src.stat().st_size < 50 * 1024 * 1024:
                shutil.copy2(str(src), str(dst))

    for item in base_data_dir.iterdir():
        if item.name == "coinvest_signals.json":
            continue
        _shadow(item, staging / item.name)

    return staging, stage_kind


def _snapshot_is_partial(date_str: str, out_dir: Path) -> bool:
    """Return True if a snapshot dir exists but rankings.csv is missing."""
    snap_dir = out_dir / date_str
    return snap_dir.exists() and not (snap_dir / "rankings.csv").exists()


def _clean_partial_dir(date_str: str, out_dir: Path) -> bool:
    """Remove a partial snapshot dir. Returns True if removed."""
    import shutil

    snap_dir = out_dir / date_str
    if snap_dir.exists() and not (snap_dir / "rankings.csv").exists():
        shutil.rmtree(snap_dir)
        return True
    return False


def run_one(
    date_str: str,
    dry_run: bool = False,
    use_bundle: bool = False,
    stage_pit_institutional: bool = False,
    out_dir: Path | None = None,
    force_overwrite: bool = False,
    allow_weekend: bool = False,
    pit_market_data: bool = False,
    isolate_side_outputs: bool = False,
    protect_live_state: bool = False,
    fresh_staging: bool = False,
) -> dict:
    """Run a regen for one date. Returns status dict.

    `pit_market_data=True` stages a market_data.json built for this date, so the run
    does not read today's market cap, price or volume. `isolate_side_outputs=True`
    sends Module 3 side outputs to <out_dir>/_side_outputs/<date>/. With
    `protect_live_state=True` the checkout files a historical run rewrites are
    restored afterwards. `fresh_staging=True` removes this date's staging dir first,
    so links to superseded input files are not reused. The CLI turns all four on.

    When `use_bundle=True` and a PIT bundle exists for the exact date, uses
    `scripts/run_screen_from_bundle.py` (Option A). Otherwise subprocesses
    `run_screen.py --as-of-date` as before.

    When `stage_pit_institutional=True` and a PIT 13F cache exists for this
    date with sufficient coverage, builds a staging data_dir that overlays a
    PIT-derived `coinvest_signals.json` on top of symlinks to the base data
    dir (Option B-lite). Mutually exclusive with `use_bundle`.

    `out_dir` defaults to PIT_V2_DIR.

    SUCCESS REQUIRES rankings.csv: even if the subprocess exits 0, the result
    is classified as `failed_false_success` if rankings.csv is absent. This
    catches run_screen.py's silent refusal to overwrite existing snapshot dirs.
    """
    if dry_run:
        partial = _snapshot_is_partial(date_str, out_dir or PIT_V2_DIR)
        return {"date": date_str, "status": "dry_run", "partial_dir": partial}

    data_dir, data_source = _resolve_data_dir(date_str)
    effective_out = out_dir or PIT_V2_DIR
    stage_result: str | None = None
    base_data_dir = data_dir
    market_cap_sources: dict | None = None

    if fresh_staging and not use_bundle and (STAGING_ROOT / date_str).exists():
        import shutil

        shutil.rmtree(STAGING_ROOT / date_str)

    # Point-in-time market data goes into the staging dir first; the shadowing below
    # never overwrites an existing file, so it stays in place.
    if pit_market_data and not use_bundle:
        try:
            _, market_cap_sources = stage_pit_market_data(date_str, base_data_dir)
        except Exception as e:  # missing prices / market data: fail the date, do not fall back to today's file
            return {
                "date": date_str,
                "status": "error",
                "returncode": None,
                "rankings_csv_exists": False,
                "error_tail": [f"point-in-time market data could not be built: {e}"],
            }

    # Option B-lite: stage a PIT-derived coinvest_signals.json over symlinks.
    if stage_pit_institutional and not use_bundle:
        staged_dir, stage_tag = stage_data_dir_with_pit_institutional(date_str, data_dir)
        stage_result = stage_tag
        if staged_dir is not None:
            data_dir = staged_dir
            # Phase 5: distinguish exact vs prior in the data_source label
            data_source = stage_tag  # "pit_13f_staged_exact" or "pit_13f_staged_prior"

    if pit_market_data and not use_bundle and data_dir == base_data_dir:
        # No PIT 13F overlay for this date: still run from the staging dir so the
        # point-in-time market_data.json is the one run_screen.py reads.
        staging = STAGING_ROOT / date_str
        for item in base_data_dir.iterdir():
            _shadow_tree(item, staging / item.name)
        data_dir = staging

    institutional_source = _resolve_institutional_source(date_str, data_dir)

    bundle_dir = _bundle_dir_for(date_str) if use_bundle else None
    if bundle_dir is not None:
        exec_path = "bundle_native"
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_screen_from_bundle.py"),
            "--bundle-dir",
            str(bundle_dir),
            "--out-root",
            str(effective_out),
            "--pit-mode",
            "lenient",
        ]
    else:
        exec_path = "run_screen_direct"
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "run_screen.py"),
            "--as-of-date",
            date_str,
            "--data-dir",
            str(data_dir),
            "--pit-mode",
            "degrade",
            "--snapshot-dir",
            str(effective_out),
            "--no-clinical-filter",
            "--diagnostics",
            "none",
        ]
        if isolate_side_outputs:
            side_dir = effective_out / SIDE_OUTPUTS_DIRNAME / date_str
            side_dir.mkdir(parents=True, exist_ok=True)
            cmd += ["--output-dir", str(side_dir)]
        if force_overwrite:
            cmd.append("--force-overwrite")
        if allow_weekend:
            cmd.append("--allow-weekend")

    t0 = time.time()
    guard = LiveStateGuard() if protect_live_state else None
    try:
        if guard is not None:
            guard.__enter__()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
                cwd=str(PROJECT_ROOT),
            )
        finally:
            if guard is not None:
                guard.__exit__(None, None, None)
        elapsed = time.time() - t0
        stdout_tail = result.stdout.strip().split("\n")[-10:]
        stderr_tail = result.stderr.strip().split("\n")[-10:]
        rankings_exists = (effective_out / date_str / "rankings.csv").exists()

        if result.returncode != 0:
            return {
                "date": date_str,
                "status": "error",
                "returncode": result.returncode,
                "rankings_csv_exists": rankings_exists,
                "exec_path": exec_path,
                "elapsed_s": round(elapsed, 1),
                "stdout_tail": stdout_tail,
                "stderr_tail": stderr_tail,
                "stage_result": stage_result,
            }

        # Exit 0 is NOT sufficient — rankings.csv must exist.
        if not rankings_exists:
            return {
                "date": date_str,
                "status": "failed_false_success",
                "returncode": 0,
                "rankings_csv_exists": False,
                "exec_path": exec_path,
                "elapsed_s": round(elapsed, 1),
                "stdout_tail": stdout_tail,
                "stderr_tail": stderr_tail,
                "stage_result": stage_result,
                "hint": (
                    "run_screen.py exited 0 but wrote no rankings.csv. "
                    "Likely cause: snapshot dir already existed (partial prior run). "
                    "Use --clean-partial or --force-overwrite to retry."
                ),
            }

        out = {
            "date": date_str,
            "status": "ok",
            "returncode": 0,
            "rankings_csv_exists": True,
            "data_source": data_source,
            "institutional_source": institutional_source,
            "exec_path": exec_path,
            "elapsed_s": round(elapsed, 1),
        }
        if stage_result:
            out["stage_result"] = stage_result
        if market_cap_sources is not None:
            out["market_data_source"] = "pit_market_data"
            out["market_cap_sources"] = market_cap_sources
        else:
            out["market_data_source"] = "current_market_data_json"
        if guard is not None:
            out["live_state_restored"] = guard.restored
        return out

    except subprocess.TimeoutExpired:
        return {"date": date_str, "status": "timeout", "elapsed_s": 600, "rankings_csv_exists": False}
    except Exception as e:
        return {"date": date_str, "status": "exception", "error": str(e), "rankings_csv_exists": False}


MANIFEST_DIR = PROJECT_ROOT / "artifacts" / "audit" / "pit_v2_regeneration"


def _write_manifest(results: list[dict], manifest_dir: Path) -> Path:
    """Write per-run manifest JSON to manifest_dir. Returns path written."""
    manifest_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = manifest_dir / f"regen_manifest_{ts}.json"
    with open(path, "w") as f:
        json.dump(
            {
                "schema": "pit_v2_regen_manifest.v1",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "results": results,
            },
            f,
            indent=2,
        )
    return path


def main():
    parser = argparse.ArgumentParser(description="Regenerate PIT v2 snapshots")
    parser.add_argument("--start", default="2020-01-01", help="Start date")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-dates", type=int, default=0, help="Max dates to process (0=all)")
    parser.add_argument(
        "--use-bundle",
        action="store_true",
        help="Option A: when a PIT bundle exists for the date, call scripts/run_screen_from_bundle.py. "
        "Otherwise use the existing run_screen.py path.",
    )
    parser.add_argument(
        "--stage-pit-institutional",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Option B-lite: when a PIT 13F cache exists with >=50%% manager coverage for "
        "this date, build a staging data_dir with a PIT-derived coinvest_signals.json overlay "
        "before calling run_screen.py. DEFAULT OFF — the shadow mechanism is fragile on /mnt/c/ "
        "NTFS (early partial files cause false-success on retry). Use only for dates with "
        "confirmed 13F PIT cache coverage (typically pre-2026-04). Validated ON as of "
        "2026-04-17 19-date quarter-end run.",
    )
    parser.add_argument(
        "--dates",
        default="",
        help="Comma-separated explicit date list to process (overrides monthly discovery)",
    )
    parser.add_argument(
        "--out-dir",
        default="",
        help="Override output snapshot dir (default: data/snapshots_pit_v2)",
    )
    parser.add_argument(
        "--clean-partial",
        action="store_true",
        help="Before running each date, remove its snapshot dir if it exists but lacks rankings.csv "
        "(partial dir from a prior failed run). Never removes a complete snapshot dir.",
    )
    parser.add_argument(
        "--force-overwrite",
        action="store_true",
        help="Pass --force-overwrite to run_screen.py so it overwrites existing snapshot dirs. "
        "Implies --clean-partial is NOT needed, but does not protect against complete dirs being "
        "re-run unintentionally. Prefer --clean-partial for safety.",
    )
    parser.add_argument(
        "--allow-weekend",
        action="store_true",
        help="Pass --allow-weekend to run_screen.py for dates that fall on Saturday/Sunday.",
    )
    parser.add_argument(
        "--pit-market-data",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Build market_data.json for each date from prices and SEC share counts available on that "
        "date (default ON). --no-pit-market-data reads today's market_data.json for every date, which "
        "puts today's market cap, price and volume into the history.",
    )
    parser.add_argument(
        "--isolate-side-outputs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Send Module 3 side outputs (catalyst_events, run_log, ctgov_state) to "
        "<out-dir>/_side_outputs/<date>/ instead of the data dir (default ON).",
    )
    parser.add_argument(
        "--protect-live-state",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Restore data/state, the expression logs, readiness history and 'latest' pointers after "
        "each date, and refuse to run while a daily production lock exists (default ON).",
    )
    parser.add_argument(
        "--fresh-staging",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove each date's staging dir before staging, so hardlinks to superseded input files "
        "are not reused (default ON).",
    )
    args = parser.parse_args()

    effective_out = Path(args.out_dir) if args.out_dir else PIT_V2_DIR

    if args.dates:
        dates = [d.strip() for d in args.dates.split(",") if d.strip()]
        todo = [d for d in dates if not (effective_out / d / "rankings.csv").exists()]
    else:
        dates = get_monthly_dates(args.start)
        todo = [d for d in dates if not (effective_out / d / "rankings.csv").exists()]
    done = [d for d in dates if (effective_out / d / "rankings.csv").exists()]
    partial = [d for d in todo if _snapshot_is_partial(d, effective_out)]

    print(f"Dates:         {len(dates)}")
    print(f"Already done:  {len(done)}")
    print(f"To process:    {len(todo)}")
    if partial:
        print(f"Partial dirs:  {len(partial)} — {partial}")
        if not args.clean_partial and not args.force_overwrite:
            print(
                "  WARNING: partial dirs exist. Use --clean-partial to remove them before "
                "re-running, or --force-overwrite to let run_screen.py overwrite them. "
                "Without one of these, those dates will likely produce FAILED_FALSE_SUCCESS."
            )
    if args.use_bundle:
        print("Mode:          bundle-native (Option A) when bundle exists, run_screen fallback otherwise")
    if args.stage_pit_institutional:
        print("Mode:          PIT 13F staging (Option B-lite) ENABLED — fragile on /mnt/c/ NTFS, use with supervision")
    else:
        print("Mode:          staging OFF (safe default) — current holdings_detailed used for institutional")
    if args.out_dir:
        print(f"Out dir:       {effective_out}")
    if args.clean_partial:
        print("Clean partial: ON")
    if args.force_overwrite:
        print("Force overwrite: ON")
    if args.allow_weekend:
        print("Allow weekend: ON")

    print(f"Market data:   {'point-in-time per date' if args.pit_market_data else 'TODAY (look-ahead)'}")
    print(f"Side outputs:  {'isolated under out dir' if args.isolate_side_outputs else 'data dir'}")
    print(f"Live state:    {'protected' if args.protect_live_state else 'NOT protected'}")

    if args.max_dates > 0:
        todo = todo[: args.max_dates]
        print(f"Limited to:    {len(todo)}")

    if not todo:
        print("Nothing to do.")
        return

    effective_out.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    results = []
    n_ok = 0
    n_err = 0
    t_start = time.time()

    for i, date_str in enumerate(todo, 1):
        prefix = f"[{i}/{len(todo)}]"
        if args.dry_run:
            partial_flag = _snapshot_is_partial(date_str, effective_out)
            print(f"{prefix} {date_str} — dry run{' (PARTIAL DIR)' if partial_flag else ''}")
            results.append(run_one(date_str, dry_run=True, out_dir=effective_out))
            continue

        # Clean partial dir before running if requested.
        cleaned = False
        if args.clean_partial and _snapshot_is_partial(date_str, effective_out):
            cleaned = _clean_partial_dir(date_str, effective_out)

        if args.protect_live_state:
            lock = production_lock_held()
            if lock:
                print(f"{prefix} {date_str} — STOPPED: daily production lock present ({lock}).")
                results.append({"date": date_str, "status": "stopped_production_lock", "rankings_csv_exists": False})
                n_err += 1
                break

        print(f"{prefix} {date_str}{' [cleaned partial]' if cleaned else ''} ...", end=" ", flush=True)
        r = run_one(
            date_str,
            use_bundle=args.use_bundle,
            stage_pit_institutional=args.stage_pit_institutional,
            out_dir=effective_out,
            force_overwrite=args.force_overwrite,
            allow_weekend=args.allow_weekend,
            pit_market_data=args.pit_market_data,
            isolate_side_outputs=args.isolate_side_outputs,
            protect_live_state=args.protect_live_state,
            fresh_staging=args.fresh_staging,
        )
        r["cleaned_partial_dir"] = cleaned
        results.append(r)

        if r["status"] == "ok":
            n_ok += 1
            src = r.get("data_source", "current")
            inst = r.get("institutional_source", "?")
            exe = r.get("exec_path", "?")
            stg = r.get("stage_result")
            stg_str = f", stage={stg}" if stg else ""
            mkt = r.get("market_data_source", "?")
            print(f"OK ({r['elapsed_s']}s, exec={exe}, data={src}, inst={inst}{stg_str}, market={mkt})")
            if r.get("live_state_restored"):
                print(f"    restored live state: {', '.join(r['live_state_restored'])}")
        else:
            n_err += 1
            label = r["status"].upper()
            print(f"FAIL [{label}]")
            if r.get("hint"):
                print(f"    HINT: {r['hint']}")
            for line in r.get("stderr_tail") or r.get("error_tail", []):
                print(f"    {line}")

    elapsed_total = time.time() - t_start

    # Institutional-source summary (diagnostic — see docs/13F_BACKFILL_PLAN.md)
    inst_counts: dict[str, int] = {}
    for r in results:
        if r.get("status") == "ok":
            tag = r.get("institutional_source", "?")
            inst_counts[tag] = inst_counts.get(tag, 0) + 1

    # Save log
    log = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_dates": len(dates),
        "processed": len(results),
        "ok": n_ok,
        "errors": n_err,
        "elapsed_total_s": round(elapsed_total, 1),
        "institutional_source_counts": inst_counts,
        "results": results,
    }
    with open(LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)

    # Write per-run audit manifest.
    manifest_path = _write_manifest(results, MANIFEST_DIR)

    print(f"\nDone: {n_ok} ok, {n_err} errors in {elapsed_total:.0f}s")
    print(f"Log:      {LOG_PATH}")
    print(f"Manifest: {manifest_path}")

    if n_err > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
