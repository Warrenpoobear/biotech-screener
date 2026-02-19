#!/usr/bin/env python3
"""Phase-2 Daily Production Runner.

Single entrypoint that orchestrates:
  1. Incremental price_history.csv refresh (including XBI)
  2. run_screen.py in phase2 mode → staging directory
  3. data_integrity_audit.py → cross-validates price-derived fields
  4. Hard gates: XBI staleness, missing-reason fraction, turnover, audit verdict
  5. Run manifest (run_manifest.json) with full provenance
  6. Atomic promotion: staging → data/snapshots/{effective_as_of_date}/ on gate pass

Exit codes:
  0 — all gates passed, snapshot promoted
  1 — hard gate FAIL (snapshot stays in staging)
  2 — gate WARN (snapshot promoted but flagged)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Repo root — all paths relative to the repo
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from archive_snapshot import get_git_info

# ---------------------------------------------------------------------------
# Gate thresholds (defaults; overridable via --gate-config)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GateConfig:
    """Hard gate thresholds for production daily runs."""
    xbi_stale_days: int = 3
    """Max trading-day gap between XBI last date and as_of_date before FAIL."""

    missing_reason_max_frac: float = 0.05
    """Max fraction of DE-critical fields with non-empty missing_reason."""

    turnover_max_pct: float = 40.0
    """Max name turnover (%) before FAIL."""

    audit_fail_is_gate_fail: bool = True
    """If data_integrity_audit exits 1, treat as gate FAIL."""

    audit_warn_is_gate_warn: bool = True
    """If data_integrity_audit exits 2, treat as gate WARN."""

    @staticmethod
    def from_json(path: Path) -> "GateConfig":
        with open(path) as f:
            d = json.load(f)
        return GateConfig(**{k: v for k, v in d.items() if k in GateConfig.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Gate result
# ---------------------------------------------------------------------------

@dataclass
class GateResult:
    name: str
    status: str  # "PASS" | "WARN" | "FAIL"
    detail: str = ""
    value: Any = None
    threshold: Any = None


# ---------------------------------------------------------------------------
# Step 1: Price refresh
# ---------------------------------------------------------------------------

def refresh_prices(
    price_csv: Path,
    through_date: str,
    universe_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Incrementally refresh price_history.csv via extend_price_csv().

    Returns stats dict from extend_price_csv plus xbi_last_date.
    """
    # Import lazily to avoid yfinance dependency at module level
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from backtest_signal_robustness import extend_price_csv

    # Collect tickers from universe.json if available
    tickers: Optional[List[str]] = None
    if universe_path and universe_path.exists():
        with open(universe_path) as f:
            universe = json.load(f)
        if isinstance(universe, list):
            tickers = [e.get("ticker", e) if isinstance(e, dict) else str(e) for e in universe]
        elif isinstance(universe, dict) and "tickers" in universe:
            tickers = universe["tickers"]
        # Always include XBI benchmark
        if tickers and "XBI" not in tickers:
            tickers.append("XBI")

    stats = extend_price_csv(
        csv_path=price_csv,
        through_date=through_date,
        tickers=tickers,
    )

    # Compute XBI last date from the CSV
    xbi_last_date = _get_ticker_last_date(price_csv, "XBI")
    stats["xbi_last_date"] = xbi_last_date

    return stats


def _get_ticker_last_date(price_csv: Path, ticker: str) -> Optional[str]:
    """Read price_history.csv and return the last date for a given ticker."""
    if not price_csv.exists():
        return None
    last_date = None
    with open(price_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (row.get("ticker") or "").strip().upper() == ticker.upper():
                d = (row.get("date") or "").strip()
                if d and (last_date is None or d > last_date):
                    last_date = d
    return last_date


# ---------------------------------------------------------------------------
# Step 2: Run screen
# ---------------------------------------------------------------------------

def run_screen(
    as_of_date: str,
    data_dir: Path,
    snapshot_dir: Path,
    price_csv: Path,
    *,
    ruleset_path: Optional[Path] = None,
    extra_args: Optional[List[str]] = None,
) -> subprocess.CompletedProcess:
    """Run run_screen.py in phase2 mode with decision ranking."""
    # run_screen.py requires --output for the raw JSON results
    output_json = snapshot_dir / as_of_date / "screen_output.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(REPO_ROOT / "run_screen.py"),
        "--as-of-date", as_of_date,
        "--data-dir", str(data_dir),
        "--output", str(output_json),
        "--decision-mode", "phase2",
        "--ranking-mode", "decision",
        "--snapshot-dir", str(snapshot_dir),
        "--strict",
    ]
    if ruleset_path:
        cmd.extend(["--ruleset", str(ruleset_path)])
    if extra_args:
        cmd.extend(extra_args)

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))
    return result


# ---------------------------------------------------------------------------
# Step 3: Run integrity audit
# ---------------------------------------------------------------------------

def run_audit(
    snapshot_date_dir: Path,
    price_csv: Path,
    as_of_date: str,
    output_dir: Path,
) -> subprocess.CompletedProcess:
    """Run tools/data_integrity_audit.py and return its result."""
    cmd = [
        sys.executable, str(REPO_ROOT / "tools" / "data_integrity_audit.py"),
        "--snapshot-dir", str(snapshot_date_dir),
        "--price-history", str(price_csv),
        "--as-of-date", as_of_date,
        "--output-dir", str(output_dir),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))
    return result


# ---------------------------------------------------------------------------
# Step 4: Hard gates
# ---------------------------------------------------------------------------

def check_xbi_staleness(
    price_csv: Path,
    as_of_date: str,
    threshold_days: int,
) -> GateResult:
    """Check if XBI data is stale beyond threshold."""
    xbi_last = _get_ticker_last_date(price_csv, "XBI")
    if xbi_last is None:
        return GateResult(
            name="xbi_staleness", status="FAIL",
            detail="XBI not found in price_history.csv",
            value=None, threshold=threshold_days,
        )

    # Count trading days gap (approximate: weekdays only)
    from datetime import timedelta
    last_dt = datetime.strptime(xbi_last, "%Y-%m-%d")
    as_of_dt = datetime.strptime(as_of_date, "%Y-%m-%d")
    delta = as_of_dt - last_dt
    # Approximate trading days (exclude weekends)
    trading_days = sum(
        1 for i in range(1, delta.days + 1)
        if (last_dt + timedelta(days=i)).weekday() < 5
    )

    if trading_days > threshold_days:
        return GateResult(
            name="xbi_staleness", status="FAIL",
            detail=f"XBI last={xbi_last}, as_of={as_of_date}, gap={trading_days} trading days",
            value=trading_days, threshold=threshold_days,
        )
    return GateResult(
        name="xbi_staleness", status="PASS",
        detail=f"XBI last={xbi_last}, gap={trading_days} trading days",
        value=trading_days, threshold=threshold_days,
    )


def check_missing_reason_fraction(
    snapshot_date_dir: Path,
    max_frac: float,
) -> GateResult:
    """Check fraction of tickers with non-empty missing_reason for DE-critical fields."""
    rankings_path = snapshot_date_dir / "rankings.csv"
    if not rankings_path.exists():
        return GateResult(
            name="missing_reason_fraction", status="FAIL",
            detail="rankings.csv not found",
        )

    critical_fields = [
        "de_beta_xbi_60d_missing_reason",
        "de_alpha_60d_missing_reason",
    ]

    total = 0
    missing_count = 0
    with open(rankings_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            for fld in critical_fields:
                val = (row.get(fld) or "").strip()
                if val and val.lower() not in ("", "nan"):
                    missing_count += 1
                    break  # count ticker once even if multiple fields missing

    if total == 0:
        return GateResult(
            name="missing_reason_fraction", status="FAIL",
            detail="rankings.csv is empty",
        )

    frac = missing_count / total
    status = "FAIL" if frac > max_frac else "PASS"
    return GateResult(
        name="missing_reason_fraction", status=status,
        detail=f"{missing_count}/{total} tickers ({frac:.1%}) have missing_reason",
        value=round(frac, 4), threshold=max_frac,
    )


def check_turnover(
    snapshot_date_dir: Path,
    max_pct: float,
) -> GateResult:
    """Check name turnover from the delta report."""
    delta_path = snapshot_date_dir / "phase2_run_delta_report.txt"
    if not delta_path.exists():
        return GateResult(
            name="turnover", status="PASS",
            detail="No delta report (first run or --no-delta); skipped",
        )

    text = delta_path.read_text()
    # Parse "Name turnover: XX.X%" from the report
    import re
    match = re.search(r"Name turnover:\s+([\d.]+)%", text)
    if not match:
        return GateResult(
            name="turnover", status="PASS",
            detail="Could not parse turnover from delta report; skipped",
        )

    turnover = float(match.group(1))
    status = "FAIL" if turnover > max_pct else "PASS"
    return GateResult(
        name="turnover", status=status,
        detail=f"Name turnover={turnover:.1f}%",
        value=turnover, threshold=max_pct,
    )


def check_audit_result(
    audit_proc: subprocess.CompletedProcess,
    config: GateConfig,
) -> GateResult:
    """Translate audit tool exit code into a gate result."""
    if audit_proc.returncode == 0:
        return GateResult(name="audit", status="PASS", detail="Audit OK")
    elif audit_proc.returncode == 2:
        status = "WARN" if config.audit_warn_is_gate_warn else "PASS"
        return GateResult(
            name="audit", status=status,
            detail="Audit WARN (invariant violations, no critical failures)",
        )
    else:
        status = "FAIL" if config.audit_fail_is_gate_fail else "WARN"
        return GateResult(
            name="audit", status=status,
            detail=f"Audit FAIL (exit code {audit_proc.returncode})",
        )


def _parse_cache_date(p: Path) -> Optional["date"]:
    """Extract and parse YYYY-MM-DD from a trial_records_{date}.json filename."""
    from datetime import date
    s = p.stem.replace("trial_records_", "")
    try:
        return date.fromisoformat(s)
    except ValueError:
        return None


def check_ctgov_cache(
    as_of_date: str,
    cache_dir: Path,
    allow_fallback: bool = False,
) -> tuple[GateResult, str]:
    """Check if PIT-filtered ctgov cache exists for the requested date.

    Returns (gate_result, effective_as_of_date).
    If allow_fallback=True and the exact date is missing, picks the latest
    cached date <= requested date and returns WARN.
    """
    from datetime import date

    exact = cache_dir / f"trial_records_{as_of_date}.json"
    if exact.exists():
        return (
            GateResult(
                name="ctgov_cache", status="PASS",
                detail=f"PIT cache found: {exact.name}",
            ),
            as_of_date,
        )

    # Exact date missing — look for fallback if allowed
    if allow_fallback:
        req = date.fromisoformat(as_of_date)
        parsed = []
        for p in cache_dir.glob("trial_records_*.json"):
            d = _parse_cache_date(p)
            if d is not None and d <= req:
                parsed.append((d, p))
        parsed.sort(key=lambda x: x[0])
        if parsed:
            fallback_date = parsed[-1][0].isoformat()
            return (
                GateResult(
                    name="ctgov_cache", status="WARN",
                    detail=(
                        f"PIT cache missing for {as_of_date}; "
                        f"falling back to {fallback_date} (--allow-date-fallback)"
                    ),
                ),
                fallback_date,
            )

    # No exact match, no fallback allowed (or no prior dates)
    return (
        GateResult(
            name="ctgov_cache", status="FAIL",
            detail=(
                f"PIT cache missing: {exact.name}. "
                f"Run: warm_caches.py --as-of-date {as_of_date} --sources ctgov"
            ),
        ),
        as_of_date,
    )


# ---------------------------------------------------------------------------
# Step 5: Run manifest
# ---------------------------------------------------------------------------

def build_run_manifest(
    as_of_date: str,
    gate_results: List[GateResult],
    price_stats: Dict[str, Any],
    screen_proc: subprocess.CompletedProcess,
    audit_proc: Optional[subprocess.CompletedProcess],
    config: GateConfig,
    snapshot_date_dir: Optional[Path] = None,
    *,
    requested_as_of_date: Optional[str] = None,
    git_pre_run: Optional[Dict[str, Any]] = None,
    git_post_run: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the run_manifest.json with full provenance.

    If requested_as_of_date differs from as_of_date, it means a date
    fallback occurred and both are recorded in the manifest.

    git_pre_run: git info captured before any artifacts are written.
    git_post_run: git info captured after screen/audit (optional).
    git.dirty == git.dirty_pre_run for backward compatibility.
    """
    git = git_pre_run if git_pre_run is not None else get_git_info(REPO_ROOT)

    # Read metadata.json from snapshot for ruleset info
    ruleset_info: Dict[str, Any] = {}
    row_counts: Dict[str, Any] = {}
    if snapshot_date_dir and (snapshot_date_dir / "metadata.json").exists():
        meta = json.loads((snapshot_date_dir / "metadata.json").read_text())
        # Ruleset ID is in clinical_sort_telemetry (primary) or health JSON
        cst = meta.get("clinical_sort_telemetry") or {}
        ruleset_info = {
            "ruleset_version": meta.get("version", ""),
            "ruleset_hash": cst.get("ruleset_id", ""),
            "ranking_mode": meta.get("ranking_mode", ""),
            "decision_mode": meta.get("decision_mode", ""),
        }
        # Also check phase2_health.json for authoritative ruleset_id
        health_path = snapshot_date_dir / "phase2_health.json"
        if health_path.exists():
            try:
                health = json.loads(health_path.read_text())
                if health.get("ruleset_id"):
                    ruleset_info["ruleset_hash"] = health["ruleset_id"]
            except (json.JSONDecodeError, OSError):
                pass
        row_counts = {
            "ticker_count": meta.get("ticker_count"),
            "total_evaluated": meta.get("total_evaluated"),
            "active_universe": meta.get("active_universe"),
        }

    # Count missing_reason from rankings.csv
    missing_reason_counts: Dict[str, int] = {}
    if snapshot_date_dir:
        rankings_path = snapshot_date_dir / "rankings.csv"
        if rankings_path.exists():
            with open(rankings_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    for col in ("de_beta_xbi_60d_missing_reason", "de_alpha_60d_missing_reason"):
                        val = (row.get(col) or "").strip()
                        if val and val.lower() not in ("nan",):
                            missing_reason_counts[val] = missing_reason_counts.get(val, 0) + 1

    overall_status = "PASS"
    for g in gate_results:
        if g.status == "FAIL":
            overall_status = "FAIL"
            break
        if g.status == "WARN" and overall_status != "FAIL":
            overall_status = "WARN"

    _requested = requested_as_of_date or as_of_date

    # Enrich git block with pre/post-run dirty flags
    git_block = dict(git)  # shallow copy to avoid mutating caller's dict
    git_block["dirty_pre_run"] = git.get("dirty")
    git_block["dirty_post_run"] = git_post_run.get("dirty") if git_post_run else None
    git_block["dirty"] = git_block["dirty_pre_run"]  # backward compat

    return {
        "manifest_version": "1.1.0",
        "requested_as_of_date": _requested,
        "effective_as_of_date": as_of_date,
        "as_of_date": as_of_date,  # backward compat alias
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git": git_block,
        "ruleset": ruleset_info,
        "row_counts": row_counts,
        "price_refresh": {
            "n_extended": price_stats.get("n_extended"),
            "n_rows_appended": price_stats.get("n_rows_appended"),
            "n_failed": price_stats.get("n_failed"),
            "failed_tickers": price_stats.get("failed_tickers", []),
            "xbi_last_date": price_stats.get("xbi_last_date"),
        },
        "missing_reason_counts": missing_reason_counts,
        "gates": [
            {
                "name": g.name,
                "status": g.status,
                "detail": g.detail,
                "value": g.value,
                "threshold": g.threshold,
            }
            for g in gate_results
        ],
        "overall_status": overall_status,
        "screen_exit_code": screen_proc.returncode,
        "audit_exit_code": audit_proc.returncode if audit_proc else None,
        "gate_config": {k: v for k, v in asdict(config).items()},
    }


# ---------------------------------------------------------------------------
# Step 6: Atomic promotion
# ---------------------------------------------------------------------------

def promote_snapshot(
    staging_date_dir: Path,
    final_snapshots_dir: Path,
    as_of_date: str,
) -> Path:
    """Atomically move staging snapshot to final location.

    Uses rename when on same filesystem, falls back to copy+delete.
    Returns the final path.
    """
    final_date_dir = final_snapshots_dir / as_of_date
    if final_date_dir.exists():
        # Archive existing by renaming with timestamp
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup = final_snapshots_dir / f"{as_of_date}__pre_{ts}"
        shutil.move(str(final_date_dir), str(backup))

    try:
        os.rename(str(staging_date_dir), str(final_date_dir))
    except OSError:
        # Cross-filesystem: copy then delete
        shutil.copytree(str(staging_date_dir), str(final_date_dir))
        shutil.rmtree(str(staging_date_dir))

    return final_date_dir


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_daily(
    as_of_date: str,
    data_dir: Path,
    price_csv: Path,
    final_snapshots_dir: Path,
    *,
    gate_config: Optional[GateConfig] = None,
    ruleset_path: Optional[Path] = None,
    skip_price_refresh: bool = False,
    skip_audit: bool = False,
    extra_screen_args: Optional[List[str]] = None,
    allow_date_fallback: bool = False,
    ctgov_cache_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute the full daily Phase-2 pipeline.

    Returns the run manifest dict. Raises SystemExit on hard gate failure
    when called from CLI.
    """
    config = gate_config or GateConfig()
    gate_results: List[GateResult] = []
    requested_as_of_date = as_of_date  # preserve the original request

    # Capture git state BEFORE any artifacts are written
    git_pre_run = get_git_info(REPO_ROOT)

    print(f"{'='*70}")
    print(f"PHASE-2 DAILY RUN — {as_of_date}")
    print(f"{'='*70}")

    # --- Step 1: Price refresh ---
    price_stats: Dict[str, Any] = {}
    if not skip_price_refresh:
        print("\n[1/5] Refreshing price_history.csv ...")
        universe_path = data_dir / "universe.json"
        price_stats = refresh_prices(price_csv, as_of_date, universe_path)
        print(f"  Extended {price_stats.get('n_extended', 0)} tickers, "
              f"{price_stats.get('n_rows_appended', 0)} rows appended, "
              f"{price_stats.get('n_failed', 0)} failures")
        if price_stats.get("failed_tickers"):
            print(f"  Failed: {', '.join(price_stats['failed_tickers'][:10])}")
    else:
        print("\n[1/5] Price refresh skipped (--skip-price-refresh)")
        price_stats["xbi_last_date"] = _get_ticker_last_date(price_csv, "XBI")

    # --- Gate: XBI staleness (check early, before expensive screen run) ---
    xbi_gate = check_xbi_staleness(price_csv, as_of_date, config.xbi_stale_days)
    gate_results.append(xbi_gate)
    print(f"  XBI gate: {xbi_gate.status} — {xbi_gate.detail}")
    if xbi_gate.status == "FAIL":
        print("\n  FATAL: XBI staleness gate FAIL. Aborting before screen run.")
        manifest = build_run_manifest(
            as_of_date, gate_results, price_stats,
            subprocess.CompletedProcess(args=[], returncode=-1),
            None, config,
            requested_as_of_date=requested_as_of_date,
            git_pre_run=git_pre_run,
        )
        return manifest

    # --- Gate: ctgov PIT cache availability ---
    _cache_dir = ctgov_cache_dir or (REPO_ROOT / "cache" / "ctgov")
    ctgov_gate, effective_as_of_date = check_ctgov_cache(
        as_of_date, _cache_dir, allow_fallback=allow_date_fallback,
    )
    gate_results.append(ctgov_gate)
    print(f"  CTGov cache gate: {ctgov_gate.status} — {ctgov_gate.detail}")
    if ctgov_gate.status == "FAIL":
        print("\n  FATAL: CTGov PIT cache not found. Aborting before screen run.")
        print(f"  Hint: run warm_caches.py --as-of-date {as_of_date} --sources ctgov")
        manifest = build_run_manifest(
            as_of_date, gate_results, price_stats,
            subprocess.CompletedProcess(args=[], returncode=-1),
            None, config,
            requested_as_of_date=requested_as_of_date,
            git_pre_run=git_pre_run,
        )
        return manifest

    if effective_as_of_date != as_of_date:
        print(f"  Date fallback: {as_of_date} → {effective_as_of_date}")
        as_of_date = effective_as_of_date

    # --- Step 2: Run screen into staging dir ---
    print(f"\n[2/5] Running screen (phase2, ranking_mode=decision) ...")
    staging_dir = Path(tempfile.mkdtemp(prefix=f"phase2_staging_{as_of_date}_"))
    screen_proc = run_screen(
        as_of_date, data_dir, staging_dir, price_csv,
        ruleset_path=ruleset_path,
        extra_args=extra_screen_args,
    )
    staging_date_dir = staging_dir / as_of_date

    if screen_proc.returncode not in (0, 2):
        print(f"  Screen FAILED (exit {screen_proc.returncode})")
        if screen_proc.stderr:
            for line in screen_proc.stderr.strip().splitlines()[-10:]:
                print(f"    {line}")
        gate_results.append(GateResult(
            name="screen", status="FAIL",
            detail=f"Screen failed (exit {screen_proc.returncode})",
            value=screen_proc.returncode,
        ))
        manifest = build_run_manifest(
            as_of_date, gate_results, price_stats, screen_proc, None, config,
            requested_as_of_date=requested_as_of_date,
            git_pre_run=git_pre_run,
        )
        return manifest

    if screen_proc.returncode == 2:
        print(f"  Screen completed with WARN (exit 2)")
    else:
        print(f"  Screen completed OK")

    if not staging_date_dir.exists():
        print(f"  ERROR: Expected snapshot at {staging_date_dir} not found")
        gate_results.append(GateResult(
            name="screen", status="FAIL",
            detail=f"Snapshot directory not created by screen: {staging_date_dir}",
        ))
        manifest = build_run_manifest(
            as_of_date, gate_results, price_stats, screen_proc, None, config,
            requested_as_of_date=requested_as_of_date,
            git_pre_run=git_pre_run,
        )
        return manifest

    # --- Step 3: Run integrity audit ---
    audit_proc = None
    if not skip_audit:
        print(f"\n[3/5] Running data integrity audit ...")
        audit_output_dir = staging_date_dir / "audit"
        audit_proc = run_audit(staging_date_dir, price_csv, as_of_date, audit_output_dir)
        audit_gate = check_audit_result(audit_proc, config)
        gate_results.append(audit_gate)
        print(f"  Audit gate: {audit_gate.status} — {audit_gate.detail}")
    else:
        print(f"\n[3/5] Audit skipped (--skip-audit)")

    # --- Step 4: Hard gates ---
    print(f"\n[4/5] Evaluating gates ...")

    missing_gate = check_missing_reason_fraction(staging_date_dir, config.missing_reason_max_frac)
    gate_results.append(missing_gate)
    print(f"  Missing-reason gate: {missing_gate.status} — {missing_gate.detail}")

    turnover_gate = check_turnover(staging_date_dir, config.turnover_max_pct)
    gate_results.append(turnover_gate)
    print(f"  Turnover gate: {turnover_gate.status} — {turnover_gate.detail}")

    # --- Step 5: Build manifest ---
    print(f"\n[5/5] Building run manifest ...")
    git_post_run = get_git_info(REPO_ROOT)
    manifest = build_run_manifest(
        as_of_date, gate_results, price_stats,
        screen_proc, audit_proc, config,
        snapshot_date_dir=staging_date_dir,
        requested_as_of_date=requested_as_of_date,
        git_pre_run=git_pre_run,
        git_post_run=git_post_run,
    )

    # Write manifest to staging dir
    manifest_path = staging_date_dir / "run_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"  Manifest → {manifest_path}")

    # --- Promotion decision ---
    overall = manifest["overall_status"]
    if overall == "FAIL":
        print(f"\n{'='*70}")
        print(f"RESULT: FAIL — snapshot NOT promoted")
        print(f"  Staging dir preserved at: {staging_date_dir}")
        for g in gate_results:
            if g.status == "FAIL":
                print(f"  [{g.status}] {g.name}: {g.detail}")
        print(f"{'='*70}")
    else:
        final_path = promote_snapshot(staging_date_dir, final_snapshots_dir, as_of_date)
        # Clean up empty staging parent
        if staging_dir.exists() and not any(staging_dir.iterdir()):
            staging_dir.rmdir()
        label = "PASS" if overall == "PASS" else "WARN"
        print(f"\n{'='*70}")
        print(f"RESULT: {label} — snapshot promoted to {final_path}")
        if overall == "WARN":
            for g in gate_results:
                if g.status == "WARN":
                    print(f"  [{g.status}] {g.name}: {g.detail}")
        print(f"{'='*70}")

    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Phase-2 Daily Production Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Exit codes:\n"
            "  0  all gates passed, snapshot promoted\n"
            "  1  hard gate FAIL (snapshot in staging)\n"
            "  2  gate WARN (snapshot promoted, flagged)\n"
        ),
    )
    parser.add_argument(
        "--as-of-date", required=True,
        help="Snapshot date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=REPO_ROOT / "production_data",
        help="Path to production_data/ (default: production_data/)",
    )
    parser.add_argument(
        "--price-history", type=Path, default=REPO_ROOT / "production_data" / "price_history.csv",
        help="Path to price_history.csv (default: production_data/price_history.csv)",
    )
    parser.add_argument(
        "--snapshot-dir", type=Path, default=REPO_ROOT / "data" / "snapshots",
        help="Final snapshot directory (default: data/snapshots/)",
    )
    parser.add_argument(
        "--ruleset", type=Path, default=None,
        help="Path to decision engine ruleset JSON",
    )
    parser.add_argument(
        "--gate-config", type=Path, default=None,
        help="Path to gate configuration JSON (overrides defaults)",
    )
    parser.add_argument(
        "--skip-price-refresh", action="store_true",
        help="Skip incremental price refresh (use existing price_history.csv)",
    )
    parser.add_argument(
        "--skip-audit", action="store_true",
        help="Skip data integrity audit step",
    )
    parser.add_argument(
        "--allow-date-fallback", action="store_true",
        help="If ctgov cache missing for --as-of-date, fall back to latest cached date (WARN).",
    )
    parser.add_argument(
        "--ctgov-cache-dir", type=Path, default=None,
        help="Path to ctgov cache directory (default: cache/ctgov/)",
    )
    args = parser.parse_args()

    config = GateConfig()
    if args.gate_config:
        config = GateConfig.from_json(args.gate_config)

    manifest = run_daily(
        as_of_date=args.as_of_date,
        data_dir=args.data_dir,
        price_csv=args.price_history,
        final_snapshots_dir=args.snapshot_dir,
        gate_config=config,
        ruleset_path=args.ruleset,
        skip_price_refresh=args.skip_price_refresh,
        skip_audit=args.skip_audit,
        allow_date_fallback=args.allow_date_fallback,
        ctgov_cache_dir=args.ctgov_cache_dir,
    )

    # Always write manifest to output/ for CI discoverability
    # (snapshot-dir manifest only exists on successful promotion)
    output_dir = REPO_ROOT / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    fallback_manifest = output_dir / "run_manifest.json"
    with open(fallback_manifest, "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    status = manifest.get("overall_status", "FAIL")
    if status == "FAIL":
        sys.exit(1)
    elif status == "WARN":
        sys.exit(2)
    sys.exit(0)


if __name__ == "__main__":
    main()
