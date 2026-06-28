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
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Repo root — all paths relative to the repo
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Load .env file from repo root (credentials, API keys)
from dotenv import load_dotenv

load_dotenv(REPO_ROOT / ".env")
from archive_snapshot import get_git_info, sha256_file

# ---------------------------------------------------------------------------
# Ops contract constants
# ---------------------------------------------------------------------------
MANIFEST_VERSION = "1.3.0"


def _deterministic_timestamp(as_of_date: str) -> str:
    """Return the canonical deterministic timestamp for an as-of date."""
    return f"{as_of_date}T00:00:00Z"


def _timestamp_for_snapshot_dir(snap_dir: Path) -> str:
    """Derive progress timestamps from the snapshot date directory name."""
    try:
        date.fromisoformat(snap_dir.name)
        return _deterministic_timestamp(snap_dir.name)
    except ValueError:
        return "1970-01-01T00:00:00Z"


# Canonical gate names — every gate emitted by run_daily() MUST be in this set.
# Adding a new gate requires updating this allowlist.
GATE_ALLOWLIST: frozenset[str] = frozenset(
    {
        "xbi_staleness",
        "ctgov_cache",
        "inputs_present",
        "market_data_schema",
        "market_data_staleness",
        "market_data_coverage",
        "screen",
        "audit",
        "missing_reason_fraction",
        "turnover",
        "drift_monitoring",
        "ctgov_pit_dates",
        "sec_13f_cache",
        "institutional_summary",
        "institutional_delta",
        "pnl_attribution",
        "price_pit_cache",
        "forward_eval",
        "pit_bundle_health",
        "decision_engine_schema",
        "sort_contrib_sanity",
        "portfolio_weights",
        "eligibility_consistency",
        "cache_health",
        "source_freshness",
        "ruleset_health",
        "exposure_missingness",
        "risk_concentration",
        "ruleset_governance",
        "regulatory_calendar",
        "canary_regression",
        "options_coverage",
        "trading_day",
        "hard_queue_artifacts",
        "hard_catalyst_supply",
        "hard_options_coverage",
        "hard_carry_state",
        "hard_queue_actionability",
        "optionality_stability",
        "phase2_health",
    }
)

# Required fields in each market_data.json record for schema gate
MARKET_DATA_REQUIRED_FIELDS: frozenset[str] = frozenset(
    {
        "ticker",
        "price",
        "market_cap",
        "collected_at",
    }
)

# Fields that must be numeric (int/float) or None — schema gate checks type
MARKET_DATA_NUMERIC_FIELDS: frozenset[str] = frozenset(
    {
        "price",
        "market_cap",
        "avg_volume",
        "beta",
        "52w_high",
        "52w_low",
        "volatility_90d",
        "returns_1m",
        "returns_3m",
    }
)

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
    """If data_integrity_audit exits 1 (critical invariant), treat as gate FAIL.

    With 4-tier exit codes, exit 1 means data model broken (critical invariants).
    Stale price recompute mismatches now exit 3 (always WARN, never FAIL).
    """

    audit_warn_is_gate_warn: bool = True
    """If data_integrity_audit exits 2, treat as gate WARN."""

    market_data_max_age_days: int = 5
    """TEMP 2026-05-19: raised from 3→5 for Yahoo Finance 429 outage."""
    # REVERT to 3 once Yahoo API recovers
    # """Max calendar-day age of market_data.json (collected_at vs as_of_date)."""

    market_data_min_coverage: float = 0.90
    """Min fraction of universe tickers that must have market_data records."""

    sec_13f_coverage_warn_pct: float = 80.0
    """Min 13F manager coverage (%) before WARN. Never FAIL."""

    institutional_summary_warn_coverage_pct: float = 50.0
    """Min institutional summary ticker coverage (%) before WARN. Never FAIL."""

    pnl_attribution_min_coverage_pct: float = 80.0
    """Min PnL attribution price coverage (%) before WARN. Never FAIL."""

    forward_eval_ic_warn_floor: float = 0.02
    """Mean IC floor for forward-eval gate. WARN if rolling IC drops below."""

    forward_eval_lookback_n: int = 10
    """Rolling window size (number of prior snapshot dates) for forward-eval."""

    forward_eval_horizon: int = 20
    """Forward-return horizon in trading days for the gate."""

    portfolio_weight_sum_tolerance: float = 1.0
    """Max absolute deviation from 100% for target_weight_pct sum (pp)."""

    sort_contrib_missing_max_frac: float = 0.01
    """WARN if > 1% of eligible rows have blank/NaN in any sort contrib column."""

    sort_contrib_sum_tolerance: float = 1e-6
    """WARN if any row's sum(contribs) != total_adj beyond this tolerance."""

    sort_contrib_hard_abs_max: float = 50.0
    """FAIL if any contrib or total_adj exceeds this absolute bound (or is non-finite)."""

    exposure_missing_warn_frac: float = 0.10
    """Per-exposure WARN threshold: fraction of eligible tickers missing a value."""

    exposure_missing_fail_frac: float = 0.25
    """Per-exposure FAIL threshold: fraction of eligible tickers missing a value."""

    risk_conc_catalyst_7d_warn: float = 0.30
    """WARN if > 30% of top-K weight has catalyst_days <= 7."""

    risk_conc_catalyst_7d_fail: float = 0.50
    """FAIL if > 50% of top-K weight has catalyst_days <= 7."""

    risk_conc_high_beta_warn: float = 0.50
    """WARN if > 50% of top-K weight has beta >= 1.5 or drawdown <= -0.30."""

    risk_conc_stacked_warn: float = 0.20
    """WARN if > 20% of top-K weight has catalyst <= 7d AND (beta >= 1.5 or drawdown <= -0.30)."""

    regulatory_calendar_min_coverage_pct: float = 3.0
    """Min regulatory coverage % of eligible tickers before WARN."""

    regulatory_calendar_max_coverage_pct: float = 12.0
    """Max regulatory coverage % of eligible tickers before WARN (over-flagging)."""

    regulatory_calendar_max_stale_days: int = 180
    """WARN if newest as_of_disclosed_at in manual calendar is older than this many days."""

    # --- Hard-catalyst production gates (Spec 018) ---

    hard_queue_min_warn: int = 8
    """Min hard-catalyst count before WARN."""

    hard_queue_min_fail: int = 3
    """Min hard-catalyst count before FAIL."""

    hard_queue_near_term_min_warn: int = 4
    """Min hard-catalyst within 90d before WARN."""

    hard_queue_near_term_min_fail: int = 1
    """Min hard-catalyst within 90d before FAIL (0 = FAIL)."""

    hard_options_coverage_warn_pct: float = 60.0
    """Min % of hard rows with opt_atm_iv before WARN."""

    hard_options_coverage_fail_pct: float = 40.0
    """Min % of hard rows with opt_atm_iv before FAIL."""

    hard_actual_straddle_warn_pct: float = 50.0
    """Min % of hard rows with actual straddle before WARN."""

    hard_actual_straddle_fail_pct: float = 20.0
    """Min % of hard rows with actual straddle before FAIL."""

    hard_reviewable_min_warn: int = 3
    """Min reviewable hard names before WARN."""

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


@dataclass(frozen=True)
class DriftThresholds:
    """Versioned thresholds for drift monitoring gate.

    WARN thresholds trigger warnings but allow snapshot promotion.
    FAIL thresholds block snapshot promotion when extreme drift is
    detected (indicating potential data corruption or pipeline error).
    Set fail_* to 0 (overlap) or a very large number (counts) to disable.
    """

    # WARN thresholds (non-blocking)
    warn_top20_overlap_pct: float = 70.0
    warn_top60_overlap_pct: float = 80.0
    warn_rank_spearman_rho: float = 0.90
    warn_mean_abs_rank_delta_top60: float = 8.0
    warn_tier_migration_count: int = 10
    warn_eligibility_change_count: int = 10

    # Stability plumbing diagnostics (observational — seed thresholds, tune after 30d)
    warn_action_change_pct: float = 25.0  # >25% of common tickers changed tier_dev
    warn_mean_abs_selector_score_delta: float = 0.08  # mean |selector_score_cur - prior|
    warn_feature_coverage_drop_pct: float = 10.0  # any feature coverage drop ≥10pp
    warn_near_miss_share_pct: float = 30.0  # >30% of top-30 within eps of K=30 cutoff

    # FAIL thresholds (blocking — extreme drift likely signals a bug)
    fail_top20_overlap_pct: float = 30.0  # <30% overlap = catastrophic churn
    fail_top60_overlap_pct: float = 50.0  # <50% overlap = data feed issue
    fail_rank_spearman_rho: float = 0.50  # <0.50 = rankings essentially scrambled
    fail_mean_abs_rank_delta_top60: float = 25.0  # >25 avg rank change = broken
    fail_tier_migration_count: int = 40  # >40 tier changes = data corruption
    fail_eligibility_change_count: int = 40  # >40 eligibility flips = broken

    # New-diagnostic FAIL thresholds — disabled (sentinel) until 30d observation
    fail_action_change_pct: float = -1.0
    fail_mean_abs_selector_score_delta: float = -1.0
    fail_feature_coverage_drop_pct: float = -1.0
    fail_near_miss_share_pct: float = -1.0

    @property
    def thresholds_id(self) -> str:
        blob = json.dumps(
            {k: getattr(self, k) for k in self.__dataclass_fields__},
            sort_keys=True,
        )
        return hashlib.sha256(blob.encode()).hexdigest()[:8]

    def to_json(self) -> dict:
        return {"thresholds_id": self.thresholds_id, **asdict(self)}

    @classmethod
    def from_json(cls, path: Path) -> "DriftThresholds":
        with open(path) as f:
            d = json.load(f)
        d.pop("thresholds_id", None)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Step 1: Price refresh
# ---------------------------------------------------------------------------

# Mode → baseline refresh breadth + warm sources. A mode never changes the
# model, ranker, selector, ruleset, or PIT integrity checks — only refresh
# breadth and snapshot location.
MODE_DEFAULTS = {
    "daily-validation": {
        "price_refresh_scope": "top30",
        "warm_sources": "ctgov",
    },
    "daily-production": {
        "price_refresh_scope": "full",
        "warm_sources": "sec_8k,ctgov,sec_13f,fda_adcom,fda_regulatory",
    },
    "full-refresh": {
        "price_refresh_scope": "full",
        "warm_sources": "sec_8k,ctgov,sec_13f,fda_adcom,fda_regulatory,euctr,ctis,isrctn,merged_trials",
    },
    "research": {
        "price_refresh_scope": "top30",
        "warm_sources": "ctgov",
    },
}


def resolve_validation_rank_depth_scope(
    validation_rank_depth: Optional[int],
    scope_explicitly_set: bool,
    current_scope: str,
) -> str:
    """Resolve the price-refresh scope after applying --validation-rank-depth.

    VALIDATION_INFRASTRUCTURE / NO_MODEL_CHANGE: governs refresh breadth only.
    Explicit --price-refresh-scope always wins. When a rank depth is given and
    scope was not set explicitly, widen to cover the cohort (>=60 -> top60,
    else top30). Otherwise the scope is left unchanged.
    """
    if validation_rank_depth is None or scope_explicitly_set:
        return current_scope
    return "top60" if validation_rank_depth >= 60 else "top30"


def _load_top_n_tickers_from_prior_snapshot(
    snapshot_dir: Path,
    as_of_date: str,
    top_n: int = 30,
) -> Optional[List[str]]:
    """Read the most recent prior snapshot's rankings.csv and return the top-N tickers.

    Returns None if no prior snapshot found or rankings.csv is missing/unreadable.
    Skips the as_of_date snapshot itself (looks for dates strictly before it).
    """
    try:
        prior_dates = sorted(
            d.name for d in snapshot_dir.iterdir() if d.is_dir() and d.name < as_of_date and len(d.name) == 10
        )
        if not prior_dates:
            return None
        prior_date = prior_dates[-1]
        rankings_path = snapshot_dir / prior_date / "rankings.csv"
        if not rankings_path.exists():
            return None
        tickers: List[str] = []
        with open(rankings_path, newline="", encoding="utf-8") as f:
            for i, row in enumerate(csv.DictReader(f)):
                if i >= top_n:
                    break
                t = (row.get("ticker") or "").strip().upper()
                if t:
                    tickers.append(t)
        return tickers if tickers else None
    except Exception:
        return None


def refresh_prices(
    price_csv: Path,
    through_date: str,
    universe_path: Optional[Path] = None,
    tickers_override: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Incrementally refresh price_history.csv via extend_price_csv().

    When tickers_override is provided, only those tickers (plus XBI) are
    refreshed. Use this for daily-validation mode where only Top-N + XBI
    prices are needed — full-universe refresh is deferred to weekly runs.

    Returns stats dict from extend_price_csv plus xbi_last_date.
    """
    # Import lazily to avoid yfinance dependency at module level
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from backtest_signal_robustness import extend_price_csv_safe as extend_price_csv

    if tickers_override is not None:
        # Scoped refresh: caller-supplied list (e.g. Top-30 + XBI)
        tickers = [t for t in tickers_override if t and not t.startswith("_")]
        if "XBI" not in tickers:
            tickers.append("XBI")
        _logger.info(f"  Scoped price refresh: {len(tickers)} tickers (Top-N + XBI)")
    else:
        # Full-universe refresh: collect tickers from universe.json
        tickers = None
        if universe_path and universe_path.exists():
            with open(universe_path) as f:
                universe = json.load(f)
            if isinstance(universe, list):
                tickers = [e.get("ticker", e) if isinstance(e, dict) else str(e) for e in universe]
            elif isinstance(universe, dict) and "tickers" in universe:
                tickers = universe["tickers"]
            if tickers:
                tickers = [t for t in tickers if t and not t.startswith("_")]
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


def validate_xbi_freshness(
    price_csv: Path,
    through_date: str,
    max_gap_days: int = 5,
) -> Dict[str, Any]:
    """Post-refresh validation: ensure XBI is fresh and re-fetch if needed.

    Returns validation result dict with validation status and any revalidation stats.
    """
    from datetime import datetime, timedelta

    xbi_last = _get_ticker_last_date(price_csv, "XBI")

    result = {
        "xbi_last_date": xbi_last,
        "through_date": through_date,
        "validation_passed": False,
        "revalidation_needed": False,
        "revalidation_stats": None,
    }

    if not xbi_last:
        result["validation_error"] = "XBI not found in price file after refresh"
        return result

    # Check staleness gap
    last_dt = datetime.strptime(xbi_last, "%Y-%m-%d")
    through_dt = datetime.strptime(through_date, "%Y-%m-%d")
    delta = through_dt - last_dt

    # Count trading days gap (weekdays only)
    trading_days = sum(1 for i in range(1, delta.days + 1) if (last_dt + timedelta(days=i)).weekday() < 5)

    result["trading_days_gap"] = trading_days
    result["max_gap_days"] = max_gap_days

    if trading_days <= max_gap_days:
        result["validation_passed"] = True
        _logger.info(f"✓ XBI validation PASS: {xbi_last} (gap={trading_days} days)")
        return result

    # XBI is stale — attempt targeted re-fetch
    _logger.warning(f"✗ XBI validation FAIL: last={xbi_last}, through={through_date}, gap={trading_days} days")
    result["revalidation_needed"] = True

    try:
        _logger.info(f"Attempting targeted XBI re-fetch from {xbi_last} through {through_date}...")

        import csv as csv_module
        import tempfile

        import yfinance as yf

        # Fetch XBI data from last known date to through_date
        start_dt = last_dt + timedelta(days=1)
        end_dt = through_dt + timedelta(days=1)

        xbi = yf.Ticker("XBI")
        hist = xbi.history(start=start_dt.strftime("%Y-%m-%d"), end=end_dt.strftime("%Y-%m-%d"))

        if hist.empty:
            result["revalidation_error"] = f"yfinance returned no XBI data for {start_dt.date()} to {through_dt.date()}"
            _logger.error(result["revalidation_error"])
            return result

        # Read existing price file
        existing_rows = []
        if price_csv.exists():
            with open(price_csv, "r", encoding="utf-8") as f:
                existing_rows = list(csv_module.DictReader(f))

        # Add new XBI rows
        new_rows = []
        for idx, row in hist.iterrows():
            dt = idx.strftime("%Y-%m-%d")
            new_rows.append(
                {
                    "date": dt,
                    "ticker": "XBI",
                    "open": str(row["Open"]),
                    "high": str(row["High"]),
                    "low": str(row["Low"]),
                    "close": str(row["Close"]),
                    "volume": str(int(row["Volume"])),
                }
            )

        # Deduplicate: keep last occurrence per (ticker, date)
        all_rows = existing_rows + new_rows
        seen = {}
        for i, row in enumerate(all_rows):
            key = ((row.get("ticker") or "").upper(), (row.get("date") or "")[:10])
            seen[key] = i
        deduped = [all_rows[i] for i in sorted(seen.values())]

        # Atomic write
        if new_rows:
            price_csv.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(suffix=".csv", dir=price_csv.parent)
            try:
                with open(fd, "w", newline="", encoding="utf-8") as f:
                    fieldnames = ["date", "ticker", "open", "high", "low", "close", "volume"]
                    writer = csv_module.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for row in deduped:
                        writer.writerow({k: row.get(k, "") for k in fieldnames})
                Path(tmp_path).replace(price_csv)

                _logger.info(f"✓ XBI re-fetch successful: added {len(new_rows)} rows")
                result["revalidation_stats"] = {
                    "rows_added": len(new_rows),
                    "xbi_last_after_revalidation": new_rows[-1]["date"],
                }

                # Re-check staleness
                new_xbi_last = _get_ticker_last_date(price_csv, "XBI")
                if new_xbi_last:
                    new_last_dt = datetime.strptime(new_xbi_last, "%Y-%m-%d")
                    new_trading_days = sum(
                        1
                        for i in range(1, (through_dt - new_last_dt).days + 1)
                        if (new_last_dt + timedelta(days=i)).weekday() < 5
                    )
                    if new_trading_days <= max_gap_days:
                        result["validation_passed"] = True
                        _logger.info(f"✓ XBI re-validation PASS: {new_xbi_last} (gap={new_trading_days} days)")
            except Exception as e:
                Path(tmp_path).unlink(missing_ok=True)
                result["revalidation_error"] = f"Failed to write XBI data: {e}"
                _logger.error(result["revalidation_error"])
    except Exception as e:
        result["revalidation_error"] = f"XBI re-fetch failed: {e}"
        _logger.error(result["revalidation_error"])

    return result


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
# Subprocess helpers — timeout + diagnostic capture
# ---------------------------------------------------------------------------

SUBPROCESS_TIMEOUT_SECONDS = 1200  # 20 minutes; prevents hung modules from burning CI budget

import logging as _logging

_logger = _logging.getLogger(__name__)


def _run_subprocess(
    cmd: List[str],
    *,
    label: str = "subprocess",
    timeout: int = SUBPROCESS_TIMEOUT_SECONDS,
    cwd: Optional[Path] = None,
) -> subprocess.CompletedProcess:
    """Run a subprocess with timeout and diagnostic capture.

    On failure (non-zero exit) or timeout, logs stderr/stdout for diagnosis.
    On timeout, raises ``subprocess.TimeoutExpired`` after logging context.
    """
    effective_cwd = str(cwd) if cwd else str(REPO_ROOT)
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=effective_cwd,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        _logger.error(
            "%s timed out after %ds. stdout (last 500 chars): %s | stderr (last 500 chars): %s",
            label,
            timeout,
            (exc.stdout or "")[-500:] if exc.stdout else "<none>",
            (exc.stderr or "")[-500:] if exc.stderr else "<none>",
        )
        _logger.error(f"{label} TIMED OUT after {timeout}s")
        raise

    if result.returncode != 0:
        _logger.warning(
            "%s exited %d. stderr (last 1000 chars): %s",
            label,
            result.returncode,
            (result.stderr or "")[-1000:],
        )
    elif label == "run_screen" and result.stderr:
        # For run_screen on success, log stderr (contains debug output from logger)
        _logger.info("%s subprocess output: %s", label, (result.stderr or "")[-2000:])

    return result


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
    prior_snapshot_dir: Optional[Path] = None,
) -> subprocess.CompletedProcess:
    """Run run_screen.py in phase2 mode with decision ranking."""
    # run_screen.py requires --output for the raw JSON results
    output_json = snapshot_dir / as_of_date / "screen_output.json"
    # Do NOT pre-create output_json.parent here — run_screen.py creates it after
    # its overwrite-policy check. Pre-creating triggers "Snapshot already exists".
    cmd = [
        sys.executable,
        str(REPO_ROOT / "run_screen.py"),
        "--as-of-date",
        as_of_date,
        "--data-dir",
        str(data_dir),
        "--output",
        str(output_json),
        "--decision-mode",
        "phase2",
        "--ranking-mode",
        "decision",
        "--snapshot-dir",
        str(snapshot_dir),
        "--strict",
        "--inputs-manifest",
        "write",
    ]
    if ruleset_path:
        cmd.extend(["--ruleset", str(ruleset_path)])
    if prior_snapshot_dir:
        cmd.extend(["--prior-snapshot-dir", str(prior_snapshot_dir)])
    if extra_args:
        cmd.extend(extra_args)

    return _run_subprocess(cmd, label="run_screen")


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
        sys.executable,
        str(REPO_ROOT / "tools" / "data_integrity_audit.py"),
        "--snapshot-dir",
        str(snapshot_date_dir),
        "--price-history",
        str(price_csv),
        "--as-of-date",
        as_of_date,
        "--output-dir",
        str(output_dir),
    ]
    return _run_subprocess(cmd, label="run_audit")


# ---------------------------------------------------------------------------
# Step 4: Hard gates
# ---------------------------------------------------------------------------


def check_trading_day(as_of_date: str) -> GateResult:
    """FAIL if as_of_date falls on a weekend.

    Prevents the pipeline from producing degenerate snapshots and
    performance rows on non-trading days (blank prices → blank excess →
    poisoned alpha-health window).
    """
    dt = datetime.strptime(as_of_date, "%Y-%m-%d")
    weekday = dt.weekday()  # 0=Mon .. 6=Sun
    if weekday >= 5:
        day_name = "Saturday" if weekday == 5 else "Sunday"
        return GateResult(
            name="trading_day",
            status="FAIL",
            detail=f"{as_of_date} is {day_name} — not a trading day",
            value={"weekday": weekday, "day_name": day_name},
            threshold=None,
        )
    return GateResult(
        name="trading_day",
        status="PASS",
        detail=f"{as_of_date} is a weekday (trading day)",
        value={"weekday": weekday},
        threshold=None,
    )


def check_xbi_staleness(
    price_csv: Path,
    as_of_date: str,
    threshold_days: int,
) -> GateResult:
    """Check if XBI data is stale beyond threshold."""
    xbi_last = _get_ticker_last_date(price_csv, "XBI")
    if xbi_last is None:
        return GateResult(
            name="xbi_staleness",
            status="FAIL",
            detail="XBI not found in price_history.csv",
            value=None,
            threshold=threshold_days,
        )

    # Count trading days gap (approximate: weekdays only)
    from datetime import timedelta

    last_dt = datetime.strptime(xbi_last, "%Y-%m-%d")
    as_of_dt = datetime.strptime(as_of_date, "%Y-%m-%d")
    delta = as_of_dt - last_dt
    # Approximate trading days (exclude weekends)
    trading_days = sum(1 for i in range(1, delta.days + 1) if (last_dt + timedelta(days=i)).weekday() < 5)

    if trading_days > threshold_days:
        return GateResult(
            name="xbi_staleness",
            status="FAIL",
            detail=f"XBI last={xbi_last}, as_of={as_of_date}, gap={trading_days} trading days",
            value=trading_days,
            threshold=threshold_days,
        )
    return GateResult(
        name="xbi_staleness",
        status="PASS",
        detail=f"XBI last={xbi_last}, gap={trading_days} trading days",
        value=trading_days,
        threshold=threshold_days,
    )


# ---------------------------------------------------------------------------
# Source-specific freshness policy (item #4)
# ---------------------------------------------------------------------------
# TTLs are calendar days. Exceeding a TTL is a NON-BLOCKING WARN — it surfaces
# stale-source drag without failing the run. Core sources (prices, ctgov) also
# have dedicated FAIL gates upstream; this gate adds whole-pipeline visibility,
# especially for slow foreign registries that should refresh weekly, not daily.
# Alpha-preserving rule: stale sources must be VISIBLE, never silently used.
FRESHNESS_TTLS: Dict[str, int] = {
    "prices": 4,  # daily (trading-day gap; XBI also has a hard FAIL gate)
    "market_data": 5,  # daily
    "ctgov": 2,  # daily clinical delta
    "sec_8k": 4,  # every few days
    "sec_13f": 100,  # quarterly-aware
    "fda_adcom": 7,  # weekly
    "euctr": 14,  # weekly unless catalyst-critical
    "ctis": 14,
    "isrctn": 14,
}


def _latest_dated_file_age(
    directory: Path,
    glob_pattern: str,
    as_of_date: str,
) -> Optional[int]:
    """Age in calendar days of the newest dated file (<= as_of_date) under directory.

    Filenames must embed a YYYY-MM-DD token. Returns None if directory missing
    or no qualifying file found.
    """
    import re as _re

    if not directory.exists():
        return None
    _date_re = _re.compile(r"(\d{4}-\d{2}-\d{2})")
    best: Optional[str] = None
    for p in directory.glob(glob_pattern):
        m = _date_re.search(p.name)
        if not m:
            continue
        d = m.group(1)
        if d <= as_of_date and (best is None or d > best):
            best = d
    if best is None:
        return None
    return (datetime.strptime(as_of_date, "%Y-%m-%d") - datetime.strptime(best, "%Y-%m-%d")).days


def check_source_freshness(
    as_of_date: str,
    *,
    price_csv: Path,
    data_dir: Path,
    ctgov_cache_dir: Optional[Path] = None,
    repo_root: Path = REPO_ROOT,
) -> GateResult:
    """Per-source freshness audit (non-blocking WARN).

    Computes the age of each tracked source against its TTL and reports any that
    are stale. Sources with no cache present are SKIPPED, not penalized, so an
    absent optional registry never creates noise. Never returns FAIL.
    """
    ages: Dict[str, Any] = {}
    stale: List[str] = []
    skipped: List[str] = []

    # prices: XBI last date in the CSV (calendar-day gap)
    _xbi_last = _get_ticker_last_date(price_csv, "XBI")
    if _xbi_last:
        ages["prices"] = (datetime.strptime(as_of_date, "%Y-%m-%d") - datetime.strptime(_xbi_last, "%Y-%m-%d")).days
    else:
        skipped.append("prices")

    # market_data: collected_at age via mtime fallback
    _mkt = data_dir / "market_data.json"
    if _mkt.exists():
        ages["market_data"] = (
            datetime.strptime(as_of_date, "%Y-%m-%d").date() - datetime.fromtimestamp(_mkt.stat().st_mtime).date()
        ).days
    else:
        skipped.append("market_data")

    # Dated-glob sources: (source, directory, glob)
    _ctgov_dir = ctgov_cache_dir or (repo_root / "cache" / "ctgov")
    _dated_specs = [
        ("ctgov", _ctgov_dir, "trial_records_*.json"),
        ("sec_8k", repo_root / "cache" / "sec" / "8k_catalysts", "8k_catalysts_*.json"),
        ("fda_adcom", repo_root / "cache" / "fda", "adcom_calendar_*.json"),
        ("euctr", repo_root / "cache" / "trials" / "euctr", "euctr_*.json"),
        ("ctis", repo_root / "cache" / "trials" / "ctis", "ctis_*.json"),
        ("isrctn", repo_root / "cache" / "trials" / "isrctn", "isrctn_*.json"),
    ]
    for _src, _dir, _glob in _dated_specs:
        _age = _latest_dated_file_age(_dir, _glob, as_of_date)
        if _age is None:
            skipped.append(_src)
        else:
            ages[_src] = _age

    # Compare each measured source to its TTL
    for _src, _age in ages.items():
        _ttl = FRESHNESS_TTLS.get(_src)
        if _ttl is not None and isinstance(_age, int) and _age > _ttl:
            stale.append(f"{_src}={_age}d>{_ttl}d")

    if stale:
        return GateResult(
            name="source_freshness",
            status="WARN",
            detail="Stale sources: " + ", ".join(sorted(stale)) + (f"; skipped={sorted(skipped)}" if skipped else ""),
            value={"ages": ages, "stale": stale, "skipped": skipped},
            threshold=FRESHNESS_TTLS,
        )
    return GateResult(
        name="source_freshness",
        status="PASS",
        detail=f"All measured sources within TTL ({len(ages)} checked, {len(skipped)} skipped)",
        value={"ages": ages, "stale": [], "skipped": skipped},
        threshold=FRESHNESS_TTLS,
    )


def check_missing_reason_fraction(
    snapshot_date_dir: Path,
    max_frac: float,
) -> GateResult:
    """Check fraction of tickers with non-empty missing_reason for DE-critical fields."""
    rankings_path = snapshot_date_dir / "rankings.csv"
    if not rankings_path.exists():
        return GateResult(
            name="missing_reason_fraction",
            status="FAIL",
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
            name="missing_reason_fraction",
            status="FAIL",
            detail="rankings.csv is empty",
        )

    frac = missing_count / total
    status = "FAIL" if frac > max_frac else "PASS"
    return GateResult(
        name="missing_reason_fraction",
        status=status,
        detail=f"{missing_count}/{total} tickers ({frac:.1%}) have missing_reason",
        value=round(frac, 4),
        threshold=max_frac,
    )


_EXPOSURE_COLUMNS_BASE: tuple[str, ...] = (
    "de_beta_xbi_60d",
    "de_drawdown",
    "de_rsi_14d",
    "de_alpha_60d",
)
"""Always-expected exposure value columns (present in all post-DE v1.3 snapshots)."""

_EXPOSURE_COLUMNS_EXTENDED: tuple[str, ...] = (
    "de_vol_60d",
    "de_drawdown_rel_xbi",
)
"""Columns only checked if they exist in the CSV header (post-fix additions)."""


def check_exposure_missingness(
    snapshot_date_dir: Path,
    warn_frac: float,
    fail_frac: float,
    *,
    held_tickers: Optional[set] = None,
    top_k: int = 20,
) -> GateResult:
    """Check per-exposure missingness among held/top-K tickers.

    Schema-aware: columns absent from the CSV header are **skipped** (not
    counted as missing).  This prevents false FAILs on legacy snapshots that
    predate a column's introduction.

    When ``held_tickers`` is provided, the check is scoped to tickers that
    are either in the held set or in the top-K by actionable_rank.  This
    avoids false WARNs from universe names that are never traded.

    For each column present in the header, computes the fraction of scoped
    tickers whose value is empty / NaN.  The *worst* column determines the
    gate verdict:

    - FAIL if worst fraction > ``fail_frac``
    - WARN if worst fraction > ``warn_frac``
    - PASS otherwise

    ``GateResult.value`` is a dict with ``eligible_n``, ``scoped_n``,
    ``missing_fracs``, and ``skipped`` (columns not in the header).
    """
    rankings_path = snapshot_date_dir / "rankings.csv"
    if not rankings_path.exists():
        return GateResult(
            name="exposure_missingness",
            status="FAIL",
            detail="rankings.csv not found",
        )

    with open(rankings_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = set(reader.fieldnames or [])
        eligible_rows = [r for r in reader if r.get("eligible") == "1"]

    n_eligible = len(eligible_rows)
    if n_eligible == 0:
        return GateResult(
            name="exposure_missingness",
            status="FAIL",
            detail="No eligible tickers in rankings.csv",
        )

    # Scope to held + top-K names (avoids false alarms from untradeable names)
    if held_tickers is not None:
        top_k_tickers = set()
        for r in eligible_rows:
            ar = (r.get("actionable_rank") or "").strip()
            try:
                if int(ar) <= top_k:
                    top_k_tickers.add(r.get("ticker", "").upper())
            except (ValueError, TypeError):
                pass
        scope_tickers = held_tickers | top_k_tickers
        scoped_rows = [r for r in eligible_rows if r.get("ticker", "").upper() in scope_tickers]
    else:
        scoped_rows = eligible_rows

    n = len(scoped_rows)
    if n == 0:
        return GateResult(
            name="exposure_missingness",
            status="PASS",
            detail=f"No held/top-K tickers to check ({n_eligible} eligible total)",
            value={"eligible_n": n_eligible, "scoped_n": 0, "missing_fracs": {}, "skipped": []},
            threshold=warn_frac,
        )

    # Determine which columns to check (skip absent ones)
    all_candidates = list(_EXPOSURE_COLUMNS_BASE) + list(_EXPOSURE_COLUMNS_EXTENDED)
    check_cols = [c for c in all_candidates if c in header]
    skipped_cols = [c for c in all_candidates if c not in header]

    if not check_cols:
        return GateResult(
            name="exposure_missingness",
            status="PASS",
            detail=f"SKIP: no exposure columns in header ({n} eligible, legacy schema)",
            value={"eligible_n": n, "missing_fracs": {}, "skipped": skipped_cols},
            threshold=warn_frac,
        )

    worst_col = ""
    worst_frac = 0.0
    missing_fracs: dict[str, float] = {}
    per_col: list[str] = []

    scope_label = f"{n} held/top-K" if held_tickers is not None else f"{n} eligible"

    for col in check_cols:
        missing = 0
        for row in scoped_rows:
            val = (row.get(col) or "").strip()
            if not val or val.lower() in ("nan", "none"):
                missing += 1
        frac = missing / n
        missing_fracs[col] = round(frac, 4)
        per_col.append(f"{col}={frac:.1%}")
        if frac > worst_frac:
            worst_frac = frac
            worst_col = col

    if skipped_cols:
        per_col.append(f"SKIP: {','.join(skipped_cols)}")

    summary = ", ".join(per_col)
    value_dict = {
        "eligible_n": n_eligible,
        "scoped_n": n,
        "missing_fracs": missing_fracs,
        "skipped": skipped_cols,
    }

    if worst_frac > fail_frac:
        return GateResult(
            name="exposure_missingness",
            status="FAIL",
            detail=f"Worst: {worst_col} ({worst_frac:.1%} missing > {fail_frac:.0%}). {summary}",
            value=value_dict,
            threshold=fail_frac,
        )
    if worst_frac > warn_frac:
        return GateResult(
            name="exposure_missingness",
            status="WARN",
            detail=f"Worst: {worst_col} ({worst_frac:.1%} missing > {warn_frac:.0%}). {summary}",
            value=value_dict,
            threshold=warn_frac,
        )
    return GateResult(
        name="exposure_missingness",
        status="PASS",
        detail=f"All exposures OK ({scope_label}). {summary}",
        value=value_dict,
        threshold=warn_frac,
    )


def check_risk_concentration(
    snapshot_date_dir: Path,
    top_k: int = 20,
    *,
    catalyst_7d_warn: float = 0.30,
    catalyst_7d_fail: float = 0.50,
    high_risk_warn: float = 0.50,
    stacked_warn: float = 0.20,
    beta_threshold: float = 1.5,
    drawdown_threshold: float = -0.30,
) -> GateResult:
    """Check risk concentration in the top-K portfolio.

    Three buckets (all measured as fraction of top-K target weight):

    1. **Catalyst <=7d**: tickers with imminent catalysts (binary outcome risk).
    2. **High-risk**: tickers with beta >= threshold OR drawdown <= threshold.
    3. **Stacked risk**: tickers in BOTH bucket 1 AND bucket 2.

    Verdicts (worst wins):

    - FAIL if catalyst_7d weight > ``catalyst_7d_fail``
    - WARN if any bucket breaches its warn threshold
    - PASS otherwise

    ``GateResult.value`` is a dict with per-bucket fractions.
    """
    rankings_path = snapshot_date_dir / "rankings.csv"
    if not rankings_path.exists():
        return GateResult(
            name="risk_concentration",
            status="FAIL",
            detail="rankings.csv not found",
        )

    with open(rankings_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Select top-K by actionable_rank
    ranked = []
    for r in rows:
        ar = r.get("actionable_rank", "")
        if ar and ar.isdigit():
            ranked.append((int(ar), r))
    ranked.sort(key=lambda x: x[0])
    topk = [r for _, r in ranked[:top_k]]

    if not topk:
        return GateResult(
            name="risk_concentration",
            status="PASS",
            detail="No ranked tickers; skipped",
            value={"catalyst_7d_wt": 0.0, "high_risk_wt": 0.0, "stacked_wt": 0.0},
        )

    def _fval(row: dict, col: str, default: float = 0.0) -> float:
        v = row.get(col, "")
        if not v or str(v).lower() in ("", "nan", "none"):
            return default
        try:
            return float(v)
        except (ValueError, TypeError):
            return default

    # Compute per-ticker weights (normalised to sum=1 within top-K)
    raw_weights = [_fval(r, "target_weight_pct", 0.0) for r in topk]
    total_wt = sum(raw_weights)
    if total_wt <= 0:
        # Fall back to equal weight
        weights = [1.0 / len(topk)] * len(topk)
    else:
        weights = [w / total_wt for w in raw_weights]

    cat7d_wt = 0.0
    high_risk_wt = 0.0
    stacked_wt = 0.0

    for r, w in zip(topk, weights):
        cat_days = _fval(r, "catalyst_days", 9999)
        beta = _fval(r, "de_beta_xbi_60d", 0.0)
        dd = _fval(r, "de_drawdown", 0.0)

        is_cat7 = cat_days <= 7
        is_high_risk = beta >= beta_threshold or dd <= drawdown_threshold

        if is_cat7:
            cat7d_wt += w
        if is_high_risk:
            high_risk_wt += w
        if is_cat7 and is_high_risk:
            stacked_wt += w

    value_dict = {
        "catalyst_7d_wt": round(cat7d_wt, 4),
        "high_risk_wt": round(high_risk_wt, 4),
        "stacked_wt": round(stacked_wt, 4),
        "top_k": len(topk),
    }

    # Determine status (worst wins)
    status = "PASS"
    detail_parts: list[str] = []

    if cat7d_wt > catalyst_7d_fail:
        status = "FAIL"
        detail_parts.append(f"catalyst<=7d={cat7d_wt:.0%} > {catalyst_7d_fail:.0%} FAIL")
    elif cat7d_wt > catalyst_7d_warn:
        status = "WARN"
        detail_parts.append(f"catalyst<=7d={cat7d_wt:.0%} > {catalyst_7d_warn:.0%} WARN")
    else:
        detail_parts.append(f"catalyst<=7d={cat7d_wt:.0%}")

    if high_risk_wt > high_risk_warn:
        if status != "FAIL":
            status = "WARN"
        detail_parts.append(f"high_risk={high_risk_wt:.0%} > {high_risk_warn:.0%} WARN")
    else:
        detail_parts.append(f"high_risk={high_risk_wt:.0%}")

    if stacked_wt > stacked_warn:
        if status != "FAIL":
            status = "WARN"
        detail_parts.append(f"stacked={stacked_wt:.0%} > {stacked_warn:.0%} WARN")
    else:
        detail_parts.append(f"stacked={stacked_wt:.0%}")

    detail = "; ".join(detail_parts)
    threshold = catalyst_7d_fail if status == "FAIL" else catalyst_7d_warn

    return GateResult(
        name="risk_concentration",
        status=status,
        detail=detail,
        value=value_dict,
        threshold=threshold,
    )


def check_turnover(
    snapshot_date_dir: Path,
    max_pct: float,
) -> GateResult:
    """Check name turnover from the delta report."""
    delta_path = snapshot_date_dir / "phase2_run_delta_report.txt"
    if not delta_path.exists():
        return GateResult(
            name="turnover",
            status="PASS",
            detail="No delta report (first run or --no-delta); skipped",
        )

    text = delta_path.read_text()
    # Parse "Name turnover: XX.X%" from the report
    import re

    match = re.search(r"Name turnover:\s+([\d.]+)%", text)
    if not match:
        return GateResult(
            name="turnover",
            status="PASS",
            detail="Could not parse turnover from delta report; skipped",
        )

    turnover = float(match.group(1))
    status = "FAIL" if turnover > max_pct else "PASS"
    return GateResult(
        name="turnover",
        status=status,
        detail=f"Name turnover={turnover:.1f}%",
        value=turnover,
        threshold=max_pct,
    )


# ---------------------------------------------------------------------------
# Drift monitoring gate (WARN-only)
# ---------------------------------------------------------------------------


def _find_prior_snapshot(snapshot_dir: Path, current_date: str) -> Optional[Path]:
    """Find most recent prior snapshot with valid rankings.csv containing tier_dev."""
    import re

    date_re = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    all_dates = sorted(
        [d.name for d in snapshot_dir.iterdir() if d.is_dir() and date_re.match(d.name)],
        reverse=True,
    )
    try:
        idx = all_dates.index(current_date)
    except ValueError:
        # current_date not in list — treat all dates before it as candidates
        all_dates = [d for d in all_dates if d < current_date]
        idx = -1

    candidates = all_dates[idx + 1 :] if idx >= 0 else all_dates
    for candidate in candidates:
        rankings_csv = snapshot_dir / candidate / "rankings.csv"
        if not rankings_csv.exists():
            continue
        with open(rankings_csv, "r") as f:
            header = f.readline().strip()
        if "tier_dev" in header.split(","):
            return snapshot_dir / candidate
    return None


def _compute_drift_metrics(
    current_csv: Path,
    prior_csv: Path,
) -> Dict[str, Any]:
    """Compute drift metrics between two rankings.csv files.

    Returns dict with overlap, rank stability, tier/eligibility changes.
    """
    import math

    def _read_rankings(path: Path) -> Dict[str, Dict[str, str]]:
        rows = {}
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ticker = row.get("ticker", "").strip()
                if ticker:
                    rows[ticker] = dict(row)
        return rows

    def _rank_col(sample_row: Dict[str, str]) -> str:
        return "actionable_rank" if "actionable_rank" in sample_row else "composite_rank"

    cur = _read_rankings(current_csv)
    pri = _read_rankings(prior_csv)

    cur_rank_col = _rank_col(next(iter(cur.values()))) if cur else "actionable_rank"
    pri_rank_col = _rank_col(next(iter(pri.values()))) if pri else "actionable_rank"

    # Parse ranks
    def _get_rank(rows: Dict, ticker: str, col: str) -> Optional[float]:
        val = rows.get(ticker, {}).get(col, "")
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    def _get_numeric(rows: Dict, ticker: str, col: str) -> Optional[float]:
        val = rows.get(ticker, {}).get(col, "")
        if val is None:
            return None
        sval = str(val).strip()
        if sval in ("", "nan", "NaN", "None"):
            return None
        try:
            return float(sval)
        except (ValueError, TypeError):
            return None

    common = set(cur.keys()) & set(pri.keys())

    # Top-N overlap (Jaccard)
    def _top_n_overlap(n: int) -> float:
        cur_ranked = sorted(
            [(t, _get_rank(cur, t, cur_rank_col)) for t in cur],
            key=lambda x: (x[1] if x[1] is not None else 1e9),
        )
        pri_ranked = sorted(
            [(t, _get_rank(pri, t, pri_rank_col)) for t in pri],
            key=lambda x: (x[1] if x[1] is not None else 1e9),
        )
        cur_top = {t for t, _ in cur_ranked[:n]}
        pri_top = {t for t, _ in pri_ranked[:n]}
        union = cur_top | pri_top
        if not union:
            return 100.0
        return round(100.0 * len(cur_top & pri_top) / len(union), 2)

    top20_overlap = _top_n_overlap(20)
    top60_overlap = _top_n_overlap(60)

    # Spearman rho (manual Pearson-of-ranks, no scipy)
    rho = None
    if common:
        cur_ranks_list = []
        pri_ranks_list = []
        for t in sorted(common):
            cr = _get_rank(cur, t, cur_rank_col)
            pr = _get_rank(pri, t, pri_rank_col)
            if cr is not None and pr is not None:
                cur_ranks_list.append(cr)
                pri_ranks_list.append(pr)

        if len(cur_ranks_list) >= 2:
            n = len(cur_ranks_list)
            mean_c = sum(cur_ranks_list) / n
            mean_p = sum(pri_ranks_list) / n
            cov = sum((c - mean_c) * (p - mean_p) for c, p in zip(cur_ranks_list, pri_ranks_list))
            var_c = sum((c - mean_c) ** 2 for c in cur_ranks_list)
            var_p = sum((p - mean_p) ** 2 for p in pri_ranks_list)
            denom = math.sqrt(var_c * var_p)
            rho = round(cov / denom, 4) if denom > 0 else 1.0

    # Mean abs rank delta for top-60 (current top-60)
    cur_ranked_all = sorted(
        [(t, _get_rank(cur, t, cur_rank_col)) for t in cur],
        key=lambda x: (x[1] if x[1] is not None else 1e9),
    )
    cur_top60 = {t for t, _ in cur_ranked_all[:60]}
    abs_deltas = []
    for t in cur_top60:
        cr = _get_rank(cur, t, cur_rank_col)
        pr = _get_rank(pri, t, pri_rank_col)
        if cr is not None and pr is not None:
            abs_deltas.append(abs(cr - pr))
    mean_abs_delta = round(sum(abs_deltas) / len(abs_deltas), 2) if abs_deltas else 0.0

    # Tier migration count
    tier_migrations = 0
    for t in common:
        cur_tier = (cur[t].get("tier_dev") or "").strip()
        pri_tier = (pri[t].get("tier_dev") or "").strip()
        if cur_tier and pri_tier and cur_tier != pri_tier:
            tier_migrations += 1

    # Eligibility change count
    elig_changes = 0
    for t in common:
        cur_elig = (cur[t].get("eligible") or "").strip().lower()
        pri_elig = (pri[t].get("eligible") or "").strip().lower()
        if cur_elig and pri_elig and cur_elig != pri_elig:
            elig_changes += 1

    # Top-20 entrants / exits
    cur_top20 = {t for t, _ in cur_ranked_all[:20]}
    pri_ranked_all = sorted(
        [(t, _get_rank(pri, t, pri_rank_col)) for t in pri],
        key=lambda x: (x[1] if x[1] is not None else 1e9),
    )
    pri_top20 = {t for t, _ in pri_ranked_all[:20]}
    top20_entrants = sorted(cur_top20 - pri_top20)
    top20_exits = sorted(pri_top20 - cur_top20)

    # --- Stability plumbing diagnostics (action, score, coverage, near-miss) ---

    # 1) Action transition matrix (prior tier_dev -> current tier_dev)
    transition_matrix: Dict[str, Dict[str, int]] = {}
    for t in common:
        pri_tier = (pri[t].get("tier_dev") or "").strip() or "_NA"
        cur_tier = (cur[t].get("tier_dev") or "").strip() or "_NA"
        transition_matrix.setdefault(pri_tier, {}).setdefault(cur_tier, 0)
        transition_matrix[pri_tier][cur_tier] += 1
    action_change_count = sum(
        cnt for from_tier, row in transition_matrix.items() for to_tier, cnt in row.items() if from_tier != to_tier
    )
    action_change_pct = round(100.0 * action_change_count / len(common), 2) if common else 0.0

    # 2) Mean absolute score delta across common tickers
    def _score_stats(col: str) -> Optional[Dict[str, float]]:
        deltas: List[float] = []
        for t in common:
            c = _get_numeric(cur, t, col)
            p = _get_numeric(pri, t, col)
            if c is not None and p is not None:
                deltas.append(abs(c - p))
        if not deltas:
            return None
        deltas_sorted = sorted(deltas)
        n = len(deltas_sorted)
        p95_idx = min(n - 1, int(round(0.95 * (n - 1))))
        return {
            "n": n,
            "mean_abs_delta": round(sum(deltas) / n, 6),
            "p95_abs_delta": round(deltas_sorted[p95_idx], 6),
            "max_abs_delta": round(deltas_sorted[-1], 6),
        }

    score_delta_stats: Dict[str, Dict[str, float]] = {}
    for col in ("selector_score", "final_score", "composite_score"):
        stats = _score_stats(col)
        if stats is not None:
            score_delta_stats[col] = stats

    # 3) Feature coverage change (% populated per field, cur vs prior)
    # Flags plumbing regressions: if coverage for an input drops, downstream
    # scores move for plumbing reasons rather than market reasons.
    ID_COLS = {
        "ticker",
        "company_name",
        "actionable_rank",
        "composite_rank",
        "decision_engine_version",
        "decision_engine_ruleset_id",
    }

    def _coverage_pct(rows: Dict[str, Dict[str, str]], col: str) -> float:
        if not rows:
            return 0.0
        populated = sum(1 for r in rows.values() if (r.get(col) or "").strip() not in ("", "nan", "NaN", "None"))
        return round(100.0 * populated / len(rows), 2)

    all_cols = set()
    for r in cur.values():
        all_cols.update(r.keys())
    for r in pri.values():
        all_cols.update(r.keys())
    tracked_cols = sorted(c for c in all_cols if c not in ID_COLS)

    coverage_cur: Dict[str, float] = {}
    coverage_pri: Dict[str, float] = {}
    coverage_delta: Dict[str, float] = {}
    for col in tracked_cols:
        cc = _coverage_pct(cur, col)
        cp = _coverage_pct(pri, col)
        coverage_cur[col] = cc
        coverage_pri[col] = cp
        coverage_delta[col] = round(cc - cp, 2)

    # Report biggest drops (negative deltas) — these are plumbing red flags
    drops_sorted = sorted(coverage_delta.items(), key=lambda kv: kv[1])
    coverage_drops_top5 = [
        {"feature": f, "cur_pct": coverage_cur[f], "prior_pct": coverage_pri[f], "delta_pp": d}
        for f, d in drops_sorted[:5]
        if d < 0
    ]
    max_coverage_drop_pp = abs(drops_sorted[0][1]) if drops_sorted and drops_sorted[0][1] < 0 else 0.0
    max_coverage_drop_feature = drops_sorted[0][0] if drops_sorted and drops_sorted[0][1] < 0 else None

    # 4) Near-miss threshold fragility at K=20 and K=30 cutoffs
    # For each score column, find the score at rank K; count names within
    # absolute score epsilon of that cutoff. A high near-miss count means
    # the day's ranking is fragile — a small input perturbation could
    # flip a number of names across the action boundary.
    near_miss: Dict[str, Any] = {}
    NEAR_MISS_EPSILONS = [0.001, 0.0025, 0.005, 0.01]
    for score_col, rank_col in (("selector_score", cur_rank_col), ("final_score", cur_rank_col)):
        # Check column presence
        sample = next(iter(cur.values()), None)
        if sample is None or score_col not in sample:
            continue
        scored = []
        for t in cur:
            r = _get_rank(cur, t, rank_col)
            s = _get_numeric(cur, t, score_col)
            if r is not None and s is not None:
                scored.append((r, s))
        if not scored:
            continue
        scored.sort(key=lambda x: x[0])
        col_stats: Dict[str, Any] = {}
        for k in (20, 30):
            if len(scored) < k:
                continue
            cutoff_score = scored[k - 1][1]
            eps_counts = {
                f"within_{int(eps * 10000)}bps": sum(1 for _, s in scored if abs(s - cutoff_score) <= eps)
                for eps in NEAR_MISS_EPSILONS
            }
            # Share of top-K that is within the tightest eps (plumbing fragility)
            tight_eps = NEAR_MISS_EPSILONS[1]  # 25bps absolute
            near_in_topk = sum(1 for r, s in scored[:k] if abs(s - cutoff_score) <= tight_eps)
            col_stats[f"K{k}"] = {
                "cutoff_score": round(cutoff_score, 6),
                "eps_counts": eps_counts,
                "share_topk_within_25bps_pct": round(100.0 * near_in_topk / k, 2),
            }
        if col_stats:
            near_miss[score_col] = col_stats

    # Headline near-miss share for threshold evaluation: selector_score K=30
    near_miss_share_pct = 0.0
    if "selector_score" in near_miss and "K30" in near_miss["selector_score"]:
        near_miss_share_pct = near_miss["selector_score"]["K30"]["share_topk_within_25bps_pct"]

    return {
        "top20_overlap_pct": top20_overlap,
        "top60_overlap_pct": top60_overlap,
        "rank_spearman_rho": rho,
        "mean_abs_rank_delta_top60": mean_abs_delta,
        "tier_migration_count": tier_migrations,
        "eligibility_change_count": elig_changes,
        "top20_entrants": top20_entrants,
        "top20_exits": top20_exits,
        "rank_column_current": cur_rank_col,
        "rank_column_prior": pri_rank_col,
        "n_common_tickers": len(common),
        # Stability plumbing diagnostics
        "action_transition_matrix": transition_matrix,
        "action_change_count": action_change_count,
        "action_change_pct": action_change_pct,
        "score_delta_stats": score_delta_stats,
        "coverage_delta": coverage_delta,
        "coverage_drops_top5": coverage_drops_top5,
        "max_coverage_drop_pp": round(max_coverage_drop_pp, 2),
        "max_coverage_drop_feature": max_coverage_drop_feature,
        "near_miss": near_miss,
        "near_miss_share_pct": near_miss_share_pct,
    }


def _write_drift_report_md(
    report: Dict[str, Any],
    path: Path,
) -> None:
    """Write human-readable drift report markdown."""
    m = report["metrics"]
    lines = [
        f"# Drift Report: {report['current_date']} vs {report['prior_date']}",
        "",
        f"**Status**: {report['status']}",
        f"**Thresholds ID**: `{report['thresholds_id']}`",
        "",
        "## Metrics",
        "",
        "| Metric | Value | Threshold | Status |",
        "|--------|-------|-----------|--------|",
    ]

    thresh = report.get("thresholds", {})
    reasons = set(report.get("warn_reasons", []))

    def _row(label: str, val, th_key: str, invert: bool = False):
        v = val if val is not None else "N/A"
        th = thresh.get(th_key, "")
        triggered = any(th_key in r for r in reasons)
        st = "WARN" if triggered else "OK"
        lines.append(f"| {label} | {v} | {th} | {st} |")

    _row("Top-20 overlap %", m.get("top20_overlap_pct"), "warn_top20_overlap_pct")
    _row("Top-60 overlap %", m.get("top60_overlap_pct"), "warn_top60_overlap_pct")
    _row("Spearman rho", m.get("rank_spearman_rho"), "warn_rank_spearman_rho")
    _row("Mean |rank delta| top-60", m.get("mean_abs_rank_delta_top60"), "warn_mean_abs_rank_delta_top60")
    _row("Tier migrations", m.get("tier_migration_count"), "warn_tier_migration_count")
    _row("Eligibility changes", m.get("eligibility_change_count"), "warn_eligibility_change_count")
    _row("Action change %", m.get("action_change_pct"), "warn_action_change_pct")
    sel_stats = (m.get("score_delta_stats") or {}).get("selector_score") or {}
    if sel_stats.get("mean_abs_delta") is not None:
        _row(
            "Mean |selector_score delta|",
            sel_stats.get("mean_abs_delta"),
            "warn_mean_abs_selector_score_delta",
        )
    _row("Max feature coverage drop (pp)", m.get("max_coverage_drop_pp"), "warn_feature_coverage_drop_pct")
    _row("Near-miss share @K=30 (25bps)", m.get("near_miss_share_pct"), "warn_near_miss_share_pct")

    lines.append("")

    if m.get("top20_entrants"):
        lines.append(f"**Top-20 entrants**: {', '.join(m['top20_entrants'])}")
    if m.get("top20_exits"):
        lines.append(f"**Top-20 exits**: {', '.join(m['top20_exits'])}")

    # Action transition matrix
    transitions = m.get("action_transition_matrix") or {}
    if transitions:
        all_tiers = sorted({t for t in transitions} | {t for row in transitions.values() for t in row})
        if all_tiers:
            lines.append("")
            lines.append("## Action Transitions (prior_tier → current_tier)")
            lines.append("")
            lines.append("| prior \\ current | " + " | ".join(all_tiers) + " |")
            lines.append("|" + "---|" * (len(all_tiers) + 1))
            for ft in all_tiers:
                row_counts = transitions.get(ft, {})
                row_cells = [str(row_counts.get(tt, 0)) for tt in all_tiers]
                lines.append(f"| {ft} | " + " | ".join(row_cells) + " |")

    # Feature coverage drops
    drops = m.get("coverage_drops_top5") or []
    if drops:
        lines.append("")
        lines.append("## Top Feature Coverage Drops (prior → current)")
        lines.append("")
        lines.append("| feature | prior % | current % | delta pp |")
        lines.append("|---|---|---|---|")
        for d in drops:
            lines.append(f"| {d['feature']} | {d['prior_pct']} | {d['cur_pct']} | {d['delta_pp']} |")

    # Near-miss fragility
    nm = m.get("near_miss") or {}
    if nm:
        lines.append("")
        lines.append("## Near-Miss Threshold Fragility")
        lines.append("")
        for score_col, stats in nm.items():
            for K_label, ks in stats.items():
                eps_parts = ", ".join(f"{k}={v}" for k, v in (ks.get("eps_counts") or {}).items())
                lines.append(
                    f"- {score_col} @ {K_label}: cutoff={ks.get('cutoff_score')}, "
                    f"within-eps counts: {eps_parts}, share_topk_25bps={ks.get('share_topk_within_25bps_pct')}%"
                )

    if report.get("warn_reasons"):
        lines.append("")
        lines.append("## Warnings")
        for r in report["warn_reasons"]:
            lines.append(f"- {r}")

    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def check_drift_monitoring(
    staging_date_dir: Path,
    snapshot_dir: Path,
    as_of_date: str,
    thresholds: DriftThresholds,
) -> GateResult:
    """Compare current snapshot vs most recent prior.

    WARN thresholds flag operational concerns. FAIL thresholds block
    snapshot promotion when extreme drift is detected (likely a data
    feed failure or pipeline bug).

    Writes drift_report.json + drift_report.md as sidecar artifacts.
    """
    prior = _find_prior_snapshot(snapshot_dir, as_of_date)
    if prior is None:
        return GateResult(
            name="drift_monitoring",
            status="PASS",
            detail="No prior snapshot; drift check skipped",
        )

    current_csv = staging_date_dir / "rankings.csv"
    prior_csv = prior / "rankings.csv"

    if not current_csv.exists():
        return GateResult(
            name="drift_monitoring",
            status="PASS",
            detail="No rankings.csv in current snapshot; drift check skipped",
        )

    metrics = _compute_drift_metrics(current_csv, prior_csv)

    # Evaluate FAIL thresholds first — extreme drift blocks promotion
    fail_reasons: List[str] = []
    if metrics["top20_overlap_pct"] < thresholds.fail_top20_overlap_pct:
        fail_reasons.append(
            f"FAIL top20_overlap_pct: {metrics['top20_overlap_pct']:.1f}% < {thresholds.fail_top20_overlap_pct}%"
        )
    if metrics["top60_overlap_pct"] < thresholds.fail_top60_overlap_pct:
        fail_reasons.append(
            f"FAIL top60_overlap_pct: {metrics['top60_overlap_pct']:.1f}% < {thresholds.fail_top60_overlap_pct}%"
        )
    if metrics["rank_spearman_rho"] is not None and metrics["rank_spearman_rho"] < thresholds.fail_rank_spearman_rho:
        fail_reasons.append(
            f"FAIL rank_spearman_rho: {metrics['rank_spearman_rho']:.4f} < {thresholds.fail_rank_spearman_rho}"
        )
    if metrics["mean_abs_rank_delta_top60"] > thresholds.fail_mean_abs_rank_delta_top60:
        fail_reasons.append(
            f"FAIL mean_abs_rank_delta_top60: {metrics['mean_abs_rank_delta_top60']:.1f} > {thresholds.fail_mean_abs_rank_delta_top60}"
        )
    if metrics["tier_migration_count"] > thresholds.fail_tier_migration_count:
        fail_reasons.append(
            f"FAIL tier_migration_count: {metrics['tier_migration_count']} > {thresholds.fail_tier_migration_count}"
        )
    if metrics["eligibility_change_count"] > thresholds.fail_eligibility_change_count:
        fail_reasons.append(
            f"FAIL eligibility_change_count: {metrics['eligibility_change_count']} > {thresholds.fail_eligibility_change_count}"
        )

    # Evaluate WARN thresholds
    warn_reasons: List[str] = []
    if metrics["top20_overlap_pct"] < thresholds.warn_top20_overlap_pct:
        warn_reasons.append(
            f"warn_top20_overlap_pct: {metrics['top20_overlap_pct']:.1f}% < {thresholds.warn_top20_overlap_pct}%"
        )
    if metrics["top60_overlap_pct"] < thresholds.warn_top60_overlap_pct:
        warn_reasons.append(
            f"warn_top60_overlap_pct: {metrics['top60_overlap_pct']:.1f}% < {thresholds.warn_top60_overlap_pct}%"
        )
    if metrics["rank_spearman_rho"] is not None and metrics["rank_spearman_rho"] < thresholds.warn_rank_spearman_rho:
        warn_reasons.append(
            f"warn_rank_spearman_rho: {metrics['rank_spearman_rho']:.4f} < {thresholds.warn_rank_spearman_rho}"
        )
    if metrics["mean_abs_rank_delta_top60"] > thresholds.warn_mean_abs_rank_delta_top60:
        warn_reasons.append(
            f"warn_mean_abs_rank_delta_top60: {metrics['mean_abs_rank_delta_top60']:.1f} > {thresholds.warn_mean_abs_rank_delta_top60}"
        )
    if metrics["tier_migration_count"] > thresholds.warn_tier_migration_count:
        warn_reasons.append(
            f"warn_tier_migration_count: {metrics['tier_migration_count']} > {thresholds.warn_tier_migration_count}"
        )
    if metrics["eligibility_change_count"] > thresholds.warn_eligibility_change_count:
        warn_reasons.append(
            f"warn_eligibility_change_count: {metrics['eligibility_change_count']} > {thresholds.warn_eligibility_change_count}"
        )

    # Stability plumbing diagnostics — WARN-only (FAIL disabled via -1 sentinel)
    if metrics.get("action_change_pct", 0.0) > thresholds.warn_action_change_pct:
        warn_reasons.append(
            f"warn_action_change_pct: {metrics['action_change_pct']:.1f}% > {thresholds.warn_action_change_pct}%"
        )
    sel_score_stats = (metrics.get("score_delta_stats") or {}).get("selector_score") or {}
    sel_mean_delta = sel_score_stats.get("mean_abs_delta")
    if sel_mean_delta is not None and sel_mean_delta > thresholds.warn_mean_abs_selector_score_delta:
        warn_reasons.append(
            f"warn_mean_abs_selector_score_delta: {sel_mean_delta:.4f} > {thresholds.warn_mean_abs_selector_score_delta}"
        )
    if metrics.get("max_coverage_drop_pp", 0.0) > thresholds.warn_feature_coverage_drop_pct:
        feat = metrics.get("max_coverage_drop_feature", "?")
        warn_reasons.append(
            f"warn_feature_coverage_drop_pct: {feat} dropped {metrics['max_coverage_drop_pp']:.1f}pp > {thresholds.warn_feature_coverage_drop_pct}pp"
        )
    if metrics.get("near_miss_share_pct", 0.0) > thresholds.warn_near_miss_share_pct:
        warn_reasons.append(
            f"warn_near_miss_share_pct: {metrics['near_miss_share_pct']:.1f}% of top-30 within 25bps of K=30 cutoff > {thresholds.warn_near_miss_share_pct}%"
        )

    if fail_reasons:
        status = "FAIL"
    elif warn_reasons:
        status = "WARN"
    else:
        status = "PASS"

    # Build and write drift report JSON
    report = {
        "version": "1.2.0",
        "thresholds_id": thresholds.thresholds_id,
        "thresholds": asdict(thresholds),
        "current_date": as_of_date,
        "prior_date": prior.name,
        "metrics": metrics,
        "fail_reasons": fail_reasons,
        "warn_reasons": warn_reasons,
        "status": status,
    }

    report_json_path = staging_date_dir / "drift_report.json"
    with open(report_json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    report_md_path = staging_date_dir / "drift_report.md"
    _write_drift_report_md(report, report_md_path)

    detail = f"vs {prior.name}: "
    if fail_reasons:
        detail += "BLOCKED — " + "; ".join(fail_reasons)
    elif warn_reasons:
        detail += "; ".join(warn_reasons)
    else:
        detail += (
            f"top20={metrics['top20_overlap_pct']:.0f}%, "
            f"top60={metrics['top60_overlap_pct']:.0f}%, "
            f"rho={metrics['rank_spearman_rho']}"
        )

    return GateResult(
        name="drift_monitoring",
        status=status,
        detail=detail,
        value=metrics,
        threshold=asdict(thresholds),
    )


# ---------------------------------------------------------------------------
# Ruleset health gate (WARN-only)
# ---------------------------------------------------------------------------


def check_ruleset_health(
    staging_date_dir: Path,
    receipts_dir: Optional[Path] = None,
    history_path: Optional[Path] = None,
    active_ruleset_id: Optional[str] = None,
    manifest_path: Optional[Path] = None,
) -> GateResult:
    """Post-promotion health check. WARN-only gate (never FAIL).

    Compares today's drift metrics against the active ruleset's promotion baseline.
    """
    from tools.ruleset_health_monitor import run_health_check

    if receipts_dir is None:
        receipts_dir = REPO_ROOT / "artifacts" / "promotions"
    if history_path is None:
        history_path = REPO_ROOT / "artifacts" / "ruleset_health_history.jsonl"
    if manifest_path is None:
        manifest_path = REPO_ROOT / "production_data" / "decision_rulesets" / "manifest.json"

    drift_report_path = staging_date_dir / "drift_report.json"

    result = run_health_check(
        drift_report_path=drift_report_path,
        receipts_dir=receipts_dir,
        history_path=history_path,
        output_dir=staging_date_dir,
        active_ruleset_id=active_ruleset_id,
        manifest_path=manifest_path,
    )

    status = result.get("status", "OK")
    # Map OK → PASS for gate consistency
    gate_status = "PASS" if status == "OK" else "WARN"
    detail = result.get("detail", "")
    if result.get("recommend_rollback"):
        detail = f"ROLLBACK RECOMMENDED: {detail}"

    return GateResult(
        name="ruleset_health",
        status=gate_status,
        detail=detail,
        value=result,
    )


# ---------------------------------------------------------------------------
# Canary regression gate (BLOCK->FAIL, WARN->WARN, INFO->PASS)
# ---------------------------------------------------------------------------


def check_canary_regression(
    staging_date_dir: Path,
    *,
    policy_path: Optional[Path] = None,
    thresholds_path: Optional[Path] = None,
    history_path: Optional[Path] = None,
    ruleset_path: Optional[Path] = None,
) -> GateResult:
    """Canary regression gate. Maps BLOCK->FAIL, WARN->WARN, INFO->PASS.

    Wraps entire canary run in try/except — degrades to WARN on crash
    (matches other advisory gates).
    """
    from scripts.replay_diff import DiffThresholds
    from scripts.run_canary_dates import DEFAULT_RULESET as CANARY_DEFAULT_RULESET
    from scripts.run_canary_dates import DEFAULT_THRESHOLDS as CANARY_DEFAULT_THRESHOLDS
    from scripts.run_canary_dates import CanaryOutcome, CanaryPolicy, run_canary_classified

    if policy_path is None:
        policy_path = REPO_ROOT / "production_data" / "canary_policy.json"
    if thresholds_path is None:
        thresholds_path = CANARY_DEFAULT_THRESHOLDS
    if history_path is None:
        history_path = REPO_ROOT / "artifacts" / "canary_regression_history.jsonl"
    if ruleset_path is None:
        ruleset_path = CANARY_DEFAULT_RULESET

    try:
        policy = CanaryPolicy.from_json(policy_path) if policy_path.exists() else CanaryPolicy.default()
        thresholds = DiffThresholds.from_json(str(thresholds_path)) if thresholds_path.exists() else DiffThresholds()
        verdict = run_canary_classified(
            thresholds,
            policy,
            ruleset_path,
            history_path,
        )
    except Exception as exc:
        return GateResult(
            name="canary_regression",
            status="WARN",
            detail=f"canary failed to run: {exc}",
        )

    outcome_to_status = {
        CanaryOutcome.BLOCK: "FAIL",
        CanaryOutcome.WARN: "WARN",
        CanaryOutcome.INFO: "PASS",
    }

    status = outcome_to_status[verdict.overall_outcome]
    detail_parts = []
    for d in sorted(verdict.per_date):
        detail_parts.append(f"{d}={verdict.per_date[d].outcome.value}")
    detail = f"overall={verdict.overall_outcome.value} ({', '.join(detail_parts)})"

    return GateResult(
        name="canary_regression",
        status=status,
        detail=detail,
    )


# ---------------------------------------------------------------------------
# CTGov PIT dates gate (WARN-only)
# ---------------------------------------------------------------------------


def check_ctgov_pit_dates(
    ctgov_cache_dir: Path,
    as_of_date: str,
    *,
    warn_first_posted_min: float = 0.80,
    warn_last_update_min: float = 0.80,
) -> GateResult:
    """Validate PIT-critical date field coverage in trial records cache.

    WARN-only gate — never returns FAIL. Checks first_posted and
    last_update_posted coverage; results_first_posted is informational only.
    """
    # Find cache file — exact match or glob fallback
    exact = ctgov_cache_dir / f"trial_records_{as_of_date}.json"
    cache_file = None
    if exact.exists():
        cache_file = exact
    else:
        # Fallback: find latest file <= as_of_date
        candidates = sorted(ctgov_cache_dir.glob("trial_records_*.json"))
        for c in reversed(candidates):
            stem_date = c.stem.replace("trial_records_", "")
            if stem_date <= as_of_date:
                cache_file = c
                break

    if cache_file is None:
        return GateResult(
            name="ctgov_pit_dates",
            status="WARN",
            detail=f"No trial_records cache found in {ctgov_cache_dir}",
        )

    try:
        with open(cache_file) as f:
            records = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        return GateResult(
            name="ctgov_pit_dates",
            status="WARN",
            detail=f"Cannot read {cache_file.name}: {e}",
        )

    if not isinstance(records, list) or len(records) == 0:
        return GateResult(
            name="ctgov_pit_dates",
            status="WARN",
            detail=f"Empty or invalid trial records in {cache_file.name}",
        )

    total = len(records)
    n_first_posted = sum(1 for r in records if (r.get("first_posted") or "").strip())
    n_last_update = sum(1 for r in records if (r.get("last_update_posted") or "").strip())
    n_results_posted = sum(1 for r in records if (r.get("results_first_posted") or "").strip())

    cov_first = n_first_posted / total
    cov_last = n_last_update / total
    cov_results = n_results_posted / total

    coverage = {
        "first_posted": round(cov_first, 4),
        "last_update_posted": round(cov_last, 4),
        "results_first_posted": round(cov_results, 4),
        "total_records": total,
        "cache_file": cache_file.name,
    }
    thresholds = {
        "warn_first_posted_min": warn_first_posted_min,
        "warn_last_update_min": warn_last_update_min,
    }

    warn_reasons: List[str] = []
    if cov_first < warn_first_posted_min:
        warn_reasons.append(f"first_posted coverage {cov_first:.1%} < {warn_first_posted_min:.0%}")
    if cov_last < warn_last_update_min:
        warn_reasons.append(f"last_update_posted coverage {cov_last:.1%} < {warn_last_update_min:.0%}")

    status = "WARN" if warn_reasons else "PASS"
    detail_parts = [
        f"first_posted={cov_first:.1%}",
        f"last_update={cov_last:.1%}",
        f"results={cov_results:.1%}",
        f"(n={total}, {cache_file.name})",
    ]
    if warn_reasons:
        detail_parts.append("; ".join(warn_reasons))

    return GateResult(
        name="ctgov_pit_dates",
        status=status,
        detail=", ".join(detail_parts),
        value=coverage,
        threshold=thresholds,
    )


# ---------------------------------------------------------------------------
# SEC 13F cache gate (WARN-only)
# ---------------------------------------------------------------------------


def check_sec_13f_cache(
    as_of_date: str,
    *,
    cache_base_dir: Optional[Path] = None,
    warn_coverage_pct: float = 80.0,
) -> GateResult:
    """Validate 13F PIT cache coverage + schema. WARN-only — never FAIL."""
    from tools.warm_13f_cache import validate_sec_13f_index_schema

    base = cache_base_dir or (REPO_ROOT / "data" / "caches" / "sec_13f" / "PIT")
    index_path = base / as_of_date / "index.json"

    if not index_path.exists():
        return GateResult(
            name="sec_13f_cache",
            status="WARN",
            detail=f"No 13F cache index at {index_path}",
            threshold={"warn_coverage_pct": warn_coverage_pct},
        )

    try:
        with open(index_path) as f:
            index = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        return GateResult(
            name="sec_13f_cache",
            status="WARN",
            detail=f"Cannot read 13F cache index: {e}",
            threshold={"warn_coverage_pct": warn_coverage_pct},
        )

    # Schema validation
    ok, schema_detail = validate_sec_13f_index_schema(
        index,
        expected_as_of_date=as_of_date,
    )
    if not ok:
        return GateResult(
            name="sec_13f_cache",
            status="WARN",
            detail=f"schema invalid: {schema_detail}",
            threshold={"warn_coverage_pct": warn_coverage_pct},
        )

    coverage = index["coverage_pct"]
    managers_ok = index["managers_with_filing"]
    total = index["total_managers"]

    detail_parts = [
        f"coverage={coverage:.1f}%",
        f"({managers_ok}/{total} managers)",
    ]

    value = {"coverage_pct": coverage, "managers_ok": managers_ok, "total": total}
    threshold = {"warn_coverage_pct": warn_coverage_pct}

    if coverage < warn_coverage_pct:
        detail_parts.append(f"below {warn_coverage_pct:.0f}% threshold")
        return GateResult(
            name="sec_13f_cache",
            status="WARN",
            detail=", ".join(detail_parts),
            value=value,
            threshold=threshold,
        )

    return GateResult(
        name="sec_13f_cache",
        status="PASS",
        detail=", ".join(detail_parts),
        value=value,
        threshold=threshold,
    )


# ---------------------------------------------------------------------------
# Institutional summary gate (WARN-only, post-screen)
# ---------------------------------------------------------------------------


def check_institutional_summary(
    snapshot_date_dir: Path,
    *,
    warn_coverage_pct: float = 50.0,
) -> GateResult:
    """Validate institutional_summary.json sidecar. WARN-only — never FAIL."""
    from institutional_summary import validate_institutional_summary_schema_v1

    sidecar_path = snapshot_date_dir / "institutional_summary.json"

    if not sidecar_path.exists():
        return GateResult(
            name="institutional_summary",
            status="WARN",
            detail=f"No institutional_summary.json at {sidecar_path}",
            threshold={"warn_coverage_pct": warn_coverage_pct},
        )

    try:
        with open(sidecar_path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        return GateResult(
            name="institutional_summary",
            status="WARN",
            detail=f"Cannot read institutional_summary.json: {e}",
            threshold={"warn_coverage_pct": warn_coverage_pct},
        )

    # Run pure schema validator
    ok, schema_detail = validate_institutional_summary_schema_v1(data)
    if not ok:
        return GateResult(
            name="institutional_summary",
            status="WARN",
            detail=f"schema invalid: {schema_detail}",
            threshold={"warn_coverage_pct": warn_coverage_pct},
        )

    coverage = data.get("signal_coverage_pct", 0.0)
    tickers_with = data.get("tickers_with_signal", 0)
    tickers_total = data.get("tickers_in_universe", 0)
    managers_used = data.get("elite_managers_with_filing", 0)
    managers_total = data.get("elite_managers_total", 0)

    detail_parts = [
        f"coverage={coverage:.1f}%",
        f"({tickers_with}/{tickers_total} tickers)",
        f"({managers_used}/{managers_total} managers)",
    ]

    value = {
        "coverage_pct": coverage,
        "tickers_with_signal": tickers_with,
        "tickers_in_universe": tickers_total,
        "managers_used": managers_used,
        "expected_managers": managers_total,
    }
    threshold = {"warn_coverage_pct": warn_coverage_pct}

    if coverage < warn_coverage_pct:
        detail_parts.append(f"below {warn_coverage_pct:.0f}% threshold")
        return GateResult(
            name="institutional_summary",
            status="WARN",
            detail=", ".join(detail_parts),
            value=value,
            threshold=threshold,
        )

    return GateResult(
        name="institutional_summary",
        status="PASS",
        detail=", ".join(detail_parts),
        value=value,
        threshold=threshold,
    )


# ---------------------------------------------------------------------------
# Institutional delta gate (WARN-only, post-screen)
# ---------------------------------------------------------------------------


def check_institutional_delta(
    snapshot_date_dir: Path,
    snapshot_dir: Path,
    current_date: str,
) -> GateResult:
    """Validate institutional_summary_delta.json sidecar. WARN-only — never FAIL.

    - Delta file exists + valid → PASS
    - Delta file exists + invalid → WARN
    - No delta + no prior with elite_holder_shares → PASS (cold-start)
    - No delta + prior with elite_holder_shares exists → WARN
    """
    delta_path = snapshot_date_dir / "institutional_summary_delta.json"
    if delta_path.exists():
        # Validate the delta sidecar
        try:
            with open(delta_path, encoding="utf-8") as f:
                delta_data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            return GateResult(
                name="institutional_delta",
                status="WARN",
                detail=f"Cannot read institutional_summary_delta.json: {e}",
            )

        from institutional_summary import validate_institutional_summary_delta_schema_v1

        ok, detail = validate_institutional_summary_delta_schema_v1(delta_data)
        if not ok:
            return GateResult(
                name="institutional_delta",
                status="WARN",
                detail=f"schema invalid: {detail}",
            )

        return GateResult(
            name="institutional_delta",
            status="PASS",
            detail=(
                "institutional_summary_delta.json valid: "
                f"{delta_data.get('as_of_date', '?')} vs {delta_data.get('prior_date', '?')}, "
                f"{delta_data.get('tickers_common', 0)} common tickers"
            ),
        )

    # No delta — check if a prior with elite_holder_shares exists
    try:
        from institutional_summary import _find_prior_institutional_summary

        prior = _find_prior_institutional_summary(snapshot_dir, current_date)
    except Exception:
        prior = None

    if prior is not None:
        return GateResult(
            name="institutional_delta",
            status="WARN",
            detail="Delta expected (prior with elite_holder_shares found) but not written",
        )

    return GateResult(
        name="institutional_delta",
        status="PASS",
        detail="No prior institutional summary with elite_holder_shares — cold-start OK",
    )


def check_pnl_attribution(
    snapshot_date_dir: Path,
    snapshot_dir: Path,
    current_date: str,
    price_csv: Path,
    min_coverage_pct: float = 80.0,
    cost_bps: float = 30.0,
) -> GateResult:
    """Generate and validate PnL attribution sidecar. WARN-only — never FAIL.

    Runs pnl_attribution between the prior snapshot and the current one,
    writes pnl_attribution.json + .md into the staging dir, then validates.
    """
    try:
        from scripts.pnl_attribution import (
            check_pnl_attribution_file,
            compute_attribution,
            find_prior_date,
            write_attribution_json,
            write_attribution_md,
        )
    except ImportError as e:
        return GateResult(
            name="pnl_attribution",
            status="WARN",
            detail=f"Cannot import pnl_attribution module: {e}",
        )

    # Find prior snapshot
    prior_date = find_prior_date(snapshot_dir, current_date)
    if prior_date is None:
        return GateResult(
            name="pnl_attribution",
            status="PASS",
            detail="No prior snapshot — cold-start OK",
        )

    prior_dir = snapshot_dir / prior_date
    if not prior_dir.exists() or not (prior_dir / "rankings.csv").exists():
        return GateResult(
            name="pnl_attribution",
            status="PASS",
            detail=f"Prior snapshot {prior_date} not usable — cold-start OK",
        )

    try:
        result = compute_attribution(
            d0_dir=prior_dir,
            d1_dir=snapshot_date_dir,
            price_csv=price_csv,
            cost_bps=cost_bps,
        )
        write_attribution_json(result, snapshot_date_dir / "pnl_attribution.json")
        write_attribution_md(result, snapshot_date_dir / "pnl_attribution.md")
    except Exception as e:
        return GateResult(
            name="pnl_attribution",
            status="WARN",
            detail=f"PnL attribution failed: {e}",
        )

    # Validate the written file
    status, detail, value, threshold = check_pnl_attribution_file(
        snapshot_date_dir,
        min_coverage_pct=min_coverage_pct,
    )
    return GateResult(
        name="pnl_attribution",
        status=status,
        detail=detail,
        value=value,
        threshold=threshold,
    )


def check_price_pit_cache(
    cache_dir: Path,
    as_of_date: str,
) -> GateResult:
    """Validate PIT price cache index for today's snapshot. WARN-only — never FAIL."""
    from tools.warm_price_cache import validate_price_pit_index

    index_path = cache_dir / "index.json"
    if not index_path.exists():
        return GateResult(
            name="price_pit_cache",
            status="WARN",
            detail=f"No PIT price cache at {cache_dir}",
        )

    try:
        with open(index_path, encoding="utf-8") as f:
            index = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        return GateResult(
            name="price_pit_cache",
            status="WARN",
            detail=f"Cannot read PIT price cache index: {e}",
        )

    ok, schema_detail = validate_price_pit_index(index, expected_as_of_date=as_of_date)
    if not ok:
        return GateResult(
            name="price_pit_cache",
            status="WARN",
            detail=f"schema invalid: {schema_detail}",
        )

    coverage = index.get("coverage_pct", 0)
    ticker_count = index.get("ticker_count", 0)
    n_missing = len(index.get("tickers_missing_anchor", []))
    anchor_date = index.get("anchor_date", "?")

    return GateResult(
        name="price_pit_cache",
        status="PASS",
        detail=(
            f"coverage={coverage:.1f}%, " f"{ticker_count - n_missing}/{ticker_count} tickers, " f"anchor={anchor_date}"
        ),
        value={
            "coverage_pct": coverage,
            "ticker_count": ticker_count,
            "missing": n_missing,
            "anchor_date": anchor_date,
        },
    )


def check_forward_eval(
    snapshot_dir: Path,
    price_cache_base: Path,
    current_date: str,
    config: GateConfig,
) -> GateResult:
    """Forward-return rolling IC gate. WARN-only — never FAIL."""
    try:
        from tools.forward_eval_gate import evaluate_rolling_ic
    except ImportError as e:
        return GateResult(
            name="forward_eval",
            status="WARN",
            detail=f"Cannot import forward_eval_gate: {e}",
        )

    status, detail, value, threshold = evaluate_rolling_ic(
        snapshot_dir=snapshot_dir,
        price_cache_base=price_cache_base,
        current_date=current_date,
        horizon=config.forward_eval_horizon,
        lookback_n=config.forward_eval_lookback_n,
        ic_warn_floor=config.forward_eval_ic_warn_floor,
    )

    return GateResult(
        name="forward_eval",
        status=status,
        detail=detail,
        value=value,
        threshold=threshold,
    )


def check_pit_bundle_health(
    as_of_date: str,
    bundle_root: Optional[Path] = None,
    ctgov_cache_dir: Optional[Path] = None,
    cache_13f_root: Optional[Path] = None,
) -> GateResult:
    """Check that a PIT bundle is buildable from available caches. WARN-only — never FAIL.

    Checks existence of required cache inputs (CTgov cache, 13F cache).
    Does NOT actually build the bundle — just validates prerequisites.
    """
    issues: List[str] = []

    # Check CTgov cache
    _ctgov_dir = ctgov_cache_dir or (REPO_ROOT / "cache" / "ctgov")
    ctgov_path = _ctgov_dir / f"trial_records_{as_of_date}.json"
    if not ctgov_path.exists():
        # Check for any prior date file
        candidates = sorted(_ctgov_dir.glob("trial_records_*.json"))
        if candidates:
            issues.append(f"CTgov cache missing for {as_of_date} (latest: {candidates[-1].stem})")
        else:
            issues.append(f"No CTgov cache files in {_ctgov_dir}")

    # Check 13F cache
    _13f_root = cache_13f_root or (REPO_ROOT / "data" / "caches" / "sec_13f" / "PIT")
    index_path = _13f_root / as_of_date / "index.json"
    if not index_path.exists():
        issues.append(f"13F PIT cache missing for {as_of_date}")

    # Check trial_records.json (needed by clinical builder)
    trial_path = REPO_ROOT / "production_data" / "trial_records.json"
    if not trial_path.exists():
        issues.append("production_data/trial_records.json not found")

    if issues:
        return GateResult(
            name="pit_bundle_health",
            status="WARN",
            detail="; ".join(issues),
        )

    return GateResult(
        name="pit_bundle_health",
        status="PASS",
        detail=f"PIT bundle prerequisites available for {as_of_date}",
    )


# ---------------------------------------------------------------------------
# Decision engine schema gate (WARN-only)
# ---------------------------------------------------------------------------


def check_decision_engine_schema(
    snapshot_date_dir: Path,
) -> GateResult:
    """Validate rankings.csv has all DE columns with valid values. WARN-only.

    Checks:
    - All DECISION_COLUMNS + ACTIONABLE_COLUMNS present in headers
    - tier_dev ∈ {A, B, C, D, ""}
    - tier_any ∈ {A, B, C, D, ""}
    - eligible ∈ {"0", "1"}
    - actionable_rank sequential 1..N for eligible rows, empty for ineligible
    """
    rankings_path = snapshot_date_dir / "rankings.csv"
    if not rankings_path.exists():
        return GateResult(
            name="decision_engine_schema",
            status="WARN",
            detail="rankings.csv not found",
        )

    # Lazy import to avoid heavy module load at init
    from decision_engine import ACTIONABLE_COLUMNS, DECISION_COLUMNS

    expected_cols = set(DECISION_COLUMNS) | set(ACTIONABLE_COLUMNS)

    warn_reasons: List[str] = []
    rows_read = 0
    # Collect eligible ranks for set-based contiguity check after the loop.
    # Rankings.csv is written in report order (weight desc, tier, rank) not
    # rank order, so sequential expected_rank tracking produces false WARNs.
    eligible_ranks: List[int] = []
    rank_errors: List[str] = []

    with open(rankings_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = set(reader.fieldnames or [])

        missing_cols = expected_cols - headers
        if missing_cols:
            warn_reasons.append(f"missing columns: {sorted(missing_cols)}")

        valid_tiers = {"A", "B", "C", "D", ""}
        valid_eligible = {"0", "1"}

        for row in reader:
            rows_read += 1
            ticker = row.get("ticker", "?")

            tier_dev = (row.get("tier_dev") or "").strip()
            if tier_dev not in valid_tiers:
                warn_reasons.append(f"{ticker}: tier_dev='{tier_dev}' invalid")

            tier_any = (row.get("tier_any") or "").strip()
            if tier_any not in valid_tiers:
                warn_reasons.append(f"{ticker}: tier_any='{tier_any}' invalid")

            eligible = (row.get("eligible") or "").strip()
            if eligible not in valid_eligible:
                warn_reasons.append(f"{ticker}: eligible='{eligible}' invalid")

            act_rank = (row.get("actionable_rank") or "").strip()
            if eligible == "1":
                if act_rank:
                    try:
                        rank_int = int(act_rank)
                        eligible_ranks.append(rank_int)
                    except ValueError:
                        rank_errors.append(f"{ticker}: actionable_rank='{act_rank}' not integer")
            else:
                if act_rank and act_rank.lower() not in ("", "nan"):
                    warn_reasons.append(f"{ticker}: ineligible but actionable_rank='{act_rank}'")

            # Cap non-rank warnings to avoid flood
            if len(warn_reasons) >= 20:
                warn_reasons.append("... (truncated)")
                break

    # Set-based rank contiguity check: eligible ranks must be exactly {1..N}
    warn_reasons.extend(rank_errors)
    if eligible_ranks:
        n = len(eligible_ranks)
        expected_set = set(range(1, n + 1))
        actual_set = set(eligible_ranks)
        missing_ranks = sorted(expected_set - actual_set)
        duplicate_ranks = sorted(r for r in actual_set if eligible_ranks.count(r) > 1)
        extra_ranks = sorted(actual_set - expected_set)
        if missing_ranks:
            warn_reasons.append(f"actionable_rank gaps: missing {missing_ranks}")
        if duplicate_ranks:
            warn_reasons.append(f"actionable_rank duplicates: {duplicate_ranks}")
        if extra_ranks:
            warn_reasons.append(f"actionable_rank out-of-range: {extra_ranks}")

    if not warn_reasons:
        return GateResult(
            name="decision_engine_schema",
            status="PASS",
            detail=f"{rows_read} rows, all DE columns present and valid",
        )

    return GateResult(
        name="decision_engine_schema",
        status="WARN",
        detail=f"{len(warn_reasons)} issues: {'; '.join(warn_reasons[:5])}",
        value=len(warn_reasons),
    )


# ---------------------------------------------------------------------------
# Sort contribution sanity gate (WARN/FAIL)
# ---------------------------------------------------------------------------


# Column names expected in rankings.csv for sort contribution diagnostics.
# Derive from engine's canonical list to prevent drift when new contributions are added.
def _get_sort_contrib_columns() -> tuple:
    from decision_engine import SORT_CONTRIB_KEYS

    return ("de_sort_total_adj",) + tuple(f"de_sort_contrib_{k}" for k in SORT_CONTRIB_KEYS)


_SORT_CONTRIB_COLUMNS = _get_sort_contrib_columns()


def check_sort_contrib_sanity(
    snapshot_date_dir: Path,
    config: GateConfig,
) -> GateResult:
    """Validate sort contribution columns in rankings.csv. WARN or FAIL.

    Checks on eligible rows:
    (i)   Column presence — WARN if any expected column missing entirely.
    (ii)  Missingness — WARN if blank/NaN fraction exceeds threshold.
    (iii) Sum identity — WARN if sum(contribs) != total_adj beyond tolerance.
    (iv)  Hard sanity — FAIL if any value is absurdly large or non-finite.
    """
    rankings_path = snapshot_date_dir / "rankings.csv"
    if not rankings_path.exists():
        return GateResult(
            name="sort_contrib_sanity",
            status="WARN",
            detail="rankings.csv not found",
        )

    with open(rankings_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = set(reader.fieldnames or [])

        # (i) Column presence
        missing_cols = [c for c in _SORT_CONTRIB_COLUMNS if c not in headers]
        if missing_cols:
            return GateResult(
                name="sort_contrib_sanity",
                status="WARN",
                detail=f"columns missing; likely older snapshot: {missing_cols}",
            )

        eligible_n = 0
        missing_count = 0
        sum_mismatch_count = 0
        hard_fail_count = 0
        hard_fail_tickers: List[str] = []
        sum_mismatch_tickers: List[str] = []

        contrib_cols = [c for c in _SORT_CONTRIB_COLUMNS if c != "de_sort_total_adj"]

        for row in reader:
            if (row.get("eligible") or "").strip() != "1":
                continue
            eligible_n += 1
            ticker = row.get("ticker", "?")

            # Parse values
            total_str = (row.get("de_sort_total_adj") or "").strip()
            contrib_strs = {c: (row.get(c) or "").strip() for c in contrib_cols}

            # (ii) Missingness: blank or NaN
            row_missing = False
            if not total_str or total_str.lower() == "nan":
                row_missing = True
            for v in contrib_strs.values():
                if not v or v.lower() == "nan":
                    row_missing = True
                    break
            if row_missing:
                missing_count += 1
                continue  # skip sum/sanity checks for incomplete rows

            # Parse floats
            try:
                total_val = float(total_str)
            except ValueError:
                hard_fail_count += 1
                if len(hard_fail_tickers) < 5:
                    hard_fail_tickers.append(ticker)
                continue

            contrib_vals = {}
            parse_ok = True
            for c, v in contrib_strs.items():
                try:
                    contrib_vals[c] = float(v)
                except ValueError:
                    hard_fail_count += 1
                    if len(hard_fail_tickers) < 5:
                        hard_fail_tickers.append(ticker)
                    parse_ok = False
                    break
            if not parse_ok:
                continue

            # (iv) Hard sanity: non-finite or absurdly large
            all_vals = [total_val] + list(contrib_vals.values())
            for av in all_vals:
                if not math.isfinite(av) or abs(av) > config.sort_contrib_hard_abs_max:
                    hard_fail_count += 1
                    if len(hard_fail_tickers) < 5:
                        hard_fail_tickers.append(ticker)
                    break
            else:
                # (iii) Sum identity: sum(contribs) == total_adj
                contrib_sum = sum(contrib_vals.values())
                if abs(contrib_sum - total_val) > config.sort_contrib_sum_tolerance:
                    sum_mismatch_count += 1
                    if len(sum_mismatch_tickers) < 5:
                        sum_mismatch_tickers.append(ticker)

    if eligible_n == 0:
        return GateResult(
            name="sort_contrib_sanity",
            status="WARN",
            detail="no eligible rows found",
        )

    value_dict = {
        "eligible_n": eligible_n,
        "missing_frac": round(missing_count / eligible_n, 4) if eligible_n else 0,
        "sum_mismatch_frac": round(sum_mismatch_count / eligible_n, 4) if eligible_n else 0,
        "hard_fail_count": hard_fail_count,
    }
    threshold_dict = {
        "missing_max_frac": config.sort_contrib_missing_max_frac,
        "sum_tolerance": config.sort_contrib_sum_tolerance,
        "hard_abs_max": config.sort_contrib_hard_abs_max,
    }

    # (iv) Hard sanity failures → FAIL
    if hard_fail_count > 0:
        return GateResult(
            name="sort_contrib_sanity",
            status="FAIL",
            detail=(
                f"{hard_fail_count} row(s) with non-finite or absurd values "
                f"(>{config.sort_contrib_hard_abs_max}): {hard_fail_tickers}"
            ),
            value=value_dict,
            threshold=threshold_dict,
        )

    # Accumulate WARN reasons
    warn_reasons: List[str] = []

    # (ii) Missingness
    missing_frac = missing_count / eligible_n
    if missing_frac > config.sort_contrib_missing_max_frac:
        warn_reasons.append(
            f"missing contribs: {missing_count}/{eligible_n} "
            f"({missing_frac:.1%} > {config.sort_contrib_missing_max_frac:.0%})"
        )

    # (iii) Sum identity
    if sum_mismatch_count > 0:
        mismatch_frac = sum_mismatch_count / eligible_n
        warn_reasons.append(
            f"sum mismatch: {sum_mismatch_count}/{eligible_n} " f"({mismatch_frac:.1%}): {sum_mismatch_tickers}"
        )

    if warn_reasons:
        return GateResult(
            name="sort_contrib_sanity",
            status="WARN",
            detail="; ".join(warn_reasons),
            value=value_dict,
            threshold=threshold_dict,
        )

    return GateResult(
        name="sort_contrib_sanity",
        status="PASS",
        detail=f"{eligible_n} eligible rows, all contribs valid",
        value=value_dict,
        threshold=threshold_dict,
    )


# ---------------------------------------------------------------------------
# Portfolio weights gate (WARN-only)
# ---------------------------------------------------------------------------


def check_portfolio_weights(
    snapshot_date_dir: Path,
    tolerance: float = 1.0,
) -> GateResult:
    """Verify target_weight_pct sums to ~100% for eligible rows. WARN-only.

    Checks:
    - Sum of target_weight_pct ≈ 100% (within tolerance pp)
    - No eligible row has empty weight
    - No ineligible row has non-empty weight
    """
    rankings_path = snapshot_date_dir / "rankings.csv"
    if not rankings_path.exists():
        return GateResult(
            name="portfolio_weights",
            status="WARN",
            detail="rankings.csv not found",
        )

    warn_reasons: List[str] = []
    weight_sum = 0.0
    eligible_count = 0
    eligible_no_weight = 0
    ineligible_with_weight = 0

    with open(rankings_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            eligible = (row.get("eligible") or "").strip()
            weight_str = (row.get("target_weight_pct") or "").strip()
            ticker = row.get("ticker", "?")

            if eligible == "1":
                eligible_count += 1
                if weight_str and weight_str.lower() not in ("", "nan"):
                    try:
                        weight_sum += float(weight_str)
                    except ValueError:
                        warn_reasons.append(f"{ticker}: invalid weight '{weight_str}'")
                else:
                    eligible_no_weight += 1
            else:
                if weight_str and weight_str.lower() not in ("", "nan"):
                    try:
                        w = float(weight_str)
                        if w > 0:
                            ineligible_with_weight += 1
                    except ValueError:
                        pass

    if eligible_count == 0:
        return GateResult(
            name="portfolio_weights",
            status="WARN",
            detail="No eligible rows found",
        )

    if abs(weight_sum - 100.0) > tolerance:
        warn_reasons.append(f"weight sum={weight_sum:.2f}%, expected ~100% (tolerance={tolerance}pp)")

    if eligible_no_weight > 0:
        warn_reasons.append(f"{eligible_no_weight} eligible row(s) missing target_weight_pct")

    if ineligible_with_weight > 0:
        warn_reasons.append(f"{ineligible_with_weight} ineligible row(s) have non-zero target_weight_pct")

    if not warn_reasons:
        return GateResult(
            name="portfolio_weights",
            status="PASS",
            detail=f"weight sum={weight_sum:.2f}%, {eligible_count} eligible rows",
        )

    return GateResult(
        name="portfolio_weights",
        status="WARN",
        detail="; ".join(warn_reasons),
        value=round(weight_sum, 2),
        threshold=tolerance,
    )


# ---------------------------------------------------------------------------
# Eligibility consistency gate (WARN-only)
# ---------------------------------------------------------------------------


def check_eligibility_consistency(
    snapshot_date_dir: Path,
) -> GateResult:
    """Verify eligible=1 ↔ ineligible_reasons="" for all rows. WARN-only."""
    rankings_path = snapshot_date_dir / "rankings.csv"
    if not rankings_path.exists():
        return GateResult(
            name="eligibility_consistency",
            status="WARN",
            detail="rankings.csv not found",
        )

    warn_reasons: List[str] = []
    total = 0

    with open(rankings_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            ticker = row.get("ticker", "?")
            eligible = (row.get("eligible") or "").strip()
            reasons = (row.get("ineligible_reasons") or "").strip()
            reasons_present = reasons not in ("", "nan")

            if eligible == "1" and reasons_present:
                warn_reasons.append(f"{ticker}: eligible=1 but ineligible_reasons='{reasons}'")
            elif eligible != "1" and not reasons_present:
                warn_reasons.append(f"{ticker}: eligible={eligible} but ineligible_reasons empty")

            if len(warn_reasons) >= 20:
                warn_reasons.append("... (truncated)")
                break

    if not warn_reasons:
        return GateResult(
            name="eligibility_consistency",
            status="PASS",
            detail=f"{total} rows, all consistent",
        )

    return GateResult(
        name="eligibility_consistency",
        status="WARN",
        detail=f"{len(warn_reasons)} inconsistencies: {'; '.join(warn_reasons[:5])}",
        value=len(warn_reasons),
    )


def check_cache_health(
    snapshot_date_dir: Path,
    *,
    fail_on_bad: bool = False,
) -> GateResult:
    """Read cache_health.json sidecar and map to GateResult.

    Default: WARN-only (degraded_run=true → WARN).
    With *fail_on_bad*: overall_status=bad → FAIL.
    """
    health_path = snapshot_date_dir / "cache_health.json"
    if not health_path.exists():
        return GateResult(
            name="cache_health",
            status="PASS",
            detail="cache_health.json not found (skipped)",
        )
    try:
        with open(health_path, "r", encoding="utf-8") as f:
            health = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        return GateResult(
            name="cache_health",
            status="WARN",
            detail=f"Could not read cache_health.json: {e}",
        )

    overall = health.get("overall_status", "ok")
    sec_status = health.get("sec8k", {}).get("status", "ok")
    ctgov_status = health.get("ctgov", {}).get("status", "ok")
    sec_reason = health.get("sec8k", {}).get("reason", "")
    ctgov_reason = health.get("ctgov", {}).get("reason", "")

    parts = []
    if sec_reason:
        parts.append(sec_reason)
    if ctgov_reason:
        parts.append(ctgov_reason)
    detail = "; ".join(parts) if parts else f"sec8k={sec_status} ctgov={ctgov_status}"

    if overall == "bad" and fail_on_bad:
        return GateResult(
            name="cache_health",
            status="FAIL",
            detail=detail,
            value=overall,
        )
    if overall != "ok":
        return GateResult(
            name="cache_health",
            status="WARN",
            detail=detail,
            value=overall,
        )
    return GateResult(
        name="cache_health",
        status="PASS",
        detail=detail,
        value=overall,
    )


def _read_invariants_summary(audit_output_dir: Path) -> Optional[Dict[str, Any]]:
    """Read invariants_summary.json written by the audit tool."""
    p = audit_output_dir / "invariants_summary.json"
    if not p.exists():
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _format_audit_detail(
    base_msg: str,
    summary: Optional[Dict[str, Any]],
) -> str:
    """Enrich audit gate detail with violation breakdown from summary JSON."""
    if not summary:
        return base_msg
    parts = []
    for sev in ("critical", "warn", "info"):
        n = summary.get(sev, 0)
        if n > 0:
            parts.append(f"{n} {sev}")
    if not parts:
        return base_msg
    by_rule = summary.get("by_rule", {})
    rule_detail = ", ".join(f"{r}:{c}" for r, c in sorted(by_rule.items(), key=lambda x: -x[1]))
    return f"{base_msg} [{', '.join(parts)}] ({rule_detail})"


def check_ruleset_governance(
    ruleset_path: Optional[Path],
    manifest_path: Path,
    allow_candidate: bool = False,
) -> GateResult:
    """Validate that the resolved ruleset has status=='active' in the manifest.

    Returns:
        GateResult with value="STRICT" or "RELAXED_CANDIDATE" for manifest stamping.
    """
    if not manifest_path.is_file():
        return GateResult(
            name="ruleset_governance",
            status="FAIL",
            detail=f"Manifest not found: {manifest_path}",
        )

    try:
        with open(manifest_path) as f:
            manifest_data = json.load(f)
    except Exception as e:
        return GateResult(
            name="ruleset_governance",
            status="FAIL",
            detail=f"Failed to load manifest: {e}",
        )

    # Resolve ruleset ID
    if ruleset_path is not None:
        from decision_engine import DecisionRuleset

        try:
            rs = DecisionRuleset.from_json(str(ruleset_path))
            resolved_id = rs.ruleset_id
        except Exception as e:
            return GateResult(
                name="ruleset_governance",
                status="FAIL",
                detail=f"Failed to load ruleset: {e}",
            )
    else:
        from run_screen import PHASE2_PINNED_RULESET_ID

        resolved_id = PHASE2_PINNED_RULESET_ID

    # Look up in manifest entries
    entries = manifest_data.get("rulesets", [])
    entry = None
    for e in entries:
        if e.get("id") == resolved_id:
            entry = e
            break

    if entry is None:
        return GateResult(
            name="ruleset_governance",
            status="FAIL",
            detail=f"Ruleset {resolved_id} not found in manifest",
        )

    status = entry.get("status", "unknown")

    if status == "active":
        return GateResult(
            name="ruleset_governance",
            status="PASS",
            detail=f"Ruleset {resolved_id} is active",
            value="STRICT",
        )

    if status == "candidate" and allow_candidate:
        return GateResult(
            name="ruleset_governance",
            status="WARN",
            detail=f"RELAXED_CANDIDATE: {resolved_id}",
            value="RELAXED_CANDIDATE",
        )

    return GateResult(
        name="ruleset_governance",
        status="FAIL",
        detail=f"Ruleset {resolved_id} has status={status}, expected active",
    )


def _stale_mismatch_held_context(
    audit_output_dir: Optional[Path],
    held_tickers: Optional[set],
) -> str:
    """Annotate STALE_MISMATCH with whether affected tickers are held."""
    if not audit_output_dir or held_tickers is None:
        return ""
    diff_csv = audit_output_dir / "price_recompute_diff.csv"
    if not diff_csv.exists():
        return ""
    try:
        with open(diff_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fail_tickers = set()
            for row in reader:
                for vk in ("dd_verdict", "rsi_verdict", "beta_verdict", "alpha_verdict"):
                    if row.get(vk) == "FAIL":
                        fail_tickers.add(row.get("ticker", "").upper())
                        break
        if not fail_tickers:
            return ""
        held_affected = fail_tickers & held_tickers
        if held_affected:
            return f" [{len(held_affected)}/{len(fail_tickers)} stale in held: {','.join(sorted(held_affected)[:5])}]"
        return f" [0/{len(fail_tickers)} stale in held — none affect portfolio]"
    except Exception:
        return ""


def check_regulatory_calendar(
    snapshot_date_dir: Path,
    as_of_date: str,
    config: GateConfig,
) -> GateResult:
    """Check regulatory calendar health: load success, coverage band, freshness.

    WARN-only gate. Fires if:
      - Manual calendar fails to load or entries_used == 0
      - Eligible regulatory coverage outside [min_pct, max_pct] band
      - Newest as_of_disclosed_at is older than configured threshold
    """
    name = "regulatory_calendar"

    # 1. Check metadata for regulatory coverage
    meta_path = snapshot_date_dir / "metadata.json"
    if not meta_path.is_file():
        return GateResult(
            name=name,
            status="WARN",
            detail="metadata.json not found — cannot check regulatory calendar",
        )

    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except Exception as e:
        return GateResult(name=name, status="WARN", detail=f"metadata.json parse error: {e}")

    reg_cov = meta.get("regulatory_coverage", {})
    manual_n = reg_cov.get("manual_calendar_n_records", 0)
    n_flagged = reg_cov.get("n_eligible_flagged", 0)
    coverage_pct = reg_cov.get("regulatory_secondary_coverage_pct", 0.0)
    entries_used = reg_cov.get("reg_calendar_entries_used", manual_n)

    warnings = []

    # 2. Calendar load check
    if entries_used == 0 and manual_n == 0:
        warnings.append("manual calendar empty or failed to load")

    # 3. Coverage band check [min_pct, max_pct]
    min_cov = config.regulatory_calendar_min_coverage_pct
    max_cov = config.regulatory_calendar_max_coverage_pct
    if min_cov > 0 and coverage_pct < min_cov:
        warnings.append(f"coverage {coverage_pct:.1f}% < {min_cov:.1f}% floor")
    if max_cov > 0 and coverage_pct > max_cov:
        warnings.append(f"coverage {coverage_pct:.1f}% > {max_cov:.1f}% ceiling (over-flagging risk)")

    # 4. Freshness check — newest disclosed_at in the calendar
    try:
        from common.regulatory_calendar import load_and_validate

        records, _ = load_and_validate(as_of_date=as_of_date)
        if records:
            newest_disclosed = max(
                (r.get("as_of_disclosed_at", "") for r in records),
                default="",
            )
            if newest_disclosed:
                try:
                    newest_dt = date.fromisoformat(newest_disclosed)
                    as_of_dt = date.fromisoformat(as_of_date)
                    age_days = (as_of_dt - newest_dt).days
                    if age_days > config.regulatory_calendar_max_stale_days:
                        warnings.append(
                            f"newest disclosed_at={newest_disclosed} "
                            f"is {age_days}d old (threshold={config.regulatory_calendar_max_stale_days}d)"
                        )
                except ValueError:
                    pass
    except Exception as exc:
        _logger.warning(f"Regulatory calendar freshness check failed: {exc}")

    if warnings:
        detail = (
            f"n_manual={manual_n}, used={entries_used}, flagged={n_flagged}, "
            f"coverage={coverage_pct:.1f}% | {'; '.join(warnings)}"
        )
        return GateResult(name=name, status="WARN", detail=detail)

    detail = f"n_manual={manual_n}, used={entries_used}, flagged={n_flagged}, coverage={coverage_pct:.1f}%"
    return GateResult(name=name, status="PASS", detail=detail)


def check_options_coverage(
    staging_date_dir: Path,
) -> GateResult:
    """WARN-only gate: check options_quality_composite population in snapshot.

    Reports TT diagnostics coverage so silent degradation (expired creds,
    API outage) is caught immediately instead of discovered weeks later
    when the clinical_plus_options A/B window is unusable.

    Never FAIL — options data is not required for the active production
    ruleset.  WARN when coverage drops to zero or credentials are missing.
    """
    name = "options_coverage"
    rankings_path = staging_date_dir / "rankings.csv"
    if not rankings_path.exists():
        return GateResult(name=name, status="PASS", detail="no rankings.csv (pre-screen)")

    try:
        with open(rankings_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except Exception as exc:
        return GateResult(name=name, status="WARN", detail=f"could not read rankings: {exc}")

    n_total = len(rows)
    if n_total == 0:
        return GateResult(name=name, status="PASS", detail="empty rankings")

    n_has_data = sum(1 for r in rows if str(r.get("opt_has_data", "0")).strip() == "1")
    n_liquid = sum(1 for r in rows if r.get("opt_liquidity_state") == "liquid")
    n_thin = sum(1 for r in rows if r.get("opt_liquidity_state") == "thin")
    n_oqc = sum(1 for r in rows if r.get("options_quality_composite", "").strip() not in ("", "0", "0.0"))
    # Step-10 eligible: secondary regulatory path (91-180d) with nonzero OQC.
    # Mirrors decision_engine.py Step 10: has_regulatory_upcoming_180d=1,
    # regulatory_days in (90, 180], and options_quality_composite > 0.
    n_step10_oqc = 0
    for r in rows:
        if r.get("options_quality_composite", "").strip() in ("", "0", "0.0"):
            continue
        if str(r.get("has_regulatory_upcoming_180d", "")).strip() != "1":
            continue
        try:
            reg_d = float(r.get("regulatory_days", ""))
        except (ValueError, TypeError):
            continue
        if reg_d > 90 and reg_d <= 180:
            n_step10_oqc += 1
    no_creds = any(r.get("opt_diagnostic_basis") == "no_credentials" for r in rows[:1])

    value = {
        "n_total": n_total,
        "n_has_data": n_has_data,
        "n_liquid": n_liquid,
        "n_thin": n_thin,
        "n_oqc_nonzero": n_oqc,
        "n_step10_eligible_oqc": n_step10_oqc,
        "ab_ready": n_oqc > 0,
        "has_credentials": not no_creds,
    }

    detail_parts = [
        f"opt_has_data={n_has_data}/{n_total}",
        f"liquid={n_liquid}",
        f"thin={n_thin}",
        f"oqc_nonzero={n_oqc}",
        f"step10_oqc={n_step10_oqc}",
    ]

    if no_creds:
        return GateResult(
            name=name,
            status="WARN",
            detail="TT credentials missing — options_quality_composite unpopulated; " + ", ".join(detail_parts),
            value=value,
        )

    if n_oqc == 0 and n_has_data == 0:
        return GateResult(
            name=name,
            status="WARN",
            detail="zero options data (API outage or no liquid chains?); " + ", ".join(detail_parts),
            value=value,
        )

    if n_oqc == 0 and n_has_data > 0:
        return GateResult(
            name=name,
            status="WARN",
            detail="options data present but zero OQC (all filtered by use_for_judgment?); " + ", ".join(detail_parts),
            value=value,
        )

    return GateResult(
        name=name,
        status="PASS",
        detail=", ".join(detail_parts),
        value=value,
    )


# ---------------------------------------------------------------------------
# Hard-catalyst production gates (Spec 018)
# ---------------------------------------------------------------------------


def _load_queue_json(staging_date_dir: Path) -> Optional[Dict[str, Any]]:
    """Load options_review_queue.json from snapshot dir."""
    path = staging_date_dir / "options_review_queue.json"
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def check_hard_queue_artifacts(staging_date_dir: Path) -> GateResult:
    """Gate: verify hard queue JSON and CSV exist and are readable."""
    name = "hard_queue_artifacts"
    json_path = staging_date_dir / "options_review_queue.json"
    csv_path = staging_date_dir / "options_review_queue.csv"

    json_ok = False
    csv_ok = False

    if json_path.exists():
        try:
            with open(json_path, encoding="utf-8") as f:
                json.load(f)
            json_ok = True
        except Exception:
            pass

    if csv_path.exists():
        try:
            with open(csv_path, encoding="utf-8") as f:
                next(csv.reader(f))  # read header
            csv_ok = True
        except Exception:
            pass

    detail = f"queue_json={'ok' if json_ok else 'MISSING'}, queue_csv={'ok' if csv_ok else 'MISSING'}"
    if json_ok and csv_ok:
        return GateResult(name=name, status="PASS", detail=detail)
    return GateResult(name=name, status="FAIL", detail=detail)


def check_hard_catalyst_supply(
    staging_date_dir: Path,
    config: GateConfig,
) -> GateResult:
    """Gate: ensure queue has enough hard-catalyst rows."""
    name = "hard_catalyst_supply"
    queue = _load_queue_json(staging_date_dir)
    if queue is None:
        return GateResult(name=name, status="FAIL", detail="queue_json missing or unreadable")

    rows = queue.get("rows", [])
    n_hard = sum(1 for r in rows if str(r.get("is_hard_catalyst", "0")).strip() == "1")
    n_hard_0_90 = 0
    for r in rows:
        if str(r.get("is_hard_catalyst", "0")).strip() != "1":
            continue
        try:
            cd = float(r.get("catalyst_days", 9999))
        except (ValueError, TypeError):
            continue
        if 0 < cd <= 90:
            n_hard_0_90 += 1

    detail = f"hard={n_hard}, hard_0_90d={n_hard_0_90}"
    value = {"n_hard": n_hard, "n_hard_0_90d": n_hard_0_90}

    status = "PASS"
    if n_hard < config.hard_queue_min_fail:
        status = "FAIL"
    elif n_hard < config.hard_queue_min_warn:
        status = "WARN"

    if n_hard_0_90 < config.hard_queue_near_term_min_fail:
        status = "FAIL"
    elif n_hard_0_90 < config.hard_queue_near_term_min_warn and status != "FAIL":
        status = "WARN"

    return GateResult(name=name, status=status, detail=detail, value=value)


def check_hard_options_coverage(
    staging_date_dir: Path,
    config: GateConfig,
) -> GateResult:
    """Gate: measure options enrichment on hard-catalyst rows."""
    name = "hard_options_coverage"
    queue = _load_queue_json(staging_date_dir)
    if queue is None:
        return GateResult(name=name, status="FAIL", detail="queue_json missing")

    rows = queue.get("rows", [])
    hard_rows = [r for r in rows if str(r.get("is_hard_catalyst", "0")).strip() == "1"]
    n_hard = len(hard_rows)
    if n_hard == 0:
        return GateResult(name=name, status="PASS", detail="no hard rows to check")

    n_with_iv = sum(1 for r in hard_rows if str(r.get("opt_atm_iv", "")).strip() not in ("", "0", "0.0"))
    n_with_straddle = sum(1 for r in hard_rows if str(r.get("cheap_vol_score", "")).strip() not in ("", "0", "0.0"))
    n_reviewable = sum(
        1
        for r in hard_rows
        if any(
            t in (r.get("review_reasons", "") or "")
            for t in ("cheap_straddle", "rich_straddle", "high_disagreement", "term_structure", "extreme_skew")
        )
    )

    opt_pct = 100.0 * n_with_iv / n_hard
    straddle_pct = 100.0 * n_with_straddle / n_hard

    detail = f"hard={n_hard}, opt_cov={opt_pct:.1f}%, straddle_cov={straddle_pct:.1f}%, reviewable={n_reviewable}"
    value = {
        "n_hard": n_hard,
        "n_with_iv": n_with_iv,
        "n_with_straddle": n_with_straddle,
        "n_reviewable": n_reviewable,
        "opt_pct": opt_pct,
        "straddle_pct": straddle_pct,
    }

    status = "PASS"
    if opt_pct < config.hard_options_coverage_fail_pct:
        status = "FAIL"
    elif opt_pct < config.hard_options_coverage_warn_pct:
        status = "WARN"

    if straddle_pct < config.hard_actual_straddle_fail_pct:
        status = "FAIL"
    elif straddle_pct < config.hard_actual_straddle_warn_pct and status != "FAIL":
        status = "WARN"

    if n_reviewable == 0 and status != "FAIL":
        status = "WARN"

    return GateResult(name=name, status=status, detail=detail, value=value)


def check_hard_carry_state(
    staging_date_dir: Path,
    as_of_date: str,
) -> GateResult:
    """Gate: verify forward-carry state health and no backslides."""
    name = "hard_carry_state"

    # Find carry state file — check multiple possible locations
    candidates = [
        staging_date_dir.parent / "state" / "hard_catalyst_carry.json",
        staging_date_dir.parent.parent / "state" / "hard_catalyst_carry.json",
        staging_date_dir.parent.parent / "data" / "state" / "hard_catalyst_carry.json",
    ]
    state_path = None
    for c in candidates:
        if c.exists():
            state_path = c
            break

    if state_path is None:
        # Check if there are hard catalysts that should have been learned
        queue = _load_queue_json(staging_date_dir)
        if queue and queue.get("summary", {}).get("n_hard_catalyst", 0) > 0:
            return GateResult(
                name=name,
                status="WARN",
                detail="carry state absent but hard queue non-empty",
            )
        return GateResult(name=name, status="PASS", detail="no carry state (no hard events yet)")

    try:
        with open(state_path, encoding="utf-8") as f:
            state = json.load(f)
    except Exception as exc:
        return GateResult(name=name, status="FAIL", detail=f"carry state unreadable: {exc}")

    n_entries = len(state)

    # Check for backslides: tickers in carry state with unexpired events
    # that still show soft source in rankings
    rankings_path = staging_date_dir / "rankings.csv"
    n_backslides = 0
    if rankings_path.exists():
        try:
            with open(rankings_path, encoding="utf-8") as f:
                rankings = {r["ticker"]: r for r in csv.DictReader(f)}
        except Exception:
            rankings = {}

        soft_sources = {"CTGOV_CALENDAR", "CTGOV_PCD_FAR", ""}
        for ticker, entry in state.items():
            est_date = entry.get("estimated_event_date", "")
            if est_date and est_date < as_of_date:
                continue  # expired, ok
            if ticker in rankings:
                src = rankings[ticker].get("catalyst_source", "")
                is_hard = str(rankings[ticker].get("is_hard_catalyst", "0")).strip() == "1"
                if src in soft_sources and not is_hard:
                    n_backslides += 1

    detail = f"state_entries={n_entries}, backslides={n_backslides}"
    if n_backslides > 0:
        return GateResult(
            name=name, status="FAIL", detail=detail, value={"n_entries": n_entries, "n_backslides": n_backslides}
        )
    return GateResult(name=name, status="PASS", detail=detail, value={"n_entries": n_entries, "n_backslides": 0})


def check_hard_queue_actionability(
    staging_date_dir: Path,
    config: GateConfig,
) -> GateResult:
    """Gate: measure whether hard queue contains reviewable names."""
    name = "hard_queue_actionability"
    queue = _load_queue_json(staging_date_dir)
    if queue is None:
        return GateResult(name=name, status="WARN", detail="queue_json missing")

    rows = queue.get("rows", [])
    hard_rows = [r for r in rows if str(r.get("is_hard_catalyst", "0")).strip() == "1"]

    n_reviewable = 0
    n_cheap_rich = 0
    n_disagree = 0
    n_ts = 0
    n_skew = 0

    for r in hard_rows:
        reasons = r.get("review_reasons", "") or ""
        has_signal = False
        if "cheap_straddle" in reasons or "rich_straddle" in reasons:
            n_cheap_rich += 1
            has_signal = True
        if "high_disagreement" in reasons:
            n_disagree += 1
            has_signal = True
        if "term_structure" in reasons:
            n_ts += 1
            has_signal = True
        if "extreme_skew" in reasons:
            n_skew += 1
            has_signal = True
        if has_signal:
            n_reviewable += 1

    detail = (
        f"reviewable={n_reviewable}, cheap_rich={n_cheap_rich}, disagreement={n_disagree}, ts={n_ts}, skew={n_skew}"
    )
    value = {
        "n_reviewable": n_reviewable,
        "n_cheap_rich": n_cheap_rich,
        "n_disagree": n_disagree,
        "n_ts": n_ts,
        "n_skew": n_skew,
    }

    if n_reviewable >= config.hard_reviewable_min_warn:
        status = "PASS"
    else:
        status = "WARN"

    return GateResult(name=name, status=status, detail=detail, value=value)


def check_audit_result(
    audit_proc: subprocess.CompletedProcess,
    config: GateConfig,
    audit_output_dir: Optional[Path] = None,
    *,
    held_tickers: Optional[set] = None,
) -> GateResult:
    """Translate audit tool exit code into a gate result."""
    summary = _read_invariants_summary(audit_output_dir) if audit_output_dir else None

    if audit_proc.returncode == 0:
        detail = _format_audit_detail("Audit OK", summary)
        return GateResult(name="audit", status="PASS", detail=detail)
    elif audit_proc.returncode == 1:
        # Critical invariant violation (data model broken)
        status = "FAIL" if config.audit_fail_is_gate_fail else "WARN"
        detail = _format_audit_detail("CRITICAL: invariant violation", summary)
        return GateResult(name="audit", status=status, detail=detail)
    elif audit_proc.returncode == 2:
        status = "WARN" if config.audit_warn_is_gate_warn else "PASS"
        detail = _format_audit_detail("Audit WARN", summary)
        return GateResult(name="audit", status=status, detail=detail)
    elif audit_proc.returncode == 3:
        # Stale recompute mismatch — always WARN (not FAIL)
        held_ctx = _stale_mismatch_held_context(audit_output_dir, held_tickers)
        detail = _format_audit_detail(f"STALE_MISMATCH: price recompute diff{held_ctx}", summary)
        return GateResult(name="audit", status="WARN", detail=detail)
    else:
        # Unknown exit code → WARN (defensive)
        detail = _format_audit_detail(
            f"Audit unknown exit code {audit_proc.returncode}",
            summary,
        )
        return GateResult(name="audit", status="WARN", detail=detail)


def _parse_cache_date(p: Path) -> Optional[date]:
    """Extract and parse YYYY-MM-DD from a trial_records_{date}.json filename."""

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
                name="ctgov_cache",
                status="PASS",
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
                    name="ctgov_cache",
                    status="WARN",
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
            name="ctgov_cache",
            status="FAIL",
            detail=(
                f"PIT cache missing: {exact.name}. " f"Run: warm_caches.py --as-of-date {as_of_date} --sources ctgov"
            ),
        ),
        as_of_date,
    )


def check_inputs_present(data_dir: Path) -> GateResult:
    """Check that required (gitignored) input files exist before running screen.

    Required files that are tracked in git (universe.json, financial_records.json,
    trial_records.json) are always present after checkout.  This gate checks for
    files that are gitignored and must come from an inputs bundle or local setup.
    """
    required = [
        ("market_data.json", "Market data (pricing, industry, volume)"),
    ]
    missing = []
    for filename, desc in required:
        if not (data_dir / filename).exists():
            missing.append(filename)

    if missing:
        return GateResult(
            name="inputs_present",
            status="FAIL",
            detail=f"missing: {', '.join(missing)}",
        )
    return GateResult(
        name="inputs_present",
        status="PASS",
        detail=f"All required inputs found in {data_dir.name}/",
    )


def check_market_data_staleness(
    data_dir: Path,
    as_of_date: str,
    max_age_days: int = 3,
) -> GateResult:
    """Check that market_data.json is fresh relative to the screen date.

    Reads the ``collected_at`` field from the first record and computes
    calendar-day age against ``as_of_date``.  FAIL if age > max_age_days.
    """
    from datetime import date as _date

    mkt_path = data_dir / "market_data.json"
    if not mkt_path.exists():
        # inputs_present gate handles this — don't double-fail
        return GateResult(
            name="market_data_staleness",
            status="PASS",
            detail="Skipped (file missing; inputs_present gate will catch)",
        )

    try:
        with open(mkt_path) as f:
            records = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        return GateResult(
            name="market_data_staleness",
            status="FAIL",
            detail=f"Cannot read market_data.json: {e}",
        )

    # Find the most common collected_at date
    collected_dates = [r.get("collected_at") for r in records if isinstance(r, dict) and r.get("collected_at")]
    if not collected_dates:
        return GateResult(
            name="market_data_staleness",
            status="FAIL",
            detail="No collected_at field found in market_data.json",
        )

    collected_at = max(collected_dates)  # latest collection date
    try:
        collected = _date.fromisoformat(collected_at)
        as_of = _date.fromisoformat(as_of_date)
    except ValueError as e:
        return GateResult(
            name="market_data_staleness",
            status="FAIL",
            detail=f"Bad date format: {e}",
        )

    age_days = (as_of - collected).days
    if age_days > max_age_days:
        return GateResult(
            name="market_data_staleness",
            status="FAIL",
            detail=(
                f"market_data.json is {age_days}d stale "
                f"(collected={collected_at}, as_of={as_of_date}, max={max_age_days}d). "
                "Run: python collect_market_data.py"
            ),
            value=age_days,
            threshold=max_age_days,
        )

    return GateResult(
        name="market_data_staleness",
        status="PASS",
        detail=f"market_data.json collected={collected_at}, age={age_days}d",
        value=age_days,
        threshold=max_age_days,
    )


def check_market_data_schema(data_dir: Path) -> GateResult:
    """Validate market_data.json record structure.

    Checks that every record has required fields (ticker, price, market_cap,
    collected_at) and that numeric fields are actually numeric or None.
    Returns FAIL if >0 records are malformed; PASS otherwise.
    """
    mkt_path = data_dir / "market_data.json"
    if not mkt_path.exists():
        return GateResult(
            name="market_data_schema",
            status="PASS",
            detail="Skipped (file missing; inputs_present gate will catch)",
        )

    try:
        with open(mkt_path) as f:
            records = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        return GateResult(
            name="market_data_schema",
            status="FAIL",
            detail=f"Cannot parse market_data.json: {e}",
        )

    if not isinstance(records, list):
        return GateResult(
            name="market_data_schema",
            status="FAIL",
            detail="market_data.json is not a JSON array",
        )

    if len(records) == 0:
        return GateResult(
            name="market_data_schema",
            status="FAIL",
            detail="market_data.json is empty",
        )

    bad_records: list[str] = []
    for i, rec in enumerate(records):
        if not isinstance(rec, dict):
            bad_records.append(f"[{i}]: not a dict")
            continue
        missing = MARKET_DATA_REQUIRED_FIELDS - set(rec.keys())
        if missing:
            bad_records.append(f"{rec.get('ticker', f'[{i}]')}: missing {sorted(missing)}")
            continue
        for fld in MARKET_DATA_NUMERIC_FIELDS:
            val = rec.get(fld)
            if val is not None and not isinstance(val, (int, float)):
                bad_records.append(f"{rec['ticker']}: {fld} is {type(val).__name__}")
                break

    if bad_records:
        sample = "; ".join(bad_records[:3])
        return GateResult(
            name="market_data_schema",
            status="FAIL",
            detail=f"{len(bad_records)} invalid records: {sample}",
            value=len(bad_records),
            threshold=0,
        )

    return GateResult(
        name="market_data_schema",
        status="PASS",
        detail=f"{len(records)} records, all valid",
        value=0,
        threshold=0,
    )


def check_market_data_coverage(
    data_dir: Path,
    min_coverage: float = 0.90,
) -> GateResult:
    """Check that market_data.json covers enough of the universe.

    Compares ticker sets: market_data tickers vs universe.json tickers.
    FAIL if coverage < min_coverage.  Synthetic tickers (prefixed with _)
    are excluded from the denominator.
    """
    mkt_path = data_dir / "market_data.json"
    uni_path = data_dir / "universe.json"

    if not mkt_path.exists():
        return GateResult(
            name="market_data_coverage",
            status="PASS",
            detail="Skipped (file missing; inputs_present gate will catch)",
        )
    if not uni_path.exists():
        return GateResult(
            name="market_data_coverage",
            status="FAIL",
            detail="universe.json not found",
        )

    try:
        with open(mkt_path) as f:
            records = json.load(f)
        with open(uni_path) as f:
            universe = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        return GateResult(
            name="market_data_coverage",
            status="FAIL",
            detail=f"Cannot read files: {e}",
        )

    mkt_tickers = {r["ticker"] for r in records if isinstance(r, dict) and r.get("ticker")}
    uni_tickers = set()
    for entry in universe:
        if isinstance(entry, dict) and entry.get("status") == "delisted":
            continue  # exclude delisted tickers from coverage denominator
        t = entry.get("ticker") if isinstance(entry, dict) else str(entry)
        if t and not t.startswith("_"):  # exclude synthetic like _XBI_BENCHMARK_
            uni_tickers.add(t)

    if not uni_tickers:
        return GateResult(
            name="market_data_coverage",
            status="FAIL",
            detail="Universe is empty",
        )

    covered = len(mkt_tickers & uni_tickers)
    coverage = covered / len(uni_tickers)
    missing = sorted(uni_tickers - mkt_tickers)

    if coverage < min_coverage:
        sample = ", ".join(missing[:5])
        return GateResult(
            name="market_data_coverage",
            status="FAIL",
            detail=(
                f"{covered}/{len(uni_tickers)} ({coverage:.1%}) coverage, "
                f"below {min_coverage:.0%} threshold. "
                f"Missing: {sample}{'...' if len(missing) > 5 else ''}"
            ),
            value=round(coverage, 4),
            threshold=min_coverage,
        )

    detail = f"{covered}/{len(uni_tickers)} ({coverage:.1%}) coverage"
    if missing:
        detail += f", {len(missing)} missing: {', '.join(missing[:5])}"
    return GateResult(
        name="market_data_coverage",
        status="PASS",
        detail=detail,
        value=round(coverage, 4),
        threshold=min_coverage,
    )


def _compute_market_data_refresh(
    data_dir: Path,
    as_of_date: str,
) -> Dict[str, Any]:
    """Build the market_data_refresh manifest block.

    Returns provenance metadata about market_data.json for the manifest.
    """
    mkt_path = data_dir / "market_data.json"
    info: Dict[str, Any] = {
        "collected_at": None,
        "age_days": None,
        "ticker_count": 0,
        "coverage_pct": None,
        "sha256": None,
    }

    if not mkt_path.exists():
        return info

    try:
        raw = mkt_path.read_bytes()
        info["sha256"] = hashlib.sha256(raw).hexdigest()
        records = json.loads(raw)
    except (json.JSONDecodeError, OSError):
        return info

    if not isinstance(records, list):
        return info

    info["ticker_count"] = len(records)

    # collected_at
    dates = [r.get("collected_at") for r in records if isinstance(r, dict) and r.get("collected_at")]
    if dates:
        info["collected_at"] = max(dates)
        try:
            from datetime import date as _date

            info["age_days"] = (_date.fromisoformat(as_of_date) - _date.fromisoformat(info["collected_at"])).days
        except ValueError:
            pass

    # coverage vs universe
    uni_path = data_dir / "universe.json"
    if uni_path.exists():
        try:
            with open(uni_path) as f:
                universe = json.load(f)
            uni_tickers = set()
            for entry in universe:
                if isinstance(entry, dict) and entry.get("status") == "delisted":
                    continue  # exclude delisted tickers from coverage denominator
                t = entry.get("ticker") if isinstance(entry, dict) else str(entry)
                if t and not t.startswith("_"):
                    uni_tickers.add(t)
            mkt_tickers = {r["ticker"] for r in records if isinstance(r, dict) and r.get("ticker")}
            if uni_tickers:
                info["coverage_pct"] = round(
                    len(mkt_tickers & uni_tickers) / len(uni_tickers),
                    4,
                )
        except (json.JSONDecodeError, OSError):
            pass

    return info


# ---------------------------------------------------------------------------
# Step 5: Run manifest
# ---------------------------------------------------------------------------


def _safe_file_hash(path: Optional[Path]) -> str:
    """SHA-256 of a file, or "" if missing/unreadable. Never raises."""
    try:
        if path and Path(path).exists() and Path(path).is_file():
            return sha256_file(Path(path))
    except (OSError, ValueError):
        pass
    return ""


def _compute_provenance_hashes(
    *,
    as_of_date: str,
    data_dir: Optional[Path],
    price_csv: Optional[Path],
    ctgov_cache_dir: Optional[Path],
    snapshot_date_dir: Optional[Path],
    model_path: Optional[Path] = None,
) -> Dict[str, str]:
    """Compute content hashes of the inputs/outputs that define a run's state.

    Pure provenance — does NOT change scoring, ranking, or gates. Lets the
    validation ledger point at the exact production state used. Every hash
    degrades to "" rather than raising, so a missing file never aborts the run.
    """
    _model = model_path or (REPO_ROOT / "production_data" / "ranker_v2_model.json")
    _universe = (data_dir / "universe.json") if data_dir else None
    _ctgov_dir = ctgov_cache_dir or (REPO_ROOT / "cache" / "ctgov")
    _clinical = Path(_ctgov_dir) / f"trial_records_{as_of_date}.json"
    _rankings = (snapshot_date_dir / "rankings.csv") if snapshot_date_dir else None
    return {
        "model_hash": _safe_file_hash(_model),
        "universe_hash": _safe_file_hash(_universe),
        "price_hash": _safe_file_hash(price_csv),
        "clinical_hash": _safe_file_hash(_clinical),
        "rankings_hash": _safe_file_hash(_rankings),
    }


def detect_no_material_input_change(
    as_of_date: str,
    *,
    data_dir: Path,
    price_csv: Path,
    ctgov_cache_dir: Optional[Path],
    final_snapshots_dir: Path,
) -> Optional[Dict[str, Any]]:
    """Logging-only no-op detector (item F).

    Compares the current run's INPUT hashes (model, universe, price, clinical)
    against the most recent prior snapshot's run_manifest.json. Returns a dict
    describing the comparison, or None if no prior manifest is available.

    CONSERVATIVE: this never skips work or snapshot creation — it only surfaces
    NO_MATERIAL_INPUT_CHANGE so operators can distinguish real rank movement
    from pipeline churn. Any introduction of skip behavior must be test-covered.
    """
    try:
        prior_dates = sorted(
            d.name for d in final_snapshots_dir.iterdir() if d.is_dir() and d.name < as_of_date and len(d.name) == 10
        )
    except OSError:
        return None
    prior_manifest_hashes: Dict[str, str] = {}
    for _pd in reversed(prior_dates):
        _mp = final_snapshots_dir / _pd / "run_manifest.json"
        if _mp.exists():
            try:
                prior_manifest_hashes = json.loads(_mp.read_text(encoding="utf-8")).get("hashes", {}) or {}
            except (json.JSONDecodeError, OSError):
                continue
            if prior_manifest_hashes:
                break
    if not prior_manifest_hashes:
        return None

    current = _compute_provenance_hashes(
        as_of_date=as_of_date,
        data_dir=data_dir,
        price_csv=price_csv,
        ctgov_cache_dir=ctgov_cache_dir,
        snapshot_date_dir=None,  # rankings is an OUTPUT — excluded from input no-op test
    )
    # Compare only INPUT hashes that are meaningful pre-screen.
    input_keys = ("model_hash", "universe_hash", "price_hash", "clinical_hash")
    changed = [
        k
        for k in input_keys
        if current.get(k) and prior_manifest_hashes.get(k) and current.get(k) != prior_manifest_hashes.get(k)
    ]
    unchanged = [k for k in input_keys if current.get(k) and current.get(k) == prior_manifest_hashes.get(k)]
    no_op = bool(unchanged) and not changed
    return {
        "no_material_input_change": no_op,
        "unchanged_inputs": unchanged,
        "changed_inputs": changed,
        "compared_against": next(
            (d for d in reversed(prior_dates) if (final_snapshots_dir / d / "run_manifest.json").exists()), None
        ),
    }


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
    data_dir: Optional[Path] = None,
    mode: str = "daily-production",
    price_csv: Optional[Path] = None,
    ctgov_cache_dir: Optional[Path] = None,
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
        # Guard the JSON parse: a corrupt/truncated metadata.json must not
        # crash the manifest build — we still want a partial manifest with
        # gate verdicts and an overall FAIL recorded downstream.
        meta: Dict[str, Any] = {}
        try:
            meta = json.loads((snapshot_date_dir / "metadata.json").read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            meta = {}
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
                health = json.loads(health_path.read_text(encoding="utf-8"))
                if health.get("ruleset_id"):
                    ruleset_info["ruleset_hash"] = health["ruleset_id"]
            except (json.JSONDecodeError, OSError, UnicodeDecodeError):
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

    # Explicit production health categories (item #11): a FAIL gate is BLOCKING
    # (snapshot not promoted), a WARN is NON-BLOCKING (promoted but flagged).
    # Lets downstream consumers react proportionally instead of treating every
    # gate verdict the same.
    health_summary = {
        "blocking_failures": [g.name for g in gate_results if g.status == "FAIL"],
        "warnings": [g.name for g in gate_results if g.status == "WARN"],
        "n_pass": sum(1 for g in gate_results if g.status == "PASS"),
        "n_warn": sum(1 for g in gate_results if g.status == "WARN"),
        "n_fail": sum(1 for g in gate_results if g.status == "FAIL"),
    }

    _requested = requested_as_of_date or as_of_date

    # Enrich git block with pre/post-run dirty flags
    git_block = dict(git)  # shallow copy to avoid mutating caller's dict
    git_block["dirty_pre_run"] = git.get("dirty")
    git_block["dirty_post_run"] = git_post_run.get("dirty") if git_post_run else None
    git_block["dirty"] = git_block["dirty_pre_run"]  # backward compat

    # Sanitize gate values for JSON serialization (Path, date, etc.)
    def _json_safe(obj: Any) -> Any:
        if isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, (datetime,)):
            return obj.isoformat()
        if hasattr(obj, "isoformat"):  # date, time
            return obj.isoformat()
        if isinstance(obj, dict):
            return {k: _json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_json_safe(v) for v in obj]
        return str(obj)

    for g in gate_results:
        g.value = _json_safe(g.value)
        g.threshold = _json_safe(g.threshold)

    # Validate gate names against allowlist
    for g in gate_results:
        if g.name not in GATE_ALLOWLIST:
            raise ValueError(f"Gate '{g.name}' not in GATE_ALLOWLIST. " "Add it to the allowlist before using it.")

    # Market data refresh provenance
    mkt_refresh: Dict[str, Any] = {}
    if data_dir:
        mkt_refresh = _compute_market_data_refresh(data_dir, as_of_date)

    # Content hashes of the inputs/outputs that define this run's state.
    # Mirror ruleset_hash into the hashes block so the ledger has one canonical
    # provenance map (model + ruleset + universe + price + clinical + rankings).
    provenance_hashes = _compute_provenance_hashes(
        as_of_date=as_of_date,
        data_dir=data_dir,
        price_csv=price_csv,
        ctgov_cache_dir=ctgov_cache_dir,
        snapshot_date_dir=snapshot_date_dir,
    )
    provenance_hashes["ruleset_hash"] = ruleset_info.get("ruleset_hash", "")

    return {
        "manifest_version": MANIFEST_VERSION,
        "mode": mode,
        "requested_as_of_date": _requested,
        "effective_as_of_date": as_of_date,
        "as_of_date": as_of_date,  # backward compat alias
        "generated_at": _deterministic_timestamp(as_of_date),
        "git": git_block,
        "hashes": provenance_hashes,
        "ruleset": ruleset_info,
        "row_counts": row_counts,
        "price_refresh": {
            "n_extended": price_stats.get("n_extended"),
            "n_rows_appended": price_stats.get("n_rows_appended"),
            "n_failed": price_stats.get("n_failed"),
            "failed_tickers": price_stats.get("failed_tickers", []),
            "xbi_last_date": price_stats.get("xbi_last_date"),
        },
        "market_data_refresh": mkt_refresh,
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
        "health_summary": health_summary,
        "screen_exit_code": screen_proc.returncode,
        "audit_exit_code": audit_proc.returncode if audit_proc else None,
        "gate_config": {k: v for k, v in asdict(config).items()},
    }


# ---------------------------------------------------------------------------
# Gate verdict ledger (JSONL time-series)
# ---------------------------------------------------------------------------

GATE_LEDGER_PATH = REPO_ROOT / "artifacts" / "gate_verdict_ledger.jsonl"


def append_gate_verdict(manifest: Dict[str, Any]) -> None:
    """Append a single-line JSON record to the gate verdict ledger.

    Each row captures the date, overall status, per-gate verdicts, and
    minimal provenance (ruleset hash, git SHA).  The ledger is the
    authoritative time-series for SLO / error-budget computation.
    """
    gates = manifest.get("gates", [])
    row = {
        "as_of_date": manifest.get("effective_as_of_date", manifest.get("as_of_date")),
        "generated_at": manifest.get("generated_at"),
        "overall_status": manifest.get("overall_status"),
        "gates": {g["name"]: g["status"] for g in gates},
        "n_pass": sum(1 for g in gates if g["status"] == "PASS"),
        "n_warn": sum(1 for g in gates if g["status"] == "WARN"),
        "n_fail": sum(1 for g in gates if g["status"] == "FAIL"),
        "ruleset_hash": (manifest.get("ruleset") or {}).get("ruleset_hash", ""),
        "git_sha": (manifest.get("git") or {}).get("sha", ""),
    }
    GATE_LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(GATE_LEDGER_PATH, "a") as f:
        f.write(json.dumps(row, default=str) + "\n")


# ---------------------------------------------------------------------------
# Step 6: Atomic promotion
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Step progress tracker — idempotent reruns + agent visibility
# ---------------------------------------------------------------------------

_PROGRESS_FILE = "_step_progress.json"


def _load_progress(snap_dir: Path) -> Dict[str, Any]:
    """Load step progress from staging/snapshot directory."""
    p = snap_dir / _PROGRESS_FILE
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {"steps": {}, "started_at": _timestamp_for_snapshot_dir(snap_dir)}


def _mark_step(snap_dir: Path, step_name: str, status: str = "done", detail: str = "") -> None:
    """Mark a pipeline step as completed. Enables idempotent reruns."""
    progress = _load_progress(snap_dir)
    timestamp = _timestamp_for_snapshot_dir(snap_dir)
    progress["steps"][step_name] = {
        "status": status,
        "completed_at": timestamp,
        "detail": detail,
    }
    progress["last_step"] = step_name
    progress["last_updated"] = timestamp
    snap_dir.mkdir(parents=True, exist_ok=True)
    (snap_dir / _PROGRESS_FILE).write_text(json.dumps(progress, indent=2) + "\n", encoding="utf-8")


def _step_done(snap_dir: Path, step_name: str) -> bool:
    """Check if a step was already completed (for idempotent reruns)."""
    progress = _load_progress(snap_dir)
    step = progress.get("steps", {}).get(step_name, {})
    return step.get("status") == "done"


# ---------------------------------------------------------------------------
# Snapshot promotion
# ---------------------------------------------------------------------------


def _snapshot_backup_path(
    final_snapshots_dir: Path,
    final_date_dir: Path,
    staging_date_dir: Path,
    as_of_date: str,
) -> Path:
    """Build a deterministic backup path from old/new snapshot content hashes."""
    existing_rankings = final_date_dir / "rankings.csv"
    staging_rankings = staging_date_dir / "rankings.csv"
    existing_hash = (
        hashlib.sha256(existing_rankings.read_bytes()).hexdigest()[:12] if existing_rankings.exists() else "noold"
    )
    staging_hash = (
        hashlib.sha256(staging_rankings.read_bytes()).hexdigest()[:12] if staging_rankings.exists() else "nonew"
    )
    base = final_snapshots_dir / f"{as_of_date}__pre_{existing_hash}_{staging_hash}"
    if not base.exists():
        return base
    suffix = 1
    while True:
        candidate = final_snapshots_dir / f"{base.name}_{suffix:02d}"
        if not candidate.exists():
            return candidate
        suffix += 1


def promote_snapshot(
    staging_date_dir: Path,
    final_snapshots_dir: Path,
    as_of_date: str,
) -> Path:
    """Atomically move staging snapshot to final location.

    Idempotency guard: if a snapshot already exists for this date with a
    rankings.csv that has the same content hash as the staging snapshot,
    skip promotion entirely (preserves the original clean snapshot for CRT
    prediction recording and April holdout integrity).

    Uses rename when on same filesystem, falls back to copy+delete.
    Returns the final path.
    """
    final_date_dir = final_snapshots_dir / as_of_date

    # --- Snapshot overwrite protection ---
    # If final snapshot already exists with rankings.csv, compare content hashes.
    # If identical, this is an idempotent rerun — skip to preserve the original.
    if final_date_dir.exists():
        existing_rankings = final_date_dir / "rankings.csv"
        staging_rankings = staging_date_dir / "rankings.csv"
        if existing_rankings.exists() and staging_rankings.exists():
            existing_hash = hashlib.sha256(existing_rankings.read_bytes()).hexdigest()[:16]
            staging_hash = hashlib.sha256(staging_rankings.read_bytes()).hexdigest()[:16]
            if existing_hash == staging_hash:
                import logging

                logging.getLogger(__name__).info(
                    "SNAPSHOT OVERWRITE PROTECTION: %s already exists with identical "
                    "rankings (hash=%s). Skipping promotion (idempotent rerun).",
                    as_of_date,
                    existing_hash,
                )
                # Clean up staging
                shutil.rmtree(str(staging_date_dir), ignore_errors=True)
                return final_date_dir

    if final_date_dir.exists():
        # Archive existing by renaming with content-derived suffix.
        backup = _snapshot_backup_path(final_snapshots_dir, final_date_dir, staging_date_dir, as_of_date)
        import logging

        logging.getLogger(__name__).info(
            "SNAPSHOT OVERWRITE: %s exists with different content. " "Archiving to %s before replacing.",
            as_of_date,
            backup.name,
        )
        try:
            shutil.move(str(final_date_dir), str(backup))
        except (OSError, PermissionError):
            # WSL2: Windows may hold directory handles; fall back to copy+delete
            try:
                shutil.copytree(str(final_date_dir), str(backup))
                shutil.rmtree(str(final_date_dir), ignore_errors=True)
            except (OSError, PermissionError):
                # Last resort: leave existing in place, overwrite individual files
                pass

    if not final_date_dir.exists():
        try:
            os.rename(str(staging_date_dir), str(final_date_dir))
        except OSError:
            # Cross-filesystem: copy then delete
            shutil.copytree(str(staging_date_dir), str(final_date_dir))
            shutil.rmtree(str(staging_date_dir))
    else:
        # Existing dir couldn't be moved; copy staging files into it
        for item in staging_date_dir.iterdir():
            dest = final_date_dir / item.name
            if item.is_file():
                shutil.copy2(str(item), str(dest))
            elif item.is_dir():
                if dest.exists():
                    shutil.rmtree(str(dest), ignore_errors=True)
                shutil.copytree(str(item), str(dest))
        shutil.rmtree(str(staging_date_dir), ignore_errors=True)

    return final_date_dir


# ---------------------------------------------------------------------------
# Phase 7B: Scientific Cartography diagnostic hook
# ---------------------------------------------------------------------------


def _scientific_cartography_output_dir(as_of_date: str) -> Path:
    """Return the date-level Scientific Cartography artifact directory."""
    return REPO_ROOT / "artifacts" / "scientific_cartography" / as_of_date


def run_scientific_cartography_phase13c_export(
    as_of_date: str,
    snapshot_dir: Path,
    ctgov_cache_dir: Path,
    output_dir: Path,
    strict: bool = False,
) -> bool:
    """Run Phase 13C disease map artifact export as optional post-snapshot hook.

    Non-blocking by default: returns False on failure but does not raise.
    With strict=True: raises exception on failure.

    Args:
        as_of_date: snapshot date (YYYY-MM-DD)
        snapshot_dir: path to promoted snapshot directory
        ctgov_cache_dir: path to CTGov cache directory (may not exist)
        output_dir: path where per-disease artifacts will be written
        strict: if True, raise on export failure; if False, log warning and return False

    Returns:
        True if export succeeded, False if export failed (non-strict mode only)
    """
    export_script = REPO_ROOT / "tools" / "run_scientific_cartography_phase13c_export.py"
    if not export_script.exists():
        msg = f"Phase 13C export script not found: {export_script}"
        if strict:
            raise RuntimeError(msg)
        _logger.warning(f"Phase 13C export hook skipped: {msg}")
        return False

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        proc = subprocess.run(
            [
                sys.executable,
                str(export_script),
                "--as-of-date",
                as_of_date,
                "--snapshot-dir",
                str(snapshot_dir),
                "--ctgov-cache",
                str(ctgov_cache_dir),
                "--output-dir",
                str(output_dir),
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if proc.returncode == 0:
            _logger.info(f"Phase 13C disease map export → {output_dir}")
            return True
        else:
            msg = f"Phase 13C export failed (exit {proc.returncode})"
            if strict:
                raise RuntimeError(msg)
            _logger.warning(f"Phase 13C hook: {msg} (non-blocking)")
            return False
    except Exception as e:
        msg = f"Phase 13C export error: {e}"
        if strict:
            raise RuntimeError(msg) from e
        _logger.warning(f"{msg} (non-blocking)")
        return False


def run_scientific_cartography_diagnostics(
    as_of_date: str,
    snapshot_dir: Path,
    ctgov_cache_dir: Path,
    output_dir: Path,
    strict: bool = False,
) -> bool:
    """Run Phase 7A diagnostic wrapper as optional post-snapshot hook.

    Non-blocking by default: returns False on failure but does not raise.
    With strict=True: raises exception on failure.

    Args:
        as_of_date: snapshot date (YYYY-MM-DD)
        snapshot_dir: path to promoted snapshot directory
        ctgov_cache_dir: path to CTGov cache directory (may not exist)
        output_dir: path where diagnostic artifacts will be written
        strict: if True, raise on wrapper failure; if False, log warning and return False

    Returns:
        True if wrapper succeeded, False if wrapper failed (non-strict mode only)
    """
    wrapper_script = REPO_ROOT / "tools" / "run_scientific_cartography_diagnostics.py"
    if not wrapper_script.exists():
        msg = f"Scientific cartography wrapper not found: {wrapper_script}"
        if strict:
            raise RuntimeError(msg)
        _logger.warning(f"Scientific cartography hook skipped: {msg}")
        return False

    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(wrapper_script),
        "--as-of-date",
        as_of_date,
        "--snapshot-dir",
        str(snapshot_dir),
        "--ctgov-cache",
        str(ctgov_cache_dir),
        "--output-dir",
        str(output_dir),
    ]

    if not strict:
        # Non-strict mode: continue even if wrapper fails
        cmd.append("--quiet")

    try:
        proc = _run_subprocess(
            cmd,
            label="scientific_cartography_diagnostics",
            timeout=300,  # 5 minutes for diagnostic generation
        )
        if proc.returncode == 0:
            _logger.info(f"Scientific cartography diagnostics → {output_dir}")
            return True
        else:
            msg = f"Scientific cartography wrapper failed (exit {proc.returncode})"
            if strict:
                raise RuntimeError(msg)
            _logger.warning(f"Scientific cartography hook: {msg} (non-blocking)")
            return False
    except Exception as e:
        msg = f"Scientific cartography hook error: {e}"
        if strict:
            raise RuntimeError(msg) from e
        _logger.warning(f"{msg} (non-blocking)")
        return False


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
    drift_thresholds: Optional[DriftThresholds] = None,
    skip_drift: bool = False,
    skip_forward_eval: bool = False,
    price_cache_dir: Optional[Path] = None,
    fail_on_bad_cache: bool = False,
    skip_pit_warm: bool = False,
    warm_sources: str = "sec_8k,ctgov,sec_13f,fda_adcom,fda_regulatory",
    warm_price_pit: bool = True,
    price_pit_backfill: bool = False,
    auto_refresh_market_data: bool = True,
    allow_candidate: bool = False,
    enrich_pubmed: bool = False,
    run_scientific_cartography: bool = False,
    scientific_cartography_strict: bool = False,
    run_scientific_cartography_phase13c: bool = False,
    scientific_cartography_phase13c_strict: bool = False,
    price_refresh_scope: str = "full",
    mode: str = "daily-production",
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

    _logger.info(f"{'='*70}")
    _logger.info(f"PHASE-2 DAILY RUN — {as_of_date}")
    _logger.info(f"{'='*70}")

    # --- Python-level lock (complements the shell lock in cron_daily_production.sh) ---
    # Prevents concurrent manual + scheduled runs from clobbering artifacts.
    _lock_path = REPO_ROOT / "logs" / ".daily_production_py.lock"
    _lock_path.parent.mkdir(parents=True, exist_ok=True)
    _lock_fd = open(_lock_path, "w")
    try:
        import fcntl

        fcntl.flock(_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (OSError, BlockingIOError):
        _logger.error("Another run_daily_production.py is already running (lock held).")
        _logger.info(f"Lock file: {_lock_path}")
        sys.exit(1)
    except ImportError:
        pass  # fcntl not available on Windows — shell lock is the fallback

    # --- Idempotent rerun check ---
    _final_snap = final_snapshots_dir / as_of_date

    def _emit_self_improvement_artifacts(snapshot_dir: Path, manifest_obj: Dict[str, Any]) -> None:
        """Write governed RSI artifacts without affecting gate outcomes."""
        try:
            from tools.self_improvement_harness import (
                DEFAULT_OUTPUT_DIR,
                week_for_as_of_date,
                write_daily_model_diagnosis,
                write_weekly_remediation_queue_from_dir,
            )

            _paths = write_daily_model_diagnosis(
                snapshot_dir,
                manifest_obj,
                as_of_date,
                DEFAULT_OUTPUT_DIR,
            )
            _logger.info(f"Self-improvement diagnosis → {_paths['markdown']}")
            _week_paths = write_weekly_remediation_queue_from_dir(
                DEFAULT_OUTPUT_DIR,
                week_for_as_of_date(as_of_date),
                DEFAULT_OUTPUT_DIR,
            )
            _logger.info(f"Self-improvement weekly queue → {_week_paths['markdown']}")
        except Exception as exc:
            _logger.warning(f"Could not write self-improvement diagnosis: {exc}")

    if _step_done(_final_snap, "manifest_written"):
        _logger.info(f"\n  Snapshot for {as_of_date} already has a completed manifest.")
        _logger.info("Skipping expensive steps (price, cache, screen, audit, gates).")
        _logger.info(f"To force a full rerun, delete: {_final_snap / _PROGRESS_FILE}")
        _rm_path = _final_snap / "run_manifest.json"
        existing_manifest = json.loads(_rm_path.read_text(encoding="utf-8")) if _rm_path.exists() else None
        if existing_manifest:
            _emit_self_improvement_artifacts(_final_snap, existing_manifest)
            return existing_manifest
        # No manifest found despite progress marker — fall through

    # --- Gate: Ruleset governance (pre-flight, before expensive work) ---
    _manifest_path = REPO_ROOT / "production_data" / "decision_rulesets" / "manifest.json"
    gov_gate = check_ruleset_governance(ruleset_path, _manifest_path, allow_candidate=allow_candidate)
    gate_results.append(gov_gate)
    _logger.info(f"Ruleset governance: {gov_gate.status} — {gov_gate.detail}")
    if gov_gate.status == "FAIL":
        _logger.error("Ruleset governance gate FAIL. Aborting before screen run.")
        manifest = build_run_manifest(
            as_of_date,
            gate_results,
            {},
            subprocess.CompletedProcess(args=[], returncode=-1),
            None,
            config,
            requested_as_of_date=requested_as_of_date,
            git_pre_run=git_pre_run,
            data_dir=data_dir,
        )
        if gov_gate.value:
            manifest["governance_mode"] = gov_gate.value
        _emit_self_improvement_artifacts(_final_snap, manifest)
        return manifest
    # Stamp governance mode for manifest
    _governance_mode = gov_gate.value or "STRICT"

    # --- Gate: Trading day (reject weekends before expensive work) ---
    td_gate = check_trading_day(as_of_date)
    gate_results.append(td_gate)
    _logger.info(f"Trading day: {td_gate.status} — {td_gate.detail}")
    if td_gate.status == "FAIL":
        _logger.error("Not a trading day. Aborting to prevent degenerate data.")
        manifest = build_run_manifest(
            as_of_date,
            gate_results,
            {},
            subprocess.CompletedProcess(args=[], returncode=-1),
            None,
            config,
            requested_as_of_date=requested_as_of_date,
            git_pre_run=git_pre_run,
            data_dir=data_dir,
        )
        _emit_self_improvement_artifacts(_final_snap, manifest)
        return manifest

    # --- Step 1: Price refresh ---
    price_stats: Dict[str, Any] = {}
    if not skip_price_refresh:
        _logger.info("\n[1/5] Refreshing price_history.csv ...")
        universe_path = data_dir / "universe.json"
        # Scoped refresh: only fetch prices for Top-N + XBI (fast daily validation).
        # top60 widens scope to the rank-depth shadow cohort (ranks 1-60) so that
        # ranks 31-60 forward returns are computed on fresh prices. Annotation/
        # validation only — never changes the model, ranker, selector, or sizing.
        _top_n_tickers: Optional[List[str]] = None
        if price_refresh_scope in ("top30", "top60"):
            _scope_n = 60 if price_refresh_scope == "top60" else 30
            _top_n_tickers = _load_top_n_tickers_from_prior_snapshot(final_snapshots_dir, as_of_date, top_n=_scope_n)
            if _top_n_tickers:
                _logger.info(
                    f"  Price refresh scope={price_refresh_scope}: "
                    f"{len(_top_n_tickers)} prior-snapshot tickers + XBI"
                )
            else:
                _logger.info(
                    f"  Price refresh scope={price_refresh_scope}: "
                    "no prior snapshot found — falling back to full universe"
                )
        price_stats = refresh_prices(price_csv, as_of_date, universe_path, tickers_override=_top_n_tickers)
        _logger.info(
            f"  Extended {price_stats.get('n_extended', 0)} tickers, "
            f"{price_stats.get('n_rows_appended', 0)} rows appended, "
            f"{price_stats.get('n_failed', 0)} failures"
        )
        if price_stats.get("failed_tickers"):
            _logger.info(f"Failed: {', '.join(price_stats['failed_tickers'][:10])}")
    else:
        _logger.info("\n[1/5] Price refresh skipped (--skip-price-refresh)")
        price_stats["xbi_last_date"] = _get_ticker_last_date(price_csv, "XBI")

    # --- Post-refresh validation: ensure XBI is fresh (re-fetch if needed) ---
    if not skip_price_refresh:
        _logger.info("[1/5] Validating XBI freshness post-refresh ...")
        xbi_validation = validate_xbi_freshness(price_csv, as_of_date, max_gap_days=config.xbi_stale_days)
        if xbi_validation["revalidation_stats"]:
            price_stats["xbi_revalidation"] = xbi_validation["revalidation_stats"]
        if not xbi_validation["validation_passed"] and xbi_validation["revalidation_needed"]:
            _logger.warning(
                f"XBI validation required targeted re-fetch (gap={xbi_validation.get('trading_days_gap', '?')} days). "
                f"Re-fetch {'successful' if xbi_validation['revalidation_stats'] else 'failed'}."
            )

    # --- Gate: XBI staleness (check early, before expensive screen run) ---
    xbi_gate = check_xbi_staleness(price_csv, as_of_date, config.xbi_stale_days)
    gate_results.append(xbi_gate)
    _logger.info(f"XBI gate: {xbi_gate.status} — {xbi_gate.detail}")
    if xbi_gate.status == "FAIL":
        _logger.error("XBI staleness gate FAIL. Aborting before screen run.")
        manifest = build_run_manifest(
            as_of_date,
            gate_results,
            price_stats,
            subprocess.CompletedProcess(args=[], returncode=-1),
            None,
            config,
            requested_as_of_date=requested_as_of_date,
            git_pre_run=git_pre_run,
            data_dir=data_dir,
        )
        _emit_self_improvement_artifacts(_final_snap, manifest)
        return manifest

    # --- Step 1.4: CTgov daily trial status poll (non-blocking) ---
    # Runs before cache warm to detect trial transitions since last cache.
    # Writes sidecar diff to artifacts/ctgov_daily/ — does not modify cache.
    try:
        from tools.poll_ctgov_daily import poll_ctgov_daily

        _ctg_result = poll_ctgov_daily(as_of_date)
        if "error" in _ctg_result:
            _logger.info(f"CTgov daily poll → skipped ({_ctg_result['error']})")
        else:
            _ctg_n = _ctg_result.get("n_changes", 0)
            _logger.info(f"CTgov daily poll → {_ctg_n} trial changes detected")
    except Exception as _ctg_err:
        _logger.warning(f"CTgov daily poll failed: {_ctg_err}")

    # --- Step 1.5: Pre-warm caches (sec_8k, ctgov, sec_13f) ---
    # Must run BEFORE the ctgov gate so the gate sees the freshly-warmed cache.
    # All three sources are idempotent (short-circuit if cache already exists).
    if not skip_pit_warm and warm_sources:
        _logger.info(f"\n[1.5] Warming caches ({warm_sources}) for {as_of_date} ...")
        _warm_proc = _run_subprocess(
            [
                sys.executable,
                str(REPO_ROOT / "warm_caches.py"),
                "--as-of-date",
                as_of_date,
                "--sources",
                warm_sources,
            ],
            label="warm_caches",
            timeout=1800,  # CTIS enrichment can exceed default 1200s on catch-up runs
        )
        if _warm_proc.returncode == 0:
            _logger.info("Cache warm OK")
        else:
            _logger.info(f"Cache warm FAILED (exit {_warm_proc.returncode}) — dependent gates may WARN")
            if _warm_proc.stderr:
                for _line in _warm_proc.stderr.strip().splitlines()[-5:]:
                    _logger.info(f"  {_line}")
    elif skip_pit_warm:
        _logger.info("\n[1.5] Cache warm skipped (--skip-pit-warm)")

    # --- Gate: ctgov PIT cache availability ---
    _cache_dir = ctgov_cache_dir or (REPO_ROOT / "cache" / "ctgov")
    ctgov_gate, effective_as_of_date = check_ctgov_cache(
        as_of_date,
        _cache_dir,
        allow_fallback=allow_date_fallback,
    )
    gate_results.append(ctgov_gate)
    _logger.info(f"CTGov cache gate: {ctgov_gate.status} — {ctgov_gate.detail}")
    if ctgov_gate.status == "FAIL":
        _logger.error("CTGov PIT cache not found. Aborting before screen run.")
        _logger.info(f"Hint: run warm_caches.py --as-of-date {as_of_date} --sources ctgov")
        manifest = build_run_manifest(
            as_of_date,
            gate_results,
            price_stats,
            subprocess.CompletedProcess(args=[], returncode=-1),
            None,
            config,
            requested_as_of_date=requested_as_of_date,
            git_pre_run=git_pre_run,
            data_dir=data_dir,
        )
        _emit_self_improvement_artifacts(_final_snap, manifest)
        return manifest

    if effective_as_of_date != as_of_date:
        _logger.info(f"Date fallback: {as_of_date} → {effective_as_of_date}")
        as_of_date = effective_as_of_date

    # --- Gate: source freshness (non-blocking; whole-pipeline staleness audit) ---
    freshness_gate = check_source_freshness(
        as_of_date,
        price_csv=price_csv,
        data_dir=data_dir,
        ctgov_cache_dir=ctgov_cache_dir,
    )
    gate_results.append(freshness_gate)
    _logger.info(f"Source freshness gate: {freshness_gate.status} — {freshness_gate.detail}")

    # --- No-op detection (logging-only; never skips work) ---
    try:
        _noop = detect_no_material_input_change(
            as_of_date,
            data_dir=data_dir,
            price_csv=price_csv,
            ctgov_cache_dir=ctgov_cache_dir,
            final_snapshots_dir=final_snapshots_dir,
        )
        if _noop and _noop["no_material_input_change"]:
            _logger.info(
                "NO_MATERIAL_INPUT_CHANGE — model/universe/price/clinical inputs match "
                f"prior snapshot {_noop['compared_against']} (unchanged={_noop['unchanged_inputs']}). "
                "Running normally; rank movement (if any) is not driven by new core inputs."
            )
        elif _noop:
            _logger.info(f"Material input change since {_noop['compared_against']}: changed={_noop['changed_inputs']}")
    except Exception as _noop_err:
        _logger.debug(f"No-op detection skipped: {_noop_err}")

    # --- Gate: required inputs present ---
    inputs_gate = check_inputs_present(data_dir)
    gate_results.append(inputs_gate)
    _logger.info(f"Inputs gate: {inputs_gate.status} — {inputs_gate.detail}")
    if inputs_gate.status == "FAIL":
        _logger.error("Required input files missing. Aborting before screen run.")
        _logger.info(f"Hint: publish an inputs bundle or copy market_data.json to {data_dir}/")
        manifest = build_run_manifest(
            as_of_date,
            gate_results,
            price_stats,
            subprocess.CompletedProcess(args=[], returncode=-1),
            None,
            config,
            requested_as_of_date=requested_as_of_date,
            git_pre_run=git_pre_run,
            data_dir=data_dir,
        )
        _emit_self_improvement_artifacts(_final_snap, manifest)
        return manifest

    # --- Gate: market data schema ---
    schema_gate = check_market_data_schema(data_dir)
    gate_results.append(schema_gate)
    _logger.info(f"Market data schema gate: {schema_gate.status} — {schema_gate.detail}")
    if schema_gate.status == "FAIL":
        _logger.error("Market data schema invalid. Aborting before screen run.")
        manifest = build_run_manifest(
            as_of_date,
            gate_results,
            price_stats,
            subprocess.CompletedProcess(args=[], returncode=-1),
            None,
            config,
            requested_as_of_date=requested_as_of_date,
            git_pre_run=git_pre_run,
            data_dir=data_dir,
        )
        _emit_self_improvement_artifacts(_final_snap, manifest)
        return manifest

    # --- Pre-patch: update market_data.json prices from price_history.csv ---
    # price_history.csv refreshes daily via a different endpoint; market_data.json
    # can get stuck when yfinance rate-limits collect_market_data.py. Patching
    # before the staleness gate keeps close_price in snapshots current even when
    # yfinance falls back to cached data.
    _logger.info("Pre-patching market_data.json prices from price_history.csv ...")
    _patch_proc = _run_subprocess(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "patch_market_data_prices.py"),
            "--as-of-date",
            as_of_date,
        ],
        label="patch_market_data_prices",
    )
    if _patch_proc.returncode != 0:
        _logger.warning("patch_market_data_prices failed (non-blocking) — market_data.json may have stale prices")

    # --- Gate: market data staleness ---
    mkt_gate = check_market_data_staleness(data_dir, as_of_date, config.market_data_max_age_days)
    if mkt_gate.status == "FAIL" and auto_refresh_market_data:
        _logger.info("Market data stale — auto-refreshing ...")
        _mkt_proc = _run_subprocess(
            [
                sys.executable,
                str(REPO_ROOT / "collect_market_data.py"),
                "--universe",
                str(data_dir / "universe.json"),
                "--output",
                str(data_dir / "market_data.json"),
            ],
            label="collect_market_data",
        )
        if _mkt_proc.returncode == 0:
            _logger.info("Market data refresh OK — re-checking staleness gate")
            mkt_gate = check_market_data_staleness(data_dir, as_of_date, config.market_data_max_age_days)
        else:
            _logger.info(f"Market data refresh FAILED (exit {_mkt_proc.returncode})")
            if _mkt_proc.stderr:
                for _line in _mkt_proc.stderr.strip().splitlines()[-3:]:
                    _logger.info(f"  {_line}")
    gate_results.append(mkt_gate)
    _logger.info(f"Market data staleness gate: {mkt_gate.status} — {mkt_gate.detail}")
    if mkt_gate.status == "FAIL":
        _logger.error("Market data too stale. Aborting before screen run.")
        _logger.info(f"Hint: python collect_market_data.py --universe {data_dir}/universe.json")
        manifest = build_run_manifest(
            as_of_date,
            gate_results,
            price_stats,
            subprocess.CompletedProcess(args=[], returncode=-1),
            None,
            config,
            requested_as_of_date=requested_as_of_date,
            git_pre_run=git_pre_run,
            data_dir=data_dir,
        )
        _emit_self_improvement_artifacts(_final_snap, manifest)
        return manifest

    # --- Gate: market data coverage ---
    cov_gate = check_market_data_coverage(data_dir, config.market_data_min_coverage)
    gate_results.append(cov_gate)
    _logger.info(f"Market data coverage gate: {cov_gate.status} — {cov_gate.detail}")
    if cov_gate.status == "FAIL":
        _logger.error("Market data coverage too low. Aborting before screen run.")
        manifest = build_run_manifest(
            as_of_date,
            gate_results,
            price_stats,
            subprocess.CompletedProcess(args=[], returncode=-1),
            None,
            config,
            requested_as_of_date=requested_as_of_date,
            git_pre_run=git_pre_run,
            data_dir=data_dir,
        )
        _emit_self_improvement_artifacts(_final_snap, manifest)
        return manifest

    # --- Step 2: Run screen into staging dir ---
    _logger.info("\n[2/5] Running screen (phase2, ranking_mode=decision) ...")
    staging_dir = Path(tempfile.mkdtemp(prefix=f"phase2_staging_{as_of_date}_"))
    screen_proc = run_screen(
        as_of_date,
        data_dir,
        staging_dir,
        price_csv,
        ruleset_path=ruleset_path,
        extra_args=extra_screen_args,
        prior_snapshot_dir=final_snapshots_dir,
    )
    staging_date_dir = staging_dir / as_of_date

    if screen_proc.returncode not in (0, 1, 2):
        _logger.info(f"Screen FAILED (exit {screen_proc.returncode})")
        if screen_proc.stderr:
            for line in screen_proc.stderr.strip().splitlines()[-10:]:
                _logger.info(f"  {line}")
        gate_results.append(
            GateResult(
                name="screen",
                status="FAIL",
                detail=f"Screen failed (exit {screen_proc.returncode})",
                value=screen_proc.returncode,
            )
        )
        manifest = build_run_manifest(
            as_of_date,
            gate_results,
            price_stats,
            screen_proc,
            None,
            config,
            requested_as_of_date=requested_as_of_date,
            git_pre_run=git_pre_run,
            data_dir=data_dir,
        )
        _emit_self_improvement_artifacts(_final_snap, manifest)
        return manifest

    if screen_proc.returncode == 1:
        # --strict Phase-2 health FAIL (policy gate, not data corruption).
        # If rankings.csv exists, the screen completed — continue with
        # downstream artifacts but record the gate failure.
        _rankings_exists = (staging_date_dir / "rankings.csv").exists()
        if _rankings_exists:
            _logger.info("Screen exited 1 (Phase-2 health FAIL) but snapshot is complete — continuing")
            gate_results.append(
                GateResult(
                    name="phase2_health",
                    status="WARN",
                    detail="Phase-2 health FAIL (exit 1) — snapshot promoted with downstream artifacts",
                    value=screen_proc.returncode,
                )
            )
        else:
            _logger.info("Screen FAILED (exit 1) and no rankings.csv — aborting")
            gate_results.append(
                GateResult(
                    name="screen",
                    status="FAIL",
                    detail="Screen failed (exit 1) with no output",
                    value=screen_proc.returncode,
                )
            )
            # Alert 1: snapshot missing after production run
            try:
                from common.alerts import send_operator_alert

                send_operator_alert(
                    severity="FAIL",
                    system="daily_production",
                    message=(
                        f"Screen exited 1 and rankings.csv is missing for {as_of_date}. "
                        f"Snapshot was NOT promoted. Manual intervention required."
                    ),
                    dedupe_key=f"daily_production:snapshot_missing:{as_of_date}",
                )
            except Exception as _alert_exc:
                _logger.debug("Operator alert skipped: %s", _alert_exc)
            manifest = build_run_manifest(
                as_of_date,
                gate_results,
                price_stats,
                screen_proc,
                None,
                config,
                requested_as_of_date=requested_as_of_date,
                git_pre_run=git_pre_run,
                data_dir=data_dir,
            )
            _emit_self_improvement_artifacts(staging_date_dir, manifest)
            return manifest

    elif screen_proc.returncode == 2:
        _logger.info("Screen completed with WARN (exit 2)")
    else:
        _logger.info("Screen completed OK")

    if not staging_date_dir.exists():
        _logger.info(f"ERROR: Expected snapshot at {staging_date_dir} not found")
        gate_results.append(
            GateResult(
                name="screen",
                status="FAIL",
                detail=f"Snapshot directory not created by screen: {staging_date_dir}",
            )
        )
        manifest = build_run_manifest(
            as_of_date,
            gate_results,
            price_stats,
            screen_proc,
            None,
            config,
            requested_as_of_date=requested_as_of_date,
            git_pre_run=git_pre_run,
            data_dir=data_dir,
        )
        _emit_self_improvement_artifacts(staging_date_dir, manifest)
        return manifest

    # --- Step 2.5: PIT price anchor from staging rankings.csv (before gates) ---
    # Creating the anchor here (not post-promotion) means price_pit_cache gate
    # can see the file and pass on the same run.
    _price_cache_base = price_cache_dir or (REPO_ROOT / "data" / "caches" / "price_pit" / "PIT")
    if not skip_pit_warm and warm_price_pit:
        _staging_rankings = staging_date_dir / "rankings.csv"
        if _staging_rankings.exists():
            _logger.info(f"\n[2.5] Creating PIT price anchor for {as_of_date} ...")
            _anchor_proc = _run_subprocess(
                [
                    sys.executable,
                    str(REPO_ROOT / "tools" / "warm_price_cache.py"),
                    "--snapshot",
                    "--as-of-date",
                    as_of_date,
                    "--rankings-csv",
                    str(_staging_rankings),
                    "--price-csv",
                    str(price_csv),
                    "--cache-base",
                    str(_price_cache_base),
                ],
                label="warm_price_cache_anchor",
            )
            if _anchor_proc.returncode == 0:
                _logger.info("PIT price anchor OK")
            else:
                _logger.info(
                    f"PIT price anchor FAILED (exit {_anchor_proc.returncode}) — price_pit_cache gate will WARN"
                )
                if _anchor_proc.stderr:
                    for _line in _anchor_proc.stderr.strip().splitlines()[-5:]:
                        _logger.info(f"  {_line}")
        else:
            _logger.info("\n[2.5] PIT price anchor skipped — rankings.csv not found in staging")

    # --- Load held tickers for scoped gates (audit + exposure) ---
    _held_tickers: Optional[set] = None
    try:
        from tools.live_shadow_portfolio import load_prior_positions

        _prior = load_prior_positions(as_of_date)
        if _prior:
            _held_tickers = {p["ticker"] for p in _prior[1]}
    except Exception as exc:
        _logger.warning(f"Could not load prior positions for held-scoped gates: {exc}")

    # --- Step 3: Run integrity audit ---
    audit_proc = None
    if not skip_audit:
        _logger.info("\n[3/5] Running data integrity audit ...")
        audit_output_dir = staging_date_dir / "audit"
        audit_proc = run_audit(staging_date_dir, price_csv, as_of_date, audit_output_dir)
        audit_gate = check_audit_result(audit_proc, config, audit_output_dir, held_tickers=_held_tickers)
        gate_results.append(audit_gate)
        _logger.info(f"Audit gate: {audit_gate.status} — {audit_gate.detail}")
    else:
        _logger.info("\n[3/5] Audit skipped (--skip-audit)")

    # --- Step 4: Hard gates ---
    _logger.info("\n[4/5] Evaluating gates ...")

    def _safe_gate(name: str, fn, *args, **kwargs) -> "GateResult":
        """Run a gate check with crash isolation. Returns WARN on exception."""
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            _logger.warning("Gate %s crashed: %s", name, exc)
            return GateResult(name=name, status="WARN", detail=f"Crashed: {exc}")

    missing_gate = _safe_gate(
        "missing_reason", check_missing_reason_fraction, staging_date_dir, config.missing_reason_max_frac
    )
    gate_results.append(missing_gate)
    _logger.info(f"Missing-reason gate: {missing_gate.status} — {missing_gate.detail}")

    turnover_gate = _safe_gate("turnover", check_turnover, staging_date_dir, config.turnover_max_pct)
    gate_results.append(turnover_gate)
    _logger.info(f"Turnover gate: {turnover_gate.status} — {turnover_gate.detail}")

    # --- Gate: drift monitoring (WARN-only) ---
    if not skip_drift:
        _drift_th = drift_thresholds or DriftThresholds()
        drift_gate = _safe_gate(
            "drift_monitoring", check_drift_monitoring, staging_date_dir, final_snapshots_dir, as_of_date, _drift_th
        )
        gate_results.append(drift_gate)
        _logger.info(f"Drift gate: {drift_gate.status} — {drift_gate.detail}")
    else:
        _logger.info("Drift gate: skipped (--skip-drift)")

    # --- Gate: ruleset_health (WARN-only) ---
    rh_gate = _safe_gate("ruleset_health", check_ruleset_health, staging_date_dir)
    gate_results.append(rh_gate)
    _logger.info(f"Ruleset health gate: {rh_gate.status} — {rh_gate.detail}")

    # --- Gate: canary_regression (BLOCK->FAIL, WARN->WARN, INFO->PASS) ---
    canary_gate = _safe_gate("canary_regression", check_canary_regression, staging_date_dir)
    gate_results.append(canary_gate)
    _logger.info(f"Canary regression gate: {canary_gate.status} — {canary_gate.detail}")

    # --- Gate: ctgov PIT dates (WARN-only) ---
    pit_dates_gate = _safe_gate("ctgov_pit_dates", check_ctgov_pit_dates, _cache_dir, as_of_date)
    gate_results.append(pit_dates_gate)
    _logger.info(f"CTGov PIT dates gate: {pit_dates_gate.status} — {pit_dates_gate.detail}")

    # --- Gate: sec_13f_cache (WARN-only) ---
    sec_13f_gate = _safe_gate(
        "sec_13f_cache", check_sec_13f_cache, as_of_date, warn_coverage_pct=config.sec_13f_coverage_warn_pct
    )
    gate_results.append(sec_13f_gate)
    _logger.info(f"13F cache gate: {sec_13f_gate.status} — {sec_13f_gate.detail}")

    # --- Gate: institutional_summary (WARN-only, post-screen) ---
    inst_gate = _safe_gate(
        "institutional_summary",
        check_institutional_summary,
        staging_date_dir,
        warn_coverage_pct=config.institutional_summary_warn_coverage_pct,
    )
    gate_results.append(inst_gate)
    _logger.info(f"Institutional summary gate: {inst_gate.status} — {inst_gate.detail}")

    # --- Gate: institutional_delta (WARN-only, post-screen) ---
    inst_delta_gate = _safe_gate(
        "institutional_delta", check_institutional_delta, staging_date_dir, final_snapshots_dir, as_of_date
    )
    gate_results.append(inst_delta_gate)
    _logger.info(f"Institutional delta gate: {inst_delta_gate.status} — {inst_delta_gate.detail}")

    # --- Gate: pnl_attribution (WARN-only, post-screen) ---
    pnl_gate = _safe_gate(
        "pnl_attribution",
        check_pnl_attribution,
        staging_date_dir,
        final_snapshots_dir,
        as_of_date,
        price_csv,
        min_coverage_pct=config.pnl_attribution_min_coverage_pct,
    )
    gate_results.append(pnl_gate)
    _logger.info(f"PnL attribution gate: {pnl_gate.status} — {pnl_gate.detail}")

    # --- Gate: price_pit_cache (WARN-only) ---
    # _price_cache_base already set in step 2.5 above
    pit_price_gate = _safe_gate("price_pit_cache", check_price_pit_cache, _price_cache_base / as_of_date, as_of_date)
    gate_results.append(pit_price_gate)
    _logger.info(f"PIT price cache gate: {pit_price_gate.status} — {pit_price_gate.detail}")

    # --- Gate: forward_eval (WARN-only) ---
    if not skip_forward_eval:
        fwd_gate = _safe_gate(
            "forward_eval", check_forward_eval, final_snapshots_dir, _price_cache_base, as_of_date, config
        )
        gate_results.append(fwd_gate)
        _logger.info(f"Forward eval gate: {fwd_gate.status} — {fwd_gate.detail}")
    else:
        _logger.info("Forward eval gate: skipped (--skip-forward-eval)")

    # --- Gate: optionality_stability (WARN-only) ---
    try:
        from tools.optionality_stability_monitor import evaluate_optionality_stability

        opt_result = evaluate_optionality_stability(
            as_of_date=as_of_date,
            snapshot_dir=final_snapshots_dir,
            lookback_n=40,
        )
        opt_gate = GateResult(
            name="optionality_stability",
            status=opt_result.gate,
            detail=opt_result.detail,
        )
    except Exception as e:
        opt_gate = GateResult(
            name="optionality_stability",
            status="PASS",
            detail=f"Monitor unavailable: {e}",
        )
    gate_results.append(opt_gate)
    _logger.info(f"Optionality stability gate: {opt_gate.status} — {opt_gate.detail}")

    # --- Gate: pit_bundle_health (WARN-only) ---
    pit_bundle_gate = _safe_gate(
        "pit_bundle_health",
        check_pit_bundle_health,
        as_of_date,
        ctgov_cache_dir=ctgov_cache_dir or (REPO_ROOT / "cache" / "ctgov"),
    )
    gate_results.append(pit_bundle_gate)
    _logger.info(f"PIT bundle health gate: {pit_bundle_gate.status} — {pit_bundle_gate.detail}")

    # --- Gate: decision_engine_schema (WARN-only) ---
    de_schema_gate = _safe_gate("de_schema", check_decision_engine_schema, staging_date_dir)
    gate_results.append(de_schema_gate)
    _logger.info(f"DE schema gate: {de_schema_gate.status} — {de_schema_gate.detail}")

    # --- Gate: sort_contrib_sanity (WARN/FAIL) ---
    sc_gate = _safe_gate("sort_contrib_sanity", check_sort_contrib_sanity, staging_date_dir, config)
    gate_results.append(sc_gate)
    _logger.info(f"Sort contrib sanity gate: {sc_gate.status} — {sc_gate.detail}")

    # --- Gate: portfolio_weights (WARN-only) ---
    pw_gate = _safe_gate(
        "portfolio_weights", check_portfolio_weights, staging_date_dir, tolerance=config.portfolio_weight_sum_tolerance
    )
    gate_results.append(pw_gate)
    _logger.info(f"Portfolio weights gate: {pw_gate.status} — {pw_gate.detail}")

    # --- Gate: eligibility_consistency (WARN-only) ---
    elig_gate = _safe_gate("eligibility_consistency", check_eligibility_consistency, staging_date_dir)
    gate_results.append(elig_gate)
    _logger.info(f"Eligibility consistency gate: {elig_gate.status} — {elig_gate.detail}")

    # --- Gate: cache_health (WARN-only by default; FAIL with --fail-on-bad-cache) ---
    ch_gate = _safe_gate("cache_health", check_cache_health, staging_date_dir, fail_on_bad=fail_on_bad_cache)
    gate_results.append(ch_gate)
    _logger.info(f"Cache health gate: {ch_gate.status} — {ch_gate.detail}")

    # --- Gate: exposure_missingness (WARN/FAIL) ---
    exp_gate = check_exposure_missingness(
        staging_date_dir,
        warn_frac=config.exposure_missing_warn_frac,
        fail_frac=config.exposure_missing_fail_frac,
        held_tickers=_held_tickers,
    )
    gate_results.append(exp_gate)
    _logger.info(f"Exposure missingness gate: {exp_gate.status} — {exp_gate.detail}")

    # --- Gate: risk_concentration (WARN/FAIL) ---
    rc_gate = check_risk_concentration(
        staging_date_dir,
        catalyst_7d_warn=config.risk_conc_catalyst_7d_warn,
        catalyst_7d_fail=config.risk_conc_catalyst_7d_fail,
        high_risk_warn=config.risk_conc_high_beta_warn,
        stacked_warn=config.risk_conc_stacked_warn,
    )
    gate_results.append(rc_gate)
    _logger.info(f"Risk concentration gate: {rc_gate.status} — {rc_gate.detail}")

    # --- Gate: regulatory_calendar (WARN-only) ---
    reg_cal_gate = check_regulatory_calendar(staging_date_dir, as_of_date, config)
    gate_results.append(reg_cal_gate)
    _logger.info(f"Regulatory calendar gate: {reg_cal_gate.status} — {reg_cal_gate.detail}")

    # --- Gate: options_coverage (WARN-only, tracks TT diagnostics health) ---
    opt_cov_gate = check_options_coverage(staging_date_dir)
    gate_results.append(opt_cov_gate)
    _logger.info(f"Options coverage gate: {opt_cov_gate.status} — {opt_cov_gate.detail}")

    # --- Hard-catalyst production gates (Spec 018) ---
    hq_artifacts = check_hard_queue_artifacts(staging_date_dir)
    gate_results.append(hq_artifacts)
    _logger.info(f"Hard queue artifacts: {hq_artifacts.status} — {hq_artifacts.detail}")

    hq_supply = check_hard_catalyst_supply(staging_date_dir, config)
    gate_results.append(hq_supply)
    _logger.info(f"Hard catalyst supply: {hq_supply.status} — {hq_supply.detail}")

    hq_opts = check_hard_options_coverage(staging_date_dir, config)
    gate_results.append(hq_opts)
    _logger.info(f"Hard options coverage: {hq_opts.status} — {hq_opts.detail}")

    hq_carry = check_hard_carry_state(staging_date_dir, as_of_date)
    gate_results.append(hq_carry)
    _logger.info(f"Hard carry state: {hq_carry.status} — {hq_carry.detail}")

    hq_action = check_hard_queue_actionability(staging_date_dir, config)
    gate_results.append(hq_action)
    _logger.info(f"Hard queue actionability: {hq_action.status} — {hq_action.detail}")

    # --- Step 5: Build manifest ---
    _logger.info("\n[5/5] Building run manifest ...")
    git_post_run = get_git_info(REPO_ROOT)
    manifest = build_run_manifest(
        as_of_date,
        gate_results,
        price_stats,
        screen_proc,
        audit_proc,
        config,
        snapshot_date_dir=staging_date_dir,
        requested_as_of_date=requested_as_of_date,
        git_pre_run=git_pre_run,
        git_post_run=git_post_run,
        data_dir=data_dir,
        mode=mode,
        price_csv=price_csv,
        ctgov_cache_dir=ctgov_cache_dir,
    )

    # Stamp governance mode
    manifest["governance_mode"] = _governance_mode

    # Write manifest to staging dir (atomic: temp-file then rename)
    manifest_path = staging_date_dir / "run_manifest.json"
    _fd, _tmp = tempfile.mkstemp(dir=staging_date_dir, prefix=".tmp_manifest_", suffix=".json")
    try:
        with os.fdopen(_fd, "w") as f:
            json.dump(manifest, f, indent=2, default=str)
        Path(_tmp).replace(manifest_path)
    except Exception:
        try:
            os.unlink(_tmp)
        except OSError:
            pass
        raise
    _logger.info(f"Manifest → {manifest_path}")
    _mark_step(staging_date_dir, "manifest_written", detail=f"overall={manifest['overall_status']}")

    # Append to gate verdict ledger (JSONL time-series for SLO tracking)
    try:
        append_gate_verdict(manifest)
        _logger.info(f"Gate ledger → {GATE_LEDGER_PATH}")
    except Exception as e:
        _logger.warning(f"Could not append gate ledger: {e}")

    _emit_self_improvement_artifacts(staging_date_dir, manifest)

    # --- Promotion decision ---
    overall = manifest["overall_status"]
    if overall == "FAIL":
        _logger.info(f"\n{'='*70}")
        _logger.info("RESULT: FAIL — snapshot NOT promoted")
        _logger.info(f"Staging dir preserved at: {staging_date_dir}")
        for g in gate_results:
            if g.status == "FAIL":
                _logger.info(f"[{g.status}] {g.name}: {g.detail}")
        _logger.info(f"{'='*70}")
    else:
        final_path = promote_snapshot(staging_date_dir, final_snapshots_dir, as_of_date)
        _mark_step(final_path, "manifest_written", detail=f"overall={overall}")
        _mark_step(final_path, "promoted", detail=f"from staging to {final_path}")
        # Clean up empty staging parent
        if staging_dir.exists() and not any(staging_dir.iterdir()):
            staging_dir.rmdir()
        label = "PASS" if overall == "PASS" else "WARN"
        _logger.info(f"\n{'='*70}")
        _logger.info(f"RESULT: {label} — snapshot promoted to {final_path}")
        if overall == "WARN":
            for g in gate_results:
                if g.status == "WARN":
                    _logger.info(f"[{g.status}] {g.name}: {g.detail}")
        _logger.info(f"{'='*70}")

        # --- Step 4.5: Scientific Cartography diagnostics (optional, disabled-by-default) ---
        # Runs Phase 7A wrapper to generate diagnostic artifacts if explicitly enabled.
        # Non-blocking by default; failures logged but do not halt pipeline.
        if run_scientific_cartography:
            try:
                _sc_output_dir = _scientific_cartography_output_dir(as_of_date)
                _sc_ctgov_cache = ctgov_cache_dir or (REPO_ROOT / "cache" / "ctgov" / as_of_date)
                run_scientific_cartography_diagnostics(
                    as_of_date=as_of_date,
                    snapshot_dir=final_path,
                    ctgov_cache_dir=_sc_ctgov_cache,
                    output_dir=_sc_output_dir,
                    strict=scientific_cartography_strict,
                )
            except Exception as _sc_err:
                if scientific_cartography_strict:
                    _logger.error(f"Scientific cartography strict mode failure: {_sc_err}")
                    raise
                else:
                    _logger.warning(f"Scientific cartography hook error (non-blocking): {_sc_err}")

        # --- Step 4.6: Phase 13C Disease Map Artifact Export (optional, disabled-by-default) ---
        # Runs Phase 13C export to generate per-disease artifacts if explicitly enabled.
        # Non-blocking by default; failures logged but do not halt pipeline.
        if run_scientific_cartography_phase13c:
            try:
                _sc13c_output_dir = _scientific_cartography_output_dir(as_of_date)
                _sc_ctgov_cache = ctgov_cache_dir or (REPO_ROOT / "cache" / "ctgov" / as_of_date)
                run_scientific_cartography_phase13c_export(
                    as_of_date=as_of_date,
                    snapshot_dir=final_path,
                    ctgov_cache_dir=_sc_ctgov_cache,
                    output_dir=_sc13c_output_dir,
                    strict=scientific_cartography_phase13c_strict,
                )
            except Exception as _sc13c_err:
                if scientific_cartography_phase13c_strict:
                    _logger.error(f"Phase 13C strict mode failure: {_sc13c_err}")
                    raise
                else:
                    _logger.warning(f"Phase 13C hook error (non-blocking): {_sc13c_err}")

        # --- Step 5a: Update forward_eval IC ledger (Path C monitoring) ---
        # Extracts mean_ic from forward_eval gate results and appends to
        # forward_eval_ic_ledger.jsonl for trend monitoring through 2026-06-03.
        # Non-blocking; failures logged but do not halt pipeline.
        try:
            from tools.forward_eval_ic_ledger import append_to_ledger, extract_forward_eval_ic

            _ic_ledger = REPO_ROOT / "artifacts" / "forward_eval_ic_ledger.jsonl"
            _existing_dates = set()
            if _ic_ledger.exists():
                with open(_ic_ledger) as f:
                    for line in f:
                        if line.strip():
                            try:
                                _entry = json.loads(line)
                                _existing_dates.add(_entry["as_of_date"])
                            except json.JSONDecodeError:
                                pass

            if as_of_date not in _existing_dates:
                _ic_data = extract_forward_eval_ic(
                    snapshot_dir=final_path,
                    as_of_date=as_of_date,
                    price_cache_base=_price_cache_base,
                    horizon=config.forward_eval_horizon,
                    lookback_n=config.forward_eval_lookback_n,
                )
                if _ic_data:
                    append_to_ledger(_ic_ledger, _ic_data)
                    _logger.info(f"Forward eval IC ledger: {as_of_date} → mean_ic={_ic_data['mean_ic']:.4f}")
        except Exception as _ic_err:
            _logger.warning(f"Forward eval IC ledger update failed (non-fatal): {_ic_err}")

        # --- Step 5a.1: Archive PIT-sensitive inputs alongside snapshot ---
        # Copies key input files so future PIT v2 regeneration and backtests
        # can use historical inputs instead of current production_data/.
        _PIT_INPUT_FILES = [
            "universe.json",
            "trial_records.json",
            "holdings_detailed.json",
            "short_interest.json",
            "ipo_dates.json",
        ]
        try:
            _inputs_dir = final_path / "inputs"
            _inputs_dir.mkdir(exist_ok=True)
            _archived = 0
            for _inp_name in _PIT_INPUT_FILES:
                _src = data_dir / _inp_name
                _dst = _inputs_dir / _inp_name
                if _src.exists() and not _dst.exists():
                    shutil.copy2(str(_src), str(_dst))
                    _archived += 1
            if _archived:
                _logger.info(f"PIT input archive: {_archived} files → {_inputs_dir}")
            else:
                _logger.info("PIT input archive: all files already present (idempotent)")
        except Exception as _pit_err:
            _logger.warning(f"PIT input archive failed (non-fatal): {_pit_err}")

        # --- Step 5b: Full drift report (optional, post-promotion) ---
        if not skip_drift:
            _drift_report_script = REPO_ROOT / "scripts" / "run_drift_report.py"
            if _drift_report_script.exists():
                _drift_out = final_path / "drift_guardrails"
                _drift_proc = _run_subprocess(
                    [
                        sys.executable,
                        str(_drift_report_script),
                        "--snapshot-dir",
                        str(final_snapshots_dir),
                        "--output-dir",
                        str(_drift_out),
                        "--window-size",
                        "5",
                    ],
                    label="run_drift_report",
                    timeout=120,
                )
                if _drift_proc.returncode == 0:
                    _logger.info(f"Full drift report → {_drift_out}")
                else:
                    _logger.warning(f"Full drift report failed (exit {_drift_proc.returncode})")

        # --- Step 5c: Action packet (non-blocking) ---
        try:
            from tools.action_packet import write_action_packet

            _action_path = write_action_packet(final_path)
            _logger.info(f"Action packet → {_action_path.parent}")
        except Exception as _ap_err:
            _logger.warning(f"Action packet generation failed: {_ap_err}")

        # --- Step 5d: Action lists — per-bucket CSVs (non-blocking) ---
        try:
            from tools.build_action_lists import build_action_lists, write_action_lists

            _al_buckets = build_action_lists(final_path)
            _al_out = final_path / "action_lists"
            write_action_lists(_al_buckets, _al_out, as_of_date=as_of_date)
            _al_total = sum(len(v) for v in _al_buckets.values())
            _logger.info(f"Action lists → {_al_out} ({_al_total} names)")
        except Exception as _al_err:
            _logger.warning(f"Action list generation failed: {_al_err}")

        # --- Step 5e: Export action lists to output/ (non-blocking) ---
        try:
            from tools.export_action_lists import export_action_lists

            _exp_out = REPO_ROOT / "output" / "action_lists"
            export_action_lists(final_path, _exp_out, as_of_date)
            _logger.info(f"Exported action lists → {_exp_out}")
        except Exception as _exp_err:
            _logger.warning(f"Action list export failed: {_exp_err}")

        # --- Step 5f: Live shadow portfolio (non-blocking) ---
        try:
            from tools.live_shadow_portfolio import run_shadow_portfolio

            _shadow_result = run_shadow_portfolio(final_path)
            _shadow_n = _shadow_result["summary"]["total_positions"]
            _logger.info(f"Shadow portfolio → {_shadow_result['positions_path']} ({_shadow_n} names)")
        except Exception as _sp_err:
            _logger.warning(f"Shadow portfolio generation failed: {_sp_err}")

        # --- Step 5g: Weekly trade packet (rebalance day only, non-blocking) ---
        try:
            from tools.run_weekly_rebalance import run_weekly_rebalance

            _reb_result = run_weekly_rebalance(as_of_date)
            _reb_decision = _reb_result["decision"]
            _reb_n = _reb_result.get("n_trades", 0)
            if _reb_decision in ("REBALANCE", "OFF_CYCLE"):
                _logger.info(f"Trade packet → {_reb_result.get('csv_path', '?')} ({_reb_n} trades)")
            else:
                _logger.info(f"Trade packet → skipped ({_reb_decision})")
        except Exception as _reb_err:
            _logger.warning(f"Weekly rebalance check failed: {_reb_err}")

        # --- Step 5h: Portfolio alerts (non-blocking) ---
        try:
            from tools.portfolio_alerts import run_portfolio_alerts

            _alert_result = run_portfolio_alerts(as_of_date)
            _n_alerts = _alert_result["alert_count"]
            if _n_alerts > 0:
                _logger.info(f"Portfolio alerts → {_alert_result['alert_path']} ({_n_alerts} alerts)")
            else:
                _logger.info("Portfolio alerts → none")
        except Exception as _alert_err:
            _logger.warning(f"Portfolio alerts failed: {_alert_err}")

        # --- Step 5i: Trade plan (daily, non-blocking) ---
        try:
            from tools.build_trade_plan import build_trade_plan

            _tp_result = build_trade_plan(as_of_date, snap_dir=final_path)
            if "error" in _tp_result:
                _logger.info(f"Trade plan → skipped ({_tp_result['error']})")
            else:
                _tp_n = len(_tp_result.get("trades", []))
                _tp_path = _tp_result.get("csv_path", "?")
                _logger.info(f"Trade plan → {_tp_path} ({_tp_n} trades)")
        except Exception as _tp_err:
            _logger.warning(f"Trade plan failed: {_tp_err}")

        # --- Step 5i.5: Surface delta monitor (non-blocking) ---
        try:
            from tools.surface_delta_monitor import run as run_surface_delta

            _sdm_result = run_surface_delta(
                as_of_date=as_of_date,
                live=False,
                json_only=True,
            )
            _sdm_n_alert = _sdm_result.get("n_alert", 0)
            _sdm_n_watch = _sdm_result.get("n_watch", 0)
            _logger.info(
                f"  Surface delta → {_sdm_result.get('n_compared', 0)} compared, "
                f"{_sdm_n_alert} alert / {_sdm_n_watch} watch"
            )
        except Exception as _sdm_err:
            _logger.warning(f"Surface delta monitor failed: {_sdm_err}")

        # --- Step 5i.6: Catalyst delta (non-blocking) ---
        try:
            from tools.build_catalyst_delta import build_catalyst_delta

            _cd_result = build_catalyst_delta(
                as_of_date,
                snapshots_dir=final_snapshots_dir,
            )
            if "error" in _cd_result:
                _logger.info(f"Catalyst delta → skipped ({_cd_result['error']})")
            else:
                _cd_n = _cd_result.get("n_filtered", 0)
                _logger.info(f"Catalyst delta → {_cd_n} changes surfaced")
        except Exception as _cd_err:
            _logger.warning(f"Catalyst delta failed: {_cd_err}")

        # --- Step 5j: Portfolio metrics update (non-blocking) ---
        try:
            from tools.build_portfolio_report import build_portfolio_report

            _pr_result = build_portfolio_report()
            _logger.info(f"Portfolio report → {_pr_result['report_path']}")
        except Exception as _pr_err:
            _logger.warning(f"Portfolio report failed: {_pr_err}")

        # --- Step 5k: Readiness scorecard (daily history, non-blocking) ---
        try:
            from tools.weekly_readiness_scorecard import append_history, build_scorecard, format_scorecard_md

            _pinned_id = ""
            try:
                from run_screen import PHASE2_PINNED_RULESET_ID

                _pinned_id = PHASE2_PINNED_RULESET_ID
            except ImportError:
                pass
            _sc_result = build_scorecard(
                as_of_date,
                snapshots_dir=final_snapshots_dir,
                artifacts_dir=REPO_ROOT / "artifacts" / "live_shadow",
                policy_path=REPO_ROOT / "production_data" / "portfolio_policy.json",
                ruleset_id=_pinned_id,
            )
            _sc_verdict = _sc_result.get("verdict", "?")
            # Write scorecard artifacts
            _sc_out_dir = REPO_ROOT / "artifacts" / "readiness"
            _sc_out_dir.mkdir(parents=True, exist_ok=True)
            _sc_md = format_scorecard_md(_sc_result)
            (_sc_out_dir / f"scorecard_{as_of_date}.md").write_text(_sc_md, encoding="utf-8")
            with open(_sc_out_dir / f"scorecard_{as_of_date}.json", "w", encoding="utf-8") as _scf:
                json.dump(_sc_result, _scf, indent=2, default=str)
            append_history(_sc_out_dir / "history.jsonl", _sc_result)
            _logger.info(f"Readiness scorecard → {_sc_verdict}")
        except Exception as _sc_err:
            _logger.warning(f"Readiness scorecard failed: {_sc_err}")

        # --- Step 5k.5a: Options watch (post-packet, non-blocking) ---
        try:
            from tools.build_options_watch import build_options_watch

            _ow_result = build_options_watch(
                as_of_date,
                mode="post_packet",
                snapshots_dir=final_snapshots_dir,
            )
            if "error" in _ow_result:
                _logger.info(f"Options watch → skipped ({_ow_result['error']})")
            else:
                _ow_n = _ow_result.get("watchlist_size", 0)
                _ow_flagged = _ow_result.get("n_flagged", 0)
                _logger.info(f"Options watch → {_ow_n} names, {_ow_flagged} flagged")
        except Exception as _ow_err:
            _logger.warning(f"Options watch failed: {_ow_err}")

        # --- Step 5k.5a-shadow: Options watch pre-open shadow (non-blocking) ---
        try:
            _ow_pre = build_options_watch(
                as_of_date,
                mode="pre_open",
                snapshots_dir=final_snapshots_dir,
            )
            if "error" not in _ow_pre:
                _ow_pre_n = _ow_pre.get("watchlist_size", 0)
                _ow_pre_flagged = _ow_pre.get("n_flagged", 0)
                _logger.info(f"Options watch (pre-open shadow) → {_ow_pre_n} names, {_ow_pre_flagged} flagged")
        except Exception as _ow_pre_err:
            _logger.warning(f"Options watch pre-open shadow failed: {_ow_pre_err}")

        # --- Step 5k.5b: Options chartbook (non-blocking) ---
        try:
            from tools.build_options_chartbook import build_chartbook

            _cb_result = build_chartbook(as_of_date, snapshots_dir=final_snapshots_dir)
            if "error" in _cb_result:
                _logger.info(f"Options chartbook → skipped ({_cb_result['error']})")
            else:
                _cb_n = _cb_result.get("scoreboard", {}).get("watchlist_size", 0)
                _cb_path = _cb_result.get("_html_path", "?")
                _logger.info(f"Options chartbook → {_cb_path} ({_cb_n} names)")
        except Exception as _cb_err:
            _logger.warning(f"Options chartbook failed: {_cb_err}")

        # --- Step 5k.5c: Price action watch (non-blocking) ---
        try:
            from tools.build_price_action_watch import build_price_action_watch

            _paw_result = build_price_action_watch(as_of_date, snapshots_dir=final_snapshots_dir)
            if "error" in _paw_result:
                _logger.info(f"Price action watch → skipped ({_paw_result['error']})")
            else:
                _paw_n = _paw_result.get("n_alerted", 0)
                _logger.info(f"Price action watch → {_paw_result['watchlist_size']} names, {_paw_n} alerted")
        except Exception as _paw_err:
            _logger.warning(f"Price action watch failed: {_paw_err}")

        # --- Step 5k.5d: Options verdict — fused multi-lens (non-blocking) ---
        try:
            from tools.build_options_verdict import build_options_verdict

            _ov_result = build_options_verdict(
                as_of_date,
                snapshots_dir=final_snapshots_dir,
            )
            _ov_n = _ov_result.get("n_tickers", 0)
            _ov_h = _ov_result.get("n_high", 0)
            _ov_new = _ov_result.get("n_new", 0)
            _logger.info(f"Options verdict → {_ov_n} active (H={_ov_h}), {_ov_new} new")
        except Exception as _ov_err:
            _logger.warning(f"Options verdict failed: {_ov_err}")

        # --- Step 5k.5e: Options Monitor v1.1 verdict artifact (non-blocking) ---
        try:
            from tools.build_options_verdict_v11 import build_verdict_v11

            _ov11_result = build_verdict_v11(
                as_of_date,
                snapshots_dir=final_snapshots_dir,
            )
            if "error" in _ov11_result:
                _logger.info(f"Options v1.1 → skipped ({_ov11_result['error']})")
            else:
                _ov11_n = _ov11_result.get("n_active", 0)
                _ov11_h = _ov11_result.get("n_high", 0)
                _ov11_new = _ov11_result.get("n_new", 0)
                _logger.info(f"Options v1.1 → {_ov11_n} active (H={_ov11_h}), {_ov11_new} new")
        except Exception as _ov11_err:
            _logger.warning(f"Options v1.1 failed: {_ov11_err}")

        # --- Step 5k.5f: Options shadow layer (TT-only; RH cache unavailable in cron) ---
        try:
            from tools.collect_options_shadow import run_options_shadow

            _os_result = run_options_shadow(
                as_of_date,
                tickers=None,  # auto-resolved from latest rankings.csv (top-30)
                rh_cache_file=None,  # no RH MCP in cron; session writes cache separately
                snapshots_dir=final_snapshots_dir,
            )
            if "error" in _os_result:
                _logger.info(f"Options shadow → skipped ({_os_result['error']})")
            else:
                _os_n = _os_result.get("n_tickers", 0)
                _os_ep = _os_result.get("n_event_premium", 0)
                _os_hc = _os_result.get("n_high_confidence", 0)
                _logger.info(f"Options shadow → {_os_n} tickers, {_os_hc} HIGH_CONFIDENCE, {_os_ep} event-premium")
        except Exception as _os_err:
            _logger.warning(f"Options shadow failed: {_os_err}")

        # --- Step 5k.5g: EES v3 daily shadow card (non-blocking) ---
        try:
            from tools.build_ees_shadow_card import run_ees_shadow_card

            _ees_card = run_ees_shadow_card(as_of_date)
            if "error" in _ees_card:
                _logger.info(f"EES shadow card → skipped ({_ees_card['error']})")
            else:
                _ees_gate = _ees_card.get("gate_20d_met", False)
                _ees_done = _ees_card.get("windows_completed", 0)
                _ees_req = _ees_card.get("gate_20d_required", 20)
                _ees_fires = _ees_card.get("n_veto_fires", 0)
                _logger.info(
                    f"EES shadow card → gate {'MET' if _ees_gate else f'UNMET ({_ees_done}/{_ees_req})'}"
                    f", {_ees_fires} shadow veto fires"
                )
        except Exception as _ees_err:
            _logger.warning(f"EES shadow card failed: {_ees_err}")

        # --- Step 5k.6: Shadow monitor (non-blocking) ---
        try:
            from tools.build_shadow_monitor import build_shadow_monitor

            _sm_result = build_shadow_monitor(as_of_date)
            if "error" in _sm_result:
                _logger.info(f"Shadow monitor → skipped ({_sm_result['error']})")
            else:
                _sm_attn = _sm_result.get("attention", "?")
                _sm_n_alerts = len(_sm_result.get("alerts", []))
                _logger.info(f"Shadow monitor → {_sm_attn} ({_sm_n_alerts} alerts)")
        except Exception as _sm_err:
            _logger.warning(f"Shadow monitor failed: {_sm_err}")

        # --- Step 5k.8: Competitive intelligence (non-blocking) ---
        try:
            from tools.build_competitive_intel import build_competitive_intel

            _ci_result = build_competitive_intel(as_of_date, snapshots_dir=final_snapshots_dir)
            if "error" in _ci_result:
                _logger.info(f"Competitive intel → skipped ({_ci_result['error']})")
            else:
                _logger.info(f"Competitive intel → {_ci_result.get('n_competitive_events', 0)} events")
        except Exception as _ci_err:
            _logger.warning(f"Competitive intel failed: {_ci_err}")

        # --- Step 5k.9: Regulatory watch (non-blocking) ---
        try:
            from tools.build_regulatory_watch import build_regulatory_watch

            _rw_result = build_regulatory_watch(as_of_date, snapshots_dir=final_snapshots_dir)
            if "error" in _rw_result:
                _logger.info(f"Regulatory watch → skipped ({_rw_result['error']})")
            else:
                _logger.info(f"Regulatory watch → {_rw_result.get('n_near_term_90d', 0)} near-term")
        except Exception as _rw_err:
            _logger.warning(f"Regulatory watch failed: {_rw_err}")

        # --- Step 5k.10: Filing watch (non-blocking) ---
        try:
            from tools.build_filing_watch import build_filing_watch

            _fw_result = build_filing_watch(as_of_date, snapshots_dir=final_snapshots_dir)
            if "error" in _fw_result:
                _logger.info(f"Filing watch → skipped ({_fw_result['error']})")
            else:
                _fw_dil = len(_fw_result.get("dilution_alerts", []))
                _logger.info(f"Filing watch → {_fw_result.get('n_relevant', 0)} relevant, {_fw_dil} dilution alerts")
        except Exception as _fw_err:
            _logger.warning(f"Filing watch failed: {_fw_err}")

        # --- Step 5k.7: IC dashboard (non-blocking) ---
        try:
            from tools.build_ic_dashboard import build_ic_dashboard

            _ic_result = build_ic_dashboard(as_of_date, lookback=60)
            if "error" in _ic_result:
                _logger.info(f"IC dashboard → skipped ({_ic_result['error']})")
            else:
                _ic_attn = _ic_result.get("attention", "?")
                _logger.info(f"IC dashboard → {_ic_attn}")
        except Exception as _ic_err:
            _logger.warning(f"IC dashboard failed: {_ic_err}")

        # --- Step 5k.11: Post-promotion monitor (non-blocking) ---
        try:
            from tools.post_promotion_monitor import compute_monitor

            _pm_result = compute_monitor(as_of_date)
            _pm_dir = REPO_ROOT / "artifacts" / "post_promotion_monitor"
            _pm_dir.mkdir(parents=True, exist_ok=True)
            _pm_path = _pm_dir / f"{as_of_date}_monitor.json"
            with open(_pm_path, "w") as _pm_f:
                json.dump(_pm_result, _pm_f, indent=2)
            _pm_day = _pm_result.get("days_since_promotion", "?")
            _pm_alerts = len(_pm_result.get("alerts", []))
            _logger.info(f"Post-promotion monitor → day {_pm_day}, {_pm_alerts} alerts")
        except Exception as _pm_err:
            _logger.warning(f"Post-promotion monitor failed: {_pm_err}")

        # --- Step 5k.11b: Coinvest anchor shadow (non-blocking, 30-day validation) ---
        try:
            from tools.coinvest_shadow_tracker import compute_shadow

            _cs_result = compute_shadow(as_of_date)
            if "error" not in _cs_result:
                _cs_dir = REPO_ROOT / "artifacts" / "coinvest_shadow"
                _cs_dir.mkdir(parents=True, exist_ok=True)
                _cs_path = _cs_dir / f"{as_of_date}.json"
                with open(_cs_path, "w") as _cs_f:
                    json.dump(_cs_result, _cs_f, indent=2, default=str)
                from tools.coinvest_shadow_tracker import append_history, write_summary

                append_history(_cs_result)
                write_summary()
                _cs_day = _cs_result.get("days_since_start", "?")
                _cs_ci = _cs_result.get("strategies", {}).get("coinvest_inst", {}).get("overlap_pct", "?")
                _logger.info(f"Coinvest shadow → day {_cs_day}, CI overlap={_cs_ci}%")
            else:
                _logger.info(f"Coinvest shadow → skipped ({_cs_result.get('error', '?')})")
        except Exception as _cs_err:
            _logger.warning(f"Coinvest shadow failed: {_cs_err}")

        # --- Step 5k.11c: Regime shadow (non-blocking, diagnostic) ---
        try:
            from tools.run_regime_shadow import run_regime_shadow

            _rs_result = run_regime_shadow(as_of_date)
            _rs_dir = REPO_ROOT / "artifacts" / "regime_shadow"
            _rs_dir.mkdir(parents=True, exist_ok=True)
            _rs_path = _rs_dir / f"{as_of_date}.json"
            with open(_rs_path, "w") as _rs_f:
                json.dump(_rs_result, _rs_f, indent=2, default=str)
            _rs_simple = _rs_result.get("simple_classifier", {}).get("regime", "?")
            _rs_rich = _rs_result.get("rich_classifier", {}).get("regime", "?")
            _rs_agree = _rs_result.get("agreement", False)
            _logger.info(f"Regime shadow → simple={_rs_simple} rich={_rs_rich} agree={_rs_agree}")
        except Exception as _rs_err:
            _logger.warning(f"Regime shadow failed: {_rs_err}")

        # --- Step 5k.11d: Regime evaluation update (non-blocking, diagnostic) ---
        try:
            import json as _json_re

            from scripts.research.regime_evaluation import run_evaluation

            _re_result = run_evaluation(horizon=5, since=None, live_only=False)
            _re_dir = REPO_ROOT / "artifacts" / "regime_evaluation"
            _re_dir.mkdir(parents=True, exist_ok=True)
            _re_path = _re_dir / "regime_eval_h5.json"
            with open(_re_path, "w") as _re_f:
                _json_re.dump(_re_result, _re_f, indent=2, default=str)
            _re_delta = _re_result.get("switching", {}).get("delta", "?")
            _re_n = _re_result.get("n_observations", 0)
            _logger.info(f"Regime eval → n={_re_n}, switching delta={_re_delta}%")
        except Exception as _re_err:
            _logger.warning(f"Regime evaluation failed: {_re_err}")

        # --- Step 5k.12: Asymmetry score (non-blocking, accumulates EPD history) ---
        try:
            from scripts.research.top30_asymmetry_score import score_snapshot

            _as_result = score_snapshot(as_of_date, top_n=30)
            if "error" not in _as_result:
                _as_dir = REPO_ROOT / "output" / "ranker_eval"
                _as_dir.mkdir(parents=True, exist_ok=True)
                _as_path = _as_dir / f"asymmetry_score_{as_of_date}.json"
                with open(_as_path, "w") as _as_f:
                    json.dump(_as_result, _as_f, indent=2, default=str)
                _as_n = _as_result.get("n_scored", 0)
                _logger.info(f"Asymmetry score → {_as_n} names scored")
            else:
                _logger.info(f"Asymmetry score → skipped ({_as_result['error']})")
        except Exception as _as_err:
            _logger.warning(f"Asymmetry score failed: {_as_err}")

        # --- Step 5k.13: AACT trial deltas (non-blocking) ---
        try:
            from tools.build_aact_trial_deltas import build_deltas

            _aact_result = build_deltas(as_of_date)
            if "error" not in _aact_result:
                _aact_dir = REPO_ROOT / "artifacts" / "aact_deltas"
                _aact_dir.mkdir(parents=True, exist_ok=True)
                _aact_path = _aact_dir / f"aact_deltas_{as_of_date}.json"
                with open(_aact_path, "w") as _aact_f:
                    json.dump(_aact_result, _aact_f, indent=2, default=str)
                _aact_n = _aact_result.get("n_with_activity", 0)
                _logger.info(f"AACT deltas → {_aact_n} tickers with activity")
            else:
                _logger.info(f"AACT deltas → skipped ({_aact_result['error']})")
        except Exception as _aact_err:
            _logger.warning(f"AACT deltas failed: {_aact_err}")

        # --- Step 5k.14: Rebalance plan (non-blocking) ---
        try:
            from tools.build_rebalance_plan import build_plan

            _rp_result = build_plan(as_of_date)
            if "error" not in _rp_result:
                _rp_dir = REPO_ROOT / "artifacts" / "rebalance_plan"
                _rp_dir.mkdir(parents=True, exist_ok=True)
                _rp_path = _rp_dir / f"{as_of_date}_plan.json"
                with open(_rp_path, "w") as _rp_f:
                    json.dump(_rp_result, _rp_f, indent=2, default=str)
                _rp_skip = "SKIP" if _rp_result.get("skip_rebalance") else "EXECUTE"
                _rp_buys = _rp_result.get("n_buys", 0)
                _logger.info(f"Rebalance plan → {_rp_skip}, {_rp_buys} buys")
                # Write standalone risk_layer artifact (Spec 052)
                _rl_data = _rp_result.get("risk_layer")
                if _rl_data:
                    _rl_dir = REPO_ROOT / "artifacts" / "risk_layer"
                    _rl_dir.mkdir(parents=True, exist_ok=True)
                    _rl_path = _rl_dir / f"{as_of_date}.json"
                    with open(_rl_path, "w") as _rl_f:
                        json.dump(_rl_data, _rl_f, indent=2, default=str)
                    _rl_nb = _rl_data.get("n_breaches", 0)
                    _logger.info(f"Risk layer artifact → {_rl_nb} breaches")
            else:
                _logger.info(f"Rebalance plan → skipped ({_rp_result['error']})")
        except Exception as _rp_err:
            _logger.warning(f"Rebalance plan failed: {_rp_err}")

        # --- Step 5k.15: Risk monitor (non-blocking) ---
        try:
            from tools.build_risk_monitor import build_risk_report

            _rm_result = build_risk_report(as_of_date)
            if "error" not in _rm_result:
                _rm_dir = REPO_ROOT / "artifacts" / "risk_monitor"
                _rm_dir.mkdir(parents=True, exist_ok=True)
                _rm_path = _rm_dir / f"{as_of_date}_risk.json"
                with open(_rm_path, "w") as _rm_f:
                    json.dump(_rm_result, _rm_f, indent=2, default=str)
                _rm_level = _rm_result.get("risk_level", "?")
                _logger.info(f"Risk monitor → {_rm_level}")
            else:
                _logger.info(f"Risk monitor → skipped ({_rm_result['error']})")
        except Exception as _rm_err:
            _logger.warning(f"Risk monitor failed: {_rm_err}")

        # --- Step 5k.16: Regime pruner recommendation (non-blocking) ---
        try:
            from tools.build_regime_pruner_recommendation import build_recommendation

            _rpr_result = build_recommendation(as_of_date)
            _rpr_dir = REPO_ROOT / "artifacts" / "regime_pruner"
            _rpr_dir.mkdir(parents=True, exist_ok=True)
            _rpr_path = _rpr_dir / f"{as_of_date}_recommendation.json"
            with open(_rpr_path, "w") as _rpr_f:
                json.dump(_rpr_result, _rpr_f, indent=2, default=str)
            _rpr_rec = _rpr_result.get("recommendation", "?")
            _rpr_regime = _rpr_result.get("regime", "?")
            _rpr_override = " [RISK OVERRIDE]" if _rpr_result.get("risk_override") else ""
            _logger.info(f"Regime pruner → {_rpr_rec} ({_rpr_regime}){_rpr_override}")
        except Exception as _rpr_err:
            _logger.warning(f"Regime pruner failed: {_rpr_err}")

        # --- Step 5k.17: Event quality shadow sizer + review priority (non-blocking, Spec 056/058+) ---
        try:
            from tools.event_quality_shadow_sizer import prioritize_reviews
            from tools.event_quality_shadow_sizer import run_shadow as _eq_run_shadow

            _eq_result = _eq_run_shadow(as_of_date)
            if "error" not in _eq_result:
                _eq_dir = REPO_ROOT / "artifacts" / "event_quality_shadow"
                _eq_dir.mkdir(parents=True, exist_ok=True)
                _eq_path = _eq_dir / f"event_quality_shadow_{as_of_date}.json"
                with open(_eq_path, "w") as _eq_f:
                    json.dump(_eq_result, _eq_f, indent=2, default=str)
                _eq_up = _eq_result.get("n_upweighted", 0)
                _eq_down = _eq_result.get("n_downweighted", 0)
                _logger.info(f"Event quality shadow → {_eq_up} up, {_eq_down} down")
            else:
                _logger.info(f"Event quality shadow → skipped ({_eq_result.get('error', '?')})")
            # Review prioritization
            _rp_result = prioritize_reviews(as_of_date)
            if "error" not in _rp_result:
                _rp_dir = REPO_ROOT / "artifacts" / "review"
                _rp_dir.mkdir(parents=True, exist_ok=True)
                _rp_path = _rp_dir / f"review_priority_{as_of_date}.json"
                with open(_rp_path, "w") as _rp_f:
                    json.dump(_rp_result, _rp_f, indent=2, default=str)
                _rp_n = _rp_result.get("n_reviewed", 0)
                _logger.info(f"Review priority → {_rp_n} flagged")
        except Exception as _eq_err:
            _logger.warning(f"Event quality shadow failed: {_eq_err}")

        # --- Step 5k.18: Timing hazard overlay + calibration dashboard (non-blocking, Spec 058+) ---
        try:
            from tools.compute_timing_hazard import (
                append_calibration_ledger,
                build_calibration_dashboard,
                compute_calibration_by_slice,
                compute_timing_hazard,
                emit_calibration_cycle_summary,
            )

            _th_result = compute_timing_hazard(as_of_date)
            if "error" not in _th_result:
                _th_dir = REPO_ROOT / "artifacts" / "timing_hazard"
                _th_dir.mkdir(parents=True, exist_ok=True)
                _th_path = _th_dir / f"timing_hazard_{as_of_date}.json"
                with open(_th_path, "w") as _th_f:
                    json.dump(_th_result, _th_f, indent=2, default=str)
                append_calibration_ledger(_th_result)
                # Calibration-by-slice
                _cal_slices = compute_calibration_by_slice(as_of_date)
                if _cal_slices["n_resolved"] > 0:
                    _cal_path = _th_dir / "calibration_by_slice.json"
                    with open(_cal_path, "w") as _cal_f:
                        json.dump(_cal_slices, _cal_f, indent=2, default=str)
                # Calibration dashboard (extended views)
                _cal_dash = build_calibration_dashboard(as_of_date)
                if _cal_dash["n_resolved"] > 0:
                    _cal_dash_path = _th_dir / "calibration_dashboard.json"
                    with open(_cal_dash_path, "w") as _cd_f:
                        json.dump(_cal_dash, _cd_f, indent=2, default=str)
                # Calibration cycle log
                emit_calibration_cycle_summary(_th_result, as_of_date)
                _th_n = _th_result.get("n_catalysts", 0)
                _th_w = _th_result.get("n_warnings", 0)
                _th_hw = _th_result.get("n_warnings_high", 0)
                _th_cal = _th_result.get("calibration_status", "?")
                _th_trend = _th_result.get("base_rate_trend")
                _trend_str = f", trend={_th_trend:+.3f}" if _th_trend is not None else ""
                _logger.info(
                    f"Timing hazard → {_th_n} catalysts, {_th_w} warnings ({_th_hw} HIGH), cal={_th_cal}{_trend_str}"
                )
            else:
                _logger.info(f"Timing hazard → skipped ({_th_result.get('error', '?')})")
        except Exception as _th_err:
            _logger.warning(f"Timing hazard failed: {_th_err}")

        # --- Step 5k.19: Production monitor (non-blocking) ---
        try:
            from tools.build_production_monitor import build_production_monitor

            _pm_result = build_production_monitor(as_of_date)
            if "error" not in _pm_result:
                _pm_attn = _pm_result.get("attention", "?")
                _pm_nalerts = len(_pm_result.get("alerts", []))
                _logger.info(f"Production monitor → {_pm_attn} attention, {_pm_nalerts} alerts")
            else:
                _logger.info(f"Production monitor → skipped ({_pm_result.get('error', '?')})")
        except Exception as _pm_err:
            _logger.warning(f"Production monitor failed: {_pm_err}")

        # --- Step 5k.20: Factor drift monitor (non-blocking) ---
        try:
            from tools.build_factor_drift import build_factor_drift

            _fd_result = build_factor_drift(as_of_date)
            if "error" not in _fd_result:
                _fd_attn = _fd_result.get("attention", "?")
                _fd_nalerts = len(_fd_result.get("alerts", []))
                _logger.info(f"Factor drift → {_fd_attn} ({_fd_nalerts} alerts)")
            else:
                _logger.info(f"Factor drift → skipped ({_fd_result.get('error', '?')})")
        except Exception as _fd_err:
            _logger.warning(f"Factor drift failed: {_fd_err}")

        # --- Step 5k.21: Event EV scoring (non-blocking, Spec 060) ---
        try:
            from tools.build_event_ev_scores import build_scores as _build_ev_scores

            _ev_dir = REPO_ROOT / "artifacts" / "event_ev"
            _ev_result = _build_ev_scores(
                as_of_date=as_of_date,
                output_dir=_ev_dir,
                enrich_pubmed=enrich_pubmed,
            )
            _ev_n = _ev_result.get("n_total", 0)
            _ev_act = _ev_result.get("n_actionable", 0)
            if _ev_n > 0:
                _ev_top = _ev_result["leaderboard"][0] if _ev_result.get("leaderboard") else {}
                _logger.info(
                    f"Event EV → {_ev_n} scored, {_ev_act} actionable, top={_ev_top.get('ticker', '?')} "
                    f"EV={_ev_top.get('scenario_ev', 0):+.1f}%"
                )
            else:
                _logger.info("Event EV → 0 catalysts in scoring window")
        except Exception as _ev_err:
            _logger.warning(f"Event EV scoring failed: {_ev_err}")

        # --- Step 5k.21b: Clinical transmission shadow (non-blocking) ---
        try:
            from tools.clinical_transmission_shadow import run_shadow as _run_shadow

            _shadow = _run_shadow(as_of_date)
            _sh_def = _shadow.get("default_actionable_n", "?")
            _sh_tx = _shadow.get("tx_actionable_n", "?")
            _sh_dropped = _shadow.get("dropped_from_actionable", [])
            _logger.info(
                "Clinical shadow → default=%s tx=%s dropped=%s t10=%s/10",
                _sh_def,
                _sh_tx,
                _sh_dropped,
                _shadow.get("top10_overlap", "?"),
            )
        except Exception as _sh_err:
            _logger.warning(f"Clinical transmission shadow failed: {_sh_err}")

        # --- Step 5k.21c: CRT options join table refresh (non-blocking) ---
        try:
            import subprocess as _sp

            _crt_join_result = _sp.run(
                [sys.executable, str(REPO_ROOT / "scripts" / "research" / "build_crt_options_join.py")],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=str(REPO_ROOT),
            )
            if _crt_join_result.returncode == 0:
                _logger.info("CRT join table refreshed")
            else:
                _logger.warning("CRT join table refresh failed (exit %d)", _crt_join_result.returncode)
        except Exception as _crt_err:
            _logger.warning(f"CRT join table refresh failed: {_crt_err}")

        # --- Step 5k.22: Event EV forward validation (non-blocking) ---
        try:
            from tools.build_ev_validation import run as _run_ev_validation

            _val_summary = _run_ev_validation()
            _val_n = _val_summary.get("n_matched", 0)
            _val_status = _val_summary.get("status", "?")
            _val_brier = _val_summary.get("brier_score")
            _brier_str = f", brier={_val_brier:.3f}" if _val_brier is not None else ""
            _logger.info(f"Event EV validation → {_val_n} matched ({_val_status}{_brier_str})")
        except Exception as _val_err:
            _logger.warning(f"Event EV validation failed: {_val_err}")

        # --- Step 5k.23: Event EV promotion readiness (non-blocking) ---
        try:
            from event_ev.promotion_ladder import evaluate_ev_readiness

            _ev_readiness = evaluate_ev_readiness()
            _ready_stage = "off"
            for _stage in ["composite", "sizing_overlay", "rank_overlay", "tiebreaker"]:
                if _ev_readiness.get(_stage, {}).get("ready"):
                    _ready_stage = _stage
                    break
            _ev_n_days = _ev_readiness.get("tiebreaker", {}).get("evidence", {}).get("n_daily_artifacts", 0)
            _logger.info(f"Event EV readiness → {_ev_n_days} days, highest ready: {_ready_stage}")

            # Write readiness artifact
            _readiness_path = REPO_ROOT / "artifacts" / "event_ev" / "ev_promotion_readiness.json"
            with open(_readiness_path, "w") as _rf:
                import json as _json

                _json.dump(_ev_readiness, _rf, indent=2, sort_keys=True, default=str)
        except Exception as _ready_err:
            _logger.warning(f"Event EV readiness check failed: {_ready_err}")

        # --- Step 5k.24: Stage 1 shadow memo (non-blocking) ---
        try:
            from tools.build_ev_shadow_memo import build_memo as _build_shadow_memo

            _memo = _build_shadow_memo(as_of_date)
            if "error" not in _memo:
                _logger.info(
                    "EV shadow → ties=%d, reordered=%s, top30_changed=%s, ev_cov=%s",
                    _memo.get("ties_at_cutoff", 0),
                    _memo.get("names_reordered", False),
                    _memo.get("top30_changed", False),
                    _memo.get("ev_coverage_boundary", "?"),
                )
                # Append to ledger
                _memo_path = REPO_ROOT / "artifacts" / "event_ev" / "ev_shadow_memo.jsonl"
                with open(_memo_path, "a") as _mf:
                    _mf.write(__import__("json").dumps(_memo, sort_keys=True, separators=(",", ":")) + "\n")
        except Exception as _memo_err:
            _logger.warning(f"EV shadow memo failed: {_memo_err}")

        # --- Step 5k.25: Stage 1 canary ledger (non-blocking) ---
        try:
            if "_memo" in dir() and _memo and "error" not in _memo:
                _canary_path = REPO_ROOT / "artifacts" / "event_ev" / "stage1_canary_ledger.jsonl"
                _canary_entry = {
                    "date": as_of_date,
                    "top30_changed": _memo.get("top30_changed", False),
                    "displaced": _memo.get("displaced", []),
                    "promoted": _memo.get("promoted", []),
                    "ties_at_cutoff": _memo.get("ties_at_cutoff", 0),
                    "gap_30_31": _memo.get("gap_30_31"),
                    "rank30": _memo.get("rank30", ""),
                    "rank31": _memo.get("rank31", ""),
                    "ev_coverage_boundary": _memo.get("ev_coverage_boundary", ""),
                    "brier": _val_summary.get("brier_score") if "_val_summary" in dir() else None,
                    "validation_matches": _val_summary.get("n_matched") if "_val_summary" in dir() else None,
                }
                with open(_canary_path, "a") as _cf:
                    _cf.write(__import__("json").dumps(_canary_entry, sort_keys=True, separators=(",", ":")) + "\n")
                _logger.info(
                    "Stage 1 canary → top30_changed=%s, ties=%d, ev_cov=%s",
                    _canary_entry["top30_changed"],
                    _canary_entry["ties_at_cutoff"],
                    _canary_entry["ev_coverage_boundary"],
                )
        except Exception as _canary_err:
            _logger.warning(f"Stage 1 canary ledger failed: {_canary_err}")

        # --- Step 5k.26: Catalyst source report (non-blocking) ---
        try:
            from tools.build_catalyst_source_report import build_report as _build_cat_report

            _cat_report = _build_cat_report(as_of_date)
            _logger.info(
                "Catalyst source → %d active nodes, %d sources",
                _cat_report.get("total_active_nodes", 0),
                len(_cat_report.get("sources", {})),
            )
        except Exception as _cat_err:
            _logger.warning(f"Catalyst source report failed: {_cat_err}")

        # --- Step 5l: Ops digest (non-blocking) ---
        try:
            from tools.build_ops_digest import run_ops_digest

            _od_result = run_ops_digest(as_of_date)
            if "error" in _od_result:
                _logger.info(f"Ops digest → skipped ({_od_result['error']})")
            else:
                _od_attention = _od_result.get("attention", "?")
                _od_path = _od_result.get("_paths", {}).get("md_path", "?")
                _logger.info(f"Ops digest → {_od_attention} ({_od_path})")
        except Exception as _od_err:
            _logger.warning(f"Ops digest failed: {_od_err}")

        # --- Step 5l.4: Options Quality Manifest (non-blocking, Spec 045) ---
        try:
            import csv as _oq_csv
            from datetime import datetime as _oq_dt
            from datetime import timezone as _oq_tz

            from common.options_quality import build_options_quality_manifest

            _oq_rankings_path = staging_date_dir / "rankings.csv"
            if not _oq_rankings_path.exists():
                _oq_rankings_path = final_snapshots_dir / as_of_date / "rankings.csv"
            if _oq_rankings_path.exists():
                with open(_oq_rankings_path, encoding="utf-8") as _oq_f:
                    _oq_rows = list(_oq_csv.DictReader(_oq_f))
                _oq_manifest = build_options_quality_manifest(_oq_rows, _oq_dt.now(_oq_tz.utc))
                _oq_out = final_snapshots_dir / as_of_date / "options_quality_manifest.json"
                _oq_out.parent.mkdir(parents=True, exist_ok=True)
                import json as _oq_json

                with open(_oq_out, "w") as _oq_wf:
                    _oq_json.dump(_oq_manifest, _oq_wf, indent=2, default=str)
                    _oq_wf.write("\n")
                _logger.info(
                    "Options quality → full=%d partial=%d absent=%d (%.1f%% coverage)",
                    _oq_manifest["state_distribution"]["full"],
                    _oq_manifest["state_distribution"]["partial"],
                    _oq_manifest["state_distribution"]["absent"],
                    _oq_manifest["coverage_pct"],
                )
        except Exception as _oq_err:
            _logger.warning(f"Options quality manifest failed: {_oq_err}")

        # --- Step 5l.4b: Event Premium Decomposition (non-blocking) ---
        try:
            import csv as _epd_csv

            from common.event_premium_decomp import compute_universe_decomp
            from common.options_surface_signals import load_historical_iv_feature_history

            _epd_rankings_path = staging_date_dir / "rankings.csv"
            if not _epd_rankings_path.exists():
                _epd_rankings_path = final_snapshots_dir / as_of_date / "rankings.csv"
            _epd_iv_hist_path = REPO_ROOT / "data" / "research" / "historical_iv_features.csv"

            if _epd_rankings_path.exists():
                with open(_epd_rankings_path, encoding="utf-8") as _epd_f:
                    _epd_all_rows = list(_epd_csv.DictReader(_epd_f))
                # Top-30 only
                _epd_ranked = [r for r in _epd_all_rows if r.get("actionable_rank", "").strip()]
                _epd_ranked.sort(key=lambda r: int(r["actionable_rank"]))
                _epd_top30 = _epd_ranked[:30]

                # Load IV history if available
                _epd_iv_histories = {}
                _epd_rr_histories = {}
                if _epd_iv_hist_path.exists():
                    _epd_iv_raw = load_historical_iv_feature_history(_epd_iv_hist_path)
                    for _tk, _hist in _epd_iv_raw.items():
                        _epd_iv_histories[_tk] = _hist
                        _epd_rr_histories[_tk] = [
                            r.get("rr_25d", 0)
                            for r in _hist
                            if r.get("rr_25d") is not None
                            and not __import__("math").isnan(r.get("rr_25d", float("nan")))
                        ]

                # Load event move table for implied-vs-realized mispricing
                _epd_emt = None
                _epd_emt_path = REPO_ROOT / "data" / "research" / "event_move_table.json"
                if _epd_emt_path.exists():
                    with open(_epd_emt_path) as _emt_f:
                        _epd_emt_raw = json.load(_emt_f)
                        _epd_emt = _epd_emt_raw.get("table", _epd_emt_raw)

                _epd_results = compute_universe_decomp(
                    _epd_top30,
                    iv_histories=_epd_iv_histories,
                    rr_histories=_epd_rr_histories,
                    event_move_table=_epd_emt,
                )
                _epd_out = final_snapshots_dir / as_of_date / "event_premium_decomp.json"
                _epd_out.parent.mkdir(parents=True, exist_ok=True)
                import json as _epd_json

                with open(_epd_out, "w") as _epd_wf:
                    _epd_json.dump(
                        {
                            "schema": "event_premium_decomp.v1",
                            "as_of_date": as_of_date,
                            "n_names": len(_epd_results),
                            "n_full": sum(1 for r in _epd_results if r.get("epd_quality") == "full"),
                            "n_partial": sum(1 for r in _epd_results if r.get("epd_quality") == "partial"),
                            "names": _epd_results,
                        },
                        _epd_wf,
                        indent=2,
                    )
                _n_eventloaded = sum(1 for r in _epd_results if "event_loaded" in (r.get("epd_surface_regime") or ""))
                _logger.info("Event premium decomp → %d names, %d event_loaded", len(_epd_results), _n_eventloaded)
        except Exception as _epd_err:
            _logger.warning(f"Event premium decomp failed (non-blocking): {_epd_err}")

        # --- Step 5l.5: Company News Ingest (Herald agent, non-blocking, Spec 044) ---
        try:
            import subprocess as _sp_herald

            _herald_result = _sp_herald.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "tools" / "fetch_company_press_releases.py"),
                    "--as-of-date",
                    as_of_date,
                ],
                capture_output=True,
                text=True,
                # 341 IR sources × ~2 sources/ticker × 2s rate limit = ~1364s
                # in time.sleep alone, plus HTTP latency. The previous 600s
                # budget was insufficient and timed out 9/10 days. 1800s is a
                # band-aid until the loop is parallelized.
                timeout=1800,
                cwd=str(REPO_ROOT),
            )
            _herald_lines = (_herald_result.stdout or "").strip().split("\n")
            _herald_summary = _herald_lines[-1] if _herald_lines else "no output"
            _logger.info(f"Herald (PR ingest) → {_herald_summary}")

            # Dedupe then classify
            _releases_path = REPO_ROOT / "data" / "press_releases" / f"releases_{as_of_date}.jsonl"
            if _releases_path.exists() and _releases_path.stat().st_size > 0:
                # Dedupe first
                _dedupe_result = _sp_herald.run(
                    [
                        sys.executable,
                        str(REPO_ROOT / "tools" / "dedupe_press_releases.py"),
                        "--input",
                        str(_releases_path),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    cwd=str(REPO_ROOT),
                )
                _dedupe_lines = (_dedupe_result.stdout or "").strip().split("\n")
                _dedupe_summary = _dedupe_lines[-1] if _dedupe_lines else "no output"
                _logger.info(f"Herald (dedupe) → {_dedupe_summary}")

                # Classify deduped output
                _deduped_path = REPO_ROOT / "data" / "press_releases" / "deduped" / f"deduped_{as_of_date}.jsonl"
                _classify_input = _deduped_path if _deduped_path.exists() else _releases_path
                _classify_result = _sp_herald.run(
                    [
                        sys.executable,
                        str(REPO_ROOT / "tools" / "classify_press_releases.py"),
                        "--input",
                        str(_classify_input),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=300,
                    cwd=str(REPO_ROOT),
                )
                _classify_lines = (_classify_result.stdout or "").strip().split("\n")
                _classify_summary = _classify_lines[-1] if _classify_lines else "no output"
                _logger.info(f"Herald (classify) → {_classify_summary}")
        except Exception as _herald_err:
            _logger.warning(f"Herald failed: {_herald_err}")

        # --- Step 5m: Catalyst Resolution Tracker (non-blocking, Spec 042) ---
        try:
            from datetime import date as _date_cls

            from tools.catalyst_resolution_tracker import run_crt

            _crt_as_of = _date_cls.fromisoformat(as_of_date)
            _crt_resolutions_dir = REPO_ROOT / "data" / "snapshots" / "resolutions"
            _crt_result = run_crt(
                as_of_date=_crt_as_of,
                snapshots_dir=final_snapshots_dir,
                resolutions_dir=_crt_resolutions_dir,
                sec_8k_cache_dir=REPO_ROOT / "cache" / "sec" / "8k_catalysts",
                ctgov_cache_dir=REPO_ROOT / "cache" / "ctgov",
                pdufa_path=REPO_ROOT / "production_data" / "pdufa_dates.json",
            )
            _crt_n_wl = _crt_result.get("n_watchlist", 0)
            _crt_n_new = _crt_result.get("n_new_records", 0)
            _logger.info(f"CRT → {_crt_n_wl} watchlist, {_crt_n_new} new resolutions")

            # Run calibration if first of month
            if _crt_as_of.day == 1:
                from datetime import timedelta as _td

                from tools.crt_calibration import build_calibration_summary, evaluate_governance_triggers

                _cal_period = (_crt_as_of - _td(days=1)).strftime("%Y-%m")
                _cal_summary = build_calibration_summary(_crt_resolutions_dir, _cal_period)
                _cal_path = _crt_resolutions_dir / "calibration_summary.json"
                with open(_cal_path, "w") as _cal_f:
                    import json as _json_mod

                    _json_mod.dump(_cal_summary, _cal_f, indent=2, default=str)
                    _cal_f.write("\n")
                _logger.info(f"CRT calibration → {_cal_summary['total_resolutions']} resolutions for {_cal_period}")

                _triggers = evaluate_governance_triggers(_crt_resolutions_dir, _cal_period)
                _met = [t for t in _triggers if t["status"] == "MET"]
                if _met:
                    for _t in _met:
                        _logger.info(f"CRT GOVERNANCE TRIGGER MET: {_t['trigger']}")
        except Exception as _crt_err:
            _logger.warning(f"CRT failed: {_crt_err}")

        # --- Step 5n: AACT trial warehouse refresh (weekly, non-blocking) ---
        # AACT download is 2.3GB+ per run; weekly is sufficient for trial-level data.
        # Runs on Sundays (weekday 6) or if snapshot is >7 days stale.
        try:
            _aact_snapshot_dir = REPO_ROOT / "data" / "aact" / "snapshots" / as_of_date
            _aact_snap_root = REPO_ROOT / "data" / "aact" / "snapshots"
            _aact_is_weekly_day = date.fromisoformat(as_of_date).weekday() == 0  # Monday
            _aact_latest_age = 999
            if _aact_snap_root.exists():
                _aact_existing = sorted(
                    (d.name for d in _aact_snap_root.iterdir() if d.is_dir() and (d / "aact_health.json").exists()),
                    reverse=True,
                )
                if _aact_existing:
                    _aact_latest_age = (date.fromisoformat(as_of_date) - date.fromisoformat(_aact_existing[0])).days
            _aact_should_run = _aact_is_weekly_day or _aact_latest_age > 7

            if _aact_snapshot_dir.exists() and (_aact_snapshot_dir / "aact_health.json").exists():
                _logger.info("AACT snapshot already exists for %s — skipping", as_of_date)
            elif not _aact_should_run:
                _logger.info(
                    "AACT skipped — not weekly day and latest snapshot is %dd old (threshold: 7d)",
                    _aact_latest_age,
                )
            else:
                _logger.info(
                    "\n[5n] AACT trial warehouse refresh (weekly, latest %dd old) ...",
                    _aact_latest_age,
                )
                if _aact_snap_root.exists():
                    _priors = sorted(
                        (d for d in _aact_snap_root.iterdir() if d.is_dir() and d.name < as_of_date),
                        reverse=True,
                    )
                    if _priors:
                        _aact_prior = _priors[0]  # noqa: F841 — used in future AACT diff step
                _aact_result = _run_subprocess(
                    [
                        sys.executable,
                        str(REPO_ROOT / "tools" / "fetch_aact_snapshot.py"),
                        "--download",
                        "--as-of-date",
                        as_of_date,
                    ],
                    timeout=1800,  # 30 min — AACT download is large
                )
                if _aact_result.returncode == 0:
                    _logger.info("AACT → refresh complete")
                else:
                    _logger.warning("AACT → exit %d", _aact_result.returncode)
        except Exception as _aact_err:
            _logger.warning(f"AACT refresh failed (non-blocking): {_aact_err}")

        # --- Step 5o: Construction v2 shadow (non-blocking) ---
        try:
            _v2_result = _run_subprocess(
                [
                    sys.executable,
                    str(REPO_ROOT / "tools" / "construction_v2_shadow.py"),
                    "--as-of-date",
                    as_of_date,
                ],
                timeout=120,
            )
            if _v2_result.returncode == 0:
                _logger.info("Construction v2 shadow → updated")
            else:
                _logger.warning("Construction v2 shadow → exit %d", _v2_result.returncode)
        except Exception as _v2_err:
            _logger.warning(f"Construction v2 shadow failed (non-blocking): {_v2_err}")

        # --- Step 5p: Construction v2 daily compare (non-blocking) ---
        try:
            _compare_result = _run_subprocess(
                [
                    sys.executable,
                    str(REPO_ROOT / "tools" / "build_daily_v2_compare.py"),
                    "--as-of-date",
                    as_of_date,
                ],
                timeout=60,
            )
            if _compare_result.returncode == 0:
                _logger.info("V2 compare → updated")
        except Exception as _compare_err:
            _logger.warning(f"V2 compare failed (non-blocking): {_compare_err}")

        # --- Step 5q: Rolling options EV summary (non-blocking) ---
        try:
            _ev_result = _run_subprocess(
                [sys.executable, str(REPO_ROOT / "scripts" / "research" / "rolling_options_ev_summary.py")],
                timeout=120,
            )
            if _ev_result.returncode == 0:
                _logger.info("Options EV summary → updated")
        except Exception as _ev_err:
            _logger.warning(f"Options EV summary failed (non-blocking): {_ev_err}")

        # --- Step 5r: Event quality confusion dashboard (non-blocking, Spec 058+) ---
        try:
            from tools.build_event_quality_confusion import build_confusion_dashboard

            _ec_result = build_confusion_dashboard(as_of_date)
            if _ec_result.get("n_labeled", 0) > 0:
                _ec_dir = REPO_ROOT / "artifacts" / "event_quality"
                _ec_dir.mkdir(parents=True, exist_ok=True)
                _ec_path = _ec_dir / "confusion_dashboard.json"
                with open(_ec_path, "w") as _ec_f:
                    json.dump(_ec_result, _ec_f, indent=2, default=str)
                _ec_acc = _ec_result.get("overall", {}).get("accuracy", "?")
                _logger.info(f"Confusion dashboard → {_ec_result['n_labeled']} labeled, accuracy={_ec_acc}")
            else:
                _logger.info("Confusion dashboard → no labeled data yet")
        except Exception as _ec_err:
            _logger.warning(f"Confusion dashboard failed: {_ec_err}")

        # --- Step 5s: Unified review packet (non-blocking, Spec 058+) ---
        try:
            from tools.build_review_packet import build_review_packet

            _rv_result = build_review_packet(as_of_date)
            if "error" not in _rv_result:
                _rv_dir = REPO_ROOT / "artifacts" / "review"
                _rv_dir.mkdir(parents=True, exist_ok=True)
                _rv_path = _rv_dir / f"{as_of_date}_review_packet.json"
                with open(_rv_path, "w") as _rv_f:
                    json.dump(_rv_result, _rv_f, indent=2, default=str)
                _rv_loaded = sum(_rv_result.get("artifacts_loaded", {}).values())
                _rv_total = len(_rv_result.get("artifacts_loaded", {}))
                _rv_warns = len(_rv_result.get("timing", {}).get("warnings", []))
                _logger.info(f"Review packet → {_rv_loaded}/{_rv_total} artifacts, {_rv_warns} timing warnings")
            else:
                _logger.info(f"Review packet → skipped ({_rv_result.get('error', '?')})")
        except Exception as _rv_err:
            _logger.warning(f"Review packet failed: {_rv_err}")

        # --- Step 6: Backfill matured PIT price forward returns (optional) ---
        # The price anchor was already created in step 2.5 (before gates).
        # Backfill is opt-in (--price-pit-backfill) since it can be slow.
        if not skip_pit_warm and price_pit_backfill:
            _logger.info(f"\n[6] Backfilling PIT price forward returns through {as_of_date} ...")
            _backfill_proc = _run_subprocess(
                [
                    sys.executable,
                    str(REPO_ROOT / "tools" / "warm_price_cache.py"),
                    "--backfill-all",
                    "--price-csv",
                    str(price_csv),
                    "--cache-base",
                    str(_price_cache_base),
                    "--through-date",
                    as_of_date,
                ],
                label="warm_price_cache_backfill",
            )
            if _backfill_proc.returncode == 0:
                _logger.info("PIT price backfill OK")
            else:
                _logger.info(f"PIT price backfill FAILED (exit {_backfill_proc.returncode})")
                if _backfill_proc.stderr:
                    for _line in _backfill_proc.stderr.strip().splitlines()[-5:]:
                        _logger.info(f"  {_line}")

        # --- Step 7: TrapOps daily monitor (non-blocking, idempotent) ---
        # Writes trapops_daily_summary.json into the promoted snapshot dir.
        # Skips if the artifact already exists (safe on reruns/watchdog recovery).
        _trapops_out = final_path / "trapops_daily_summary.json"
        if _trapops_out.exists():
            _logger.info("[TrapOps] artifact already present — skipping (idempotent)")
        else:
            try:
                from tools.trapops_monitor import print_report, run_trapops

                _trapops = run_trapops(as_of_date, final_snapshots_dir)
                if "error" not in _trapops:
                    _trapops_state = _trapops.get("health", {}).get("state", "?")
                    _trapops_n = _trapops.get("health", {}).get("n_eligible", 0)
                    _logger.info(
                        "[TrapOps] %s — %d eligible, %d alerts",
                        _trapops_state,
                        _trapops_n,
                        len(_trapops.get("health", {}).get("alerts", [])),
                    )
                    print_report(_trapops)
                    # Write artifact — this is the canonical write path.
                    # trapops_monitor.main() also writes it when run standalone;
                    # the production pipeline was previously logging only (no write).
                    with open(_trapops_out, "w") as _to_f:
                        json.dump(_trapops, _to_f, indent=2, default=str)
                    _logger.info("[TrapOps] written → %s", _trapops_out)
                else:
                    _logger.warning("[TrapOps] Error: %s", _trapops.get("error"))
                    # Write a failure sentinel so watchdog can detect and report.
                    with open(_trapops_out, "w") as _to_f:
                        json.dump({"error": _trapops.get("error"), "snapshot_date": as_of_date}, _to_f, indent=2)
                    _logger.warning("[TrapOps] failure sentinel written → %s", _trapops_out)
            except Exception as _trapops_err:
                _logger.warning(f"TrapOps monitor failed: {_trapops_err}")
                # Write failure sentinel so watchdog knows it ran but errored.
                try:
                    with open(_trapops_out, "w") as _to_f:
                        json.dump({"error": str(_trapops_err), "snapshot_date": as_of_date}, _to_f, indent=2)
                except OSError:
                    pass

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
        "--as-of-date",
        required=True,
        help="Snapshot date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "production_data",
        help="Path to production_data/ (default: production_data/)",
    )
    parser.add_argument(
        "--price-history",
        type=Path,
        default=REPO_ROOT / "production_data" / "price_history.csv",
        help="Path to price_history.csv (default: production_data/price_history.csv)",
    )
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=REPO_ROOT / "data" / "snapshots",
        help="Final snapshot directory (default: data/snapshots/)",
    )
    parser.add_argument(
        "--ruleset",
        type=Path,
        default=None,
        help="Path to decision engine ruleset JSON",
    )
    parser.add_argument(
        "--gate-config",
        type=Path,
        default=None,
        help="Path to gate configuration JSON (overrides defaults)",
    )
    parser.add_argument(
        "--skip-price-refresh",
        action="store_true",
        help="Skip incremental price refresh (use existing price_history.csv)",
    )
    parser.add_argument(
        "--mode",
        default="daily-production",
        choices=["daily-validation", "daily-production", "full-refresh", "research"],
        help=(
            "Run mode (composes price scope, warm sources, and snapshot target):\n"
            "  daily-validation — fast frozen-model proof: Top-30+XBI prices, ctgov-only warm.\n"
            "  daily-production — normal screen (DEFAULT): core sources, full price refresh.\n"
            "  full-refresh     — weekly/deep: all sources incl. slow registries (euctr/ctis/isrctn).\n"
            "  research         — experimental: writes to data/research_snapshots, drift/forward gates skipped.\n"
            "Explicit --price-refresh-scope / --warm-sources / --snapshot-dir override the mode default."
        ),
    )
    parser.add_argument(
        "--price-refresh-scope",
        default=None,
        choices=["full", "top30", "top60"],
        help=(
            "Price refresh scope: 'full' (all universe tickers), 'top30' "
            "(prior-snapshot Top-30 + XBI — fastest), or 'top60' (prior-snapshot "
            "Top-60 + XBI — covers the rank-depth shadow cohort so ranks 31-60 "
            "forward returns use fresh prices). Falls back to full if no prior "
            "snapshot found. Default is set by --mode (top30 for daily-validation, "
            "full otherwise). Annotation/validation only — never changes the model."
        ),
    )
    parser.add_argument(
        "--validation-rank-depth",
        type=int,
        default=None,
        choices=[30, 60],
        help=(
            "Explicit rank-depth for daily validation runs. When set, widens the "
            "price-refresh scope to cover the cohort (30 -> top30, 60 -> top60 + XBI) "
            "unless --price-refresh-scope is given explicitly. The rank-depth shadow "
            "cohorts (top30 / rank31_60 / top60) are tracked in the forward "
            "validation ledger regardless. VALIDATION_INFRASTRUCTURE / NO_MODEL_CHANGE: "
            "does not change ranking, selection, scoring, eligibility, sizing, or trading."
        ),
    )
    parser.add_argument(
        "--skip-audit",
        action="store_true",
        help="Skip data integrity audit step",
    )
    parser.add_argument(
        "--allow-date-fallback",
        action="store_true",
        help="If ctgov cache missing for --as-of-date, fall back to latest cached date (WARN).",
    )
    parser.add_argument(
        "--ctgov-cache-dir",
        type=Path,
        default=None,
        help="Path to ctgov cache directory (default: cache/ctgov/)",
    )
    parser.add_argument(
        "--drift-thresholds",
        type=Path,
        default=None,
        help="Path to drift monitoring thresholds JSON (default: built-in)",
    )
    parser.add_argument(
        "--skip-drift",
        action="store_true",
        help="Skip drift monitoring gate",
    )
    parser.add_argument(
        "--skip-forward-eval",
        action="store_true",
        help="Skip forward-return rolling IC gate",
    )
    parser.add_argument(
        "--price-cache-dir",
        type=Path,
        default=None,
        help="Base dir for PIT price caches (default: data/caches/price_pit/PIT/)",
    )
    parser.add_argument(
        "--fail-on-bad-cache",
        action="store_true",
        help="Exit non-zero if cache health sentinel detects BAD status (SEC 8-K outage or extreme CTGov shift).",
    )
    parser.add_argument(
        "--skip-pit-warm",
        action="store_true",
        help=(
            "Skip all PIT warm steps (cache warm, price anchor, backfill). "
            "Use when CI handles PIT warming externally."
        ),
    )
    parser.add_argument(
        "--warm-sources",
        default=None,
        help=(
            "Comma-separated sources passed to warm_caches.py in step 1.5. "
            "Default is set by --mode (ctgov for daily-validation; "
            "sec_8k,ctgov,sec_13f,fda_adcom,fda_regulatory for daily-production; "
            "+euctr,ctis,isrctn,merged_trials for full-refresh). "
            "Slow registry sources are excluded from the daily path — they run in "
            "cron_data_refresh.sh (14:00 ET) to avoid blocking production. "
            "Use empty string to skip."
        ),
    )
    parser.add_argument(
        "--no-warm-price-pit",
        action="store_true",
        help="Skip PIT price anchor creation in step 2.5.",
    )
    parser.add_argument(
        "--price-pit-backfill",
        action="store_true",
        help="After promotion, backfill matured forward returns for all PIT price caches.",
    )
    parser.add_argument(
        "--no-auto-refresh-market-data",
        action="store_true",
        help="Do not auto-refresh market_data.json when staleness gate fires; abort instead.",
    )
    parser.add_argument(
        "--allow-candidate",
        action="store_true",
        help="Allow candidate rulesets (WARN instead of FAIL in governance gate).",
    )
    parser.add_argument(
        "--json-logs",
        action="store_true",
        help="Emit JSON-structured logs to stdout (also activated by LOG_FORMAT=json env var).",
    )
    parser.add_argument(
        "--enrich-pubmed",
        action="store_true",
        help="Enable PubMed literature enrichment in Event EV scoring (optional, cached).",
    )
    parser.add_argument(
        "--run-scientific-cartography",
        action="store_true",
        help="Enable optional Scientific Cartography Phase 7A diagnostic artifacts (post-snapshot, non-blocking).",
    )
    parser.add_argument(
        "--scientific-cartography-strict",
        action="store_true",
        help="If enabled with --run-scientific-cartography, fail the run if diagnostics fail.",
    )
    parser.add_argument(
        "--run-scientific-cartography-phase13c",
        action="store_true",
        help="Enable optional Phase 13C disease map artifact export (per-disease JSON/CSV/MD, post-snapshot, non-blocking).",
    )
    parser.add_argument(
        "--scientific-cartography-phase13c-strict",
        action="store_true",
        help="If enabled with --run-scientific-cartography-phase13c, fail the run if export fails.",
    )
    args = parser.parse_args()

    # -- Logging setup (must be before any logger calls) --
    if args.json_logs or os.environ.get("LOG_FORMAT", "").lower() == "json":
        from common.logging_config import setup_structured_logging

        setup_structured_logging()

    # --- Resolve --mode into concrete flags ---
    # The mode sets a baseline for price scope + warm sources + snapshot target.
    # An explicit --price-refresh-scope / --warm-sources / --snapshot-dir on the
    # command line always overrides the mode default (detected via argparse
    # sentinel None for scope/sources, and sys.argv for snapshot-dir).
    # NOTE: mode never changes the model, ranker, selector, ruleset, or PIT
    # integrity checks — it only governs refresh breadth and snapshot location.
    _mode_cfg = MODE_DEFAULTS[args.mode]
    if args.price_refresh_scope is None:
        args.price_refresh_scope = _mode_cfg["price_refresh_scope"]
    if args.warm_sources is None:
        args.warm_sources = _mode_cfg["warm_sources"]

    # --validation-rank-depth makes rank-depth shadow tracking explicit and widens
    # the price-refresh scope to cover the cohort so ranks 31-60 use fresh prices.
    # Explicit --price-refresh-scope still wins. NO_MODEL_CHANGE: this only governs
    # refresh breadth — never ranking, selection, scoring, sizing, or trading.
    _resolved_scope = resolve_validation_rank_depth_scope(
        args.validation_rank_depth,
        scope_explicitly_set="--price-refresh-scope" in sys.argv,
        current_scope=args.price_refresh_scope,
    )
    if _resolved_scope != args.price_refresh_scope:
        args.price_refresh_scope = _resolved_scope
        _logger.info(
            "[MODE] --validation-rank-depth=%s → price_refresh_scope=%s",
            args.validation_rank_depth,
            args.price_refresh_scope,
        )

    # Research mode: never write a production snapshot. Redirect to a scratch
    # snapshot tree and skip drift/forward-eval gates (they compare against
    # production priors that won't exist there).
    if args.mode == "research":
        if "--snapshot-dir" not in sys.argv:
            args.snapshot_dir = REPO_ROOT / "data" / "research_snapshots"
        args.skip_drift = True
        args.skip_forward_eval = True
        _logger.info(
            "[MODE] research → snapshot_dir=%s, drift/forward-eval gates skipped",
            args.snapshot_dir,
        )
    _logger.info(
        "[MODE] %s → price_refresh_scope=%s, warm_sources=%s",
        args.mode,
        args.price_refresh_scope,
        args.warm_sources,
    )

    config = GateConfig()
    if args.gate_config:
        config = GateConfig.from_json(args.gate_config)

    _drift_th = None
    if args.drift_thresholds:
        _drift_th = DriftThresholds.from_json(args.drift_thresholds)

    try:
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
            drift_thresholds=_drift_th,
            skip_drift=args.skip_drift,
            skip_forward_eval=args.skip_forward_eval,
            price_cache_dir=args.price_cache_dir,
            fail_on_bad_cache=args.fail_on_bad_cache,
            skip_pit_warm=args.skip_pit_warm,
            warm_sources=args.warm_sources,
            warm_price_pit=not args.no_warm_price_pit,
            price_pit_backfill=args.price_pit_backfill,
            auto_refresh_market_data=not args.no_auto_refresh_market_data,
            allow_candidate=args.allow_candidate,
            enrich_pubmed=args.enrich_pubmed,
            run_scientific_cartography=args.run_scientific_cartography,
            scientific_cartography_strict=args.scientific_cartography_strict,
            run_scientific_cartography_phase13c=args.run_scientific_cartography_phase13c,
            scientific_cartography_phase13c_strict=args.scientific_cartography_phase13c_strict,
            price_refresh_scope=args.price_refresh_scope,
            mode=args.mode,
        )
    except Exception as exc:
        # Ensure a FAIL manifest + ledger entry exist even on unhandled crash
        import traceback

        _logger.error("Unhandled exception in run_daily: %s", exc, exc_info=True)
        _logger.error(f"\nFATAL: {exc}")
        traceback.print_exc()
        manifest = {
            "manifest_version": MANIFEST_VERSION,
            "as_of_date": args.as_of_date,
            "generated_at": _deterministic_timestamp(args.as_of_date),
            "overall_status": "FAIL",
            "gates": [],
            "crash": {
                "exception": str(exc),
                "type": type(exc).__name__,
            },
        }
        try:
            append_gate_verdict(manifest)
        except Exception:
            pass  # best-effort; don't mask the original crash

    # Always write manifest to output/ for CI discoverability
    # (snapshot-dir manifest only exists on successful promotion)
    output_dir = REPO_ROOT / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    fallback_manifest = output_dir / "run_manifest.json"
    _fd2, _tmp2 = tempfile.mkstemp(dir=output_dir, prefix=".tmp_manifest_", suffix=".json")
    try:
        with os.fdopen(_fd2, "w") as f:
            json.dump(manifest, f, indent=2, default=str)
        Path(_tmp2).replace(fallback_manifest)
    except Exception:
        try:
            os.unlink(_tmp2)
        except OSError:
            pass
        raise

    status = manifest.get("overall_status", "FAIL")
    if status == "FAIL":
        sys.exit(1)
    elif status == "WARN":
        sys.exit(2)
    sys.exit(0)


if __name__ == "__main__":
    main()
