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
# Ops contract constants
# ---------------------------------------------------------------------------
MANIFEST_VERSION = "1.3.0"

# Canonical gate names — every gate emitted by run_daily() MUST be in this set.
# Adding a new gate requires updating this allowlist.
GATE_ALLOWLIST: frozenset[str] = frozenset({
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
})

# Required fields in each market_data.json record for schema gate
MARKET_DATA_REQUIRED_FIELDS: frozenset[str] = frozenset({
    "ticker", "price", "market_cap", "collected_at",
})

# Fields that must be numeric (int/float) or None — schema gate checks type
MARKET_DATA_NUMERIC_FIELDS: frozenset[str] = frozenset({
    "price", "market_cap", "avg_volume", "beta",
    "52w_high", "52w_low", "volatility_90d", "returns_1m", "returns_3m",
})

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

    market_data_max_age_days: int = 3
    """Max calendar-day age of market_data.json (collected_at vs as_of_date)."""

    market_data_min_coverage: float = 0.90
    """Min fraction of universe tickers that must have market_data records."""

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
    """Versioned thresholds for drift monitoring gate (WARN-only)."""
    warn_top20_overlap_pct: float = 70.0
    warn_top60_overlap_pct: float = 80.0
    warn_rank_spearman_rho: float = 0.90
    warn_mean_abs_rank_delta_top60: float = 8.0
    warn_tier_migration_count: int = 10
    warn_eligibility_change_count: int = 10

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

    candidates = all_dates[idx + 1:] if idx >= 0 else all_dates
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

    lines.append("")

    if m.get("top20_entrants"):
        lines.append(f"**Top-20 entrants**: {', '.join(m['top20_entrants'])}")
    if m.get("top20_exits"):
        lines.append(f"**Top-20 exits**: {', '.join(m['top20_exits'])}")

    if report.get("warn_reasons"):
        lines.append("")
        lines.append("## Warnings")
        for r in report["warn_reasons"]:
            lines.append(f"- {r}")

    lines.append("")
    path.write_text("\n".join(lines))


def check_drift_monitoring(
    staging_date_dir: Path,
    snapshot_dir: Path,
    as_of_date: str,
    thresholds: DriftThresholds,
) -> GateResult:
    """Compare current snapshot vs most recent prior. WARN-only gate (never FAIL).

    Writes drift_report.json + drift_report.md as sidecar artifacts.
    """
    prior = _find_prior_snapshot(snapshot_dir, as_of_date)
    if prior is None:
        return GateResult(
            name="drift_monitoring", status="PASS",
            detail="No prior snapshot; drift check skipped",
        )

    current_csv = staging_date_dir / "rankings.csv"
    prior_csv = prior / "rankings.csv"

    if not current_csv.exists():
        return GateResult(
            name="drift_monitoring", status="PASS",
            detail="No rankings.csv in current snapshot; drift check skipped",
        )

    metrics = _compute_drift_metrics(current_csv, prior_csv)

    # Evaluate thresholds — collect warn reasons
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

    status = "WARN" if warn_reasons else "PASS"

    # Build and write drift report JSON
    report = {
        "version": "1.0.0",
        "thresholds_id": thresholds.thresholds_id,
        "thresholds": asdict(thresholds),
        "current_date": as_of_date,
        "prior_date": prior.name,
        "metrics": metrics,
        "warn_reasons": warn_reasons,
        "status": status,
    }

    report_json_path = staging_date_dir / "drift_report.json"
    with open(report_json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    report_md_path = staging_date_dir / "drift_report.md"
    _write_drift_report_md(report, report_md_path)

    detail = f"vs {prior.name}: "
    if warn_reasons:
        detail += "; ".join(warn_reasons)
    else:
        detail += (
            f"top20={metrics['top20_overlap_pct']:.0f}%, "
            f"top60={metrics['top60_overlap_pct']:.0f}%, "
            f"rho={metrics['rank_spearman_rho']}"
        )

    return GateResult(
        name="drift_monitoring", status=status,
        detail=detail,
        value=metrics,
        threshold=asdict(thresholds),
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
            name="ctgov_pit_dates", status="WARN",
            detail=f"No trial_records cache found in {ctgov_cache_dir}",
        )

    try:
        with open(cache_file) as f:
            records = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        return GateResult(
            name="ctgov_pit_dates", status="WARN",
            detail=f"Cannot read {cache_file.name}: {e}",
        )

    if not isinstance(records, list) or len(records) == 0:
        return GateResult(
            name="ctgov_pit_dates", status="WARN",
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
        warn_reasons.append(
            f"first_posted coverage {cov_first:.1%} < {warn_first_posted_min:.0%}"
        )
    if cov_last < warn_last_update_min:
        warn_reasons.append(
            f"last_update_posted coverage {cov_last:.1%} < {warn_last_update_min:.0%}"
        )

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
        name="ctgov_pit_dates", status=status,
        detail=", ".join(detail_parts),
        value=coverage,
        threshold=thresholds,
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


def check_audit_result(
    audit_proc: subprocess.CompletedProcess,
    config: GateConfig,
    audit_output_dir: Optional[Path] = None,
) -> GateResult:
    """Translate audit tool exit code into a gate result."""
    summary = _read_invariants_summary(audit_output_dir) if audit_output_dir else None

    if audit_proc.returncode == 0:
        detail = _format_audit_detail("Audit OK", summary)
        return GateResult(name="audit", status="PASS", detail=detail)
    elif audit_proc.returncode == 2:
        status = "WARN" if config.audit_warn_is_gate_warn else "PASS"
        detail = _format_audit_detail("Audit WARN", summary)
        return GateResult(name="audit", status=status, detail=detail)
    else:
        status = "FAIL" if config.audit_fail_is_gate_fail else "WARN"
        detail = _format_audit_detail(
            f"Audit FAIL (exit code {audit_proc.returncode})", summary,
        )
        return GateResult(name="audit", status=status, detail=detail)


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
            name="inputs_present", status="FAIL",
            detail=f"missing: {', '.join(missing)}",
        )
    return GateResult(
        name="inputs_present", status="PASS",
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
            name="market_data_staleness", status="PASS",
            detail="Skipped (file missing; inputs_present gate will catch)",
        )

    try:
        with open(mkt_path) as f:
            records = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        return GateResult(
            name="market_data_staleness", status="FAIL",
            detail=f"Cannot read market_data.json: {e}",
        )

    # Find the most common collected_at date
    collected_dates = [
        r.get("collected_at") for r in records
        if isinstance(r, dict) and r.get("collected_at")
    ]
    if not collected_dates:
        return GateResult(
            name="market_data_staleness", status="FAIL",
            detail="No collected_at field found in market_data.json",
        )

    collected_at = max(collected_dates)  # latest collection date
    try:
        collected = _date.fromisoformat(collected_at)
        as_of = _date.fromisoformat(as_of_date)
    except ValueError as e:
        return GateResult(
            name="market_data_staleness", status="FAIL",
            detail=f"Bad date format: {e}",
        )

    age_days = (as_of - collected).days
    if age_days > max_age_days:
        return GateResult(
            name="market_data_staleness", status="FAIL",
            detail=(
                f"market_data.json is {age_days}d stale "
                f"(collected={collected_at}, as_of={as_of_date}, max={max_age_days}d). "
                f"Run: python collect_market_data.py"
            ),
            value=age_days, threshold=max_age_days,
        )

    return GateResult(
        name="market_data_staleness", status="PASS",
        detail=f"market_data.json collected={collected_at}, age={age_days}d",
        value=age_days, threshold=max_age_days,
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
            name="market_data_schema", status="PASS",
            detail="Skipped (file missing; inputs_present gate will catch)",
        )

    try:
        with open(mkt_path) as f:
            records = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        return GateResult(
            name="market_data_schema", status="FAIL",
            detail=f"Cannot parse market_data.json: {e}",
        )

    if not isinstance(records, list):
        return GateResult(
            name="market_data_schema", status="FAIL",
            detail="market_data.json is not a JSON array",
        )

    if len(records) == 0:
        return GateResult(
            name="market_data_schema", status="FAIL",
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
            name="market_data_schema", status="FAIL",
            detail=f"{len(bad_records)} invalid records: {sample}",
            value=len(bad_records), threshold=0,
        )

    return GateResult(
        name="market_data_schema", status="PASS",
        detail=f"{len(records)} records, all valid",
        value=0, threshold=0,
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
            name="market_data_coverage", status="PASS",
            detail="Skipped (file missing; inputs_present gate will catch)",
        )
    if not uni_path.exists():
        return GateResult(
            name="market_data_coverage", status="FAIL",
            detail="universe.json not found",
        )

    try:
        with open(mkt_path) as f:
            records = json.load(f)
        with open(uni_path) as f:
            universe = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        return GateResult(
            name="market_data_coverage", status="FAIL",
            detail=f"Cannot read files: {e}",
        )

    mkt_tickers = {r["ticker"] for r in records if isinstance(r, dict) and r.get("ticker")}
    uni_tickers = set()
    for entry in universe:
        t = entry.get("ticker") if isinstance(entry, dict) else str(entry)
        if t and not t.startswith("_"):  # exclude synthetic like _XBI_BENCHMARK_
            uni_tickers.add(t)

    if not uni_tickers:
        return GateResult(
            name="market_data_coverage", status="FAIL",
            detail="Universe is empty",
        )

    covered = len(mkt_tickers & uni_tickers)
    coverage = covered / len(uni_tickers)
    missing = sorted(uni_tickers - mkt_tickers)

    if coverage < min_coverage:
        sample = ", ".join(missing[:5])
        return GateResult(
            name="market_data_coverage", status="FAIL",
            detail=(
                f"{covered}/{len(uni_tickers)} ({coverage:.1%}) coverage, "
                f"below {min_coverage:.0%} threshold. "
                f"Missing: {sample}{'...' if len(missing) > 5 else ''}"
            ),
            value=round(coverage, 4), threshold=min_coverage,
        )

    detail = f"{covered}/{len(uni_tickers)} ({coverage:.1%}) coverage"
    if missing:
        detail += f", {len(missing)} missing: {', '.join(missing[:5])}"
    return GateResult(
        name="market_data_coverage", status="PASS",
        detail=detail,
        value=round(coverage, 4), threshold=min_coverage,
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
    dates = [r.get("collected_at") for r in records
             if isinstance(r, dict) and r.get("collected_at")]
    if dates:
        info["collected_at"] = max(dates)
        try:
            from datetime import date as _date
            info["age_days"] = (_date.fromisoformat(as_of_date) -
                                _date.fromisoformat(info["collected_at"])).days
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
                t = entry.get("ticker") if isinstance(entry, dict) else str(entry)
                if t and not t.startswith("_"):
                    uni_tickers.add(t)
            mkt_tickers = {r["ticker"] for r in records
                           if isinstance(r, dict) and r.get("ticker")}
            if uni_tickers:
                info["coverage_pct"] = round(
                    len(mkt_tickers & uni_tickers) / len(uni_tickers), 4,
                )
        except (json.JSONDecodeError, OSError):
            pass

    return info


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
    data_dir: Optional[Path] = None,
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

    # Validate gate names against allowlist
    for g in gate_results:
        if g.name not in GATE_ALLOWLIST:
            raise ValueError(
                f"Gate '{g.name}' not in GATE_ALLOWLIST. "
                f"Add it to the allowlist before using it."
            )

    # Market data refresh provenance
    mkt_refresh: Dict[str, Any] = {}
    if data_dir:
        mkt_refresh = _compute_market_data_refresh(data_dir, as_of_date)

    return {
        "manifest_version": MANIFEST_VERSION,
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
    drift_thresholds: Optional[DriftThresholds] = None,
    skip_drift: bool = False,
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
            data_dir=data_dir,
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
            data_dir=data_dir,
        )
        return manifest

    if effective_as_of_date != as_of_date:
        print(f"  Date fallback: {as_of_date} → {effective_as_of_date}")
        as_of_date = effective_as_of_date

    # --- Gate: required inputs present ---
    inputs_gate = check_inputs_present(data_dir)
    gate_results.append(inputs_gate)
    print(f"  Inputs gate: {inputs_gate.status} — {inputs_gate.detail}")
    if inputs_gate.status == "FAIL":
        print("\n  FATAL: Required input files missing. Aborting before screen run.")
        print(f"  Hint: publish an inputs bundle or copy market_data.json to {data_dir}/")
        manifest = build_run_manifest(
            as_of_date, gate_results, price_stats,
            subprocess.CompletedProcess(args=[], returncode=-1),
            None, config,
            requested_as_of_date=requested_as_of_date,
            git_pre_run=git_pre_run,
            data_dir=data_dir,
        )
        return manifest

    # --- Gate: market data schema ---
    schema_gate = check_market_data_schema(data_dir)
    gate_results.append(schema_gate)
    print(f"  Market data schema gate: {schema_gate.status} — {schema_gate.detail}")
    if schema_gate.status == "FAIL":
        print("\n  FATAL: Market data schema invalid. Aborting before screen run.")
        manifest = build_run_manifest(
            as_of_date, gate_results, price_stats,
            subprocess.CompletedProcess(args=[], returncode=-1),
            None, config,
            requested_as_of_date=requested_as_of_date,
            git_pre_run=git_pre_run,
            data_dir=data_dir,
        )
        return manifest

    # --- Gate: market data staleness ---
    mkt_gate = check_market_data_staleness(data_dir, as_of_date, config.market_data_max_age_days)
    gate_results.append(mkt_gate)
    print(f"  Market data staleness gate: {mkt_gate.status} — {mkt_gate.detail}")
    if mkt_gate.status == "FAIL":
        print("\n  FATAL: Market data too stale. Aborting before screen run.")
        print(f"  Hint: python collect_market_data.py --universe {data_dir}/universe.json")
        manifest = build_run_manifest(
            as_of_date, gate_results, price_stats,
            subprocess.CompletedProcess(args=[], returncode=-1),
            None, config,
            requested_as_of_date=requested_as_of_date,
            git_pre_run=git_pre_run,
            data_dir=data_dir,
        )
        return manifest

    # --- Gate: market data coverage ---
    cov_gate = check_market_data_coverage(data_dir, config.market_data_min_coverage)
    gate_results.append(cov_gate)
    print(f"  Market data coverage gate: {cov_gate.status} — {cov_gate.detail}")
    if cov_gate.status == "FAIL":
        print("\n  FATAL: Market data coverage too low. Aborting before screen run.")
        manifest = build_run_manifest(
            as_of_date, gate_results, price_stats,
            subprocess.CompletedProcess(args=[], returncode=-1),
            None, config,
            requested_as_of_date=requested_as_of_date,
            git_pre_run=git_pre_run,
            data_dir=data_dir,
        )
        return manifest

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
            data_dir=data_dir,
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
            data_dir=data_dir,
        )
        return manifest

    # --- Step 3: Run integrity audit ---
    audit_proc = None
    if not skip_audit:
        print(f"\n[3/5] Running data integrity audit ...")
        audit_output_dir = staging_date_dir / "audit"
        audit_proc = run_audit(staging_date_dir, price_csv, as_of_date, audit_output_dir)
        audit_gate = check_audit_result(audit_proc, config, audit_output_dir)
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

    # --- Gate: drift monitoring (WARN-only) ---
    if not skip_drift:
        _drift_th = drift_thresholds or DriftThresholds()
        drift_gate = check_drift_monitoring(
            staging_date_dir, final_snapshots_dir, as_of_date, _drift_th,
        )
        gate_results.append(drift_gate)
        print(f"  Drift gate: {drift_gate.status} — {drift_gate.detail}")
    else:
        print("  Drift gate: skipped (--skip-drift)")

    # --- Gate: ctgov PIT dates (WARN-only) ---
    pit_dates_gate = check_ctgov_pit_dates(_cache_dir, as_of_date)
    gate_results.append(pit_dates_gate)
    print(f"  CTGov PIT dates gate: {pit_dates_gate.status} — {pit_dates_gate.detail}")

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
        data_dir=data_dir,
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
    parser.add_argument(
        "--drift-thresholds", type=Path, default=None,
        help="Path to drift monitoring thresholds JSON (default: built-in)",
    )
    parser.add_argument(
        "--skip-drift", action="store_true",
        help="Skip drift monitoring gate",
    )
    args = parser.parse_args()

    config = GateConfig()
    if args.gate_config:
        config = GateConfig.from_json(args.gate_config)

    _drift_th = None
    if args.drift_thresholds:
        _drift_th = DriftThresholds.from_json(args.drift_thresholds)

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
