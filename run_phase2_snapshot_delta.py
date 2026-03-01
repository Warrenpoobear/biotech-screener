#!/usr/bin/env python3
"""Phase-2 Snapshot Delta Report.

Compares two Phase-2 snapshots (current vs prior) and produces:
  - Human-readable delta report  (phase2_run_delta_report.txt)
  - Machine-readable JSON        (phase2_run_delta_details.json)
  - Per-ticker delta CSV          (phase2_run_delta.csv)

Handles three cases:
  1. Both snapshots have decision_portfolio.csv (future steady-state)
  2. Current has portfolio, prior has only rankings.csv with tier_dev (reconstruct)
  3. No comparable prior exists (single-snapshot summary mode)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from dataclasses import fields as dc_fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Guardrail thresholds
# ---------------------------------------------------------------------------
PHASE2_PINNED_RULESET_ID = "31f399e0"
PHASE2_A_FLOOR = 0.60  # Phase-2 ruleset a_floor; used by optionality diagnostic
WARN_NAME_TURNOVER_PCT = 40.0
WARN_A_COUNT_MIN = 1
WARN_CATALYST_COVERAGE_MIN = 50.0  # specific_days % among dev-stage

# ---------------------------------------------------------------------------
# Health gate thresholds (externalized, versioned)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Phase2HealthThresholds:
    """Immutable, versioned configuration for health gate thresholds.

    Frozen so instances are hashable and produce a deterministic thresholds_id.
    Defaults are calibrated against the 2025 steady-state archive chain.
    """
    # FAIL gates — data feed is broken
    fail_catalyst_coverage_min: float = 20.0   # below this = catalyst feed broken
    fail_a_count_min: int = 1                  # Phase-2 requires tier separation
    fail_optionality_coverage_min: float = 80.0  # below this = optionality feed broken

    # WARN gates — operational concern
    warn_a_count_low: int = 2                  # too concentrated (calibrated: P50=2, P90=5)
    warn_turnover_pct: float = 50.0            # name turnover (calibrated: P95=17.4)
    warn_weight_l1_pct: float = 55.0           # weight turnover (calibrated: P90=50.8, P95=52.7)
    warn_catalyst_drop_pp: float = 5.0         # coverage drop vs prior (calibrated: max=1.9)

    # Coinvest coverage gate (WARN-only, never FAIL)
    warn_coinvest_coverage_min: float = 70.0   # WARN if < 70% tickers have signal

    # Missingness coverage gates (dev-stage component coverage)
    fail_drawdown_coverage_min: float = 95.0   # below this = risk feed broken
    warn_drawdown_coverage_min: float = 99.0   # mild degradation
    warn_sponsor_coverage_min: float = 90.0    # dev-stage sponsorship coverage
    warn_catalyst_comp_coverage_min: float = 85.0  # dev-stage catalyst component coverage

    # Exposure guardrails — catalyst timing
    warn_catalyst_le_7d_weight_pct: float = 20.0
    fail_catalyst_le_7d_weight_pct: float = 35.0
    warn_catalyst_le_7d_count: int = 4
    fail_catalyst_le_7d_count: int = 7
    # Risk-flag exposure
    warn_high_vol_weight_pct: float = 25.0
    fail_high_vol_weight_pct: float = 35.0
    warn_high_beta_weight_pct: float = 25.0
    fail_high_beta_weight_pct: float = 35.0
    warn_high_vol_or_beta_weight_pct: float = 35.0
    fail_high_vol_or_beta_weight_pct: float = 50.0
    warn_drawdown_rel_xbi_weight_pct: float = 25.0
    fail_drawdown_rel_xbi_weight_pct: float = 40.0
    warn_overbought_rsi_weight_pct: float = 25.0
    fail_overbought_rsi_weight_pct: float = 40.0
    # Stacked risk
    warn_catalyst_le_7d_and_drawdown_weight_pct: float = 10.0
    fail_catalyst_le_7d_and_drawdown_weight_pct: float = 20.0
    # Momentum / concentration
    warn_headwind_weight_pct: float = 30.0
    fail_headwind_weight_pct: float = 45.0
    warn_top5_weight_pct: float = 40.0
    fail_top5_weight_pct: float = 55.0

    def _canonical_json(self) -> str:
        """Deterministic JSON representation for hashing."""
        d = {f.name: getattr(self, f.name) for f in dc_fields(self)}
        return json.dumps(d, sort_keys=True, separators=(",", ":"))

    @property
    def thresholds_id(self) -> str:
        """8-char hex hash of all parameters for change detection."""
        return hashlib.sha256(self._canonical_json().encode()).hexdigest()[:8]

    def to_json(self, path: str) -> None:
        """Write thresholds to a JSON file."""
        d = {f.name: getattr(self, f.name) for f in dc_fields(self)}
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(d, fh, indent=2, sort_keys=True)
            fh.write("\n")

    @classmethod
    def from_json(cls, path: str) -> Phase2HealthThresholds:
        """Load thresholds from a JSON file."""
        with open(path, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        return cls(**d)


DEFAULT_HEALTH_THRESHOLDS = Phase2HealthThresholds()

# Backward-compat module-level aliases (used by tests and calibration harness)
FAIL_CATALYST_COVERAGE_MIN = DEFAULT_HEALTH_THRESHOLDS.fail_catalyst_coverage_min
FAIL_A_COUNT_MIN = DEFAULT_HEALTH_THRESHOLDS.fail_a_count_min
FAIL_OPTIONALITY_COVERAGE_MIN = DEFAULT_HEALTH_THRESHOLDS.fail_optionality_coverage_min
WARN_A_COUNT_LOW = DEFAULT_HEALTH_THRESHOLDS.warn_a_count_low
WARN_TURNOVER_PCT = DEFAULT_HEALTH_THRESHOLDS.warn_turnover_pct
WARN_WEIGHT_L1_PCT = DEFAULT_HEALTH_THRESHOLDS.warn_weight_l1_pct
WARN_CATALYST_DROP_PP = DEFAULT_HEALTH_THRESHOLDS.warn_catalyst_drop_pp

# Portfolio reconstruction defaults (mirrors run_screen.py logic)
RECON_TIER_FILTER = ("A", "B")
RECON_TOP_K = 20

# Risk flag vocabulary
RISK_FLAGS_ALL = [
    "deep_drawdown",
    "deep_drawdown_rel_xbi",
    "high_vol",
    "high_beta",
    "overbought_rsi",
    "low_confidence",
]

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SnapshotData:
    date: str
    path: Path
    rankings: "pd.DataFrame"
    portfolio: "pd.DataFrame"
    metadata: dict
    ruleset_id: str
    has_native_portfolio: bool


@dataclass
class CatalystCoverage:
    n_dev: int
    n_specific: int
    pct: float
    mode_dist: Dict[str, int]
    backfill_share_pct: float = 0.0  # % of non-missing signals from backfill


@dataclass
class DeltaResult:
    entrants: List[str]
    exits: List[str]
    name_turnover_pct: float
    weight_l1_delta: float

    tier_counts_current: Dict[str, int]
    tier_counts_prior: Dict[str, int]
    tier_in_portfolio_current: Dict[str, int]
    tier_in_portfolio_prior: Dict[str, int]

    size_band_current: Dict[str, int]
    size_band_prior: Dict[str, int]

    catalyst_coverage_current: CatalystCoverage
    catalyst_coverage_prior: CatalystCoverage
    top_catalysts_current: List[Dict]
    top_catalysts_prior: List[Dict]

    risk_current: Dict[str, int]
    risk_prior: Dict[str, int]

    warnings: List[str] = field(default_factory=list)


@dataclass
class SingleResult:
    """Summary for a single snapshot (no prior available)."""
    tier_counts: Dict[str, int]
    tier_in_portfolio: Dict[str, int]
    size_band: Dict[str, int]
    catalyst_coverage: CatalystCoverage
    top_catalysts: List[Dict]
    risk: Dict[str, int]
    warnings: List[str] = field(default_factory=list)


@dataclass
class HealthResult:
    """Deterministic health gate outcome."""
    status: str                 # "OK" | "WARN" | "FAIL"
    reasons: List[str]          # e.g. ["high_turnover", "catalyst_drop"]
    metrics: Dict[str, object]  # key numbers for logging


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
import pandas as pd


def _parse_risk_flags(val) -> List[str]:
    """Parse pipe-separated risk_flags string into a list."""
    if pd.isna(val) or val == "":
        return []
    return [f.strip() for f in str(val).split("|") if f.strip()]


def _count_risk_flags(portfolio: pd.DataFrame) -> Dict[str, int]:
    """Count occurrences of each risk flag in portfolio."""
    counts = {f: 0 for f in RISK_FLAGS_ALL}
    for _, row in portfolio.iterrows():
        for flag in _parse_risk_flags(row.get("risk_flags", "")):
            if flag in counts:
                counts[flag] += 1
    return counts


def _safe_float(val, default=0.0) -> float:
    try:
        if pd.isna(val) or val == "":
            return default
        return float(val)
    except (ValueError, TypeError):
        return default


def _safe_int(val, default=0) -> int:
    try:
        if pd.isna(val) or val == "":
            return default
        return int(float(val))
    except (ValueError, TypeError):
        return default


def _tier_counts(rankings: pd.DataFrame) -> Dict[str, int]:
    """Count tier_dev distribution among drug_developer rows."""
    dev = rankings[rankings["archetype"] == "drug_developer"]
    counts = {"A": 0, "B": 0, "C": 0, "D": 0}
    for tier in dev["tier_dev"]:
        t = str(tier).strip()
        if t in counts:
            counts[t] += 1
    return counts


def _tier_in_portfolio(portfolio: pd.DataFrame) -> Dict[str, int]:
    """Count tier distribution within the portfolio."""
    counts = {"A": 0, "B": 0, "C": 0, "D": 0}
    for tier in portfolio["tier_dev"]:
        t = str(tier).strip()
        if t in counts:
            counts[t] += 1
    return counts


def _size_band_counts(portfolio: pd.DataFrame) -> Dict[str, int]:
    counts = {"L": 0, "M": 0, "S": 0, "XS": 0}
    for band in portfolio["size_band"]:
        b = str(band).strip()
        if b in counts:
            counts[b] += 1
    return counts


def _catalyst_coverage(rankings: pd.DataFrame) -> CatalystCoverage:
    """Compute catalyst coverage stats for drug_developer rows."""
    dev = rankings[rankings["archetype"] == "drug_developer"].copy()
    n_dev = len(dev)
    if n_dev == 0:
        return CatalystCoverage(0, 0, 0.0, {})

    mode_col = "catalyst_mode"
    mode_dist: Dict[str, int] = {}
    n_specific = 0
    for mode in dev[mode_col]:
        m = str(mode).strip() if pd.notna(mode) and str(mode).strip() else "missing"
        mode_dist[m] = mode_dist.get(m, 0) + 1
        if m == "specific_days":
            n_specific += 1

    pct = (n_specific / n_dev * 100) if n_dev > 0 else 0.0

    # Backfill share: % of non-missing catalyst signals sourced from backfill
    backfill_share = 0.0
    if "catalyst_source" in dev.columns:
        sources = [str(s).strip() for s in dev["catalyst_source"] if str(s).strip()]
        if sources:
            n_bf = sum(1 for s in sources if s in ("trial_cd", "trial_active"))
            backfill_share = round(n_bf / len(sources) * 100, 1)

    return CatalystCoverage(n_dev, n_specific, round(pct, 1), mode_dist, backfill_share)


def _top_catalysts(rankings: pd.DataFrame, n: int = 10) -> List[Dict]:
    """Return nearest N specific_days catalysts among drug_developer rows."""
    dev = rankings[
        (rankings["archetype"] == "drug_developer")
        & (rankings["catalyst_mode"] == "specific_days")
    ].copy()
    dev["_cat_days_int"] = dev["catalyst_days"].apply(lambda v: _safe_int(v, 99999))
    dev = dev.nsmallest(n, "_cat_days_int")
    result = []
    for _, row in dev.iterrows():
        result.append({
            "ticker": row["ticker"],
            "catalyst_days": _safe_int(row["catalyst_days"]),
            "tier_dev": str(row.get("tier_dev", "")).strip(),
        })
    return result


def _reconstruct_portfolio(rankings: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct portfolio from rankings when portfolio_positions.csv is missing.

    Mirrors the logic in run_screen.py save_validation_snapshot():
    filter eligible + tier_any A/B, sort by actionable_rank, take top 20.
    Falls back to drug_developer + tier_dev for old snapshots without tier_any.
    """
    if "tier_any" in rankings.columns:
        eligible = rankings[
            (rankings["eligible"].astype(str) == "1")
            & (rankings["tier_any"].isin(RECON_TIER_FILTER))
        ].copy()
    else:
        # Legacy fallback for old snapshots
        eligible = rankings[
            (rankings["archetype"] == "drug_developer")
            & (rankings["tier_dev"].isin(RECON_TIER_FILTER))
        ].copy()
    eligible["_ar"] = eligible["actionable_rank"].apply(lambda v: _safe_int(v, 99999))
    portfolio = eligible.nsmallest(RECON_TOP_K, "_ar").copy()
    portfolio.drop(columns=["_ar"], inplace=True)
    return portfolio.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def is_snapshot_degraded(snapshot_dir: Path) -> bool:
    """Return True if snapshot has degraded cache health."""
    health_path = snapshot_dir / "cache_health.json"
    if not health_path.exists():
        return False  # no sentinel → assume healthy (legacy snapshots)
    try:
        health = json.loads(health_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return health.get("degraded_run", False)


def find_snapshots(
    snapshot_dir: Path,
    current_date: Optional[str],
    prior_date: Optional[str],
) -> Tuple[Optional[Path], Optional[Path]]:
    """Locate current and prior snapshot directories.

    Returns (current_path, prior_path).  prior_path may be None.
    """
    date_re = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    all_dates = sorted(
        [d.name for d in snapshot_dir.iterdir() if d.is_dir() and date_re.match(d.name)],
        reverse=True,
    )

    if not all_dates:
        return None, None

    # Resolve current
    if current_date:
        if current_date not in all_dates:
            print(f"ERROR: snapshot directory not found for {current_date}", file=sys.stderr)
            return None, None
        current_name = current_date
    else:
        current_name = all_dates[0]

    current_path = snapshot_dir / current_name

    # Resolve prior
    if prior_date is not None:
        if prior_date.lower() == "none":
            return current_path, None
        if prior_date not in all_dates:
            print(f"ERROR: snapshot directory not found for {prior_date}", file=sys.stderr)
            return current_path, None
        prior_path = snapshot_dir / prior_date
        return current_path, prior_path

    # Auto-find: scan backwards from current for first dir with tier_dev column
    # Skip degraded snapshots — don't use as baseline
    idx = all_dates.index(current_name)
    for candidate in all_dates[idx + 1 :]:
        candidate_path = snapshot_dir / candidate
        if is_snapshot_degraded(candidate_path):
            continue  # skip degraded — don't use as baseline
        rankings_csv = candidate_path / "rankings.csv"
        if not rankings_csv.exists():
            continue
        with open(rankings_csv, "r") as f:
            header = f.readline().strip()
        if "tier_dev" in header.split(","):
            return current_path, candidate_path
    return current_path, None


def load_snapshot(snap_path: Path) -> Optional[SnapshotData]:
    """Load a snapshot directory into a SnapshotData object."""
    rankings_csv = snap_path / "rankings.csv"
    if not rankings_csv.exists():
        print(f"WARNING: {rankings_csv} not found", file=sys.stderr)
        return None

    rankings = pd.read_csv(rankings_csv, dtype=str)
    if "tier_dev" not in rankings.columns:
        print(
            f"WARNING: {snap_path.name}: snapshot schema too old "
            f"(missing 'tier_dev'). Re-enrich or skip.",
            file=sys.stderr,
        )
        return None
    if "actionable_rank" not in rankings.columns:
        print(
            f"WARNING: {snap_path.name}: snapshot schema too old "
            f"(missing 'actionable_rank'). Re-enrich or skip.",
            file=sys.stderr,
        )
        return None

    # Load metadata
    metadata_file = snap_path / "metadata.json"
    metadata = {}
    if metadata_file.exists():
        with open(metadata_file, "r") as f:
            metadata = json.load(f)

    # Determine ruleset_id
    ruleset_id = ""
    if "decision_engine_ruleset_id" in rankings.columns:
        ids = rankings["decision_engine_ruleset_id"].dropna().unique()
        ids = [i for i in ids if i.strip()]
        if ids:
            ruleset_id = ids[0].strip()

    # Load portfolio positions (prefer portfolio_positions.csv, fall back to
    # decision_portfolio.csv for older snapshots, then reconstruct from rankings)
    positions_csv = snap_path / "portfolio_positions.csv"
    legacy_csv = snap_path / "decision_portfolio.csv"
    if positions_csv.exists():
        portfolio = pd.read_csv(positions_csv, dtype=str)
        has_native = True
    elif legacy_csv.exists():
        portfolio = pd.read_csv(legacy_csv, dtype=str)
        has_native = True
    else:
        portfolio = _reconstruct_portfolio(rankings)
        has_native = False

    date_str = snap_path.name

    return SnapshotData(
        date=date_str,
        path=snap_path,
        rankings=rankings,
        portfolio=portfolio,
        metadata=metadata,
        ruleset_id=ruleset_id,
        has_native_portfolio=has_native,
    )


def compute_delta(current: SnapshotData, prior: SnapshotData) -> DeltaResult:
    """Compute the delta between two snapshots."""
    cur_tickers = set(current.portfolio["ticker"].tolist())
    pri_tickers = set(prior.portfolio["ticker"].tolist())

    entrants = sorted(cur_tickers - pri_tickers)
    exits = sorted(pri_tickers - cur_tickers)
    avg_size = (len(cur_tickers) + len(pri_tickers)) / 2
    name_turnover_pct = (
        (len(entrants) + len(exits)) / (2 * avg_size) * 100 if avg_size > 0 else 0.0
    )

    # Weight L1 delta
    cur_weights = {}
    for _, row in current.portfolio.iterrows():
        cur_weights[row["ticker"]] = _safe_float(row.get("target_weight_pct", 0))
    pri_weights = {}
    for _, row in prior.portfolio.iterrows():
        pri_weights[row["ticker"]] = _safe_float(row.get("target_weight_pct", 0))
    all_tickers = cur_tickers | pri_tickers
    weight_l1 = sum(
        abs(cur_weights.get(t, 0.0) - pri_weights.get(t, 0.0)) for t in all_tickers
    )

    # Tier counts (universe)
    tier_counts_current = _tier_counts(current.rankings)
    tier_counts_prior = _tier_counts(prior.rankings)

    # Tier in portfolio
    tier_in_portfolio_current = _tier_in_portfolio(current.portfolio)
    tier_in_portfolio_prior = _tier_in_portfolio(prior.portfolio)

    # Size bands
    size_band_current = _size_band_counts(current.portfolio)
    size_band_prior = _size_band_counts(prior.portfolio)

    # Catalyst coverage
    cat_cov_current = _catalyst_coverage(current.rankings)
    cat_cov_prior = _catalyst_coverage(prior.rankings)
    top_cat_current = _top_catalysts(current.rankings)
    top_cat_prior = _top_catalysts(prior.rankings)

    # Risk
    risk_current = _count_risk_flags(current.portfolio)
    risk_prior = _count_risk_flags(prior.portfolio)

    # Guardrail warnings
    warnings = _compute_warnings(
        current, prior, name_turnover_pct, tier_counts_current, cat_cov_current
    )

    return DeltaResult(
        entrants=entrants,
        exits=exits,
        name_turnover_pct=round(name_turnover_pct, 1),
        weight_l1_delta=round(weight_l1, 1),
        tier_counts_current=tier_counts_current,
        tier_counts_prior=tier_counts_prior,
        tier_in_portfolio_current=tier_in_portfolio_current,
        tier_in_portfolio_prior=tier_in_portfolio_prior,
        size_band_current=size_band_current,
        size_band_prior=size_band_prior,
        catalyst_coverage_current=cat_cov_current,
        catalyst_coverage_prior=cat_cov_prior,
        top_catalysts_current=top_cat_current,
        top_catalysts_prior=top_cat_prior,
        risk_current=risk_current,
        risk_prior=risk_prior,
        warnings=warnings,
    )


def _compute_warnings(
    current: SnapshotData,
    prior: Optional[SnapshotData],
    name_turnover_pct: float,
    tier_counts: Dict[str, int],
    cat_cov: CatalystCoverage,
) -> List[str]:
    warnings = []
    if prior and current.ruleset_id != prior.ruleset_id:
        warnings.append(
            f"Ruleset changed: {prior.ruleset_id} -> {current.ruleset_id}"
        )
    if current.ruleset_id and current.ruleset_id != PHASE2_PINNED_RULESET_ID:
        warnings.append(
            f"Ruleset {current.ruleset_id} does not match pinned Phase-2 ID ({PHASE2_PINNED_RULESET_ID})"
        )
    if tier_counts.get("A", 0) < WARN_A_COUNT_MIN:
        warnings.append(
            f"A-tier count is {tier_counts.get('A', 0)} (minimum: {WARN_A_COUNT_MIN})"
        )
    if cat_cov.pct < WARN_CATALYST_COVERAGE_MIN:
        warnings.append(
            f"Catalyst specific_days coverage {cat_cov.pct}% < {WARN_CATALYST_COVERAGE_MIN}% threshold"
        )
    if name_turnover_pct > WARN_NAME_TURNOVER_PCT:
        warnings.append(
            f"Name turnover {name_turnover_pct}% > {WARN_NAME_TURNOVER_PCT}% threshold"
        )
    return warnings


def compute_single_snapshot_summary(snap: SnapshotData) -> SingleResult:
    """Produce summary metrics for a single snapshot (no prior)."""
    tier_counts = _tier_counts(snap.rankings)
    tier_port = _tier_in_portfolio(snap.portfolio)
    sb = _size_band_counts(snap.portfolio)
    cat_cov = _catalyst_coverage(snap.rankings)
    top_cat = _top_catalysts(snap.rankings)
    risk = _count_risk_flags(snap.portfolio)

    warnings = _compute_warnings(snap, None, 0.0, tier_counts, cat_cov)

    return SingleResult(
        tier_counts=tier_counts,
        tier_in_portfolio=tier_port,
        size_band=sb,
        catalyst_coverage=cat_cov,
        top_catalysts=top_cat,
        risk=risk,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Optionality diagnostic
# ---------------------------------------------------------------------------

def _optionality_diagnostic(rankings: pd.DataFrame, a_floor: float) -> dict:
    """Compute optionality coverage and A-candidate stats for dev-stage names."""
    dev = rankings[rankings["archetype"] == "drug_developer"]
    n_dev = len(dev)
    if n_dev == 0:
        return {
            "n_dev": 0,
            "n_with_optionality": 0,
            "dev_optionality_coverage_pct": 0.0,
            "dev_above_a_floor_count": 0,
            "a_floor": a_floor,
            "quantiles": {},
        }
    opts = pd.to_numeric(dev["clinical_optionality_pct_dev"], errors="coerce")
    n_with_opt = int(opts.notna().sum())
    coverage_pct = round(n_with_opt / n_dev * 100, 1)
    above_floor = int((opts >= a_floor).sum())
    quantiles = {} if opts.dropna().empty else {
        "p50": round(float(opts.quantile(0.5)), 4),
        "p75": round(float(opts.quantile(0.75)), 4),
        "p90": round(float(opts.quantile(0.9)), 4),
    }
    return {
        "n_dev": n_dev,
        "n_with_optionality": n_with_opt,
        "dev_optionality_coverage_pct": coverage_pct,
        "dev_above_a_floor_count": above_floor,
        "a_floor": a_floor,
        "quantiles": quantiles,
    }


# ---------------------------------------------------------------------------
# Exposure metrics
# ---------------------------------------------------------------------------

def compute_exposure_metrics(portfolio_df) -> Dict[str, Any]:
    """Compute portfolio-level exposure concentrations.

    Args:
        portfolio_df: DataFrame with columns target_weight_pct, catalyst_days,
                      risk_flags, mom_state.

    Returns:
        Flat dict of exposure metric values (all floats/ints).
    """
    if portfolio_df is None or len(portfolio_df) == 0:
        return {
            "catalyst_le_7d_weight_pct": 0.0,
            "catalyst_le_7d_count": 0,
            "catalyst_le_30d_weight_pct": 0.0,
            "high_vol_weight_pct": 0.0,
            "high_beta_weight_pct": 0.0,
            "high_vol_or_beta_weight_pct": 0.0,
            "drawdown_rel_xbi_weight_pct": 0.0,
            "overbought_rsi_weight_pct": 0.0,
            "catalyst_le_7d_and_drawdown_weight_pct": 0.0,
            "headwind_weight_pct": 0.0,
            "top5_weight_pct": 0.0,
            "max_single_weight_pct": 0.0,
        }

    cat7_w = 0.0
    cat7_n = 0
    cat30_w = 0.0
    hvol_w = 0.0
    hbeta_w = 0.0
    hvol_or_beta_w = 0.0
    dd_rel_w = 0.0
    rsi_w = 0.0
    stacked_w = 0.0
    headwind_w = 0.0
    weights = []

    for _, row in portfolio_df.iterrows():
        w = float(row.get("target_weight_pct") or 0)
        weights.append(w)

        # Parse catalyst_days
        cat_raw = row.get("catalyst_days")
        cat_days = None
        if cat_raw is not None and str(cat_raw).strip() not in ("", "nan"):
            try:
                cat_days = int(float(str(cat_raw)))
            except (ValueError, TypeError):
                pass

        is_cat7 = cat_days is not None and cat_days <= 7
        is_cat30 = cat_days is not None and cat_days <= 30

        if is_cat7:
            cat7_w += w
            cat7_n += 1
        if is_cat30:
            cat30_w += w

        # Parse risk_flags (pipe-separated)
        rf_raw = str(row.get("risk_flags") or "")
        rf_set = set(rf_raw.split("|")) if rf_raw.strip() else set()

        has_hvol = "high_vol" in rf_set
        has_hbeta = "high_beta" in rf_set
        has_dd_rel = "deep_drawdown_rel_xbi" in rf_set
        has_rsi = "overbought_rsi" in rf_set

        if has_hvol:
            hvol_w += w
        if has_hbeta:
            hbeta_w += w
        if has_hvol or has_hbeta:
            hvol_or_beta_w += w
        if has_dd_rel:
            dd_rel_w += w
        if has_rsi:
            rsi_w += w

        # Stacked: catalyst<=7d AND drawdown
        if is_cat7 and has_dd_rel:
            stacked_w += w

        # Momentum headwind
        mom = str(row.get("mom_state") or "").lower()
        if mom == "headwind":
            headwind_w += w

    # Top-5 concentration
    weights_sorted = sorted(weights, reverse=True)
    top5_w = sum(weights_sorted[:5])
    max_w = weights_sorted[0] if weights_sorted else 0.0

    return {
        "catalyst_le_7d_weight_pct": round(cat7_w, 2),
        "catalyst_le_7d_count": cat7_n,
        "catalyst_le_30d_weight_pct": round(cat30_w, 2),
        "high_vol_weight_pct": round(hvol_w, 2),
        "high_beta_weight_pct": round(hbeta_w, 2),
        "high_vol_or_beta_weight_pct": round(hvol_or_beta_w, 2),
        "drawdown_rel_xbi_weight_pct": round(dd_rel_w, 2),
        "overbought_rsi_weight_pct": round(rsi_w, 2),
        "catalyst_le_7d_and_drawdown_weight_pct": round(stacked_w, 2),
        "headwind_weight_pct": round(headwind_w, 2),
        "top5_weight_pct": round(top5_w, 2),
        "max_single_weight_pct": round(max_w, 2),
    }


# Exposure check table: (metric_key, warn_threshold_attr, fail_threshold_attr, reason_tag)
_EXPOSURE_CHECKS = [
    ("catalyst_le_7d_weight_pct", "warn_catalyst_le_7d_weight_pct", "fail_catalyst_le_7d_weight_pct", "catalyst_7d_weight_high"),
    ("catalyst_le_7d_count", "warn_catalyst_le_7d_count", "fail_catalyst_le_7d_count", "catalyst_7d_count_high"),
    ("high_vol_weight_pct", "warn_high_vol_weight_pct", "fail_high_vol_weight_pct", "high_vol_exposure"),
    ("high_beta_weight_pct", "warn_high_beta_weight_pct", "fail_high_beta_weight_pct", "high_beta_exposure"),
    ("high_vol_or_beta_weight_pct", "warn_high_vol_or_beta_weight_pct", "fail_high_vol_or_beta_weight_pct", "high_vol_or_beta_exposure"),
    ("drawdown_rel_xbi_weight_pct", "warn_drawdown_rel_xbi_weight_pct", "fail_drawdown_rel_xbi_weight_pct", "drawdown_rel_xbi_exposure"),
    ("overbought_rsi_weight_pct", "warn_overbought_rsi_weight_pct", "fail_overbought_rsi_weight_pct", "overbought_rsi_exposure"),
    ("catalyst_le_7d_and_drawdown_weight_pct", "warn_catalyst_le_7d_and_drawdown_weight_pct", "fail_catalyst_le_7d_and_drawdown_weight_pct", "stacked_catalyst_drawdown"),
    ("headwind_weight_pct", "warn_headwind_weight_pct", "fail_headwind_weight_pct", "headwind_exposure"),
    ("top5_weight_pct", "warn_top5_weight_pct", "fail_top5_weight_pct", "top5_concentration"),
]


# ---------------------------------------------------------------------------
# Health gate
# ---------------------------------------------------------------------------

def compute_health_gate(
    current: SnapshotData,
    prior: Optional[SnapshotData],
    result,  # DeltaResult | SingleResult
    thresholds: Optional[Phase2HealthThresholds] = None,
    expected_ruleset_id: Optional[str] = None,
) -> HealthResult:
    """Evaluate FAIL / WARN / OK health status for the snapshot.

    When *expected_ruleset_id* is provided (e.g. from an explicit ``--ruleset``
    override), the mismatch check compares against that id instead of the
    module-level ``PHASE2_PINNED_RULESET_ID``.  This prevents spurious
    ``ruleset_mismatch`` FAILs when intentionally testing a candidate.
    """
    th = thresholds or DEFAULT_HEALTH_THRESHOLDS
    is_delta = isinstance(result, DeltaResult)
    fail_reasons: List[str] = []
    warn_reasons: List[str] = []
    opt_diag = None  # populated lazily when A-count is below threshold

    tier_counts = result.tier_counts_current if is_delta else result.tier_counts
    cat_cov = result.catalyst_coverage_current if is_delta else result.catalyst_coverage

    # --- FAIL conditions (checked first) ---
    _expected_id = expected_ruleset_id or PHASE2_PINNED_RULESET_ID
    if current.ruleset_id != _expected_id:
        fail_reasons.append("ruleset_mismatch")

    if len(current.portfolio) == 0:
        fail_reasons.append("no_portfolio")

    if tier_counts.get("A", 0) < th.fail_a_count_min:
        opt_diag = _optionality_diagnostic(current.rankings, a_floor=PHASE2_A_FLOOR)
        if opt_diag["dev_optionality_coverage_pct"] < th.fail_optionality_coverage_min:
            fail_reasons.append("optionality_broken")
        else:
            # Coverage is fine but nothing clears all A requirements — legit regime
            warn_reasons.append("no_a_tier_regime")

    if cat_cov.pct < th.fail_catalyst_coverage_min:
        fail_reasons.append("catalyst_broken")

    # --- WARN conditions (only meaningful in delta mode) ---
    if is_delta:
        if result.name_turnover_pct > th.warn_turnover_pct:
            warn_reasons.append("high_turnover")

        if result.weight_l1_delta > th.warn_weight_l1_pct:
            warn_reasons.append("high_weight_churn")

        if prior is not None:
            cat_cov_prior = result.catalyst_coverage_prior
            cat_drop = cat_cov_prior.pct - cat_cov.pct
            if cat_drop > th.warn_catalyst_drop_pp:
                warn_reasons.append("catalyst_drop")

    # low_a_tier: above zero but below WARN threshold
    # (when a_count == 0 with good coverage, no_a_tier_regime already fires)
    a_count = tier_counts.get("A", 0)
    if a_count > 0 and a_count < th.warn_a_count_low:
        warn_reasons.append("low_a_tier")

    # --- Resolve status ---
    if fail_reasons:
        status = "FAIL"
        reasons = fail_reasons
    elif warn_reasons:
        status = "WARN"
        reasons = warn_reasons
    else:
        status = "OK"
        reasons = []

    metrics = {
        "a_count": a_count,
        "catalyst_coverage_pct": cat_cov.pct,
        "portfolio_size": len(current.portfolio),
        "ruleset_id": current.ruleset_id,
        "thresholds_id": th.thresholds_id,
    }
    if is_delta:
        metrics["name_turnover_pct"] = result.name_turnover_pct
        metrics["weight_l1_delta"] = result.weight_l1_delta

    # Attach optionality diagnostic when computed (A-count below threshold)
    if opt_diag is not None:
        metrics["dev_optionality_coverage_pct"] = opt_diag["dev_optionality_coverage_pct"]
        metrics["dev_above_a_floor_count"] = opt_diag["dev_above_a_floor_count"]
        metrics["optionality_diagnostic"] = opt_diag

    # Component coverage (dev-stage) — guarded by column existence
    if "missing_components" in current.rankings.columns:
        dev_mask = current.rankings.get("archetype", pd.Series(dtype=str)) == "drug_developer"
        dev_rows = current.rankings[dev_mask] if dev_mask.any() else current.rankings
        n_dev = len(dev_rows)
        if n_dev > 0:
            for comp in ("catalyst", "sponsor", "drawdown"):
                n_miss = dev_rows["missing_components"].apply(
                    lambda x, c=comp: pd.notna(x) and c in str(x)
                ).sum()
                metrics[f"coverage_{comp}_pct"] = round(100 * (1 - n_miss / n_dev), 1)
        # Portfolio missing count
        port_tickers = set(current.portfolio["ticker"])
        port_mask = current.rankings["ticker"].isin(port_tickers)
        port_mc = current.rankings.loc[port_mask, "missing_components"]
        metrics["portfolio_missing_count"] = int(
            port_mc.apply(lambda x: pd.notna(x) and bool(str(x).strip())).sum()
        )

    # --- Missingness guardrails (guarded by column existence) ---
    if "missing_components" in current.rankings.columns:
        dd_cov = metrics.get("coverage_drawdown_pct")
        sp_cov = metrics.get("coverage_sponsor_pct")
        ct_cov = metrics.get("coverage_catalyst_pct")
        pm_cnt = metrics.get("portfolio_missing_count", 0)

        if dd_cov is not None and dd_cov < th.fail_drawdown_coverage_min:
            fail_reasons.append("drawdown_coverage_low")
        if dd_cov is not None and dd_cov < th.warn_drawdown_coverage_min:
            warn_reasons.append("drawdown_coverage_low")
        if sp_cov is not None and sp_cov < th.warn_sponsor_coverage_min:
            warn_reasons.append("sponsor_coverage_low")
        if ct_cov is not None and ct_cov < th.warn_catalyst_comp_coverage_min:
            warn_reasons.append("catalyst_coverage_low")
        if pm_cnt > 0:
            warn_reasons.append("portfolio_missing_data")

        # Re-resolve status after missingness checks
        if fail_reasons:
            status = "FAIL"
            reasons = fail_reasons
        elif warn_reasons:
            status = "WARN"
            reasons = warn_reasons

    # --- Coinvest coverage gate (WARN-only, non-blocking) ---
    # Uses sponsor_tier1_count column if available
    if "sponsor_tier1_count" in current.rankings.columns:
        _t1 = current.rankings["sponsor_tier1_count"]
        _n_total = len(_t1)
        if _n_total > 0:
            _n_with = int(_t1.apply(
                lambda x: pd.notna(x) and str(x).strip() not in ("", "0")
            ).sum())
            _coinvest_cov = 100.0 * _n_with / _n_total
            metrics["coinvest_coverage_pct"] = round(_coinvest_cov, 1)
            if _coinvest_cov < th.warn_coinvest_coverage_min:
                warn_reasons.append("coinvest_coverage_low")
                if not fail_reasons:
                    status = "WARN"
                    reasons = warn_reasons

    # --- Exposure guardrails (portfolio concentration) ---
    if len(current.portfolio) > 0:
        exposure = compute_exposure_metrics(current.portfolio)
        metrics["exposure"] = exposure

        for metric_key, warn_key, fail_key, reason in _EXPOSURE_CHECKS:
            val = exposure[metric_key]
            if val >= getattr(th, fail_key):
                fail_reasons.append(reason)
            elif val >= getattr(th, warn_key):
                warn_reasons.append(reason)

        # Re-resolve status after exposure checks
        if fail_reasons:
            status = "FAIL"
            reasons = fail_reasons
        elif warn_reasons:
            status = "WARN"
            reasons = warn_reasons

    return HealthResult(status=status, reasons=reasons, metrics=metrics)


def generate_health_json(
    health: HealthResult,
    thresholds: Optional[Phase2HealthThresholds] = None,
) -> dict:
    """Produce a machine-readable dict for phase2_health.json."""
    th = thresholds or DEFAULT_HEALTH_THRESHOLDS
    return {
        "status": health.status,
        "reasons": health.reasons,
        "metrics": health.metrics,
        "thresholds_id": th.thresholds_id,
        "thresholds": {
            "fail_catalyst_coverage_min": th.fail_catalyst_coverage_min,
            "fail_a_count_min": th.fail_a_count_min,
            "fail_optionality_coverage_min": th.fail_optionality_coverage_min,
            "fail_drawdown_coverage_min": th.fail_drawdown_coverage_min,
            "warn_a_count_low": th.warn_a_count_low,
            "warn_turnover_pct": th.warn_turnover_pct,
            "warn_weight_l1_pct": th.warn_weight_l1_pct,
            "warn_catalyst_drop_pp": th.warn_catalyst_drop_pp,
            "warn_drawdown_coverage_min": th.warn_drawdown_coverage_min,
            "warn_sponsor_coverage_min": th.warn_sponsor_coverage_min,
            "warn_catalyst_comp_coverage_min": th.warn_catalyst_comp_coverage_min,
        },
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(
    current: SnapshotData,
    prior: Optional[SnapshotData],
    result,  # DeltaResult | SingleResult
    health: Optional[HealthResult] = None,
    churn_suppressed: bool = False,
) -> str:
    """Generate human-readable text report."""
    lines: List[str] = []
    W = 72

    lines.append("=" * W)
    lines.append("PHASE-2 RUN DELTA REPORT")
    lines.append("=" * W)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines.append(f"Generated: {now}")
    lines.append(f"Current:   {current.date} (ruleset {current.ruleset_id})")
    if prior:
        tag = "" if prior.has_native_portfolio else " [reconstructed portfolio]"
        lines.append(f"Prior:     {prior.date} (ruleset {prior.ruleset_id}){tag}")
    else:
        lines.append("Prior:     (none)")
    lines.append("")

    if churn_suppressed:
        lines.append("*** CACHE HEALTH: DEGRADED ***")
        lines.append("*** CHURN COMPARISONS: SUPPRESSED ***")
        lines.append("*** BASELINES: NOT UPDATED ***")
        lines.append("")

    # Health headline box
    if health is not None:
        if health.reasons:
            lines.append(f"HEALTH: {health.status} [{', '.join(health.reasons)}]")
        else:
            lines.append(f"HEALTH: {health.status}")
        m = health.metrics
        parts = []
        if "name_turnover_pct" in m:
            parts.append(f"turnover={m['name_turnover_pct']}%")
        parts.append(f"catalyst={m['catalyst_coverage_pct']}%")
        parts.append(f"A={m['a_count']}")
        parts.append(f"ruleset={m['ruleset_id']}")
        lines.append("  " + "  ".join(parts))
        lines.append("")

    is_delta = isinstance(result, DeltaResult)

    # Warnings
    if result.warnings:
        lines.append("WARNINGS:")
        for w in result.warnings:
            lines.append(f"  - {w}")
        lines.append("")

    # --- Section 1: Portfolio Turnover ---
    lines.append("1. PORTFOLIO TURNOVER")
    lines.append("-" * W)
    if is_delta:
        n_cur = len(current.portfolio)
        n_pri = len(prior.portfolio) if prior else 0
        lines.append(f"  Current portfolio: {n_cur} names")
        if prior:
            lines.append(f"  Prior portfolio:   {n_pri} names")
        _format_list(lines, "Entrants", result.entrants)
        _format_list(lines, "Exits", result.exits)
        lines.append(f"  Name turnover: {result.name_turnover_pct}%")
        lines.append(f"  Weight L1 delta: {result.weight_l1_delta}%")
    else:
        lines.append(f"  Portfolio: {len(current.portfolio)} names (no prior for comparison)")
    lines.append("")

    # --- Section 2: Tier Distribution ---
    lines.append("2. TIER DISTRIBUTION (dev-stage universe)")
    lines.append("-" * W)
    if is_delta:
        lines.append(f"  {'':12s} {'Current':>8s} {'Prior':>8s} {'Delta':>8s}")
        for tier in ["A", "B", "C", "D"]:
            c = result.tier_counts_current.get(tier, 0)
            p = result.tier_counts_prior.get(tier, 0)
            d = c - p
            sign = "+" if d > 0 else ""
            lines.append(f"  {tier:12s} {c:8d} {p:8d} {sign}{d:>7d}")
        lines.append("")
        lines.append("  In portfolio:")
        for tier in ["A", "B"]:
            c = result.tier_in_portfolio_current.get(tier, 0)
            p = result.tier_in_portfolio_prior.get(tier, 0)
            n_c = len(current.portfolio)
            n_p = len(prior.portfolio) if prior else 0
            pct_c = (c / n_c * 100) if n_c else 0
            pct_p = (p / n_p * 100) if n_p else 0
            lines.append(
                f"    {tier}: {c} ({pct_c:.0f}%)   Prior: {p} ({pct_p:.0f}%)"
            )
    else:
        lines.append(f"  {'':12s} {'Current':>8s}")
        for tier in ["A", "B", "C", "D"]:
            c = result.tier_counts.get(tier, 0)
            lines.append(f"  {tier:12s} {c:8d}")
        lines.append("")
        lines.append("  In portfolio:")
        for tier in ["A", "B"]:
            c = result.tier_in_portfolio.get(tier, 0)
            n_c = len(current.portfolio)
            pct_c = (c / n_c * 100) if n_c else 0
            lines.append(f"    {tier}: {c} ({pct_c:.0f}%)")
    lines.append("")

    # --- Optionality diagnostic (when A-count is low) ---
    if health is not None and health.metrics.get("optionality_diagnostic"):
        od = health.metrics["optionality_diagnostic"]
        lines.append("OPTIONALITY DIAGNOSTIC")
        lines.append("-" * W)
        lines.append(f"  Dev-stage names:       {od['n_dev']}")
        lines.append(
            f"  With optionality data: {od['n_with_optionality']}  ({od['dev_optionality_coverage_pct']}%)"
        )
        lines.append(f"  Above a_floor ({od['a_floor']}):   {od['dev_above_a_floor_count']}")
        if od.get("quantiles"):
            q = od["quantiles"]
            lines.append(
                f"  Quantiles: P50={q.get('p50', 'N/A')}  P75={q.get('p75', 'N/A')}  P90={q.get('p90', 'N/A')}"
            )
        lines.append("")

    # --- Section 3: Sizing ---
    lines.append("3. SIZING (portfolio)")
    lines.append("-" * W)
    if is_delta:
        lines.append(f"  {'':12s} {'Current':>8s} {'Prior':>8s} {'Delta':>8s}")
        for band in ["L", "M", "S", "XS"]:
            c = result.size_band_current.get(band, 0)
            p = result.size_band_prior.get(band, 0)
            d = c - p
            sign = "+" if d > 0 else ""
            lines.append(f"  {band:12s} {c:8d} {p:8d} {sign}{d:>7d}")
    else:
        lines.append(f"  {'':12s} {'Current':>8s}")
        for band in ["L", "M", "S", "XS"]:
            c = result.size_band.get(band, 0)
            lines.append(f"  {band:12s} {c:8d}")
    lines.append("")

    # --- Section 4: Catalyst Coverage ---
    lines.append("4. CATALYST COVERAGE (dev-stage)")
    lines.append("-" * W)
    if is_delta:
        cc = result.catalyst_coverage_current
        cp = result.catalyst_coverage_prior
        lines.append(f"  specific_days:  {cc.pct}%  (Prior: {cp.pct}%)")
        if cc.backfill_share_pct > 0:
            lines.append(f"  backfill_share:  {cc.backfill_share_pct}%")
        lines.append("  Mode distribution (current):")
        for mode, count in sorted(cc.mode_dist.items()):
            lines.append(f"    {mode}: {count}")
        lines.append("")
        lines.append("  Nearest 10 catalysts (current):")
        _format_catalysts(lines, result.top_catalysts_current)
    else:
        cc = result.catalyst_coverage
        lines.append(f"  specific_days:  {cc.pct}%")
        if cc.backfill_share_pct > 0:
            lines.append(f"  backfill_share:  {cc.backfill_share_pct}%")
        lines.append("  Mode distribution:")
        for mode, count in sorted(cc.mode_dist.items()):
            lines.append(f"    {mode}: {count}")
        lines.append("")
        lines.append("  Nearest 10 catalysts:")
        _format_catalysts(lines, result.top_catalysts)
    lines.append("")

    # --- Section 5: Risk Sanity ---
    lines.append("5. RISK SANITY (portfolio)")
    lines.append("-" * W)
    if is_delta:
        lines.append(f"  {'Flag':20s} {'Current':>8s} {'Prior':>8s}")
        for flag in RISK_FLAGS_ALL:
            c = result.risk_current.get(flag, 0)
            p = result.risk_prior.get(flag, 0)
            lines.append(f"  {flag:20s} {c:8d} {p:8d}")
    else:
        lines.append(f"  {'Flag':20s} {'Current':>8s}")
        for flag in RISK_FLAGS_ALL:
            c = result.risk.get(flag, 0)
            lines.append(f"  {flag:20s} {c:8d}")
    lines.append("")

    # --- Section 6: Guardrails ---
    lines.append("6. GUARDRAILS")
    lines.append("-" * W)
    _format_guardrails(lines, current, result)
    lines.append("")

    return "\n".join(lines)


def _format_list(lines: List[str], label: str, items: List[str], max_show: int = 10):
    """Format a list of tickers for the report."""
    n = len(items)
    if n == 0:
        lines.append(f"  {label} (0):  (none)")
    elif n <= max_show:
        lines.append(f"  {label} ({n}):  {', '.join(items)}")
    else:
        shown = ", ".join(items[:max_show])
        lines.append(f"  {label} ({n}):  {shown}, ... (+{n - max_show} more)")


def _format_catalysts(lines: List[str], catalysts: List[Dict]):
    if not catalysts:
        lines.append("    (none)")
        return
    # Two-column layout
    for i in range(0, len(catalysts), 2):
        parts = []
        for j in range(2):
            if i + j < len(catalysts):
                c = catalysts[i + j]
                parts.append(f"  {c['ticker']:8s} {c['catalyst_days']:3d}d  tier={c['tier_dev']}")
        lines.append("  " + "    ".join(parts))


def _format_guardrails(lines: List[str], current: SnapshotData, result):
    """Format guardrail checks."""
    is_delta = isinstance(result, DeltaResult)

    # Ruleset check
    if current.ruleset_id == PHASE2_PINNED_RULESET_ID:
        lines.append(f"  [PASS] Ruleset matches pinned Phase-2 ID ({PHASE2_PINNED_RULESET_ID})")
    elif current.ruleset_id:
        lines.append(
            f"  [WARN] Ruleset {current.ruleset_id} != pinned {PHASE2_PINNED_RULESET_ID}"
        )
    else:
        lines.append("  [WARN] No ruleset ID found")

    # A-tier check
    tier_counts = result.tier_counts_current if is_delta else result.tier_counts
    a_count = tier_counts.get("A", 0)
    if a_count >= WARN_A_COUNT_MIN:
        lines.append(f"  [PASS] A-tier count > 0 ({a_count})")
    else:
        lines.append(f"  [WARN] A-tier count is {a_count}")

    # Catalyst coverage
    cat_cov = result.catalyst_coverage_current if is_delta else result.catalyst_coverage
    if cat_cov.pct >= WARN_CATALYST_COVERAGE_MIN:
        lines.append(
            f"  [PASS] Catalyst specific_days coverage > {WARN_CATALYST_COVERAGE_MIN}% ({cat_cov.pct}%)"
        )
    else:
        lines.append(
            f"  [WARN] Catalyst specific_days coverage {cat_cov.pct}% < {WARN_CATALYST_COVERAGE_MIN}%"
        )

    # Turnover (only in delta mode)
    if is_delta:
        if result.name_turnover_pct <= WARN_NAME_TURNOVER_PCT:
            lines.append(
                f"  [PASS] Name turnover {result.name_turnover_pct}% <= {WARN_NAME_TURNOVER_PCT}%"
            )
        else:
            lines.append(
                f"  [WARN] Name turnover {result.name_turnover_pct}% > {WARN_NAME_TURNOVER_PCT}%"
            )


# ---------------------------------------------------------------------------
# JSON details
# ---------------------------------------------------------------------------

def generate_details_json(
    current: SnapshotData,
    prior: Optional[SnapshotData],
    result,
    churn_suppressed: bool = False,
) -> dict:
    """Generate machine-readable JSON output."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    is_delta = isinstance(result, DeltaResult)

    out: dict = {
        "generated_at": now,
        "current": {
            "date": current.date,
            "ruleset_id": current.ruleset_id,
            "has_native_portfolio": current.has_native_portfolio,
            "portfolio_size": len(current.portfolio),
        },
    }
    if prior:
        out["prior"] = {
            "date": prior.date,
            "ruleset_id": prior.ruleset_id,
            "has_native_portfolio": prior.has_native_portfolio,
            "portfolio_size": len(prior.portfolio),
        }
    else:
        out["prior"] = None

    if is_delta:
        out["portfolio_turnover"] = {
            "entrants": result.entrants,
            "exits": result.exits,
            "name_turnover_pct": result.name_turnover_pct,
            "weight_l1_delta": result.weight_l1_delta,
        }
        out["tier_distribution"] = {
            "current": result.tier_counts_current,
            "prior": result.tier_counts_prior,
        }
        out["tier_in_portfolio"] = {
            "current": result.tier_in_portfolio_current,
            "prior": result.tier_in_portfolio_prior,
        }
        out["sizing"] = {
            "current": result.size_band_current,
            "prior": result.size_band_prior,
        }
        out["catalyst_coverage"] = {
            "current": _cat_cov_dict(result.catalyst_coverage_current),
            "prior": _cat_cov_dict(result.catalyst_coverage_prior),
        }
        out["top_catalysts"] = {
            "current": result.top_catalysts_current,
            "prior": result.top_catalysts_prior,
        }
        out["risk_flags"] = {
            "current": result.risk_current,
            "prior": result.risk_prior,
        }
    else:
        out["tier_distribution"] = {"current": result.tier_counts}
        out["tier_in_portfolio"] = {"current": result.tier_in_portfolio}
        out["sizing"] = {"current": result.size_band}
        out["catalyst_coverage"] = {"current": _cat_cov_dict(result.catalyst_coverage)}
        out["top_catalysts"] = {"current": result.top_catalysts}
        out["risk_flags"] = {"current": result.risk}

    out["churn_suppressed"] = churn_suppressed
    if churn_suppressed:
        out["churn_suppressed_reason"] = "cache_degraded"

    # Guardrails
    passed = []
    warnings = []
    if current.ruleset_id == PHASE2_PINNED_RULESET_ID:
        passed.append(f"Ruleset matches pinned Phase-2 ID ({PHASE2_PINNED_RULESET_ID})")
    elif current.ruleset_id:
        warnings.append(f"Ruleset {current.ruleset_id} != pinned {PHASE2_PINNED_RULESET_ID}")

    tier_counts = result.tier_counts_current if is_delta else result.tier_counts
    a_count = tier_counts.get("A", 0)
    if a_count >= WARN_A_COUNT_MIN:
        passed.append(f"A-tier count >= {WARN_A_COUNT_MIN} ({a_count})")
    else:
        warnings.append(f"A-tier count is {a_count}")

    cat_cov = result.catalyst_coverage_current if is_delta else result.catalyst_coverage
    if cat_cov.pct >= WARN_CATALYST_COVERAGE_MIN:
        passed.append(f"Catalyst coverage {cat_cov.pct}% >= {WARN_CATALYST_COVERAGE_MIN}%")
    else:
        warnings.append(f"Catalyst coverage {cat_cov.pct}% < {WARN_CATALYST_COVERAGE_MIN}%")

    if is_delta:
        if result.name_turnover_pct <= WARN_NAME_TURNOVER_PCT:
            passed.append(f"Name turnover {result.name_turnover_pct}% <= {WARN_NAME_TURNOVER_PCT}%")
        else:
            warnings.append(f"Name turnover {result.name_turnover_pct}% > {WARN_NAME_TURNOVER_PCT}%")

    out["guardrails"] = {
        "passed": passed,
        "warnings": warnings + result.warnings,
    }

    return out


def _cat_cov_dict(cc: CatalystCoverage) -> dict:
    return {
        "n_dev": cc.n_dev,
        "n_specific": cc.n_specific,
        "pct": cc.pct,
        "mode_dist": cc.mode_dist,
        "backfill_share_pct": cc.backfill_share_pct,
    }


# ---------------------------------------------------------------------------
# Delta CSV
# ---------------------------------------------------------------------------

def generate_delta_csv(
    current: SnapshotData,
    prior: Optional[SnapshotData],
    output_path: Path,
):
    """Write per-ticker delta CSV (union of both portfolios)."""
    cur_map = {}
    for _, row in current.portfolio.iterrows():
        cur_map[row["ticker"]] = row

    pri_map = {}
    if prior is not None:
        for _, row in prior.portfolio.iterrows():
            pri_map[row["ticker"]] = row

    all_tickers = sorted(set(cur_map.keys()) | set(pri_map.keys()))

    fieldnames = [
        "ticker",
        "in_current",
        "in_prior",
        "weight_current",
        "weight_prior",
        "weight_delta",
        "tier_current",
        "tier_prior",
        "rank_current",
        "rank_prior",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ticker in all_tickers:
            in_cur = ticker in cur_map
            in_pri = ticker in pri_map
            w_c = _safe_float(cur_map[ticker].get("target_weight_pct")) if in_cur else 0.0
            w_p = _safe_float(pri_map[ticker].get("target_weight_pct")) if in_pri else 0.0
            writer.writerow({
                "ticker": ticker,
                "in_current": 1 if in_cur else 0,
                "in_prior": 1 if in_pri else 0,
                "weight_current": round(w_c, 2),
                "weight_prior": round(w_p, 2),
                "weight_delta": round(w_c - w_p, 2),
                "tier_current": str(cur_map[ticker].get("tier_dev", "")).strip() if in_cur else "",
                "tier_prior": str(pri_map[ticker].get("tier_dev", "")).strip() if in_pri else "",
                "rank_current": _safe_int(cur_map[ticker].get("actionable_rank")) if in_cur else "",
                "rank_prior": _safe_int(pri_map[ticker].get("actionable_rank")) if in_pri else "",
            })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Phase-2 Snapshot Delta Report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--snapshot-dir",
        default="data/snapshots",
        help="Base directory for snapshots (default: data/snapshots)",
    )
    parser.add_argument(
        "--current",
        default=None,
        help="Current snapshot date (default: newest)",
    )
    parser.add_argument(
        "--prior",
        default=None,
        help='Prior snapshot date (default: auto-find; "none" to skip)',
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Output directory (default: output)",
    )
    parser.add_argument(
        "--health-thresholds",
        default=None,
        help="Path to Phase2HealthThresholds JSON (default: built-in defaults)",
    )
    parser.add_argument(
        "--expected-ruleset-id",
        default=None,
        help="Override expected ruleset ID for health gate (skips pinned mismatch check)",
    )
    args = parser.parse_args()

    snapshot_dir = Path(args.snapshot_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load health thresholds
    if args.health_thresholds:
        ht_path = Path(args.health_thresholds)
        if not ht_path.exists():
            print(f"ERROR: Health thresholds not found: {ht_path}", file=sys.stderr)
            sys.exit(1)
        health_thresholds = Phase2HealthThresholds.from_json(str(ht_path))
        print(f"Health thresholds: {ht_path.name} (id={health_thresholds.thresholds_id})")
    else:
        health_thresholds = DEFAULT_HEALTH_THRESHOLDS
        print(f"Health thresholds: defaults (id={health_thresholds.thresholds_id})")

    # Find snapshots
    current_path, prior_path = find_snapshots(
        snapshot_dir, args.current, args.prior
    )
    if current_path is None:
        print("ERROR: No snapshot found.", file=sys.stderr)
        sys.exit(1)

    print(f"Current snapshot: {current_path.name}")

    # Load current
    current = load_snapshot(current_path)
    if current is None:
        print(f"ERROR: Could not load snapshot at {current_path}", file=sys.stderr)
        sys.exit(1)

    # Load prior (may be None)
    prior = None
    if prior_path is not None:
        print(f"Prior snapshot:   {prior_path.name}")
        prior = load_snapshot(prior_path)
        if prior is None:
            print(
                f"WARNING: Prior snapshot {prior_path.name} lacks tier_dev, running single-snapshot mode",
                file=sys.stderr,
            )
    else:
        print("Prior snapshot:   (none)")

    # Compute
    if prior is not None:
        result = compute_delta(current, prior)
    else:
        result = compute_single_snapshot_summary(current)

    # Health gate
    health = compute_health_gate(
        current, prior, result,
        thresholds=health_thresholds,
        expected_ruleset_id=args.expected_ruleset_id,
    )
    health_json = generate_health_json(health, thresholds=health_thresholds)

    # Generate outputs
    _churn_suppressed = is_snapshot_degraded(current_path)
    report_txt = generate_report(current, prior, result, health=health, churn_suppressed=_churn_suppressed)
    details = generate_details_json(current, prior, result, churn_suppressed=_churn_suppressed)

    # Write artifacts
    report_path = output_dir / "phase2_run_delta_report.txt"
    details_path = output_dir / "phase2_run_delta_details.json"
    csv_path = output_dir / "phase2_run_delta.csv"
    health_path = output_dir / "phase2_health.json"

    with open(report_path, "w") as f:
        f.write(report_txt)
    print(f"  -> {report_path}")

    with open(details_path, "w") as f:
        json.dump(details, f, indent=2)
    print(f"  -> {details_path}")

    generate_delta_csv(current, prior, csv_path)
    print(f"  -> {csv_path}")

    with open(health_path, "w") as f:
        json.dump(health_json, f, indent=2)
    print(f"  -> {health_path}")

    # Print health headline
    if health.reasons:
        headline = f"PHASE2 HEALTH: {health.status} [{', '.join(health.reasons)}]"
    else:
        headline = f"PHASE2 HEALTH: {health.status}"
    print(headline)

    # Print report to stdout
    print()
    print(report_txt)


if __name__ == "__main__":
    main()
