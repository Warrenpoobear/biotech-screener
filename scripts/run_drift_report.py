#!/usr/bin/env python3
"""Daily Drift Report — post-promotion monitoring for the active ruleset.

Loads the last N snapshots from the snapshot directory, computes per-snapshot
drift metrics (tier distribution, catalyst coverage, top-25 overlap, score
dispersion), evaluates hard guardrails, and produces JSON + Markdown reports.

Alternatively, accepts a walkforward panel CSV (``--panel``) to replay
historical drift metrics across archived dates.

Usage:
    python scripts/run_drift_report.py \
        --snapshot-dir data/snapshots \
        --output-dir output \
        [--window-size 5] \
        [--guardrails path/to/guardrails.json]

    python scripts/run_drift_report.py \
        --panel artifacts/walkforward_panel.csv \
        --output-dir output \
        [--window-size 10]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
from dataclasses import asdict, dataclass
from dataclasses import fields as dc_fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Imports from existing code (snapshot loading + helpers)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from run_phase2_snapshot_delta import (
    SnapshotData,
    _catalyst_coverage,
    _safe_float,
    _safe_int,
    _tier_counts,
    load_snapshot,
)

# ---------------------------------------------------------------------------
# Guardrails dataclass
# ---------------------------------------------------------------------------
MANIFEST_PATH = (
    PROJECT_ROOT / "production_data" / "decision_rulesets" / "manifest.json"
)


@dataclass(frozen=True)
class DriftGuardrails:
    """Immutable, versioned guardrail thresholds for drift monitoring.

    FAIL triggers indicate an immediate rollback candidate — the tier
    structure or data feed has diverged far enough from expectations that
    the active ruleset may be mis-specified.
    """

    # FAIL triggers
    fail_a_pct_low: float = 0.0        # A-tier % among dev = 0% (pipeline broken)
    fail_a_pct_high: float = 25.0      # A-tier % among dev > 25%
    fail_catalyst_missing_high: float = 85.0   # catalyst missing % among eligible > 85%
    fail_overlap_low: float = 50.0     # top-25 overlap vs prior < 50%
    fail_dispersion_low: float = 0.10  # optionality std < 0.10

    # WARN triggers
    warn_a_pct_low: float = 1.5              # WARN if A-tier % < 1.5% (sparse catalyst regime)

    # Cost / eligibility / drawdown coverage
    warn_median_cost_bps_high: float = 60.0   # median round-trip cost > 60 bps
    warn_cost_coverage_low: float = 80.0      # WARN if cost coverage < 80%
    warn_cap_binding_high: float = 20.0       # WARN if cap-binding (floor bucket) > 20%
    warn_eligible_dev_pct_low: float = 10.0   # eligible dev % < 10% → WARN
    warn_drawdown_rel_coverage_low: float = 80.0  # WARN if XBI relative DD coverage < 80%
    fail_drawdown_rel_coverage_low: float = 50.0  # FAIL if XBI relative DD coverage < 50%

    # Rolling window size
    window_size: int = 5

    # Backfill telemetry
    warn_backfill_share_high: float = 60.0  # WARN if backfill-sourced catalyst > 60%

    # dd_rel_margin rescue telemetry
    warn_dd_rel_margin_rescue_share_high: float = 5.0  # WARN if rescued share > 5%

    # Returns-source mix
    warn_rs_unknown_share_high: float = 8.5        # WARN if unknown source > 8.5%
    warn_rs_csv_outlier_override_share_high: float = 1.0  # WARN if outlier overrides > 1%
    warn_rs_morningstar_share_low: float = 85.0    # WARN if Morningstar share < 85%

    # Catalyst coverage
    warn_cat_eligible_share_low: float = 95.0         # WARN if eligible % of dev < 95%
    warn_cat_specific_days_share_low: float = 40.0    # WARN if specific_days % of eligible < 40%

    # Catalyst source mix
    warn_cs_ctgov_share_low: float = 37.0             # WARN if CTGOV_CALENDAR share < 37%
    warn_cs_unknown_share_high: float = 5.0           # WARN if unknown source share > 5%

    # Catalyst event type mix (regime-gated: only when cat_specific_days ≥ 40%)
    warn_ct_unknown_share_high: float = 5.0           # WARN if unknown event type share > 5%
    warn_ct_fda_share_spike: float = 30.0             # WARN if FDA_DECISION + FDA_ADCOM share > 30%

    # Adaptive WARN layer
    warn_iqr_k: float = 2.0             # WARN if |delta| > k * max(IQR, floor)
    warn_iqr_floor: float = 1.0         # minimum IQR (prevents spurious WARN from flat windows)
    warn_min_window: int = 3            # minimum snapshots for adaptive WARN to activate
    fail_corroboration_count: int = 2   # FAIL metrics needed for ROLLBACK_RECOMMENDED

    @property
    def guardrails_id(self) -> str:
        """Deterministic 8-char SHA256 of all threshold fields."""
        parts = []
        for f in dc_fields(self):
            parts.append(f"{f.name}={getattr(self, f.name)}")
        blob = "|".join(parts).encode()
        return hashlib.sha256(blob).hexdigest()[:8]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["guardrails_id"] = self.guardrails_id
        return d

    @classmethod
    def from_json(cls, path: str) -> "DriftGuardrails":
        with open(path) as f:
            data = json.load(f)
        data.pop("guardrails_id", None)
        valid_fields = {f.name for f in dc_fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
            f.write("\n")


# ---------------------------------------------------------------------------
# Snapshot window loading
# ---------------------------------------------------------------------------
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def load_snapshot_window(
    snap_dir: Path, window_size: int
) -> List[SnapshotData]:
    """Load the last *window_size* snapshots sorted by date (ascending).

    Only considers directories whose name matches YYYY-MM-DD and that
    contain a loadable snapshot (rankings.csv with tier_dev).
    """
    if not snap_dir.is_dir():
        return []

    date_dirs = sorted(
        [d for d in snap_dir.iterdir() if d.is_dir() and _DATE_RE.match(d.name)],
        key=lambda d: d.name,
    )

    # Take the last N directories
    candidates = date_dirs[-window_size:] if window_size > 0 else date_dirs

    snapshots: List[SnapshotData] = []
    for d in candidates:
        snap = load_snapshot(d)
        if snap is not None:
            snapshots.append(snap)

    return snapshots


# ---------------------------------------------------------------------------
# Panel-as-snapshots loader
# ---------------------------------------------------------------------------

# Column mapping: panel name → snapshot name expected by metrics helpers.
_PANEL_COL_MAP = {
    "tier": "tier_dev",
    "optionality": "clinical_optionality_pct_dev",
    "band": "size_band",
    "drawdown_rel_xbi": "de_drawdown_rel_xbi",
}


def _synthesize_actionable_rank(df: pd.DataFrame) -> pd.Series:
    """Derive actionable_rank from weight (higher weight → lower rank).

    Rows with weight > 0 are ranked 1..N by descending weight.
    Rows with weight = 0 or missing get rank 9999.
    """
    weights = df["weight"].apply(lambda v: _safe_float(v, 0.0))
    ranks = pd.Series(9999, index=df.index, dtype=int)
    mask = weights > 0
    if mask.any():
        # Rank descending: highest weight = rank 1
        ranked = weights[mask].rank(ascending=False, method="min").astype(int)
        ranks.loc[mask] = ranked
    return ranks


def load_panel_as_snapshots(
    panel_path: Path, window_size: int = 0,
) -> List[SnapshotData]:
    """Load a walkforward panel CSV and convert to SnapshotData objects.

    The panel has one row per (date, ticker).  Column names differ from
    snapshot rankings — this function remaps them so all downstream drift
    metrics work unchanged.

    Args:
        panel_path: Path to the panel CSV (date column required — accepts
            ``as_of_date``, ``snapshot_date``, or ``date``).
        window_size: If > 0, keep only the last *window_size* dates.

    Returns:
        List of SnapshotData, sorted by date ascending.
    """
    df = pd.read_csv(panel_path, dtype=str)

    # Flexible date column detection
    date_col = None
    for candidate in ("as_of_date", "snapshot_date", "date"):
        if candidate in df.columns:
            date_col = candidate
            break
    if date_col is None:
        raise ValueError(
            f"Panel CSV missing date column (expected one of: "
            f"as_of_date, snapshot_date, date): {panel_path}"
        )
    if date_col != "as_of_date":
        df = df.rename(columns={date_col: "as_of_date"})

    # Rename columns to match snapshot expectations
    df = df.rename(columns=_PANEL_COL_MAP)

    # Panel is dev-only — synthesize archetype
    df["archetype"] = "drug_developer"

    # Synthesize actionable_rank from weight (for top-25 overlap)
    if "weight" in df.columns and "actionable_rank" not in df.columns:
        df["actionable_rank"] = _synthesize_actionable_rank(df)

    # Group by date, build one SnapshotData per date
    dates = sorted(df["as_of_date"].unique())
    if window_size > 0:
        dates = dates[-window_size:]

    snapshots: List[SnapshotData] = []
    for date_str in dates:
        date_df = df[df["as_of_date"] == date_str].copy()

        # Drop the date column — real snapshot rankings.csv never has it
        date_df.drop(columns=["as_of_date"], inplace=True, errors="ignore")

        # Determine ruleset_id (first non-blank value)
        ruleset_id = ""
        if "ruleset_id" in date_df.columns:
            ids = [
                str(v).strip()
                for v in date_df["ruleset_id"].dropna().unique()
                if str(v).strip()
            ]
            if ids:
                ruleset_id = ids[0]

        snap = SnapshotData(
            date=date_str,
            path=Path("<panel>"),
            rankings=date_df,
            portfolio=pd.DataFrame(),
            metadata={},
            ruleset_id=ruleset_id,
            has_native_portfolio=False,
        )
        snapshots.append(snap)

    return snapshots


# ---------------------------------------------------------------------------
# Drift metrics computation
# ---------------------------------------------------------------------------
def _top_n_tickers(rankings: pd.DataFrame, n: int = 25) -> set:
    """Return the top-N tickers by actionable rank (dev-stage only)."""
    dev = rankings[rankings["archetype"] == "drug_developer"].copy()
    if "actionable_rank" not in dev.columns:
        return set()
    dev["_arank"] = dev["actionable_rank"].apply(lambda v: _safe_int(v, 9999))
    dev = dev.sort_values("_arank").head(n)
    return set(dev["ticker"])


def _optionality_std(rankings: pd.DataFrame) -> Optional[float]:
    """Standard deviation of clinical_optionality_pct_dev among dev tickers."""
    dev = rankings[rankings["archetype"] == "drug_developer"]
    if "clinical_optionality_pct_dev" not in dev.columns:
        return None
    vals = [
        _safe_float(v, None)
        for v in dev["clinical_optionality_pct_dev"]
        if _safe_float(v, None) is not None
    ]
    if len(vals) < 2:
        return None
    return round(statistics.stdev(vals), 4)


def _composite_iqr(rankings: pd.DataFrame) -> Optional[float]:
    """IQR of composite_score among dev tickers."""
    dev = rankings[rankings["archetype"] == "drug_developer"]
    if "composite_score" not in dev.columns:
        return None
    vals = sorted(
        _safe_float(v, None)
        for v in dev["composite_score"]
        if _safe_float(v, None) is not None
    )
    if len(vals) < 4:
        return None
    q1 = vals[len(vals) // 4]
    q3 = vals[3 * len(vals) // 4]
    return round(q3 - q1, 4)


def _catalyst_missing_pct_eligible(rankings: pd.DataFrame) -> Optional[float]:
    """Catalyst missing % among eligible dev tickers."""
    dev = rankings[rankings["archetype"] == "drug_developer"].copy()
    if "eligible" not in dev.columns:
        return None
    eligible = dev[dev["eligible"].astype(str).str.strip() == "1"]
    n_elig = len(eligible)
    if n_elig == 0:
        return None
    if "catalyst_mode" not in eligible.columns:
        return None
    n_missing = sum(
        1
        for m in eligible["catalyst_mode"]
        if str(m).strip() in ("missing", "")
    )
    return round(n_missing / n_elig * 100, 1)


def _catalyst_backfill_share_pct(rankings: pd.DataFrame) -> Optional[float]:
    """Percentage of non-missing catalyst signals sourced from backfill.

    Reads ``catalyst_source`` from rankings; counts trial_cd and trial_active
    as backfill sources.  Returns None when column is absent or no non-missing
    catalyst signals exist.
    """
    if "catalyst_source" not in rankings.columns:
        return None
    dev = rankings[rankings["archetype"] == "drug_developer"]
    if "eligible" in dev.columns:
        dev = dev[dev["eligible"].astype(str).str.strip() == "1"]
    # Non-missing: has a catalyst_source value that isn't blank
    sources = [str(s).strip() for s in dev["catalyst_source"] if str(s).strip()]
    if not sources:
        return None
    backfill_sources = {"trial_cd", "trial_active"}
    n_backfill = sum(1 for s in sources if s in backfill_sources)
    return round(n_backfill / len(sources) * 100, 1)


def _drawdown_coverage_pct(rankings: pd.DataFrame) -> Optional[float]:
    """Drawdown coverage % among dev tickers (has a non-missing reason)."""
    dev = rankings[rankings["archetype"] == "drug_developer"]
    n_dev = len(dev)
    if n_dev == 0:
        return None
    if "de_drawdown_missing_reason" not in dev.columns:
        # Older snapshots without this column — assume coverage unknown
        return None
    n_covered = sum(
        1
        for r in dev["de_drawdown_missing_reason"]
        if str(r).strip() == ""
    )
    return round(n_covered / n_dev * 100, 1)


def _drawdown_rel_coverage_pct(rankings: pd.DataFrame) -> Optional[float]:
    """Relative drawdown (vs XBI) coverage % among dev tickers."""
    dev = rankings[rankings["archetype"] == "drug_developer"]
    n_dev = len(dev)
    if n_dev == 0:
        return None
    if "de_drawdown_rel_xbi" not in dev.columns:
        return None
    n_covered = sum(
        1
        for v in dev["de_drawdown_rel_xbi"]
        if v is not None and str(v).strip() not in ("", "nan")
    )
    return round(n_covered / n_dev * 100, 1)


_RETURNS_SOURCE_KEYS = ("morningstar", "csv", "csv_outlier_override", "unknown")


def _returns_source_metrics(rankings: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Returns-source mix metrics from the ``returns_source`` column.

    Computes per-source counts and share percentages among dev-stage tickers.
    Returns None when the column is absent.
    """
    if "returns_source" not in rankings.columns:
        return None

    dev = rankings[rankings["archetype"] == "drug_developer"]
    n_dev = len(dev)
    if n_dev == 0:
        result: Dict[str, Any] = {"rs_n_dev": 0}
        for src in _RETURNS_SOURCE_KEYS:
            result[f"rs_{src}_count"] = 0
            result[f"rs_{src}_share_pct"] = None
        return result

    from collections import Counter

    def _norm_source(v) -> str:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "unknown"
        s = str(v).strip()
        return s if s else "unknown"

    sources = [_norm_source(v) for v in dev["returns_source"]]
    counts = Counter(sources)

    result = {"rs_n_dev": n_dev}
    for src in _RETURNS_SOURCE_KEYS:
        c = counts.get(src, 0)
        result[f"rs_{src}_count"] = c
        result[f"rs_{src}_share_pct"] = round(c / n_dev * 100, 1)

    return result


_CATALYST_MODE_KEYS = ("specific_days", "blended_window", "no_upcoming", "missing")


def _catalyst_coverage_metrics(rankings: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Catalyst coverage metrics from ``tier_dev`` and ``catalyst_mode`` columns.

    Among dev-stage tickers, counts how many are "eligible" (non-blank tier_dev)
    and breaks down the catalyst_mode distribution among eligible tickers.
    Returns None when required columns are absent.
    """
    if "tier_dev" not in rankings.columns or "catalyst_mode" not in rankings.columns:
        return None

    dev = rankings[rankings["archetype"] == "drug_developer"]
    n_dev = len(dev)
    if n_dev == 0:
        result: Dict[str, Any] = {"cat_n_dev": 0, "cat_n_eligible": 0, "cat_eligible_share_pct": None}
        for mode in _CATALYST_MODE_KEYS:
            result[f"cat_{mode}_count"] = 0
            result[f"cat_{mode}_share_pct"] = None
        return result

    # Eligible = non-blank tier_dev
    def _is_eligible(v) -> bool:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return False
        return str(v).strip() != ""

    eligible_mask = dev["tier_dev"].apply(_is_eligible)
    eligible_dev = dev[eligible_mask]
    n_eligible = len(eligible_dev)

    result = {
        "cat_n_dev": n_dev,
        "cat_n_eligible": n_eligible,
        "cat_eligible_share_pct": round(n_eligible / n_dev * 100, 1) if n_dev > 0 else None,
    }

    if n_eligible == 0:
        for mode in _CATALYST_MODE_KEYS:
            result[f"cat_{mode}_count"] = 0
            result[f"cat_{mode}_share_pct"] = None
        return result

    from collections import Counter

    def _norm_mode(v) -> str:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "missing"
        s = str(v).strip()
        return s if s else "missing"

    modes = [_norm_mode(v) for v in eligible_dev["catalyst_mode"]]
    counts = Counter(modes)

    for mode in _CATALYST_MODE_KEYS:
        c = counts.get(mode, 0)
        result[f"cat_{mode}_count"] = c
        result[f"cat_{mode}_share_pct"] = round(c / n_eligible * 100, 1)

    return result


_CATALYST_SOURCE_KEYS = (
    "CTGOV_CALENDAR", "FDA_CALENDAR", "SEC_8K_FILING",
    "CORPORATE_CALENDAR", "none", "unknown",
)


def _catalyst_source_metrics(
    rankings: pd.DataFrame,
    *,
    strict_missing: bool = False,
) -> Optional[Dict[str, Any]]:
    """Catalyst source mix metrics from the ``catalyst_source`` column.

    Among eligible dev tickers (non-blank ``tier_dev``), counts per-source
    distribution.  Maps ``""`` → ``"none"`` (no catalyst).

    When ``strict_missing=True`` and ``catalyst_mode`` column exists,
    tickers with a dated catalyst (``specific_days`` or ``blended_window``)
    but missing ``catalyst_source`` are counted as ``"unknown"`` (broken
    data).  Otherwise missing → ``"none"``.

    Returns None when column is absent.
    """
    if "catalyst_source" not in rankings.columns or "tier_dev" not in rankings.columns:
        return None

    dev = rankings[rankings["archetype"] == "drug_developer"]
    n_dev = len(dev)
    if n_dev == 0:
        result: Dict[str, Any] = {"cs_n_eligible": 0}
        for src in _CATALYST_SOURCE_KEYS:
            key = src.lower()
            result[f"cs_{key}_count"] = 0
            result[f"cs_{key}_share_pct"] = None
        return result

    # Eligible = non-blank tier_dev
    def _is_eligible(v) -> bool:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return False
        return str(v).strip() != ""

    eligible_dev = dev[dev["tier_dev"].apply(_is_eligible)]
    n_eligible = len(eligible_dev)

    if n_eligible == 0:
        result = {"cs_n_eligible": 0}
        for src in _CATALYST_SOURCE_KEYS:
            key = src.lower()
            result[f"cs_{key}_count"] = 0
            result[f"cs_{key}_share_pct"] = None
        return result

    from collections import Counter

    # Map enrichment-pipeline names → canonical live-pipeline names
    _SOURCE_ALIASES: Dict[str, str] = {
        "trial_pcd": "CTGOV_CALENDAR",
        "pdufa": "FDA_CALENDAR",
        "sec_8k": "SEC_8K_FILING",
        "adcom": "FDA_CALENDAR",
    }

    modes: Optional[pd.Series] = None
    if strict_missing and "catalyst_mode" in eligible_dev.columns:
        modes = eligible_dev["catalyst_mode"].astype(str).fillna("")

    def _norm_source(v, mode: str = "") -> str:
        missing = (v is None) or (isinstance(v, float) and pd.isna(v)) or (str(v).strip() == "")
        if missing:
            if strict_missing and mode in ("specific_days", "blended_window"):
                return "unknown"
            return "none"
        s = str(v).strip()
        return _SOURCE_ALIASES.get(s, s)

    if modes is None:
        sources = [_norm_source(v) for v in eligible_dev["catalyst_source"]]
    else:
        sources = [
            _norm_source(v, m)
            for v, m in zip(
                eligible_dev["catalyst_source"].tolist(),
                modes.tolist(),
            )
        ]
    counts = Counter(sources)

    result = {"cs_n_eligible": n_eligible}
    for src in _CATALYST_SOURCE_KEYS:
        key = src.lower()
        c = counts.get(src, 0)
        result[f"cs_{key}_count"] = c
        result[f"cs_{key}_share_pct"] = round(c / n_eligible * 100, 1)

    return result


# --- Catalyst event type mix ---------------------------------------------------

_CATALYST_EVENT_TYPE_KEYS = (
    "DATA_READOUT", "FDA_DECISION", "FDA_ADCOM",
    "TRIAL_ONGOING", "none", "unknown",
)

# Map enrichment-pipeline event type names → canonical names
_EVENT_TYPE_ALIASES: Dict[str, str] = {
    "data_readout": "DATA_READOUT",
    "fda_decision": "FDA_DECISION",
    "fda_adcom": "FDA_ADCOM",
    "trial_ongoing": "TRIAL_ONGOING",
}


def _catalyst_event_type_metrics(
    rankings: pd.DataFrame,
    *,
    strict_missing: bool = False,
) -> Optional[Dict[str, Any]]:
    """Catalyst event type mix metrics from the ``catalyst_event_type`` column.

    Among eligible dev tickers (non-blank ``tier_dev``), counts per-type
    distribution.  Maps ``""`` → ``"none"`` (no catalyst).

    When ``strict_missing=True`` and ``catalyst_mode`` column exists,
    tickers with a dated catalyst but missing event type are counted as
    ``"unknown"`` (broken data).  Otherwise missing → ``"none"``.

    Returns None when column is absent.
    """
    if "catalyst_event_type" not in rankings.columns or "tier_dev" not in rankings.columns:
        return None

    dev = rankings[rankings["archetype"] == "drug_developer"]
    n_dev = len(dev)
    if n_dev == 0:
        result: Dict[str, Any] = {"ct_n_eligible": 0}
        for et in _CATALYST_EVENT_TYPE_KEYS:
            key = et.lower()
            result[f"ct_{key}_count"] = 0
            result[f"ct_{key}_share_pct"] = None
        return result

    def _is_eligible(v) -> bool:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return False
        return str(v).strip() != ""

    eligible_dev = dev[dev["tier_dev"].apply(_is_eligible)]
    n_eligible = len(eligible_dev)

    if n_eligible == 0:
        result = {"ct_n_eligible": 0}
        for et in _CATALYST_EVENT_TYPE_KEYS:
            key = et.lower()
            result[f"ct_{key}_count"] = 0
            result[f"ct_{key}_share_pct"] = None
        return result

    from collections import Counter

    modes: Optional[pd.Series] = None
    if strict_missing and "catalyst_mode" in eligible_dev.columns:
        modes = eligible_dev["catalyst_mode"].astype(str).fillna("")

    def _norm_event_type(v, mode: str = "") -> str:
        missing = (v is None) or (isinstance(v, float) and pd.isna(v)) or (str(v).strip() == "")
        if missing:
            if strict_missing and mode in ("specific_days", "blended_window"):
                return "unknown"
            return "none"
        s = str(v).strip()
        return _EVENT_TYPE_ALIASES.get(s.lower(), s)

    if modes is None:
        types = [_norm_event_type(v) for v in eligible_dev["catalyst_event_type"]]
    else:
        types = [
            _norm_event_type(v, m)
            for v, m in zip(
                eligible_dev["catalyst_event_type"].tolist(),
                modes.tolist(),
            )
        ]
    counts = Counter(types)

    result = {"ct_n_eligible": n_eligible}
    for et in _CATALYST_EVENT_TYPE_KEYS:
        key = et.lower()
        c = counts.get(et, 0)
        result[f"ct_{key}_count"] = c
        result[f"ct_{key}_share_pct"] = round(c / n_eligible * 100, 1)

    return result


def _catalyst_priority_metrics(
    rankings: pd.DataFrame,
) -> Optional[Dict[str, Any]]:
    """Priority distribution for the cat_priority column (when present).

    Returns counts per priority bucket among eligible dev-stage tickers.
    Returns None when cat_priority column is absent.
    """
    if "cat_priority" not in rankings.columns or "tier_dev" not in rankings.columns:
        return None

    dev = rankings[rankings["archetype"] == "drug_developer"]
    eligible_dev = dev[dev["eligible"] == "1"] if "eligible" in dev.columns else dev
    n = len(eligible_dev)
    if n == 0:
        return {"cp_n_eligible": 0}

    vals = eligible_dev["cat_priority"].fillna(9).astype(int)
    result: Dict[str, Any] = {"cp_n_eligible": n}
    # Named buckets for common priority values
    _LABELS = {1: "fda", 2: "readout", 3: "ongoing", 9: "none", 99: "unknown"}
    for prio, label in _LABELS.items():
        c = int((vals == prio).sum())
        result[f"cp_{label}_count"] = c
        result[f"cp_{label}_share_pct"] = round(c / n * 100, 1)
    return result


def _cost_metrics(
    rankings: pd.DataFrame, cap_bps: float = 1000.0,
) -> Optional[Dict[str, Any]]:
    """Cost telemetry metrics from pipeline-produced columns.

    Reads est_cost_bps, cost_mult, cost_bucket, cost_haircut_applied
    directly from rankings — no recomputation.

    Returns None when required cost columns are absent.
    """
    required = {"est_cost_bps", "cost_mult", "cost_bucket"}
    if not required.issubset(rankings.columns):
        return None

    dev = rankings[rankings["archetype"] == "drug_developer"]
    n_dev = len(dev)
    result: Dict[str, Any] = {}

    if n_dev == 0:
        for k in (
            "cost_coverage_pct", "est_cost_bps_p10", "est_cost_bps_p50",
            "est_cost_bps_p90", "median_cost_bps", "mean_cost_mult",
            "cap_binding_pct",
            "cost_bucket_no_pct", "cost_bucket_mild_pct",
            "cost_bucket_heavy_pct", "cost_bucket_floor_pct",
        ):
            result[k] = None
        return result

    # --- est_cost_bps percentiles + coverage ---
    if "est_cost_bps" in dev.columns:
        cost_vals = [
            fv for v in dev["est_cost_bps"]
            if (fv := _safe_float(v, None)) is not None
        ]
        result["cost_coverage_pct"] = round(len(cost_vals) / n_dev * 100, 1)

        if cost_vals:
            cost_sorted = sorted(cost_vals)
            n = len(cost_sorted)
            result["est_cost_bps_p10"] = round(
                cost_sorted[max(0, int(n * 0.10))], 1
            )
            result["est_cost_bps_p50"] = round(statistics.median(cost_sorted), 1)
            result["est_cost_bps_p90"] = round(
                cost_sorted[min(n - 1, int(n * 0.90))], 1
            )
            result["median_cost_bps"] = result["est_cost_bps_p50"]
            eps = 1e-9
            result["cap_binding_pct"] = round(
                sum(1 for v in cost_vals if v >= (cap_bps - eps))
                / len(cost_vals) * 100, 1
            )
        else:
            for k in ("est_cost_bps_p10", "est_cost_bps_p50",
                       "est_cost_bps_p90", "median_cost_bps",
                       "cap_binding_pct"):
                result[k] = None
    else:
        for k in ("cost_coverage_pct", "est_cost_bps_p10", "est_cost_bps_p50",
                   "est_cost_bps_p90", "median_cost_bps", "cap_binding_pct"):
            result[k] = None

    # --- cost_mult stats + bucket shares ---
    if "cost_mult" in dev.columns:
        mults = [
            fv for v in dev["cost_mult"]
            if (fv := _safe_float(v, None)) is not None
        ]
        result["mean_cost_mult"] = (
            round(statistics.mean(mults), 4) if mults else None
        )

        n_with = len(mults)
        if n_with > 0:
            n_no = sum(1 for m in mults if m >= 1.0)
            n_mild = sum(1 for m in mults if 0.70 < m < 1.0)
            n_heavy = sum(1 for m in mults if 0.55 < m <= 0.70)
            n_floor = sum(1 for m in mults if m <= 0.55)
            result["cost_bucket_no_pct"] = round(n_no / n_with * 100, 1)
            result["cost_bucket_mild_pct"] = round(n_mild / n_with * 100, 1)
            result["cost_bucket_heavy_pct"] = round(n_heavy / n_with * 100, 1)
            result["cost_bucket_floor_pct"] = round(n_floor / n_with * 100, 1)
        else:
            for b in ("no", "mild", "heavy", "floor"):
                result[f"cost_bucket_{b}_pct"] = None
    else:
        result["mean_cost_mult"] = None
        for b in ("no", "mild", "heavy", "floor"):
            result[f"cost_bucket_{b}_pct"] = None

    return result


def compute_snapshot_metrics(
    snap: SnapshotData,
    *,
    strict_cs_missing: bool = False,
) -> Dict[str, Any]:
    """Compute drift metrics for a single snapshot."""
    rankings = snap.rankings
    tc = _tier_counts(rankings)
    n_dev = sum(tc.values())

    metrics: Dict[str, Any] = {
        "date": snap.date,
        "ruleset_id": snap.ruleset_id,
        "n_dev": n_dev,
    }

    # Tier counts and percentages
    for tier in ("A", "B", "C", "D"):
        metrics[f"tier_{tier}_count"] = tc[tier]
        metrics[f"tier_{tier}_pct"] = (
            round(tc[tier] / n_dev * 100, 1) if n_dev > 0 else 0.0
        )

    # Eligible percentage
    dev = rankings[rankings["archetype"] == "drug_developer"]
    if "eligible" in dev.columns:
        n_elig = sum(
            1 for e in dev["eligible"] if str(e).strip() == "1"
        )
        metrics["eligible_pct"] = round(n_elig / n_dev * 100, 1) if n_dev > 0 else 0.0
    else:
        metrics["eligible_pct"] = None

    # Catalyst missing % among eligible
    metrics["catalyst_missing_pct"] = _catalyst_missing_pct_eligible(rankings)

    # Catalyst backfill share
    metrics["catalyst_backfill_share_pct"] = _catalyst_backfill_share_pct(rankings)

    # Drawdown coverage
    metrics["drawdown_coverage_pct"] = _drawdown_coverage_pct(rankings)

    # Relative drawdown (vs XBI) coverage
    metrics["drawdown_rel_coverage_pct"] = _drawdown_rel_coverage_pct(rankings)

    # Catalyst strength distribution (when column exists)
    if "catalyst_strength" in dev.columns:
        n_elig = sum(1 for e in dev["eligible"] if str(e).strip() == "1") if "eligible" in dev.columns else n_dev
        if n_elig > 0:
            eligible_dev = dev[dev["eligible"].astype(str).str.strip() == "1"] if "eligible" in dev.columns else dev
            for band in ("near", "mid", "far", "missing"):
                n_band = sum(1 for v in eligible_dev["catalyst_strength"] if str(v).strip() == band)
                metrics[f"catalyst_strength_{band}_pct"] = round(n_band / n_elig * 100, 1) if n_elig else 0.0

    # Score dispersion
    metrics["optionality_std"] = _optionality_std(rankings)
    metrics["composite_iqr"] = _composite_iqr(rankings)

    # Gate margin metrics (when columns exist)
    margin_summary = _compute_margin_summary(dev)
    for key in ("median_dd_abs_margin", "median_dd_rel_margin"):
        metrics[key] = margin_summary.get(key)
    metrics["rescued_count"] = margin_summary.get("rescued_count")
    for key in ("dd_abs_near_gate_pct", "dd_rel_near_gate_pct",
                "optionality_near_a_floor_pct", "rescued_share_pct",
                "dd_rel_margin_rescue_count", "dd_rel_margin_rescue_share_pct"):
        metrics[key] = margin_summary.get(key)

    # Cost telemetry
    cost = _cost_metrics(rankings)
    if cost is not None:
        metrics.update(cost)

    # Returns-source mix
    rs = _returns_source_metrics(rankings)
    if rs is not None:
        metrics.update(rs)

    # Catalyst coverage
    cat = _catalyst_coverage_metrics(rankings)
    if cat is not None:
        metrics.update(cat)

    # Catalyst source mix
    cs = _catalyst_source_metrics(rankings, strict_missing=strict_cs_missing)
    if cs is not None:
        metrics.update(cs)

    # Catalyst event type mix
    ct = _catalyst_event_type_metrics(rankings, strict_missing=strict_cs_missing)
    if ct is not None:
        metrics.update(ct)

    # Catalyst priority distribution
    cp = _catalyst_priority_metrics(rankings)
    if cp is not None:
        metrics.update(cp)

    # Top-25 tickers (for overlap computation)
    metrics["_top25"] = _top_n_tickers(rankings, 25)

    return metrics


def compute_drift_metrics(
    snapshots: List[SnapshotData],
    *,
    strict_cs_missing: bool = False,
) -> Dict[str, Any]:
    """Compute per-snapshot metrics plus rolling aggregates.

    Returns:
        {
            "snapshots": [per-snapshot metrics (oldest first)],
            "rolling": {metric_name: {"min": ..., "max": ..., "mean": ..., "current": ...}},
            "current": latest snapshot metrics,
        }
    """
    if not snapshots:
        return {"snapshots": [], "rolling": {}, "current": {}}

    all_metrics: List[Dict[str, Any]] = []
    for snap in snapshots:
        m = compute_snapshot_metrics(snap, strict_cs_missing=strict_cs_missing)
        all_metrics.append(m)

    # Compute top-25 overlap (vs prior snapshot)
    for i, m in enumerate(all_metrics):
        if i == 0:
            m["top25_overlap_pct"] = None
        else:
            prev_top = all_metrics[i - 1].get("_top25", set())
            cur_top = m.get("_top25", set())
            if prev_top and cur_top:
                union = prev_top | cur_top
                intersection = prev_top & cur_top
                # Jaccard-style: intersection over 25 (fixed denominator)
                m["top25_overlap_pct"] = round(
                    len(intersection) / 25 * 100, 1
                )
            else:
                m["top25_overlap_pct"] = None

    # Remove internal _top25 sets (not serializable)
    for m in all_metrics:
        m.pop("_top25", None)

    current = all_metrics[-1]

    # Rolling aggregates for key numeric metrics
    roll_keys = [
        "tier_A_pct", "tier_B_pct", "tier_C_pct", "tier_D_pct",
        "eligible_pct", "catalyst_missing_pct", "catalyst_backfill_share_pct",
        "top25_overlap_pct",
        "optionality_std", "composite_iqr", "drawdown_coverage_pct",
        "drawdown_rel_coverage_pct",
        "catalyst_strength_near_pct", "catalyst_strength_mid_pct",
        "catalyst_strength_far_pct", "catalyst_strength_missing_pct",
        "median_dd_abs_margin", "median_dd_rel_margin", "rescued_count",
        "dd_abs_near_gate_pct", "dd_rel_near_gate_pct",
        "optionality_near_a_floor_pct", "rescued_share_pct",
        "dd_rel_margin_rescue_count", "dd_rel_margin_rescue_share_pct",
        "cost_coverage_pct", "cap_binding_pct", "mean_cost_mult",
        "est_cost_bps_p50", "median_cost_bps",
        "rs_morningstar_share_pct", "rs_unknown_share_pct",
        "rs_csv_outlier_override_share_pct",
        "cat_eligible_share_pct", "cat_specific_days_share_pct",
        "cat_no_upcoming_share_pct", "cat_missing_share_pct",
        "cs_ctgov_calendar_share_pct", "cs_fda_calendar_share_pct",
        "cs_sec_8k_filing_share_pct", "cs_corporate_calendar_share_pct",
        "cs_none_share_pct", "cs_unknown_share_pct",
        "ct_data_readout_share_pct", "ct_fda_decision_share_pct",
        "ct_fda_adcom_share_pct", "ct_trial_ongoing_share_pct",
        "ct_none_share_pct", "ct_unknown_share_pct",
    ]
    rolling: Dict[str, Dict[str, Any]] = {}
    for key in roll_keys:
        vals = [m[key] for m in all_metrics if m.get(key) is not None]
        if vals:
            cur_val = current.get(key)
            med = round(statistics.median(vals), 2)
            entry: Dict[str, Any] = {
                "min": round(min(vals), 2),
                "max": round(max(vals), 2),
                "mean": round(statistics.mean(vals), 2),
                "median": med,
                "current": cur_val,
            }
            # IQR requires >= 3 data points for statistics.quantiles
            if len(vals) >= 3:
                q1, _q2, q3 = statistics.quantiles(vals, n=4)
                entry["iqr"] = round(q3 - q1, 2)
            else:
                entry["iqr"] = None
            # Delta = current - median (positive = increased vs window baseline)
            if cur_val is not None:
                entry["delta"] = round(cur_val - med, 2)
            else:
                entry["delta"] = None
            rolling[key] = entry

    return {
        "snapshots": all_metrics,
        "rolling": rolling,
        "current": current,
    }


# ---------------------------------------------------------------------------
# Guardrail evaluation
# ---------------------------------------------------------------------------
def find_rollback_candidate(
    manifest_path: Path = MANIFEST_PATH,
) -> Optional[Dict[str, Any]]:
    """Find the most recently retired ruleset from the manifest.

    Returns the manifest entry dict, or None if no retired entries exist.
    """
    if not manifest_path.exists():
        return None
    with open(manifest_path) as f:
        manifest = json.load(f)

    retired = [
        r for r in manifest.get("rulesets", []) if r.get("status") == "retired"
    ]
    if not retired:
        return None

    # Sort by updated_at (descending) to find the most recently retired
    def _sort_key(r: Dict) -> str:
        return r.get("updated_at", r.get("created_at", ""))

    retired.sort(key=_sort_key, reverse=True)
    return retired[0]


def evaluate_guardrails(
    metrics: Dict[str, Any],
    guardrails: DriftGuardrails,
) -> Tuple[str, List[str], Optional[Dict[str, Any]], str]:
    """Evaluate drift guardrails against current snapshot metrics.

    Returns:
        (status, reasons, rollback_candidate, recommended_action)
        status: "OK" | "WARN" | "FAIL"
        reasons: list of human-readable reason strings
        rollback_candidate: manifest entry for most recently retired ruleset (on FAIL)
        recommended_action: "NONE" | "ROLLBACK_RECOMMENDED" | "INVESTIGATE"
    """
    current = metrics.get("current", {})
    reasons: List[str] = []

    a_pct = current.get("tier_A_pct")
    cat_missing = current.get("catalyst_missing_pct")
    overlap = current.get("top25_overlap_pct")
    opt_std = current.get("optionality_std")

    # FAIL checks
    if a_pct is not None and a_pct <= guardrails.fail_a_pct_low:
        reasons.append(
            f"A-tier % = {a_pct:.1f}% <= {guardrails.fail_a_pct_low}% floor"
        )
    if a_pct is not None and a_pct > guardrails.fail_a_pct_high:
        reasons.append(
            f"A-tier % = {a_pct:.1f}% > {guardrails.fail_a_pct_high}% ceiling"
        )
    if cat_missing is not None and cat_missing > guardrails.fail_catalyst_missing_high:
        reasons.append(
            f"Catalyst missing = {cat_missing:.1f}% > {guardrails.fail_catalyst_missing_high}% ceiling"
        )
    if overlap is not None and overlap < guardrails.fail_overlap_low:
        reasons.append(
            f"Top-25 overlap = {overlap:.1f}% < {guardrails.fail_overlap_low}% floor"
        )
    if opt_std is not None and opt_std < guardrails.fail_dispersion_low:
        reasons.append(
            f"Optionality std = {opt_std:.4f} < {guardrails.fail_dispersion_low} floor"
        )

    dd_rel_cov = current.get("drawdown_rel_coverage_pct")
    if dd_rel_cov is not None and dd_rel_cov < guardrails.fail_drawdown_rel_coverage_low:
        reasons.append(
            f"Relative DD coverage = {dd_rel_cov:.1f}% < "
            f"{guardrails.fail_drawdown_rel_coverage_low}% floor"
        )

    if reasons:
        rollback = find_rollback_candidate()
        n_fail = len(reasons)
        if n_fail >= guardrails.fail_corroboration_count and rollback:
            action = "ROLLBACK_RECOMMENDED"
        else:
            action = "INVESTIGATE"
        return "FAIL", reasons, rollback, action

    # WARN checks (non-FAIL, advisory only)
    warn_reasons: List[str] = []

    if a_pct is not None and a_pct > guardrails.fail_a_pct_low and a_pct < guardrails.warn_a_pct_low:
        warn_reasons.append(
            f"A-tier % = {a_pct:.1f}% < {guardrails.warn_a_pct_low}% (sparse catalyst regime)"
        )

    median_cost = current.get("median_cost_bps")
    if median_cost is not None and median_cost > guardrails.warn_median_cost_bps_high:
        warn_reasons.append(
            f"Median cost = {median_cost:.1f} bps > "
            f"{guardrails.warn_median_cost_bps_high} bps ceiling"
        )

    cost_coverage = current.get("cost_coverage_pct")
    if cost_coverage is not None and cost_coverage < guardrails.warn_cost_coverage_low:
        warn_reasons.append(
            f"Cost coverage = {cost_coverage:.1f}% < "
            f"{guardrails.warn_cost_coverage_low}% floor"
        )

    cap_binding = current.get("cap_binding_pct")
    if cap_binding is not None and cap_binding > guardrails.warn_cap_binding_high:
        warn_reasons.append(
            f"Cap binding = {cap_binding:.1f}% > "
            f"{guardrails.warn_cap_binding_high}% ceiling"
        )

    backfill_share = current.get("catalyst_backfill_share_pct")
    if backfill_share is not None and backfill_share > guardrails.warn_backfill_share_high:
        warn_reasons.append(
            f"Backfill share = {backfill_share:.1f}% > "
            f"{guardrails.warn_backfill_share_high}% ceiling"
        )

    rescue_share = current.get("dd_rel_margin_rescue_share_pct")
    if rescue_share is not None and rescue_share > guardrails.warn_dd_rel_margin_rescue_share_high:
        warn_reasons.append(
            f"DD rel-margin rescue share = {rescue_share:.1f}% > "
            f"{guardrails.warn_dd_rel_margin_rescue_share_high}% ceiling"
        )

    eligible_pct = current.get("eligible_pct")
    if eligible_pct is not None and eligible_pct < guardrails.warn_eligible_dev_pct_low:
        warn_reasons.append(
            f"eligible_dev_pct={eligible_pct:.1f}% < "
            f"{guardrails.warn_eligible_dev_pct_low}% floor"
        )

    if (dd_rel_cov is not None
            and dd_rel_cov < guardrails.warn_drawdown_rel_coverage_low
            and dd_rel_cov >= guardrails.fail_drawdown_rel_coverage_low):
        warn_reasons.append(
            f"Relative DD coverage = {dd_rel_cov:.1f}% < "
            f"{guardrails.warn_drawdown_rel_coverage_low}% floor"
        )

    # Returns-source mix checks
    rs_unknown = current.get("rs_unknown_share_pct")
    if rs_unknown is not None and rs_unknown > guardrails.warn_rs_unknown_share_high:
        warn_reasons.append(
            f"Returns source unknown = {rs_unknown:.1f}% > "
            f"{guardrails.warn_rs_unknown_share_high}% ceiling"
        )
    rs_outlier = current.get("rs_csv_outlier_override_share_pct")
    if rs_outlier is not None and rs_outlier > guardrails.warn_rs_csv_outlier_override_share_high:
        warn_reasons.append(
            f"Returns source csv_outlier_override = {rs_outlier:.1f}% > "
            f"{guardrails.warn_rs_csv_outlier_override_share_high}% ceiling"
        )
    rs_mstar = current.get("rs_morningstar_share_pct")
    if rs_mstar is not None and rs_mstar < guardrails.warn_rs_morningstar_share_low:
        warn_reasons.append(
            f"Returns source morningstar = {rs_mstar:.1f}% < "
            f"{guardrails.warn_rs_morningstar_share_low}% floor"
        )

    # Catalyst coverage checks
    cat_elig = current.get("cat_eligible_share_pct")
    if cat_elig is not None and cat_elig < guardrails.warn_cat_eligible_share_low:
        warn_reasons.append(
            f"Catalyst eligible share = {cat_elig:.1f}% < "
            f"{guardrails.warn_cat_eligible_share_low}% floor"
        )
    cat_sd = current.get("cat_specific_days_share_pct")
    if cat_sd is not None and cat_sd < guardrails.warn_cat_specific_days_share_low:
        warn_reasons.append(
            f"Catalyst specific_days share = {cat_sd:.1f}% < "
            f"{guardrails.warn_cat_specific_days_share_low}% floor"
        )

    # Catalyst source mix checks — only evaluate when dated catalysts are
    # prevalent enough for source mix to be meaningful.
    cat_sd_current = current.get("cat_specific_days_share_pct")
    _cs_regime_ok = (
        cat_sd_current is not None
        and cat_sd_current >= _SUGGEST_CS_MIN_CAT_SPECIFIC_DAYS_MEDIAN
    )
    if _cs_regime_ok:
        cs_ctgov = current.get("cs_ctgov_calendar_share_pct")
        if cs_ctgov is not None and cs_ctgov < guardrails.warn_cs_ctgov_share_low:
            warn_reasons.append(
                f"Catalyst source CTGOV share = {cs_ctgov:.1f}% < "
                f"{guardrails.warn_cs_ctgov_share_low}% floor"
            )
        cs_unknown = current.get("cs_unknown_share_pct")
        if cs_unknown is not None and cs_unknown > guardrails.warn_cs_unknown_share_high:
            warn_reasons.append(
                f"Catalyst source unknown = {cs_unknown:.1f}% > "
                f"{guardrails.warn_cs_unknown_share_high}% ceiling"
            )

    # Catalyst event type mix checks — same regime gate as cs_*
    if _cs_regime_ok:
        ct_unknown = current.get("ct_unknown_share_pct")
        if ct_unknown is not None and ct_unknown > guardrails.warn_ct_unknown_share_high:
            warn_reasons.append(
                f"Catalyst event type unknown = {ct_unknown:.1f}% > "
                f"{guardrails.warn_ct_unknown_share_high}% ceiling"
            )
        # FDA spike tripwire: FDA_DECISION + FDA_ADCOM combined
        ct_fda_dec = current.get("ct_fda_decision_share_pct") or 0.0
        ct_fda_adc = current.get("ct_fda_adcom_share_pct") or 0.0
        ct_fda_combined = ct_fda_dec + ct_fda_adc
        if ct_fda_combined > guardrails.warn_ct_fda_share_spike:
            warn_reasons.append(
                f"Catalyst event type FDA share = {ct_fda_combined:.1f}% > "
                f"{guardrails.warn_ct_fda_share_spike}% ceiling"
            )

    if warn_reasons:
        return "WARN", warn_reasons, None, "INVESTIGATE"

    return "OK", [], None, "NONE"


# ---------------------------------------------------------------------------
# Adaptive WARN evaluation
# ---------------------------------------------------------------------------
ADAPTIVE_WARN_METRICS = [
    "eligible_pct",
    "tier_A_pct",
    "catalyst_missing_pct",
    "catalyst_strength_near_pct",
]


def evaluate_adaptive_warnings(
    metrics: Dict[str, Any],
    guardrails: DriftGuardrails,
) -> Tuple[str, List[str]]:
    """IQR-based adaptive warnings for key health indicators.

    Fires when the current value drifts more than ``warn_iqr_k * max(IQR, floor)``
    from the rolling window median.  Requires at least ``warn_min_window``
    snapshots in the window to activate.

    Returns:
        (status, reasons) where status is "OK" or "WARN".
    """
    rolling = metrics.get("rolling", {})
    n_snaps = len(metrics.get("snapshots", []))
    reasons: List[str] = []

    if n_snaps < guardrails.warn_min_window:
        return "OK", []

    for key in ADAPTIVE_WARN_METRICS:
        entry = rolling.get(key)
        if entry is None:
            continue
        iqr = entry.get("iqr")
        delta = entry.get("delta")
        median = entry.get("median")
        cur = entry.get("current")

        if iqr is None or delta is None:
            continue

        effective_iqr = max(iqr, guardrails.warn_iqr_floor)
        threshold = guardrails.warn_iqr_k * effective_iqr

        if abs(delta) > threshold:
            direction = "above" if delta > 0 else "below"
            label = key.replace("_", " ").replace("pct", "%")
            reasons.append(
                f"{label} = {cur} ({direction} median {median} by "
                f"{abs(delta):.1f}pp; threshold {threshold:.1f}pp = "
                f"{guardrails.warn_iqr_k} * max(IQR={iqr}, floor={guardrails.warn_iqr_floor}))"
            )

    if reasons:
        return "WARN", reasons
    return "OK", []


# ---------------------------------------------------------------------------
# Suggested guardrails — data-driven threshold proposals (informational only)
# ---------------------------------------------------------------------------

# Each spec: (metric_key, direction, guardrail_field, label)
# direction: "low_floor" → suggested = max(0, median - k*IQR)
#            "high_ceiling" → suggested = min(100, median + k*IQR)
_SUGGESTION_SPECS: List[Tuple[str, str, str, str]] = [
    ("tier_A_pct", "low_floor", "warn_a_pct_low", "A-tier %"),
    ("cat_eligible_share_pct", "low_floor", "warn_cat_eligible_share_low", "Catalyst eligible share"),
    ("cat_specific_days_share_pct", "low_floor", "warn_cat_specific_days_share_low", "Specific-days share"),
    ("rs_morningstar_share_pct", "low_floor", "warn_rs_morningstar_share_low", "Morningstar share"),
    ("rs_unknown_share_pct", "high_ceiling", "warn_rs_unknown_share_high", "Returns unknown share"),
    ("cs_ctgov_calendar_share_pct", "low_floor", "warn_cs_ctgov_share_low", "CTGOV calendar share"),
    ("cs_unknown_share_pct", "high_ceiling", "warn_cs_unknown_share_high", "Unknown source share"),
    ("ct_unknown_share_pct", "high_ceiling", "warn_ct_unknown_share_high", "Unknown event type share"),
]

SUGGESTION_MIN_POINTS = 3

# Suppress cs_ctgov suggestion when dated catalysts are too sparse to be meaningful
_SUGGEST_CS_MIN_CAT_SPECIFIC_DAYS_MEDIAN = 40.0


def compute_suggested_guardrails(
    all_metrics: List[Dict[str, Any]],
    guardrails: DriftGuardrails,
    k: float = 2.0,
    min_points: int = SUGGESTION_MIN_POINTS,
) -> List[Dict[str, Any]]:
    """Compute data-driven threshold suggestions from prior snapshots.

    Uses ``all_metrics[:-1]`` (excludes current) to build a baseline.
    For each metric in ``_SUGGESTION_SPECS``, computes median/IQR and
    proposes a tighter threshold if enough history exists.

    Returns a list of suggestion dicts, one per spec with enough data.
    Each dict: {metric, label, direction, n, median, iqr, suggested,
                current_threshold, tighter}.
    """
    if len(all_metrics) < 2:
        return []

    # Prior snapshots only (exclude current)
    prior = all_metrics[:-1]

    suggestions: List[Dict[str, Any]] = []
    for metric_key, direction, guardrail_field, label in _SUGGESTION_SPECS:
        vals = [
            m[metric_key]
            for m in prior
            if m.get(metric_key) is not None
        ]
        if len(vals) < min_points:
            continue

        # Gate: suppress cs_*/ct_* suggestions when dated catalysts are sparse
        if metric_key in (
            "cs_ctgov_calendar_share_pct",
            "cs_unknown_share_pct",
            "ct_unknown_share_pct",
        ):
            cat_vals = [
                m["cat_specific_days_share_pct"]
                for m in prior
                if m.get("cat_specific_days_share_pct") is not None
            ]
            cat_median = statistics.median(cat_vals) if cat_vals else 0.0
            if cat_median < _SUGGEST_CS_MIN_CAT_SPECIFIC_DAYS_MEDIAN:
                current_threshold = getattr(guardrails, guardrail_field, None)
                suggestions.append({
                    "metric": metric_key,
                    "label": label,
                    "direction": direction,
                    "n": len(vals),
                    "median": None,
                    "iqr": None,
                    "suggested": None,
                    "current_threshold": current_threshold,
                    "tighter": False,
                    "suppressed": True,
                    "suppressed_reason": (
                        f"prior median cat_specific_days={cat_median:.1f}%"
                        f" < {_SUGGEST_CS_MIN_CAT_SPECIFIC_DAYS_MEDIAN:.1f}%"
                    ),
                })
                continue

        median = statistics.median(vals)
        if len(vals) >= 3:
            q1, _q2, q3 = statistics.quantiles(vals, n=4)
            iqr = q3 - q1
        else:
            iqr = 0.0

        current_threshold = getattr(guardrails, guardrail_field, None)

        if direction == "low_floor":
            suggested = max(0.0, median - k * iqr)
        else:  # high_ceiling
            suggested = min(100.0, median + k * iqr)
        suggested = round(suggested, 1)

        # Is the suggestion tighter than current?
        tighter = False
        if current_threshold is not None:
            if direction == "low_floor":
                tighter = suggested > current_threshold
            else:
                tighter = suggested < current_threshold

        suggestions.append({
            "metric": metric_key,
            "label": label,
            "direction": direction,
            "n": len(vals),
            "median": round(median, 1),
            "iqr": round(iqr, 1),
            "suggested": suggested,
            "current_threshold": current_threshold,
            "tighter": tighter,
        })

    return suggestions


# ---------------------------------------------------------------------------
# Drift attribution — root-cause breadcrumbs for pairwise snapshot diffs
# ---------------------------------------------------------------------------
_GATE_KEYS = ("fundamental_red_flag", "sev3", "deep_drawdown", "adv_fail")
_STRENGTH_RANK = {"near": 0, "mid": 1, "far": 2, "missing": 3}


def _parse_pipe_separated(value) -> List[str]:
    """Split a pipe-separated CSV string, handling blank/NaN/None."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    s = str(value).strip()
    if not s:
        return []
    return [part.strip() for part in s.split("|") if part.strip()]


def _compute_gate_counts(dev: pd.DataFrame) -> Dict[str, int]:
    """Count each eligibility gate reason among dev tickers.

    Returns dict with 4 keys always present (zeros for missing gates).
    Graceful: returns all-zero if ``ineligible_reasons`` column is absent.
    """
    counts = {k: 0 for k in _GATE_KEYS}
    if "ineligible_reasons" not in dev.columns:
        return counts
    for val in dev["ineligible_reasons"]:
        for reason in _parse_pipe_separated(val):
            if reason in counts:
                counts[reason] += 1
    return counts


def _compute_strength_counts(dev: pd.DataFrame) -> Dict[str, int]:
    """Count catalyst_strength buckets among eligible dev tickers.

    Returns dict with 4 keys: near, mid, far, missing.
    Graceful: all-zero if column missing.
    """
    counts = {"near": 0, "mid": 0, "far": 0, "missing": 0}
    if "catalyst_strength" not in dev.columns:
        return counts
    # Filter to eligible only
    if "eligible" in dev.columns:
        eligible = dev[dev["eligible"].astype(str).str.strip() == "1"]
    else:
        eligible = dev
    for val in eligible["catalyst_strength"]:
        band = str(val).strip()
        if band in counts:
            counts[band] += 1
    return counts


def _compute_margin_summary(dev: pd.DataFrame) -> Dict[str, Any]:
    """Compute margin statistics among dev tickers.

    Graceful: returns all-None if margin columns are absent (older snapshots).
    """
    result: Dict[str, Any] = {}

    for col in ("dd_abs_margin", "dd_rel_margin", "optionality_margin_a"):
        if col not in dev.columns:
            result[f"median_{col}"] = None
            result[f"p10_{col}"] = None
            continue
        vals = []
        for v in dev[col]:
            fv = _safe_float(v, None)
            if fv is not None:
                vals.append(fv)
        if vals:
            vals_sorted = sorted(vals)
            result[f"median_{col}"] = round(statistics.median(vals), 4)
            # P10 (10th percentile — worst-case margin)
            idx = max(0, int(len(vals_sorted) * 0.10))
            result[f"p10_{col}"] = round(vals_sorted[min(idx, len(vals_sorted) - 1)], 4)
        else:
            result[f"median_{col}"] = None
            result[f"p10_{col}"] = None

    # Rescued count
    if "rescued_by_rel" in dev.columns:
        result["rescued_count"] = sum(
            1 for v in dev["rescued_by_rel"]
            if str(v).strip() == "1"
        )
    else:
        result["rescued_count"] = None

    # --- Gate pressure metrics (share within ±5pp of threshold) ---
    n_dev = len(dev)

    for col, key in [
        ("dd_abs_margin", "dd_abs_near_gate_pct"),
        ("dd_rel_margin", "dd_rel_near_gate_pct"),
        ("optionality_margin_a", "optionality_near_a_floor_pct"),
    ]:
        if col not in dev.columns:
            result[key] = None
            continue
        n_near = 0
        n_valid = 0
        for v in dev[col]:
            fv = _safe_float(v, None)
            if fv is not None:
                n_valid += 1
                if -0.05 <= fv <= 0.05:
                    n_near += 1
        result[key] = round(n_near / n_valid * 100, 1) if n_valid > 0 else None

    # rescued_share_pct = rescued / eligible
    if "eligible" in dev.columns and result.get("rescued_count") is not None:
        n_elig = sum(1 for e in dev["eligible"] if str(e).strip() == "1")
        result["rescued_share_pct"] = (
            round(result["rescued_count"] / n_elig * 100, 1)
            if n_elig > 0 else 0.0
        )
    else:
        result["rescued_share_pct"] = None

    # dd_rel_margin_rescued count + share
    if "dd_rel_margin_rescued" in dev.columns:
        result["dd_rel_margin_rescue_count"] = sum(
            1 for v in dev["dd_rel_margin_rescued"]
            if str(v).strip() == "1"
        )
    else:
        result["dd_rel_margin_rescue_count"] = None

    if "eligible" in dev.columns and result.get("dd_rel_margin_rescue_count") is not None:
        n_elig = sum(1 for e in dev["eligible"] if str(e).strip() == "1")
        result["dd_rel_margin_rescue_share_pct"] = (
            round(result["dd_rel_margin_rescue_count"] / n_elig * 100, 1)
            if n_elig > 0 else 0.0
        )
    else:
        result["dd_rel_margin_rescue_share_pct"] = None

    return result


def _compute_strength_transitions(
    prior_dev: pd.DataFrame, current_dev: pd.DataFrame
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Find tickers whose catalyst_strength changed between snapshots.

    Only considers tickers eligible in BOTH snapshots (intersection).
    Returns (degraded, improved) lists of {"ticker", "prior", "current"} dicts,
    sorted by magnitude of transition, capped at 10 each.
    """
    degraded: List[Dict[str, str]] = []
    improved: List[Dict[str, str]] = []

    for df in (prior_dev, current_dev):
        if "catalyst_strength" not in df.columns:
            return [], []

    # Build eligible-only lookups: ticker -> strength
    def _eligible_strength_map(df: pd.DataFrame) -> Dict[str, str]:
        if "eligible" in df.columns:
            elig = df[df["eligible"].astype(str).str.strip() == "1"]
        else:
            elig = df
        out: Dict[str, str] = {}
        for _, row in elig.iterrows():
            t = str(row.get("ticker", "")).strip()
            s = str(row.get("catalyst_strength", "")).strip()
            if t and s:
                out[t] = s
        return out

    prior_map = _eligible_strength_map(prior_dev)
    current_map = _eligible_strength_map(current_dev)
    common = set(prior_map) & set(current_map)

    for ticker in common:
        p, c = prior_map[ticker], current_map[ticker]
        if p == c:
            continue
        p_rank = _STRENGTH_RANK.get(p, 3)
        c_rank = _STRENGTH_RANK.get(c, 3)
        entry = {"ticker": ticker, "prior": p, "current": c}
        if c_rank > p_rank:
            degraded.append(entry)
        else:
            improved.append(entry)

    # Sort by magnitude (biggest rank change first), then ticker for stability
    degraded.sort(key=lambda e: (-abs(_STRENGTH_RANK.get(e["current"], 3) - _STRENGTH_RANK.get(e["prior"], 3)), e["ticker"]))
    improved.sort(key=lambda e: (-abs(_STRENGTH_RANK.get(e["current"], 3) - _STRENGTH_RANK.get(e["prior"], 3)), e["ticker"]))

    return degraded[:10], improved[:10]


def _compute_churn_details(
    prior_rankings: pd.DataFrame, current_rankings: pd.DataFrame
) -> Dict[str, Any]:
    """Compute portfolio churn between two snapshots.

    Uses top-25 by actionable_rank among dev tickers.
    Returns dropped/added ticker lists with metadata.
    """
    prior_top = _top_n_tickers(prior_rankings, 25)
    current_top = _top_n_tickers(current_rankings, 25)

    dropped_tickers = prior_top - current_top
    added_tickers = current_top - prior_top

    def _ticker_meta(rankings: pd.DataFrame, ticker: str) -> Dict[str, str]:
        dev = rankings[rankings["archetype"] == "drug_developer"]
        rows = dev[dev["ticker"] == ticker]
        if rows.empty:
            return {"ticker": ticker, "tier_dev": "", "size_band": "", "tier_reason": ""}
        row = rows.iloc[0]
        return {
            "ticker": ticker,
            "tier_dev": str(row.get("tier_dev", "")),
            "size_band": str(row.get("size_band", "")),
            "tier_reason": str(row.get("tier_reason", "")),
        }

    dropped = sorted(
        [_ticker_meta(prior_rankings, t) for t in dropped_tickers],
        key=lambda d: d["ticker"],
    )
    added = sorted(
        [_ticker_meta(current_rankings, t) for t in added_tickers],
        key=lambda d: d["ticker"],
    )

    return {
        "prior_top25": sorted(prior_top),
        "current_top25": sorted(current_top),
        "dropped": dropped,
        "added": added,
        "overlap_count": len(prior_top & current_top),
    }


def compute_attribution(
    snapshots: List[SnapshotData],
) -> Optional[Dict[str, Any]]:
    """Compute root-cause attribution by diffing the two most recent snapshots.

    Returns None if fewer than 2 snapshots are available.
    """
    if len(snapshots) < 2:
        return None

    prior, current = snapshots[-2], snapshots[-1]
    prior_dev = prior.rankings[prior.rankings["archetype"] == "drug_developer"]
    current_dev = current.rankings[current.rankings["archetype"] == "drug_developer"]

    # Eligibility gates
    prior_gates = _compute_gate_counts(prior_dev)
    current_gates = _compute_gate_counts(current_dev)
    gate_delta = {k: current_gates[k] - prior_gates[k] for k in _GATE_KEYS}

    # Catalyst strength
    prior_strength = _compute_strength_counts(prior_dev)
    current_strength = _compute_strength_counts(current_dev)
    strength_delta = {k: current_strength[k] - prior_strength[k] for k in _STRENGTH_RANK}
    degraded, improved = _compute_strength_transitions(prior_dev, current_dev)

    # Portfolio churn
    churn = _compute_churn_details(prior.rankings, current.rankings)

    # Gate margin summary
    prior_margins = _compute_margin_summary(prior_dev)
    current_margins = _compute_margin_summary(current_dev)
    margin_delta: Dict[str, Any] = {}
    for key in prior_margins:
        p = prior_margins[key]
        c = current_margins.get(key)
        if isinstance(p, (int, float)) and isinstance(c, (int, float)):
            margin_delta[key] = round(c - p, 4)
        else:
            margin_delta[key] = None

    return {
        "prior_date": prior.date,
        "current_date": current.date,
        "eligibility_gates": {
            "prior": prior_gates,
            "current": current_gates,
            "delta": gate_delta,
        },
        "catalyst_strength": {
            "prior": prior_strength,
            "current": current_strength,
            "delta": strength_delta,
            "degraded": degraded,
            "improved": improved,
        },
        "portfolio_churn": churn,
        "margin_summary": {
            "prior": prior_margins,
            "current": current_margins,
            "delta": margin_delta,
        },
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------
def generate_drift_report_md(
    metrics: Dict[str, Any],
    status: str,
    reasons: List[str],
    guardrails: DriftGuardrails,
    rollback_candidate: Optional[Dict[str, Any]] = None,
    recommended_action: str = "NONE",
    adaptive_warnings: Optional[List[str]] = None,
    attribution: Optional[Dict[str, Any]] = None,
    suggestions: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Generate a Markdown drift report."""
    current = metrics.get("current", {})
    rolling = metrics.get("rolling", {})
    n_snaps = len(metrics.get("snapshots", []))

    lines: List[str] = []
    lines.append("# Daily Drift Report")
    lines.append(
        f"Date: {current.get('date', 'N/A')} | "
        f"Ruleset: {current.get('ruleset_id', 'N/A')} | "
        f"Window: {n_snaps} snapshots"
    )
    lines.append("")

    # Current snapshot table
    lines.append("## Current Snapshot")
    lines.append("")
    lines.append("| Metric                    | Value  |")
    lines.append("|---------------------------|--------|")

    def _fmt(val: Any, decimals: int = 1) -> str:
        if val is None:
            return "N/A"
        if isinstance(val, float):
            if abs(val) < 0.01 and val != 0.0:
                return f"{val:.4f}"
            return f"{val:.{decimals}f}"
        return str(val)

    rows = [
        ("A-tier count (dev)", current.get("tier_A_count")),
        ("A-tier % (dev)", f"{current.get('tier_A_pct', 'N/A')}%"
         if current.get("tier_A_pct") is not None else "N/A"),
        ("B-tier count (dev)", current.get("tier_B_count")),
        ("Eligible % (dev)", f"{current.get('eligible_pct', 'N/A')}%"
         if current.get("eligible_pct") is not None else "N/A"),
        ("Catalyst missing (elig)", f"{current.get('catalyst_missing_pct', 'N/A')}%"
         if current.get("catalyst_missing_pct") is not None else "N/A"),
        ("Backfill share (elig)", f"{current.get('catalyst_backfill_share_pct', 'N/A')}%"
         if current.get("catalyst_backfill_share_pct") is not None else "N/A"),
        ("Drawdown coverage (dev)", f"{current.get('drawdown_coverage_pct', 'N/A')}%"
         if current.get("drawdown_coverage_pct") is not None else "N/A"),
        ("Rel DD coverage (dev)", f"{current.get('drawdown_rel_coverage_pct', 'N/A')}%"
         if current.get("drawdown_rel_coverage_pct") is not None else "N/A"),
        ("Top-25 overlap (vs prior)", f"{current.get('top25_overlap_pct', 'N/A')}%"
         if current.get("top25_overlap_pct") is not None else "N/A"),
        ("Optionality std", _fmt(current.get("optionality_std"), 2)),
        ("Composite IQR", _fmt(current.get("composite_iqr"))),
    ]
    for label, val in rows:
        lines.append(f"| {label:<25} | {str(val):>6} |")
    lines.append("")

    # Cost Context table (only if cost columns present)
    has_cost = current.get("cost_coverage_pct") is not None
    if has_cost:
        lines.append("## Cost Context")
        lines.append("")
        lines.append("| Metric                    | Value  |")
        lines.append("|---------------------------|--------|")

        cov = current.get("cost_coverage_pct")
        lines.append(f"| {'Cost coverage (dev)':<25} | {_fmt(cov)}% |")

        p10 = current.get("est_cost_bps_p10")
        p50 = current.get("est_cost_bps_p50")
        p90 = current.get("est_cost_bps_p90")
        bps_str = f"{_fmt(p10)} / {_fmt(p50)} / {_fmt(p90)} bps"
        lines.append(f"| {'Est cost P10 / P50 / P90':<25} | {bps_str} |")

        lines.append(f"| {'Mean cost mult':<25} | {_fmt(current.get('mean_cost_mult'), 4)} |")

        bk_no = _fmt(current.get("cost_bucket_no_pct"))
        bk_mild = _fmt(current.get("cost_bucket_mild_pct"))
        bk_heavy = _fmt(current.get("cost_bucket_heavy_pct"))
        bk_floor = _fmt(current.get("cost_bucket_floor_pct"))
        lines.append(
            f"| {'Bucket no/mild/heavy/floor':<25} | "
            f"{bk_no}% / {bk_mild}% / {bk_heavy}% / {bk_floor}% |"
        )

        lines.append(f"| {'Cap binding':<25} | {_fmt(current.get('cap_binding_pct'))}% |")
        lines.append("")

    # Returns Source Mix table (only if returns_source data present)
    has_rs = current.get("rs_morningstar_share_pct") is not None
    if has_rs:
        lines.append("## Returns Source Mix")
        lines.append("")
        lines.append("| Source                 | Count | Share |")
        lines.append("|------------------------|-------|-------|")
        for src in _RETURNS_SOURCE_KEYS:
            cnt = current.get(f"rs_{src}_count", 0)
            share = current.get(f"rs_{src}_share_pct")
            share_str = f"{share:.1f}%" if share is not None else "N/A"
            lines.append(f"| {src:<22} | {cnt:>5} | {share_str:>5} |")
        lines.append("")

    # Catalyst Coverage table (only if tier_dev + catalyst_mode present)
    has_cat = current.get("cat_n_dev") is not None
    if has_cat:
        lines.append("## Catalyst Coverage")
        lines.append("")
        n_dev_cat = current.get("cat_n_dev", 0)
        n_elig_cat = current.get("cat_n_eligible", 0)
        elig_share = current.get("cat_eligible_share_pct")
        elig_str = f"{elig_share:.1f}%" if elig_share is not None else "N/A"
        lines.append(f"Dev tickers: {n_dev_cat} | Eligible: {n_elig_cat} ({elig_str})")
        lines.append("Eligible = dev tickers with non-empty `tier_dev`.")
        lines.append("")
        lines.append("| Catalyst Mode     | Count | Share (elig) |")
        lines.append("|-------------------|-------|--------------|")
        for mode in _CATALYST_MODE_KEYS:
            cnt = current.get(f"cat_{mode}_count", 0)
            share = current.get(f"cat_{mode}_share_pct")
            share_str = f"{share:.1f}%" if share is not None else "N/A"
            lines.append(f"| {mode:<17} | {cnt:>5} | {share_str:>12} |")
        lines.append("")

    # Catalyst Source Mix table (only if catalyst_source data present)
    has_cs = current.get("cs_n_eligible") is not None
    if has_cs:
        lines.append("## Catalyst Source Mix")
        lines.append("")
        n_elig_cs = current.get("cs_n_eligible", 0)
        lines.append(f"Eligible dev tickers: {n_elig_cs}")
        lines.append("")
        lines.append("| Source                 | Count | Share (elig) |")
        lines.append("|------------------------|-------|--------------|")
        for src in _CATALYST_SOURCE_KEYS:
            key = src.lower()
            cnt = current.get(f"cs_{key}_count", 0)
            share = current.get(f"cs_{key}_share_pct")
            share_str = f"{share:.1f}%" if share is not None else "N/A"
            lines.append(f"| {src:<22} | {cnt:>5} | {share_str:>12} |")
        lines.append("")

    # Catalyst Event Type Mix table (only if catalyst_event_type data present)
    has_ct = current.get("ct_n_eligible") is not None
    if has_ct:
        lines.append("## Catalyst Event Type Mix")
        lines.append("")
        n_elig_ct = current.get("ct_n_eligible", 0)
        lines.append(f"Eligible dev tickers: {n_elig_ct}")
        lines.append("")
        lines.append("| Event Type             | Count | Share (elig) |")
        lines.append("|------------------------|-------|--------------|")
        for et in _CATALYST_EVENT_TYPE_KEYS:
            key = et.lower()
            cnt = current.get(f"ct_{key}_count", 0)
            share = current.get(f"ct_{key}_share_pct")
            share_str = f"{share:.1f}%" if share is not None else "N/A"
            lines.append(f"| {et:<22} | {cnt:>5} | {share_str:>12} |")
        lines.append("")

    # Catalyst Priority Distribution (only if cat_priority data present)
    has_cp = current.get("cp_n_eligible") is not None
    if has_cp:
        lines.append("## Catalyst Priority Distribution")
        lines.append("")
        n_elig_cp = current.get("cp_n_eligible", 0)
        lines.append(f"Eligible dev tickers: {n_elig_cp}")
        lines.append("")
        lines.append("| Priority | Label    | Count | Share (elig) |")
        lines.append("|----------|----------|-------|--------------|")
        _CP_ROWS = [
            (1, "fda", "FDA"),
            (2, "readout", "Readout"),
            (3, "ongoing", "Ongoing"),
            (9, "none", "None"),
            (99, "unknown", "Unknown"),
        ]
        for prio, key, label in _CP_ROWS:
            cnt = current.get(f"cp_{key}_count", 0)
            share = current.get(f"cp_{key}_share_pct")
            share_str = f"{share:.1f}%" if share is not None else "N/A"
            lines.append(f"| {prio:>8} | {label:<8} | {cnt:>5} | {share_str:>12} |")
        lines.append("")

    # Rolling window table
    if rolling:
        lines.append(f"## Rolling Window (last {n_snaps} runs)")
        lines.append("")
        lines.append("| Metric              | Min   | Max   | Mean  | Median | IQR   | Delta | Current |")
        lines.append("|---------------------|-------|-------|-------|--------|-------|-------|---------|")
        for key in [
            "tier_A_pct", "catalyst_missing_pct",
            "top25_overlap_pct", "optionality_std",
            "catalyst_strength_near_pct",
        ]:
            if key not in rolling:
                continue
            r = rolling[key]
            label = key.replace("_", " ").replace("tier ", "").title()
            lines.append(
                f"| {label:<19} "
                f"| {r['min']:>5} "
                f"| {r['max']:>5} "
                f"| {r['mean']:>5} "
                f"| {_fmt(r.get('median')):>6} "
                f"| {_fmt(r.get('iqr')):>5} "
                f"| {_fmt(r.get('delta')):>5} "
                f"| {_fmt(r['current']):>7} |"
            )
        lines.append("")

    # Mixed rulesets warning
    rulesets_in_window = set(
        s.get("ruleset_id", "") for s in metrics.get("snapshots", [])
    )
    rulesets_in_window.discard("")
    if len(rulesets_in_window) > 1:
        lines.append("## Warning: Mixed Rulesets in Window")
        lines.append(f"Rulesets observed: {', '.join(sorted(rulesets_in_window))}")
        lines.append("")

    # Adaptive Warnings
    if adaptive_warnings:
        lines.append("## Adaptive Warnings")
        lines.append("")
        for w in adaptive_warnings:
            lines.append(f"- {w}")
        lines.append("")

    # Drift Attribution (pairwise diff of last two snapshots)
    if attribution is not None:
        lines.append("## Drift Attribution")
        lines.append(
            f"Comparing {attribution['prior_date']} → {attribution['current_date']}"
        )
        lines.append("")

        # Eligibility Gate Changes
        eg = attribution.get("eligibility_gates", {})
        if eg:
            lines.append("### Eligibility Gate Changes")
            lines.append("")
            lines.append("| Gate | Prior | Current | Delta |")
            lines.append("|------|-------|---------|-------|")
            for gate in _GATE_KEYS:
                p = eg.get("prior", {}).get(gate, 0)
                c = eg.get("current", {}).get(gate, 0)
                d = eg.get("delta", {}).get(gate, 0)
                delta_str = f"+{d}" if d > 0 else str(d)
                lines.append(f"| {gate} | {p} | {c} | {delta_str} |")
            lines.append("")

        # Catalyst Strength Shifts
        cs = attribution.get("catalyst_strength", {})
        if cs:
            lines.append("### Catalyst Strength Shifts")
            lines.append("")
            lines.append("| Band | Prior | Current | Delta |")
            lines.append("|------|-------|---------|-------|")
            for band in ("near", "mid", "far", "missing"):
                p = cs.get("prior", {}).get(band, 0)
                c = cs.get("current", {}).get(band, 0)
                d = cs.get("delta", {}).get(band, 0)
                delta_str = f"+{d}" if d > 0 else str(d)
                lines.append(f"| {band} | {p} | {c} | {delta_str} |")
            lines.append("")

            degraded = cs.get("degraded", [])
            if degraded:
                lines.append("**Degraded:**")
                for e in degraded:
                    lines.append(f"- {e['ticker']}: {e['prior']} → {e['current']}")
                lines.append("")

            improved = cs.get("improved", [])
            if improved:
                lines.append("**Improved:**")
                for e in improved:
                    lines.append(f"- {e['ticker']}: {e['prior']} → {e['current']}")
                lines.append("")

        # Portfolio Churn
        churn = attribution.get("portfolio_churn", {})
        if churn:
            lines.append("### Portfolio Churn")
            lines.append(f"Top-25 overlap: {churn.get('overlap_count', 0)}/25")
            lines.append("")

            dropped = churn.get("dropped", [])
            if dropped:
                lines.append("**Dropped:**")
                lines.append("")
                lines.append("| Ticker | Tier | Band | Reason |")
                lines.append("|--------|------|------|--------|")
                for d in dropped:
                    lines.append(
                        f"| {d['ticker']} | {d['tier_dev']} | {d['size_band']} | {d['tier_reason']} |"
                    )
                lines.append("")

            added = churn.get("added", [])
            if added:
                lines.append("**Added:**")
                lines.append("")
                lines.append("| Ticker | Tier | Band | Reason |")
                lines.append("|--------|------|------|--------|")
                for a in added:
                    lines.append(
                        f"| {a['ticker']} | {a['tier_dev']} | {a['size_band']} | {a['tier_reason']} |"
                    )
                lines.append("")

        # Gate Margin Shifts
        ms = attribution.get("margin_summary", {})
        prior_ms = ms.get("prior", {})
        current_ms = ms.get("current", {})
        delta_ms = ms.get("delta", {})
        # Only show if at least one non-None value exists
        has_data = any(
            v is not None
            for v in list(prior_ms.values()) + list(current_ms.values())
        )
        if has_data:
            lines.append("### Gate Margin Shifts")
            lines.append("")
            lines.append("| Metric | Prior | Current | Delta |")
            lines.append("|--------|-------|---------|-------|")
            margin_rows = [
                ("Median dd_abs_margin", "median_dd_abs_margin"),
                ("P10 dd_abs_margin", "p10_dd_abs_margin"),
                ("Median dd_rel_margin", "median_dd_rel_margin"),
                ("Rescued count", "rescued_count"),
                ("DD rel-margin rescue count", "dd_rel_margin_rescue_count"),
            ]
            for label, key in margin_rows:
                p = prior_ms.get(key)
                c = current_ms.get(key)
                d = delta_ms.get(key)
                p_s = f"{p:+.4f}" if isinstance(p, float) else (str(p) if p is not None else "N/A")
                c_s = f"{c:+.4f}" if isinstance(c, float) else (str(c) if c is not None else "N/A")
                d_s = f"{d:+.4f}" if isinstance(d, float) else (f"+{d}" if isinstance(d, int) and d > 0 else str(d) if d is not None else "N/A")
                lines.append(f"| {label} | {p_s} | {c_s} | {d_s} |")
            lines.append("")

        # Gate Pressure (share within ±5pp of thresholds)
        pressure_rows = [
            ("DD abs near gate %", "dd_abs_near_gate_pct"),
            ("DD rel near gate %", "dd_rel_near_gate_pct"),
            ("Optionality near A-floor %", "optionality_near_a_floor_pct"),
            ("Rescued share %", "rescued_share_pct"),
            ("DD rel-margin rescue share %", "dd_rel_margin_rescue_share_pct"),
        ]
        has_pressure = any(
            current_ms.get(k) is not None or prior_ms.get(k) is not None
            for _, k in pressure_rows
        )
        if has_pressure:
            lines.append("### Gate Pressure")
            lines.append("")
            lines.append("Share of dev tickers within ±5pp of each gate threshold.")
            lines.append("")
            lines.append("| Metric | Prior | Current | Delta |")
            lines.append("|--------|-------|---------|-------|")
            for label, key in pressure_rows:
                p = prior_ms.get(key)
                c = current_ms.get(key)
                d = delta_ms.get(key)
                p_s = f"{p:.1f}%" if isinstance(p, (int, float)) else "N/A"
                c_s = f"{c:.1f}%" if isinstance(c, (int, float)) else "N/A"
                d_s = f"{d:+.1f}pp" if isinstance(d, (int, float)) else "N/A"
                lines.append(f"| {label} | {p_s} | {c_s} | {d_s} |")
            lines.append("")

    # Guardrails
    lines.append(f"## Guardrails: {status}")
    lines.append(f"Recommended action: {recommended_action}")
    lines.append("")
    if reasons:
        for r in reasons:
            lines.append(f"- {r}")
    if rollback_candidate:
        lines.append("")
        lines.append(f"**Rollback candidate**: {rollback_candidate['id']} "
                      f"({rollback_candidate.get('file', 'N/A')})")
        lines.append(f"  To rollback: `python scripts/promote_ruleset.py "
                      f"{rollback_candidate['id']} --rollback --force`")
    lines.append("")

    # Guardrails config
    lines.append("## Guardrails Config")
    lines.append(f"ID: {guardrails.guardrails_id}")
    lines.append(f"- fail_a_pct_low: {guardrails.fail_a_pct_low}%")
    lines.append(f"- warn_a_pct_low: {guardrails.warn_a_pct_low}%")
    lines.append(f"- fail_a_pct_high: {guardrails.fail_a_pct_high}%")
    lines.append(f"- fail_catalyst_missing_high: {guardrails.fail_catalyst_missing_high}%")
    lines.append(f"- fail_overlap_low: {guardrails.fail_overlap_low}%")
    lines.append(f"- fail_dispersion_low: {guardrails.fail_dispersion_low}")
    lines.append(f"- warn_median_cost_bps_high: {guardrails.warn_median_cost_bps_high} bps")
    lines.append(f"- warn_cost_coverage_low: {guardrails.warn_cost_coverage_low}%")
    lines.append(f"- warn_cap_binding_high: {guardrails.warn_cap_binding_high}%")
    lines.append(f"- warn_backfill_share_high: {guardrails.warn_backfill_share_high}%")
    lines.append(f"- warn_dd_rel_margin_rescue_share_high: {guardrails.warn_dd_rel_margin_rescue_share_high}%")
    lines.append(f"- warn_rs_unknown_share_high: {guardrails.warn_rs_unknown_share_high}%")
    lines.append(f"- warn_rs_csv_outlier_override_share_high: {guardrails.warn_rs_csv_outlier_override_share_high}%")
    lines.append(f"- warn_rs_morningstar_share_low: {guardrails.warn_rs_morningstar_share_low}%")
    lines.append(f"- warn_cat_eligible_share_low: {guardrails.warn_cat_eligible_share_low}%")
    lines.append(f"- warn_cat_specific_days_share_low: {guardrails.warn_cat_specific_days_share_low}%")
    lines.append(f"- warn_cs_ctgov_share_low: {guardrails.warn_cs_ctgov_share_low}%")
    lines.append(f"- warn_cs_unknown_share_high: {guardrails.warn_cs_unknown_share_high}%")
    lines.append(f"- warn_ct_unknown_share_high: {guardrails.warn_ct_unknown_share_high}%")
    lines.append(f"- warn_ct_fda_share_spike: {guardrails.warn_ct_fda_share_spike}%")
    lines.append(f"- warn_iqr_k: {guardrails.warn_iqr_k}")
    lines.append(f"- warn_iqr_floor: {guardrails.warn_iqr_floor}")
    lines.append(f"- warn_min_window: {guardrails.warn_min_window}")
    lines.append(f"- fail_corroboration_count: {guardrails.fail_corroboration_count}")
    lines.append("")

    # Suggested guardrails (informational only — never affects status)
    if suggestions:
        lines.append("## Suggested Guardrails")
        lines.append("")
        lines.append("Data-driven suggestions from prior snapshots (excludes current).")
        lines.append("These are **informational only** and do not affect drift status.")
        lines.append("")
        lines.append("| Metric | N | Median | IQR | Suggested | Current | Tighter? |")
        lines.append("|--------|---|--------|-----|-----------|---------|----------|")
        for s in suggestions:
            cur = s["current_threshold"]
            cur_str = f"{cur}%" if cur is not None else "N/A"
            if s.get("suppressed"):
                lines.append(
                    f"| {s['label']} | {s['n']} |  |  "
                    f"| _suppressed ({s['suppressed_reason']})_ "
                    f"| {cur_str} |  |"
                )
                continue
            tighter_flag = "**yes**" if s["tighter"] else ""
            lines.append(
                f"| {s['label']} | {s['n']} | {s['median']}% "
                f"| {s['iqr']} | {s['suggested']}% "
                f"| {cur_str} | {tighter_flag} |"
            )
        # Show suppression config so the report is self-explanatory
        any_suppressed = any(s.get("suppressed") for s in suggestions)
        if any_suppressed:
            lines.append(
                f"Suppression threshold: "
                f"suggest_cs_min_cat_specific_days_median ="
                f" {_SUGGEST_CS_MIN_CAT_SPECIFIC_DAYS_MEDIAN:.1f}%"
            )
        lines.append("")

    return "\n".join(lines)


def generate_drift_json(
    metrics: Dict[str, Any],
    status: str,
    reasons: List[str],
    guardrails: DriftGuardrails,
    rollback_candidate: Optional[Dict[str, Any]] = None,
    recommended_action: str = "NONE",
    adaptive_warnings: Optional[List[str]] = None,
    attribution: Optional[Dict[str, Any]] = None,
    suggestions: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Generate a JSON-serializable drift report dict."""
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "reasons": reasons,
        "recommended_action": recommended_action,
        "rollback_candidate": rollback_candidate,
        "adaptive_warnings": adaptive_warnings or [],
        "attribution": attribution,
        "suggested_guardrails": suggestions or [],
        "guardrails": guardrails.to_dict(),
        "current": metrics.get("current", {}),
        "rolling": metrics.get("rolling", {}),
        "window_size": len(metrics.get("snapshots", [])),
        "snapshots": metrics.get("snapshots", []),
    }


def generate_rollback_packet_md(
    status: str,
    reasons: List[str],
    rollback_candidate: Optional[Dict[str, Any]],
    recommended_action: str,
    current_metrics: Dict[str, Any],
    guardrails: DriftGuardrails,
) -> str:
    """Generate a self-contained rollback ops page (only called on FAIL)."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines: List[str] = []
    lines.append("# Rollback Packet")
    lines.append(f"Generated: {ts} | Status: {status} | Action: {recommended_action}")
    lines.append("")

    # --- Guardrails tripped ---
    lines.append("## Guardrails Tripped")
    lines.append("")
    lines.append("| Guardrail | Threshold | Actual | Status |")
    lines.append("|-----------|-----------|--------|--------|")
    for reason in reasons:
        # Parse reason strings like "A-tier % = 1.5% < 2.0% floor"
        lines.append(f"| {reason} | - | - | FAIL |")
    lines.append("")

    if recommended_action == "ROLLBACK_RECOMMENDED" and rollback_candidate:
        # --- Recommended rollback ---
        rid = rollback_candidate["id"]
        rfile = rollback_candidate.get("file", "N/A")
        engine_ver = rollback_candidate.get("engine_version", "N/A")
        desc = rollback_candidate.get("description", "N/A")

        lines.append("## Recommended Rollback")
        lines.append("")
        lines.append("| Field | Value |")
        lines.append("|-------|-------|")
        lines.append(f"| Ruleset ID | `{rid}` |")
        lines.append(f"| File | `production_data/decision_rulesets/{rfile}` |")
        lines.append(f"| Engine version | {engine_ver} |")
        lines.append(f"| Description | {desc} |")
        lines.append("")

        # --- Fingerprint verification ---
        lines.append("## Fingerprint Verification")
        lines.append("")
        lines.append("```bash")
        lines.append(
            f'python -c "import json; '
            f"d=json.load(open('production_data/decision_rulesets/{rfile}')); "
            f"assert d['ruleset_id']=='{rid}', f'ID mismatch: {{d[\"ruleset_id\"]}}'; "
            f"print('OK:', d['ruleset_id'])\""
        )
        lines.append("```")
        lines.append("")

        # --- Rollback commands ---
        lines.append("## Rollback Commands")
        lines.append("")
        lines.append(
            f"1. Verify fingerprint (above)"
        )
        lines.append(
            f"2. `python scripts/promote_ruleset.py {rid} --rollback --force`"
        )
        lines.append(
            "3. Re-run screen: `python run_screen.py --strict`"
        )
        lines.append(
            "4. Re-run drift: `python scripts/run_drift_report.py "
            "--snapshot-dir data/snapshots --output-dir output`"
        )
        lines.append("")
    else:
        # No candidate available
        lines.append("## No Rollback Candidate Available")
        lines.append("")
        lines.append("No retired ruleset available. Manual investigation required.")
        lines.append("")

    # --- Current snapshot ---
    lines.append("## Current Snapshot")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    for key in (
        "tier_A_pct", "tier_B_pct", "catalyst_missing_pct",
        "top25_overlap_pct", "optionality_std",
    ):
        val = current_metrics.get(key)
        if val is not None:
            label = key.replace("_", " ").title()
            lines.append(f"| {label} | {val} |")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate daily drift report for post-promotion monitoring."
    )

    # Input source: snapshot directory OR panel CSV (mutually exclusive)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--snapshot-dir",
        type=Path,
        help="Directory containing dated snapshot subdirectories.",
    )
    source.add_argument(
        "--panel",
        type=Path,
        help="Walkforward panel CSV (as_of_date column required).",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write drift_report.json and drift_report.md.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=5,
        help="Number of most recent snapshots to include (default: 5).",
    )
    parser.add_argument(
        "--guardrails",
        type=str,
        default=None,
        help="Path to guardrails JSON override (optional).",
    )
    args = parser.parse_args()

    # Load guardrails
    if args.guardrails:
        guardrails = DriftGuardrails.from_json(args.guardrails)
    else:
        guardrails = DriftGuardrails(window_size=args.window_size)

    window_size = guardrails.window_size

    # Load snapshots (from directory or panel)
    if args.panel:
        snapshots = load_panel_as_snapshots(args.panel, window_size)
        source_label = f"panel: {args.panel.name}"
    else:
        snapshots = load_snapshot_window(args.snapshot_dir, window_size)
        source_label = f"snapshot-dir: {args.snapshot_dir}"

    if not snapshots:
        print(f"No loadable snapshots found ({source_label}).", file=sys.stderr)
        return 1

    print(f"Loaded {len(snapshots)} snapshots (window={window_size}, {source_label})")

    # Compute metrics
    is_panel = args.panel is not None
    metrics = compute_drift_metrics(
        snapshots, strict_cs_missing=not is_panel,
    )

    # Evaluate guardrails (FAIL + static WARN checks)
    gr_status, gr_reasons, rollback, recommended_action = evaluate_guardrails(
        metrics, guardrails
    )

    # Evaluate adaptive warnings (WARN layer — window-relative)
    aw_status, aw_reasons = evaluate_adaptive_warnings(metrics, guardrails)

    # Split guardrail reasons into FAIL vs WARN buckets
    if gr_status == "FAIL":
        fail_reasons = gr_reasons
        warn_reasons = aw_reasons or []
    else:
        fail_reasons = []
        # Merge guardrail WARNs + adaptive WARNs
        warn_reasons = (gr_reasons or []) + (aw_reasons or [])

    # Combine: FAIL > WARN > OK
    if fail_reasons:
        final_status = "FAIL"
    elif warn_reasons:
        final_status = "WARN"
    else:
        final_status = "OK"

    # Compute attribution (pairwise diff of last two snapshots)
    attribution = compute_attribution(snapshots)

    # Compute suggested guardrails (informational — uses prior snapshots only)
    suggestions = compute_suggested_guardrails(
        metrics.get("snapshots", []), guardrails,
    )

    # Generate reports
    md = generate_drift_report_md(
        metrics, final_status, fail_reasons, guardrails, rollback,
        recommended_action, adaptive_warnings=warn_reasons or None,
        attribution=attribution, suggestions=suggestions or None,
    )
    if is_panel and "## Catalyst Source Mix" in md:
        md = md.replace(
            "## Catalyst Source Mix\n\n",
            "## Catalyst Source Mix\n\n"
            "_Panel CSV: empty catalyst_source treated as `none`"
            " (CSV ambiguity)._\n\n",
            1,
        )
    if is_panel and "## Catalyst Event Type Mix" in md:
        md = md.replace(
            "## Catalyst Event Type Mix\n\n",
            "## Catalyst Event Type Mix\n\n"
            "_Panel CSV: empty catalyst_event_type treated as `none`"
            " (CSV ambiguity)._\n\n",
            1,
        )
    if args.panel:
        d0, d1 = snapshots[0].date, snapshots[-1].date
        md = (
            f"> Panel mode: {args.panel.name} | "
            f"dates={len(snapshots)} | range={d0}..{d1} | "
            f"window={window_size}\n\n"
        ) + md
    report_json = generate_drift_json(
        metrics, final_status, fail_reasons, guardrails, rollback,
        recommended_action, adaptive_warnings=warn_reasons or None,
        attribution=attribution, suggestions=suggestions or None,
    )

    # Write outputs
    args.output_dir.mkdir(parents=True, exist_ok=True)
    md_path = args.output_dir / "drift_report.md"
    json_path = args.output_dir / "drift_report.json"

    with open(md_path, "w") as f:
        f.write(md)

    with open(json_path, "w") as f:
        json.dump(report_json, f, indent=2, default=str)
        f.write("\n")

    # Rollback packet (only on FAIL)
    if final_status == "FAIL":
        packet = generate_rollback_packet_md(
            final_status, fail_reasons, rollback, recommended_action,
            metrics.get("current", {}), guardrails,
        )
        packet_path = args.output_dir / "rollback_packet.md"
        packet_path.write_text(packet, encoding="utf-8")
        print(f"Wrote rollback packet: {packet_path}")

    print(f"Drift status: {final_status}")
    if fail_reasons:
        for r in fail_reasons:
            print(f"  FAIL: {r}")
    if warn_reasons:
        for r in warn_reasons:
            print(f"  WARN: {r}")
    if attribution:
        gate_delta = attribution.get("eligibility_gates", {}).get("delta", {})
        nonzero_gates = {k: v for k, v in gate_delta.items() if v != 0}
        if nonzero_gates:
            print(f"  Gate deltas: {nonzero_gates}")
        churn = attribution.get("portfolio_churn", {})
        n_dropped = len(churn.get("dropped", []))
        n_added = len(churn.get("added", []))
        if n_dropped or n_added:
            print(f"  Churn: {n_dropped} dropped, {n_added} added")
    print(f"Reports written to {args.output_dir}/")

    return 1 if final_status == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main())
