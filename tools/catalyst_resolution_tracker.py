#!/usr/bin/env python3
"""Catalyst Resolution Tracker (CRT) — Spec 042.

Closes the prediction -> resolution -> calibration loop by detecting
when binary catalysts resolve and recording structured outcomes.

Phase 1: schemas, watchlist construction, deterministic outcome classification.
Phase 2: source adapters, prediction snapshots, price capture, main runner.

Usage:
    python tools/catalyst_resolution_tracker.py --as-of-date 2026-03-31
    python tools/catalyst_resolution_tracker.py --as-of-date 2026-03-31 --dry-run
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import re
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SCHEMA_VERSION = "1.0.0"

OUTCOMES = frozenset({"HIT", "MISS", "MIXED", "DELAYED", "WITHDRAWN", "NEEDS_REVIEW", "INFORMATIONAL"})

CATALYST_TYPES = frozenset(
    {
        "PDUFA_ACTION",
        "PHASE_3_READOUT",
        "PHASE_2_READOUT",
        "PHASE_1_DATA",
        "ADVISORY_COMMITTEE",
        "NDA_BLA_FILING",
        "REGULATORY_DESIGNATION",
        "CORPORATE_UPDATE",
        "EARNINGS",
        "CONFERENCE_PRESENTATION",
    }
)

SOURCE_TYPES = frozenset({"SEC_8K", "PRESS_RELEASE", "CTGOV_STATUS", "FDA_ACTION", "MANUAL"})

# Detection window: T-30 to T+7 (look back 30 days for past-due catalysts,
# look ahead 7 days for early announcements)
WINDOW_LOOKBACK_DAYS = 30
WINDOW_LOOKAHEAD_DAYS = 7

# Keyword lists for deterministic outcome classification
_HIT_KEYWORDS = [
    "met primary endpoint",
    "positive topline",
    "statistically significant",
    "achieved primary",
    "met the primary",
    "demonstrated superiority",
    "approved",
]

_MISS_KEYWORDS = [
    "did not meet",
    "failed to achieve",
    "not statistically significant",
    "discontinued",
    "discontinuation",
    "terminated",
    "complete response letter",
    "did not achieve",
    "negative topline",
    "clinical hold",
    "partial clinical hold",
    "serious adverse event",
    "safety signal",
    "dose-limiting toxicit",
    "voluntary pause",
    "fatal",
]

# Keywords indicating an informational update, NOT a catalyst resolution.
# These suppress NEEDS_REVIEW in favor of skipping the record entirely.
_INFORMATIONAL_KEYWORDS = [
    "data expected",
    "enrollment continu",
    "enrollment underway",
    "on track to",
    "anticipated in",
    "submission anticipated",
    "submission in",
    "resubmission in",
    "initiating phase",
    "initiated phase",
    "expects to report",
    "expects to initiate",
    "conference call",
    "financial results",
    "quarterly report",
    "annual report",
    "corporate update",
    "investor presentation",
    "business update",
    "regulatory options",
    "development program",
]


@dataclass
class ResolutionRecord:
    """A single catalyst resolution record."""

    ticker: str
    catalyst_date: str
    catalyst_type: str
    resolution_date: Optional[str] = None
    outcome: str = "NEEDS_REVIEW"
    outcome_detail: str = ""
    source_type: str = "MANUAL"
    source_id: str = ""
    catalyst_description: str = ""
    prediction_snapshot_date: Optional[str] = None
    prediction_dem_rank: Optional[int] = None
    prediction_composite_score: Optional[float] = None
    prediction_match_type: Optional[str] = None  # "exact" | "fallback"
    event_ev_node_id: Optional[str] = None
    event_ev_p_hit: Optional[float] = None
    event_ev_p_miss: Optional[float] = None
    event_ev_p_mixed: Optional[float] = None
    event_ev_confidence: Optional[float] = None
    event_ev_asof_date: Optional[str] = None
    event_ev_match_type: Optional[str] = (
        None  # "exact_node" | "ticker_date_7d" | "ambiguous" | "no_match" | "no_snap" | "no_ev_artifact"
    )
    price_t_minus_1: Optional[float] = None
    price_t_0: Optional[float] = None
    price_t_plus_5: Optional[float] = None
    price_direction: Optional[str] = None  # up / down / flat (separate from outcome)
    days_from_expected: Optional[int] = None
    as_of_date: str = ""
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.outcome not in OUTCOMES:
            raise ValueError(f"Invalid outcome: {self.outcome!r}. Must be one of {OUTCOMES}")
        if self.catalyst_type not in CATALYST_TYPES:
            raise ValueError(f"Invalid catalyst_type: {self.catalyst_type!r}. Must be one of {CATALYST_TYPES}")
        if self.source_type not in SOURCE_TYPES:
            raise ValueError(f"Invalid source_type: {self.source_type!r}. Must be one of {SOURCE_TYPES}")

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        d["schema_version"] = self.schema_version
        d["ticker"] = self.ticker
        d["catalyst_date"] = self.catalyst_date
        d["catalyst_type"] = self.catalyst_type
        d["catalyst_description"] = self.catalyst_description
        d["resolution_date"] = self.resolution_date
        d["outcome"] = self.outcome
        d["outcome_detail"] = self.outcome_detail
        d["source_type"] = self.source_type
        d["source_id"] = self.source_id
        d["prediction_snapshot_date"] = self.prediction_snapshot_date
        d["prediction_dem_rank"] = self.prediction_dem_rank
        d["prediction_composite_score"] = self.prediction_composite_score
        d["prediction_match_type"] = self.prediction_match_type
        d["event_ev_node_id"] = self.event_ev_node_id
        d["event_ev_p_hit"] = self.event_ev_p_hit
        d["event_ev_p_miss"] = self.event_ev_p_miss
        d["event_ev_p_mixed"] = self.event_ev_p_mixed
        d["event_ev_confidence"] = self.event_ev_confidence
        d["event_ev_asof_date"] = self.event_ev_asof_date
        d["event_ev_match_type"] = self.event_ev_match_type
        d["price_t_minus_1"] = self.price_t_minus_1
        d["price_t_0"] = self.price_t_0
        d["price_t_plus_5"] = self.price_t_plus_5
        d["price_direction"] = self.price_direction
        d["days_from_expected"] = self.days_from_expected
        d["as_of_date"] = self.as_of_date
        return d


def compute_record_hash(record: ResolutionRecord) -> str:
    """Compute deterministic SHA256 hash of a resolution record."""
    d = record.to_dict()
    # Remove any existing hash field to avoid circularity
    d.pop("record_hash", None)
    canonical = json.dumps(d, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_watchlist(
    catalyst_events: List[Dict[str, Any]],
    as_of_date: date,
    existing_resolutions: Set[Tuple[str, str]],
) -> List[Dict[str, Any]]:
    """Select catalysts in the resolution detection window.

    Window: as_of_date - LOOKBACK to as_of_date + LOOKAHEAD.
    Excludes catalysts already resolved.

    Args:
        catalyst_events: List of dicts with ticker, catalyst_date (str), catalyst_type.
        as_of_date: Current snapshot date.
        existing_resolutions: Set of (ticker, catalyst_date) already resolved.

    Returns:
        Filtered list of events in the detection window.
    """
    window_start = as_of_date - timedelta(days=WINDOW_LOOKBACK_DAYS)
    window_end = as_of_date + timedelta(days=WINDOW_LOOKAHEAD_DAYS)

    result = []
    for event in catalyst_events:
        ticker = event.get("ticker", "")
        cat_date_str = event.get("catalyst_date", "")
        if not ticker or not cat_date_str:
            continue

        try:
            cat_date = date.fromisoformat(cat_date_str[:10])
        except ValueError:
            continue

        if cat_date < window_start or cat_date > window_end:
            continue

        if (ticker, cat_date_str[:10]) in existing_resolutions:
            continue

        result.append(event)

    return result


def classify_outcome(
    catalyst_type: str,
    *,
    headline: str = "",
    fda_action: Optional[str] = None,
    ctgov_status_from: Optional[str] = None,
    ctgov_status_to: Optional[str] = None,
) -> str:
    """Deterministic rules-based outcome classification.

    CRITICAL: This is keyword matching, not LLM inference. If keywords
    are ambiguous, returns NEEDS_REVIEW for human classification.
    """
    # FDA action (PDUFA)
    if catalyst_type == "PDUFA_ACTION" and fda_action:
        fda_upper = fda_action.upper()
        if fda_upper == "APPROVED":
            return "HIT"
        if fda_upper in ("CRL", "COMPLETE_RESPONSE_LETTER"):
            return "MISS"

    # CT.gov status transitions
    if ctgov_status_to:
        status_to = ctgov_status_to.upper()
        if status_to in ("TERMINATED", "SUSPENDED"):
            return "MISS"
        if status_to == "WITHDRAWN":
            return "MISS"
        # COMPLETED alone is ambiguous — need headline to determine HIT/MISS
        if status_to == "COMPLETED" and not headline:
            return "NEEDS_REVIEW"

    # Headline keyword matching
    if headline:
        headline_lower = headline.lower()

        # Check informational first — these are NOT resolutions
        if any(kw in headline_lower for kw in _INFORMATIONAL_KEYWORDS):
            return "INFORMATIONAL"

        for kw in _HIT_KEYWORDS:
            if kw in headline_lower:
                return "HIT"

        for kw in _MISS_KEYWORDS:
            if kw in headline_lower:
                return "MISS"

    # FDA action without known result
    if catalyst_type == "PDUFA_ACTION" and fda_action is None:
        return "NEEDS_REVIEW"

    return "NEEDS_REVIEW"


# ---------------------------------------------------------------------------
# Phase 2: Source adapters
# ---------------------------------------------------------------------------


def _safe_float(v: Any) -> Optional[float]:
    """Best-effort float coercion. Returns None for non-numeric or NaN."""
    try:
        f = float(v)
        return None if f != f else f
    except (TypeError, ValueError):
        return None


_EV_FULL_PATTERN = re.compile(r"^(\d{4}-\d{2}-\d{2})_event_ev_full\.json$")
_EV_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "event_ev"
_EV_DATE_WINDOW = 7  # days: (ticker, expected_date ±7d) fallback window


def _bind_event_ev_p_hit(
    ticker: str,
    catalyst_date_str: str,
    prediction_snapshot_date: Optional[str],
    node_id: Optional[str] = None,
    ev_artifacts_dir: Path = _EV_ARTIFACTS_DIR,
) -> Dict[str, Any]:
    """Bind event-level p_hit from EV artifacts into a resolution record.

    Uses the closest EV artifact dated <= prediction_snapshot_date to avoid
    look-ahead. Primary strategy: exact node_id match. Fallback: (ticker,
    expected_date ±7d). Leaves all fields None if no unambiguous match is found.
    Never fabricates values.

    Returns a dict with the seven event_ev_* fields.
    """
    null = {
        "event_ev_node_id": None,
        "event_ev_p_hit": None,
        "event_ev_p_miss": None,
        "event_ev_p_mixed": None,
        "event_ev_confidence": None,
        "event_ev_asof_date": None,
        "event_ev_match_type": "no_match",
    }

    if not prediction_snapshot_date:
        null["event_ev_match_type"] = "no_snap"
        return null

    if not ev_artifacts_dir.exists():
        null["event_ev_match_type"] = "no_ev_artifact"
        return null

    # Collect valid artifact dates <= prediction_snapshot_date, newest first
    valid: List[Tuple[str, Path]] = []
    for fp in ev_artifacts_dir.iterdir():
        m = _EV_FULL_PATTERN.match(fp.name)
        if m and m.group(1) <= prediction_snapshot_date:
            valid.append((m.group(1), fp))
    if not valid:
        null["event_ev_match_type"] = "no_ev_artifact"
        return null

    ev_asof, ev_fp = max(valid, key=lambda x: x[0])

    try:
        with open(ev_fp) as f:
            ev_data = json.load(f)
    except Exception:
        null["event_ev_match_type"] = "no_ev_artifact"
        return null

    events = ev_data.get("events", [])
    if not events:
        null["event_ev_match_type"] = "no_ev_artifact"
        return null

    # Primary: exact node_id match (immune to expected_date slippage)
    if node_id:
        for e in events:
            if not isinstance(e, dict):
                continue
            n = e.get("node", {})
            if isinstance(n, dict) and n.get("node_id") == node_id:
                outcome = e.get("outcome", {})
                return {
                    "event_ev_node_id": node_id,
                    "event_ev_p_hit": _safe_float(outcome.get("p_hit")),
                    "event_ev_p_miss": _safe_float(outcome.get("p_miss")),
                    "event_ev_p_mixed": _safe_float(outcome.get("p_mixed")),
                    "event_ev_confidence": _safe_float(outcome.get("confidence")),
                    "event_ev_asof_date": ev_asof,
                    "event_ev_match_type": "exact_node",
                }

    # Fallback: (ticker, expected_date ±7d)
    try:
        cat_dt = date.fromisoformat(catalyst_date_str[:10])
    except ValueError:
        return null

    matches: List[Tuple[int, str, Dict[str, Any]]] = []
    for e in events:
        if not isinstance(e, dict):
            continue
        n = e.get("node", {})
        if not isinstance(n, dict) or n.get("ticker") != ticker:
            continue
        ed_str = (n.get("expected_date") or "")[:10]
        if not ed_str:
            continue
        try:
            ev_dt = date.fromisoformat(ed_str)
        except ValueError:
            continue
        dist = abs((ev_dt - cat_dt).days)
        if dist <= _EV_DATE_WINDOW:
            matches.append((dist, n.get("node_id") or "", e.get("outcome", {})))

    if not matches:
        null["event_ev_match_type"] = "no_match"
        return null

    if len(matches) > 1:
        null["event_ev_match_type"] = "ambiguous"
        return null

    _dist, node_id, outcome = matches[0]
    return {
        "event_ev_node_id": node_id or None,
        "event_ev_p_hit": _safe_float(outcome.get("p_hit")),
        "event_ev_p_miss": _safe_float(outcome.get("p_miss")),
        "event_ev_p_mixed": _safe_float(outcome.get("p_mixed")),
        "event_ev_confidence": _safe_float(outcome.get("confidence")),
        "event_ev_asof_date": ev_asof,
        "event_ev_match_type": "ticker_date_7d",
    }


def _compute_price_direction(
    t_minus_1: Optional[float], t_0: Optional[float], threshold: float = 0.02
) -> Optional[str]:
    """Classify price direction from T-1 to T0. Separate from event outcome."""
    if t_minus_1 is None or t_0 is None or t_minus_1 <= 0:
        return None
    ret = (t_0 - t_minus_1) / t_minus_1
    if ret > threshold:
        return "up"
    if ret < -threshold:
        return "down"
    return "flat"


def load_8k_events(cache_dir: Path, as_of_date: date) -> List[Dict[str, Any]]:
    """Load 8-K catalyst events from the SEC cache."""
    candidates = sorted(cache_dir.glob("8k_catalysts_*.json"), reverse=True)
    for f in candidates:
        parts = f.stem.split("_")
        if len(parts) >= 3:
            file_date = parts[2]
            if file_date <= as_of_date.isoformat():
                with open(f) as fh:
                    data = json.load(fh)
                if isinstance(data, list):
                    return data
    return []


def load_ctgov_trials(cache_dir: Path, as_of_date: date) -> List[Dict[str, Any]]:
    """Load CTGov trial records from cache."""
    trial_path = cache_dir / f"trial_records_{as_of_date.isoformat()}.json"
    if not trial_path.exists():
        candidates = sorted(cache_dir.glob("trial_records_*.json"), reverse=True)
        candidates = [c for c in candidates if c.stem.split("_")[-1] <= as_of_date.isoformat()]
        trial_path = candidates[0] if candidates else None
    if trial_path and trial_path.exists():
        with open(trial_path) as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    return []


def load_pdufa_dates(path: Path) -> List[Dict[str, Any]]:
    """Load PDUFA dates from production data."""
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def load_existing_resolutions(resolutions_dir: Path) -> Set[Tuple[str, str]]:
    """Load set of (ticker, catalyst_date) from existing resolution records."""
    resolved: Set[Tuple[str, str]] = set()
    if not resolutions_dir.exists():
        return resolved
    for f in resolutions_dir.glob("**/*.json"):
        if f.name in ("calibration_summary.json", "manual_overrides.json", "watchlist_current.json"):
            continue
        try:
            with open(f) as fh:
                rec = json.load(fh)
            ticker = rec.get("ticker", "")
            cat_date = rec.get("catalyst_date", "")
            if ticker and cat_date:
                resolved.add((ticker, cat_date[:10]))
        except Exception:
            continue
    return resolved


def load_manual_overrides(resolutions_dir: Path) -> Dict[Tuple[str, str], Dict]:
    """Load manual override entries from resolutions dir and production_data."""
    overrides: Dict[Tuple[str, str], Dict] = {}
    # Check both locations: resolutions dir and production_data (tracked in git)
    candidates = [
        resolutions_dir / "manual_overrides.json",
        PROJECT_ROOT / "production_data" / "crt_manual_overrides.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            with open(path) as f:
                data = json.load(f)
            entries = data if isinstance(data, list) else data.get("overrides", [])
            for entry in entries:
                key = (entry.get("ticker", ""), entry.get("catalyst_date", "")[:10])
                overrides[key] = entry
        except Exception:
            continue
    return overrides


def get_prediction_snapshot(
    ticker: str,
    catalyst_date: date,
    snapshots_dir: Path,
    max_lookback_days: int = 14,
) -> Dict[str, Any]:
    """Find the most recent rankings snapshot before catalyst_date containing ticker.

    Scans backward from catalyst_date - 1 day through up to max_lookback_days
    calendar days. Returns the first snapshot where the ticker appears.
    match_type is "exact" when the ticker is in the immediate prior snapshot,
    "fallback" when an older snapshot was needed (e.g. ticker recently dropped
    from universe). Returns TICKER_NOT_IN_SNAPSHOT if no snapshot within the
    window contains the ticker.
    """
    snap_dates = sorted(
        [
            d.name
            for d in snapshots_dir.iterdir()
            if d.is_dir()
            and not d.name.startswith("_")
            and "__pre" not in d.name
            and len(d.name) == 10
            and (d / "rankings.csv").exists()
        ],
        reverse=True,
    )

    target = catalyst_date.isoformat()
    # Collect candidate snapshot dates within the lookback window, newest first
    candidates = []
    for sd in snap_dates:
        if sd >= target:
            continue
        if (catalyst_date - date.fromisoformat(sd)).days > max_lookback_days:
            break
        candidates.append(sd)

    if not candidates:
        return {"status": "MISSING_SNAPSHOT"}

    first_snap = candidates[0]
    for i, snapshot_date in enumerate(candidates):
        rankings_path = snapshots_dir / snapshot_date / "rankings.csv"
        with open(rankings_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("ticker") == ticker:
                    match_type = "exact" if snapshot_date == first_snap else "fallback"
                    return {
                        "status": "OK",
                        "snapshot_date": snapshot_date,
                        "match_type": match_type,
                        "dem_rank": row.get("actionable_rank", ""),
                        "tier": row.get("tier_any", ""),
                        "composite_score": row.get("composite_score", ""),
                    }

    return {"status": "TICKER_NOT_IN_SNAPSHOT", "snapshot_date": first_snap}


def get_price_reaction(
    ticker: str,
    resolution_date: date,
    prices: Dict[str, Dict[str, float]],
    as_of_date: date,
) -> Dict[str, Optional[float]]:
    """Capture T-1, T+0, T+5 prices around resolution date."""
    tp = prices.get(ticker, {})
    sorted_dates = sorted(tp.keys())
    if not sorted_dates:
        return {"price_t_minus_1": None, "price_t_0": None, "price_t_plus_5": None}

    def _closest(target_date: date, offset: int) -> Optional[float]:
        t = (target_date + timedelta(days=offset)).isoformat()
        if t > as_of_date.isoformat():
            return None
        candidates = [d for d in sorted_dates if d <= t]
        return tp[candidates[-1]] if candidates else None

    return {
        "price_t_minus_1": _closest(resolution_date, -1),
        "price_t_0": _closest(resolution_date, 0),
        "price_t_plus_5": _closest(resolution_date, 7),
    }


def build_catalyst_calendar(
    pdufa_entries: List[Dict[str, Any]],
    sec_8k_events: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Build unified catalyst calendar from PDUFA + 8-K sources."""
    calendar: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, str]] = set()

    type_map = {
        "DATA_READOUT": "PHASE_3_READOUT",
        "FDA_PDUFA_DATE": "PDUFA_ACTION",
        "FDA_ADVISORY_COMMITTEE": "ADVISORY_COMMITTEE",
        "NDA_BLA_FILING": "NDA_BLA_FILING",
        "REGULATORY_DESIGNATION": "REGULATORY_DESIGNATION",
    }

    for p in pdufa_entries:
        ticker = p.get("ticker", "")
        pdufa_date = p.get("pdufa_date", "")
        if ticker and pdufa_date:
            key = (ticker, pdufa_date[:10])
            if key not in seen:
                calendar.append(
                    {
                        "ticker": ticker,
                        "catalyst_date": pdufa_date[:10],
                        "catalyst_type": "PDUFA_ACTION",
                        "description": f"PDUFA: {p.get('drug_name', '')} — {p.get('indication', '')}",
                        "source": "PDUFA",
                        "is_hard": True,
                    }
                )
                seen.add(key)

    for ev in sec_8k_events:
        ticker = ev.get("ticker", "")
        ev_date = ev.get("event_date", "")
        ev_type = ev.get("event_type", "")
        if not ticker or not ev_date:
            continue
        key = (ticker, ev_date[:10])
        if key in seen:
            continue
        cat_type = type_map.get(ev_type, "CORPORATE_UPDATE")
        if cat_type not in CATALYST_TYPES:
            cat_type = "CORPORATE_UPDATE"
        calendar.append(
            {
                "ticker": ticker,
                "catalyst_date": ev_date[:10],
                "catalyst_type": cat_type,
                "description": ev.get("event_name", ""),
                "source": "SEC_8K",
                "is_hard": ev.get("confidence", "") == "HIGH",
            }
        )
        seen.add(key)

    return calendar


def check_8k_for_resolution(
    ticker: str,
    sec_8k_events: List[Dict[str, Any]],
    catalyst_date: date,
    as_of_date: date,
) -> Optional[Dict[str, Any]]:
    """Check 8-K events for a resolution signal near the catalyst date."""
    window_start = catalyst_date - timedelta(days=7)
    window_end = min(catalyst_date + timedelta(days=30), as_of_date)

    for ev in sec_8k_events:
        if ev.get("ticker") != ticker:
            continue
        ev_date_str = ev.get("disclosed_at", ev.get("event_date", ""))
        if not ev_date_str:
            continue
        try:
            ev_date = date.fromisoformat(ev_date_str[:10])
        except ValueError:
            continue
        if window_start <= ev_date <= window_end:
            return {
                "headline": ev.get("event_name", ""),
                "source_type": "SEC_8K",
                "source_id": f"8K_{ev_date_str[:10]}_{ticker}",
                "resolution_date": ev_date_str[:10],
            }
    return None


def check_ctgov_for_resolution(
    ticker: str,
    trial_records: List[Dict[str, Any]],
    catalyst_date: date,
) -> Optional[Dict[str, Any]]:
    """Check CTGov trial status for resolution signals."""
    for tr in trial_records:
        if tr.get("ticker") != ticker:
            continue
        status = tr.get("status", "")
        pcd = tr.get("primary_completion_date", "")
        if not pcd:
            continue
        try:
            pcd_date = date.fromisoformat(pcd[:10])
        except ValueError:
            continue
        if abs((pcd_date - catalyst_date).days) > 30:
            continue
        if status in ("TERMINATED", "WITHDRAWN", "SUSPENDED", "COMPLETED"):
            return {
                "ctgov_status_to": status,
                "source_type": "CTGOV_STATUS",
                "source_id": tr.get("nct_id", ""),
                "resolution_date": catalyst_date.isoformat(),
            }
    return None


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run_crt(
    as_of_date: date,
    snapshots_dir: Path,
    resolutions_dir: Path,
    sec_8k_cache_dir: Path,
    ctgov_cache_dir: Path,
    pdufa_path: Path,
    price_series: Optional[Dict[str, Dict[str, float]]] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Run the CRT for a given date. Returns summary dict."""
    resolutions_dir.mkdir(parents=True, exist_ok=True)

    sec_8k_events = load_8k_events(sec_8k_cache_dir, as_of_date)
    trial_records = load_ctgov_trials(ctgov_cache_dir, as_of_date)
    pdufa_entries = load_pdufa_dates(pdufa_path)
    existing = load_existing_resolutions(resolutions_dir)
    overrides = load_manual_overrides(resolutions_dir)

    logger.info(
        "CRT sources: %d 8K events, %d trials, %d PDUFA, %d existing",
        len(sec_8k_events),
        len(trial_records),
        len(pdufa_entries),
        len(existing),
    )

    calendar = build_catalyst_calendar(pdufa_entries, sec_8k_events)
    watchlist = build_watchlist(calendar, as_of_date, existing)
    logger.info("Watchlist: %d catalysts in window", len(watchlist))

    new_records: List[ResolutionRecord] = []

    # Process manual overrides first — these bypass the watchlist entirely.
    # Human-curated outcomes for events the automated calendar may not cover.
    _override_keys_used: Set[Tuple[str, str]] = set()
    for key, ov in overrides.items():
        if key in existing:
            continue
        ticker, cat_date_str = key
        outcome = ov.get("outcome", "")
        if outcome not in OUTCOMES:
            continue
        cat_type = ov.get("catalyst_type", "PHASE_3_READOUT")
        if cat_type not in CATALYST_TYPES:
            cat_type = "PHASE_3_READOUT"

        snap = get_prediction_snapshot(ticker, date.fromisoformat(cat_date_str), snapshots_dir)
        dem_rank = None
        if snap.get("dem_rank"):
            try:
                dem_rank = int(snap["dem_rank"])
            except (ValueError, TypeError):
                pass

        ev_bind = _bind_event_ev_p_hit(ticker, cat_date_str, snap.get("snapshot_date"))

        prices_data: Dict[str, Optional[float]] = {
            "price_t_minus_1": None,
            "price_t_0": None,
            "price_t_plus_5": None,
        }
        if price_series:
            prices_data = get_price_reaction(ticker, date.fromisoformat(cat_date_str), price_series, as_of_date)

        new_records.append(
            ResolutionRecord(
                ticker=ticker,
                catalyst_date=cat_date_str,
                catalyst_type=cat_type,
                catalyst_description=ov.get("outcome_detail", ""),
                resolution_date=ov.get("resolution_date", cat_date_str),
                outcome=outcome,
                outcome_detail=ov.get("outcome_detail", "manual override"),
                source_type="MANUAL",
                source_id=ov.get("source", "manual_override"),
                prediction_snapshot_date=snap.get("snapshot_date"),
                prediction_dem_rank=dem_rank,
                prediction_composite_score=_safe_float(snap.get("composite_score")),
                prediction_match_type=snap.get("match_type"),
                event_ev_node_id=ev_bind["event_ev_node_id"],
                event_ev_p_hit=ev_bind["event_ev_p_hit"],
                event_ev_p_miss=ev_bind["event_ev_p_miss"],
                event_ev_p_mixed=ev_bind["event_ev_p_mixed"],
                event_ev_confidence=ev_bind["event_ev_confidence"],
                event_ev_asof_date=ev_bind["event_ev_asof_date"],
                event_ev_match_type=ev_bind["event_ev_match_type"],
                price_t_minus_1=prices_data.get("price_t_minus_1"),
                price_t_0=prices_data.get("price_t_0"),
                price_t_plus_5=prices_data.get("price_t_plus_5"),
                price_direction=_compute_price_direction(
                    prices_data.get("price_t_minus_1"), prices_data.get("price_t_0")
                ),
                as_of_date=as_of_date.isoformat(),
            )
        )
        _override_keys_used.add(key)
    if _override_keys_used:
        logger.info("Manual overrides: %d processed", len(_override_keys_used))

    for event in watchlist:
        ticker = event["ticker"]
        cat_date_str = event["catalyst_date"]
        cat_date = date.fromisoformat(cat_date_str[:10])
        cat_type = event.get("catalyst_type", "CORPORATE_UPDATE")

        override_key = (ticker, cat_date_str[:10])
        if override_key in overrides:
            ov = overrides[override_key]
            outcome = ov.get("outcome", "NEEDS_REVIEW")
            if outcome in OUTCOMES:
                snap = get_prediction_snapshot(ticker, cat_date, snapshots_dir)
                dem_rank = None
                if snap.get("dem_rank"):
                    try:
                        dem_rank = int(snap["dem_rank"])
                    except (ValueError, TypeError):
                        pass
                ev_bind = _bind_event_ev_p_hit(ticker, cat_date_str[:10], snap.get("snapshot_date"))
                new_records.append(
                    ResolutionRecord(
                        ticker=ticker,
                        catalyst_date=cat_date_str[:10],
                        catalyst_type=cat_type,
                        catalyst_description=event.get("description", ""),
                        resolution_date=ov.get("resolution_date", cat_date_str[:10]),
                        outcome=outcome,
                        outcome_detail=ov.get("outcome_detail", "manual override"),
                        source_type="MANUAL",
                        source_id="manual_override",
                        prediction_snapshot_date=snap.get("snapshot_date"),
                        prediction_dem_rank=dem_rank,
                        prediction_composite_score=_safe_float(snap.get("composite_score")),
                        prediction_match_type=snap.get("match_type"),
                        event_ev_node_id=ev_bind["event_ev_node_id"],
                        event_ev_p_hit=ev_bind["event_ev_p_hit"],
                        event_ev_p_miss=ev_bind["event_ev_p_miss"],
                        event_ev_p_mixed=ev_bind["event_ev_p_mixed"],
                        event_ev_confidence=ev_bind["event_ev_confidence"],
                        event_ev_asof_date=ev_bind["event_ev_asof_date"],
                        event_ev_match_type=ev_bind["event_ev_match_type"],
                        as_of_date=as_of_date.isoformat(),
                    )
                )
                continue

        _8k = check_8k_for_resolution(ticker, sec_8k_events, cat_date, as_of_date)
        _ctgov = check_ctgov_for_resolution(ticker, trial_records, cat_date)

        if not _8k and not _ctgov:
            if (as_of_date - cat_date).days > 7:
                outcome = "DELAYED"
            else:
                continue
        else:
            headline = _8k.get("headline", "") if _8k else ""
            ctgov_to = _ctgov.get("ctgov_status_to") if _ctgov else None
            outcome = classify_outcome(cat_type, headline=headline, ctgov_status_to=ctgov_to)

        source_type = "SEC_8K"
        source_id = ""
        resolution_date = cat_date_str[:10]
        if _8k:
            source_type = _8k["source_type"]
            source_id = _8k["source_id"]
            resolution_date = _8k.get("resolution_date", cat_date_str[:10])
        elif _ctgov:
            source_type = _ctgov["source_type"]
            source_id = _ctgov["source_id"]

        snap = get_prediction_snapshot(ticker, cat_date, snapshots_dir)
        dem_rank = None
        if snap.get("dem_rank"):
            try:
                dem_rank = int(snap["dem_rank"])
            except (ValueError, TypeError):
                pass

        ev_bind = _bind_event_ev_p_hit(ticker, cat_date_str[:10], snap.get("snapshot_date"))

        prices_data: Dict[str, Optional[float]] = {
            "price_t_minus_1": None,
            "price_t_0": None,
            "price_t_plus_5": None,
        }
        if price_series:
            prices_data = get_price_reaction(ticker, cat_date, price_series, as_of_date)

        new_records.append(
            ResolutionRecord(
                ticker=ticker,
                catalyst_date=cat_date_str[:10],
                catalyst_type=cat_type,
                catalyst_description=event.get("description", ""),
                resolution_date=resolution_date,
                outcome=outcome,
                outcome_detail=(_8k.get("headline", "") if _8k else "")[:200],
                source_type=source_type,
                source_id=source_id,
                prediction_snapshot_date=snap.get("snapshot_date"),
                prediction_dem_rank=dem_rank,
                prediction_composite_score=_safe_float(snap.get("composite_score")),
                prediction_match_type=snap.get("match_type"),
                event_ev_node_id=ev_bind["event_ev_node_id"],
                event_ev_p_hit=ev_bind["event_ev_p_hit"],
                event_ev_p_miss=ev_bind["event_ev_p_miss"],
                event_ev_p_mixed=ev_bind["event_ev_p_mixed"],
                event_ev_confidence=ev_bind["event_ev_confidence"],
                event_ev_asof_date=ev_bind["event_ev_asof_date"],
                event_ev_match_type=ev_bind["event_ev_match_type"],
                price_t_minus_1=prices_data.get("price_t_minus_1"),
                price_t_0=prices_data.get("price_t_0"),
                price_t_plus_5=prices_data.get("price_t_plus_5"),
                price_direction=_compute_price_direction(
                    prices_data.get("price_t_minus_1"), prices_data.get("price_t_0")
                ),
                days_from_expected=(as_of_date - cat_date).days if _8k or _ctgov else None,
                as_of_date=as_of_date.isoformat(),
            )
        )

    written = 0
    skipped_informational = 0
    for record in new_records:
        # Skip INFORMATIONAL records — these are pipeline updates, not resolutions
        if record.outcome == "INFORMATIONAL":
            skipped_informational += 1
            continue

        record_hash = compute_record_hash(record)
        month_dir = resolutions_dir / record.catalyst_date[:7]
        month_dir.mkdir(parents=True, exist_ok=True)
        out_path = month_dir / f"{record.ticker}_{record.catalyst_date}.json"

        if out_path.exists():
            logger.info("  SKIP (exists): %s", out_path.name)
            continue

        out_data = record.to_dict()
        out_data["record_hash"] = record_hash

        if dry_run:
            logger.info("  DRY-RUN: %s [%s]", out_path.name, record.outcome)
        else:
            with open(out_path, "w") as f:
                json.dump(out_data, f, indent=2, default=str)
                f.write("\n")
            logger.info("  WROTE: %s [%s]", out_path.name, record.outcome)
        written += 1

    wl_path = resolutions_dir / "watchlist_current.json"
    wl_data = {
        "as_of_date": as_of_date.isoformat(),
        "n_watchlist": len(watchlist),
        "n_resolved_today": written,
        "n_existing": len(existing),
        "watchlist": [
            {"ticker": w["ticker"], "catalyst_date": w["catalyst_date"], "catalyst_type": w.get("catalyst_type", "")}
            for w in watchlist
        ],
    }
    if not dry_run:
        with open(wl_path, "w") as f:
            json.dump(wl_data, f, indent=2)
            f.write("\n")

    return {
        "as_of_date": as_of_date.isoformat(),
        "n_watchlist": len(watchlist),
        "n_new_records": written,
        "n_informational_skipped": skipped_informational,
        "n_existing": len(existing),
        "records": [r.to_dict() for r in new_records],
    }


def main() -> int:
    import time

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    started = time.perf_counter()

    parser = argparse.ArgumentParser(description="Catalyst Resolution Tracker (Spec 042)")
    parser.add_argument("--as-of-date", required=True, help="Snapshot date (YYYY-MM-DD)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--resolutions-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "snapshots" / "resolutions",
    )
    args = parser.parse_args()
    as_of = date.fromisoformat(args.as_of_date)

    price_series: Dict[str, Dict[str, float]] = {}
    price_path = PROJECT_ROOT / "production_data" / "price_history.csv"
    if price_path.exists():
        with open(price_path) as f:
            for row in csv.DictReader(f):
                t = row.get("ticker") or row.get("symbol", "")
                d = row.get("date", "")
                c = row.get("close", "")
                if t and d and c:
                    try:
                        price_series.setdefault(t, {})[d] = float(c)
                    except ValueError:
                        pass

    result = run_crt(
        as_of_date=as_of,
        snapshots_dir=PROJECT_ROOT / "data" / "snapshots",
        resolutions_dir=args.resolutions_dir,
        sec_8k_cache_dir=PROJECT_ROOT / "cache" / "sec" / "8k_catalysts",
        ctgov_cache_dir=PROJECT_ROOT / "cache" / "ctgov",
        pdufa_path=PROJECT_ROOT / "production_data" / "pdufa_dates.json",
        price_series=price_series,
        dry_run=args.dry_run,
    )

    print(f"CRT: {result['n_watchlist']} in watchlist, {result['n_new_records']} new resolutions")
    for rec in result["records"]:
        print(f"  {rec['ticker']} {rec['catalyst_type']} -> {rec['outcome']} ({rec['source_type']})")

    try:
        from tools.agent_skill_telemetry import log_agent_run
        from tools.record_skill_feedback import attach_outcome_verdict

        n_wl = result.get("n_watchlist", 0)
        n_new = result.get("n_new_records", 0)
        exec_id = log_agent_run(
            "catalyst_resolution_tracker",
            f"CRT run for {args.as_of_date}",
            inputs={"as_of_date": args.as_of_date, "dry_run": args.dry_run},
            outputs={"n_watchlist": n_wl, "n_new_records": n_new},
            success=True,
            latency_ms=(time.perf_counter() - started) * 1000,
        )
        if exec_id:
            attach_outcome_verdict(
                exec_id,
                was_correct=n_wl > 0,
                evidence=f"watchlist={n_wl} new_records={n_new}",
            )
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
