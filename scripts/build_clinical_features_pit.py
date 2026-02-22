#!/usr/bin/env python3
"""Build PIT-safe clinical features from trial_records.json.

Reads trial_records.json and produces per-ticker clinical features:
- readout_days (min days to nearest PCD)
- clinical_alpha_raw (weighted phase + proximity + quality)
- far_horizon_days (PIT-safe far-window catalyst hydration)

PIT Convention (canonical):
  A trial is admitted iff pit_date is not None AND pit_date <= as_of_date,
  where pit_date = first_posted OR last_update_posted (in priority order).

Usage:
    python3 scripts/build_clinical_features_pit.py \\
        --as-of-date 2026-01-15 \\
        --trial-records production_data/trial_records.json \\
        --out /tmp/clinical_features_2026-01-15.json \\
        --universe production_data/universe.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

VERSION = "1.0.0"
SCHEMA_VERSION = "clinical_features_pit.v1"

# ---------------------------------------------------------------------------
# Constants — replicated from run_screen.py:2141-2155
# ---------------------------------------------------------------------------

_CLINICAL_PHASES = frozenset({
    "PHASE1", "PHASE2", "PHASE3",
    "PHASE 1", "PHASE 2", "PHASE 3",
    "PHASE1/PHASE2", "PHASE 1/PHASE 2",
    "PHASE2/PHASE3", "PHASE 2/PHASE 3",
    "EARLY_PHASE1",
})

_PHASE_NUM = {
    "approved": 1.0, "phase 3": 0.83, "phase 2/3": 0.67,
    "phase 2": 0.50, "phase 1/2": 0.33, "phase 1": 0.17,
    "preclinical": 0.08,
}

_CE_WEIGHTS = {"phase": 0.35, "proximity": 0.35, "quality": 0.30}

_TERMINAL_NEGATIVE_STATUSES = frozenset({
    "WITHDRAWN", "TERMINATED", "SUSPENDED",
})

PIT_DATE_PRIORITY = ["first_posted", "last_update_posted"]

DEFAULT_QUALITY = 0.5
DEFAULT_FAR_WINDOW_DAYS = 730


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


def _parse_trial_date(s) -> Optional[date]:
    """Parse YYYY-MM-DD or YYYY-MM string to date, or None.

    Replicates run_screen.py:2158-2165.
    Month-only dates (YYYY-MM) are parsed as first-of-month.
    """
    if not s:
        return None
    raw = str(s).strip()[:10]
    # Handle month-only: YYYY-MM → YYYY-MM-01
    if len(raw) == 7 and raw[4] == "-":
        raw = raw + "-01"
    try:
        return datetime.strptime(raw, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def _clinical_proximity(days: Optional[int]) -> float:
    """Exponential decay proximity score: days to readout → [0, 1].

    Replicates run_screen.py:2168-2174.
    """
    if days is None:
        return 0.0
    if days <= 0:
        return 1.0
    return math.exp(-days / 90)


def _is_pit_admitted(trial: dict, as_of: date) -> Tuple[bool, str]:
    """Canonical PIT gate: trial admitted iff pit_date <= as_of_date.

    Returns (admitted, pit_date_field_used).
    pit_date_field_used is the field name that was used ("first_posted",
    "last_update_posted"), or "" if not admitted.
    """
    for field_name in PIT_DATE_PRIORITY:
        d = _parse_trial_date(trial.get(field_name))
        if d is not None:
            if d <= as_of:
                return True, field_name
            else:
                return False, field_name
    return False, ""


def _phase_label_to_num(phase_str: str) -> Tuple[float, str]:
    """Convert trial phase string to numeric score and canonical label.

    Returns (phase_num, label).  Falls back to checking _PHASE_NUM keys.
    """
    raw = str(phase_str).strip().upper()
    # Map ClinicalTrials.gov phases to canonical labels
    mapping = {
        "PHASE1": "phase 1",
        "PHASE 1": "phase 1",
        "EARLY_PHASE1": "phase 1",
        "PHASE2": "phase 2",
        "PHASE 2": "phase 2",
        "PHASE1/PHASE2": "phase 1/2",
        "PHASE 1/PHASE 2": "phase 1/2",
        "PHASE3": "phase 3",
        "PHASE 3": "phase 3",
        "PHASE2/PHASE3": "phase 2/3",
        "PHASE 2/PHASE 3": "phase 2/3",
    }
    label = mapping.get(raw, raw.lower())
    num = _PHASE_NUM.get(label, 0.0)
    return num, label


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def _compute_readout_for_ticker(
    trials: List[dict],
    as_of: date,
) -> Tuple[Optional[int], Optional[str], int, int, str]:
    """Compute readout_days for a single ticker's trials.

    Replicates run_screen.py:2197-2238 logic.

    Returns:
        (best_days, best_nct_id, n_admitted, n_filtered, pit_date_used)
    """
    best_days: Optional[int] = None
    best_nct_id: Optional[str] = None
    n_admitted = 0
    n_filtered = 0
    pit_field_used = ""

    for t in trials:
        # PIT gate
        admitted, pit_field = _is_pit_admitted(t, as_of)
        if not admitted:
            n_filtered += 1
            continue

        if pit_field and not pit_field_used:
            pit_field_used = pit_field

        # Study type filter
        if (t.get("study_type") or "").upper() != "INTERVENTIONAL":
            n_admitted += 1
            continue

        # Phase filter
        phase = str(t.get("phase", "")).upper()
        if phase not in _CLINICAL_PHASES:
            n_admitted += 1
            continue

        n_admitted += 1

        pcd = _parse_trial_date(t.get("primary_completion_date"))
        cd = _parse_trial_date(t.get("completion_date"))
        rfp = _parse_trial_date(t.get("results_first_posted"))

        # Priority 1: PCD
        if pcd is not None:
            if pcd > as_of:
                days = (pcd - as_of).days
            elif rfp is None:
                # Past PCD, no results posted → imminent readout
                days = 0
            else:
                # Past PCD + results posted → already known, skip
                continue
        elif cd is not None and cd > as_of:
            # Fallback: future completion_date
            days = (cd - as_of).days
        else:
            continue

        if best_days is None or days < best_days:
            best_days = days
            best_nct_id = t.get("nct_id", "")

    return best_days, best_nct_id, n_admitted, n_filtered, pit_field_used


def _compute_far_horizon_days(
    trials: List[dict],
    as_of: date,
    far_max: int = DEFAULT_FAR_WINDOW_DAYS,
) -> Optional[int]:
    """PIT-safe far-horizon catalyst: min future PCD within far_max days.

    Fixes the gap in _hydrate_far_horizon_catalysts (run_screen.py:2313-2379)
    which has NO PIT check.  This version applies the canonical PIT gate.
    """
    best: Optional[int] = None

    for t in trials:
        # PIT gate
        admitted, _ = _is_pit_admitted(t, as_of)
        if not admitted:
            continue

        # Study type filter
        stype = (t.get("study_type") or "").upper()
        if "INTERVENTIONAL" not in stype:
            continue

        # Terminal status filter
        status = (t.get("status") or "?").upper()
        if status in _TERMINAL_NEGATIVE_STATUSES:
            continue

        pcd = _parse_trial_date(t.get("primary_completion_date"))
        if pcd is None:
            continue

        days = (pcd - as_of).days
        if days <= 0:
            continue
        if days > far_max:
            continue

        if best is None or days < best:
            best = days

    return best


def _highest_phase(trials: List[dict], as_of: date) -> Tuple[float, str]:
    """Find the highest phase score across PIT-admitted trials."""
    best_num = 0.0
    best_label = ""
    for t in trials:
        admitted, _ = _is_pit_admitted(t, as_of)
        if not admitted:
            continue
        if (t.get("study_type") or "").upper() != "INTERVENTIONAL":
            continue
        phase = str(t.get("phase", ""))
        num, label = _phase_label_to_num(phase)
        if num > best_num:
            best_num = num
            best_label = label
    return best_num, best_label


def _has_active_interventional(trials: List[dict], as_of: date) -> bool:
    """Check if any PIT-admitted interventional trial is non-terminal."""
    for t in trials:
        admitted, _ = _is_pit_admitted(t, as_of)
        if not admitted:
            continue
        if (t.get("study_type") or "").upper() != "INTERVENTIONAL":
            continue
        status = (t.get("status") or "?").upper()
        if status not in _TERMINAL_NEGATIVE_STATUSES:
            return True
    return False


def _collected_at_range(trial_records: List[dict]) -> Tuple[str, str]:
    """Find min/max collected_at dates from trial records."""
    dates: List[str] = []
    for t in trial_records:
        ca = t.get("collected_at") or t.get("last_update_posted") or ""
        if ca:
            dates.append(str(ca)[:10])
    if not dates:
        return "", ""
    return min(dates), max(dates)


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_clinical_features(
    as_of_date: str,
    trial_records: List[dict],
    universe_tickers: Set[str],
    quality_mode: str = "neutral",
    quality_value: float = DEFAULT_QUALITY,
    far_window_days: int = DEFAULT_FAR_WINDOW_DAYS,
) -> dict:
    """Build PIT-safe clinical features for all universe tickers.

    Args:
        as_of_date: ISO date string (YYYY-MM-DD).
        trial_records: List of trial record dicts.
        universe_tickers: Set of ticker strings (upper-cased).
        quality_mode: "neutral" (constant) or "from_module4" (not yet supported).
        quality_value: Quality score when quality_mode is "neutral".
        far_window_days: Max days for far-horizon catalyst scan.

    Returns:
        Full output dict matching clinical_features_pit.v1 schema.
    """
    as_of = datetime.strptime(as_of_date[:10], "%Y-%m-%d").date()

    # Group trials by ticker
    by_ticker: Dict[str, List[dict]] = {}
    for trial in trial_records:
        tk = (trial.get("ticker") or "").upper()
        if tk:
            by_ticker.setdefault(tk, []).append(trial)

    # Track totals
    total_admitted = 0
    total_filtered = 0
    tickers_data: Dict[str, dict] = {}

    for ticker in sorted(universe_tickers):
        trials = by_ticker.get(ticker, [])
        if not trials:
            continue

        # Readout days
        rd, nct_id, n_adm, n_filt, pit_field = _compute_readout_for_ticker(
            trials, as_of,
        )
        total_admitted += n_adm
        total_filtered += n_filt

        # Phase
        phase_num, lead_phase = _highest_phase(trials, as_of)

        # Proximity
        proximity = _clinical_proximity(rd)

        # Quality (neutral default)
        quality = quality_value

        # Clinical alpha raw
        raw = (
            _CE_WEIGHTS["phase"] * phase_num
            + _CE_WEIGHTS["proximity"] * proximity
            + _CE_WEIGHTS["quality"] * quality
        )

        # Far horizon (PIT-safe)
        far_days = _compute_far_horizon_days(trials, as_of, far_window_days)

        # Active interventional check
        has_active = _has_active_interventional(trials, as_of)

        tickers_data[ticker] = {
            "readout_days": rd,
            "readout_nct_id": nct_id or "",
            "phase_num": round(phase_num, 4),
            "lead_phase": lead_phase,
            "proximity": round(proximity, 6),
            "quality": round(quality, 4),
            "clinical_alpha_raw": round(raw, 6),
            "n_trials_admitted": n_adm,
            "n_trials_filtered": n_filt,
            "pit_date_used": pit_field,
            "far_horizon_days": far_days,
            "has_active_interventional": has_active,
        }

    tickers_with_signal = len(tickers_data)
    tickers_in_universe = len(universe_tickers)

    # Trial-level totals (count all trials, not just universe)
    all_trials_total = len(trial_records)

    col_min, col_max = _collected_at_range(trial_records)

    output = {
        "schema_version": SCHEMA_VERSION,
        "version": VERSION,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "as_of_date": as_of_date,
        "trial_records_source": "",  # filled by caller
        "pit_convention": "first_posted_or_last_update_lte_as_of",
        "tickers_in_universe": tickers_in_universe,
        "tickers_with_signal": tickers_with_signal,
        "signal_coverage_pct": (
            round(tickers_with_signal / tickers_in_universe * 100, 1)
            if tickers_in_universe > 0 else 0.0
        ),
        "trials_total": all_trials_total,
        "trials_admitted": total_admitted,
        "trials_filtered": total_filtered,
        "tickers": tickers_data,
        "provenance": {
            "builder": "build_clinical_features_pit.py",
            "builder_version": VERSION,
            "quality_mode": quality_mode,
            "quality_value": quality_value,
            "pit_date_priority": list(PIT_DATE_PRIORITY),
            "trial_records_file": "",
            "trial_records_collected_at_range": [col_min, col_max],
            "far_window_days": far_window_days,
        },
    }

    return output


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

def validate_clinical_features_schema(
    features: dict,
) -> Tuple[bool, str]:
    """Validate output matches ``clinical_features_pit.v1`` schema.

    Returns ``(True, "OK")`` on success, ``(False, reason)`` on failure.
    """
    if not isinstance(features, dict):
        return False, "features must be a dict"

    required_top = [
        "schema_version", "version", "created_at", "as_of_date",
        "tickers_in_universe", "tickers_with_signal", "signal_coverage_pct",
        "trials_total", "trials_admitted", "trials_filtered",
        "tickers", "provenance",
    ]
    for k in required_top:
        if k not in features:
            return False, f"missing required field: {k}"

    if features["schema_version"] != SCHEMA_VERSION:
        return False, (
            f"expected schema {SCHEMA_VERSION}, "
            f"got {features['schema_version']}"
        )

    # Coverage consistency
    n_uni = features["tickers_in_universe"]
    n_sig = features["tickers_with_signal"]
    expected_pct = (
        round(n_sig / n_uni * 100, 1) if n_uni > 0 else 0.0
    )
    if abs(features["signal_coverage_pct"] - expected_pct) > 0.2:
        return False, (
            f"coverage inconsistent: {features['signal_coverage_pct']} "
            f"vs expected {expected_pct}"
        )

    # Per-ticker required fields
    required_ticker = [
        "readout_days", "readout_nct_id", "phase_num", "lead_phase",
        "proximity", "quality", "clinical_alpha_raw",
        "n_trials_admitted", "n_trials_filtered", "pit_date_used",
        "far_horizon_days", "has_active_interventional",
    ]
    for tk, td in features.get("tickers", {}).items():
        for f in required_ticker:
            if f not in td:
                return False, f"ticker {tk} missing field: {f}"

    # Provenance required fields
    prov = features.get("provenance", {})
    prov_required = [
        "builder", "builder_version", "quality_mode", "quality_value",
        "pit_date_priority",
    ]
    for k in prov_required:
        if k not in prov:
            return False, f"provenance missing field: {k}"

    return True, "OK"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build PIT-safe clinical features from trial_records.json",
    )
    parser.add_argument(
        "--as-of-date", required=True, help="As-of date (ISO format)",
    )
    parser.add_argument(
        "--trial-records", default="production_data/trial_records.json",
        help="Path to trial_records.json",
    )
    parser.add_argument("--out", required=True, help="Output JSON path")
    parser.add_argument(
        "--universe", default="production_data/universe.json",
        help="Universe JSON path",
    )
    parser.add_argument(
        "--quality-mode", default="neutral",
        choices=["neutral", "from_module4"],
        help="Quality score mode (default: neutral=0.5)",
    )
    parser.add_argument(
        "--far-window-days", type=int, default=DEFAULT_FAR_WINDOW_DAYS,
        help=f"Far horizon window (default: {DEFAULT_FAR_WINDOW_DAYS})",
    )
    args = parser.parse_args(argv)

    # Import project modules (add project root to path).
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    # Resolve paths relative to project root
    trial_path = Path(args.trial_records)
    if not trial_path.is_absolute():
        trial_path = project_root / trial_path
    uni_path = Path(args.universe)
    if not uni_path.is_absolute():
        uni_path = project_root / uni_path

    # Load inputs
    universe = _load_universe(uni_path)
    if not universe:
        print(f"ERROR: empty universe from {uni_path}", file=sys.stderr)
        return 1

    trial_data = _load_json(trial_path)
    if trial_data is None:
        print(f"ERROR: cannot load trial records from {trial_path}",
              file=sys.stderr)
        return 1
    if not isinstance(trial_data, list):
        print(f"ERROR: trial_records must be a list, got {type(trial_data)}",
              file=sys.stderr)
        return 1

    print(f"Loaded {len(trial_data)} trial records, "
          f"{len(universe)} universe tickers")

    # Build features
    features = build_clinical_features(
        as_of_date=args.as_of_date,
        trial_records=trial_data,
        universe_tickers=universe,
        quality_mode=args.quality_mode,
        far_window_days=args.far_window_days,
    )
    features["trial_records_source"] = str(trial_path)
    features["provenance"]["trial_records_file"] = trial_path.name

    # Validate
    ok, reason = validate_clinical_features_schema(features)
    if not ok:
        print(f"WARNING: schema validation failed: {reason}", file=sys.stderr)

    # Write output
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(features, f, indent=2)

    print(f"Output: {out_path}")
    print(f"  tickers_with_signal: {features['tickers_with_signal']}"
          f" / {features['tickers_in_universe']}"
          f" ({features['signal_coverage_pct']}%)")
    print(f"  trials: {features['trials_admitted']} admitted, "
          f"{features['trials_filtered']} filtered")
    return 0


if __name__ == "__main__":
    sys.exit(main())
