#!/usr/bin/env python3
"""Build PIT-safe catalyst event features from the event ledger.

Wraps event_ledger.py build_event_ledger() + query_nearest_catalyst() into
a standalone feature file, producing per-ticker catalyst features that match
what save_validation_snapshot() computes in run_screen.py / decision_engine.py.

PIT Safety:
  Inherited from event_ledger.py: entries with disclosed_at > as_of_date
  are excluded (line 579).

Usage:
    python3 scripts/build_catalyst_events_pit.py \\
        --as-of-date 2026-01-15 \\
        --out /tmp/catalyst_events_2026-01-15.json \\
        --universe production_data/universe.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

VERSION = "1.0.0"
SCHEMA_VERSION = "catalyst_events_pit.v1"

DEFAULT_CATALYST_NEAR_DAYS = 120
DEFAULT_CATALYST_MID_DAYS = 180
DEFAULT_DECAY_MODE = "step"

# Project root (parent of scripts/)
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


# ---------------------------------------------------------------------------
# Catalyst feature derivation
# ---------------------------------------------------------------------------

def _derive_catalyst_mode(
    days_to_catalyst: Optional[int],
) -> str:
    """Derive catalyst_mode from days_to_catalyst.

    Replicates decision_engine.py:638-646 logic.
    When days_to_catalyst comes from query_nearest_catalyst, it is always
    a positive int (future event).  None means no upcoming catalyst.
    """
    if days_to_catalyst is not None and days_to_catalyst > 0:
        return "specific_days"
    elif days_to_catalyst is not None and days_to_catalyst == 0:
        # Day-of event: treat as blended_window
        return "blended_window"
    else:
        return "missing"


def _derive_catalyst_strength(
    mode: str,
    days: Optional[int],
    near_days: int,
    mid_days: int,
) -> str:
    """Derive catalyst_strength from mode and days.

    Replicates decision_engine.py:648-659 logic.
    """
    if mode == "blended_window":
        return "near"
    elif mode == "specific_days" and days is not None and days > 0:
        if days <= near_days:
            return "near"
        elif days <= mid_days:
            return "mid"
        else:
            return "far"
    else:
        return "missing"


def _derive_catalyst_decay_w(
    mode: str,
    strength: str,
    days: Optional[int],
    decay_mode: str,
    near_days: int = DEFAULT_CATALYST_NEAR_DAYS,
    mid_days: int = DEFAULT_CATALYST_MID_DAYS,
) -> float:
    """Derive catalyst decay weight.

    For "step" decay_mode, uses the hard-weight mapping from
    decision_engine.py:682-684.  For "linear"/"exp", delegates
    to compute_catalyst_decay_weight from run_screen.py.
    """
    if decay_mode == "step":
        _hard_w = {"near": 1.0, "mid": 0.7, "far": 0.3, "missing": 0.0}
        return _hard_w.get(strength, 0.0)

    # For linear/exp, import from run_screen
    if days is None or days < 0 or mode == "missing":
        return 0.0

    try:
        from run_screen import compute_catalyst_decay_weight
        return round(
            compute_catalyst_decay_weight(days, 0, near_days, decay_mode), 4,
        )
    except ImportError:
        # Fallback to step if import fails
        _hard_w = {"near": 1.0, "mid": 0.7, "far": 0.3, "missing": 0.0}
        return _hard_w.get(strength, 0.0)


def _derive_catalyst_features(
    nearest: Optional[Dict],
    near_days: int = DEFAULT_CATALYST_NEAR_DAYS,
    mid_days: int = DEFAULT_CATALYST_MID_DAYS,
    decay_mode: str = DEFAULT_DECAY_MODE,
) -> dict:
    """From query_nearest_catalyst result to per-ticker feature dict.

    Replicates run_screen.py / decision_engine.py catalyst logic.
    """
    if nearest is None:
        return {
            "days_to_catalyst": None,
            "catalyst_mode": "missing",
            "catalyst_strength": "missing",
            "catalyst_decay_w": 0.0,
            "nearest_event_type": "",
            "nearest_event_source": "",
            "nearest_event_name": "",
            "nearest_disclosed_at": "",
            "nearest_event_date": "",
        }

    days = nearest["days_to_catalyst"]
    mode = _derive_catalyst_mode(days)
    strength = _derive_catalyst_strength(mode, days, near_days, mid_days)
    decay_w = _derive_catalyst_decay_w(
        mode, strength, days, decay_mode, near_days, mid_days,
    )

    return {
        "days_to_catalyst": days,
        "catalyst_mode": mode,
        "catalyst_strength": strength,
        "catalyst_decay_w": decay_w,
        "nearest_event_type": nearest.get("event_type", ""),
        "nearest_event_source": nearest.get("source", ""),
        "nearest_event_name": nearest.get("event_name", ""),
        "nearest_disclosed_at": nearest.get("disclosed_at", ""),
        "nearest_event_date": nearest.get("event_date", ""),
    }


def _count_source_mix(
    ledger: list,
    ticker: str,
) -> Dict[str, int]:
    """Count events per source for a given ticker."""
    counts: Dict[str, int] = {}
    for entry in ledger:
        if entry.ticker == ticker:
            counts[entry.source] = counts.get(entry.source, 0) + 1
    return counts


def _count_ledger_sources(ledger: list) -> Dict[str, int]:
    """Count total entries per source in the full ledger."""
    counts: Dict[str, int] = {}
    for entry in ledger:
        counts[entry.source] = counts.get(entry.source, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_catalyst_events(
    as_of_date: str,
    universe_tickers: Set[str],
    ledger: Optional[list] = None,
    config: Optional[Any] = None,
    near_days: int = DEFAULT_CATALYST_NEAR_DAYS,
    mid_days: int = DEFAULT_CATALYST_MID_DAYS,
    decay_mode: str = DEFAULT_DECAY_MODE,
) -> dict:
    """Build PIT-safe catalyst event features for all universe tickers.

    Args:
        as_of_date: ISO date string (YYYY-MM-DD).
        universe_tickers: Set of ticker strings (upper-cased).
        ledger: Pre-built event ledger (list of LedgerEntry). If None,
            builds from config.
        config: LedgerConfig for building the event ledger (used if ledger
            is None).
        near_days: Days threshold for "near" catalyst.
        mid_days: Days threshold for "mid" catalyst.
        decay_mode: "step", "linear", or "exp".

    Returns:
        Full output dict matching catalyst_events_pit.v1 schema.
    """
    as_of = date.fromisoformat(as_of_date[:10])

    # Build or use provided ledger
    if ledger is None:
        try:
            from event_ledger import build_event_ledger, query_nearest_catalyst, LedgerConfig
            if config is None:
                config = LedgerConfig()
            ledger = build_event_ledger(as_of, config)
        except ImportError:
            ledger = []
    else:
        # Import query function
        try:
            from event_ledger import query_nearest_catalyst
        except ImportError:
            # Provide a stub for testing
            def query_nearest_catalyst(lg, tk, d):
                return None

    # If we didn't import query_nearest_catalyst above (ledger was None path)
    try:
        query_nearest_catalyst
    except NameError:
        try:
            from event_ledger import query_nearest_catalyst
        except ImportError:
            def query_nearest_catalyst(lg, tk, d):
                return None

    # Per-ticker features
    tickers_data: Dict[str, dict] = {}
    tickers_with_catalyst = 0

    for ticker in sorted(universe_tickers):
        nearest = query_nearest_catalyst(ledger, ticker, as_of)
        features = _derive_catalyst_features(nearest, near_days, mid_days, decay_mode)

        # Add event count and source mix
        source_mix = _count_source_mix(ledger, ticker)
        features["event_count"] = sum(source_mix.values())
        features["source_mix"] = source_mix

        tickers_data[ticker] = features

        if features["catalyst_mode"] != "missing":
            tickers_with_catalyst += 1

    tickers_in_universe = len(universe_tickers)
    ledger_sources = _count_ledger_sources(ledger)

    output = {
        "schema_version": SCHEMA_VERSION,
        "version": VERSION,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "as_of_date": as_of_date,
        "tickers_in_universe": tickers_in_universe,
        "tickers_with_catalyst": tickers_with_catalyst,
        "catalyst_coverage_pct": (
            round(tickers_with_catalyst / tickers_in_universe * 100, 1)
            if tickers_in_universe > 0 else 0.0
        ),
        "ledger_total_entries": len(ledger),
        "tickers": tickers_data,
        "provenance": {
            "builder": "build_catalyst_events_pit.py",
            "builder_version": VERSION,
            "ledger_sources": ledger_sources,
            "catalyst_near_days": near_days,
            "catalyst_mid_days": mid_days,
            "decay_mode": decay_mode,
        },
    }

    return output


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

def validate_catalyst_events_schema(
    features: dict,
) -> Tuple[bool, str]:
    """Validate output matches ``catalyst_events_pit.v1`` schema.

    Returns ``(True, "OK")`` on success, ``(False, reason)`` on failure.
    """
    if not isinstance(features, dict):
        return False, "features must be a dict"

    required_top = [
        "schema_version", "version", "created_at", "as_of_date",
        "tickers_in_universe", "tickers_with_catalyst", "catalyst_coverage_pct",
        "ledger_total_entries", "tickers", "provenance",
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
    n_cat = features["tickers_with_catalyst"]
    expected_pct = (
        round(n_cat / n_uni * 100, 1) if n_uni > 0 else 0.0
    )
    if abs(features["catalyst_coverage_pct"] - expected_pct) > 0.2:
        return False, (
            f"coverage inconsistent: {features['catalyst_coverage_pct']} "
            f"vs expected {expected_pct}"
        )

    # Per-ticker required fields
    required_ticker = [
        "days_to_catalyst", "catalyst_mode", "catalyst_strength",
        "catalyst_decay_w", "nearest_event_type", "nearest_event_source",
        "nearest_event_name", "nearest_disclosed_at", "nearest_event_date",
        "event_count", "source_mix",
    ]
    for tk, td in features.get("tickers", {}).items():
        for f in required_ticker:
            if f not in td:
                return False, f"ticker {tk} missing field: {f}"

    # Provenance required fields
    prov = features.get("provenance", {})
    prov_required = [
        "builder", "builder_version", "ledger_sources",
        "catalyst_near_days", "catalyst_mid_days", "decay_mode",
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
        description="Build PIT-safe catalyst event features from event ledger",
    )
    parser.add_argument(
        "--as-of-date", required=True, help="As-of date (ISO format)",
    )
    parser.add_argument("--out", required=True, help="Output JSON path")
    parser.add_argument(
        "--universe", default="production_data/universe.json",
        help="Universe JSON path",
    )
    parser.add_argument(
        "--sec-cache-dir", default=None,
        help="SEC cache directory (default: auto-detect)",
    )
    parser.add_argument(
        "--fda-cache-dir", default=None,
        help="FDA cache directory (default: auto-detect)",
    )
    parser.add_argument(
        "--ctgov-cache-dir", default=None,
        help="CTGov cache directory (default: auto-detect)",
    )
    parser.add_argument(
        "--catalyst-near-days", type=int, default=DEFAULT_CATALYST_NEAR_DAYS,
        help=f"Near catalyst threshold (default: {DEFAULT_CATALYST_NEAR_DAYS})",
    )
    parser.add_argument(
        "--catalyst-mid-days", type=int, default=DEFAULT_CATALYST_MID_DAYS,
        help=f"Mid catalyst threshold (default: {DEFAULT_CATALYST_MID_DAYS})",
    )
    parser.add_argument(
        "--decay-mode", default=DEFAULT_DECAY_MODE,
        choices=["step", "linear", "exp"],
        help=f"Decay mode (default: {DEFAULT_DECAY_MODE})",
    )
    args = parser.parse_args(argv)

    # Import project modules (add project root to path).
    sys.path.insert(0, str(_PROJECT_ROOT))

    # Resolve paths
    uni_path = Path(args.universe)
    if not uni_path.is_absolute():
        uni_path = _PROJECT_ROOT / uni_path

    # Load universe
    universe = _load_universe(uni_path)
    if not universe:
        print(f"ERROR: empty universe from {uni_path}", file=sys.stderr)
        return 1

    print(f"Loaded {len(universe)} universe tickers")

    # Build ledger config
    try:
        from event_ledger import LedgerConfig
        config = LedgerConfig(
            ctgov_cache_dir=(
                Path(args.ctgov_cache_dir) if args.ctgov_cache_dir else None
            ),
            sec_cache_dir=(
                Path(args.sec_cache_dir) if args.sec_cache_dir else None
            ),
            fda_cache_dir=(
                Path(args.fda_cache_dir) if args.fda_cache_dir else None
            ),
        )
    except ImportError:
        print("ERROR: event_ledger.py not found on path", file=sys.stderr)
        return 1

    # Build features
    features = build_catalyst_events(
        as_of_date=args.as_of_date,
        universe_tickers=universe,
        config=config,
        near_days=args.catalyst_near_days,
        mid_days=args.catalyst_mid_days,
        decay_mode=args.decay_mode,
    )

    # Validate
    ok, reason = validate_catalyst_events_schema(features)
    if not ok:
        print(f"WARNING: schema validation failed: {reason}", file=sys.stderr)

    # Write output
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(features, f, indent=2)

    print(f"Output: {out_path}")
    print(f"  tickers_with_catalyst: {features['tickers_with_catalyst']}"
          f" / {features['tickers_in_universe']}"
          f" ({features['catalyst_coverage_pct']}%)")
    print(f"  ledger_total_entries: {features['ledger_total_entries']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
