#!/usr/bin/env python3
"""
audit_catalyst_coverage.py — Comprehensive catalyst coverage diagnostic

Reads offline cached data and reports:
  - Universe size, % with any catalyst, % with dated catalysts
  - % with FDA events, % with CTGOV readouts
  - Breakdown by source: CTGOV_CALENDAR, FDA_CALENDAR, SEC_8K_FILING, etc.
  - Breakdown by event_type: DATA_READOUT, FDA_*, CT_*, TRIAL_ONGOING, etc.
  - Unknown/blank counts for event_type and source

Usage:
  python3 scripts/audit_catalyst_coverage.py
  python3 scripts/audit_catalyst_coverage.py --as-of-date 2026-02-07
  python3 scripts/audit_catalyst_coverage.py --vnext-file path/to/summaries.json

Outputs:
  - artifacts/coverage_audit_{date}.json  (machine-readable)
  - Console summary
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# Default data locations
UNIVERSE_PATH = PROJECT_ROOT / "production_data" / "universe.json"
PDUFA_PATH = PROJECT_ROOT / "production_data" / "pdufa_dates.json"
CORPORATE_CATALYSTS_PATH = PROJECT_ROOT / "production_data" / "corporate_catalysts.json"

# Cache directories to search (both project-root and pipeline subdirectory)
CACHE_DIRS_8K = [
    PROJECT_ROOT / "cache" / "sec" / "8k_catalysts",
    PROJECT_ROOT / "wake_robin_data_pipeline" / "cache" / "sec" / "8k_catalysts",
]
CACHE_DIRS_ADCOM = [
    PROJECT_ROOT / "cache" / "fda",
    PROJECT_ROOT / "wake_robin_data_pipeline" / "cache" / "fda",
]

# Well-known event types (for classifying unknown/blank)
KNOWN_EVENT_TYPES = {
    "DATA_READOUT", "FDA_PDUFA_DATE", "FDA_ADCOM", "FDA_CRL", "FDA_RTF",
    "FDA_WARNING_LETTER", "FDA_APPROVAL", "FDA_SUBMISSION", "FDA_DESIGNATION",
    "FDA_DECISION", "CT_PRIMARY_COMPLETION", "CT_STUDY_COMPLETION",
    "TRIAL_ONGOING", "CLINICAL_HOLD", "SAFETY_SIGNAL",
}

FDA_EVENT_TYPES = {
    "FDA_PDUFA_DATE", "FDA_ADCOM", "FDA_CRL", "FDA_RTF",
    "FDA_WARNING_LETTER", "FDA_APPROVAL", "FDA_SUBMISSION",
    "FDA_DESIGNATION", "FDA_DECISION",
}

# Well-known sources
KNOWN_SOURCES = {
    "CTGOV_CALENDAR", "FDA_CALENDAR", "FDA_ADCOM_CALENDAR",
    "SEC_8K_FILING", "SEC_10Q_FILING", "SEC_10K_FILING", "SEC_6K_FILING",
    "SEC_8K_ADCOM", "CORPORATE_CALENDAR", "FEDERAL_REGISTER",
}


def _load_json(path):
    """Load JSON file, return empty list on failure."""
    if not path or not Path(path).exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _find_latest_cache(dirs, prefix):
    """Find the most recent cache file matching prefix across directories."""
    candidates = []
    for d in dirs:
        if d.exists():
            for f in d.glob(f"{prefix}*.json"):
                candidates.append(f)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _find_cache_for_date(dirs, prefix, as_of_date):
    """Find cache file for a specific date (any version)."""
    target = f"{prefix}{as_of_date}"
    for d in dirs:
        if not d.exists():
            continue
        matches = sorted(d.glob(f"{target}*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if matches:
            return matches[0]
    return None


def _load_universe():
    """Load universe tickers as a set of uppercase strings."""
    data = _load_json(UNIVERSE_PATH)
    if isinstance(data, dict):
        entries = data.get("tickers", [])
    elif isinstance(data, list):
        entries = data
    else:
        return set()
    tickers = set()
    for e in entries:
        if isinstance(e, dict):
            t = e.get("ticker", "")
        else:
            t = str(e)
        if t:
            tickers.add(t.upper())
    return tickers


def _find_vnext_file(as_of_date=None):
    """Find catalyst_events_vnext_{date}.json in production_data/."""
    prod_dir = PROJECT_ROOT / "production_data"
    if as_of_date:
        candidate = prod_dir / f"catalyst_events_vnext_{as_of_date}.json"
        if candidate.exists():
            return candidate
    # Fall back to latest by mtime
    candidates = list(prod_dir.glob("catalyst_events_vnext_*.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def compute_coverage_metrics(
    universe_tickers,
    vnext_summaries=None,
    sec_8k_events=None,
    adcom_events=None,
    pdufa_entries=None,
    corp_entries=None,
):
    """Compute comprehensive coverage metrics.

    Args:
        universe_tickers: set of uppercase tickers
        vnext_summaries: dict of {ticker: summary} from catalyst_events_vnext
        sec_8k_events: list of SEC 8-K event dicts
        adcom_events: list of ADCOM event dicts
        pdufa_entries: list of PDUFA date entries
        corp_entries: list of corporate catalyst entries

    Returns:
        dict of coverage metrics
    """
    universe_size = len(universe_tickers)
    if universe_size == 0:
        return {"universe_size": 0, "error": "empty_universe"}

    vnext_summaries = vnext_summaries or {}
    sec_8k_events = sec_8k_events or []
    adcom_events = adcom_events or []
    pdufa_entries = pdufa_entries or []
    corp_entries = corp_entries or []

    # ── Aggregate events from all sources ──
    all_events = []  # list of (ticker, event_type, source, has_date)

    # vNext summaries: each ticker has events list
    for ticker, summary in vnext_summaries.items():
        if ticker.upper() not in universe_tickers:
            continue
        events = []
        if isinstance(summary, dict):
            events = summary.get("events", [])
        elif isinstance(summary, list):
            events = summary
        for ev in events:
            et = ev.get("event_type", "")
            src = ev.get("source", "")
            has_date = bool(ev.get("event_date"))
            all_events.append((ticker.upper(), et, src, has_date))

    # SEC 8-K events
    sec_tickers = set()
    for ev in sec_8k_events:
        t = ev.get("ticker", "").upper()
        if t and t in universe_tickers:
            et = ev.get("event_type", "")
            src = ev.get("source", "SEC_8K_FILING")
            has_date = bool(ev.get("event_date"))
            all_events.append((t, et, src, has_date))
            sec_tickers.add(t)

    # ADCOM events
    adcom_tickers = set()
    for ev in adcom_events:
        t = ev.get("ticker", "").upper()
        if t and t in universe_tickers:
            et = ev.get("event_type", "FDA_ADCOM")
            src = ev.get("source", "FEDERAL_REGISTER")
            has_date = bool(ev.get("event_date"))
            all_events.append((t, et, src, has_date))
            adcom_tickers.add(t)

    # PDUFA entries
    pdufa_tickers = set()
    for entry in pdufa_entries:
        if isinstance(entry, dict):
            t = entry.get("ticker", entry.get("symbol", "")).upper()
            if t and t in universe_tickers:
                all_events.append((t, "FDA_PDUFA_DATE", "FDA_CALENDAR", True))
                pdufa_tickers.add(t)

    # Corporate catalysts
    corp_tickers = set()
    for entry in corp_entries:
        if isinstance(entry, dict):
            t = entry.get("ticker", "").upper()
            if t and t in universe_tickers:
                et = entry.get("event_type", "")
                has_date = bool(entry.get("event_date") or entry.get("catalyst_date"))
                all_events.append((t, et, "CORPORATE_CALENDAR", has_date))
                corp_tickers.add(t)

    # ── Compute metrics ──
    tickers_with_any = {t for t, _, _, _ in all_events}
    tickers_with_dated = {t for t, _, _, hd in all_events if hd}

    # By source
    source_counter = Counter()
    source_tickers = defaultdict(set)
    for t, et, src, hd in all_events:
        s = src if src else "(blank)"
        source_counter[s] += 1
        source_tickers[s].add(t)

    # By event_type
    type_counter = Counter()
    type_tickers = defaultdict(set)
    for t, et, src, hd in all_events:
        e = et if et else "(blank)"
        type_counter[e] += 1
        type_tickers[e].add(t)

    # FDA tickers
    fda_tickers = {t for t, et, _, _ in all_events if et in FDA_EVENT_TYPES}

    # CTGOV tickers
    ctgov_tickers = set()
    for t, et, src, _ in all_events:
        if src == "CTGOV_CALENDAR" or et in ("CT_PRIMARY_COMPLETION", "CT_STUDY_COMPLETION"):
            ctgov_tickers.add(t)

    # Unknown/blank event_type
    unknown_type_events = [
        (t, et, src) for t, et, src, _ in all_events
        if not et or et not in KNOWN_EVENT_TYPES
    ]
    blank_type_events = [(t, et, src) for t, et, src, _ in all_events if not et]

    # Unknown/blank source
    unknown_source_events = [
        (t, et, src) for t, et, src, _ in all_events
        if not src or src not in KNOWN_SOURCES
    ]
    blank_source_events = [(t, et, src) for t, et, src, _ in all_events if not src]

    metrics = {
        "universe_size": universe_size,
        "total_events": len(all_events),
        "tickers_with_any_catalyst": len(tickers_with_any),
        "pct_with_any_catalyst": round(len(tickers_with_any) / universe_size * 100, 2),
        "tickers_with_dated_catalyst": len(tickers_with_dated),
        "pct_with_dated_catalyst": round(len(tickers_with_dated) / universe_size * 100, 2),
        "tickers_with_fda": len(fda_tickers),
        "pct_with_fda": round(len(fda_tickers) / universe_size * 100, 2),
        "tickers_with_ctgov": len(ctgov_tickers),
        "pct_with_ctgov": round(len(ctgov_tickers) / universe_size * 100, 2),
        "by_source": {
            s: {"events": c, "tickers": len(source_tickers[s])}
            for s, c in source_counter.most_common()
        },
        "by_event_type": {
            e: {"events": c, "tickers": len(type_tickers[e])}
            for e, c in type_counter.most_common()
        },
        "unknown_events": {
            "unknown_type_count": len(unknown_type_events),
            "blank_type_count": len(blank_type_events),
            "unknown_source_count": len(unknown_source_events),
            "blank_source_count": len(blank_source_events),
            "unknown_type_examples": [
                {"ticker": t, "event_type": et, "source": src}
                for t, et, src in unknown_type_events[:20]
            ],
            "unknown_source_examples": [
                {"ticker": t, "event_type": et, "source": src}
                for t, et, src in unknown_source_events[:20]
            ],
        },
        "source_ticker_lists": {
            "SEC_8K": sorted(sec_tickers & universe_tickers),
            "ADCOM": sorted(adcom_tickers & universe_tickers),
            "PDUFA": sorted(pdufa_tickers & universe_tickers),
            "CORPORATE": sorted(corp_tickers & universe_tickers),
        },
    }
    return metrics


def print_summary(metrics):
    """Print human-readable summary to console."""
    print(f"\n{'='*60}")
    print("CATALYST COVERAGE AUDIT")
    print(f"{'='*60}")
    print(f"  Universe size: {metrics['universe_size']}")
    print(f"  With any catalyst:   {metrics['tickers_with_any_catalyst']} "
          f"({metrics['pct_with_any_catalyst']:.1f}%)")
    print(f"  With dated catalyst: {metrics['tickers_with_dated_catalyst']} "
          f"({metrics['pct_with_dated_catalyst']:.1f}%)")
    print(f"  With FDA events:     {metrics['tickers_with_fda']} "
          f"({metrics['pct_with_fda']:.1f}%)")
    print(f"  With CTGOV readouts: {metrics['tickers_with_ctgov']} "
          f"({metrics['pct_with_ctgov']:.1f}%)")

    print(f"\n  BY SOURCE:")
    for src, data in metrics.get("by_source", {}).items():
        print(f"    {src:25s}  events={data['events']:4d}  tickers={data['tickers']:3d}")

    print(f"\n  BY EVENT TYPE:")
    for et, data in metrics.get("by_event_type", {}).items():
        print(f"    {et:30s}  events={data['events']:4d}  tickers={data['tickers']:3d}")

    unk = metrics.get("unknown_events", {})
    print(f"\n  UNKNOWN/BLANK:")
    print(f"    Unknown event_type:  {unk.get('unknown_type_count', 0)}")
    print(f"    Blank event_type:    {unk.get('blank_type_count', 0)}")
    print(f"    Unknown source:      {unk.get('unknown_source_count', 0)}")
    print(f"    Blank source:        {unk.get('blank_source_count', 0)}")

    if unk.get("unknown_type_examples"):
        print(f"\n    Unknown type examples:")
        for ex in unk["unknown_type_examples"][:10]:
            print(f"      {ex['ticker']:6s}  type={ex['event_type']!r:20s}  src={ex['source']!r}")

    if unk.get("unknown_source_examples"):
        print(f"\n    Unknown source examples:")
        for ex in unk["unknown_source_examples"][:10]:
            print(f"      {ex['ticker']:6s}  type={ex['event_type']!r:20s}  src={ex['source']!r}")


def main():
    parser = argparse.ArgumentParser(description="Comprehensive catalyst coverage audit")
    parser.add_argument(
        "--as-of-date", type=str, default=None,
        help="Specific as-of date (YYYY-MM-DD). Default: latest available.",
    )
    parser.add_argument(
        "--vnext-file", type=str, default=None,
        help="Path to catalyst_events_vnext_{date}.json. Default: auto-detect.",
    )
    args = parser.parse_args()

    # Load universe
    universe_tickers = _load_universe()
    print(f"Universe: {len(universe_tickers)} tickers")

    # Load vnext summaries
    vnext_summaries = {}
    if args.vnext_file:
        vnext_path = Path(args.vnext_file)
    else:
        vnext_path = _find_vnext_file(args.as_of_date)
    if vnext_path and vnext_path.exists():
        vnext_data = _load_json(vnext_path)
        if isinstance(vnext_data, dict):
            vnext_summaries = vnext_data
        print(f"vNext summaries: {vnext_path.name} ({len(vnext_summaries)} tickers)")
    else:
        print("vNext summaries: NOT FOUND")

    # Load SEC 8-K cache
    if args.as_of_date:
        sec_cache = _find_cache_for_date(CACHE_DIRS_8K, "8k_catalysts_", args.as_of_date)
    else:
        sec_cache = _find_latest_cache(CACHE_DIRS_8K, "8k_catalysts_")
    sec_events = _load_json(sec_cache) if sec_cache else []
    print(f"SEC 8-K cache: {sec_cache.name if sec_cache else 'NOT FOUND'} ({len(sec_events)} events)")

    # Load ADCOM cache
    if args.as_of_date:
        adcom_cache = _find_cache_for_date(CACHE_DIRS_ADCOM, "adcom_calendar_", args.as_of_date)
    else:
        adcom_cache = _find_latest_cache(CACHE_DIRS_ADCOM, "adcom_calendar_")
    adcom_events = _load_json(adcom_cache) if adcom_cache else []
    print(f"ADCOM cache: {adcom_cache.name if adcom_cache else 'NOT FOUND'} ({len(adcom_events)} events)")

    # Load PDUFA dates
    pdufa_data = _load_json(PDUFA_PATH)
    if isinstance(pdufa_data, dict):
        pdufa_entries = pdufa_data.get("events", pdufa_data.get("dates", []))
        if not isinstance(pdufa_entries, list):
            pdufa_entries = []
    elif isinstance(pdufa_data, list):
        pdufa_entries = pdufa_data
    else:
        pdufa_entries = []
    print(f"PDUFA dates: {len(pdufa_entries)} entries")

    # Load corporate catalysts
    corp_data = _load_json(CORPORATE_CATALYSTS_PATH)
    if isinstance(corp_data, dict):
        corp_entries = corp_data.get("events", corp_data.get("catalysts", []))
        if not isinstance(corp_entries, list):
            corp_entries = []
    elif isinstance(corp_data, list):
        corp_entries = corp_data
    else:
        corp_entries = []
    print(f"Corporate catalysts: {len(corp_entries)} entries")

    # Compute metrics
    metrics = compute_coverage_metrics(
        universe_tickers=universe_tickers,
        vnext_summaries=vnext_summaries,
        sec_8k_events=sec_events,
        adcom_events=adcom_events,
        pdufa_entries=pdufa_entries,
        corp_entries=corp_entries,
    )

    # Print summary
    print_summary(metrics)

    # Write artifact
    as_of_label = args.as_of_date or "latest"
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ARTIFACTS_DIR / f"coverage_audit_{as_of_label}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nArtifact written: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
