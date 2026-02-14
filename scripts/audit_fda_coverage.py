#!/usr/bin/env python3
"""
audit_fda_coverage.py — FDA event coverage diagnostic

Reads the latest SEC 8-K cache, pdufa_dates.json, and ADCOM cache,
cross-references against universe tickers, and reports:
  - Tickers with FDA events by source
  - Event type distribution
  - Overall FDA coverage % of the universe

Usage:
  python3 scripts/audit_fda_coverage.py                        # latest cache
  python3 scripts/audit_fda_coverage.py --as-of-date 2025-10-31  # specific date
  python3 scripts/audit_fda_coverage.py --as-of-date 2025-10-31 --compare  # old vs new patterns

Outputs:
  - artifacts/fda_coverage_audit.json  (machine-readable)
  - Console summary
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from wake_robin_data_pipeline.collectors.sec_8k_catalyst_collector import PATTERN_VERSION

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OLD_PATTERN_VERSION = "221244b7"  # Pre-expansion hash (for --compare)

# Search both cache locations (project root and pipeline subdirectory)
CACHE_DIRS_8K = [
    PROJECT_ROOT / "cache" / "sec" / "8k_catalysts",
    PROJECT_ROOT / "wake_robin_data_pipeline" / "cache" / "sec" / "8k_catalysts",
]
CACHE_DIRS_ADCOM = [
    PROJECT_ROOT / "cache" / "fda",
    PROJECT_ROOT / "wake_robin_data_pipeline" / "cache" / "fda",
]
PDUFA_PATH = PROJECT_ROOT / "production_data" / "pdufa_dates.json"
CORPORATE_CATALYSTS_PATH = PROJECT_ROOT / "production_data" / "corporate_catalysts.json"
UNIVERSE_PATH = PROJECT_ROOT / "production_data" / "universe.json"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

FDA_EVENT_TYPES = {
    "FDA_PDUFA_DATE", "FDA_ADCOM", "FDA_CRL", "FDA_RTF",
    "FDA_WARNING_LETTER", "FDA_APPROVAL", "FDA_SUBMISSION",
    "FDA_DESIGNATION",
}


def _find_cache_for_date(dirs, prefix, as_of_date, version_suffix=None):
    """Find cache file for a specific date and optional version suffix."""
    if as_of_date and version_suffix:
        target = f"{prefix}{as_of_date}_{version_suffix}.json"
    elif as_of_date:
        target = f"{prefix}{as_of_date}_"
    else:
        target = None

    for d in dirs:
        if not d.exists():
            continue
        if target and version_suffix:
            candidate = d / target
            if candidate.exists():
                return candidate
        elif target:
            # Match any version for this date
            for f in sorted(d.glob(f"{target}*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
                return f
        else:
            # Latest by mtime
            candidates = list(d.glob(f"{prefix}*.json"))
            if candidates:
                return max(candidates, key=lambda p: p.stat().st_mtime)
    return None


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


def _load_json(path):
    """Load JSON file, return empty list/dict on failure."""
    if not path or not Path(path).exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_fda_tickers(events):
    """Extract universe-intersected FDA tickers from event list."""
    fda_events = [e for e in events if e.get("event_type") in FDA_EVENT_TYPES]
    fda_tickers = {e.get("ticker") for e in fda_events if e.get("ticker")}
    return fda_events, fda_tickers


def _cache_version_suffix(path):
    """Extract pattern version suffix from cache filename."""
    if not path:
        return ""
    m = re.search(r"_([0-9a-f]{8})\.json$", path.name)
    return m.group(1) if m else "(unversioned)"


def main():
    parser = argparse.ArgumentParser(description="FDA event coverage diagnostic")
    parser.add_argument(
        "--as-of-date", type=str, default=None,
        help="Specific as-of date (YYYY-MM-DD). Default: latest available cache.",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Compare old-pattern vs new-pattern cache for the same date.",
    )
    args = parser.parse_args()

    # Load universe
    universe_data = _load_json(UNIVERSE_PATH)
    if isinstance(universe_data, dict):
        universe_tickers = set(universe_data.get("tickers", []))
    elif isinstance(universe_data, list):
        universe_tickers = {
            e.get("ticker", e) if isinstance(e, dict) else str(e)
            for e in universe_data
        }
    else:
        universe_tickers = set()

    universe_size = len(universe_tickers)
    print(f"Universe: {universe_size} tickers")

    # ── Comparison mode ──
    if args.compare:
        if not args.as_of_date:
            print("ERROR: --compare requires --as-of-date", file=sys.stderr)
            return 1

        old_cache = _find_cache_for_date(CACHE_DIRS_8K, "8k_catalysts_", args.as_of_date, OLD_PATTERN_VERSION)
        new_cache = _find_cache_for_date(CACHE_DIRS_8K, "8k_catalysts_", args.as_of_date, PATTERN_VERSION)

        old_events = _load_json(old_cache) if old_cache else []
        new_events = _load_json(new_cache) if new_cache else []

        old_fda, old_fda_tickers = _extract_fda_tickers(old_events)
        new_fda, new_fda_tickers = _extract_fda_tickers(new_events)
        old_in_u = old_fda_tickers & universe_tickers
        new_in_u = new_fda_tickers & universe_tickers

        print(f"\n{'='*60}")
        print(f"PATTERN COMPARISON for as_of_date={args.as_of_date}")
        print(f"{'='*60}")
        print(f"  Old patterns ({OLD_PATTERN_VERSION}): {old_cache.name if old_cache else 'NOT FOUND'}")
        print(f"    Total events: {len(old_events)}, FDA events: {len(old_fda)}")
        print(f"    FDA tickers in universe: {len(old_in_u)}")
        print(f"    Types: {dict(Counter(e['event_type'] for e in old_fda).most_common())}")
        print(f"  New patterns ({PATTERN_VERSION}): {new_cache.name if new_cache else 'NOT FOUND'}")
        print(f"    Total events: {len(new_events)}, FDA events: {len(new_fda)}")
        print(f"    FDA tickers in universe: {len(new_in_u)}")
        print(f"    Types: {dict(Counter(e['event_type'] for e in new_fda).most_common())}")

        gained = new_in_u - old_in_u
        lost = old_in_u - new_in_u
        print(f"\n  DELTA:")
        print(f"    Gained: {len(gained)} tickers — {sorted(gained)}" if gained else "    Gained: 0")
        print(f"    Lost:   {len(lost)} tickers — {sorted(lost)}" if lost else "    Lost:   0")
        print(f"    Net:    {len(new_in_u) - len(old_in_u):+d}")

        # Show new events that don't exist in old (by ticker+type+date key)
        old_keys = {(e["ticker"], e["event_type"], e["event_date"]) for e in old_fda}
        new_only = [e for e in new_fda if (e["ticker"], e["event_type"], e["event_date"]) not in old_keys]
        if new_only:
            print(f"\n  NEW FDA events (not in old cache):")
            for e in sorted(new_only, key=lambda x: x["ticker"]):
                print(f"    {e['ticker']:6s} {e['event_type']:20s} {e['event_date']} prec={e.get('date_precision','?')}")

        # Write comparison artifact
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        comp = {
            "as_of_date": args.as_of_date,
            "old_version": OLD_PATTERN_VERSION,
            "new_version": PATTERN_VERSION,
            "old_fda_count": len(old_fda),
            "new_fda_count": len(new_fda),
            "old_tickers": sorted(old_in_u),
            "new_tickers": sorted(new_in_u),
            "gained_tickers": sorted(gained),
            "lost_tickers": sorted(lost),
            "new_only_events": new_only,
        }
        out_path = ARTIFACTS_DIR / f"fda_coverage_comparison_{args.as_of_date}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(comp, f, indent=2)
        print(f"\nArtifact written: {out_path}")
        return 0

    # ── Standard audit mode ──

    # Find SEC 8-K cache
    if args.as_of_date:
        latest_8k = _find_cache_for_date(CACHE_DIRS_8K, "8k_catalysts_", args.as_of_date, PATTERN_VERSION)
        if not latest_8k:
            # Fall back to any version for this date
            latest_8k = _find_cache_for_date(CACHE_DIRS_8K, "8k_catalysts_", args.as_of_date)
    else:
        latest_8k = _find_latest_cache(CACHE_DIRS_8K, "8k_catalysts_")

    sec_events = _load_json(latest_8k) if latest_8k else []
    cache_suffix = _cache_version_suffix(latest_8k)

    print(f"\nPATTERN_VERSION in code: {PATTERN_VERSION}")
    print(f"SEC 8-K cache: {latest_8k.name if latest_8k else 'NOT FOUND'}")
    if cache_suffix:
        match_status = "MATCH" if cache_suffix == PATTERN_VERSION else f"STALE (code={PATTERN_VERSION})"
        print(f"  Cache pattern version: {cache_suffix} — {match_status}")
    print(f"  Total events: {len(sec_events)}")

    sec_fda_events, sec_fda_tickers = _extract_fda_tickers(sec_events)
    sec_all_type_counts = Counter(e.get("event_type") for e in sec_events)
    sec_fda_type_counts = Counter(e.get("event_type") for e in sec_fda_events)
    sec_fda_in_universe = sec_fda_tickers & universe_tickers

    print(f"  FDA events: {len(sec_fda_events)}")
    print(f"  FDA tickers: {len(sec_fda_tickers)} ({len(sec_fda_in_universe)} in universe)")
    print(f"  All event types: {dict(sec_all_type_counts.most_common())}")
    print(f"  FDA event types: {dict(sec_fda_type_counts.most_common())}")

    # ── Source 2: PDUFA dates ──
    pdufa_data = _load_json(PDUFA_PATH)
    if isinstance(pdufa_data, dict):
        pdufa_entries = pdufa_data.get("events", pdufa_data.get("dates", []))
        if not isinstance(pdufa_entries, list):
            pdufa_entries = list(pdufa_data.values()) if pdufa_data else []
    else:
        pdufa_entries = pdufa_data

    pdufa_tickers = set()
    for entry in pdufa_entries:
        if isinstance(entry, dict):
            t = entry.get("ticker", entry.get("symbol", ""))
            if t:
                pdufa_tickers.add(t.upper())
    pdufa_in_universe = pdufa_tickers & universe_tickers

    print(f"\nPDUFA dates: {PDUFA_PATH.name}")
    print(f"  Entries: {len(pdufa_entries)}")
    print(f"  Tickers: {len(pdufa_tickers)} ({len(pdufa_in_universe)} in universe)")

    # ── Source 3: Corporate catalysts ──
    corp_data = _load_json(CORPORATE_CATALYSTS_PATH)
    if isinstance(corp_data, dict):
        corp_entries = corp_data.get("events", corp_data.get("catalysts", []))
        if not isinstance(corp_entries, list):
            corp_entries = list(corp_data.values()) if corp_data else []
    else:
        corp_entries = corp_data

    corp_fda_entries = [
        e for e in corp_entries
        if isinstance(e, dict) and "fda" in str(e.get("event_type", "")).lower()
    ]
    corp_fda_tickers = {e.get("ticker", "").upper() for e in corp_fda_entries if e.get("ticker")}
    corp_fda_in_universe = corp_fda_tickers & universe_tickers

    print(f"\nCorporate catalysts: {CORPORATE_CATALYSTS_PATH.name}")
    print(f"  Total entries: {len(corp_entries)}")
    print(f"  FDA-related: {len(corp_fda_entries)}")
    print(f"  FDA tickers: {len(corp_fda_tickers)} ({len(corp_fda_in_universe)} in universe)")

    # ── Source 4: ADCOM cache ──
    if args.as_of_date:
        latest_adcom = _find_cache_for_date(CACHE_DIRS_ADCOM, "adcom_calendar_", args.as_of_date)
    else:
        latest_adcom = _find_latest_cache(CACHE_DIRS_ADCOM, "adcom_calendar_")
    adcom_events = _load_json(latest_adcom) if latest_adcom else []
    adcom_tickers = set()
    for e in adcom_events:
        if isinstance(e, dict):
            t = e.get("ticker", "")
            if t:
                adcom_tickers.add(t.upper())
    adcom_in_universe = adcom_tickers & universe_tickers

    print(f"\nADCOM cache: {latest_adcom.name if latest_adcom else 'NOT FOUND'}")
    print(f"  Events: {len(adcom_events)}")
    print(f"  Tickers: {len(adcom_tickers)} ({len(adcom_in_universe)} in universe)")

    # ── Combined coverage ──
    all_fda_tickers = sec_fda_in_universe | pdufa_in_universe | corp_fda_in_universe | adcom_in_universe
    coverage_pct = (len(all_fda_tickers) / universe_size * 100) if universe_size else 0

    print("\n" + "=" * 60)
    print("COMBINED FDA COVERAGE")
    print("=" * 60)
    print(f"  Tickers with any FDA event: {len(all_fda_tickers)} / {universe_size}")
    print(f"  Coverage: {coverage_pct:.1f}%")
    if all_fda_tickers:
        source_breakdown = defaultdict(set)
        for t in sec_fda_in_universe:
            source_breakdown["SEC_8K"].add(t)
        for t in pdufa_in_universe:
            source_breakdown["PDUFA"].add(t)
        for t in corp_fda_in_universe:
            source_breakdown["CORPORATE"].add(t)
        for t in adcom_in_universe:
            source_breakdown["ADCOM"].add(t)
        for src, tickers in sorted(source_breakdown.items()):
            print(f"    {src}: {len(tickers)} tickers — {sorted(tickers)[:10]}{'...' if len(tickers) > 10 else ''}")

    # ── Write artifact ──
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    audit = {
        "as_of_date": args.as_of_date or "(latest)",
        "pattern_version": PATTERN_VERSION,
        "universe_size": universe_size,
        "sec_8k_cache": str(latest_8k) if latest_8k else None,
        "sec_8k_total_events": len(sec_events),
        "sec_8k_fda_events": len(sec_fda_events),
        "sec_8k_fda_tickers_in_universe": sorted(sec_fda_in_universe),
        "sec_8k_event_type_distribution": dict(sec_all_type_counts.most_common()),
        "sec_8k_fda_type_distribution": dict(sec_fda_type_counts.most_common()),
        "pdufa_entries": len(pdufa_entries),
        "pdufa_tickers_in_universe": sorted(pdufa_in_universe),
        "corporate_fda_entries": len(corp_fda_entries),
        "corporate_fda_tickers_in_universe": sorted(corp_fda_in_universe),
        "adcom_events": len(adcom_events),
        "adcom_tickers_in_universe": sorted(adcom_in_universe),
        "combined_fda_tickers": sorted(all_fda_tickers),
        "combined_fda_coverage_pct": round(coverage_pct, 2),
    }

    out_path = ARTIFACTS_DIR / "fda_coverage_audit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)
    print(f"\nArtifact written: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
