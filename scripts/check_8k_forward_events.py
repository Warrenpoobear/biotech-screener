#!/usr/bin/env python3
"""Post-run sanity check for SEC 8-K forward-looking catalyst events.

Reads the 8-K cache and prints a summary of future events relative to
the given as-of date. Exits 0 (PASS) if future events exist, 1 (FAIL) otherwise.

Usage:
    python scripts/check_8k_forward_events.py --as-of-date 2026-02-28
    python scripts/check_8k_forward_events.py --as-of-date 2026-02-28 --snapshot-dir data/snapshots
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path


def _find_cache_file(cache_dir: Path, as_of: date) -> Path | None:
    """Find the 8-K cache file for the given date (any pattern version)."""
    prefix = f"8k_catalysts_{as_of.isoformat()}_"
    for p in sorted(cache_dir.glob(f"{prefix}*.json"), reverse=True):
        if p.name.endswith(".meta.json"):
            continue
        return p
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check SEC 8-K forward-looking event coverage.",
    )
    parser.add_argument(
        "--as-of-date", required=True,
        help="Analysis date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--cache-dir", type=Path,
        default=Path("cache/sec/8k_catalysts"),
        help="8-K cache directory (default: cache/sec/8k_catalysts)",
    )
    parser.add_argument(
        "--snapshot-dir", type=Path, default=None,
        help="Snapshot directory to check shadow metrics alert",
    )
    args = parser.parse_args(argv)

    as_of = date.fromisoformat(args.as_of_date)
    cache_file = _find_cache_file(args.cache_dir, as_of)

    if not cache_file:
        print(f"FAIL: No 8-K cache file found for {as_of} in {args.cache_dir}")
        return 1

    events = json.loads(cache_file.read_text(encoding="utf-8"))
    if not isinstance(events, list):
        events = events.get("events", [])

    future = [
        e for e in events
        if e.get("event_date") and e["event_date"] > as_of.isoformat()
    ]

    tickers = sorted({e["ticker"] for e in future})
    soonest = sorted(future, key=lambda e: e["event_date"])[:5]

    print(f"Cache file: {cache_file.name}")
    print(f"Total events: {len(events)}")
    print(f"Future events (after {as_of}): {len(future)}")
    print(f"Distinct tickers with future events: {len(tickers)}")

    if soonest:
        print(f"\nTop {len(soonest)} soonest future events:")
        for e in soonest:
            print(
                f"  {e.get('ticker', '?'):<6} "
                f"{e.get('event_type', '?'):<20} "
                f"{e.get('event_date', '?'):<12} "
                f"{e.get('date_precision', '?'):<10} "
                f"{e.get('confidence', '?')}"
            )

    # Check shadow metrics alert if snapshot dir provided
    if args.snapshot_dir:
        shadow_path = args.snapshot_dir / as_of.isoformat() / "catalyst_shadow_metrics.json"
        if shadow_path.exists():
            shadow = json.loads(shadow_path.read_text(encoding="utf-8"))
            alert = shadow.get("sec8k_future_alert")
            if alert:
                print(f"\nShadow metrics alert: {alert}")

    verdict = "PASS" if len(future) > 0 else "FAIL"
    print(f"\nVerdict: {verdict}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
