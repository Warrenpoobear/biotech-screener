#!/usr/bin/env python3
"""
Backfill IR events + press release future-date caches per day.

Usage:
    python3 scripts/backfill_company_calendar.py \
        --start 2026-01-15 --end 2026-02-28 \
        --sources ir_events,press_releases

Produces:
    cache/ir/ir_events_{YYYY-MM-DD}.json
    cache/press/press_release_events_{YYYY-MM-DD}.json

When the universe lacks ir_url / pr_rss_url entries, the collectors return []
early without writing cache. This script writes empty sentinel caches in that
case so future warm_caches calls short-circuit properly.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List

# Allow importing project root modules when run as script
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DAY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _all_days(start: date, end: date):
    """Yield every calendar day in [start, end] inclusive."""
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def _load_universe(data_dir: Path) -> List[Dict]:
    """Load universe.json from data_dir."""
    p = data_dir / "universe.json"
    if not p.exists():
        logger.error("universe.json not found at %s", p)
        return []
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_empty_cache(cache_path: Path, schema: str, collector_version: str,
                       as_of_date: date) -> None:
    """Write a minimal empty cache wrapper so warm_caches short-circuits."""
    wrapper = {
        "schema": schema,
        "as_of_date": as_of_date.isoformat(),
        "collector_version": collector_version,
        "events": [],
        "stats": {
            "fetched_pages": 0,
            "parsed_items": 0,
            "matched": 0,
            "unmatched": 0,
            "unmatched_samples": [],
        },
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(wrapper, f, sort_keys=True, indent=2)


def _validate_cache(cache_path: Path, as_of_str: str) -> Dict:
    """Validate a cache file. Returns {events, past_dates, bad_precision}."""
    if not cache_path.exists():
        return {"events": 0, "past_dates": 0, "bad_precision": 0, "exists": False}
    data = json.loads(cache_path.read_text())
    events = data.get("events", [])
    past = [e for e in events if (e.get("event_date") or "") < as_of_str]
    bad = [e for e in events if not _DAY_RE.match(e.get("event_date", ""))]
    unmatched = data.get("stats", {}).get("unmatched_samples", [])
    return {
        "events": len(events),
        "past_dates": len(past),
        "bad_precision": len(bad),
        "unmatched_cap_ok": len(unmatched) <= 100,
        "exists": True,
    }


# ---------------------------------------------------------------------------
# Core backfill
# ---------------------------------------------------------------------------

def backfill(
    start: date,
    end: date,
    sources: List[str],
    data_dir: Path,
    ir_cache_dir: Path,
    press_cache_dir: Path,
    sleep_seconds: float = 1.0,
    skip_existing: bool = True,
) -> Dict:
    """Run backfill for the given date range. Returns summary stats."""
    from wake_robin_data_pipeline.collectors.ir_events_collector import (
        collect_ir_events, SCHEMA as IR_SCHEMA, COLLECTOR_VERSION as IR_VERSION,
    )
    from wake_robin_data_pipeline.collectors.press_release_collector import (
        collect_press_release_events, SCHEMA as PR_SCHEMA,
        COLLECTOR_VERSION as PR_VERSION,
    )

    universe = _load_universe(data_dir)
    if not universe:
        print("ERROR: empty universe", file=sys.stderr)
        return {}

    # Merge curated company calendar source URLs (ir_url, pr_rss_url)
    sources_path = data_dir / "company_calendar_sources.json"
    if sources_path.exists():
        try:
            from wake_robin_data_pipeline.company_calendar_sources import (
                load_company_calendar_sources,
                merge_universe_with_company_calendar_sources,
            )
            cal_sources = load_company_calendar_sources(sources_path)
            if cal_sources:
                universe, merge_stats = merge_universe_with_company_calendar_sources(
                    universe, cal_sources,
                )
                print(f"Calendar sources merged: ir_added={merge_stats['ir_url_added']} "
                      f"pr_added={merge_stats['pr_rss_url_added']}")
        except Exception as e:
            logger.warning("Failed to merge company calendar sources: %s", e)

    do_ir = "ir_events" in sources
    do_pr = "press_releases" in sources

    stats = {
        "days_attempted": 0,
        "ir_days_ok": 0, "ir_total_events": 0, "ir_skipped": 0,
        "pr_days_ok": 0, "pr_total_events": 0, "pr_skipped": 0,
        "failures": 0, "violations": 0,
    }

    for d in _all_days(start, end):
        stats["days_attempted"] += 1
        as_of_str = d.isoformat()
        ir_n = 0
        pr_n = 0

        # --- IR Events ---
        if do_ir:
            ir_path = ir_cache_dir / f"ir_events_{as_of_str}.json"
            if skip_existing and ir_path.exists():
                v = _validate_cache(ir_path, as_of_str)
                ir_n = v["events"]
                stats["ir_skipped"] += 1
            else:
                try:
                    events = collect_ir_events(
                        universe=universe,
                        as_of_date=d,
                        cache_dir=ir_cache_dir,
                    )
                    ir_n = len(events)
                    # Write empty sentinel if collector returned early
                    if not ir_path.exists():
                        _write_empty_cache(ir_path, IR_SCHEMA, IR_VERSION, d)
                except Exception as e:
                    logger.error("IR backfill failed for %s: %s", d, e)
                    stats["failures"] += 1

            # Validate
            v = _validate_cache(ir_path, as_of_str)
            if v["past_dates"] > 0 or v["bad_precision"] > 0:
                print(f"VIOLATION {d}: IR past_dates={v['past_dates']} "
                      f"bad_precision={v['bad_precision']}", file=sys.stderr)
                stats["violations"] += 1
            stats["ir_days_ok"] += 1
            stats["ir_total_events"] += ir_n

        # --- Press Releases ---
        if do_pr:
            pr_path = press_cache_dir / f"press_release_events_{as_of_str}.json"
            if skip_existing and pr_path.exists():
                v = _validate_cache(pr_path, as_of_str)
                pr_n = v["events"]
                stats["pr_skipped"] += 1
            else:
                try:
                    events = collect_press_release_events(
                        universe=universe,
                        as_of_date=d,
                        cache_dir=press_cache_dir,
                    )
                    pr_n = len(events)
                    # Write empty sentinel if collector returned early
                    if not pr_path.exists():
                        _write_empty_cache(pr_path, PR_SCHEMA, PR_VERSION, d)
                except Exception as e:
                    logger.error("PR backfill failed for %s: %s", d, e)
                    stats["failures"] += 1

            # Validate
            v = _validate_cache(pr_path, as_of_str)
            if v["past_dates"] > 0 or v["bad_precision"] > 0:
                print(f"VIOLATION {d}: PR past_dates={v['past_dates']} "
                      f"bad_precision={v['bad_precision']}", file=sys.stderr)
                stats["violations"] += 1
            stats["pr_days_ok"] += 1
            stats["pr_total_events"] += pr_n

        print(f"{d}: ir={ir_n} pr={pr_n}")

        if sleep_seconds > 0 and d < end:
            time.sleep(sleep_seconds)

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Backfill IR events + press release caches per day"
    )
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument(
        "--sources", default="ir_events,press_releases",
        help="Comma-separated: ir_events,press_releases (default: both)",
    )
    parser.add_argument("--data-dir", default="production_data")
    parser.add_argument("--ir-cache-dir", default="cache/ir")
    parser.add_argument("--press-cache-dir", default="cache/press")
    parser.add_argument("--sleep-seconds", type=float, default=0.0,
                        help="Sleep between dates (default: 0, no network needed)")
    parser.add_argument("--max-days", type=int, default=0,
                        help="Cap on number of days to process (0=unlimited)")
    parser.add_argument("--no-skip-existing", action="store_true",
                        help="Reprocess even if cache file exists")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    sources = [s.strip() for s in args.sources.split(",")]

    if args.max_days > 0:
        capped_end = start + timedelta(days=args.max_days - 1)
        end = min(end, capped_end)

    print(f"Backfill: {start} → {end} ({(end - start).days + 1} days)")
    print(f"Sources: {sources}")
    print()

    stats = backfill(
        start=start,
        end=end,
        sources=sources,
        data_dir=Path(args.data_dir),
        ir_cache_dir=Path(args.ir_cache_dir),
        press_cache_dir=Path(args.press_cache_dir),
        sleep_seconds=args.sleep_seconds,
        skip_existing=not args.no_skip_existing,
    )

    print()
    print("=" * 60)
    print("BACKFILL SUMMARY")
    print("=" * 60)
    print(f"Days attempted:   {stats.get('days_attempted', 0)}")
    print(f"IR days OK:       {stats.get('ir_days_ok', 0)}")
    print(f"IR total events:  {stats.get('ir_total_events', 0)}")
    print(f"IR skipped:       {stats.get('ir_skipped', 0)}")
    print(f"PR days OK:       {stats.get('pr_days_ok', 0)}")
    print(f"PR total events:  {stats.get('pr_total_events', 0)}")
    print(f"PR skipped:       {stats.get('pr_skipped', 0)}")
    print(f"Failures:         {stats.get('failures', 0)}")
    print(f"Violations:       {stats.get('violations', 0)}")

    if stats.get("violations", 0) > 0:
        print("\nFAIL: Invariant violations detected!")
        sys.exit(1)
    else:
        print("\nPASS: All invariants hold (past_dates=0, bad_precision=0)")

    sys.exit(0)


if __name__ == "__main__":
    main()
