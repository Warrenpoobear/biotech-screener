#!/usr/bin/env python3
"""Offline parity check: would delta warming reproduce full warm results?

For each consecutive pair of cached dates (N, N+1), simulates what delta
warming would produce and compares against the actual full-warm cache.

Usage:
    python3 scripts/verify_delta_parity.py
    python3 scripts/verify_delta_parity.py --delta-lookback 7
    python3 scripts/verify_delta_parity.py --cache-dir cache/sec/8k_catalysts

No EDGAR calls — purely local file analysis.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CACHE_DIR = SCRIPT_DIR / "cache" / "sec" / "8k_catalysts"
EXPIRE_WINDOW_DAYS = 180


@dataclass
class PairResult:
    seed_date: str
    target_date: str
    full_count: int
    seed_count: int
    delta_narrow_count: int  # events in full_N+1 within delta window
    simulated_merged_count: int
    expired_count: int
    simulated_final_count: int
    missing_keys: list[tuple[str, str, str]] = field(default_factory=list)
    extra_keys: list[tuple[str, str, str]] = field(default_factory=list)

    @property
    def is_parity(self) -> bool:
        return len(self.missing_keys) == 0 and len(self.extra_keys) == 0


def _event_key(e: dict) -> tuple[str, str, str]:
    return (e["ticker"], e["event_type"], e["event_date"])


def _load_cache(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_cache_pairs(
    cache_dir: Path, *, version: str | None = None,
) -> list[tuple[Path, Path]]:
    """Find consecutive versioned cache files sorted by date.

    When *version* is given, only files with that PATTERN_VERSION are used.
    Otherwise, auto-detect the most common version across files.
    """
    import re as _re

    files = sorted(cache_dir.glob("8k_catalysts_*_*.json"))
    # Parse (date, version, path) triples
    parsed: list[tuple[str, str, Path]] = []
    for f in files:
        m = _re.match(r"8k_catalysts_(\d{4}-\d{2}-\d{2})_([a-f0-9]{8})$", f.stem)
        if m:
            parsed.append((m.group(1), m.group(2), f))

    if not parsed:
        return []

    # Pick version to use
    if version is None:
        from collections import Counter
        ver_counts = Counter(v for _, v, _ in parsed)
        version = ver_counts.most_common(1)[0][0]

    # Filter to chosen version, dedup by date (keep first)
    seen_dates: set[str] = set()
    dated: list[tuple[str, Path]] = []
    for d, v, f in sorted(parsed):
        if v == version and d not in seen_dates:
            seen_dates.add(d)
            dated.append((d, f))

    pairs = []
    for i in range(len(dated) - 1):
        pairs.append((dated[i][1], dated[i + 1][1]))
    return pairs


def simulate_delta(
    seed_events: list[dict],
    full_target_events: list[dict],
    target_date: date,
    delta_lookback: int,
) -> PairResult:
    """Simulate what delta warm would produce and compare to full warm."""
    seed_date_str = ""  # will be set by caller
    target_date_str = target_date.isoformat()

    # Build key sets
    seed_key_set = set()
    seed_by_key: dict[tuple, dict] = {}
    for e in seed_events:
        k = _event_key(e)
        if k not in seed_key_set:
            seed_key_set.add(k)
            seed_by_key[k] = e

    full_key_set = set()
    full_by_key: dict[tuple, dict] = {}
    for e in full_target_events:
        k = _event_key(e)
        if k not in full_key_set:
            full_key_set.add(k)
            full_by_key[k] = e

    # Simulate narrow EDGAR search: events in full_N+1 whose disclosed_at
    # falls within [target - delta_lookback, target]
    delta_cutoff = (target_date - timedelta(days=delta_lookback)).isoformat()
    narrow_events = []
    for e in full_target_events:
        disclosed = e.get("disclosed_at", "")
        if disclosed >= delta_cutoff:
            narrow_events.append(e)

    # Merge: seed + narrow, dedup by key (seed wins on collision)
    merged_by_key: dict[tuple, dict] = {}
    for e in seed_events:
        k = _event_key(e)
        if k not in merged_by_key:
            merged_by_key[k] = e
    for e in narrow_events:
        k = _event_key(e)
        if k not in merged_by_key:
            merged_by_key[k] = e

    # Expire: drop disclosed_at < (target - 180 days)
    expire_cutoff = (target_date - timedelta(days=EXPIRE_WINDOW_DAYS)).isoformat()
    expired_count = 0
    final_by_key: dict[tuple, dict] = {}
    for k, e in merged_by_key.items():
        disclosed = e.get("disclosed_at", "")
        if disclosed and disclosed < expire_cutoff:
            expired_count += 1
        else:
            final_by_key[k] = e

    # Compare
    simulated_keys = set(final_by_key.keys())
    # Also dedup the full target for fair comparison
    full_deduped_keys = set()
    for e in full_target_events:
        full_deduped_keys.add(_event_key(e))

    missing = sorted(full_deduped_keys - simulated_keys)
    extra = sorted(simulated_keys - full_deduped_keys)

    return PairResult(
        seed_date="",  # set by caller
        target_date=target_date_str,
        full_count=len(full_deduped_keys),
        seed_count=len(seed_key_set),
        delta_narrow_count=len(narrow_events),
        simulated_merged_count=len(merged_by_key),
        expired_count=expired_count,
        simulated_final_count=len(final_by_key),
        missing_keys=missing,
        extra_keys=extra,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline delta-vs-full parity check.")
    parser.add_argument(
        "--cache-dir", type=Path, default=DEFAULT_CACHE_DIR,
        help="8-K cache directory",
    )
    parser.add_argument(
        "--delta-lookback", type=int, default=7,
        help="Delta lookback days (default: 7)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show per-event detail for mismatches",
    )
    args = parser.parse_args()

    pairs = find_cache_pairs(args.cache_dir)
    if not pairs:
        print("No consecutive cache pairs found.")
        return 1

    print(f"Found {len(pairs)} consecutive pairs in {args.cache_dir}")
    print(f"Delta lookback: {args.delta_lookback} days | Expire window: {EXPIRE_WINDOW_DAYS} days")
    print()

    all_parity = True
    results: list[PairResult] = []

    for seed_path, target_path in pairs:
        seed_date_str = seed_path.stem.split("_")[2]
        target_date_str = target_path.stem.split("_")[2]
        target_dt = date.fromisoformat(target_date_str)

        seed_events = _load_cache(seed_path)
        full_target = _load_cache(target_path)

        r = simulate_delta(seed_events, full_target, target_dt, args.delta_lookback)
        r.seed_date = seed_date_str

        results.append(r)

        status = "PARITY" if r.is_parity else "MISMATCH"
        if not r.is_parity:
            all_parity = False

        print(
            f"  {r.seed_date} → {r.target_date}  "
            f"full={r.full_count}  seed={r.seed_count}  "
            f"narrow={r.delta_narrow_count}  merged={r.simulated_merged_count}  "
            f"expired={r.expired_count}  final={r.simulated_final_count}  "
            f"[{status}]"
        )

        if not r.is_parity:
            if r.missing_keys:
                print(f"    MISSING ({len(r.missing_keys)} events in full but not in delta):")
                for k in r.missing_keys[:10]:
                    print(f"      {k}")
                    if args.verbose:
                        # Find the event in full target
                        for e in full_target:
                            if _event_key(e) == k:
                                print(f"        disclosed_at={e['disclosed_at']}  "
                                      f"confidence={e['confidence']}  "
                                      f"precision={e['date_precision']}")
                                break
                if len(r.missing_keys) > 10:
                    print(f"      ... and {len(r.missing_keys) - 10} more")
            if r.extra_keys:
                print(f"    EXTRA ({len(r.extra_keys)} events in delta but not in full):")
                for k in r.extra_keys[:10]:
                    print(f"      {k}")

    # Summary
    n_parity = sum(1 for r in results if r.is_parity)
    n_mismatch = len(results) - n_parity
    total_missing = sum(len(r.missing_keys) for r in results)
    total_extra = sum(len(r.extra_keys) for r in results)

    print()
    print(f"Summary: {n_parity}/{len(results)} pairs at parity, "
          f"{n_mismatch} mismatches")
    if total_missing:
        print(f"  Total missing: {total_missing} events (in full, not delta)")
    if total_extra:
        print(f"  Total extra: {total_extra} events (in delta, not full)")

    if all_parity:
        print("\nAll pairs at parity. Delta warming is safe.")
    else:
        # Analyze missing events: how old are they?
        missing_disclosed: list[str] = []
        for seed_path, target_path in pairs:
            target_date_str = target_path.stem.split("_")[2]
            full_target = _load_cache(target_path)
            r = [x for x in results if x.target_date == target_date_str][0]
            for k in r.missing_keys:
                for e in full_target:
                    if _event_key(e) == k:
                        missing_disclosed.append(e["disclosed_at"])
                        break

        if missing_disclosed:
            missing_disclosed.sort()
            print(f"\n  Missing events disclosed_at range: "
                  f"{missing_disclosed[0]} → {missing_disclosed[-1]}")
            # How many would be caught by wider lookback?
            for window in [14, 30, 60]:
                caught = 0
                for seed_p, target_p in pairs:
                    td = target_p.stem.split("_")[2]
                    target_dt = date.fromisoformat(td)
                    cutoff = (target_dt - timedelta(days=window)).isoformat()
                    full_events = _load_cache(target_p)
                    r = [x for x in results if x.target_date == td][0]
                    for k in r.missing_keys:
                        for e in full_events:
                            if _event_key(e) == k and e["disclosed_at"] >= cutoff:
                                caught += 1
                                break
                print(f"  With lookback={window}d: would catch {caught}/{total_missing} missing")

    return 0 if all_parity else 2


if __name__ == "__main__":
    raise SystemExit(main())
