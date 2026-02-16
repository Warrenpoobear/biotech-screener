#!/usr/bin/env python3
"""Offline parity check: would delta warming reproduce full warm results?

For each consecutive pair of cached dates (D0, D1), decomposes new events
into "recent" (caught by narrow lookback) vs "late arrivals" (missed),
then simulates the full seed+merge+expire+dedup path.

Usage:
    python3 scripts/verify_delta_parity.py
    python3 scripts/verify_delta_parity.py --delta-lookback 7
    python3 scripts/verify_delta_parity.py --cache-dir cache/sec/8k_catalysts

No EDGAR calls — purely local file analysis.
"""
from __future__ import annotations

import argparse
import json
import re as _re
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CACHE_DIR = SCRIPT_DIR / "cache" / "sec" / "8k_catalysts"
EXPIRE_WINDOW_DAYS = 180

Key = tuple[str, str, str]  # (ticker, event_type, event_date)


@dataclass
class LateArrival:
    ticker: str
    event_type: str
    event_date: str
    disclosed_at: str
    age_days: int  # target_date - disclosed_at
    confidence: str
    precision: str


@dataclass
class PairResult:
    seed_date: str
    target_date: str
    seed_count: int
    full_count: int  # deduped keys in target
    new_in_target: int  # keys in target but not seed
    recent_new: int  # new keys with disclosed_at within lookback
    late_arrivals: list[LateArrival] = field(default_factory=list)
    expired_from_seed: int = 0
    disappeared_from_seed: int = 0  # in seed but not target (EDGAR flicker)
    simulated_final_count: int = 0
    missing_keys: list[Key] = field(default_factory=list)  # in target, not simulated
    extra_keys: list[Key] = field(default_factory=list)  # in simulated, not target

    @property
    def is_parity(self) -> bool:
        return len(self.missing_keys) == 0 and len(self.extra_keys) == 0


def _event_key(e: dict) -> Key:
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
    files = sorted(cache_dir.glob("8k_catalysts_*_*.json"))
    parsed: list[tuple[str, str, Path]] = []
    for f in files:
        m = _re.match(r"8k_catalysts_(\d{4}-\d{2}-\d{2})_([a-f0-9]{8})$", f.stem)
        if m:
            parsed.append((m.group(1), m.group(2), f))

    if not parsed:
        return []

    if version is None:
        ver_counts = Counter(v for _, v, _ in parsed)
        version = ver_counts.most_common(1)[0][0]

    seen_dates: set[str] = set()
    dated: list[tuple[str, Path]] = []
    for d, v, f in sorted(parsed):
        if v == version and d not in seen_dates:
            seen_dates.add(d)
            dated.append((d, f))

    return [(dated[i][1], dated[i + 1][1]) for i in range(len(dated) - 1)]


def analyze_pair(
    seed_events: list[dict],
    full_target_events: list[dict],
    seed_date: date,
    target_date: date,
    delta_lookback: int,
) -> PairResult:
    """Decompose D0→D1 into recent/late/expired/disappeared buckets."""
    # Dedup both sides by key
    seed_by_key: dict[Key, dict] = {}
    for e in seed_events:
        k = _event_key(e)
        if k not in seed_by_key:
            seed_by_key[k] = e

    full_by_key: dict[Key, dict] = {}
    for e in full_target_events:
        k = _event_key(e)
        if k not in full_by_key:
            full_by_key[k] = e

    seed_keys = set(seed_by_key)
    full_keys = set(full_by_key)

    # Step 1: new keys in target vs seed
    new_keys = full_keys - seed_keys
    delta_cutoff = (target_date - timedelta(days=delta_lookback)).isoformat()

    recent_new: list[Key] = []
    late_arrivals: list[LateArrival] = []
    for k in sorted(new_keys):
        e = full_by_key[k]
        disclosed = e.get("disclosed_at", "")
        if disclosed >= delta_cutoff:
            recent_new.append(k)
        else:
            age = (target_date - date.fromisoformat(disclosed)).days if disclosed else -1
            late_arrivals.append(LateArrival(
                ticker=k[0], event_type=k[1], event_date=k[2],
                disclosed_at=disclosed, age_days=age,
                confidence=e.get("confidence", "?"),
                precision=e.get("date_precision", "?"),
            ))

    # Step 2: disappeared from seed (EDGAR flicker)
    disappeared_keys = seed_keys - full_keys

    # Step 3: simulate delta output
    # merged = seed + recent_new events, dedup (seed wins)
    merged_by_key: dict[Key, dict] = dict(seed_by_key)
    for k in recent_new:
        if k not in merged_by_key:
            merged_by_key[k] = full_by_key[k]

    # expire
    expire_cutoff = (target_date - timedelta(days=EXPIRE_WINDOW_DAYS)).isoformat()
    expired_count = 0
    final_by_key: dict[Key, dict] = {}
    for k, e in merged_by_key.items():
        disclosed = e.get("disclosed_at", "")
        if disclosed and disclosed < expire_cutoff:
            expired_count += 1
        else:
            final_by_key[k] = e

    # Step 4: compare simulated vs full
    simulated_keys = set(final_by_key)
    missing = sorted(full_keys - simulated_keys)
    extra = sorted(simulated_keys - full_keys)

    return PairResult(
        seed_date=seed_date.isoformat(),
        target_date=target_date.isoformat(),
        seed_count=len(seed_keys),
        full_count=len(full_keys),
        new_in_target=len(new_keys),
        recent_new=len(recent_new),
        late_arrivals=late_arrivals,
        expired_from_seed=expired_count,
        disappeared_from_seed=len(disappeared_keys),
        simulated_final_count=len(final_by_key),
        missing_keys=missing,
        extra_keys=extra,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Offline delta-vs-full parity check for SEC 8-K caches.",
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=DEFAULT_CACHE_DIR,
        help="8-K cache directory",
    )
    parser.add_argument(
        "--delta-lookback", type=int, default=7,
        help="Delta lookback days (default: 7)",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Fail on ANY mismatch (default: only fail on high-quality late arrivals)",
    )
    args = parser.parse_args()

    pairs = find_cache_pairs(args.cache_dir)
    if not pairs:
        print("No consecutive cache pairs found.")
        return 1

    print(f"Found {len(pairs)} consecutive pairs in {args.cache_dir}")
    print(f"Delta lookback: {args.delta_lookback}d | Expire window: {EXPIRE_WINDOW_DAYS}d\n")

    # Header
    print(f"  {'pair':<27}  {'seed':>4}  {'full':>4}  "
          f"{'new':>3}  {'rcnt':>4}  {'late':>4}  "
          f"{'expd':>4}  {'gone':>4}  {'sim':>4}  status")
    print("  " + "-" * 90)

    all_results: list[PairResult] = []
    total_late: list[LateArrival] = []

    for seed_path, target_path in pairs:
        seed_date_str = seed_path.stem.split("_")[2]
        target_date_str = target_path.stem.split("_")[2]
        seed_dt = date.fromisoformat(seed_date_str)
        target_dt = date.fromisoformat(target_date_str)

        r = analyze_pair(
            _load_cache(seed_path), _load_cache(target_path),
            seed_dt, target_dt, args.delta_lookback,
        )
        all_results.append(r)
        total_late.extend(r.late_arrivals)

        status = "OK" if r.is_parity else "MISMATCH"
        print(
            f"  {r.seed_date} → {r.target_date}  "
            f"{r.seed_count:4d}  {r.full_count:4d}  "
            f"{r.new_in_target:3d}  {r.recent_new:4d}  {len(r.late_arrivals):4d}  "
            f"{r.expired_from_seed:4d}  {r.disappeared_from_seed:4d}  "
            f"{r.simulated_final_count:4d}  {status}"
        )

    # ── Late arrivals detail ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"LATE ARRIVALS: {len(total_late)} total across {len(pairs)} pairs")
    print(f"{'='*60}")

    if not total_late:
        print(f"  None. lookback={args.delta_lookback}d catches all new events.")
    else:
        print(f"\n  {'ticker':<6}  {'event_type':<20}  {'event_date':<11}  "
              f"{'disclosed':<11}  {'age':>4}  {'conf':<4}  {'prec'}")
        print("  " + "-" * 80)
        for la in sorted(total_late, key=lambda x: x.age_days, reverse=True):
            print(
                f"  {la.ticker:<6}  {la.event_type:<20}  {la.event_date:<11}  "
                f"{la.disclosed_at:<11}  {la.age_days:4d}d  {la.confidence:<4}  "
                f"{la.precision}"
            )

        # What lookback would catch them?
        print(f"\n  Lookback sensitivity:")
        for window in [14, 21, 30, 60, 90]:
            caught = sum(1 for la in total_late if la.age_days <= window)
            print(f"    {window:3d}d → catches {caught}/{len(total_late)} late arrivals")

    # ── Disappeared events (EDGAR flicker) ────────────────────────────
    total_disappeared = sum(r.disappeared_from_seed for r in all_results)
    total_extra = sum(len(r.extra_keys) for r in all_results)
    if total_disappeared:
        print(f"\n  EDGAR flicker: {total_disappeared} events disappeared from seed "
              f"({total_extra} show as 'extra' in delta output)")

    # ── Summary ───────────────────────────────────────────────────────
    n_parity = sum(1 for r in all_results if r.is_parity)
    n_mismatch = len(all_results) - n_parity
    total_missing = sum(len(r.missing_keys) for r in all_results)

    print(f"\n{'='*60}")
    print(f"VERDICT: {n_parity}/{len(all_results)} pairs at parity, "
          f"{n_mismatch} mismatches")
    print(f"  Late arrivals: {len(total_late)} "
          f"(events delta would miss)")
    print(f"  EDGAR flicker: {total_disappeared} disappeared, "
          f"{total_extra} extra in delta")
    print(f"  Missing from simulated: {total_missing} "
          f"(= late arrivals + expired late)")

    # Classify late arrivals by quality
    _JUNK_QUALITY = {"LOW"}
    _JUNK_PRECISION = {"HALF_YEAR"}
    high_quality_late = [
        la for la in total_late
        if la.confidence not in _JUNK_QUALITY or la.precision not in _JUNK_PRECISION
    ]

    if len(total_late) == 0:
        print(f"\n  lookback={args.delta_lookback}d is SAFE for this universe.")
    elif not high_quality_late:
        print(f"\n  All {len(total_late)} late arrivals are LOW/HALF_YEAR "
              f"(gated out by hard filter).")
        print(f"  lookback={args.delta_lookback}d is SAFE in practice.")
    else:
        print(f"\n  WARNING: {len(high_quality_late)} late arrivals are "
              f"NOT low-quality — consider widening lookback.")

    # Exit code: --strict fails on any mismatch; default only on real risk
    if args.strict:
        return 0 if n_mismatch == 0 else 2
    return 2 if high_quality_late else 0


if __name__ == "__main__":
    raise SystemExit(main())
