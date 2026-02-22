#!/usr/bin/env python3
"""
warm_price_cache.py — PIT price cache builder.

Freezes anchor close prices at snapshot time and backfills forward-horizon
prices as they mature.  Write-once: once a price cell is filled it is never
overwritten, making forward-return attribution immune to later yfinance
split/dividend adjustments.

Usage:
    # Create anchor cache for today's snapshot
    python tools/warm_price_cache.py --snapshot \
        --as-of-date 2026-02-19 \
        --rankings-csv data/snapshots/2026-02-19/rankings.csv \
        --price-csv production_data/price_history.csv

    # Backfill matured forward prices for all existing caches
    python tools/warm_price_cache.py --backfill-all \
        --price-csv production_data/price_history.csv \
        --cache-base data/caches/price_pit/PIT/ \
        --through-date 2026-02-20

Output structure:
    data/caches/price_pit/PIT/{as_of_date}/
        index.json      # manifest
        prices.csv      # one row per ticker, anchor + horizon columns
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import re
import sys
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Repo root — same pattern as run_daily_production.py
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("warm_price_cache")

# ---------------------------------------------------------------------------
# Schema contract constants
# ---------------------------------------------------------------------------
SCHEMA_VERSION = "price_pit_index.v1"
CACHE_TYPE = "price_pit"
DEFAULT_HORIZONS = [5, 20, 63]

# Split detection thresholds (same as repair_price_history_splits.py)
SPLIT_JUMP_THRESHOLD = 3.0      # +300% single-day
SPLIT_DROP_THRESHOLD = -0.75    # -75% single-day

# CSV columns
ANCHOR_COLS = ["ticker", "anchor_date", "anchor_close"]
HORIZON_COLS_TEMPLATE = ["h{h}_date", "h{h}_close"]


def _horizon_cols(horizons: List[int]) -> List[str]:
    """Build ordered CSV column names for the given horizons."""
    cols = list(ANCHOR_COLS)
    for h in sorted(horizons):
        cols.extend([f"h{h}_date", f"h{h}_close"])
    return cols


# ---------------------------------------------------------------------------
# Price CSV helpers
# ---------------------------------------------------------------------------

def _sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_price_map(price_csv: Path) -> Dict[str, Dict[str, float]]:
    """Load price_history.csv → {ticker: {date_str: close}}."""
    prices: Dict[str, Dict[str, float]] = {}
    with open(price_csv, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ticker = (row.get("ticker") or "").strip().upper()
            close_str = (row.get("close") or "").strip()
            date_str = (row.get("date") or "").strip()
            if not ticker or not close_str or not date_str:
                continue
            try:
                close = float(close_str)
            except (ValueError, TypeError):
                continue
            prices.setdefault(ticker, {})[date_str] = close
    return prices


def _sorted_trading_dates(prices: Dict[str, Dict[str, float]]) -> List[str]:
    """Extract sorted unique trading dates from price map."""
    all_dates: set = set()
    for td in prices.values():
        all_dates.update(td.keys())
    return sorted(all_dates)


def _trading_days_after(sorted_dates: List[str], start: str, n: int) -> Optional[str]:
    """Return the date n trading days after start, or None."""
    try:
        idx = sorted_dates.index(start)
    except ValueError:
        return None
    target = idx + n
    if target < len(sorted_dates):
        return sorted_dates[target]
    return None


def _find_anchor_date(sorted_dates: List[str], as_of: str) -> Optional[str]:
    """Find the latest trading date <= as_of_date."""
    best = None
    for d in sorted_dates:
        if d <= as_of:
            best = d
        else:
            break
    return best


# ---------------------------------------------------------------------------
# Schema validation (12 invariants)
# ---------------------------------------------------------------------------

def validate_price_pit_index(
    index: Dict[str, Any],
    expected_as_of_date: str,
    configured_horizons: Optional[List[int]] = None,
) -> Tuple[bool, str]:
    """Validate PIT price cache index.json against 12 invariants.

    Returns (ok, detail). detail is empty on success.
    """
    horizons = configured_horizons or DEFAULT_HORIZONS

    # 1. schema_version
    if index.get("schema_version") != SCHEMA_VERSION:
        return False, f"schema_version: expected {SCHEMA_VERSION}, got {index.get('schema_version')}"

    # 2. cache_type
    if index.get("cache_type") != CACHE_TYPE:
        return False, f"cache_type: expected {CACHE_TYPE}, got {index.get('cache_type')}"

    # 3. as_of_date valid ISO date
    aod = index.get("as_of_date", "")
    try:
        date.fromisoformat(aod)
    except (ValueError, TypeError):
        return False, f"as_of_date not a valid ISO date: {aod!r}"

    if aod != expected_as_of_date:
        return False, f"as_of_date mismatch: index={aod}, expected={expected_as_of_date}"

    # 4. created_at ends with Z
    ca = index.get("created_at", "")
    if not isinstance(ca, str) or not ca.endswith("Z"):
        return False, f"created_at must end with Z: {ca!r}"

    # 5. ticker_count int >= 0
    tc = index.get("ticker_count")
    if not isinstance(tc, int) or tc < 0:
        return False, f"ticker_count must be int >= 0: {tc!r}"

    # 6. anchor_date <= as_of_date
    ad = index.get("anchor_date", "")
    try:
        date.fromisoformat(ad)
    except (ValueError, TypeError):
        return False, f"anchor_date not valid ISO date: {ad!r}"
    if ad > aod:
        return False, f"anchor_date {ad} > as_of_date {aod}"

    # 7. coverage_pct in [0, 100]
    cov = index.get("coverage_pct")
    if not isinstance(cov, (int, float)) or cov < 0 or cov > 100:
        return False, f"coverage_pct must be in [0,100]: {cov!r}"

    # 8. coverage_pct consistent with ticker_count and tickers_missing_anchor
    missing = index.get("tickers_missing_anchor", [])
    if not isinstance(missing, list):
        return False, f"tickers_missing_anchor must be a list"
    n_missing = len(missing)
    if tc > 0:
        expected_cov = round(100.0 * (tc - n_missing) / tc, 2)
        if abs(cov - expected_cov) > 0.1:
            return False, (
                f"coverage_pct {cov} inconsistent with ticker_count={tc}, "
                f"missing={n_missing} (expected ~{expected_cov})"
            )

    # 9. horizons_filled is subset of configured horizons
    filled = index.get("horizons_filled", [])
    if not isinstance(filled, list):
        return False, "horizons_filled must be a list"
    if not set(filled).issubset(set(horizons)):
        return False, f"horizons_filled {filled} not subset of configured {horizons}"

    # 10. horizons_filled union horizons_pending == all configured horizons
    pending = index.get("horizons_pending", [])
    if not isinstance(pending, list):
        return False, "horizons_pending must be a list"
    if set(filled) | set(pending) != set(horizons):
        return False, (
            f"horizons_filled {filled} | horizons_pending {pending} "
            f"!= configured {horizons}"
        )

    # 11. source_csv_sha256 is 64-char hex
    sha = index.get("source_csv_sha256", "")
    if not isinstance(sha, str) or not re.match(r"^[0-9a-f]{64}$", sha):
        return False, f"source_csv_sha256 must be 64-char hex: {sha!r}"

    # 12. split_warnings is list of {ticker, horizon} dicts
    sw = index.get("split_warnings", [])
    if not isinstance(sw, list):
        return False, "split_warnings must be a list"
    for item in sw:
        if not isinstance(item, dict) or "ticker" not in item or "horizon" not in item:
            return False, f"split_warnings item must have ticker+horizon: {item!r}"

    return True, ""


# ---------------------------------------------------------------------------
# Split detection
# ---------------------------------------------------------------------------

def detect_split_warnings(
    rows: List[Dict[str, str]],
    jump_threshold: float = SPLIT_JUMP_THRESHOLD,
    drop_threshold: float = SPLIT_DROP_THRESHOLD,
) -> List[Dict[str, Any]]:
    """Flag tickers with suspicious anchor-to-forward price jumps.

    Returns list of {ticker, horizon, anchor_close, forward_close, pct_change}.
    """
    warnings: List[Dict[str, Any]] = []
    horizons_present = set()
    for row in rows:
        for key in row:
            m = re.match(r"h(\d+)_close", key)
            if m:
                horizons_present.add(int(m.group(1)))

    for row in rows:
        ticker = row.get("ticker", "")
        anchor_str = row.get("anchor_close", "")
        if not anchor_str:
            continue
        try:
            anchor = float(anchor_str)
        except (ValueError, TypeError):
            continue
        if anchor <= 0:
            continue

        for h in sorted(horizons_present):
            fwd_str = row.get(f"h{h}_close", "")
            if not fwd_str:
                continue
            try:
                fwd = float(fwd_str)
            except (ValueError, TypeError):
                continue
            if fwd <= 0:
                continue

            pct = (fwd / anchor) - 1.0
            if pct > jump_threshold or pct < drop_threshold:
                warnings.append({
                    "ticker": ticker,
                    "horizon": h,
                    "anchor_close": anchor,
                    "forward_close": fwd,
                    "pct_change": round(pct, 4),
                })

    return warnings


# ---------------------------------------------------------------------------
# Core: snapshot_price_cache
# ---------------------------------------------------------------------------

def snapshot_price_cache(
    as_of_date: str,
    price_csv: Path,
    rankings_csv: Path,
    out_dir: Path,
    horizons: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Create PIT price cache with anchor prices for all ranked tickers.

    Returns the index dict.
    """
    horizons = horizons or list(DEFAULT_HORIZONS)

    # Load ranked tickers from rankings.csv
    tickers: List[str] = []
    with open(rankings_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = (row.get("ticker") or "").strip().upper()
            if t:
                tickers.append(t)

    if not tickers:
        raise ValueError(f"No tickers found in {rankings_csv}")

    # Load prices
    prices = _load_price_map(price_csv)
    sorted_dates = _sorted_trading_dates(prices)

    # Find anchor date (latest trading date <= as_of_date)
    anchor_date = _find_anchor_date(sorted_dates, as_of_date)
    if anchor_date is None:
        raise ValueError(f"No trading dates <= {as_of_date} in {price_csv}")

    # Build rows
    columns = _horizon_cols(horizons)
    rows: List[Dict[str, str]] = []
    missing_tickers: List[str] = []

    for ticker in tickers:
        row: Dict[str, str] = {"ticker": ticker, "anchor_date": anchor_date}
        ticker_prices = prices.get(ticker, {})
        anchor_close = ticker_prices.get(anchor_date)

        if anchor_close is not None:
            row["anchor_close"] = str(round(anchor_close, 6))
        else:
            row["anchor_close"] = ""
            missing_tickers.append(ticker)

        # Horizon columns start empty
        for h in horizons:
            row[f"h{h}_date"] = ""
            row[f"h{h}_close"] = ""

        rows.append(row)

    # Compute coverage
    n_with_anchor = len(tickers) - len(missing_tickers)
    coverage_pct = round(100.0 * n_with_anchor / len(tickers), 2) if tickers else 0.0

    # Source hash
    source_sha = _sha256_file(price_csv)

    # Build index
    index = {
        "schema_version": SCHEMA_VERSION,
        "cache_type": CACHE_TYPE,
        "as_of_date": as_of_date,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_csv_sha256": source_sha,
        "ticker_count": len(tickers),
        "anchor_date": anchor_date,
        "horizons_filled": [],
        "horizons_pending": sorted(horizons),
        "split_warnings": [],
        "coverage_pct": coverage_pct,
        "tickers_missing_anchor": sorted(missing_tickers),
    }

    # Atomic write
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write prices.csv atomically
    prices_path = out_dir / "prices.csv"
    with tempfile.NamedTemporaryFile(
        mode="w", newline="", suffix=".csv", dir=str(out_dir), delete=False,
    ) as tmp:
        writer = csv.DictWriter(tmp, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})
        tmp_prices = tmp.name
    os.replace(tmp_prices, str(prices_path))

    # Write index.json atomically
    index_path = out_dir / "index.json"
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", dir=str(out_dir), delete=False,
    ) as tmp:
        json.dump(index, tmp, indent=2)
        tmp_index = tmp.name
    os.replace(tmp_index, str(index_path))

    logger.info(
        f"PIT price cache created: {as_of_date}, anchor={anchor_date}, "
        f"{n_with_anchor}/{len(tickers)} tickers, coverage={coverage_pct:.1f}%"
    )
    return index


# ---------------------------------------------------------------------------
# Core: backfill_forward_prices
# ---------------------------------------------------------------------------

def backfill_forward_prices(
    cache_dir: Path,
    price_csv: Path,
    horizon: int,
) -> Dict[str, Any]:
    """Fill matured forward prices for a single horizon. Write-once.

    Returns dict with {filled, skipped, split_warnings_added}.
    """
    index_path = cache_dir / "index.json"
    prices_path = cache_dir / "prices.csv"

    with open(index_path, encoding="utf-8") as f:
        index = json.load(f)

    # Skip if already filled
    if horizon in index.get("horizons_filled", []):
        return {"filled": 0, "skipped": True, "split_warnings_added": 0}

    # Load source prices
    prices = _load_price_map(price_csv)
    sorted_dates = _sorted_trading_dates(prices)

    # Read existing CSV rows
    rows: List[Dict[str, str]] = []
    with open(prices_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            rows.append(dict(row))

    date_col = f"h{horizon}_date"
    close_col = f"h{horizon}_close"
    filled = 0

    for row in rows:
        # Write-once: skip if already has a value
        if row.get(close_col, "").strip():
            continue

        anchor_date = row.get("anchor_date", "")
        ticker = row.get("ticker", "")
        if not anchor_date or not ticker:
            continue

        # Find forward date
        fwd_date = _trading_days_after(sorted_dates, anchor_date, horizon)
        if fwd_date is None:
            continue

        ticker_prices = prices.get(ticker, {})
        fwd_close = ticker_prices.get(fwd_date)
        if fwd_close is None:
            continue

        row[date_col] = fwd_date
        row[close_col] = str(round(fwd_close, 6))
        filled += 1

    # Check split warnings on newly filled data
    new_splits = detect_split_warnings(rows)
    # Only add split warnings for this horizon that aren't already recorded
    existing_sw = {(w["ticker"], w["horizon"]) for w in index.get("split_warnings", [])}
    new_horizon_splits = [
        w for w in new_splits
        if w["horizon"] == horizon and (w["ticker"], w["horizon"]) not in existing_sw
    ]

    # Update index
    if filled > 0:
        # Check if ALL rows are now filled for this horizon
        all_filled = all(
            row.get(close_col, "").strip()
            for row in rows
            if row.get("anchor_close", "").strip()  # only count tickers with anchors
        )

        if all_filled:
            index["horizons_filled"] = sorted(set(index["horizons_filled"]) | {horizon})
            index["horizons_pending"] = sorted(set(index["horizons_pending"]) - {horizon})
        elif filled > 0:
            # Partial fill — still mark as filled (forward date exists, some may be missing)
            # We consider a horizon "filled" once the forward date has matured
            index["horizons_filled"] = sorted(set(index["horizons_filled"]) | {horizon})
            index["horizons_pending"] = sorted(set(index["horizons_pending"]) - {horizon})

    if new_horizon_splits:
        index["split_warnings"].extend(
            {"ticker": w["ticker"], "horizon": w["horizon"]}
            for w in new_horizon_splits
        )

    # Write back atomically
    with tempfile.NamedTemporaryFile(
        mode="w", newline="", suffix=".csv", dir=str(cache_dir), delete=False,
    ) as tmp:
        writer = csv.DictWriter(tmp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        tmp_name = tmp.name
    os.replace(tmp_name, str(prices_path))

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", dir=str(cache_dir), delete=False,
    ) as tmp:
        json.dump(index, tmp, indent=2)
        tmp_name = tmp.name
    os.replace(tmp_name, str(index_path))

    logger.info(
        f"Backfill h{horizon}: {filled} prices filled, "
        f"{len(new_horizon_splits)} split warnings"
    )
    return {
        "filled": filled,
        "skipped": False,
        "split_warnings_added": len(new_horizon_splits),
    }


# ---------------------------------------------------------------------------
# Core: backfill_all_caches
# ---------------------------------------------------------------------------

def backfill_all_caches(
    cache_base: Path,
    price_csv: Path,
    through_date: str,
    horizons: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Iterate all existing PIT cache dirs, fill any matured horizons.

    Returns summary dict.
    """
    horizons = horizons or list(DEFAULT_HORIZONS)

    # Load prices once
    prices = _load_price_map(price_csv)
    sorted_dates = _sorted_trading_dates(prices)

    # Find through_date's position in sorted_dates
    through_idx = None
    for i, d in enumerate(sorted_dates):
        if d <= through_date:
            through_idx = i

    if through_idx is None:
        return {"dirs_processed": 0, "total_filled": 0, "detail": "no trading dates <= through_date"}

    results: Dict[str, Any] = {"dirs_processed": 0, "total_filled": 0, "by_date": {}}

    # Discover cache dirs
    if not cache_base.exists():
        return results

    date_re = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    for entry in sorted(cache_base.iterdir()):
        if not entry.is_dir() or not date_re.match(entry.name):
            continue
        cache_date = entry.name
        index_path = entry / "index.json"
        if not index_path.exists():
            continue

        with open(index_path, encoding="utf-8") as f:
            index = json.load(f)

        anchor_date = index.get("anchor_date", "")
        if not anchor_date:
            continue

        # Find anchor position in sorted dates
        try:
            anchor_idx = sorted_dates.index(anchor_date)
        except ValueError:
            continue

        date_results: Dict[str, Any] = {}
        for h in horizons:
            if h in index.get("horizons_filled", []):
                continue
            # Check if enough trading days have elapsed
            if anchor_idx + h <= through_idx:
                res = backfill_forward_prices(entry, price_csv, h)
                date_results[f"h{h}"] = res
                results["total_filled"] += res.get("filled", 0)

        results["dirs_processed"] += 1
        if date_results:
            results["by_date"][cache_date] = date_results

    logger.info(
        f"Backfill complete: {results['dirs_processed']} dirs, "
        f"{results['total_filled']} total prices filled"
    )
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="PIT price cache: snapshot anchors + backfill forward prices",
    )
    parser.add_argument(
        "--as-of-date", type=str,
        default=date.today().isoformat(),
        help="Snapshot date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--price-csv", type=Path,
        default=REPO_ROOT / "production_data" / "price_history.csv",
        help="Path to price_history.csv",
    )
    parser.add_argument(
        "--rankings-csv", type=Path, default=None,
        help="Path to rankings.csv (required for --snapshot)",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help="Output directory (default: data/caches/price_pit/PIT/{as_of_date})",
    )
    parser.add_argument(
        "--cache-base", type=Path,
        default=REPO_ROOT / "data" / "caches" / "price_pit" / "PIT",
        help="Base directory for all PIT price caches",
    )
    parser.add_argument(
        "--horizons", type=str, default="5,20,63",
        help="Comma-separated forward horizons in trading days",
    )
    parser.add_argument(
        "--snapshot", action="store_true",
        help="Create anchor cache for --as-of-date",
    )
    parser.add_argument(
        "--backfill-all", action="store_true",
        help="Backfill matured horizons for all existing caches",
    )
    parser.add_argument(
        "--through-date", type=str, default=None,
        help="Through-date for backfill (default: as-of-date)",
    )

    args = parser.parse_args()
    horizons = [int(h.strip()) for h in args.horizons.split(",")]

    if args.snapshot:
        if not args.rankings_csv:
            parser.error("--rankings-csv required for --snapshot")
        out_dir = args.out_dir or (args.cache_base / args.as_of_date)
        index = snapshot_price_cache(
            as_of_date=args.as_of_date,
            price_csv=args.price_csv,
            rankings_csv=args.rankings_csv,
            out_dir=out_dir,
            horizons=horizons,
        )
        print(json.dumps(index, indent=2))
        return 0

    if args.backfill_all:
        through = args.through_date or args.as_of_date
        result = backfill_all_caches(
            cache_base=args.cache_base,
            price_csv=args.price_csv,
            through_date=through,
            horizons=horizons,
        )
        print(json.dumps(result, indent=2, default=str))
        return 0

    parser.error("Must specify --snapshot or --backfill-all")
    return 1


if __name__ == "__main__":
    sys.exit(main())
