"""
Assemble PIT forward returns for the gap period 2026-01-16 to 2026-05-07.

Reads gap snapshot rankings and pit_archive price histories to produce a
forward returns panel compatible with data/snapshots/_forward_returns_panel.csv.

RESEARCH_ONLY — do not import from or write to any production module.
Production model freeze is ACTIVE; this script is read-only w.r.t. production data.

Usage:
    python3 scripts/research/assemble_gap_forward_returns.py [--dry-run] [--out PATH]

    --dry-run    Run validation only; do not write output file.
    --out PATH   Output CSV path (default: artifacts/audit/gap_forward_returns_panel.csv)

Governance:
    - RESEARCH_ONLY_NO_MODEL_CHANGE
    - Output is quarantined until manually reviewed and explicitly accepted
    - Do not merge output into _forward_returns_panel.csv without operator approval
    - Feasibility memo: artifacts/audit/PIT_PRICE_GAP_CLOSURE_FEASIBILITY_2026_06_22.md
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
SNAP_ROOT = REPO_ROOT / "data" / "snapshots"
ARCH_ROOT = REPO_ROOT / "data" / "pit_archives"

GAP_START = "2026-01-16"
GAP_END = "2026-05-07"

# ATXS acquired 2026-01-23; no prices exist after this date
ATXS_LAST_TRADING_DATE = "2026-01-23"

# Forward horizons to compute (trading days)
HORIZONS = [1, 3, 5, 20, 60]

# Minimum top-30 tickers with non-null anchor for a snapshot to be valid
MIN_ANCHOR_COVERAGE = 28

DEFAULT_OUT = REPO_ROOT / "artifacts" / "audit" / "gap_forward_returns_panel.csv"


# ---------------------------------------------------------------------------
# Price loading (self-contained; does not import production modules)
# ---------------------------------------------------------------------------


def load_price_series(csv_path: Path) -> Dict[str, Dict[str, float]]:
    """Load price_history.csv -> {ticker: {date_str: close}}."""
    prices: Dict[str, Dict[str, float]] = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_archive_manifest(arch_date: str) -> Optional[str]:
    """Return None if manifest SHA256 checks pass; else error string."""
    arch_dir = ARCH_ROOT / arch_date
    manifest_path = arch_dir / "manifest.json"
    price_path = arch_dir / "price_history.csv"
    if not manifest_path.exists():
        return f"no manifest.json in {arch_date}"
    with open(manifest_path) as f:
        manifest = json.load(f)
    expected = (manifest.get("files", {}).get("price_history.csv") or {}).get("sha256")
    if not expected:
        return f"manifest missing price_history.csv sha256 for {arch_date}"
    actual = sha256_file(price_path)
    if actual != expected:
        return f"SHA256 mismatch for {arch_date}/price_history.csv: {actual} != {expected}"
    return None


# ---------------------------------------------------------------------------
# Snapshot discovery
# ---------------------------------------------------------------------------


def discover_gap_snapshots() -> List[str]:
    """Return sorted list of canonical YYYY-MM-DD gap snapshot dates."""
    dates = []
    for entry in os.listdir(SNAP_ROOT):
        if (
            len(entry) == 10
            and entry[4] == "-"
            and entry[7] == "-"
            and GAP_START <= entry <= GAP_END
            and (SNAP_ROOT / entry / "rankings.csv").is_file()
        ):
            dates.append(entry)
    return sorted(dates)


def resolve_archive(snap_date: str) -> Tuple[str, bool]:
    """
    Return (archive_date, is_fallback).
    If the snapshot's own archive exists, use it (is_fallback=False).
    Otherwise find nearest prior archive (is_fallback=True).
    """
    if (ARCH_ROOT / snap_date).is_dir():
        return snap_date, False
    all_arch = sorted(d for d in os.listdir(ARCH_ROOT) if len(d) == 10 and d < snap_date)
    if not all_arch:
        raise RuntimeError(f"No prior pit_archive found for {snap_date}")
    return all_arch[-1], True


# ---------------------------------------------------------------------------
# Anchor price resolution
# ---------------------------------------------------------------------------


def resolve_anchor(
    ticker: str,
    snap_date: str,
    prices: Dict[str, Dict[str, float]],
    sorted_dates: List[str],
) -> Tuple[Optional[float], Optional[str]]:
    """
    Return (anchor_close, anchor_date).
    Uses snap_date close if available; else most recent prior trading date in archive.
    """
    ticker_prices = prices.get(ticker, {})
    if not ticker_prices:
        return None, None

    # Exact date
    if snap_date in ticker_prices:
        return ticker_prices[snap_date], snap_date

    # Most recent prior trading date in archive (handles market holidays)
    candidates = [d for d in sorted_dates if d <= snap_date and d in ticker_prices]
    if candidates:
        best = candidates[-1]
        return ticker_prices[best], best

    return None, None


# ---------------------------------------------------------------------------
# Forward return computation
# ---------------------------------------------------------------------------


def trading_days_after(sorted_dates: List[str], start: str, n: int) -> Optional[str]:
    """Return the date n trading days after start, or None."""
    try:
        idx = sorted_dates.index(start)
    except ValueError:
        return None
    target = idx + n
    return sorted_dates[target] if target < len(sorted_dates) else None


def compute_return(
    ticker: str,
    anchor_date: str,
    horizon: int,
    prices: Dict[str, Dict[str, float]],
    anchor_close: float,
    sorted_dates: List[str],
) -> Optional[float]:
    fwd_date = trading_days_after(sorted_dates, anchor_date, horizon)
    if fwd_date is None:
        return None
    fwd_close = prices.get(ticker, {}).get(fwd_date)
    if fwd_close is None:
        return None
    return (fwd_close - anchor_close) / anchor_close


# ---------------------------------------------------------------------------
# Validation suite (§6 of feasibility memo)
# ---------------------------------------------------------------------------


def run_validations(
    rows: List[dict],
    prices_by_arch: Dict[str, Dict[str, Dict[str, float]]],
) -> List[str]:
    """
    Run all 6 feasibility memo §6 validation checks.
    Returns list of warning/error strings (empty = clean).
    """
    issues = []

    # 1. Price continuity: no > 50% single-day jump
    for arch_date, prices in prices_by_arch.items():
        for ticker, tprices in prices.items():
            sorted_dates = sorted(tprices.keys())
            for i in range(1, len(sorted_dates)):
                p0 = tprices[sorted_dates[i - 1]]
                p1 = tprices[sorted_dates[i]]
                if p0 > 0 and abs(p1 / p0 - 1) > 0.50:
                    issues.append(
                        f"CONTINUITY: {ticker} in arch {arch_date}: "
                        f"{sorted_dates[i-1]}={p0:.4f} -> {sorted_dates[i]}={p1:.4f} "
                        f"({(p1/p0-1)*100:.1f}%)"
                    )

    # 2. Coverage completeness: each snapshot has >= MIN_ANCHOR_COVERAGE non-null anchors
    by_snap: Dict[str, List[dict]] = {}
    for r in rows:
        by_snap.setdefault(r["snap_date"], []).append(r)
    for snap_date, snap_rows in by_snap.items():
        covered = sum(1 for r in snap_rows if r["anchor_close"] != "")
        if covered < MIN_ANCHOR_COVERAGE:
            issues.append(f"COVERAGE: {snap_date} has only {covered}/{len(snap_rows)} non-null anchors")

    # 3. XBI sanity: coverage check (all snap_dates should have XBI anchor)
    for r in rows:
        if r["ticker"] == "XBI_BENCHMARK" and r["anchor_close"] == "":
            issues.append(f"XBI_MISSING: no XBI anchor for {r['snap_date']}")

    # 4. ATXS exclusion confirmation
    for r in rows:
        if r["ticker"] == "ATXS" and r["snap_date"] > ATXS_LAST_TRADING_DATE:
            if r["anchor_close"] != "":
                issues.append(f"ATXS_NOT_EXCLUDED: {r['snap_date']} has non-null anchor for ATXS post-acquisition")

    # 5. Archive integrity already done per-archive via verify_archive_manifest()
    # (issues logged separately before this function is called)

    # 6. Era reconciliation: check 5 overlap dates around 2026-01-15
    # Compare pit_archives/2026-01-15 vs universe_prices.csv for same tickers
    era_overlap_path = REPO_ROOT / "data" / "universe_prices.csv"
    arch_check_date = "2026-01-15"
    if era_overlap_path.exists() and (ARCH_ROOT / arch_check_date).exists():
        era_prices: Dict[str, float] = {}
        with open(era_overlap_path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            for row in reader:
                if row.get("date") == arch_check_date:
                    for col in fieldnames:
                        if col != "date":
                            v = (row.get(col) or "").strip()
                            if v:
                                try:
                                    era_prices[col.upper()] = float(v)
                                except ValueError:
                                    pass
                    break
        arch_prices = prices_by_arch.get(arch_check_date, {})
        discrepancies = []
        for ticker, era_p in era_prices.items():
            arch_p = arch_prices.get(ticker, {}).get(arch_check_date)
            if arch_p is not None and era_p > 0:
                diff_pct = abs(arch_p / era_p - 1)
                if diff_pct > 0.02:  # > 2% discrepancy
                    discrepancies.append(f"{ticker}: era1={era_p:.4f} arch={arch_p:.4f} ({diff_pct*100:.1f}%)")
        if discrepancies:
            issues.append(
                f"ERA_RECONCILIATION: {len(discrepancies)} tickers have >2% price discrepancy "
                f"on {arch_check_date}: {discrepancies[:5]}"
            )
    else:
        issues.append(f"ERA_RECONCILIATION: skipped (universe_prices.csv or {arch_check_date} archive missing)")

    return issues


# ---------------------------------------------------------------------------
# Main assembly
# ---------------------------------------------------------------------------


def assemble(dry_run: bool, out_path: Path) -> int:
    print(f"Gap period: {GAP_START} to {GAP_END}")
    gap_dates = discover_gap_snapshots()
    print(f"Gap snapshots found: {len(gap_dates)}")

    output_rows: List[dict] = []
    prices_by_arch: Dict[str, Dict[str, Dict[str, float]]] = {}
    sorted_dates_by_arch: Dict[str, List[str]] = {}
    manifest_errors: List[str] = []

    archive_cache: Dict[str, Dict[str, Dict[str, float]]] = {}

    for snap_date in gap_dates:
        arch_date, is_fallback = resolve_archive(snap_date)

        # Load archive (cached)
        if arch_date not in archive_cache:
            print(f"  Loading archive {arch_date}...")
            err = verify_archive_manifest(arch_date)
            if err:
                manifest_errors.append(err)
                print(f"    MANIFEST WARNING: {err}")
            arch_prices = load_price_series(ARCH_ROOT / arch_date / "price_history.csv")
            archive_cache[arch_date] = arch_prices
            prices_by_arch[arch_date] = arch_prices
            sorted_dates_by_arch[arch_date] = sorted(set().union(*[set(v.keys()) for v in arch_prices.values()]))

        prices = archive_cache[arch_date]
        sorted_dates = sorted_dates_by_arch[arch_date]

        # Read top-30 rankings
        rankings_path = SNAP_ROOT / snap_date / "rankings.csv"
        top30: List[dict] = []
        with open(rankings_path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    rank = int(row.get("actionable_rank") or "9999")
                except ValueError:
                    continue
                if rank <= 30:
                    top30.append(
                        {
                            "ticker": (row.get("ticker") or "").strip().upper(),
                            "actionable_rank": rank,
                            "target_weight_pct": row.get("target_weight_pct", ""),
                        }
                    )

        # Compute XBI benchmark returns for this snapshot
        xbi_anchor_close, xbi_anchor_date = resolve_anchor("XBI", snap_date, prices, sorted_dates)
        xbi_returns: Dict[int, Optional[float]] = {}
        if xbi_anchor_close is not None:
            for h in HORIZONS:
                xbi_returns[h] = compute_return("XBI", xbi_anchor_date, h, prices, xbi_anchor_close, sorted_dates)
        else:
            for h in HORIZONS:
                xbi_returns[h] = None

        for stock in top30:
            ticker = stock["ticker"]

            # ATXS exclusion post-acquisition
            if ticker == "ATXS" and snap_date > ATXS_LAST_TRADING_DATE:
                row = {
                    "snap_date": snap_date,
                    "ticker": ticker,
                    "actionable_rank": stock["actionable_rank"],
                    "target_weight_pct": stock["target_weight_pct"],
                    "archive_date": arch_date,
                    "archive_fallback": str(is_fallback),
                    "anchor_date": "",
                    "anchor_close": "",
                    "atxs_excluded": "true",
                }
                for h in HORIZONS:
                    row[f"actual_return_{h}d"] = ""
                    row[f"xbi_return_{h}d"] = _fmt(xbi_returns[h])
                    row[f"excess_return_{h}d"] = ""
                row["forward_complete"] = "false"
                output_rows.append(row)
                continue

            anchor_close, anchor_date = resolve_anchor(ticker, snap_date, prices, sorted_dates)

            row = {
                "snap_date": snap_date,
                "ticker": ticker,
                "actionable_rank": stock["actionable_rank"],
                "target_weight_pct": stock["target_weight_pct"],
                "archive_date": arch_date,
                "archive_fallback": str(is_fallback),
                "anchor_date": anchor_date or "",
                "anchor_close": _fmt(anchor_close),
                "atxs_excluded": "false",
            }

            for h in HORIZONS:
                if anchor_close is not None and anchor_date is not None:
                    ret = compute_return(ticker, anchor_date, h, prices, anchor_close, sorted_dates)
                    xbi_ret = xbi_returns[h]
                    excess = (ret - xbi_ret) if (ret is not None and xbi_ret is not None) else None
                    row[f"actual_return_{h}d"] = _fmt(ret)
                    row[f"xbi_return_{h}d"] = _fmt(xbi_ret)
                    row[f"excess_return_{h}d"] = _fmt(excess)
                else:
                    row[f"actual_return_{h}d"] = ""
                    row[f"xbi_return_{h}d"] = _fmt(xbi_returns[h])
                    row[f"excess_return_{h}d"] = ""

            # forward_complete = True only if 60d return is available
            row["forward_complete"] = "true" if _fmt(row.get("actual_return_60d")) != "" else "false"
            output_rows.append(row)

    print(f"\nTotal output rows: {len(output_rows)}")

    # Run validations
    print("\nRunning validation suite (§6 of feasibility memo)...")
    issues = run_validations(output_rows, prices_by_arch)
    if manifest_errors:
        issues = manifest_errors + issues

    if issues:
        print(f"\n  VALIDATION ISSUES ({len(issues)}):")
        for issue in issues:
            print(f"    {issue}")
    else:
        print("  All validation checks PASSED")

    # Coverage summary
    snap_coverage: Dict[str, int] = {}
    for r in output_rows:
        sd = r["snap_date"]
        snap_coverage[sd] = snap_coverage.get(sd, 0) + (1 if r["anchor_close"] != "" else 0)

    low_coverage = [(sd, n) for sd, n in snap_coverage.items() if n < MIN_ANCHOR_COVERAGE]
    if low_coverage:
        print(f"\n  LOW COVERAGE SNAPSHOTS ({len(low_coverage)}):")
        for sd, n in sorted(low_coverage):
            print(f"    {sd}: {n} tickers with anchor")
    else:
        print(f"  All {len(snap_coverage)} snapshots have >= {MIN_ANCHOR_COVERAGE} non-null anchors")

    if dry_run:
        print("\n  DRY RUN — no output written")
        return 0 if not issues else 1

    # Write output
    fieldnames = [
        "snap_date",
        "ticker",
        "actionable_rank",
        "target_weight_pct",
        "archive_date",
        "archive_fallback",
        "anchor_date",
        "anchor_close",
        "atxs_excluded",
    ]
    for h in HORIZONS:
        fieldnames += [f"actual_return_{h}d", f"xbi_return_{h}d", f"excess_return_{h}d"]
    fieldnames.append("forward_complete")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:  # nosemgrep: sc-mcp-tool-writes-filesystem
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"\nOutput written to: {out_path}")
    print("\nGOVERNANCE: Output is QUARANTINED until operator reviews and explicitly accepts.")
    print("  Do not merge into _forward_returns_panel.csv without operator approval.")

    return 0 if not issues else 1


def _fmt(v) -> str:
    if v is None or v == "":
        return ""
    try:
        return f"{float(v):.6f}"
    except (TypeError, ValueError):
        return str(v)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="Validate only; do not write output")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output CSV path")
    args = parser.parse_args()

    rc = assemble(dry_run=args.dry_run, out_path=args.out)
    sys.exit(rc)


if __name__ == "__main__":
    main()
