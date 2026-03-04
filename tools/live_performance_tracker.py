"""
Live Performance Tracker — tools/live_performance_tracker.py

Reads PIT price caches + snapshot rankings, computes portfolio metrics per date,
and appends rows to output/live_performance.csv (write-once: never overwrites
existing rows).

Also writes output/live_performance_summary.json with rolling stats.

Schema: live_performance_row.v1
Columns: date, horizon, n_held, anchor_close_mean, forward_close_mean,
         gross_return, net_return, ic, xbi_return, excess_return, turnover,
         ruleset_id, notes

Usage:
    python3 tools/live_performance_tracker.py
    python3 tools/live_performance_tracker.py --as-of-date 2026-03-05
    python3 tools/live_performance_tracker.py --dry-run
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Project root on path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval_forward_returns import (
    spearman_ic,
    top_k_portfolio_return,
    compute_turnover,
    net_return,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PRICE_PIT_BASE = PROJECT_ROOT / "data" / "caches" / "price_pit" / "PIT"
SNAPSHOTS_ROOT = PROJECT_ROOT / "data" / "snapshots"
PRICE_HISTORY_CSV = PROJECT_ROOT / "price_history.csv"
OUTPUT_CSV = PROJECT_ROOT / "output" / "live_performance.csv"
OUTPUT_SUMMARY = PROJECT_ROOT / "output" / "live_performance_summary.json"

HORIZON = 20          # trading days forward (h20)
TOP_K = 20
COST_BPS = 30.0
SCHEMA_VERSION = "live_performance_row.v1"

CSV_FIELDS = [
    "schema_version",
    "date",
    "horizon",
    "n_held",
    "anchor_close_mean",
    "forward_close_mean",
    "gross_return",
    "net_return",
    "ic",
    "xbi_return",
    "excess_return",
    "turnover",
    "ruleset_id",
    "notes",
]

# Approximately 4 and 13 weeks of trading days
ROLLING_4W_N = 20
ROLLING_13W_N = 65


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_rankings(snapshot_dir: Path) -> List[Dict]:
    """Load rankings.csv from a snapshot directory, sorted by actionable_rank asc."""
    rankings_csv = snapshot_dir / "rankings.csv"
    if not rankings_csv.exists():
        return []
    rows = []
    with rankings_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    # Sort by actionable_rank (ascending integer)
    try:
        rows.sort(key=lambda r: int(r.get("actionable_rank", 9999)))
    except (ValueError, TypeError):
        pass
    return rows


def _load_pit_prices(pit_dir: Path) -> Dict[str, Dict]:
    """Load prices.csv from a PIT date directory. Returns {ticker: {col: val}}."""
    prices_csv = pit_dir / "prices.csv"
    if not prices_csv.exists():
        return {}
    result = {}
    with prices_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            result[row["ticker"].upper()] = row
    return result


def _load_pit_index(pit_dir: Path) -> Dict:
    """Load index.json from a PIT date directory."""
    index_json = pit_dir / "index.json"
    if not index_json.exists():
        return {}
    with index_json.open() as f:
        return json.load(f)


def _load_xbi_prices() -> Dict[str, float]:
    """Load XBI close prices from price_history.csv. Returns {date_str: close}."""
    if not PRICE_HISTORY_CSV.exists():
        return {}
    result = {}
    with PRICE_HISTORY_CSV.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("ticker", "").upper() == "XBI":
                try:
                    result[row["date"]] = float(row["close"])
                except (ValueError, KeyError):
                    pass
    return result


def _find_split_warnings(pit_dir: Path, horizon: int) -> set:
    """Return set of tickers with split warnings for this horizon."""
    idx = _load_pit_index(pit_dir)
    warnings = idx.get("split_warnings", [])
    result = set()
    for w in warnings:
        if isinstance(w, dict):
            # Check if horizon is relevant
            h_key = f"h{horizon}"
            if w.get(h_key) or w.get("horizon") == horizon:
                result.add(w.get("ticker", "").upper())
        elif isinstance(w, str):
            result.add(w.upper())
    return result


# ---------------------------------------------------------------------------
# Forward return computation
# ---------------------------------------------------------------------------

def _compute_fwd_returns(
    pit_prices: Dict[str, Dict],
    split_tickers: set,
    horizon: int,
) -> Dict[str, float]:
    """Compute per-ticker forward returns from PIT prices. Skip split-warning tickers."""
    h_close_col = f"h{horizon}_close"
    result = {}
    for ticker, row in pit_prices.items():
        if ticker in split_tickers:
            continue
        try:
            anchor = float(row["anchor_close"])
            fwd = float(row[h_close_col])
            if anchor > 0:
                result[ticker] = fwd / anchor - 1.0
        except (ValueError, TypeError, KeyError):
            pass
    return result


def _get_xbi_forward_return(
    xbi_prices: Dict[str, float],
    anchor_date: str,
    horizon: int,
) -> Optional[float]:
    """Compute XBI forward return at anchor_date for given horizon (trading days).

    Walks forward up to horizon+10 calendar days to find h trading days.
    Returns None if insufficient data.
    """
    if not xbi_prices:
        return None
    sorted_dates = sorted(xbi_prices.keys())
    if anchor_date not in xbi_prices:
        return None

    # Find all trading dates after anchor
    dates_after = [d for d in sorted_dates if d > anchor_date]
    if len(dates_after) < horizon:
        return None

    fwd_date = dates_after[horizon - 1]
    anchor_price = xbi_prices[anchor_date]
    fwd_price = xbi_prices[fwd_date]
    if anchor_price <= 0:
        return None
    return fwd_price / anchor_price - 1.0


# ---------------------------------------------------------------------------
# Snapshot metadata
# ---------------------------------------------------------------------------

def _load_ruleset_id(snapshot_dir: Path) -> str:
    """Extract ruleset_id from snapshot metadata.json."""
    meta = snapshot_dir / "metadata.json"
    if meta.exists():
        try:
            with meta.open() as f:
                data = json.load(f)
            return data.get("ruleset_id", data.get("decision_engine_ruleset_id", ""))
        except Exception:
            pass
    return ""


def _discover_pit_dates_with_horizon(horizon: int) -> List[str]:
    """Find PIT cache dates where horizon is filled."""
    if not PRICE_PIT_BASE.exists():
        return []
    result = []
    for date_dir in sorted(PRICE_PIT_BASE.iterdir()):
        if not date_dir.is_dir():
            continue
        idx = _load_pit_index(date_dir)
        filled = idx.get("horizons_filled", [])
        if horizon in filled:
            result.append(date_dir.name)
    return sorted(result)


def _load_existing_dates(horizon: int) -> set:
    """Load dates already in live_performance.csv for this horizon."""
    if not OUTPUT_CSV.exists():
        return set()
    existing = set()
    with OUTPUT_CSV.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row.get("horizon", 0)) == horizon:
                existing.add(row["date"])
    return existing


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_row(
    snap_date: str,
    horizon: int,
    xbi_prices: Dict[str, float],
    prev_top_k: Optional[List[str]],
) -> Optional[Dict]:
    """Compute one performance row for snap_date. Returns None if data incomplete."""
    pit_dir = PRICE_PIT_BASE / snap_date
    if not pit_dir.exists():
        return None

    idx = _load_pit_index(pit_dir)
    if horizon not in idx.get("horizons_filled", []):
        return None

    anchor_date = idx.get("anchor_date", snap_date)
    split_tickers = _find_split_warnings(pit_dir, horizon)

    pit_prices = _load_pit_prices(pit_dir)
    if not pit_prices:
        return None

    # Forward returns
    fwd_rets = _compute_fwd_returns(pit_prices, split_tickers, horizon)
    if not fwd_rets:
        return None

    # Load rankings
    snap_dir = SNAPSHOTS_ROOT / snap_date
    rankings = _load_rankings(snap_dir)
    if not rankings:
        return None

    # Eligible tickers in rank order
    tickers_ranked = [
        r["ticker"].upper()
        for r in rankings
        if r.get("eligible", "").strip() == "1"
    ]

    # IC: signal = negative actionable_rank (lower rank = better signal)
    signal = []
    ret_vals = []
    for r in rankings:
        tk = r["ticker"].upper()
        if tk in split_tickers or tk not in fwd_rets:
            continue
        try:
            rank = float(r["actionable_rank"])
            signal.append(-rank)  # negate: higher rank → lower value → higher IC
            ret_vals.append(fwd_rets[tk])
        except (ValueError, TypeError):
            pass

    ic = spearman_ic(signal, ret_vals) if len(signal) >= 3 else None

    # Top-K portfolio return
    gross, n_held = top_k_portfolio_return(tickers_ranked, fwd_rets, TOP_K)
    if gross is None:
        return None

    # Mean anchor / forward prices for top-K
    top_k_tickers = [t for t in tickers_ranked[:TOP_K] if t in pit_prices]
    anchor_closes = []
    fwd_closes = []
    h_close_col = f"h{horizon}_close"
    for tk in top_k_tickers:
        try:
            anchor_closes.append(float(pit_prices[tk]["anchor_close"]))
            fwd_closes.append(float(pit_prices[tk][h_close_col]))
        except (ValueError, TypeError, KeyError):
            pass
    anchor_close_mean = statistics.mean(anchor_closes) if anchor_closes else None
    forward_close_mean = statistics.mean(fwd_closes) if fwd_closes else None

    # Turnover
    curr_top_k = list(tickers_ranked[:TOP_K])
    turnover = compute_turnover(prev_top_k or [], curr_top_k) if prev_top_k is not None else 0.0

    # Net return
    net = net_return(gross, turnover, COST_BPS)

    # XBI benchmark
    xbi_ret = _get_xbi_forward_return(xbi_prices, anchor_date, horizon)
    excess = (gross - xbi_ret) if (xbi_ret is not None) else None

    ruleset_id = _load_ruleset_id(snap_dir)

    return {
        "schema_version": SCHEMA_VERSION,
        "date": snap_date,
        "horizon": horizon,
        "n_held": n_held,
        "anchor_close_mean": round(anchor_close_mean, 4) if anchor_close_mean is not None else "",
        "forward_close_mean": round(forward_close_mean, 4) if forward_close_mean is not None else "",
        "gross_return": round(gross, 6),
        "net_return": round(net, 6),
        "ic": round(ic, 6) if ic is not None else "",
        "xbi_return": round(xbi_ret, 6) if xbi_ret is not None else "",
        "excess_return": round(excess, 6) if excess is not None else "",
        "turnover": round(turnover, 6),
        "ruleset_id": ruleset_id,
        "notes": "",
        # keep for turnover chain
        "_top_k_tickers": curr_top_k,
    }


# ---------------------------------------------------------------------------
# Rolling summary
# ---------------------------------------------------------------------------

def _mean_safe(vals: List[float]) -> Optional[float]:
    filtered = [v for v in vals if v is not None and not math.isnan(v)]
    return statistics.mean(filtered) if filtered else None


def build_summary(rows: List[Dict]) -> Dict:
    """Build rolling summary from all rows (sorted ascending by date)."""
    rows_sorted = sorted(rows, key=lambda r: r["date"])

    def _window(n: int) -> List[Dict]:
        return rows_sorted[-n:] if len(rows_sorted) >= n else rows_sorted

    def _stats(window: List[Dict]) -> Dict:
        gross_list = [float(r["gross_return"]) for r in window if r.get("gross_return") not in ("", None)]
        net_list = [float(r["net_return"]) for r in window if r.get("net_return") not in ("", None)]
        ic_list = [float(r["ic"]) for r in window if r.get("ic") not in ("", None)]
        excess_list = [float(r["excess_return"]) for r in window if r.get("excess_return") not in ("", None)]
        return {
            "n_dates": len(window),
            "mean_gross_return": round(_mean_safe(gross_list), 6) if gross_list else None,
            "mean_net_return": round(_mean_safe(net_list), 6) if net_list else None,
            "mean_ic": round(_mean_safe(ic_list), 6) if ic_list else None,
            "mean_excess_return": round(_mean_safe(excess_list), 6) if excess_list else None,
        }

    return {
        "schema_version": "live_performance_summary.v1",
        "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_dates": len(rows_sorted),
        "horizon": HORIZON,
        "first_date": rows_sorted[0]["date"] if rows_sorted else None,
        "last_date": rows_sorted[-1]["date"] if rows_sorted else None,
        "last_4w": _stats(_window(ROLLING_4W_N)),
        "last_13w": _stats(_window(ROLLING_13W_N)),
        "inception": _stats(rows_sorted),
    }


# ---------------------------------------------------------------------------
# Write-once CSV helpers
# ---------------------------------------------------------------------------

def _load_all_rows() -> List[Dict]:
    """Load all rows from existing live_performance.csv."""
    if not OUTPUT_CSV.exists():
        return []
    rows = []
    with OUTPUT_CSV.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


def _write_rows(rows: List[Dict]) -> None:
    """Write rows to OUTPUT_CSV (creates file with header)."""
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_tracker(
    *,
    dry_run: bool = False,
    as_of_date: Optional[str] = None,
    horizon: int = HORIZON,
) -> Dict:
    """Run the tracker and return summary dict."""
    xbi_prices = _load_xbi_prices()

    # Discover PIT dates with filled horizon
    filled_dates = _discover_pit_dates_with_horizon(horizon)
    if as_of_date:
        filled_dates = [d for d in filled_dates if d <= as_of_date]

    if not filled_dates:
        print(f"[tracker] No PIT dates with horizon={horizon} filled. Nothing to do.")
        return {}

    # Load existing rows (write-once)
    existing_rows = _load_all_rows()
    existing_dates = {
        r["date"] for r in existing_rows
        if int(r.get("horizon", 0)) == horizon
    }

    new_dates = [d for d in filled_dates if d not in existing_dates]
    if not new_dates:
        print(f"[tracker] All {len(filled_dates)} dates already computed. Nothing new.")
    else:
        print(f"[tracker] {len(new_dates)} new date(s) to process: {new_dates}")

    # Build ordered list of all known dates for turnover chain
    all_dates_sorted = sorted(set(filled_dates) | existing_dates)

    # Build prev_top_k map from existing rows
    prev_top_k_map: Dict[str, List[str]] = {}
    for row in existing_rows:
        if int(row.get("horizon", 0)) == horizon:
            # We don't store top_k list in CSV; turnover for new rows will be
            # computed relative to the immediately prior date's rankings
            pass

    new_rows = []
    prev_top_k: Optional[List[str]] = None

    # We need to process dates in order to chain turnover
    # Seed prev_top_k from the date just before the first new date
    if new_dates:
        first_new = min(new_dates)
        prior_dates = [d for d in all_dates_sorted if d < first_new]
        if prior_dates:
            prior_date = max(prior_dates)
            prior_snap = SNAPSHOTS_ROOT / prior_date
            prior_rankings = _load_rankings(prior_snap)
            if prior_rankings:
                prev_top_k = [
                    r["ticker"].upper()
                    for r in prior_rankings
                    if r.get("eligible", "").strip() == "1"
                ][:TOP_K]

    for snap_date in sorted(new_dates):
        row = compute_row(snap_date, horizon, xbi_prices, prev_top_k)
        if row is None:
            print(f"[tracker] Skipping {snap_date}: incomplete data")
            continue

        prev_top_k = row.pop("_top_k_tickers", None)

        # Annotate if no prior row (fresh start)
        if not existing_rows and not new_rows:
            row["notes"] = "fresh_start"

        new_rows.append(row)
        print(f"[tracker] {snap_date}: gross={row['gross_return']:.4f} "
              f"net={row['net_return']:.4f} IC={row['ic']} "
              f"XBI_excess={row['excess_return']} turnover={row['turnover']:.3f}")

    if not dry_run and new_rows:
        all_rows = existing_rows + new_rows
        _write_rows(all_rows)
        print(f"[tracker] Wrote {len(new_rows)} new row(s) to {OUTPUT_CSV}")

    # Build and write summary
    all_display_rows = existing_rows + new_rows
    summary = build_summary(all_display_rows)
    if not dry_run:
        OUTPUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
        with OUTPUT_SUMMARY.open("w") as f:
            json.dump(summary, f, indent=2)
        print(f"[tracker] Summary written to {OUTPUT_SUMMARY}")

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live performance tracker — computes portfolio metrics from PIT price caches."
    )
    parser.add_argument(
        "--as-of-date",
        default=None,
        metavar="YYYY-MM-DD",
        help="Only process dates up to this date",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute but do not write output files",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=HORIZON,
        help=f"Forward return horizon in trading days (default: {HORIZON})",
    )
    args = parser.parse_args()

    summary = run_tracker(
        dry_run=args.dry_run,
        as_of_date=args.as_of_date,
        horizon=args.horizon,
    )
    if summary:
        print("\n=== Summary ===")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
