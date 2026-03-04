#!/usr/bin/env python3
"""Clinical sort signal monotonicity diagnostic.

RESEARCH ONLY — not for production use.

For each snapshot date in a window, pairs eligible tickers' signal column
with their H-day forward returns, bins into quintiles, and reports:
  • Per-quintile mean return (averaged across all dates)
  • Q1 (highest signal) − Q5 (lowest signal) spread
  • Monotonicity verdict: PASS / WARN / FAIL
  • Mean IC of signal vs forward return (with t-stat)

Filters to drug_developer archetype, mid/late stage_bucket only (where the
clinical sort signal can be nonzero in production).  Optionally include all
eligible tickers for a full-universe view.

Modes:
  Single column (default):
    python3 scripts/research/clinical_monotonicity_report.py \\
        --date-from 2025-01-01 --date-to 2025-12-31

  All clinical sub-components (batch summary table):
    python3 scripts/research/clinical_monotonicity_report.py \\
        --date-from 2025-01-01 --date-to 2025-12-31 --batch

  Batch + readout-day stratification:
    python3 scripts/research/clinical_monotonicity_report.py \\
        --date-from 2025-01-01 --date-to 2025-12-31 --batch --readout-bins

  OOS:
    python3 scripts/research/clinical_monotonicity_report.py \\
        --date-from 2020-01-01 --date-to 2024-12-31 --batch
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Batch sweep configuration
# ---------------------------------------------------------------------------

# Ordered list of clinical columns to sweep in --batch mode.
# (column_name, short_label, flip_sign)
# flip_sign=True means multiply signal by -1 before computing IC/quintiles
# (useful for competitive_intensity_z where lower = better).
BATCH_COLUMNS: List[Tuple[str, str, bool]] = [
    ("clinical_score_z_tier",   "cz_tier*",        False),  # active signal
    ("clinical_score_z",        "cz",              False),
    ("clinical_score",          "clin_raw",        False),
    ("clinical_score_v2_z",     "cv2_z",           False),
    ("clinical_score_v2",       "cv2_raw",         False),
    ("readout_curve_score",     "readout_curve",   False),
    ("readout_density_90",      "density_90",      False),
    ("late_stage_readouts_180", "late_rdouts_180", False),
    ("execution_momentum",      "exec_mom",        False),
    ("design_quality_score",    "design_q",        False),
    ("endpoint_strength_score", "endpoint_str",    False),
    ("competitive_intensity_z", "comp_int_z",      True),   # flipped: low = less competitive = better
    ("clinical_alpha_z",        "clin_alpha_z",    False),
]

# (* = active production sort signal)

# Readout-day bins: (low_inclusive, high_inclusive_or_None, label)
READOUT_BINS: List[Tuple[Optional[int], Optional[int], str]] = [
    (None, None,  "all"),
    (0,   30,    "0-30d"),
    (31,  90,    "31-90d"),
    (91,  180,   "91-180d"),
    (181, None,  ">180d"),
]


# ---------------------------------------------------------------------------
# Price helpers
# ---------------------------------------------------------------------------

def load_prices(price_csv: Path) -> Dict[str, Dict[str, float]]:
    """Return {ticker: {date_str: close_price}}."""
    prices: Dict[str, Dict[str, float]] = {}
    with open(price_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ticker = row["ticker"]
            date = row["date"]
            close = row.get("close", "")
            if close and close not in ("", "None"):
                prices.setdefault(ticker, {})[date] = float(close)
    return prices


def sorted_trading_dates(prices: Dict[str, Dict[str, float]]) -> List[str]:
    """Sorted unique dates across all tickers."""
    dates: set = set()
    for d in prices.values():
        dates.update(d.keys())
    return sorted(dates)


def forward_return(
    ticker: str,
    snap_date: str,
    horizon: int,
    prices: Dict[str, Dict[str, float]],
    trading_dates: List[str],
) -> Optional[float]:
    """H-day forward return: (close[t+H] / close[t]) − 1."""
    ticker_prices = prices.get(ticker)
    if ticker_prices is None:
        return None
    anchor = ticker_prices.get(snap_date)
    if anchor is None or anchor <= 0:
        return None
    try:
        idx = trading_dates.index(snap_date)
    except ValueError:
        return None
    fwd_idx = idx + horizon
    if fwd_idx >= len(trading_dates):
        return None
    fwd_date = trading_dates[fwd_idx]
    fwd_price = ticker_prices.get(fwd_date)
    if fwd_price is None or fwd_price <= 0:
        return None
    return fwd_price / anchor - 1.0


# ---------------------------------------------------------------------------
# Per-date analysis
# ---------------------------------------------------------------------------

def _safe_float(v: str, default: float = 0.0) -> float:
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


def _in_readout_bin(
    r: Dict[str, str],
    low: Optional[int],
    high: Optional[int],
) -> bool:
    """Check if ticker's clinical_readout_days falls within [low, high]."""
    if low is None and high is None:
        return True
    val_str = r.get("clinical_readout_days", "")
    if not val_str or val_str in ("", "None"):
        return False
    try:
        val = float(val_str)
    except (ValueError, TypeError):
        return False
    if low is not None and val < low:
        return False
    if high is not None and val > high:
        return False
    return True


def analyze_snapshot(
    csv_path: Path,
    snap_date: str,
    horizon: int,
    prices: Dict[str, Dict[str, float]],
    trading_dates: List[str],
    all_eligible: bool = False,
    signal_col: str = "clinical_score_z_tier",
    flip_sign: bool = False,
    readout_bin: Tuple[Optional[int], Optional[int]] = (None, None),
) -> Optional[List[Tuple[float, float]]]:
    """Return list of (signal_value, forward_return) pairs for eligible tickers.

    Filters to drug_developer archetype, mid/late stage_bucket by default.
    With all_eligible=True, includes all eligible tickers with a non-missing signal.
    readout_bin=(low, high) optionally filters by clinical_readout_days.
    flip_sign=True negates the signal (for columns where lower = better).
    """
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    low, high = readout_bin
    pairs: List[Tuple[float, float]] = []
    for r in rows:
        if r.get("eligible") != "1":
            continue
        # Archetype / stage filter
        arch = r.get("archetype", "")
        stage = r.get("stage_bucket", "")
        if not all_eligible:
            if arch != "drug_developer":
                continue
            if stage not in ("mid", "late"):
                continue
        # Readout-day bin filter
        if not _in_readout_bin(r, low, high):
            continue
        # Signal
        sig_val = r.get(signal_col, "")
        if not sig_val or sig_val in ("", "None"):
            continue
        sig = _safe_float(sig_val)
        if flip_sign:
            sig = -sig
        # Forward return
        ticker = r.get("ticker", "")
        ret = forward_return(ticker, snap_date, horizon, prices, trading_dates)
        if ret is None:
            continue
        pairs.append((sig, ret))

    return pairs if pairs else None


# ---------------------------------------------------------------------------
# Quintile aggregation
# ---------------------------------------------------------------------------

def quintile_profile(
    all_pairs: List[Tuple[float, float]],
    n_bins: int = 5,
) -> List[Dict]:
    """Bin (signal, return) pairs into n_bins quintiles by signal.

    Returns list of dicts: {label, n, mean_return, q_rank (1=top signal)}.
    """
    if not all_pairs:
        return []
    sorted_pairs = sorted(all_pairs, key=lambda x: x[0], reverse=True)
    bin_size = len(sorted_pairs) // n_bins
    results = []
    for i in range(n_bins):
        start = i * bin_size
        end = (i + 1) * bin_size if i < n_bins - 1 else len(sorted_pairs)
        bucket = sorted_pairs[start:end]
        returns = [p[1] for p in bucket]
        results.append({
            "q_rank": i + 1,
            "label": f"Q{i+1} (top)" if i == 0 else (f"Q{i+1} (bot)" if i == n_bins - 1 else f"Q{i+1}"),
            "n": len(returns),
            "mean_return": sum(returns) / len(returns),
        })
    return results


def spearman_ic(pairs: List[Tuple[float, float]]) -> Optional[float]:
    """Spearman rank correlation between signal and return."""
    if len(pairs) < 3:
        return None
    n = len(pairs)

    def rank_list(vals):
        sorted_vals = sorted(range(n), key=lambda i: vals[i])
        ranks = [0.0] * n
        for rank, idx in enumerate(sorted_vals):
            ranks[idx] = rank + 1.0
        return ranks

    sigs = [p[0] for p in pairs]
    rets = [p[1] for p in pairs]
    rank_s = rank_list(sigs)
    rank_r = rank_list(rets)
    mean_s = sum(rank_s) / n
    mean_r = sum(rank_r) / n
    num = sum((rank_s[i] - mean_s) * (rank_r[i] - mean_r) for i in range(n))
    den_s = sum((rank_s[i] - mean_s) ** 2 for i in range(n)) ** 0.5
    den_r = sum((rank_r[i] - mean_r) ** 2 for i in range(n)) ** 0.5
    if den_s == 0 or den_r == 0:
        return 0.0
    return num / (den_s * den_r)


def monotonicity_verdict(qprofile: List[Dict], n_bins: int = 5) -> str:
    """PASS / WARN / FAIL based on mean returns across quintiles.

    PASS: monotone (each quintile return ≥ next), i.e. Q1 ≥ Q2 ≥ ... ≥ Q5
    WARN: mostly monotone (at most 1 inversion)
    FAIL: 2+ inversions or large U-shape
    """
    returns = [q["mean_return"] for q in qprofile]
    inversions = sum(1 for i in range(len(returns) - 1) if returns[i] < returns[i + 1])
    if inversions == 0:
        return "PASS"
    elif inversions == 1:
        return "WARN"
    else:
        return "FAIL"


# ---------------------------------------------------------------------------
# Core accumulation logic (shared by single and batch modes)
# ---------------------------------------------------------------------------

def accumulate_pairs(
    snap_dates: List[str],
    snapshot_root: Path,
    horizons: List[int],
    prices: Dict[str, Dict[str, float]],
    trading_dates: List[str],
    all_eligible: bool,
    signal_col: str,
    flip_sign: bool = False,
    readout_bin: Tuple[Optional[int], Optional[int]] = (None, None),
) -> Tuple[Dict[int, List[Tuple[float, float]]], Dict[int, List[float]], Dict[int, int]]:
    """Accumulate (signal, return) pairs and per-date ICs across all snapshot dates.

    Returns:
        all_pairs_by_horizon: {h: [(sig, ret), ...]}
        ic_by_date: {h: [ic_date1, ic_date2, ...]}
        n_dates_with_data: {h: int}
    """
    all_pairs: Dict[int, List[Tuple[float, float]]] = {h: [] for h in horizons}
    ic_by_date: Dict[int, List[float]] = {h: [] for h in horizons}
    n_dates: Dict[int, int] = {h: 0 for h in horizons}

    for snap_date in snap_dates:
        csv_path = snapshot_root / snap_date / "rankings.csv"
        for h in horizons:
            pairs = analyze_snapshot(
                csv_path, snap_date, h, prices, trading_dates,
                all_eligible=all_eligible,
                signal_col=signal_col,
                flip_sign=flip_sign,
                readout_bin=readout_bin,
            )
            if pairs:
                all_pairs[h].extend(pairs)
                n_dates[h] += 1
                ic = spearman_ic(pairs)
                if ic is not None:
                    ic_by_date[h].append(ic)

    return all_pairs, ic_by_date, n_dates


def horizon_stats(
    pairs: List[Tuple[float, float]],
    ics: List[float],
    n_bins: int = 5,
) -> Dict:
    """Compute IC, t-stat, Q1/Q5 return, spread, verdict for one horizon."""
    if not pairs or not ics:
        return {
            "mean_ic": math.nan, "t_ic": math.nan,
            "q1_ret": math.nan, "q5_ret": math.nan,
            "spread": math.nan, "verdict": "N/A",
            "n_pairs": 0, "n_dates": len(ics),
        }
    mean_ic = sum(ics) / len(ics)
    var_ic = sum((v - mean_ic) ** 2 for v in ics) / max(len(ics) - 1, 1)
    t_ic = mean_ic / (var_ic ** 0.5 / len(ics) ** 0.5) if (var_ic > 0 and len(ics) > 1) else math.nan

    qprofile = quintile_profile(pairs, n_bins=n_bins)
    verdict = monotonicity_verdict(qprofile, n_bins=n_bins)
    q1_ret = qprofile[0]["mean_return"] if qprofile else math.nan
    q5_ret = qprofile[-1]["mean_return"] if qprofile else math.nan

    return {
        "mean_ic": mean_ic,
        "t_ic": t_ic,
        "q1_ret": q1_ret,
        "q5_ret": q5_ret,
        "spread": q1_ret - q5_ret,
        "verdict": verdict,
        "n_pairs": len(pairs),
        "n_dates": len(ics),
    }


# ---------------------------------------------------------------------------
# Batch mode: summary table
# ---------------------------------------------------------------------------

def _verdict_icon(v: str) -> str:
    return {"PASS": "✓", "WARN": "~", "FAIL": "✗", "N/A": "?"}.get(v, "?")


def run_batch(
    snap_dates: List[str],
    snapshot_root: Path,
    horizons: List[int],
    prices: Dict[str, Dict[str, float]],
    trading_dates: List[str],
    all_eligible: bool,
    n_bins: int,
    readout_bins_mode: bool,
) -> None:
    """Run all BATCH_COLUMNS × (optionally) READOUT_BINS and print summary table."""

    # Which readout bins to run
    bins_to_run = READOUT_BINS if readout_bins_mode else [READOUT_BINS[0]]  # just "all"

    # Header
    h_labels = [f"{h}d" for h in horizons]
    col_w = 18  # feature label width
    hz_w = 26   # per-horizon block width: "IC  t    Q1%  Q5%  V"

    def print_header(bin_label: str) -> None:
        sep_width = col_w + 1 + len(horizons) * (hz_w + 1)
        sep = "─" * sep_width
        print(f"\n{sep}")
        print(f"  Readout bin: {bin_label}   filter: {'all eligible' if all_eligible else 'dev mid/late'}")
        print(sep)
        hdr = f"{'Feature':<{col_w}}"
        for h in horizons:
            hdr += f"  {'──── ' + str(h) + 'd ─────────────────':<{hz_w}}"
        print(hdr)
        sub = " " * col_w
        for _ in horizons:
            sub += f"  {'IC':>6}  {'t':>5}  {'Q1%':>5} {'Q5%':>5} {'V':<4}"
        print(sub)
        print(sep)

    for bin_low, bin_high, bin_label in bins_to_run:
        print_header(bin_label)
        readout_bin = (bin_low, bin_high)

        for col_name, short_label, flip_sign in BATCH_COLUMNS:
            all_pairs, ic_by_date, n_dates = accumulate_pairs(
                snap_dates, snapshot_root, horizons, prices, trading_dates,
                all_eligible, col_name, flip_sign=flip_sign,
                readout_bin=readout_bin,
            )

            label = short_label + (" [↑]" if flip_sign else "")
            row = f"{label:<{col_w}}"
            for h in horizons:
                stats = horizon_stats(all_pairs[h], ic_by_date[h], n_bins=n_bins)
                if math.isnan(stats["mean_ic"]):
                    row += f"  {'no data':<{hz_w}}"
                else:
                    v_icon = _verdict_icon(stats["verdict"])
                    row += (
                        f"  {stats['mean_ic']:>+6.4f}"
                        f"  {stats['t_ic']:>+5.2f}"
                        f"  {stats['q1_ret']:>+5.1%}"
                        f" {stats['q5_ret']:>+5.1%}"
                        f" {v_icon:<4}"
                    )
            print(row)

        print()


# ---------------------------------------------------------------------------
# Single-column detailed mode
# ---------------------------------------------------------------------------

def run_single(
    snap_dates: List[str],
    snapshot_root: Path,
    horizons: List[int],
    prices: Dict[str, Dict[str, float]],
    trading_dates: List[str],
    all_eligible: bool,
    signal_col: str,
    n_bins: int,
) -> None:
    """Original detailed output for a single signal column."""
    all_pairs_by_horizon, ic_by_date, n_dates_with_data = accumulate_pairs(
        snap_dates, snapshot_root, horizons, prices, trading_dates,
        all_eligible, signal_col,
    )

    # Coverage stats
    print("Dates with priced data:")
    for h in horizons:
        n = n_dates_with_data[h]
        ics = ic_by_date[h]
        mean_ic = sum(ics) / len(ics) if ics else float("nan")
        print(f"  {h:3d}d: {n}/{len(snap_dates)} dates  mean_IC={mean_ic:+.4f}")

    print()

    for h in horizons:
        pairs = all_pairs_by_horizon[h]
        if not pairs:
            print(f"\n=== {h}d: NO DATA ===")
            continue

        qprofile = quintile_profile(pairs, n_bins=n_bins)
        verdict = monotonicity_verdict(qprofile)
        q1_ret = qprofile[0]["mean_return"] if qprofile else float("nan")
        q5_ret = qprofile[-1]["mean_return"] if qprofile else float("nan")
        spread = q1_ret - q5_ret

        ics = ic_by_date[h]
        mean_ic = sum(ics) / len(ics) if ics else float("nan")
        t_ic = (mean_ic / ((sum((v - mean_ic)**2 for v in ics) / max(len(ics)-1, 1))**0.5 / len(ics)**0.5)
                if len(ics) > 1 else float("nan"))

        print(f"=== {h}d horizon  (n_pairs={len(pairs)}, n_dates={n_dates_with_data[h]}) ===")
        print(f"    mean IC = {mean_ic:+.4f}  t = {t_ic:+.2f}   Q1−Q5 spread = {spread:+.2f}%*100")
        print(f"    Monotonicity: {verdict}")
        print(f"    {'Quintile':<12} {'N':>6} {'Mean Return':>12}")
        print(f"    {'-'*32}")
        for q in qprofile:
            bar = "█" * max(0, int(q["mean_return"] * 200 + 10))
            print(f"    {q['label']:<12} {q['n']:>6} {q['mean_return']:>+11.1%}  {bar}")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clinical sort signal monotonicity diagnostic",
    )
    parser.add_argument("--snapshot-root", type=Path,
                        default=PROJECT_ROOT / "data" / "snapshots")
    parser.add_argument("--price-csv", type=Path,
                        default=PROJECT_ROOT / "production_data" / "price_history.csv")
    parser.add_argument("--date-from", type=str, default=None)
    parser.add_argument("--date-to", type=str, default=None)
    parser.add_argument("--horizons", type=str, default="63,84,126",
                        help="Comma-separated horizons in trading days")
    parser.add_argument("--n-bins", type=int, default=5,
                        help="Number of quintile bins (default 5)")
    parser.add_argument("--signal-col", type=str, default="clinical_score_z_tier",
                        help="Column to use as signal in single-column mode")
    parser.add_argument("--all-eligible", action="store_true", default=False,
                        help="Include all eligible tickers (not just dev mid/late)")
    parser.add_argument("--batch", action="store_true", default=False,
                        help="Sweep all predefined clinical columns (summary table)")
    parser.add_argument("--readout-bins", action="store_true", default=False,
                        help="Also stratify by clinical_readout_days bins (requires --batch)")
    args = parser.parse_args()

    horizons = [int(h.strip()) for h in args.horizons.split(",")]

    # Discover snapshot dates
    snap_dates = sorted([
        p.name for p in args.snapshot_root.iterdir()
        if p.is_dir() and len(p.name) == 10 and p.name[4] == "-"
        and (p / "rankings.csv").exists()
    ])
    if args.date_from:
        snap_dates = [d for d in snap_dates if d >= args.date_from]
    if args.date_to:
        snap_dates = [d for d in snap_dates if d <= args.date_to]

    mode = "batch" if args.batch else "single"
    print(f"Snapshots: {len(snap_dates)} dates"
          f"  ({snap_dates[0] if snap_dates else 'none'} – {snap_dates[-1] if snap_dates else 'none'})")
    print(f"Mode:      {mode}")
    if not args.batch:
        print(f"Signal:    {args.signal_col}")
    print(f"Filter:    {'all eligible' if args.all_eligible else 'drug_developer, mid/late stage only'}")
    print(f"Horizons:  {horizons}")
    print(f"Bins:      {args.n_bins}")
    print()

    print("Loading price history...", end=" ", flush=True)
    prices = load_prices(args.price_csv)
    trading_dates = sorted_trading_dates(prices)
    print(f"{len(trading_dates)} trading dates, {len(prices)} tickers")

    if args.batch:
        run_batch(
            snap_dates, args.snapshot_root, horizons,
            prices, trading_dates, args.all_eligible,
            args.n_bins, args.readout_bins,
        )
    else:
        run_single(
            snap_dates, args.snapshot_root, horizons,
            prices, trading_dates, args.all_eligible,
            args.signal_col, args.n_bins,
        )


if __name__ == "__main__":
    main()
