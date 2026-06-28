#!/usr/bin/env python3
"""
fill_forward_returns.py — Fill pending forward returns in the DEM Top-30 EW validation ledger.

For each capture where 1d/5d/20d returns are not yet filled, checks if the
forward endpoint is now available in universe_prices.csv and indices_prices.csv.
Appends fills to artifacts/forward_validation/fills.jsonl.
Regenerates truth cards for affected dates.

Usage:
    python3 tools/fill_forward_returns.py
    python3 tools/fill_forward_returns.py --date 2026-06-26   # fill one date
    python3 tools/fill_forward_returns.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import date as ddate
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
# Long-format (date, ticker, close — split-adjusted). Includes XBI.
PRICE_HISTORY = REPO_ROOT / "production_data" / "price_history.csv"
ARTIFACTS = REPO_ROOT / "artifacts" / "forward_validation"
CAPTURES_LEDGER = ARTIFACTS / "captures.jsonl"
FILLS_LEDGER = ARTIFACTS / "fills.jsonl"

HORIZONS = {"1d": 1, "5d": 5, "20d": 20}


# ---------------------------------------------------------------------------
# Price loaders — production_data/price_history.csv (long: date, ticker, close)
# ---------------------------------------------------------------------------

_PH_BY_DATE: dict[str, dict[str, float]] = {}
_PH_DATES_SORTED: list[str] = []


def _load_price_history() -> None:
    global _PH_DATES_SORTED
    if _PH_BY_DATE:
        return
    if not PRICE_HISTORY.exists():
        return
    with open(PRICE_HISTORY, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            d = row.get("date", "")
            t = row.get("ticker", "")
            try:
                c = float(row["close"])
            except (KeyError, ValueError, TypeError):
                continue
            if d not in _PH_BY_DATE:
                _PH_BY_DATE[d] = {}
            _PH_BY_DATE[d][t] = c
    _PH_DATES_SORTED = sorted(_PH_BY_DATE)


def _all_dates() -> list[str]:
    _load_price_history()
    return _PH_DATES_SORTED


def get_universe_prices(date: str) -> dict[str, float]:
    _load_price_history()
    return _PH_BY_DATE.get(date, {})


def get_xbi_price(date: str) -> float | None:
    _load_price_history()
    return _PH_BY_DATE.get(date, {}).get("XBI")


def _load_all_universe_dates() -> list[str]:
    return _all_dates()


def _load_all_xbi_dates() -> list[str]:
    return _all_dates()


def nth_trading_day_after(start_date: str, n: int, available: list[str]) -> str | None:
    """Return the nth trading date strictly after start_date."""
    count = 0
    for d in available:
        if d > start_date:
            count += 1
            if count == n:
                return d
    return None


# ---------------------------------------------------------------------------
# Fills ledger
# ---------------------------------------------------------------------------


def load_fills() -> dict[str, dict]:
    """Return dict keyed by capture_date of latest fill record."""
    fills: dict[str, dict] = {}
    if not FILLS_LEDGER.exists():
        return fills
    with open(FILLS_LEDGER) as f:
        for line in f:
            if line.strip():
                try:
                    rec = json.loads(line)
                    fills[rec["capture_date"]] = rec
                except (json.JSONDecodeError, KeyError):
                    pass
    return fills


def append_fill(record: dict, dry_run: bool = False) -> None:
    if dry_run:
        print(f"  [dry-run] would append fill: {record['capture_date']}")
        return
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    with open(FILLS_LEDGER, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, separators=(",", ":")) + "\n")


# ---------------------------------------------------------------------------
# Return computation
# ---------------------------------------------------------------------------


def compute_basket_return(tickers: list[str], start_date: str, end_date: str) -> float | None:
    """Equal-weight basket return from start to end date (adj_close)."""
    start_prices = get_universe_prices(start_date)
    end_prices = get_universe_prices(end_date)
    if not start_prices or not end_prices:
        return None
    returns = []
    for t in tickers:
        p0 = start_prices.get(t)
        p1 = end_prices.get(t)
        if p0 and p1 and p0 > 0:
            returns.append((p1 - p0) / p0)
    if not returns:
        return None
    return sum(returns) / len(returns)


def compute_control_returns(
    control_tickers: list[str],
    start_date: str,
    end_date: str,
) -> float | None:
    return compute_basket_return(control_tickers, start_date, end_date)


# ---------------------------------------------------------------------------
# Rank-depth shadow cohorts
# ---------------------------------------------------------------------------
# VALIDATION_INFRASTRUCTURE / RANK_DEPTH_SHADOW_TRACKING / NO_MODEL_CHANGE.
COHORT_RANGES = {
    "top30": (1, 30),
    "rank31_60": (31, 60),
    "top60": (1, 60),
}


def compute_cohort_returns(
    cohorts: dict[str, list[str]],
    effective_start: str,
    universe_dates: list[str],
    xbi_start: float | None,
) -> dict:
    """Per-cohort equal-weight forward returns vs XBI at each horizon.

    Returns {cohort: {horizon: {rank_min, rank_max, n_names, ew_return,
    xbi_return, xs_return, end_date}}}. Return fields are None until the
    forward endpoint exists in price_history.csv. Annotation only.
    """
    out: dict[str, dict] = {}
    for cohort, tickers in cohorts.items():
        rank_min, rank_max = COHORT_RANGES.get(cohort, (None, None))
        out[cohort] = {}
        for label, n_days in HORIZONS.items():
            end_date = nth_trading_day_after(effective_start, n_days, universe_dates)
            rec: dict[str, Any] = {
                "rank_min": rank_min,
                "rank_max": rank_max,
                "n_names": len(tickers),
                "ew_return": None,
                "xbi_return": None,
                "xs_return": None,
                "end_date": end_date,
            }
            if end_date is not None and xbi_start:
                ew = compute_basket_return(tickers, effective_start, end_date)
                xbi_end = get_xbi_price(end_date)
                if ew is not None and xbi_end is not None and xbi_start:
                    xbi_ret = (xbi_end - xbi_start) / xbi_start
                    rec["ew_return"] = round(ew, 6)
                    rec["xbi_return"] = round(xbi_ret, 6)
                    rec["xs_return"] = round(ew - xbi_ret, 6)
            out[cohort][label] = rec
    return out


def compute_bootstrap_percentile(
    top30_xs: float,
    bootstraps: list[list[str]],
    xbi_return: float,
    start_date: str,
    end_date: str,
) -> float | None:
    """Fraction of bootstrap samples that the Top-30 EW beats."""
    bootstrap_xs = []
    for sample in bootstraps:
        br = compute_basket_return(sample, start_date, end_date)
        if br is not None:
            bootstrap_xs.append(br - xbi_return)
    if not bootstrap_xs:
        return None
    beats = sum(1 for x in bootstrap_xs if top30_xs > x)
    return beats / len(bootstrap_xs)


# ---------------------------------------------------------------------------
# Truth card regeneration (import from capture script)
# ---------------------------------------------------------------------------


def regenerate_truth_card(capture: dict, fill: dict) -> None:
    from tools.run_forward_validation import generate_truth_card

    date = capture["date"]
    card_dir = ARTIFACTS / date
    card_dir.mkdir(parents=True, exist_ok=True)
    card = generate_truth_card(capture, [fill])
    (card_dir / "TRUTH_CARD.md").write_text(card, encoding="utf-8")


# ---------------------------------------------------------------------------
# Fill one capture
# ---------------------------------------------------------------------------


def fill_capture(capture: dict, existing_fill: dict | None, dry_run: bool) -> dict | None:
    date = capture["date"]
    effective_start = capture.get("effective_price_date")
    if not effective_start:
        return None

    tickers = [t["ticker"] for t in capture["top30"]]
    bottom30_tickers = capture.get("adversarial", {}).get("bottom30_tickers", [])
    bootstraps = capture.get("adversarial", {}).get("bootstrap_samples", [])

    universe_dates = _load_all_universe_dates()  # same as xbi dates — same price file

    fill: dict[str, Any] = {
        "capture_date": date,
        "filled_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "effective_start_date": effective_start,
    }

    changed = False

    for label, n_days in HORIZONS.items():
        # Skip if already filled
        if existing_fill and existing_fill.get(f"basket_{label}") is not None:
            fill[f"basket_{label}"] = existing_fill[f"basket_{label}"]
            fill[f"xbi_{label}"] = existing_fill[f"xbi_{label}"]
            fill[f"xs_{label}"] = existing_fill[f"xs_{label}"]
            fill[f"end_date_{label}"] = existing_fill.get(f"end_date_{label}")
            fill[f"control_bottom30_{label}"] = existing_fill.get(f"control_bottom30_{label}")
            fill[f"control_bootstrap_pct_{label}"] = existing_fill.get(f"control_bootstrap_pct_{label}")
            continue

        # Basket and XBI both come from price_history.csv — same date = guaranteed parity.
        end_date = nth_trading_day_after(effective_start, n_days, universe_dates)

        if end_date is None:
            fill[f"basket_{label}"] = None
            fill[f"xbi_{label}"] = None
            fill[f"xs_{label}"] = None
            fill[f"end_date_{label}"] = None
            fill[f"control_bottom30_{label}"] = None
            fill[f"control_bootstrap_pct_{label}"] = None
            continue

        basket_ret = compute_basket_return(tickers, effective_start, end_date)
        xbi_start = capture.get("xbi_price_at_capture") or get_xbi_price(effective_start)
        xbi_end_price = get_xbi_price(end_date)

        if basket_ret is None or xbi_start is None or xbi_end_price is None or xbi_start == 0:
            fill[f"basket_{label}"] = None
            fill[f"xbi_{label}"] = None
            fill[f"xs_{label}"] = None
            fill[f"end_date_{label}"] = end_date
            fill[f"control_bottom30_{label}"] = None
            fill[f"control_bootstrap_pct_{label}"] = None
            continue

        xbi_ret = (xbi_end_price - xbi_start) / xbi_start
        xs_ret = basket_ret - xbi_ret

        # Adversarial controls
        b30_ret = compute_control_returns(bottom30_tickers, effective_start, end_date) if bottom30_tickers else None
        boot_pct = None
        if bootstraps and len(bootstraps) >= 100:
            boot_pct = compute_bootstrap_percentile(xs_ret, bootstraps[:200], xbi_ret, effective_start, end_date)

        fill[f"basket_{label}"] = round(basket_ret, 6)
        fill[f"xbi_{label}"] = round(xbi_ret, 6)
        fill[f"xs_{label}"] = round(xs_ret, 6)
        fill[f"end_date_{label}"] = end_date
        fill[f"control_bottom30_{label}"] = round(b30_ret, 6) if b30_ret is not None else None
        fill[f"control_bootstrap_pct_{label}"] = round(boot_pct, 4) if boot_pct is not None else None
        changed = True

    # --- Rank-depth shadow cohorts (top30 / rank31_60 / top60) ---
    # Computed fresh each pass (cheap price lookups). Only flips `changed`
    # when the cohort returns actually differ from the prior fill, so a fully
    # matured capture is not re-appended every run.
    cohorts = capture.get("cohorts")
    if cohorts:
        xbi_start = capture.get("xbi_price_at_capture") or get_xbi_price(effective_start)
        cohort_returns = compute_cohort_returns(cohorts, effective_start, universe_dates, xbi_start)
        fill["cohorts"] = cohort_returns
        if not existing_fill or existing_fill.get("cohorts") != cohort_returns:
            changed = True

    if not changed and existing_fill is not None:
        return None  # Nothing new

    return fill


# ---------------------------------------------------------------------------
# Weekly non-overlapping summary
# ---------------------------------------------------------------------------


def week_key(date_str: str) -> str:
    d = ddate.fromisoformat(date_str)
    year, week, _ = d.isocalendar()
    return f"{year}-W{week:02d}"


def compute_weekly_stats(
    captures: list[dict],
    fills: dict[str, dict],
) -> dict:
    """One 5d result per calendar week (earliest capture in week with a completed 5d fill)."""
    by_week: dict[str, tuple[str, float, float]] = {}  # week → (date, xs, xbi_ret)
    for cap in sorted(captures, key=lambda c: c["date"]):
        date = cap["date"]
        fill = fills.get(date, {})
        xs = fill.get("xs_5d")
        xbi = fill.get("xbi_5d")
        if xs is None or xbi is None:
            continue
        wk = week_key(date)
        if wk not in by_week:
            by_week[wk] = (date, xs, xbi)

    rows = list(by_week.values())
    if not rows:
        return {"n": 0}

    xs_vals = [r[1] for r in rows]
    n = len(xs_vals)
    mean_xs = sum(xs_vals) / n
    variance = sum((x - mean_xs) ** 2 for x in xs_vals) / max(n - 1, 1)
    std_xs = variance**0.5
    t_stat = mean_xs / (std_xs / n**0.5) if std_xs > 0 else 0.0
    hit_rate = sum(1 for x in xs_vals if x > 0) / n

    return {
        "n": n,
        "mean_xs": round(mean_xs, 6),
        "std_xs": round(std_xs, 6),
        "t_stat": round(t_stat, 4),
        "hit_rate": round(hit_rate, 4),
        "cum_xs": round(sum(xs_vals), 4),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Fill forward returns for DEM Top-30 EW validation")
    parser.add_argument("--date", help="Fill only this capture date (YYYY-MM-DD)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not CAPTURES_LEDGER.exists():
        print("No captures ledger found. Run run_forward_validation.py first.")
        return 1

    # Load captures
    captures = []
    with open(CAPTURES_LEDGER) as f:
        for line in f:
            if line.strip():
                try:
                    captures.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    if args.date:
        captures = [c for c in captures if c["date"] == args.date]

    existing_fills = load_fills()
    new_fills = dict(existing_fills)
    filled_count = 0

    for capture in captures:
        date = capture["date"]
        existing_fill = existing_fills.get(date)
        fill = fill_capture(capture, existing_fill, dry_run=args.dry_run)
        if fill is None:
            continue

        append_fill(fill, dry_run=args.dry_run)
        new_fills[date] = fill
        filled_count += 1

        # Print 5d result if complete
        xs5 = fill.get("xs_5d")
        b5 = fill.get("basket_5d")
        x5 = fill.get("xbi_5d")
        if xs5 is not None:
            print(f"  {date}: basket={b5:+.2%} XBI={x5:+.2%} excess={xs5:+.2%}")
        else:
            print(f"  {date}: 5d still PENDING")

        # Regenerate truth card
        if not args.dry_run:
            try:
                regenerate_truth_card(capture, fill)
            except Exception as e:
                print(f"  Warning: truth card regeneration failed for {date}: {e}")

    print(f"\nFilled {filled_count} capture(s).")

    # Print rolling stats
    all_captures = []
    with open(CAPTURES_LEDGER) as f:
        for line in f:
            if line.strip():
                try:
                    all_captures.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    stats = compute_weekly_stats(all_captures, new_fills)
    if stats["n"] > 0:
        print("\nForward validation (non-overlapping 5d windows):")
        print(
            f"  n={stats['n']}  mean_xs={stats['mean_xs']:+.3%}  "
            f"t={stats['t_stat']:.2f}  hit={stats['hit_rate']:.0%}  "
            f"cum_xs={stats['cum_xs']:+.2%}"
        )
    else:
        print("\nNo completed 5d windows yet.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
