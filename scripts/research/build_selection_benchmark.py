"""Selection-only benchmark: equal-weight top-N by DEM rank vs XBI.

Answers the key question: is the selection layer generating alpha
before construction drag?

Runs alongside the constructed shadow portfolio using the same date range, but
with simple equal-weight allocation to the top-N DEM-ranked names, rebalanced
at each snapshot.

Reporting convention (schema selection_benchmark.v2):
  - Net of trading costs first. One-way cost of 50 bps (also 25 and 100),
    charged on actual name turnover: the initial buy, then sells plus buys at
    each rebalance. XBI is gross.
  - Returns are compounded.
  - Prices are split-adjusted closes (production_data/price_history_split_adj.csv)
    when that file exists.
  - A snapshot date with no price row uses the last close on or before it
    (at most MAX_PRICE_GAP_DAYS old). A name that stops trading inside a period
    exits at its last close. A name with no entry price is left out of the
    average; a period with too few priced names is skipped and counted.
  - Historical and live periods are reported separately (--live-start).

Usage:
    python scripts/research/build_selection_benchmark.py
    python scripts/research/build_selection_benchmark.py --top-n 30
    python scripts/research/build_selection_benchmark.py --start-date 2026-03-01
    python scripts/research/build_selection_benchmark.py --pit-mode survivorship \\
        --top-n 30 --snapshot-dir data/snapshots_pit_v2
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import logging
import math
import statistics
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

SNAPSHOT_DIR = REPO_ROOT / "data" / "snapshots"
SHADOW_PERF_PATH = REPO_ROOT / "artifacts" / "live_shadow" / "performance.csv"
OUTPUT_DIR = REPO_ROOT / "output" / "benchmarks"
IPO_DATES_PATH = REPO_ROOT / "production_data" / "ipo_dates.json"

SCHEMA_VERSION = "selection_benchmark.v2"
BENCHMARK = "XBI"
DEFAULT_COST_BPS: Tuple[int, ...] = (50, 25, 100)  # base first
DEFAULT_EXCLUDE: Tuple[str, ...] = ("JBIO",)
DEFAULT_LIVE_START = "2024-10-01"
MAX_PRICE_GAP_DAYS = 7
MIN_PRICED_FRACTION = 0.8

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("selection_benchmark")

# Module-level PIT state (set from CLI args)
_ipo_dates: dict[str, str] | None = None
_pit_mode: str = "off"


def default_price_path(production_dir: Optional[Path] = None) -> Path:
    """Split-adjusted closes when present, else the raw history.

    price_history.csv carries unadjusted reverse splits, which read as
    multi-thousand-percent days in a buy-and-hold basket.
    """
    base = Path(production_dir) if production_dir is not None else REPO_ROOT / "production_data"
    adjusted = base / "price_history_split_adj.csv"
    return adjusted if adjusted.exists() else base / "price_history.csv"


PRICE_PATH = default_price_path()


def load_ipo_dates() -> dict[str, str]:
    """Load ipo_dates.json → {ticker: first_price_date}.

    File format: {tickers: {TICKER: {first_price_date, last_price_date}}}
    Returns flat {TICKER: first_price_date} for easy lookup.
    """
    if not IPO_DATES_PATH.exists():
        log.warning("ipo_dates.json not found — PIT survivorship filter disabled")
        return {}
    with open(IPO_DATES_PATH, encoding="utf-8") as f:
        raw = json.load(f)
    tickers = raw.get("tickers", {})
    return {t: v.get("first_price_date", "") for t, v in tickers.items()}


def filter_rankings_pit(rankings: list[dict], snapshot_date: str) -> list[dict]:
    """Remove tickers that hadn't IPO'd by snapshot_date (survivorship filter)."""
    if _pit_mode == "off" or not _ipo_dates:
        return rankings
    filtered = []
    for r in rankings:
        ticker = r.get("ticker", "").upper()
        ipo = _ipo_dates.get(ticker)
        if ipo and ipo > snapshot_date:
            continue  # pre-IPO: exclude
        filtered.append(r)
    return filtered


# ── Prices ───────────────────────────────────────────────────────────


class PriceBook:
    """Closes per ticker with an on-or-before lookup."""

    def __init__(self, series: Dict[str, List[Tuple[str, float]]]):
        self._dates = {t: [d for d, _ in rows] for t, rows in series.items()}
        self._closes = {t: [c for _, c in rows] for t, rows in series.items()}

    @classmethod
    def load(cls, price_path: Path) -> "PriceBook":
        series: Dict[str, Dict[str, float]] = {}
        with open(price_path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                ticker = (row.get("ticker") or "").strip().upper()
                day = (row.get("date") or "").strip()
                raw = (row.get("close") or "").strip()
                if not (ticker and day and raw):
                    continue
                try:
                    close = float(raw)
                except ValueError:
                    continue
                if close > 0 and not math.isnan(close):
                    series.setdefault(ticker, {})[day] = close
        return cls({t: sorted(v.items()) for t, v in series.items()})

    def __len__(self) -> int:
        return len(self._dates)

    def on_or_before(self, ticker: str, day: str, not_before: Optional[str] = None) -> Optional[Tuple[str, float]]:
        """Last (date, close) on or before ``day``, optionally not earlier than ``not_before``."""
        dates = self._dates.get(ticker)
        if not dates:
            return None
        i = bisect.bisect_right(dates, day) - 1
        if i < 0:
            return None
        if not_before is not None and dates[i] < not_before:
            return None
        return dates[i], self._closes[ticker][i]


def _gap_days(later: str, earlier: str) -> int:
    return (date.fromisoformat(later) - date.fromisoformat(earlier)).days


# ── Snapshots ────────────────────────────────────────────────────────


def load_rankings(snapshot_date: str) -> list[dict]:
    """Load rankings.csv for a given snapshot date, sorted by actionable_rank.

    When PIT mode is active, pre-IPO tickers are excluded before ranking.
    """
    rpath = SNAPSHOT_DIR / snapshot_date / "rankings.csv"
    if not rpath.exists():
        return []
    with open(rpath, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    rows = filter_rankings_pit(rows, snapshot_date)
    ranked = []
    for r in rows:
        ar = (r.get("actionable_rank") or "").strip()
        if ar:
            try:
                r["_rank"] = int(float(ar))
                ranked.append(r)
            except ValueError:
                pass
    ranked.sort(key=lambda r: (r["_rank"], r.get("ticker", "")))
    return ranked


def load_shadow_performance() -> list[dict]:
    """Load constructed shadow performance.csv."""
    if not SHADOW_PERF_PATH.exists():
        return []
    rows = []
    with open(SHADOW_PERF_PATH, encoding="utf-8") as f:
        for line in csv.reader(f):
            if len(line) >= 10 and line[0] == "live_shadow_perf.v1":
                try:
                    rows.append(
                        {
                            "date": line[1],
                            "prior_date": line[2],
                            "pnl_pct": float(line[4]) if line[4] else 0,
                            "xbi_pct": float(line[5]) if line[5] else 0,
                            "excess": float(line[6]) if line[6] else 0,
                            "n_held": int(line[7]) if line[7] else 0,
                            "turnover": float(line[8]) if line[8] else 0,
                        }
                    )
                except (ValueError, IndexError):
                    pass
    return rows


def get_snapshot_dates(start_date: str = "2000-01-01") -> list[str]:
    """Get available snapshot dates, sorted ascending."""
    if not SNAPSHOT_DIR.exists():
        return []
    return sorted(
        d.name
        for d in SNAPSHOT_DIR.iterdir()
        if d.is_dir() and len(d.name) == 10 and d.name >= start_date and (d / "rankings.csv").exists()
    )


def build_ew_positions(rankings: list[dict], top_n: int, exclude: Iterable[str] = ()) -> list[dict]:
    """Build equal-weight top-N positions, skipping excluded tickers."""
    skip = {t.upper() for t in exclude}
    selected = [r for r in rankings if r.get("ticker", "").upper() not in skip][:top_n]
    if not selected:
        return []
    weight = 100.0 / len(selected)
    return [
        {
            "ticker": r.get("ticker", "").upper(),
            "rank": r["_rank"],
            "weight_pct": weight,
            "tier": r.get("tier_any", ""),
            "catalyst_days": r.get("catalyst_days", ""),
            "archetype": r.get("archetype", ""),
        }
        for r in selected
    ]


# ── Returns ──────────────────────────────────────────────────────────


def compute_ew_return(positions: list[dict], prices: PriceBook, start: str, end: str) -> dict:
    """Equal-weight return of ``positions`` from ``start`` to ``end``.

    Names with no usable entry price are left out of the average.
    """
    contributors = []
    n_stale_exit = 0
    for pos in positions:
        ticker = pos["ticker"]
        entry = prices.on_or_before(ticker, start)
        if entry is None or _gap_days(start, entry[0]) > MAX_PRICE_GAP_DAYS:
            continue
        exit_ = prices.on_or_before(ticker, end, not_before=entry[0])
        if exit_ is None:
            continue
        if _gap_days(end, exit_[0]) > MAX_PRICE_GAP_DAYS:
            n_stale_exit += 1  # stopped trading inside the period: last close carried
        contributors.append({"ticker": ticker, "return": exit_[1] / entry[1] - 1.0})

    n_priced = len(contributors)
    pnl = sum(c["return"] for c in contributors) / n_priced if n_priced else None
    for c in contributors:
        c["contrib"] = c["return"] / n_priced
    contributors.sort(key=lambda c: c["contrib"], reverse=True)
    return {
        "pnl": pnl,
        "n_held": len(positions),
        "n_priced": n_priced,
        "n_stale_exit": n_stale_exit,
        "contributors": contributors,
    }


def compute_xbi_return(prices: PriceBook, start: str, end: str) -> Optional[float]:
    """Benchmark return between two dates, or None when it cannot be priced."""
    entry = prices.on_or_before(BENCHMARK, start)
    exit_ = prices.on_or_before(BENCHMARK, end)
    if entry is None or exit_ is None:
        return None
    if _gap_days(start, entry[0]) > MAX_PRICE_GAP_DAYS or _gap_days(end, exit_[0]) > MAX_PRICE_GAP_DAYS:
        return None
    return exit_[1] / entry[1] - 1.0


def _series_stats(periods: List[dict], key: str) -> dict:
    """Compounded statistics for one return series against the benchmark."""
    n = len(periods)
    if n == 0:
        return {"n_periods": 0}
    rets = [p[key] / 100.0 for p in periods]
    bench = [p["xbi_pct"] / 100.0 for p in periods]
    excess = [r - b for r, b in zip(rets, bench)]

    cum = math.prod(1 + r for r in rets) - 1
    cum_b = math.prod(1 + b for b in bench) - 1
    years = max(_gap_days(periods[-1]["date"], periods[0]["prior_date"]) / 365.25, 1e-9)
    ann = (1 + cum) ** (1 / years) - 1 if cum > -1 else -1.0
    ann_b = (1 + cum_b) ** (1 / years) - 1 if cum_b > -1 else -1.0

    def _max_dd(values: Sequence[float]) -> float:
        peak, worst, level = 1.0, 0.0, 1.0
        for v in values:
            level *= 1 + v
            peak = max(peak, level)
            worst = min(worst, level / peak - 1)
        return worst

    relative = [(1 + r) / (1 + b) - 1 for r, b in zip(rets, bench)]
    mean_ex = sum(excess) / n
    t_stat = None
    if n >= 3:
        sd = statistics.stdev(excess)
        if sd > 0:
            t_stat = round(mean_ex / (sd / math.sqrt(n)), 2)
    return {
        "n_periods": n,
        "years": round(years, 2),
        "cumulative_pct": round(cum * 100, 4),
        "annualized_pct": round(ann * 100, 2),
        "xbi_cumulative_pct": round(cum_b * 100, 4),
        "xbi_annualized_pct": round(ann_b * 100, 2),
        "cumulative_excess_pct": round((cum - cum_b) * 100, 4),
        "annualized_excess_pct": round((ann - ann_b) * 100, 2),
        "mean_period_excess_pct": round(mean_ex * 100, 4),
        "t_stat": t_stat,
        "hit_rate": round(sum(1 for e in excess if e > 0) / n, 4),
        "max_drawdown_pct": round(_max_dd(rets) * 100, 2),
        "max_relative_drawdown_pct": round(_max_dd(relative) * 100, 2),
    }


def run_benchmark(
    top_n: int = 20,
    start_date: str = "2000-01-01",
    *,
    price_csv: Optional[Path] = None,
    cost_bps: Sequence[int] = DEFAULT_COST_BPS,
    exclude: Iterable[str] = DEFAULT_EXCLUDE,
    live_start: str = DEFAULT_LIVE_START,
) -> dict:
    """Run the selection-only benchmark across all available snapshots."""
    price_path = Path(price_csv) if price_csv is not None else PRICE_PATH
    log.info("Loading prices from %s...", price_path.name)
    prices = PriceBook.load(price_path)
    log.info("Loaded prices for %d tickers", len(prices))

    dates = get_snapshot_dates(start_date)
    log.info("Found %d snapshot dates (from %s)", len(dates), start_date)
    if len(dates) < 2:
        log.error("Need at least 2 snapshots to compute returns")
        return {}

    exclude = tuple(t.upper() for t in exclude)
    positions_by_date = {d: build_ew_positions(load_rankings(d), top_n, exclude) for d in dates}

    periods: List[dict] = []
    skipped: List[dict] = []
    held_before: Optional[set] = None
    for prior_date, current_date in zip(dates[:-1], dates[1:]):
        positions = positions_by_date.get(prior_date, [])
        if not positions:
            skipped.append({"prior_date": prior_date, "date": current_date, "reason": "no ranked names"})
            continue
        ew = compute_ew_return(positions, prices, prior_date, current_date)
        xbi = compute_xbi_return(prices, prior_date, current_date)
        min_priced = max(1, math.ceil(MIN_PRICED_FRACTION * len(positions)))
        if ew["pnl"] is None or ew["n_priced"] < min_priced or xbi is None:
            reason = "benchmark not priced" if xbi is None else f"{ew['n_priced']}/{len(positions)} names priced"
            skipped.append({"prior_date": prior_date, "date": current_date, "reason": reason})
            continue

        held = {p["ticker"] for p in positions}
        # One-way turnover into this period's basket. The first basket is a full buy.
        turnover_in = 1.0 if held_before is None else 1.0 - len(held & held_before) / max(len(held), 1)
        traded = 1.0 if held_before is None else 2.0 * turnover_in  # buys only, then sells + buys
        held_before = held

        row = {
            "date": current_date,
            "prior_date": prior_date,
            "ew_pnl_pct": round(ew["pnl"] * 100, 4),
            "xbi_pct": round(xbi * 100, 4),
            "ew_excess": round((ew["pnl"] - xbi) * 100, 4),
            "n_held": ew["n_held"],
            "n_priced": ew["n_priced"],
            "n_stale_exit": ew["n_stale_exit"],
            "turnover": round(turnover_in, 4),
            "top_contrib": ew["contributors"][0] if ew["contributors"] else None,
            "bottom_contrib": ew["contributors"][-1] if ew["contributors"] else None,
        }
        for bps in cost_bps:
            cost = traded * bps / 100.0  # percent
            row[f"cost_pct_net{bps}"] = round(cost, 6)
            row[f"ew_pnl_pct_net{bps}"] = round(ew["pnl"] * 100 - cost, 4)
        periods.append(row)

    # Constructed shadow, when present
    shadow_by_date = {s["date"]: s for s in load_shadow_performance()}
    for p in periods:
        shadow = shadow_by_date.get(p["date"], {})
        p["shadow_pnl_pct"] = shadow.get("pnl_pct", None)
        p["shadow_excess"] = shadow.get("excess", None)

    # Compounded running totals
    level_ew = level_xbi = level_shadow = 1.0
    for p in periods:
        level_ew *= 1 + p["ew_pnl_pct"] / 100.0
        level_xbi *= 1 + p["xbi_pct"] / 100.0
        p["cum_ew"] = round((level_ew - 1) * 100, 4)
        p["cum_xbi"] = round((level_xbi - 1) * 100, 4)
        p["cum_ew_excess"] = round((level_ew - level_xbi) * 100, 4)
        if p["shadow_pnl_pct"] is not None:
            level_shadow *= 1 + p["shadow_pnl_pct"] / 100.0
            p["cum_shadow"] = round((level_shadow - 1) * 100, 4)

    shadow_periods = [p for p in periods if p["shadow_pnl_pct"] is not None]
    n_shadow = len(shadow_periods)
    gross = _series_stats(periods, "ew_pnl_pct")

    summary = {
        "schema": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pseudo_pit_version": 2 if _pit_mode != "off" else 1,
        "pit_mode": _pit_mode,
        "top_n": top_n,
        "price_source": price_path.name,
        "excluded_tickers": list(exclude),
        "cost_convention": "one-way bps on actual name turnover; first basket is a full buy; XBI gross",
        "cost_bps": list(cost_bps),
        "primary": f"net{cost_bps[0]}" if cost_bps else "gross",
        "live_start": live_start,
        "n_periods": len(periods),
        "n_skipped_periods": len(skipped),
        "skipped_periods": skipped,
        "date_range": f"{periods[0]['prior_date']} to {periods[-1]['date']}" if periods else "",
        "mean_turnover": round(sum(p["turnover"] for p in periods[1:]) / max(len(periods) - 1, 1), 4),
        "gross": gross,
        # Gross headline fields kept for readers of the v1 layout (now compounded).
        "ew_cumulative_pct": gross.get("cumulative_pct"),
        "ew_cumulative_excess_pct": gross.get("cumulative_excess_pct"),
        "ew_win_rate": gross.get("hit_rate"),
        "ew_avg_excess_pct": gross.get("mean_period_excess_pct"),
        "xbi_cumulative_pct": gross.get("xbi_cumulative_pct"),
        "shadow_cumulative_pct": round((level_shadow - 1) * 100, 4) if n_shadow else None,
        "shadow_win_rate": (
            round(sum(1 for p in shadow_periods if (p["shadow_excess"] or 0) > 0) / n_shadow, 4) if n_shadow else None
        ),
        "construction_drag_pct": None,
        "diagnosis": "SELECTION_ONLY: no constructed-shadow periods to compare",
        "segments": {},
    }
    for bps in cost_bps:
        summary[f"net{bps}"] = _series_stats(periods, f"ew_pnl_pct_net{bps}")

    segments = {
        "historical": [p for p in periods if p["date"] < live_start],
        "live": [p for p in periods if p["prior_date"] >= live_start],
    }
    for name, rows in segments.items():
        block = {"gross": _series_stats(rows, "ew_pnl_pct")}
        for bps in cost_bps:
            block[f"net{bps}"] = _series_stats(rows, f"ew_pnl_pct_net{bps}")
        summary["segments"][name] = block

    if n_shadow:
        ew_on_shadow = math.prod(1 + p["ew_pnl_pct"] / 100.0 for p in shadow_periods) - 1
        xbi_on_shadow = math.prod(1 + p["xbi_pct"] / 100.0 for p in shadow_periods) - 1
        drag = (ew_on_shadow - (level_shadow - 1)) * 100
        ew_excess = (ew_on_shadow - xbi_on_shadow) * 100
        summary["construction_drag_pct"] = round(drag, 4)
        if ew_excess > 0 and drag > 0:
            summary["diagnosis"] = "CONSTRUCTION_DRAG: Selection is generating alpha, construction is losing it"
        elif ew_excess > 0 and drag <= 0:
            summary["diagnosis"] = "BOTH_OK: Selection and construction both adding value"
        elif ew_excess <= 0 and drag > 0:
            summary["diagnosis"] = "SIGNAL_COLD_PLUS_DRAG: Signal is cold AND construction is dragging"
        else:
            summary["diagnosis"] = (
                "SIGNAL_COLD: Selection itself is underperforming, construction is not the primary problem"
            )

    return {"summary": summary, "periods": periods}


# ── Report ───────────────────────────────────────────────────────────


def _fmt(value: Optional[float], spec: str = "+.1f", suffix: str = "") -> str:
    return "n/a" if value is None else f"{value:{spec}}{suffix}"


def format_report(summary: dict, top_n: int) -> str:
    lines = [
        "=" * 96,
        f"SELECTION-ONLY BENCHMARK: Equal-Weight Top-{top_n} vs {BENCHMARK}",
        "=" * 96,
        f"Date range: {summary['date_range']}   Periods: {summary['n_periods']}"
        f"   Skipped: {summary['n_skipped_periods']}   Mean one-way turnover: {summary['mean_turnover']:.1%}",
        f"Prices: {summary['price_source']}   PIT mode: {summary['pit_mode']}"
        f"   Excluded: {', '.join(summary['excluded_tickers']) or 'none'}",
        "Costs: one-way bps on actual turnover (first basket is a full buy). XBI is gross. Returns compounded.",
        "",
    ]
    header = (
        f"{'Series':<24}{'Periods':>8}{'Cum':>10}{'XBI cum':>10}{'Ann xs':>9}"
        f"{'Mean xs':>9}{'t':>7}{'Hit':>7}{'MaxDD':>9}{'RelDD':>9}"
    )

    def _row(label: str, s: dict) -> str:
        if not s or not s.get("n_periods"):
            return f"{label:<24}{0:>8}"
        return (
            f"{label:<24}{s['n_periods']:>8}{_fmt(s['cumulative_pct'], '+.1f', '%'):>10}"
            f"{_fmt(s['xbi_cumulative_pct'], '+.1f', '%'):>10}{_fmt(s['annualized_excess_pct'], '+.1f', 'pt'):>9}"
            f"{_fmt(s['mean_period_excess_pct'], '+.2f', 'pt'):>9}{_fmt(s['t_stat'], '+.2f'):>7}"
            f"{s['hit_rate']:>7.0%}{_fmt(s['max_drawdown_pct'], '.1f', '%'):>9}"
            f"{_fmt(s['max_relative_drawdown_pct'], '.1f', '%'):>9}"
        )

    series = [f"net{b}" for b in summary["cost_bps"]]
    blocks = [("FULL PERIOD", summary)]
    blocks += [
        (f"HISTORICAL (before {summary['live_start']})", summary["segments"]["historical"]),
        (f"LIVE PERIOD (from {summary['live_start']})", summary["segments"]["live"]),
    ]
    for title, block in blocks:
        lines += [title, header, "-" * len(header)]
        for key in series:
            label = f"Net {key[3:]} bps" + (" (primary)" if key == summary["primary"] else "")
            lines.append(_row(label, block.get(key, {})))
        lines.append(_row("Gross (before costs)", block.get("gross", {})))
        lines.append("")
    if summary["construction_drag_pct"] is not None:
        lines.append(f"Construction drag (EW - Shadow, shadow periods): {summary['construction_drag_pct']:+.2f}%")
    lines.append(f"DIAGNOSIS: {summary['diagnosis']}")
    if summary["n_skipped_periods"]:
        lines.append(f"Skipped periods: {[(s['prior_date'], s['reason']) for s in summary['skipped_periods'][:8]]}")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Selection-only benchmark")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--start-date", default="2000-01-01")
    parser.add_argument("--also-top30", action="store_true", help="Also run top-30 benchmark")
    parser.add_argument(
        "--snapshot-dir", type=Path, default=None, help="Override snapshot directory (default: data/snapshots)"
    )
    parser.add_argument(
        "--pit-mode",
        choices=["off", "survivorship", "full"],
        default="off",
        help="PIT filtering mode: off=no filter, survivorship=IPO date filter, "
        "full=survivorship + PIT financials (default: off)",
    )
    parser.add_argument(
        "--price-csv",
        type=Path,
        default=None,
        help="Price file (default: production_data/price_history_split_adj.csv, else price_history.csv)",
    )
    parser.add_argument(
        "--cost-bps",
        default=",".join(str(b) for b in DEFAULT_COST_BPS),
        help="One-way cost levels in bps, base case first (default: 50,25,100)",
    )
    parser.add_argument(
        "--exclude-tickers",
        default=",".join(DEFAULT_EXCLUDE),
        help="Comma-separated tickers left out of every basket (default: JBIO; pass '' for none)",
    )
    parser.add_argument("--live-start", default=DEFAULT_LIVE_START, help="First date of the live period")
    args = parser.parse_args()

    # Initialize PIT state
    global _pit_mode, _ipo_dates, SNAPSHOT_DIR
    _pit_mode = args.pit_mode
    if _pit_mode != "off":
        _ipo_dates = load_ipo_dates()
        log.info("PIT mode: %s (%d IPO dates loaded)", _pit_mode, len(_ipo_dates))

    # Allow overriding the snapshot directory for PIT-fixed runs etc.
    if args.snapshot_dir is not None:
        SNAPSHOT_DIR = args.snapshot_dir.resolve()
        log.info("Using snapshot dir: %s", SNAPSHOT_DIR)

    cost_bps = tuple(int(b) for b in args.cost_bps.split(",") if b.strip())
    exclude = tuple(t.strip().upper() for t in args.exclude_tickers.split(",") if t.strip())

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for n in [args.top_n, 30] if args.also_top30 else [args.top_n]:
        log.info("Running EW top-%d benchmark...", n)
        result = run_benchmark(
            top_n=n,
            start_date=args.start_date,
            price_csv=args.price_csv,
            cost_bps=cost_bps,
            exclude=exclude,
            live_start=args.live_start,
        )

        if not result:
            log.error("No results produced")
            continue

        s = result["summary"]
        pit_suffix = f"_pseudo_pit_v{s['pseudo_pit_version']}" if s.get("pseudo_pit_version", 1) > 1 else ""
        output_path = OUTPUT_DIR / f"selection_only_top{n}{pit_suffix}.json"
        with open(output_path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(result, f, indent=2)
        log.info("Wrote %s", output_path)

        print()
        print(format_report(s, n))


if __name__ == "__main__":
    main()
