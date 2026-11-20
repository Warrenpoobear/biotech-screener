"""Backfill point-in-time market_snapshot.json files for past dates.

refresh_market_snapshot.py always pulls the most recent window, so its
--as-of-date only relabels today's data. This tool rebuilds each date's
snapshot from history truncated at that date, using the same windows and
math as the live refresh, and writes it as a frozen per-date file.

PIT rules:
    - every input row used has date <= as_of_date (window is cut, not shifted)
    - fund flows use the volume proxy; XBI/IBB AUM has no history to replay
    - nothing is carried from another snapshot; a missing feed stays null
    - existing files are replaced, never written through (staged copies may be
      hard links to production_data/market_snapshot.json)

Output:
    data/staging/pit_regen/<date>/market_snapshot.json   (one per date)
    artifacts/market_snapshot_backfill/inputs.json       (raw fetched series)
    artifacts/market_snapshot_backfill/manifest.json     (date -> sha256)

Usage:
    python tools/backfill_market_snapshots.py
    python tools/backfill_market_snapshots.py --dates 2026-09-15 2026-09-16
    python tools/backfill_market_snapshots.py --inputs artifacts/market_snapshot_backfill/inputs.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
from datetime import date, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from refresh_market_snapshot import _realized_vol, _return_over_period, _validate_snapshot  # noqa: E402

STAGING_ROOT = PROJECT_ROOT / "data" / "staging" / "pit_regen"
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "market_snapshot_backfill"
VERSION = "pit-1.0.0"

# Trading-row windows, matching the yfinance periods in refresh_market_snapshot.py
# (yfinance period="70d" returns the last 70 rows, not 70 calendar days)
WINDOW_ROWS = {"^VIX": 10, "SPY": 40, "XBI": 40, "HYG": 40, "^TNX": 70, "^IRX": 70}
VOLUME_WINDOW_ROWS = 30
VOLUME_PROXY_CAP_MM = 300.0
FETCH_LOOKBACK_DAYS = 120

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("backfill_market_snapshots")


# ---------------------------------------------------------------------------
# PIT reconstruction (pure)
# ---------------------------------------------------------------------------


def _window(rows: list, as_of: str, n_rows: int) -> list:
    """Last n_rows rows dated on or before as_of, sorted by date."""
    return sorted((r for r in rows if r[0] <= as_of), key=lambda r: r[0])[-n_rows:]


def _volume_flow_proxy(volume_series: dict, as_of: str) -> tuple[float | None, str | None]:
    """PIT copy of refresh_market_snapshot._compute_volume_flow_proxy.

    Returns ($MM flow capped at +/-300, last input date used).
    """
    total_flow = 0.0
    used = False
    last_date = None
    for ticker in ("XBI", "IBB"):
        h = _window(volume_series.get(ticker, []), as_of, VOLUME_WINDOW_ROWS)
        if len(h) < 15:
            continue
        head = h[:-5]
        avg_vol = sum(r[2] for r in head) / len(head)
        if avg_vol <= 0:
            continue
        used = True
        last_date = max(last_date or "", h[-1][0])
        for idx in range(len(h) - 5, len(h)):
            close, vol = h[idx][1], h[idx][2]
            prev_close = h[idx - 1][1] if idx > 0 else close
            if prev_close <= 0:
                continue
            daily_return = (close / prev_close) - 1
            dollar_excess = max(vol - avg_vol, 0) * close
            if daily_return > 0.001:
                total_flow += dollar_excess
            elif daily_return < -0.001:
                total_flow -= dollar_excess
    if not used:
        return None, None
    flow = round(total_flow / 1e6, 1)
    return max(min(flow, VOLUME_PROXY_CAP_MM), -VOLUME_PROXY_CAP_MM), last_date


def _hy_oas(observations: list, as_of: str) -> tuple[float | None, str | None]:
    """Latest valid FRED BAMLH0A0HYM2 observation on or before as_of, in bps."""
    for obs_date, val in sorted(observations, key=lambda r: r[0], reverse=True):
        if obs_date <= as_of and val != ".":
            return float(val) * 100, obs_date
    return None, None


def build_pit_snapshot(as_of_date: str, inputs: dict) -> dict:
    """Rebuild the market snapshot as it would have read on as_of_date."""
    closes = inputs["closes"]
    hist = {t: [(r[0], r[1]) for r in _window(closes.get(t, []), as_of_date, d)] for t, d in WINDOW_ROWS.items()}
    if not hist["^VIX"] or not hist["XBI"]:
        raise ValueError(f"No VIX/XBI history on or before {as_of_date} within the live windows")

    vix_date, vix_current = hist["^VIX"][-1]

    xbi_30d = _return_over_period(hist["XBI"], 21)
    spy_30d = _return_over_period(hist["SPY"], 21)
    xbi_vs_spy_30d = (xbi_30d - spy_30d) if (xbi_30d is not None and spy_30d is not None) else None
    if xbi_30d is None:
        raise ValueError(f"Insufficient XBI history for {as_of_date}")

    xbi_mom_10d = _return_over_period(hist["XBI"], 7)
    spy_mom_10d = _return_over_period(hist["SPY"], 7)
    xbi_vol = _realized_vol(hist["XBI"])

    hyg_30d = _return_over_period(hist["HYG"], 21) if hist["HYG"] else None
    credit_spread_change = -hyg_30d / 10 if hyg_30d is not None else None

    hy_oas_current, hy_oas_date = _hy_oas(inputs.get("hy_oas", []), as_of_date)
    biotech_fund_flows, flow_date = _volume_flow_proxy(inputs.get("volumes", {}), as_of_date)

    tnx_current = hist["^TNX"][-1][1] if hist["^TNX"] else None
    irx_hist = hist["^IRX"]
    irx_current = irx_hist[-1][1] if irx_hist else None
    yield_curve_slope = (
        (tnx_current - irx_current) * 100 if (tnx_current is not None and irx_current is not None) else None
    )
    fed_rate_change_3m = irx_current - irx_hist[0][1] if (irx_hist and len(irx_hist) >= 50) else None

    def _fmt(v):
        return None if v is None else f"{v:.2f}"

    def _feed(v, ok="pit"):
        return ok if v is not None else "failed"

    input_last_dates = {
        **{t: (hist[t][-1][0] if hist[t] else None) for t in WINDOW_ROWS},
        "hy_oas_fred": hy_oas_date,
        "volume_proxy": flow_date,
    }
    data_available_timestamp = max(d for d in input_last_dates.values() if d)

    return {
        "provenance": {
            "source": "yfinance (VIX, SPY, XBI, HYG, TNX, IRX, IBB) + FRED (BAMLH0A0HYM2), truncated at as_of_date",
            "as_of_date": as_of_date,
            "vix_date": vix_date,
            "data_available_timestamp": data_available_timestamp,
            "input_last_dates": input_last_dates,
            "pit_mode": "reconstructed",
            "generated_by": "backfill_market_snapshots.py",
            "version": VERSION,
        },
        "vix": _fmt(vix_current),
        "xbi_vs_spy_30d": _fmt(xbi_vs_spy_30d),
        "fed_rate_change_3m": _fmt(fed_rate_change_3m),
        "xbi_momentum_10d": _fmt(xbi_mom_10d),
        "spy_momentum_10d": _fmt(spy_mom_10d),
        "credit_spread_change": _fmt(credit_spread_change),
        "hy_credit_spread": _fmt(hy_oas_current),
        "biotech_fund_flows": _fmt(biotech_fund_flows),
        "xbi_realized_vol_20d": _fmt(xbi_vol),
        "yield_curve_slope_bps": _fmt(yield_curve_slope),
        "tnx_10y_yield": _fmt(tnx_current),
        "irx_13w_yield": _fmt(irx_current),
        "feeds": {
            "vix": _feed(vix_current),
            "spy": _feed(spy_30d),
            "xbi": _feed(xbi_30d),
            "hyg": _feed(hyg_30d),
            "tnx": _feed(tnx_current),
            "irx": _feed(irx_current),
            "fed_rate": _feed(fed_rate_change_3m),
            "hy_oas_fred": _feed(hy_oas_current),
            "biotech_fund_flows": _feed(biotech_fund_flows, "pit_volume_proxy"),
        },
        "notes": {
            "vix": f"CBOE VIX close {vix_date}: {vix_current:.2f}",
            "xbi_vs_spy_30d": (
                f"XBI {xbi_30d:+.1f}% vs SPY {spy_30d:+.1f}% = {xbi_vs_spy_30d:+.1f}pp"
                if xbi_vs_spy_30d is not None
                else "Feed failed"
            ),
            "credit_spread_change": (
                f"HYG 30d return {hyg_30d:+.1f}% -> spread change proxy {credit_spread_change:+.3f}"
                if hyg_30d is not None
                else "Feed failed"
            ),
            "hy_credit_spread": (
                f"ICE BofA HY OAS {hy_oas_date}: {hy_oas_current:.0f} bps (FRED BAMLH0A0HYM2)"
                if hy_oas_current is not None
                else "FRED feed failed"
            ),
            "biotech_fund_flows": (
                f"XBI+IBB volume proxy, capped +/-{VOLUME_PROXY_CAP_MM:.0f}: ${biotech_fund_flows:+.0f}MM "
                "(AUM ledger has no history to replay)"
                if biotech_fund_flows is not None
                else "Insufficient data"
            ),
            "yield_curve": (
                f"10Y {tnx_current:.2f}% - 13W {irx_current:.2f}% = {yield_curve_slope:+.0f} bps"
                if yield_curve_slope is not None
                else "Feed failed"
            ),
            "fed_rate_change_3m": (
                f"13W T-bill now {irx_current:.2f}% vs {irx_hist[0][0]} {irx_hist[0][1]:.2f}% = {fed_rate_change_3m:+.2f}pp"
                if fed_rate_change_3m is not None
                else "Insufficient IRX history in window; not carried"
            ),
        },
    }


def serialize_snapshot(snapshot: dict) -> bytes:
    return (json.dumps(snapshot, indent=2) + "\n").encode("utf-8")


def write_frozen(path: Path, snapshot: dict) -> str:
    """Write snapshot to path without writing through a hard link. Returns sha256."""
    data = serialize_snapshot(snapshot)
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    # os.replace swaps the directory entry, so any other path hard-linked to
    # the old file keeps its content.
    os.replace(tmp, path)
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Fetching (network)
# ---------------------------------------------------------------------------


def fetch_inputs(start: str, end: str) -> dict:
    """Fetch every series needed for dates in [start, end]."""
    import yfinance as yf

    fetch_start = (date.fromisoformat(start) - timedelta(days=FETCH_LOOKBACK_DAYS)).isoformat()
    fetch_end = (date.fromisoformat(end) + timedelta(days=1)).isoformat()

    closes, volumes = {}, {}
    for ticker in list(WINDOW_ROWS) + ["IBB"]:
        h = yf.Ticker(ticker).history(start=fetch_start, end=fetch_end)
        if h.empty:
            raise RuntimeError(f"yfinance returned no history for {ticker}")
        rows = [(d.strftime("%Y-%m-%d"), round(float(r["Close"]), 6), float(r["Volume"])) for d, r in h.iterrows()]
        if ticker in WINDOW_ROWS:
            closes[ticker] = [[d, c] for d, c, _ in rows]
        if ticker in ("XBI", "IBB"):
            volumes[ticker] = [[d, c, v] for d, c, v in rows]

    return {
        "fetch_window": [fetch_start, end],
        "closes": closes,
        "volumes": volumes,
        "hy_oas": _fetch_fred_hy_oas_history(fetch_start, end),
    }


def _fetch_fred_hy_oas_history(start: str, end: str) -> list:
    import urllib.request

    api_key = os.environ.get("FRED_API_KEY", "")
    if not api_key:
        try:
            from dotenv import load_dotenv

            load_dotenv(PROJECT_ROOT / ".env")
            api_key = os.environ.get("FRED_API_KEY", "")
        except ImportError:
            pass
    if not api_key:
        log.warning("FRED_API_KEY not set — HY OAS will be null in every snapshot")
        return []
    url = (
        "https://api.stlouisfed.org/fred/series/observations"
        f"?series_id=BAMLH0A0HYM2&observation_start={start}&observation_end={end}"
        f"&file_type=json&api_key={api_key}"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "WakeRobin-Market/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
    except Exception as exc:
        # The request URL carries the API key. Log the error type only.
        log.warning("FRED HY OAS fetch failed (%s)", type(exc).__name__)
        return []
    return [[o["date"], o["value"]] for o in data.get("observations", [])]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _staged_dates(root: Path) -> list[str]:
    return sorted(p.name for p in root.iterdir() if p.is_dir() and re.fullmatch(r"\d{4}-\d{2}-\d{2}", p.name))


def _validated_dates(dates: list[str]) -> list[str]:
    """Accept only real ISO dates. Reject path segments such as ``../outside``."""
    clean = []
    for raw in dates:
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw):
            raise ValueError(f"date must be YYYY-MM-DD, got {raw!r}")
        try:
            date.fromisoformat(raw)
        except ValueError as exc:
            raise ValueError(f"date must be YYYY-MM-DD, got {raw!r}") from exc
        clean.append(raw)
    return sorted(clean)


def _snapshot_output_path(staging_root: Path, as_of: str) -> Path:
    """Return the snapshot path, which must stay inside staging_root."""
    root = Path(staging_root).resolve()
    out = (root / as_of / "market_snapshot.json").resolve()
    if not out.is_relative_to(root):
        raise ValueError(f"snapshot path escapes staging root: {as_of}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dates",
        nargs="*",
        help="Dates to backfill (default: every dated dir under --staging-root)",
    )
    parser.add_argument("--staging-root", type=Path, default=STAGING_ROOT)
    parser.add_argument("--inputs", type=Path, help="Replay a saved inputs.json instead of fetching")
    parser.add_argument("--dry-run", action="store_true", help="Build and validate, write nothing")
    args = parser.parse_args()

    try:
        dates = _validated_dates(args.dates) if args.dates else _staged_dates(args.staging_root)
    except ValueError as exc:
        log.error("%s", exc)
        return 1
    if not dates:
        log.error("No dates to backfill")
        return 1

    if args.inputs:
        inputs = json.loads(args.inputs.read_text(encoding="utf-8"))
    else:
        inputs = fetch_inputs(dates[0], dates[-1])

    snapshots, failures = {}, {}
    for d in dates:
        try:
            snap = build_pit_snapshot(d, inputs)
        except ValueError as exc:
            failures[d] = str(exc)
            continue
        issues = _validate_snapshot(snap)
        if issues:
            failures[d] = "; ".join(issues)
            continue
        snapshots[d] = snap

    for d, why in failures.items():
        log.error("%s: %s", d, why)
    if failures:
        log.error("%d of %d dates failed — nothing written", len(failures), len(dates))
        return 1

    if args.dry_run:
        log.info("Dry run: %d snapshots built and validated, nothing written", len(snapshots))
        return 0

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    inputs_bytes = (json.dumps(inputs, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if not args.inputs:
        (OUTPUT_DIR / "inputs.json").write_bytes(inputs_bytes)

    manifest = {
        "generated_by": "backfill_market_snapshots.py",
        "version": VERSION,
        "inputs_sha256": hashlib.sha256(inputs_bytes).hexdigest(),
        "snapshots": {},
    }
    project_root = PROJECT_ROOT.resolve()
    staging_root = Path(args.staging_root).resolve()
    for d, snap in snapshots.items():
        out = _snapshot_output_path(staging_root, d)
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            rel = out.relative_to(project_root).as_posix()
        except ValueError:
            rel = out.relative_to(staging_root).as_posix()
        manifest["snapshots"][d] = {
            "path": rel,
            "sha256": write_frozen(out, snap),
        }
        log.info(
            "%s: vix=%s xbi_vs_spy_30d=%s hy_oas=%s",
            d,
            snap["vix"],
            snap["xbi_vs_spy_30d"],
            snap["hy_credit_spread"],
        )

    (OUTPUT_DIR / "manifest.json").write_bytes((json.dumps(manifest, indent=2) + "\n").encode("utf-8"))
    log.info(
        "Wrote %d frozen snapshots; manifest at %s",
        len(snapshots),
        OUTPUT_DIR / "manifest.json",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
