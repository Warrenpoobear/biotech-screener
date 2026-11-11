#!/usr/bin/env python3
"""Daily options IV snapshot collector — tastytrade market metrics.

Fetches ATM IV, IV rank, term structure, event premium, and liquidity state
for all active universe tickers via tastytrade market metrics API.

Every universe ticker receives an explicit row regardless of whether options
data exists. Tickers with no listed contracts are recorded with:
    opt_has_options = 0
    opt_coverage_status = NO_LISTED_OPTIONS
    (all true options-derived fields = null)

Coverage status values:
    VALID_OPTIONS       — options exist and passed quote-quality checks
    LOW_LIQUIDITY_CHAIN — options exist but liquidity is absent/negligible
    NO_LISTED_OPTIONS   — tastytrade returned no IV data (no listed chain)
    FETCH_FAILED        — batch exception prevented any fetch for this ticker

Classification:
    OPTIONS_COVERAGE_NORMALIZATION / NO_MODEL_CHANGE / NO_RANKER_CHANGE /
    NO_SELECTOR_CHANGE / NO_SCORING_CHANGE / NO_TRADING_CHANGE

Writes dated snapshot to:
    production_data/options_snapshot_{as_of_date}.json

Also writes/overwrites the rolling latest:
    production_data/options_snapshot_latest.json

PIT guard:
    tastytrade market metrics are CURRENT-state only. A snapshot is therefore
    written only when ``as_of_date`` equals today's date in America/New_York.
    Any other date (e.g. a ``--catch-up`` backfill) is refused: nothing is
    fetched or written, and ``metadata.skipped`` is set. Without this guard,
    live chains get labelled with an earlier as-of date (look-ahead); see
    production_data/_QUARANTINED_2026-11-01_catchup_options/ and
    _QUARANTINED_2026-11-11_late_fetch_options/.

Usage:
    python3 tools/collect_options_snapshot.py
    python3 tools/collect_options_snapshot.py --batch-size 100 --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv

load_dotenv(REPO_ROOT / ".env")


def _d(v) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _parse_term(exp_ivs, today: date):
    """Return (front_iv, back_iv, term_slope, event_premium) from expiry list."""
    if not exp_ivs:
        return None, None, None, None
    future = [
        (e.expiration_date, _d(e.implied_volatility))
        for e in exp_ivs
        if e.expiration_date >= today and _d(e.implied_volatility) is not None
    ]
    if not future:
        return None, None, None, None
    future.sort()
    front_date, front_iv = future[0]
    back_iv = None
    for exp_date, iv in future[1:]:
        if (exp_date - front_date).days >= 14:
            back_iv = iv
            break
    slope = event_prem = None
    if front_iv and back_iv:
        slope = round((back_iv - front_iv) / front_iv, 6)
        event_prem = "YES" if (front_iv - back_iv) / back_iv > 0.10 else "NO"
    return front_iv, back_iv, slope, event_prem


def _iv_regime(atm_iv: float | None) -> str | None:
    if atm_iv is None:
        return None
    if atm_iv > 2.0:
        return "EXTREME"
    if atm_iv > 1.0:
        return "ELEVATED"
    if atm_iv > 0.4:
        return "NORMAL"
    return "LOW"


def _liq_state(liq: float | None) -> str:
    if liq is None or liq == 0:
        return "absent"
    if liq > 0.005:
        return "liquid"
    if liq > 0.0002:
        return "thin"
    return "absent"


def _coverage_status(atm_iv: float | None, liq_state: str) -> str:
    """Classify options coverage quality. VALID / LOW_LIQUIDITY / NO_LISTED_OPTIONS."""
    if atm_iv is None:
        return "NO_LISTED_OPTIONS"
    if liq_state == "absent":
        return "LOW_LIQUIDITY_CHAIN"
    return "VALID_OPTIONS"


def _null_record(status: str, fetch_ts: str) -> dict:
    """Explicit null row for a ticker with no options data."""
    return {
        "opt_has_options": 0,
        "opt_coverage_status": status,
        "opt_has_data": 0,
        "opt_quote_ts": fetch_ts,
        "opt_atm_iv": None,
        "opt_iv_rank_tos": None,
        "opt_iv_rank_tw": None,
        "opt_iv_percentile": None,
        "opt_iv_5d_change": None,
        "opt_iv_30d": None,
        "opt_hv_30d": None,
        "opt_hv_60d": None,
        "opt_hv_90d": None,
        "opt_iv_hv_spread": None,
        "opt_front_iv": None,
        "opt_back_iv": None,
        "opt_term_slope": None,
        "opt_event_premium": None,
        "opt_iv_regime": None,
        "opt_liquidity_rank": None,
        "opt_liquidity_state": None,
        "opt_use_for_judgment": "NO",
    }


def load_universe_tickers(universe_path: Path) -> list[str]:
    u = json.loads(universe_path.read_text())
    items = u if isinstance(u, list) else u.get("universe", u.get("tickers", []))
    tickers = []
    for item in items:
        if isinstance(item, dict):
            t = item.get("ticker", "")
            s = item.get("status", "active")
            if t and not t.startswith("_") and s != "delisted":
                tickers.append(t)
        elif isinstance(item, str) and not item.startswith("_"):
            tickers.append(item)
    return tickers


async def _fetch(tickers: list[str], batch_size: int, verbose: bool) -> dict:
    from tastytrade import Session
    from tastytrade.metrics import get_market_metrics

    session = Session(is_test=False)
    snapshot: dict = {}
    today = date.today()
    fetch_ts = datetime.now(timezone.utc).isoformat()
    n_batches = math.ceil(len(tickers) / batch_size)

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        bn = i // batch_size + 1
        if verbose:
            print(f"  Batch {bn}/{n_batches} ({len(batch)} tickers)...", end=" ", flush=True)
        try:
            metrics = await get_market_metrics(session, batch)
            returned_syms = set()
            for m in metrics:
                sym = m.symbol
                returned_syms.add(sym)
                front_iv, back_iv, slope, event_prem = _parse_term(
                    m.option_expiration_implied_volatilities or [], today
                )
                atm_iv = _d(m.implied_volatility_index)
                liq = _d(m.liquidity_rank)
                ls = _liq_state(liq)
                regime = _iv_regime(atm_iv)
                cov = _coverage_status(atm_iv, ls)
                usable = "YES" if (atm_iv is not None and ls != "absent" and regime != "EXTREME") else "NO"
                snapshot[sym] = {
                    "opt_has_options": 1 if atm_iv is not None else 0,
                    "opt_coverage_status": cov,
                    "opt_has_data": 1 if atm_iv is not None else 0,
                    "opt_quote_ts": fetch_ts,
                    "opt_atm_iv": atm_iv,
                    "opt_iv_rank_tos": _d(m.tos_implied_volatility_index_rank),
                    "opt_iv_rank_tw": _d(m.tw_implied_volatility_index_rank),
                    "opt_iv_percentile": (
                        float(m.implied_volatility_percentile) if m.implied_volatility_percentile else None
                    ),
                    "opt_iv_5d_change": _d(m.implied_volatility_index_5_day_change),
                    "opt_iv_30d": _d(m.implied_volatility_30_day),
                    "opt_hv_30d": _d(m.historical_volatility_30_day),
                    "opt_hv_60d": _d(m.historical_volatility_60_day),
                    "opt_hv_90d": _d(m.historical_volatility_90_day),
                    "opt_iv_hv_spread": _d(m.iv_hv_30_day_difference),
                    "opt_front_iv": front_iv,
                    "opt_back_iv": back_iv,
                    "opt_term_slope": slope,
                    "opt_event_premium": event_prem,
                    "opt_iv_regime": regime,
                    "opt_liquidity_rank": liq,
                    "opt_liquidity_state": ls,
                    "opt_use_for_judgment": usable,
                }
            # Tickers in the batch but absent from the API response → no listed chain
            for t in batch:
                if t not in returned_syms:
                    snapshot[t] = _null_record("NO_LISTED_OPTIONS", fetch_ts)
            if verbose:
                n_valid = sum(1 for t in batch if snapshot.get(t, {}).get("opt_has_options"))
                n_no_chain = sum(
                    1 for t in batch if snapshot.get(t, {}).get("opt_coverage_status") == "NO_LISTED_OPTIONS"
                )
                print(f"OK ({len(metrics)} returned | valid={n_valid} no_chain={n_no_chain})")
        except Exception as e:
            # Entire batch failed — mark all as FETCH_FAILED
            for t in batch:
                if t not in snapshot:
                    snapshot[t] = _null_record("FETCH_FAILED", fetch_ts)
            if verbose:
                print(f"FAIL [{len(batch)} tickers marked FETCH_FAILED]: {str(e)[:80]}")

    # Defensive pass: any ticker still missing gets NO_LISTED_OPTIONS
    for t in tickers:
        if t not in snapshot:
            snapshot[t] = _null_record("NO_LISTED_OPTIONS", fetch_ts)

    return snapshot


def build_output(snapshot: dict, tickers: list[str], as_of_date: str) -> dict:
    fetch_ts = next(
        (v["opt_quote_ts"] for v in snapshot.values()),
        datetime.now(timezone.utc).isoformat(),
    )
    valid = sum(1 for v in snapshot.values() if v["opt_coverage_status"] == "VALID_OPTIONS")
    low_liq = sum(1 for v in snapshot.values() if v["opt_coverage_status"] == "LOW_LIQUIDITY_CHAIN")
    no_chain = sum(1 for v in snapshot.values() if v["opt_coverage_status"] == "NO_LISTED_OPTIONS")
    fetch_failed = sum(1 for v in snapshot.values() if v["opt_coverage_status"] == "FETCH_FAILED")
    liquid = sum(1 for v in snapshot.values() if v["opt_liquidity_state"] == "liquid")
    thin = sum(1 for v in snapshot.values() if v["opt_liquidity_state"] == "thin")
    usable = sum(1 for v in snapshot.values() if v["opt_use_for_judgment"] == "YES")
    extreme = sum(1 for v in snapshot.values() if v["opt_iv_regime"] == "EXTREME")
    elevated = sum(1 for v in snapshot.values() if v["opt_iv_regime"] == "ELEVATED")
    event_p = sum(1 for v in snapshot.values() if v["opt_event_premium"] == "YES")
    return {
        "metadata": {
            "fetch_timestamp": fetch_ts,
            "as_of_date": as_of_date,
            "source": "tastytrade_market_metrics",
            "classification": "OPTIONS_COVERAGE_NORMALIZATION/NO_MODEL_CHANGE",
            "universe_count": len(tickers),
            "returned_count": len(snapshot),
            # Coverage breakdown — every ticker accounted for
            "coverage_valid": valid,
            "coverage_low_liquidity": low_liq,
            "coverage_no_chain": no_chain,
            "coverage_fetch_failed": fetch_failed,
            # Liquidity sub-counts (within VALID_OPTIONS)
            "liquid_count": liquid,
            "thin_count": thin,
            "usable_for_judgment": usable,
            # IV regime counts
            "iv_regime_extreme": extreme,
            "iv_regime_elevated": elevated,
            "event_premium_detected": event_p,
            # Legacy compat
            "has_options_data": valid + low_liq,
        },
        "tickers": snapshot,
    }


MARKET_TZ = ZoneInfo("America/New_York")


def market_today() -> date:
    """Today's date in the market time zone (not UTC, not the host's zone)."""
    return datetime.now(MARKET_TZ).date()


def skipped_output(as_of_date: str, today: date) -> dict:
    """Result returned when the PIT guard refuses a non-current as_of_date."""
    return {
        "metadata": {
            "skipped": True,
            "skip_reason": "NON_CURRENT_AS_OF_DATE",
            "as_of_date": as_of_date,
            "market_today": today.isoformat(),
            "source": "tastytrade_market_metrics",
            "universe_count": 0,
            "returned_count": 0,
        },
        "tickers": {},
    }


def run(
    as_of_date: str,
    batch_size: int,
    dry_run: bool,
    verbose: bool,
    today: date | None = None,
) -> dict:
    today = today or market_today()
    if as_of_date != today.isoformat():
        # PIT guard: the source only serves current-state metrics, so any
        # other label would be look-ahead (backfill) or a mislabel.
        if verbose:
            print(
                f"REFUSED: as_of_date={as_of_date} != market today {today}; "
                "tastytrade metrics are current-state only (no PIT history). "
                "Nothing fetched or written."
            )
        return skipped_output(as_of_date, today)

    data_dir = REPO_ROOT / "production_data"
    universe_path = data_dir / "universe.json"

    tickers = load_universe_tickers(universe_path)
    if verbose:
        print(f"Universe tickers: {len(tickers)}")

    snapshot = asyncio.run(_fetch(tickers, batch_size, verbose))
    output = build_output(snapshot, tickers, as_of_date)

    if not dry_run:
        dated = data_dir / f"options_snapshot_{as_of_date}.json"
        latest = data_dir / "options_snapshot_latest.json"
        text = json.dumps(output, indent=2, default=str)
        dated.write_text(text)
        latest.write_text(text)
        if verbose:
            print(f"Written: {dated} ({dated.stat().st_size / 1e6:.2f} MB)")
            print(f"Written: {latest}")

    meta = output["metadata"]
    if verbose:
        print(
            f"\nCoverage: valid={meta['coverage_valid']} "
            f"low_liq={meta['coverage_low_liquidity']} "
            f"no_chain={meta['coverage_no_chain']} "
            f"failed={meta['coverage_fetch_failed']} "
            f"| usable={meta['usable_for_judgment']} "
            f"| event_premium={meta['event_premium_detected']}"
        )

    return output


def main():
    parser = argparse.ArgumentParser(description="Daily options IV snapshot — tastytrade")
    parser.add_argument(
        "--date",
        default=None,
        help="As-of date (YYYY-MM-DD); must equal today in America/New_York (default)",
    )
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--dry-run", action="store_true", help="Fetch but do not write files")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    as_of = args.date or market_today().isoformat()
    out = run(as_of, args.batch_size, args.dry_run, verbose=not args.quiet)
    if out["metadata"].get("skipped"):
        sys.exit(2)


if __name__ == "__main__":
    main()
