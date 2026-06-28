#!/usr/bin/env python3
"""Robinhood feed sync — writer functions + daily report for DEM pilot workflow.

Feeds integrated:
  fills          — actual order fills → blotter cost-basis ground truth
  pnl            — realized P&L → net-of-cost reporting
  earnings       — earnings calendar → catalyst date cross-check vs rankings
  iv             — options IV for Top-30 → EES shadow card enrichment
  backfill       — historical prices for new/IPO tickers → price_history.csv
  intraday       — real-time quotes for binary_now names within 5d → alert card
  tradability    — per-ticker tradability flags → soft checklist item
  index_quotes   — XBI/SPY live quotes → header context
  price_drift    — 5d post-snapshot drift vs snap close → HIGH/MEDIUM alerts
  fundamentals   — live market cap tier → tier drift detection
  regime_inputs  — VIX + XBI/SPY momentum → live regime classification

Architecture:
  Writer functions are called from a Claude session AFTER MCP data is fetched.
  Each writer normalizes + persists to data/caches/robinhood/{date}/.
  CLI --report reads all cache files and emits a daily summary artifact.
  The action card (build_personal_pilot_action_card.py) reads from those caches.

Classification: PILOT_ACTION_INFRASTRUCTURE / NO_MODEL_CHANGE / NO_AUTONOMOUS_TRADING

Usage (CLI):
    python3 tools/rh_feed_sync.py --report 2026-06-30
    python3 tools/rh_feed_sync.py --list-caches 2026-06-30
    python3 tools/rh_feed_sync.py --blotter-pending 2026-06-30

Usage (import from Claude session after MCP calls):
    from tools.rh_feed_sync import write_fills, write_pnl, write_earnings
    from tools.rh_feed_sync import write_iv, write_backfill, write_intraday
    from tools.rh_feed_sync import write_regime_inputs
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

RH_CACHE_ROOT = REPO_ROOT / "data" / "caches" / "robinhood"
PILOT_ROOT = REPO_ROOT / "artifacts" / "live_pilot"
BLOTTER_CSV = PILOT_ROOT / "dem_personal_pilot_blotter.csv"
PRODUCTION_DATA = REPO_ROOT / "production_data"
PRICE_HISTORY = PRODUCTION_DATA / "price_history.csv"
PRICE_HISTORY_SPLIT = PRODUCTION_DATA / "price_history_split_adj.csv"
OPTIONS_SHADOW_DIR = REPO_ROOT / "artifacts" / "options_shadow"

ACCOUNT = "802349084"
BINARY_NOW_ALERT_DAYS = 5
INTRADAY_MOVE_ALERT_PCT = 0.05  # 5% intraday move triggers alert

SCHEMA_VERSION = "rh_feed_sync.v1"


# ---------------------------------------------------------------------------
# Cache path helpers
# ---------------------------------------------------------------------------


def _cache_dir(as_of: str) -> Path:
    d = RH_CACHE_ROOT / as_of
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cache_path(as_of: str, feed: str) -> Path:
    return _cache_dir(as_of) / f"{as_of}_rh_{feed}.json"


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def _load_cache(as_of: str, feed: str) -> Optional[dict]:
    p = _cache_path(as_of, feed)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 1. FILLS — actual order fills
# ---------------------------------------------------------------------------


def write_fills(as_of: str, orders_response: Any, account: str = ACCOUNT) -> Path:
    """Persist Robinhood equity orders response.

    Call after: mcp__robinhood-trading__get_equity_orders(account_number=ACCOUNT,
                    state="filled", created_at_gte=as_of)

    orders_response: the full MCP response dict or list of order objects.
    """
    raw_orders = (
        orders_response.get("orders", orders_response) if isinstance(orders_response, dict) else orders_response
    )
    normalized = []
    for o in raw_orders or []:
        ticker = o.get("symbol") or o.get("ticker") or (o.get("instrument_data") or {}).get("symbol", "")
        normalized.append(
            {
                "order_id": o.get("id") or o.get("order_id", ""),
                "ticker": ticker.upper() if ticker else "",
                "side": o.get("side", ""),
                "state": o.get("state", ""),
                "filled_qty": _safe_float(o.get("filled_quantity") or o.get("quantity")),
                "average_price": _safe_float(
                    o.get("average_price") or o.get("price") or o.get("executed_notional", {}).get("amount")
                ),
                "notional": _safe_float(o.get("executed_notional", {}).get("amount") or o.get("notional")),
                "created_at": o.get("created_at", ""),
                "last_transaction_at": o.get("last_transaction_at", ""),
                "type": o.get("type", ""),
                "time_in_force": o.get("time_in_force", ""),
            }
        )

    payload = {
        "schema": "rh_fills.v1",
        "as_of_date": as_of,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "account": account,
        "n_orders": len(normalized),
        "orders": normalized,
    }
    return _write_json(_cache_path(as_of, "fills"), payload)


def pending_blotter_fills(as_of: str) -> List[dict]:
    """Return fills not yet recorded in the blotter."""
    cache = _load_cache(as_of, "fills")
    if not cache:
        return []

    # Read already-logged order IDs from blotter
    logged_ids: set = set()
    if BLOTTER_CSV.exists():
        with open(BLOTTER_CSV, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                note = row.get("operator_note", "")
                if "order_id=" in note:
                    oid = note.split("order_id=")[-1].split()[0]
                    logged_ids.add(oid)

    pending = [o for o in cache.get("orders", []) if o["state"] == "filled" and o["order_id"] not in logged_ids]
    return pending


def write_fills_to_blotter(fills: List[dict], as_of: str, model_hash: str = "", ruleset_hash: str = "") -> int:
    """Append confirmed fills to the blotter. Returns number of rows written.

    Operator must call this explicitly — never called automatically.
    """
    if not fills:
        return 0
    BLOTTER_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not BLOTTER_CSV.exists() or BLOTTER_CSV.stat().st_size == 0

    fieldnames = [
        "date",
        "action",
        "ticker",
        "rank",
        "target_weight",
        "actual_weight",
        "price",
        "shares",
        "notional",
        "reason",
        "data_quality_status",
        "model_hash",
        "ruleset_hash",
        "ees_status",
        "repeat_offender_status",
        "replacement_candidate",
        "operator_note",
    ]
    with open(BLOTTER_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for fill in fills:
            action = "BUY" if fill.get("side", "").lower() == "buy" else "SELL"
            writer.writerow(
                {
                    "date": as_of,
                    "action": action,
                    "ticker": fill["ticker"],
                    "rank": "",
                    "target_weight": "",
                    "actual_weight": "",
                    "price": fill.get("average_price", ""),
                    "shares": fill.get("filled_qty", ""),
                    "notional": fill.get("notional", ""),
                    "reason": "rh_fill_sync",
                    "data_quality_status": "CONFIRMED",
                    "model_hash": model_hash,
                    "ruleset_hash": ruleset_hash,
                    "ees_status": "",
                    "repeat_offender_status": "",
                    "replacement_candidate": "",
                    "operator_note": f"order_id={fill.get('order_id', '')} "
                    f"state={fill.get('state', '')} "
                    f"tif={fill.get('time_in_force', '')}",
                }
            )
    return len(fills)


# ---------------------------------------------------------------------------
# 2. P&L — realized profit & loss
# ---------------------------------------------------------------------------


def write_pnl(as_of: str, pnl_response: Any, account: str = ACCOUNT) -> Path:
    """Persist Robinhood realized P&L response.

    Call after: mcp__robinhood-trading__get_realized_pnl(account_number=ACCOUNT,
                    span="all")
    """
    raw = pnl_response if isinstance(pnl_response, dict) else {}

    payload = {
        "schema": "rh_pnl.v1",
        "as_of_date": as_of,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "account": account,
        "span": raw.get("span", "all"),
        "total_realized_gain_usd": _safe_float(raw.get("total") or raw.get("total_gain")),
        "total_realized_gain_pct": _safe_float(raw.get("total_percentage") or raw.get("total_gain_pct")),
        "n_closing_trades": raw.get("total_trades") or raw.get("n_trades"),
        "buckets": raw.get("buckets") or raw.get("results") or [],
        "raw": raw,
    }
    return _write_json(_cache_path(as_of, "pnl"), payload)


# ---------------------------------------------------------------------------
# 3. EARNINGS CALENDAR — cross-check vs catalyst dates in rankings
# ---------------------------------------------------------------------------


def write_earnings(as_of: str, events_response: Any, window_days: int = 30) -> Path:
    """Persist Robinhood earnings calendar response.

    Call after: mcp__robinhood-trading__get_earnings_calendar(
                    start_date=as_of, days=window_days)
    """
    # Unwrap {data: {results: [...]}} or {results: [...]} or plain list
    if isinstance(events_response, dict):
        raw_events = (
            events_response.get("results")
            or (events_response.get("data") or {}).get("results")
            or events_response.get("events")
            or []
        )
    else:
        raw_events = events_response or []
    if isinstance(raw_events, dict):
        raw_events = list(raw_events.values())

    normalized = []
    for e in raw_events or []:
        ticker = (e.get("symbol") or e.get("ticker") or "").upper()
        # Robinhood schema: report.date / report.timing / report.verified
        report = e.get("report") or {}
        eps = e.get("eps") or {}
        normalized.append(
            {
                "ticker": ticker,
                "report_date": report.get("date") or e.get("report_date") or e.get("date", ""),
                "timing": report.get("timing") or e.get("timing") or "",
                "eps_estimate": _safe_float(eps.get("estimate") or e.get("eps_estimate")),
                "eps_actual": _safe_float(eps.get("actual") or e.get("eps_actual")),
                "verified": bool(report.get("verified") or e.get("verified") or e.get("confirmed")),
            }
        )

    payload = {
        "schema": "rh_earnings.v1",
        "as_of_date": as_of,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "window_days": window_days,
        "n_events": len(normalized),
        "events": normalized,
    }
    return _write_json(_cache_path(as_of, "earnings"), payload)


def cross_check_earnings(as_of: str, rankings_path: Path) -> List[dict]:
    """Compare RH earnings dates against model catalyst_days for Top-30+bench.

    Returns list of discrepancy dicts:
      - model says catalyst within 10d but no RH earnings found
      - RH shows earnings within 10d but model catalyst_days > 30
    """
    cache = _load_cache(as_of, "earnings")
    if not cache or not rankings_path.exists():
        return []

    rh_by_ticker: Dict[str, dict] = {e["ticker"]: e for e in cache.get("events", [])}

    discrepancies = []
    with open(rankings_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rank = _safe_int(row.get("actionable_rank"))
            if rank is None or rank > 60:
                continue
            ticker = row.get("ticker", "").upper()
            cat_days = _safe_int(row.get("catalyst_days"))
            model_in_window = cat_days is not None and cat_days <= 10

            if ticker in rh_by_ticker:
                rh_date = rh_by_ticker[ticker].get("report_date", "")
                rh_days = _days_until(as_of, rh_date) if rh_date else None
                rh_in_window = rh_days is not None and rh_days <= 10

                if rh_in_window and not model_in_window:
                    discrepancies.append(
                        {
                            "ticker": ticker,
                            "rank": rank,
                            "type": "RH_ONLY",
                            "detail": f"RH earnings in {rh_days}d ({rh_date}) but model catalyst_days={cat_days}",
                        }
                    )
                elif model_in_window and not rh_in_window:
                    discrepancies.append(
                        {
                            "ticker": ticker,
                            "rank": rank,
                            "type": "MODEL_ONLY",
                            "detail": f"Model catalyst_days={cat_days} but no RH earnings within 10d",
                        }
                    )
            elif model_in_window:
                discrepancies.append(
                    {
                        "ticker": ticker,
                        "rank": rank,
                        "type": "NOT_IN_RH_CALENDAR",
                        "detail": f"Model catalyst_days={cat_days} but ticker absent from RH earnings calendar",
                    }
                )

    return discrepancies


def _days_until(as_of: str, target: str) -> Optional[int]:
    try:
        from datetime import date

        d0 = date.fromisoformat(as_of[:10])
        d1 = date.fromisoformat(target[:10])
        return (d1 - d0).days
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 4. OPTIONS IV — EES shadow enrichment
# ---------------------------------------------------------------------------


def write_iv(as_of: str, tickers_data: Dict[str, dict]) -> Path:
    """Persist options IV data for Top-30 names.

    tickers_data format mirrors write_rh_options_cache.py (rh_quotes_cache.v1).

    Call after: for each ticker, mcp__robinhood-trading__get_option_chains +
                mcp__robinhood-trading__get_option_quotes.
    """
    try:
        from tools.write_rh_options_cache import write_rh_cache

        path = write_rh_cache(as_of, tickers_data, out_dir=_cache_dir(as_of), validate=False)
    except ImportError:
        # Fallback: write directly
        payload = {
            "schema": "rh_quotes_cache.v1",
            "as_of_date": as_of,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "n_tickers": len(tickers_data),
            "tickers": tickers_data,
        }
        path = _write_json(_cache_path(as_of, "iv"), payload)
    # Also copy to options_shadow dir for collect_options_shadow.py
    OPTIONS_SHADOW_DIR.mkdir(parents=True, exist_ok=True)
    shadow_path = OPTIONS_SHADOW_DIR / f"{as_of}_rh_quotes_cache.json"
    shadow_path.write_text(path.read_text(), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# 5. PRICE BACKFILL — historical prices for new/IPO tickers
# ---------------------------------------------------------------------------


def write_backfill(ticker: str, bars_response: Any, price_csv: Optional[Path] = None) -> int:
    """Append daily bars for a ticker to price_history.csv.

    Returns number of new rows appended.

    bars_response: mcp__robinhood-trading__get_equity_historicals response
    for a single ticker with interval="day".
    """
    if price_csv is None:
        price_csv = PRICE_HISTORY_SPLIT if PRICE_HISTORY_SPLIT.exists() else PRICE_HISTORY

    raw_bars = (
        bars_response.get("results", [])
        or bars_response.get("historicals", [])
        or (bars_response if isinstance(bars_response, list) else [])
    )

    # Read existing dates for this ticker
    existing: set = set()
    if price_csv.exists():
        with open(price_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("ticker", "").upper() == ticker.upper():
                    existing.add(row.get("date", "")[:10])

    new_rows = []
    for bar in raw_bars or []:
        date_str = (bar.get("begins_at") or bar.get("date") or "")[:10]
        if not date_str or date_str in existing:
            continue
        close = _safe_float(bar.get("close_price") or bar.get("close"))
        if close is None:
            continue
        new_rows.append(
            {
                "date": date_str,
                "ticker": ticker.upper(),
                "close": close,
                "open": _safe_float(bar.get("open_price") or bar.get("open")) or "",
                "high": _safe_float(bar.get("high_price") or bar.get("high")) or "",
                "low": _safe_float(bar.get("low_price") or bar.get("low")) or "",
                "volume": _safe_float(bar.get("volume")) or "",
            }
        )

    if not new_rows:
        return 0

    new_rows.sort(key=lambda r: r["date"])
    write_header = not price_csv.exists() or price_csv.stat().st_size == 0
    with open(price_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["date", "ticker", "close", "open", "high", "low", "volume"],
            extrasaction="ignore",
        )
        if write_header:
            writer.writeheader()
        writer.writerows(new_rows)

    return len(new_rows)


# ---------------------------------------------------------------------------
# 6. INTRADAY — real-time quotes for binary_now names
# ---------------------------------------------------------------------------


def write_intraday(
    as_of: str,
    quotes_response: Any,
    binary_now_tickers: Optional[List[str]] = None,
    catalyst_days_map: Optional[Dict[str, int]] = None,
) -> Path:
    """Persist real-time quote snapshot for binary_now names.

    Call after: mcp__robinhood-trading__get_equity_quotes(symbols=[...binary_now...])

    quotes_response: MCP get_equity_quotes response (dict with 'quotes' key or list).
    binary_now_tickers: list of tickers expected (for context in the artifact).
    catalyst_days_map: {ticker: catalyst_days} for alert annotation.
    """
    # Unwrap {data: {results: [...]}} or {results: [...]} or plain list/dict
    if isinstance(quotes_response, dict):
        inner = (
            quotes_response.get("quotes")
            or (quotes_response.get("data") or {}).get("results")
            or quotes_response.get("results")
            or quotes_response
        )
    else:
        inner = quotes_response or []
    # Each element may be {"quote": {...}, "close": {...}} or a flat quote dict
    flat: list = []
    for item in (inner if isinstance(inner, list) else [inner]):
        if isinstance(item, dict) and "quote" in item:
            q = item["quote"]
            q["_close"] = item.get("close", {})
            flat.append(q)
        elif isinstance(item, dict):
            flat.append(item)
    raw_quotes = {q.get("symbol", q.get("ticker", "")): q for q in flat}

    normalized: Dict[str, dict] = {}
    alerts: List[dict] = []

    for sym, q in (raw_quotes or {}).items():
        ticker = sym.upper()
        last = _safe_float(q.get("last_trade_price") or q.get("last_extended_hours_trade_price"))
        ask = _safe_float(q.get("ask_price"))
        bid = _safe_float(q.get("bid_price"))
        prev_close = _safe_float(q.get("adjusted_previous_close") or q.get("previous_close"))
        open_price = _safe_float(q.get("open_price") or q.get("open"))

        intraday_move = None
        if last and prev_close and prev_close > 0:
            intraday_move = (last - prev_close) / prev_close

        cat_days = (catalyst_days_map or {}).get(ticker)
        alert_level = None
        if cat_days is not None and cat_days <= BINARY_NOW_ALERT_DAYS:
            if intraday_move is not None and abs(intraday_move) >= INTRADAY_MOVE_ALERT_PCT:
                alert_level = "MOVE_ALERT"
            else:
                alert_level = "WATCH"

        entry = {
            "ticker": ticker,
            "last": last,
            "bid": bid,
            "ask": ask,
            "prev_close": prev_close,
            "open": open_price,
            "intraday_move_pct": round(intraday_move * 100, 2) if intraday_move is not None else None,
            "catalyst_days": cat_days,
            "alert": alert_level,
            "updated_at": q.get("updated_at") or q.get("last_trade_price_source") or "",
        }
        normalized[ticker] = entry
        if alert_level == "MOVE_ALERT":
            alerts.append({"ticker": ticker, "move_pct": entry["intraday_move_pct"], "catalyst_days": cat_days})

    payload = {
        "schema": "rh_intraday.v1",
        "as_of_date": as_of,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "n_tickers": len(normalized),
        "binary_now_tickers": binary_now_tickers or [],
        "alerts": alerts,
        "quotes": normalized,
    }
    return _write_json(_cache_path(as_of, "intraday"), payload)


# ---------------------------------------------------------------------------
# New model feeds: tradability, index quotes, price drift, fundamentals
# ---------------------------------------------------------------------------


def _cap_bucket(mm: float) -> str:
    if mm >= 10_000:
        return "large"
    if mm >= 2_000:
        return "mid"
    if mm > 0:
        return "small"
    return "unknown"


def write_tradability(
    as_of: str,
    tradability_response: Any,
    account: str = ACCOUNT,
) -> Path:
    """Normalize get_equity_tradability response → {date}_rh_tradability.json."""
    raw = tradability_response or {}

    # Unwrap: {data: {results: [...]}} or plain list
    items: list = []
    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, dict):
        data = raw.get("data") or raw
        results = data.get("results") if isinstance(data, dict) else None
        if isinstance(results, list):
            items = results
        elif isinstance(data, list):
            items = data

    normalized: Dict[str, Any] = {}
    untradeable: List[str] = []

    for item in items:
        if not isinstance(item, dict):
            continue
        ticker = (item.get("symbol") or item.get("ticker") or "").upper()
        if not ticker:
            continue
        tradeable = bool(item.get("tradability") == "tradable" or item.get("tradeable", False))
        fractional = bool(item.get("fractional_tradability") == "tradable" or item.get("fractional", False))
        reason = item.get("reason") or item.get("untradable_reason") or ""
        normalized[ticker] = {
            "ticker": ticker,
            "tradeable": tradeable,
            "fractional": fractional,
            "reason": reason,
        }
        if not tradeable:
            untradeable.append(ticker)

    payload = {
        "schema": "rh_tradability.v1",
        "as_of_date": as_of,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "account": account,
        "n_tickers": len(normalized),
        "n_untradeable": len(untradeable),
        "untradeable": untradeable,
        "tickers": normalized,
    }
    return _write_json(_cache_path(as_of, "tradability"), payload)


def write_index_quotes(as_of: str, quotes_response: Any) -> Path:
    """Normalize get_equity_quotes for XBI/SPY → {date}_rh_index_quotes.json."""
    raw = quotes_response or {}

    # Unwrap to list of quote dicts
    items: list = []
    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, dict):
        data = raw.get("data") or raw
        results = data.get("results") if isinstance(data, dict) else None
        if isinstance(results, list):
            items = results
        elif isinstance(data, list):
            items = data
        else:
            items = [raw]

    by_symbol: Dict[str, Dict] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        sym = (item.get("symbol") or item.get("ticker") or "").upper()
        if not sym:
            continue
        last = _safe_float(
            item.get("last_trade_price") or item.get("last") or item.get("last_extended_hours_trade_price")
        )
        prev = _safe_float(item.get("previous_close") or item.get("adjusted_previous_close") or item.get("prev_close"))
        if last is not None and prev is not None and prev != 0:
            daily_move = (last - prev) / prev
        else:
            daily_move = None
        by_symbol[sym] = {
            "symbol": sym,
            "last": last,
            "prev_close": prev,
            "daily_move_pct": round(daily_move * 100, 3) if daily_move is not None else None,
            "ask": _safe_float(item.get("ask_price") or item.get("ask")),
            "bid": _safe_float(item.get("bid_price") or item.get("bid")),
        }

    xbi = by_symbol.get("XBI", {})
    spy = by_symbol.get("SPY", {})
    xbi_last = xbi.get("last")
    spy_last = spy.get("last")
    xbi_spy_ratio = round(xbi_last / spy_last, 6) if (xbi_last and spy_last) else None

    payload = {
        "schema": "rh_index_quotes.v1",
        "as_of_date": as_of,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "xbi": xbi,
        "spy": spy,
        "xbi_spy_ratio": xbi_spy_ratio,
        "quotes": by_symbol,
    }
    return _write_json(_cache_path(as_of, "index_quotes"), payload)


def write_price_drift(
    as_of: str,
    historicals_response: Any,
    snap_closes: Dict[str, float],
) -> Path:
    """Compare 5d RH historicals to snapshot closes → {date}_rh_price_drift.json.

    snap_closes: {ticker: close_price_from_snapshot}
    Alert thresholds: HIGH >15%, MEDIUM >10%, LOW >5%, OK otherwise.
    """
    raw = historicals_response or {}

    # Unwrap to list of {symbol, historicals: [...]} objects
    items: list = []
    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, dict):
        data = raw.get("data") or raw
        results = data.get("results") if isinstance(data, dict) else None
        if isinstance(results, list):
            items = results
        elif isinstance(data, list):
            items = data

    drifts: Dict[str, Dict] = {}
    alerts: List[Dict] = []

    for item in items:
        if not isinstance(item, dict):
            continue
        ticker = (item.get("symbol") or item.get("ticker") or "").upper()
        if not ticker:
            continue
        bars = item.get("historicals") or item.get("bars") or []
        if not bars:
            continue
        # Find the most recent bar's close
        latest_bar = max(bars, key=lambda b: b.get("begins_at") or b.get("date") or "", default=None)
        if latest_bar is None:
            continue
        live_close = _safe_float(latest_bar.get("close_price") or latest_bar.get("close"))
        if live_close is None:
            continue
        snap_close = snap_closes.get(ticker)
        if snap_close is None or snap_close == 0:
            drift_pct = None
            alert = "NO_SNAP"
        else:
            drift_pct = (live_close - snap_close) / snap_close
            drift_abs = abs(drift_pct)
            if drift_abs > 0.15:
                alert = "HIGH"
            elif drift_abs > 0.10:
                alert = "MEDIUM"
            elif drift_abs > 0.05:
                alert = "LOW"
            else:
                alert = "OK"

        entry = {
            "ticker": ticker,
            "snap_close": round(snap_close, 2) if snap_close is not None else None,
            "live_close": round(live_close, 2) if live_close is not None else None,
            "drift_pct": round(drift_pct * 100, 3) if drift_pct is not None else None,
            "alert": alert,
        }
        drifts[ticker] = entry
        if alert in ("HIGH", "MEDIUM"):
            alerts.append(entry)

    payload = {
        "schema": "rh_price_drift.v1",
        "as_of_date": as_of,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "n_tickers": len(drifts),
        "n_alerts": len(alerts),
        "alerts": alerts,
        "drifts": drifts,
    }
    return _write_json(_cache_path(as_of, "price_drift"), payload)


def write_fundamentals(
    as_of: str,
    fundamentals_response: Any,
    snap_market_caps: Dict[str, float],
) -> Path:
    """Normalize get_equity_fundamentals response → {date}_rh_fundamentals.json.

    snap_market_caps: {ticker: market_cap_mm_from_snapshot}
    Flags tier_drift if live and snap buckets differ.
    RH market_cap field is in USD (not mm); divide by 1e6.
    """
    raw = fundamentals_response or {}

    items: list = []
    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, dict):
        data = raw.get("data") or raw
        results = data.get("results") if isinstance(data, dict) else None
        if isinstance(results, list):
            items = results
        elif isinstance(data, list):
            items = data

    fundamentals: Dict[str, Dict] = {}
    tier_drifts: List[Dict] = []

    for item in items:
        if not isinstance(item, dict):
            continue
        ticker = (item.get("symbol") or item.get("ticker") or "").upper()
        if not ticker:
            continue
        # RH returns market_cap in dollars
        market_cap_raw = _safe_float(item.get("market_cap") or item.get("market_capitalization"))
        live_cap_mm = round(market_cap_raw / 1e6, 2) if market_cap_raw is not None else None
        live_bucket = _cap_bucket(live_cap_mm) if live_cap_mm is not None else "unknown"

        snap_cap_mm = snap_market_caps.get(ticker)
        snap_bucket = _cap_bucket(snap_cap_mm) if snap_cap_mm is not None else "unknown"

        tier_drift = live_bucket != snap_bucket and live_bucket != "unknown" and snap_bucket != "unknown"

        pe_ratio = _safe_float(item.get("pe_ratio"))
        volume = _safe_float(item.get("average_volume_2_weeks") or item.get("volume"))
        entry = {
            "ticker": ticker,
            "live_cap_mm": live_cap_mm,
            "live_bucket": live_bucket,
            "snap_cap_mm": snap_cap_mm,
            "snap_bucket": snap_bucket,
            "tier_drift": tier_drift,
            "pe_ratio": pe_ratio,
            "avg_volume_2w": volume,
            "description": (item.get("description") or "")[:200],
        }
        fundamentals[ticker] = entry
        if tier_drift:
            tier_drifts.append(
                {
                    "ticker": ticker,
                    "from": snap_bucket,
                    "to": live_bucket,
                    "snap_cap_mm": snap_cap_mm,
                    "live_cap_mm": live_cap_mm,
                }
            )

    payload = {
        "schema": "rh_fundamentals.v1",
        "as_of_date": as_of,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "n_tickers": len(fundamentals),
        "n_tier_drifts": len(tier_drifts),
        "tier_drifts": tier_drifts,
        "fundamentals": fundamentals,
    }
    return _write_json(_cache_path(as_of, "fundamentals"), payload)


# ---------------------------------------------------------------------------
# Regime inputs writer
# ---------------------------------------------------------------------------


def write_regime_inputs(
    as_of: str,
    historicals_response: Any,
    index_quotes_response: Any,
) -> Path:
    """Compute live regime classification → {date}_rh_regime_inputs.json.

    historicals_response: get_equity_historicals(["XBI","SPY"], 30d, interval=day)
    index_quotes_response: get_index_quotes([VIX_id, SPX_id, ...])

    Computes from historicals:
      - xbi_30d_return, spy_30d_return, xbi_vs_spy_30d
      - xbi_10d_momentum, spy_10d_momentum
    Extracts VIX from index quotes.
    Calls regime_engine.detect_regime() if available.
    """
    from decimal import Decimal, InvalidOperation

    # ── Unwrap historicals ────────────────────────────────────────────────
    hist_raw = historicals_response or {}
    hist_items: list = []
    if isinstance(hist_raw, list):
        hist_items = hist_raw
    elif isinstance(hist_raw, dict):
        data = hist_raw.get("data") or hist_raw
        results = data.get("results") if isinstance(data, dict) else None
        if isinstance(results, list):
            hist_items = results
        elif isinstance(data, list):
            hist_items = data

    bars_by_sym: Dict[str, list] = {}
    for item in hist_items:
        if not isinstance(item, dict):
            continue
        sym = (item.get("symbol") or item.get("ticker") or "").upper()
        bars = item.get("bars") or item.get("historicals") or []
        if sym and bars:
            bars_by_sym[sym] = sorted(bars, key=lambda b: b.get("begins_at") or "")

    def _momentum(bars: list, lookback: int) -> Optional[float]:
        if len(bars) < 2:
            return None
        last = _safe_float(bars[-1].get("close_price") or bars[-1].get("close"))
        idx = max(0, len(bars) - 1 - lookback)
        ref = _safe_float(bars[idx].get("close_price") or bars[idx].get("close"))
        if last is None or ref is None or ref == 0:
            return None
        return round((last - ref) / ref * 100, 3)

    xbi_bars = bars_by_sym.get("XBI", [])
    spy_bars = bars_by_sym.get("SPY", [])

    xbi_close = _safe_float((xbi_bars[-1].get("close_price") or xbi_bars[-1].get("close")) if xbi_bars else None)
    spy_close = _safe_float((spy_bars[-1].get("close_price") or spy_bars[-1].get("close")) if spy_bars else None)
    xbi_10d = _momentum(xbi_bars, 10)
    spy_10d = _momentum(spy_bars, 10)
    xbi_30d = _momentum(xbi_bars, 30)
    spy_30d = _momentum(spy_bars, 30)
    xbi_vs_spy_30d = round(xbi_30d - spy_30d, 3) if (xbi_30d is not None and spy_30d is not None) else None

    # ── Unwrap index quotes (VIX, SPX, RUT, NDX) ─────────────────────────
    idx_raw = index_quotes_response or {}
    idx_items: list = []
    if isinstance(idx_raw, list):
        idx_items = idx_raw
    elif isinstance(idx_raw, dict):
        data = idx_raw.get("data") or idx_raw
        quotes = data.get("quotes") if isinstance(data, dict) else None
        if isinstance(quotes, list):
            idx_items = quotes
        elif isinstance(data, list):
            idx_items = data

    idx_by_sym: Dict[str, float] = {}
    for q in idx_items:
        if not isinstance(q, dict):
            continue
        sym = (q.get("symbol") or "").upper()
        val = _safe_float(q.get("value") or q.get("current_value"))
        if sym and val is not None:
            idx_by_sym[sym] = val

    vix = idx_by_sym.get("VIX")
    spx = idx_by_sym.get("SPX")
    rut = idx_by_sym.get("RUT")

    # ── Run regime engine ─────────────────────────────────────────────────
    regime_result: Dict[str, Any] = {}
    regime_error: Optional[str] = None
    inputs_complete = vix is not None and xbi_vs_spy_30d is not None

    if inputs_complete:
        try:
            from regime_engine import RegimeDetectionEngine

            engine = RegimeDetectionEngine()

            def _dec(v: Optional[float]) -> Optional[Decimal]:
                if v is None:
                    return None
                try:
                    return Decimal(str(round(v, 4)))
                except InvalidOperation:
                    return None

            result = engine.detect_regime(
                vix_current=Decimal(str(round(vix, 2))),
                xbi_vs_spy_30d=Decimal(str(round(xbi_vs_spy_30d, 3))),
                xbi_momentum_10d=_dec(xbi_10d),
                spy_momentum_10d=_dec(spy_10d),
            )

            # Convert Decimal values to float for JSON serialisation
            def _jsonify(obj: Any) -> Any:
                if isinstance(obj, Decimal):
                    return float(obj)
                if isinstance(obj, dict):
                    return {k: _jsonify(v) for k, v in obj.items()}
                if isinstance(obj, (list, tuple)):
                    return [_jsonify(i) for i in obj]
                return obj

            regime_result = _jsonify(result)
        except Exception as exc:
            regime_error = str(exc)

    payload = {
        "schema": "rh_regime_inputs.v1",
        "as_of_date": as_of,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "inputs_complete": inputs_complete,
        "regime_error": regime_error,
        # Raw inputs
        "vix": vix,
        "spx": spx,
        "rut": rut,
        "xbi_close": xbi_close,
        "spy_close": spy_close,
        "xbi_10d_momentum": xbi_10d,
        "spy_10d_momentum": spy_10d,
        "xbi_30d_return": xbi_30d,
        "spy_30d_return": spy_30d,
        "xbi_vs_spy_30d": xbi_vs_spy_30d,
        # Engine output (empty dict if engine not run)
        "regime": regime_result.get("regime"),
        "regime_description": regime_result.get("regime_description"),
        "confidence": regime_result.get("confidence"),
        "signal_adjustments": regime_result.get("signal_adjustments", {}),
        "indicators": regime_result.get("indicators", {}),
    }
    return _write_json(_cache_path(as_of, "regime_inputs"), payload)


# ---------------------------------------------------------------------------
# Unified cache loader (consumed by build_personal_pilot_action_card.py)
# ---------------------------------------------------------------------------


def load_rh_feed(as_of: str) -> Dict[str, Any]:
    """Return all available RH cache data for as_of date."""
    feeds = [
        "fills",
        "pnl",
        "earnings",
        "iv",
        "intraday",
        "tradability",
        "index_quotes",
        "price_drift",
        "fundamentals",
        "regime_inputs",
    ]
    result: Dict[str, Any] = {"as_of_date": as_of, "available": []}
    for feed in feeds:
        data = _load_cache(as_of, feed)
        if data:
            result[feed] = data
            result["available"].append(feed)
    # IV may also live under options_shadow dir (written by write_rh_options_cache.py)
    if "iv" not in result:
        iv_candidates = sorted(OPTIONS_SHADOW_DIR.glob(f"{as_of}_rh_quotes_cache.json"))
        if iv_candidates:
            try:
                result["iv"] = json.loads(iv_candidates[-1].read_text())
                result["available"].append("iv")
            except Exception:
                pass
    return result


# ---------------------------------------------------------------------------
# CLI — report / blotter-pending / list-caches
# ---------------------------------------------------------------------------


def _report(as_of: str, snap_date: Optional[str] = None) -> None:
    feed = load_rh_feed(as_of)
    snap_dir = _find_snap(snap_date or as_of)

    lines = [
        "# RH Feed Daily Summary",
        "",
        f"**As-of**: {as_of}  ",
        f"**Generated**: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}  ",
        f"**Feeds loaded**: {', '.join(feed['available']) or 'none'}",
        "",
        "---",
        "",
    ]

    # Fills
    fills_cache = feed.get("fills")
    if fills_cache:
        orders = fills_cache.get("orders", [])
        filled = [o for o in orders if o.get("state") == "filled"]
        pending = pending_blotter_fills(as_of)
        lines += [
            "## Fills",
            "",
            f"Orders fetched: {len(orders)} | Filled: {len(filled)} | Pending blotter update: {len(pending)}",
            "",
        ]
        if pending:
            lines += ["**Pending fills (not yet in blotter):**", ""]
            lines += ["| Ticker | Side | Qty | Avg Price | Notional |", "|--------|------|-----|-----------|---------|"]
            for o in pending:
                lines.append(
                    f"| {o['ticker']} | {o['side']} | {o['filled_qty']} "
                    f"| ${o['average_price']} | ${o['notional']} |"
                )
            lines += [
                "",
                "*Run `write_fills_to_blotter()` to record these fills.*",
                "",
            ]
    else:
        lines += ["## Fills", "", "*No fills cache for this date.*", ""]

    # P&L
    pnl_cache = feed.get("pnl")
    if pnl_cache:
        gain = pnl_cache.get("total_realized_gain_usd")
        gain_pct = pnl_cache.get("total_realized_gain_pct")
        lines += [
            "## Realized P&L",
            "",
            f"Total realized gain: **${gain:+,.2f}** ({gain_pct:+.2%})" if gain is not None else "No P&L data.",
            f"Span: {pnl_cache.get('span', 'all')}",
            "",
        ]
    else:
        lines += ["## Realized P&L", "", "*No P&L cache for this date.*", ""]

    # Earnings cross-check
    earnings_cache = feed.get("earnings")
    if earnings_cache and snap_dir:
        rankings_path = snap_dir / "rankings.csv"
        discrepancies = cross_check_earnings(as_of, rankings_path)
        lines += [
            "## Earnings Calendar Cross-Check",
            "",
            f"RH events in window: {earnings_cache.get('n_events', 0)} | Discrepancies: {len(discrepancies)}",
            "",
        ]
        if discrepancies:
            lines += ["| Ticker | Rank | Type | Detail |", "|--------|------|------|--------|"]
            for d in discrepancies:
                lines.append(f"| {d['ticker']} | {d['rank']} | {d['type']} | {d['detail']} |")
            lines += [""]
        else:
            lines += ["All catalyst dates consistent between model and RH calendar.", ""]
    else:
        lines += ["## Earnings Calendar Cross-Check", "", "*No earnings cache for this date.*", ""]

    # IV
    iv_cache = feed.get("iv")
    if iv_cache:
        tickers_with_iv = list(iv_cache.get("tickers", {}).keys())
        lines += [
            "## Options IV",
            "",
            f"IV data for {len(tickers_with_iv)} tickers: {', '.join(tickers_with_iv[:15])}{'…' if len(tickers_with_iv) > 15 else ''}",
            "",
        ]
    else:
        lines += ["## Options IV", "", "*No IV cache for this date.*", ""]

    # Intraday alerts
    intraday_cache = feed.get("intraday")
    if intraday_cache:
        alerts = intraday_cache.get("alerts", [])
        quotes = intraday_cache.get("quotes", {})
        lines += [
            "## Intraday Binary-Now Monitor",
            "",
            f"Tickers monitored: {intraday_cache.get('n_tickers', 0)} | Move alerts: {len(alerts)}",
            "",
        ]
        if alerts:
            lines += ["**MOVE ALERTS (>5% intraday, catalyst within 5d):**", ""]
            for a in alerts:
                lines.append(f"- **{a['ticker']}**: {a['move_pct']:+.1f}% intraday, catalyst in {a['catalyst_days']}d")
            lines += [""]
        lines += ["| Ticker | Last | Move% | Cat Days | Alert |", "|--------|------|-------|----------|-------|"]
        for ticker, q in sorted(quotes.items()):
            move = f"{q['intraday_move_pct']:+.1f}%" if q.get("intraday_move_pct") is not None else "—"
            lines.append(
                f"| {ticker} | ${q.get('last', '—')} | {move} "
                f"| {q.get('catalyst_days', '—')} | {q.get('alert', '—')} |"
            )
        lines += [""]
    else:
        lines += ["## Intraday Binary-Now Monitor", "", "*No intraday cache for this date.*", ""]

    lines += [
        "---",
        "",
        "*Classification: PILOT_ACTION_INFRASTRUCTURE / NO_MODEL_CHANGE / NO_AUTONOMOUS_TRADING*  ",
        f"*Schema: {SCHEMA_VERSION}*",
    ]

    md = "\n".join(lines)
    PILOT_ROOT.mkdir(parents=True, exist_ok=True)
    md_path = PILOT_ROOT / f"RH_FEED_{as_of}.md"
    md_path.write_text(md, encoding="utf-8")
    print(f"Wrote: {md_path}")

    json_path = PILOT_ROOT / f"RH_FEED_{as_of}.json"
    json_path.write_text(
        json.dumps(
            {"schema": SCHEMA_VERSION, "as_of_date": as_of, "feeds": feed, "available": feed["available"]},
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"Wrote: {json_path}")
    print()
    print(f"Feeds available: {', '.join(feed['available']) or 'none'}")


def _list_caches(as_of: str) -> None:
    d = RH_CACHE_ROOT / as_of
    if not d.exists():
        print(f"No cache directory for {as_of}")
        return
    files = sorted(d.iterdir())
    print(f"Cache files for {as_of}:")
    for f in files:
        size = f.stat().st_size
        print(f"  {f.name}  ({size:,} bytes)")


def _blotter_pending(as_of: str) -> None:
    pending = pending_blotter_fills(as_of)
    if not pending:
        print(f"No pending fills for {as_of} (blotter is up to date).")
        return
    print(f"{len(pending)} fills not yet in blotter:")
    for o in pending:
        print(
            f"  {o['side'].upper():4s} {o['ticker']:6s}  qty={o['filled_qty']}  "
            f"avg=${o['average_price']}  notional=${o['notional']}"
        )


def _find_snap(before_date: str) -> Optional[Path]:
    snap_root = REPO_ROOT / "data" / "snapshots"
    candidates = sorted(
        p
        for p in snap_root.iterdir()
        if p.is_dir() and p.name[:4].isdigit() and "__" not in p.name and p.name <= before_date
    )
    return candidates[-1] if candidates else None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_float(v: Any) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _safe_int(v: Any) -> Optional[int]:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--report", metavar="DATE", help="Generate daily RH feed summary")
    grp.add_argument("--list-caches", metavar="DATE", help="List cache files for a date")
    grp.add_argument("--blotter-pending", metavar="DATE", help="Show fills not yet recorded in blotter")
    parser.add_argument("--snap-date", help="Override snapshot date for cross-checks")
    args = parser.parse_args()

    if args.report:
        _report(args.report, snap_date=args.snap_date)
    elif args.list_caches:
        _list_caches(args.list_caches)
    elif args.blotter_pending:
        _blotter_pending(args.blotter_pending)


if __name__ == "__main__":
    main()
