#!/usr/bin/env python3
"""Intraday mover watch — real-time absolute + relative (vs XBI) alert agent.

Polls real-time quotes for the model-relevant watchlist plus XBI, classifies
large intraday moves, links same-day catalyst/news context from Herald and
Grok, and emits read-only artifacts + (when enabled) email alerts.

Spec: specs/changes/spec_063_intraday_mover_watch.md

Phase 1 scaffolding state
-------------------------
- Builder, classifier, and artifact writer are live.
- Quote source is provider-agnostic via `common.realtime_quote_client`.
- Email sending is disabled by default. `--send-email` is a no-op unless
  both:
    1. MASSIVE_API_KEY or POLYGON_API_KEY is set, and
    2. BIOTECH_INTRADAY_REALTIME_TIER=1 is confirmed.
- Cron wrapper: `tools/cron_intraday_mover.sh` (see header for crontab lines).

Usage
-----
    # Dry-run artifact-only (safe with no credentials)
    python tools/build_intraday_mover_watch.py --as-of-ts 2026-04-17T14:30:00Z

    # End-of-day digest rollup
    python tools/build_intraday_mover_watch.py --as-of-date 2026-04-17 --digest-only

    # Live mode (requires key + tier confirmation)
    python tools/build_intraday_mover_watch.py --as-of-ts 2026-04-17T14:30:00Z --send-email
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from common.alert_dedupe import AlertDedupeStore  # noqa: E402
from common.alert_email import is_smtp_configured, send_email  # noqa: E402
from common.realtime_quote_client import HealthStatus, QuoteRecord, RealtimeQuoteClient, make_quote_client  # noqa: E402
from common.watchlist_config import WATCHLIST_MAX, build_model_relevant_watchlist  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("intraday_mover_watch")

SCHEMA_VERSION = "intraday_mover_watch.v1"
BENCHMARK_TICKER = "XBI"

# Default poll cadence (minutes). Configurable via BIOTECH_INTRADAY_POLL_MINUTES.
DEFAULT_POLL_MINUTES = 15

# ---------------------------------------------------------------------------
# Thresholds (policy-chosen; see Spec 063)
# ---------------------------------------------------------------------------
THRESHOLDS: Dict[str, float] = {
    "abs_move_medium_up": 5.0,
    "abs_move_high_up": 10.0,
    "abs_move_medium_down": -5.0,
    "abs_move_high_down": -10.0,
    "rel_move_medium_up": 4.0,
    "rel_move_high_up": 7.0,
    "rel_move_medium_down": -4.0,
    "rel_move_high_down": -7.0,
    "rvol_spike": 2.5,
    "min_price": 1.00,
    "dedup_window_hours": 4.0,
    "step_up_pp": 3.0,
    "max_immediate_emails_per_hour": 5,
}


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------
def _sf(val: Any) -> float:
    if val is None or val == "":
        return math.nan
    try:
        return float(val)
    except (ValueError, TypeError):
        return math.nan


def _num_or_none(val: Any) -> Optional[float]:
    v = _sf(val)
    return None if math.isnan(v) else v


def _iso_z(ts: datetime) -> str:
    return ts.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _poll_filename(as_of_ts: str) -> str:
    # 2026-04-17T14:30:00Z -> 2026-04-17T14-30-00Z_poll.json
    return as_of_ts.replace(":", "-") + "_poll.json"


def _load_rankings(snapshots_dir: Path, as_of_date: str) -> Dict[str, Dict[str, str]]:
    rk = snapshots_dir / as_of_date / "rankings.csv"
    if not rk.exists():
        return {}
    with open(rk, encoding="utf-8") as f:
        return {row["ticker"]: row for row in csv.DictReader(f) if row.get("ticker")}


# ---------------------------------------------------------------------------
# Metrics + classification
# ---------------------------------------------------------------------------
def compute_intraday_metrics(
    stock: QuoteRecord,
    xbi: Optional[QuoteRecord],
    ranking_row: Dict[str, str],
) -> Dict[str, Any]:
    if stock.prev_close <= 0:
        return {"skip_reason": "invalid_prev_close"}

    stock_abs_move_pct = 100.0 * (stock.last / stock.prev_close - 1.0)
    gap_pct = 100.0 * (stock.open / stock.prev_close - 1.0) if stock.prev_close > 0 else None

    rel_move_vs_xbi_pct: Optional[float] = None
    if xbi is not None and xbi.prev_close > 0:
        xbi_abs = 100.0 * (xbi.last / xbi.prev_close - 1.0)
        rel_move_vs_xbi_pct = stock_abs_move_pct - xbi_abs

    avg_vol = stock.avg_volume_20d
    rvol: Optional[float] = None
    if avg_vol and avg_vol > 0:
        rvol = stock.volume / avg_vol

    return {
        "ticker": stock.ticker,
        "tier": ranking_row.get("tier_dev", ""),
        "actionable_rank": (
            int(_sf(ranking_row.get("actionable_rank", "0"))) if ranking_row.get("actionable_rank") else None
        ),
        "catalyst_days": int(_sf(ranking_row.get("catalyst_days", ""))) if ranking_row.get("catalyst_days") else None,
        "is_hard_catalyst": ranking_row.get("is_hard_catalyst", "") == "1",
        "last": round(stock.last, 4),
        "prev_close": round(stock.prev_close, 4),
        "open": round(stock.open, 4),
        "high": round(stock.high, 4),
        "low": round(stock.low, 4),
        "volume": stock.volume,
        "quote_ts": stock.quote_ts,
        "market_status": stock.market_status,
        "source": stock.source,
        "xbi_source": xbi.source if xbi is not None else None,
        "stock_abs_move_pct": round(stock_abs_move_pct, 2),
        "rel_move_vs_xbi_pct": round(rel_move_vs_xbi_pct, 2) if rel_move_vs_xbi_pct is not None else None,
        "gap_pct": round(gap_pct, 2) if gap_pct is not None else None,
        "rvol": round(rvol, 2) if rvol is not None else None,
        "market_cap_mm": _num_or_none(ranking_row.get("market_cap_mm", "")),
        "short_interest_pct": _num_or_none(ranking_row.get("short_interest_pct", "")),
        "pre_event_put_call_ratio": _num_or_none(ranking_row.get("pre_event_put_call_ratio", "")),
        "opt_put_call_skew": _num_or_none(ranking_row.get("opt_put_call_skew", "")),
        "priced_move_pct": _num_or_none(ranking_row.get("priced_move_pct", "")),
    }


def classify_intraday_alerts(row: Dict[str, Any]) -> List[str]:
    codes: List[str] = []
    t = THRESHOLDS
    abs_move = row.get("stock_abs_move_pct")
    rel_move = row.get("rel_move_vs_xbi_pct")
    rvol = row.get("rvol")

    if row.get("last") is not None and row["last"] < t["min_price"]:
        return codes

    if abs_move is not None:
        if abs_move >= t["abs_move_high_up"]:
            codes.append("INTRADAY_ABS_MOVE_UP_HIGH")
        elif abs_move >= t["abs_move_medium_up"]:
            codes.append("INTRADAY_ABS_MOVE_UP_MEDIUM")
        elif abs_move <= t["abs_move_high_down"]:
            codes.append("INTRADAY_ABS_MOVE_DOWN_HIGH")
        elif abs_move <= t["abs_move_medium_down"]:
            codes.append("INTRADAY_ABS_MOVE_DOWN_MEDIUM")

    if rel_move is not None:
        if rel_move >= t["rel_move_high_up"]:
            codes.append("INTRADAY_REL_MOVE_UP_HIGH")
        elif rel_move >= t["rel_move_medium_up"]:
            codes.append("INTRADAY_REL_MOVE_UP_MEDIUM")
        elif rel_move <= t["rel_move_high_down"]:
            codes.append("INTRADAY_REL_MOVE_DOWN_HIGH")
        elif rel_move <= t["rel_move_medium_down"]:
            codes.append("INTRADAY_REL_MOVE_DOWN_MEDIUM")

    # RVOL only fires when true volume is available (avg_volume_20d populated)
    if rvol is not None and rvol >= t["rvol_spike"]:
        codes.append("INTRADAY_RVOL_SPIKE")

    return codes


def compute_severity(row: Dict[str, Any]) -> str:
    codes = row.get("trigger_codes", [])
    news_status = row.get("news_status", "NONE")
    has_high = any(c.endswith("_HIGH") for c in codes)
    has_medium = any(c.endswith("_MEDIUM") for c in codes)

    if has_high:
        return "HIGH"
    if has_medium and news_status == "OFFICIAL":
        return "HIGH"
    if "INTRADAY_RVOL_SPIKE" in codes and has_medium:
        return "MEDIUM"
    if has_medium:
        return "MEDIUM"
    return "LOW"


# ---------------------------------------------------------------------------
# News lookup (Herald classified > Herald raw > Grok watch > NONE)
# ---------------------------------------------------------------------------
def _null_news_record() -> Dict[str, Any]:
    return {
        "news_status": "NONE",
        "source_rank": None,
        "source_type": None,
        "headline": "",
        "published_at_utc": "",
        "url": "",
        "summary": "",
        "confidence": "low",
        "is_same_day": False,
        "catalyst_tag": "none",
    }


def lookup_same_day_news(
    ticker: str,
    as_of_date: str,
    *,
    herald_classified_dir: Path,
    herald_raw_dir: Path,
    grok_watch_dir: Path,
) -> Dict[str, Any]:
    """Look up same-day news for a ticker in source-priority order.

    Scaffolding-level implementation: scans JSON files by date. Production
    path should use indexed lookups; Phase 2 will optimize.
    """
    # 1. Herald classified
    classified_path = herald_classified_dir / f"{as_of_date}.json"
    hit = _find_hit_in_herald(classified_path, ticker)
    if hit:
        return _mark(hit, status="OFFICIAL", rank=_rank_for_source(hit.get("source_type")))

    # 2. Herald raw
    raw_path = herald_raw_dir / f"{as_of_date}.json"
    hit = _find_hit_in_herald(raw_path, ticker)
    if hit:
        return _mark(hit, status="OFFICIAL", rank=2)

    # 3. Grok watch
    grok_path = grok_watch_dir / f"{as_of_date}_watch.json"
    hit = _find_hit_in_grok(grok_path, ticker)
    if hit:
        return _mark(hit, status="SUPPORTING", rank=4, source_type="grok")

    return _null_news_record()


def _find_hit_in_herald(path: Path, ticker: str) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    rows = data.get("releases") or data.get("items") or data.get("rows") or []
    for row in rows:
        if row.get("ticker") == ticker:
            return row
    return None


def _find_hit_in_grok(path: Path, ticker: str) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    alerts = data.get("alerts") or data.get("rows") or []
    for row in alerts:
        if row.get("ticker") == ticker:
            return row
    return None


def _rank_for_source(src: Optional[str]) -> int:
    return {"company_ir": 1, "wire": 2, "fda": 3, "sec": 3, "ctgov": 3}.get(src or "", 3)


def _mark(raw: Dict[str, Any], *, status: str, rank: int, source_type: Optional[str] = None) -> Dict[str, Any]:
    return {
        "news_status": status,
        "source_rank": rank,
        "source_type": source_type or raw.get("source_type"),
        "headline": raw.get("headline") or raw.get("title") or "",
        "published_at_utc": raw.get("published_at_utc") or raw.get("published_at") or "",
        "url": raw.get("url", ""),
        "summary": raw.get("summary", ""),
        "confidence": raw.get("confidence", "medium"),
        "is_same_day": True,
        "catalyst_tag": raw.get("catalyst_tag", "none"),
    }


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------
def build_intraday_mover_watch(
    as_of_ts: str,
    *,
    quote_client: Optional[RealtimeQuoteClient] = None,
    snapshots_dir: Path = REPO_ROOT / "data" / "snapshots",
    artifacts_dir: Path = REPO_ROOT / "artifacts",
    herald_classified_dir: Path = REPO_ROOT / "artifacts" / "herald" / "classified",
    herald_raw_dir: Path = REPO_ROOT / "artifacts" / "herald" / "raw",
    grok_watch_dir: Path = REPO_ROOT / "artifacts" / "grok_biotech_watch",
    send_email: bool = False,
) -> Dict[str, Any]:
    as_of_date = as_of_ts[:10]
    rankings = _load_rankings(snapshots_dir, as_of_date)
    if not rankings:
        return _empty_artifact(as_of_ts, status="NO_DATA", detail="rankings missing")

    watchlist, wl_sources = build_model_relevant_watchlist(
        as_of_date,
        snapshots_dir=snapshots_dir,
        artifacts_dir=artifacts_dir,
        rankings=rankings,
        max_size=WATCHLIST_MAX,
    )

    client = quote_client or make_quote_client()
    health: HealthStatus = client.health()

    fetch_tickers: Set[str] = set(watchlist) | {BENCHMARK_TICKER}
    quotes = client.get_quotes(sorted(fetch_tickers))

    if not quotes:
        return _empty_artifact(
            as_of_ts,
            status="NO_DATA",
            detail=f"no quotes returned (mode={health.mode}: {health.detail})",
            watchlist_size=len(watchlist),
            sources=wl_sources,
            health=health,
        )

    xbi = quotes.get(BENCHMARK_TICKER)

    rows: List[Dict[str, Any]] = []
    for ticker in sorted(watchlist):
        q = quotes.get(ticker)
        if q is None:
            continue
        row = compute_intraday_metrics(q, xbi, rankings.get(ticker, {}))
        if "skip_reason" in row:
            continue
        codes = classify_intraday_alerts(row)
        if not codes:
            continue
        row["trigger_codes"] = codes

        news = lookup_same_day_news(
            ticker,
            as_of_date,
            herald_classified_dir=herald_classified_dir,
            herald_raw_dir=herald_raw_dir,
            grok_watch_dir=grok_watch_dir,
        )
        row.update(news)

        if news["news_status"] == "OFFICIAL":
            row["trigger_codes"].append("INTRADAY_MOVE_WITH_OFFICIAL_NEWS")
        elif news["news_status"] == "SUPPORTING":
            row["trigger_codes"].append("INTRADAY_MOVE_WITH_SUPPORTING_NEWS")
        else:
            row["trigger_codes"].append("INTRADAY_MOVE_NO_OFFICIAL_NEWS")

        row["severity"] = compute_severity(row)
        row["dedupe_key"] = _build_dedupe_key(row, as_of_date)
        rows.append(row)

    xbi_abs_move = None
    if xbi is not None and xbi.prev_close > 0:
        xbi_abs_move = round(100.0 * (xbi.last / xbi.prev_close - 1.0), 2)

    # Derive unambiguous feed provenance from the actual quotes returned
    # rather than from client health alone. This lets the artifact reflect
    # reality: if the Polygon path degrades mid-poll the source field will
    # show what actually delivered the data.
    observed_sources = {q.source for q in quotes.values()}
    feed_source = (
        (next(iter(observed_sources)) if len(observed_sources) == 1 else ",".join(sorted(observed_sources)))
        if observed_sources
        else "none"
    )

    artifact: Dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "as_of_ts": as_of_ts,
        "as_of_date": as_of_date,
        "generated_at": _iso_z(datetime.now(timezone.utc)),
        "status": "OK",
        "market_status": (xbi.market_status if xbi else "unknown"),
        "provider": health.mode,
        "feed_source": feed_source,
        "feed_detail": health.detail,
        "health": {"ok": health.ok, "mode": health.mode, "detail": health.detail},
        "watchlist_size": len(watchlist),
        "n_triggered": len(rows),
        "xbi_abs_move_pct": xbi_abs_move,
        "xbi_source": xbi.source if xbi is not None else None,
        "thresholds": THRESHOLDS,
        "watchlist_sources": wl_sources,
        "rows": rows,
    }

    out_dir = artifacts_dir / "intraday_mover_watch"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / _poll_filename(as_of_ts)
    out_path.write_text(json.dumps(artifact, indent=2, default=str), encoding="utf-8")
    logger.info("Wrote %s", out_path)
    artifact["_json_path"] = str(out_path)

    # Email is triple-gated: flag + live mode + SMTP creds present.
    if send_email:
        if health.mode != "live":
            logger.info(
                "send-email requested but client mode=%s; no email sent (%s)",
                health.mode,
                health.detail,
            )
        elif not is_smtp_configured():
            logger.warning("send-email requested but SMTP not configured; no email sent")
        else:
            dedupe_store = AlertDedupeStore(
                artifacts_dir / "intraday_mover_watch" / "sent_alerts.json",
                window_hours=float(THRESHOLDS["dedup_window_hours"]),
                step_up_pp=float(THRESHOLDS["step_up_pp"]),
            )
            # Prune before deciding — keep state file bounded
            pruned = dedupe_store.prune_older_than(days=7)
            if pruned:
                logger.info("pruned %d stale dedupe entries", pruned)
            _send_immediate_alerts(
                rows,
                feed_source=feed_source,
                feed_detail=health.detail,
                artifact_path=str(out_path),
                dedupe_store=dedupe_store,
            )

    return artifact


def _empty_artifact(
    as_of_ts: str,
    *,
    status: str,
    detail: str,
    watchlist_size: int = 0,
    sources: Optional[Dict[str, int]] = None,
    health: Optional[HealthStatus] = None,
) -> Dict[str, Any]:
    return {
        "schema": SCHEMA_VERSION,
        "as_of_ts": as_of_ts,
        "as_of_date": as_of_ts[:10],
        "generated_at": _iso_z(datetime.now(timezone.utc)),
        "status": status,
        "detail": detail,
        "provider": health.mode if health else "unknown",
        "feed_source": "none",
        "feed_detail": health.detail if health else "no client",
        "watchlist_size": watchlist_size,
        "n_triggered": 0,
        "watchlist_sources": sources or {},
        "health": ({"ok": health.ok, "mode": health.mode, "detail": health.detail} if health else None),
        "rows": [],
    }


def _build_dedupe_key(row: Dict[str, Any], as_of_date: str) -> str:
    import hashlib

    direction = "up" if (row.get("stock_abs_move_pct") or 0) >= 0 else "down"
    headline = row.get("headline") or "none"
    key = f"{row['ticker']}|{direction}|{row.get('severity', 'LOW')}|{headline}|{as_of_date}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def _send_immediate_alerts(
    rows: List[Dict[str, Any]],
    *,
    feed_source: str,
    feed_detail: str,
    artifact_path: str,
    dedupe_store: Optional[AlertDedupeStore] = None,
) -> int:
    """Send one email per HIGH-severity row, capped per hour.

    Consults `dedupe_store` (persistent across invocations) before each
    send. Records each successful send back into the store and flushes to
    disk once at the end.

    Returns the count actually sent. Every email body explicitly stamps:
      - feed source (alpaca / polygon / massive / etc.)
      - news status (OFFICIAL / SUPPORTING / NONE / UNKNOWN_SOURCE_STATE)

    so "no official news found" stays honest in the outbound message.
    """
    high_rows = [r for r in rows if r.get("severity") == "HIGH"]
    cap = int(THRESHOLDS["max_immediate_emails_per_hour"])
    sent = 0
    considered = 0
    for r in high_rows:
        if sent >= cap:
            logger.info(
                "throttle cap reached (%d/hour); %d HIGH rows not emailed this poll",
                cap,
                len(high_rows) - sent,
            )
            break
        considered += 1
        dedupe_key = r.get("dedupe_key", "")
        abs_pct = r.get("stock_abs_move_pct") or 0.0
        rel_pct = r.get("rel_move_vs_xbi_pct")

        if dedupe_store is not None and dedupe_key:
            decision = dedupe_store.decide(dedupe_key, abs_move_pct=abs_pct, rel_move_pct=rel_pct)
            if not decision.should_send:
                logger.info(
                    "[dedupe suppressed] %s — %s (abs=%.2f%% rel=%s)",
                    r.get("ticker", "?"),
                    decision.reason,
                    abs_pct,
                    f"{rel_pct:+.2f}pp" if rel_pct is not None else "N/A",
                )
                continue

        subject, body = format_immediate_alert(
            r, feed_source=feed_source, feed_detail=feed_detail, artifact_path=artifact_path
        )
        if send_email(subject, body):
            sent += 1
            if dedupe_store is not None and dedupe_key:
                dedupe_store.record_sent(
                    dedupe_key,
                    ticker=r.get("ticker", "?"),
                    severity=r.get("severity", "HIGH"),
                    news_status=r.get("news_status", "NONE"),
                    abs_move_pct=abs_pct,
                    rel_move_pct=rel_pct,
                )
    if dedupe_store is not None and sent > 0:
        dedupe_store.save()
    return sent


def derive_potential_drivers(row: Dict[str, Any]) -> List[str]:
    """Return heuristic, unconfirmed driver hints for a HIGH-severity mover row.

    Output order is fixed: News, Options flow, Technical/liquidity, Event proximity,
    Other. Missing inputs render as explicit "unavailable" / "N/A" rather than
    being inferred. Language is probabilistic ("possible", "potential"); caller
    is expected to render these under a "heuristic, unconfirmed" header.
    """
    lines: List[str] = []
    abs_move = row.get("stock_abs_move_pct") or 0.0
    rel_move = row.get("rel_move_vs_xbi_pct")

    # News
    news_status = row.get("news_status", "NONE")
    headline = row.get("headline") or ""
    if news_status == "OFFICIAL":
        source_type = row.get("source_type") or "unknown source"
        news_line = f"Official same-day catalyst ({source_type})" + (f": {headline}" if headline else "")
    elif news_status == "SUPPORTING":
        news_line = "Same-day supporting context only (grok, unverified)"
    else:
        news_line = "No primary or secondary same-day catalyst identified"
    lines.append(f"- News: {news_line}")

    # Options flow (put/call ratio if available; skew as weak fallback)
    pcr = row.get("pre_event_put_call_ratio")
    skew = row.get("opt_put_call_skew")
    if pcr is not None:
        if abs_move < 0 and pcr >= 1.5:
            opt_line = f"Elevated put activity (P/C ratio {pcr:.2f}) — possible downside positioning / hedging"
        elif abs_move > 0 and pcr <= 0.67:
            opt_line = f"Elevated call activity (P/C ratio {pcr:.2f}) — possible speculative upside positioning"
        else:
            opt_line = f"No unusual options imbalance (P/C ratio {pcr:.2f})"
    elif skew is not None:
        opt_line = f"Put/call skew {skew:+.2f} — limited signal without full flow data"
    else:
        opt_line = "No options data available"
    lines.append(f"- Options flow: {opt_line}")

    # Technical / liquidity
    mcap = row.get("market_cap_mm")
    if mcap is None:
        tech_line = "Market-cap data unavailable"
    elif mcap < 500 and abs(abs_move) > 10:
        tech_line = f"Small-cap (${mcap:.0f}M) with {abs(abs_move):.1f}% move — possible liquidity-driven air pocket"
    elif mcap < 500:
        tech_line = f"Small-cap profile (${mcap:.0f}M) — liquidity risk elevated"
    elif mcap < 2000:
        tech_line = f"Mid-cap (${mcap:.0f}M) — liquidity less likely primary driver"
    else:
        tech_line = f"Large-cap (${mcap:.0f}M) — liquidity unlikely driver"
    lines.append(f"- Technical/liquidity: {tech_line}")

    # Event proximity (uses catalyst_days already in row)
    cd = row.get("catalyst_days")
    if cd is None or cd < 0:
        prox_line = "No scheduled catalyst in forward window"
    elif cd <= 5:
        prox_line = f"Near-term catalyst ({cd}d) — potential positioning unwind / binary risk"
    elif cd <= 15:
        prox_line = f"Catalyst in {cd}d — possible pre-positioning or de-risking"
    else:
        prox_line = f"Catalyst in {cd}d — move likely unrelated to scheduled catalyst"
    lines.append(f"- Event proximity: {prox_line}")

    # Other: short interest + sector sympathy (from rel-vs-XBI already present)
    others: List[str] = []
    si = row.get("short_interest_pct")
    if si is not None and si >= 20.0:
        if abs_move < 0:
            others.append(f"High short interest ({si:.1f}%) — possible short reinforcement")
        else:
            others.append(f"High short interest ({si:.1f}%) — possible squeeze dynamics")
    if rel_move is not None and abs(abs_move) >= 5.0 and abs(rel_move) < 2.0:
        others.append("Move largely tracks XBI — possible sector sympathy")
    if not others:
        others.append("No additional signals flagged")
    lines.append(f"- Other: {'; '.join(others)}")

    return lines


def format_immediate_alert(
    r: Dict[str, Any],
    *,
    feed_source: str,
    feed_detail: str,
    artifact_path: str,
) -> tuple:
    """Format one HIGH row as (subject, plain-text body).

    News status labeling (Phase 2 guardrail #2):
      OFFICIAL            → "same-day official catalyst (<source_type>)"
      SUPPORTING          → "same-day supporting context only (grok)"
      NONE                → "no official same-day source found"
      UNKNOWN_SOURCE_STATE→ "herald source state unknown"
    """
    ticker = r["ticker"]
    abs_move = r.get("stock_abs_move_pct") or 0.0
    rel_move = r.get("rel_move_vs_xbi_pct")
    news_status = r.get("news_status", "NONE")

    news_tag = {
        "OFFICIAL": f"official catalyst ({r.get('source_type') or 'unknown source'})",
        "SUPPORTING": "supporting context only (grok, unverified)",
        "NONE": "no official same-day source found",
        "UNKNOWN_SOURCE_STATE": "herald source state unknown",
    }.get(news_status, "no official same-day source found")

    if rel_move is not None:
        subject = f"[HIGH] {ticker} {abs_move:+.1f}% intraday ({rel_move:+.1f}pp vs XBI) — {news_tag}"
    else:
        subject = f"[HIGH] {ticker} {abs_move:+.1f}% intraday — {news_tag}"

    lines: List[str] = [
        f"Intraday Mover Alert — {r.get('quote_ts', '')}",
        "",
        f"Ticker: {ticker}",
        f"Tier / Rank: {r.get('tier', '?')} / {r.get('actionable_rank', '?')}",
        f"Catalyst days: {r.get('catalyst_days', '?')}",
        f"Price: {r.get('last', '?')} (prev close {r.get('prev_close', '?')})",
        f"Move: {abs_move:+.2f}% intraday",
    ]
    if rel_move is not None:
        lines.append(f"Relative vs XBI: {rel_move:+.2f}pp")
    else:
        lines.append("Relative vs XBI: N/A (XBI quote missing)")
    if r.get("rvol") is not None:
        lines.append(f"RVOL: {r['rvol']:.2f}x")
    lines.extend(
        [
            "",
            f"Feed source: {feed_source} ({feed_detail})",
            f"News status: {news_status} — {news_tag}",
        ]
    )
    if r.get("headline"):
        lines.extend(
            [
                f"Headline: {r['headline']}",
                f"Published: {r.get('published_at_utc', '')}",
                f"URL: {r.get('url', '')}",
            ]
        )
    lines.extend(["", "Potential drivers (heuristic, unconfirmed):"])
    lines.extend(derive_potential_drivers(r))
    lines.extend(
        [
            "",
            "Interpretation:",
            "Read-only alert only. Not a trade recommendation.",
            "",
            f"Artifact: {artifact_path}",
        ]
    )
    return subject, "\n".join(lines)


def format_daily_digest_email(d: Dict[str, Any], feed_source: str = "unknown") -> tuple:
    """Format the EOD digest as (subject, plain-text body)."""
    date = d.get("as_of_date", "")
    counts = d.get("severity_counts", {})
    n_unique = d.get("unique_movers", 0)
    n_polls = d.get("polls_seen", 0)

    subject = (
        f"[Digest {date}] {n_unique} movers ({counts.get('HIGH', 0)} HIGH, "
        f"{counts.get('MEDIUM', 0)} MEDIUM) over {n_polls} polls"
    )

    lines: List[str] = [
        f"Intraday Mover Digest — {date}",
        "",
        f"Feed source: {feed_source}",
        f"Polls observed: {n_polls}",
        f"Unique movers: {n_unique}",
        f"Severity: HIGH={counts.get('HIGH', 0)}, MEDIUM={counts.get('MEDIUM', 0)}, LOW={counts.get('LOW', 0)}",
        "",
        "Top absolute movers:",
    ]
    for r in d.get("top_absolute_movers", []):
        lines.append(
            f"  {r['ticker']:6s} {r.get('stock_abs_move_pct', 0):+6.2f}%  "
            f"(rel {r.get('rel_move_vs_xbi_pct', 0):+5.2f}pp)  "
            f"news={r.get('news_status', 'NONE')}"
        )
    lines += ["", "Top relative movers vs XBI:"]
    for r in d.get("top_relative_movers_vs_xbi", []):
        lines.append(
            f"  {r['ticker']:6s} rel {r.get('rel_move_vs_xbi_pct', 0):+5.2f}pp  "
            f"(abs {r.get('stock_abs_move_pct', 0):+6.2f}%)  "
            f"news={r.get('news_status', 'NONE')}"
        )
    lines += ["", "Movers without official news found:"]
    for r in d.get("movers_without_official_news", []):
        lines.append(f"  {r['ticker']:6s} {r.get('stock_abs_move_pct', 0):+6.2f}% — no same-day official source")
    lines += [
        "",
        "Read-only digest only. Not trade recommendations.",
    ]
    return subject, "\n".join(lines)


# ---------------------------------------------------------------------------
# Digest
# ---------------------------------------------------------------------------
def build_daily_digest(
    as_of_date: str,
    artifacts_dir: Path,
    *,
    send_email_flag: bool = False,
) -> Dict[str, Any]:
    out_dir = artifacts_dir / "intraday_mover_watch"
    polls = sorted(out_dir.glob(f"{as_of_date}T*_poll.json"))

    all_rows: List[Dict[str, Any]] = []
    feed_sources: Set[str] = set()
    for p in polls:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        all_rows.extend(data.get("rows", []))
        fs = data.get("feed_source")
        if fs and fs != "none":
            feed_sources.add(fs)

    by_ticker: Dict[str, Dict[str, Any]] = {}
    for r in all_rows:
        t = r["ticker"]
        current = by_ticker.get(t)
        if current is None or abs(r.get("stock_abs_move_pct") or 0) > abs(current.get("stock_abs_move_pct") or 0):
            by_ticker[t] = r

    unique_rows = list(by_ticker.values())

    def _abs_key(r):
        return abs(r.get("stock_abs_move_pct") or 0)

    def _rel_key(r):
        return abs(r.get("rel_move_vs_xbi_pct") or 0)

    top_abs = sorted(unique_rows, key=_abs_key, reverse=True)[:5]
    top_rel = sorted(unique_rows, key=_rel_key, reverse=True)[:5]
    with_official = [r for r in unique_rows if r.get("news_status") == "OFFICIAL"]
    no_official = [r for r in unique_rows if r.get("news_status") in ("NONE", "UNKNOWN_SOURCE_STATE")]

    counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for r in unique_rows:
        counts[r.get("severity", "LOW")] = counts.get(r.get("severity", "LOW"), 0) + 1

    feed_source_label = (
        (next(iter(feed_sources)) if len(feed_sources) == 1 else ",".join(sorted(feed_sources)))
        if feed_sources
        else "none"
    )

    digest = {
        "schema": "intraday_mover_watch_digest.v1",
        "as_of_date": as_of_date,
        "generated_at": _iso_z(datetime.now(timezone.utc)),
        "polls_seen": len(polls),
        "unique_movers": len(unique_rows),
        "feed_source": feed_source_label,
        "severity_counts": counts,
        "top_absolute_movers": top_abs,
        "top_relative_movers_vs_xbi": top_rel,
        "movers_with_official_news": with_official,
        "movers_without_official_news": no_official,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{as_of_date}_digest.json").write_text(json.dumps(digest, indent=2, default=str), encoding="utf-8")
    (out_dir / f"{as_of_date}_digest.md").write_text(_format_digest_md(digest), encoding="utf-8")
    logger.info("Wrote digest for %s (%d unique movers across %d polls)", as_of_date, len(unique_rows), len(polls))

    if send_email_flag:
        if not is_smtp_configured():
            logger.warning("send-email requested for digest but SMTP not configured; no email sent")
        else:
            subject, body = format_daily_digest_email(digest, feed_source=feed_source_label)
            send_email(subject, body)

    return digest


def _format_digest_md(d: Dict[str, Any]) -> str:
    lines = [
        f"# Intraday Mover Watch — {d['as_of_date']}",
        "",
        f"Polls: {d['polls_seen']} | Unique movers: {d['unique_movers']}",
        f"Severity: HIGH={d['severity_counts'].get('HIGH', 0)}, MEDIUM={d['severity_counts'].get('MEDIUM', 0)}, LOW={d['severity_counts'].get('LOW', 0)}",
        "",
        "## Top Absolute Movers",
        "",
    ]
    for r in d.get("top_absolute_movers", []):
        lines.append(
            f"- {r['ticker']}: {r.get('stock_abs_move_pct'):+.1f}% "
            f"(rel vs XBI {r.get('rel_move_vs_xbi_pct'):+.1f}pp) — news={r.get('news_status')}"
        )
    lines.extend(["", "## Top Relative Movers vs XBI", ""])
    for r in d.get("top_relative_movers_vs_xbi", []):
        lines.append(
            f"- {r['ticker']}: rel {r.get('rel_move_vs_xbi_pct'):+.1f}pp "
            f"(abs {r.get('stock_abs_move_pct'):+.1f}%) — news={r.get('news_status')}"
        )
    lines.extend(["", "*Generated: " + d.get("generated_at", "") + "*", ""])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Intraday mover watch (Spec 063)")
    parser.add_argument("--as-of-ts", help="ISO8601 UTC timestamp, e.g. 2026-04-17T14:30:00Z")
    parser.add_argument("--as-of-date", help="YYYY-MM-DD (used with --digest-only)")
    parser.add_argument("--digest-only", action="store_true", help="Build end-of-day digest only")
    parser.add_argument("--send-email", action="store_true", help="Opt-in email sending (gated)")
    parser.add_argument("--snapshots-dir", type=Path, default=REPO_ROOT / "data" / "snapshots")
    parser.add_argument("--artifacts-dir", type=Path, default=REPO_ROOT / "artifacts")
    parser.add_argument(
        "--poll-minutes",
        type=int,
        default=int(os.environ.get("BIOTECH_INTRADAY_POLL_MINUTES", DEFAULT_POLL_MINUTES)),
        help="Advisory poll cadence; informs artifact metadata only (no cron scheduling)",
    )
    args = parser.parse_args()
    started = time.perf_counter()

    if args.digest_only:
        as_of_date = args.as_of_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        build_daily_digest(as_of_date, args.artifacts_dir, send_email_flag=args.send_email)
        try:
            from tools.agent_skill_telemetry import log_agent_run
            from tools.record_skill_feedback import attach_outcome_verdict

            exec_id = log_agent_run(
                "build_intraday_mover_watch",
                f"Intraday digest for {as_of_date}",
                inputs={"as_of_date": as_of_date, "digest_only": True},
                success=True,
                latency_ms=(time.perf_counter() - started) * 1000,
            )
            if exec_id:
                attach_outcome_verdict(exec_id, was_correct=True, evidence=f"digest_only as_of={as_of_date}")
        except Exception:
            pass
        return

    as_of_ts = args.as_of_ts or _iso_z(datetime.now(timezone.utc))
    result = build_intraday_mover_watch(
        as_of_ts,
        snapshots_dir=args.snapshots_dir,
        artifacts_dir=args.artifacts_dir,
        send_email=args.send_email,
    )
    logger.info(
        "status=%s provider=%s triggered=%d",
        result.get("status"),
        result.get("provider"),
        result.get("n_triggered", 0),
    )
    try:
        from tools.agent_skill_telemetry import log_agent_run
        from tools.record_skill_feedback import attach_outcome_verdict

        status = result.get("status")
        exec_id = log_agent_run(
            "build_intraday_mover_watch",
            f"Intraday watch at {as_of_ts}",
            inputs={"as_of_ts": as_of_ts},
            outputs={"n_triggered": result.get("n_triggered"), "status": status},
            success=status != "error",
            latency_ms=(time.perf_counter() - started) * 1000,
        )
        if exec_id:
            attach_outcome_verdict(
                exec_id,
                was_correct=status == "OK",
                evidence=f"status={status} n_triggered={result.get('n_triggered', 0)}",
            )
    except Exception:
        pass


if __name__ == "__main__":
    main()
