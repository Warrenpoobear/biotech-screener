#!/usr/bin/env python3
"""Grok biotech watch — watchlist-scoped news search with email alerts.

Searches xAI Grok for catalyst-relevant news on model-held names, enriches
matches with DEM context (tier, rank, catalyst proximity, policy status),
and optionally emails actionable alerts.

Read-only — does not affect rankings, scoring, or execution.

Output:
    artifacts/grok_watch/{date}_alerts.json
    artifacts/grok_watch/{date}_alerts.md

Usage:
    python tools/build_grok_biotech_watch.py --as-of-date 2026-03-30
    python tools/build_grok_biotech_watch.py --as-of-date 2026-03-30 --send-email
    python tools/build_grok_biotech_watch.py --as-of-date 2026-03-30 --digest-only

Requires:
    XAI_API_KEY environment variable (from console.x.ai)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import smtplib
import sys
import time
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("grok_biotech_watch")

SCHEMA_VERSION = "grok_watch.v2"
WATCHLIST_MAX = 40
DEDUP_WINDOW_HOURS = 4
MAX_IMMEDIATE_EMAILS_PER_HOUR = 5

# Catalyst keywords that elevate severity
CATALYST_KEYWORDS = [
    "topline",
    "phase 2",
    "phase 3",
    "phase ii",
    "phase iii",
    "pdufa",
    "fda",
    "adcom",
    "advisory committee",
    "crl",
    "complete response",
    "approval",
    "approved",
    "reject",
    "primary endpoint",
    "enrollment",
    "breakthrough therapy",
    "accelerated approval",
    "priority review",
    "hold",
    "clinical hold",
    "partial hold",
    "readout",
    "interim",
    "futility",
    "acquire",
    "acquisition",
    "merger",
    "tender offer",
    "buyout",
    "cvr",
    "contingent value",
    "definitive agreement",
]

# Source patterns that indicate high credibility
HIGH_CREDIBILITY_PATTERNS = [
    "fda.gov",
    "sec.gov",
    "clinicaltrials.gov",
    "businesswire",
    "prnewswire",
    "globenewswire",
    "reuters",
    "statnews",
    "endpoints news",
    "fierce biotech",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sf(val: Any) -> float:
    if val is None or val == "":
        return math.nan
    try:
        return float(val)
    except (ValueError, TypeError):
        return math.nan


def _load_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _load_csv_tickers(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    try:
        with open(path, encoding="utf-8") as f:
            return {row["ticker"] for row in csv.DictReader(f) if row.get("ticker")}
    except (KeyError, OSError):
        return set()


def _topic_hash(ticker: str, text: str) -> str:
    """Stable hash for dedup: ticker + first 100 chars of normalized text."""
    normalized = text.lower().strip()[:100]
    return hashlib.sha256(f"{ticker}:{normalized}".encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Watchlist construction
# ---------------------------------------------------------------------------


def build_watchlist(
    as_of_date: str,
    snapshots_dir: Path,
    artifacts_dir: Path,
) -> Dict[str, Dict[str, Any]]:
    """Build watchlist from current model state. Returns {ticker: context_dict}."""
    snap_dir = snapshots_dir / as_of_date
    rankings_path = snap_dir / "rankings.csv"
    if not rankings_path.exists():
        return {}

    # Load full rankings for context
    rankings: Dict[str, Dict[str, str]] = {}
    with open(rankings_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("ticker"):
                rankings[row["ticker"]] = row

    # Gather watchlist sources
    review_queue = _load_csv_tickers(snap_dir / "review_queue.csv")
    trade_plan = _load_csv_tickers(artifacts_dir / "live_shadow" / "trade_plan" / as_of_date / "trade_plan.csv")

    position_tickers: Set[str] = set()
    pos_data = _load_json(artifacts_dir / "live_shadow" / "positions" / f"{as_of_date}.json")
    if pos_data:
        position_tickers = {p["ticker"] for p in pos_data.get("positions", []) if p.get("ticker")}

    catalyst_delta_tickers: Set[str] = set()
    cd_data = _load_json(artifacts_dir / "catalyst_delta" / f"{as_of_date}_delta.json")
    if cd_data:
        catalyst_delta_tickers = {d["ticker"] for d in cd_data.get("deltas", []) if d.get("ticker")}

    # Near-term catalyst names (A/B tier, <=30 days)
    near_catalyst = {
        t
        for t, r in rankings.items()
        if r.get("tier_dev") in ("A", "B")
        and not math.isnan(_sf(r.get("catalyst_days", "")))
        and _sf(r.get("catalyst_days", "")) <= 30
    }

    # Union, capped by rank
    watchlist = review_queue | trade_plan | position_tickers | catalyst_delta_tickers | near_catalyst
    watchlist = {t for t in watchlist if t in rankings}
    if len(watchlist) > WATCHLIST_MAX:
        ranked = sorted(
            watchlist,
            key=lambda t: _sf(rankings[t].get("actionable_rank", "9999")),
        )
        watchlist = set(ranked[:WATCHLIST_MAX])

    # Build context for each ticker
    result = {}
    for ticker in sorted(watchlist):
        r = rankings.get(ticker, {})
        result[ticker] = {
            "tier": r.get("tier_dev", ""),
            "actionable_rank": (int(_sf(r.get("actionable_rank", "0"))) if r.get("actionable_rank") else None),
            "catalyst_days": (int(_sf(r.get("catalyst_days", ""))) if r.get("catalyst_days") else None),
            "catalyst_family": r.get("catalyst_family", ""),
            "is_hard_catalyst": r.get("is_hard_catalyst", "") == "1",
            "in_shadow": ticker in position_tickers,
            "in_trade_plan": ticker in trade_plan,
            "in_review_queue": ticker in review_queue,
        }
    return result


# ---------------------------------------------------------------------------
# Grok API search
# ---------------------------------------------------------------------------


_STRUCTURED_PROMPT = """\
Search for the latest biotech/pharma news about: {query}

Return ONLY a JSON array. Each element must have exactly these fields:
- "title": string, headline text
- "snippet": string, 1-2 sentence summary
- "source": string, source name or domain (e.g. "reuters.com", "Fierce Biotech")
- "source_type": string, one of: "official_company", "fda", "wire_service", "journalist", "analyst", "social", "unknown"
- "official_confirmation": boolean, true if the source is a company PR, SEC filing, FDA notice, or ClinicalTrials.gov
- "catalyst_keyword_hit": string or null, the specific catalyst term found (e.g. "topline", "PDUFA", "phase 3", "CRL", "approval") or null if none
- "topic": string, brief topic label (e.g. "Phase 3 readout", "FDA approval", "enrollment update")
- "date": string, publication date YYYY-MM-DD if known, or ""

Rules:
- Return at most 5 results. If no relevant results, return [].
- Only include biotech/pharma/FDA/clinical trial news.
- Exclude stock price commentary, market analysis, and general financial news.
- official_confirmation must be true ONLY for verified primary sources, not journalist reports about company news.
"""


def search_grok(
    query: str,
    api_key: str,
    max_results: int = 10,
) -> List[Dict[str, Any]]:
    """Search using xAI Grok API. Returns list of typed result dicts.

    Uses the xAI chat completions endpoint with web search enabled.
    Returns structured results with source_type, official_confirmation,
    catalyst_keyword_hit, and topic fields for deterministic classification.
    """
    try:
        import requests
    except ImportError:
        logger.error("requests library required: pip install requests")
        return []

    url = "https://api.x.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    prompt = _STRUCTURED_PROMPT.format(query=query)

    payload = {
        "model": "grok-3",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "search": {"mode": "auto"},
    }

    import time

    max_retries = 3
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            if resp.status_code == 429:
                wait = 2 ** (attempt + 1)
                logger.info("Rate limited, waiting %ds (attempt %d/%d)", wait, attempt + 1, max_retries)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")

            # Parse JSON from response (Grok may wrap in markdown code block)
            content = content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[-1].rsplit("```", 1)[0]
            raw = json.loads(content)
            if not isinstance(raw, list):
                return []
            # Validate and normalize each result to the typed schema
            results = []
            for r in raw[:max_results]:
                if not isinstance(r, dict) or not r.get("title"):
                    continue
                results.append(
                    {
                        "title": str(r.get("title", "")),
                        "snippet": str(r.get("snippet", "")),
                        "source": str(r.get("source", "")),
                        "source_type": str(r.get("source_type", "unknown")),
                        "official_confirmation": bool(r.get("official_confirmation", False)),
                        "catalyst_keyword_hit": r.get("catalyst_keyword_hit") or None,
                        "topic": str(r.get("topic", "")),
                        "date": str(r.get("date", "")),
                    }
                )
            return results
        except requests.exceptions.RequestException as exc:
            if attempt < max_retries - 1:
                time.sleep(2 ** (attempt + 1))
                continue
            logger.warning("Grok API request failed for %r: %s", query, exc)
            return []
        except (json.JSONDecodeError, KeyError, IndexError) as exc:
            logger.warning("Grok API response parse failed for %r: %s", query, exc)
            return []
    logger.warning("Grok API exhausted retries for %r", query)
    return []


# ---------------------------------------------------------------------------
# Alert classification
# ---------------------------------------------------------------------------


def classify_severity(
    result: Dict[str, Any],
    ticker_context: Dict[str, Any],
) -> str:
    """Classify alert severity using structured fields from Grok response.

    Uses the typed schema (source_type, official_confirmation,
    catalyst_keyword_hit) for primary classification, with local
    keyword/source fallback for robustness.
    """
    # Structured fields from Grok (preferred)
    source_type = result.get("source_type", "unknown")
    is_official = result.get("official_confirmation", False)
    catalyst_hit = result.get("catalyst_keyword_hit")

    # Local fallback: verify against our own keyword/source lists
    title = (result.get("title") or "").lower()
    snippet = (result.get("snippet") or "").lower()
    source = (result.get("source") or "").lower()
    text = f"{title} {snippet}"

    has_catalyst_keyword = bool(catalyst_hit) or any(kw in text for kw in CATALYST_KEYWORDS)
    is_credible_source = (
        is_official
        or source_type
        in (
            "official_company",
            "fda",
            "wire_service",
        )
        or any(s in source for s in HIGH_CREDIBILITY_PATTERNS)
    )

    catalyst_days = ticker_context.get("catalyst_days")
    near_catalyst = catalyst_days is not None and catalyst_days <= 14

    # HIGH: official/credible + catalyst language, or near-term + catalyst
    if is_credible_source and has_catalyst_keyword:
        return "HIGH"
    if near_catalyst and has_catalyst_keyword:
        return "HIGH"
    if is_credible_source and near_catalyst:
        return "HIGH"

    # MEDIUM: credible source OR catalyst keyword
    if is_credible_source or has_catalyst_keyword:
        return "MEDIUM"

    return "LOW"


# ---------------------------------------------------------------------------
# Dedup
# ---------------------------------------------------------------------------


def load_dedup_state(path: Path) -> Dict[str, str]:
    """Load dedup state: {topic_hash: last_alert_iso_timestamp}."""
    data = _load_json(path)
    return data if isinstance(data, dict) else {}


def save_dedup_state(path: Path, state: Dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)


def is_duplicate(
    topic_hash: str,
    dedup_state: Dict[str, str],
    now_iso: str,
) -> bool:
    """Check if this topic was already alerted within DEDUP_WINDOW_HOURS."""
    last = dedup_state.get(topic_hash)
    if not last:
        return False
    try:
        last_dt = datetime.fromisoformat(last.replace("Z", "+00:00"))
        now_dt = datetime.fromisoformat(now_iso.replace("Z", "+00:00"))
        delta_hours = (now_dt - last_dt).total_seconds() / 3600
        return delta_hours < DEDUP_WINDOW_HOURS
    except (ValueError, TypeError):
        return False


# ---------------------------------------------------------------------------
# Email
# ---------------------------------------------------------------------------


def send_alert_email(
    subject: str,
    body_html: str,
    body_text: str,
    to_addr: Optional[str] = None,
) -> bool:
    """Send alert email via SMTP. Returns True on success."""
    smtp_host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(os.environ.get("SMTP_PORT", "587"))
    smtp_user = os.environ.get("SMTP_USER", "")
    smtp_pass = os.environ.get("SMTP_PASSWORD", "")
    to_str = to_addr or os.environ.get("ALERT_EMAIL_TO") or os.environ.get("ALERT_RECIPIENT", "")

    if not to_str:
        logger.warning("No recipient configured — skipping email")
        return False

    if not smtp_user or not smtp_pass:
        logger.warning("SMTP credentials not configured — skipping email")
        return False

    recipients = [e.strip() for e in to_str.split(",") if e.strip()]
    if not recipients:
        logger.warning("No valid recipients — skipping email")
        return False

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = smtp_user
    msg["To"] = to_str
    msg.attach(MIMEText(body_text, "plain"))
    msg.attach(MIMEText(body_html, "html"))

    try:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.sendmail(smtp_user, recipients, msg.as_string())
        logger.info("Email sent to %s: %s", ", ".join(recipients), subject)
        return True
    except Exception as exc:
        logger.warning("Email send failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Alert formatting
# ---------------------------------------------------------------------------


def format_alert_email(alert: Dict[str, Any]) -> tuple:
    """Format one alert as (subject, html_body, text_body)."""
    ticker = alert["ticker"]
    severity = alert["severity"]
    ctx = alert["context"]
    title = alert.get("title", "")

    cat_days = ctx.get("catalyst_days")
    cat_str = f"{cat_days}d to catalyst" if cat_days is not None else "no near catalyst"

    topic = alert.get("topic", "")
    subject = f"[{severity}] {ticker} — {topic or title[:60]}"

    official = "YES" if alert.get("official_confirmation") else "no"
    kw_hit = alert.get("catalyst_keyword_hit") or "-"
    src_type = alert.get("source_type", "unknown")

    text_body = (
        f"Ticker: {ticker}\n"
        f"Alert: {severity}\n"
        f"Topic: {topic}\n"
        f"Source: {alert.get('source', '?')} ({src_type})\n"
        f"Official: {official}\n"
        f"Catalyst keyword: {kw_hit}\n"
        f"Title: {title}\n"
        f"Snippet: {alert.get('snippet', '')}\n"
        "\n"
        "-- Model Context --\n"
        f"Tier: {ctx.get('tier', '?')}, Rank: {ctx.get('actionable_rank', '?')}\n"
        f"Catalyst: {cat_str}, Family: {ctx.get('catalyst_family', '?')}\n"
        f"Hard catalyst: {ctx.get('is_hard_catalyst', False)}\n"
        f"In shadow: {ctx.get('in_shadow', False)}\n"
        f"In trade plan: {ctx.get('in_trade_plan', False)}\n"
        f"In review queue: {ctx.get('in_review_queue', False)}\n"
    )

    html_body = """<h3>[{severity}] {ticker} — {topic}</h3>
<p><b>{title}</b></p>
<p>{alert.get('snippet', '')}</p>
<p><small>Source: {alert.get('source', '?')} ({src_type}) | Official: {official} | Keyword: {kw_hit}</small></p>
<hr>
<table>
<tr><td>Tier</td><td><b>{ctx.get('tier', '?')}</b></td></tr>
<tr><td>Rank</td><td>{ctx.get('actionable_rank', '?')}</td></tr>
<tr><td>Catalyst</td><td>{cat_str}</td></tr>
<tr><td>Family</td><td>{ctx.get('catalyst_family', '?')}</td></tr>
<tr><td>Hard catalyst</td><td>{ctx.get('is_hard_catalyst', False)}</td></tr>
<tr><td>Shadow</td><td>{ctx.get('in_shadow', False)}</td></tr>
<tr><td>Trade plan</td><td>{ctx.get('in_trade_plan', False)}</td></tr>
<tr><td>Review queue</td><td>{ctx.get('in_review_queue', False)}</td></tr>
</table>
<p><small>Wake Robin Grok Watch — {alert.get('as_of_date', '')}</small></p>"""

    return subject, html_body, text_body


def format_digest_md(alerts: List[Dict[str, Any]], as_of_date: str) -> str:
    """Format all alerts as markdown digest."""
    lines = [f"# Grok Biotech Watch — {as_of_date}", ""]

    if not alerts:
        lines.append("No alerts triggered.")
        lines.append("")
        return "\n".join(lines)

    lines.append(f"**{len(alerts)} alerts** across {len(set(a['ticker'] for a in alerts))} tickers")
    lines.append("")

    # Group by severity
    for sev in ("HIGH", "MEDIUM", "LOW"):
        sev_alerts = [a for a in alerts if a["severity"] == sev]
        if not sev_alerts:
            continue
        lines.append(f"## {sev}")
        lines.append("")
        lines.append("| Ticker | Tier | Rank | Cat | Topic | Source | Official | Keyword |")
        lines.append("|--------|------|------|-----|-------|--------|----------|---------|")
        for a in sev_alerts:
            ctx = a["context"]
            topic = (a.get("topic") or a.get("title", ""))[:40]
            source = (a.get("source") or "")[:18]
            rank = ctx.get("actionable_rank", "?")
            cat = ctx.get("catalyst_days", "-")
            official = "Y" if a.get("official_confirmation") else ""
            kw = (a.get("catalyst_keyword_hit") or "")[:12]
            lines.append(
                f"| {a['ticker']} | {ctx.get('tier', '?')} | {rank} | {cat} | {topic} | {source} | {official} | {kw} |"
            )
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------


def build_grok_biotech_watch(
    as_of_date: str,
    *,
    snapshots_dir: Path = REPO_ROOT / "data" / "snapshots",
    artifacts_dir: Path = REPO_ROOT / "artifacts",
    send_email: bool = False,
    digest_only: bool = False,
) -> Dict[str, Any]:
    """Build Grok biotech watch artifact.

    Args:
        as_of_date: Snapshot date
        snapshots_dir: Base snapshots directory
        artifacts_dir: Base artifacts directory
        send_email: If True, send email for HIGH alerts
        digest_only: If True, skip Grok search and just build digest from today's alerts
    """
    api_key = os.environ.get("XAI_API_KEY", "")
    if not api_key and not digest_only:
        return {"error": "XAI_API_KEY not set. Get one from console.x.ai"}

    # Build watchlist
    watchlist = build_watchlist(as_of_date, snapshots_dir, artifacts_dir)
    if not watchlist:
        return {"error": f"No watchlist for {as_of_date} — snapshot may be missing"}

    logger.info("Watchlist: %d names", len(watchlist))

    # Load dedup state
    out_dir = artifacts_dir / "grok_watch"
    out_dir.mkdir(parents=True, exist_ok=True)
    dedup_path = out_dir / "dedup_state.json"
    dedup_state = load_dedup_state(dedup_path)
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Search for each ticker
    all_alerts: List[Dict[str, Any]] = []
    emails_sent_this_run = 0

    import time as _time

    for i, (ticker, context) in enumerate(watchlist.items()):
        if digest_only:
            break

        # Rate-limit: pause between requests to avoid 429s
        if i > 0:
            _time.sleep(2)

        # Build query: ticker + catalyst keywords
        cat_terms = "topline OR phase 3 OR FDA OR PDUFA OR adcom OR approval OR CRL OR acquisition OR merger OR buyout"
        query = f"{ticker} biotech ({cat_terms})"

        results = search_grok(query, api_key, max_results=5)
        logger.info("  %s: %d results", ticker, len(results))

        for result in results:
            title = result.get("title", "")
            snippet = result.get("snippet", "")
            source = result.get("source", "")

            # Dedup check
            th = _topic_hash(ticker, f"{title} {snippet}")
            if is_duplicate(th, dedup_state, now_iso):
                continue

            severity = classify_severity(result, context)

            alert = {
                "ticker": ticker,
                "severity": severity,
                "title": title,
                "snippet": snippet,
                "source": source,
                "source_type": result.get("source_type", "unknown"),
                "official_confirmation": result.get("official_confirmation", False),
                "catalyst_keyword_hit": result.get("catalyst_keyword_hit"),
                "topic": result.get("topic", ""),
                "date": result.get("date", ""),
                "topic_hash": th,
                "context": context,
                "as_of_date": as_of_date,
            }
            all_alerts.append(alert)

            # Update dedup state
            dedup_state[th] = now_iso

            # Immediate email for HIGH
            if send_email and severity == "HIGH" and emails_sent_this_run < MAX_IMMEDIATE_EMAILS_PER_HOUR:
                subj, html, text = format_alert_email(alert)
                if send_alert_email(subj, html, text):
                    emails_sent_this_run += 1

    # Prune old dedup entries (> 24h)
    pruned = {}
    for k, v in dedup_state.items():
        if not is_duplicate("_probe_", {k: v}, now_iso):
            # Entry is old enough to prune — but we want to KEEP recent ones
            pass
        pruned[k] = v
    # Actually: keep entries less than 24h old
    fresh_state = {}
    for k, v in dedup_state.items():
        try:
            entry_dt = datetime.fromisoformat(v.replace("Z", "+00:00"))
            now_dt = datetime.fromisoformat(now_iso.replace("Z", "+00:00"))
            if (now_dt - entry_dt).total_seconds() < 86400:
                fresh_state[k] = v
        except (ValueError, TypeError):
            pass
    save_dedup_state(dedup_path, fresh_state)

    # Sort alerts: HIGH first, then MEDIUM, then LOW
    severity_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    all_alerts.sort(key=lambda a: (severity_order.get(a["severity"], 9), a["ticker"]))

    n_high = sum(1 for a in all_alerts if a["severity"] == "HIGH")
    n_med = sum(1 for a in all_alerts if a["severity"] == "MEDIUM")
    n_low = sum(1 for a in all_alerts if a["severity"] == "LOW")

    artifact = {
        "schema": SCHEMA_VERSION,
        "as_of_date": as_of_date,
        "generated_at": now_iso,
        "watchlist_size": len(watchlist),
        "n_alerts": len(all_alerts),
        "n_high": n_high,
        "n_medium": n_med,
        "n_low": n_low,
        "emails_sent": emails_sent_this_run,
        "alerts": all_alerts,
    }

    # Write artifacts
    json_path = out_dir / f"{as_of_date}_alerts.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2, sort_keys=True, default=str)
    logger.info("Wrote %s", json_path)

    md_path = out_dir / f"{as_of_date}_alerts.md"
    md_path.write_text(format_digest_md(all_alerts, as_of_date), encoding="utf-8")
    logger.info("Wrote %s", md_path)

    # Send digest email if requested
    if send_email and digest_only and all_alerts:
        digest_text = format_digest_md(all_alerts, as_of_date)
        send_alert_email(
            f"[DIGEST] Grok Biotech Watch — {as_of_date} ({len(all_alerts)} alerts)",
            f"<pre>{digest_text}</pre>",
            digest_text,
        )

    return artifact


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Grok biotech watch — watchlist-scoped news search with email alerts")
    parser.add_argument(
        "--as-of-date",
        default=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    )
    parser.add_argument(
        "--snapshots-dir",
        type=Path,
        default=REPO_ROOT / "data" / "snapshots",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=REPO_ROOT / "artifacts",
    )
    parser.add_argument(
        "--send-email",
        action="store_true",
        help="Send email for HIGH severity alerts (immediate) and digest",
    )
    parser.add_argument(
        "--digest-only",
        action="store_true",
        help="Skip search, build digest from existing alerts",
    )
    args = parser.parse_args()
    started = time.perf_counter()

    result = build_grok_biotech_watch(
        args.as_of_date,
        snapshots_dir=args.snapshots_dir,
        artifacts_dir=args.artifacts_dir,
        send_email=args.send_email,
        digest_only=args.digest_only,
    )

    if "error" in result:
        logger.error(result["error"])
        try:
            from tools.agent_skill_telemetry import log_agent_run

            log_agent_run(
                "build_grok_biotech_watch",
                f"Grok watch for {args.as_of_date}",
                inputs={"as_of_date": args.as_of_date},
                outputs={"error": result["error"]},
                success=False,
                error=result["error"],
                latency_ms=(time.perf_counter() - started) * 1000,
            )
        except Exception:
            pass
        sys.exit(1)

    logger.info(
        "Watch: %d names, %d alerts (H=%d M=%d L=%d), %d emails",
        result["watchlist_size"],
        result["n_alerts"],
        result["n_high"],
        result["n_medium"],
        result["n_low"],
        result["emails_sent"],
    )
    try:
        from tools.agent_skill_telemetry import log_agent_run

        log_agent_run(
            "build_grok_biotech_watch",
            f"Grok watch for {args.as_of_date}",
            inputs={"as_of_date": args.as_of_date},
            outputs={"n_alerts": result.get("n_alerts"), "n_high": result.get("n_high")},
            success=True,
            latency_ms=(time.perf_counter() - started) * 1000,
        )
    except Exception:
        pass


if __name__ == "__main__":
    main()
