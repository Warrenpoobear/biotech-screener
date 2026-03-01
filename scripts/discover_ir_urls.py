#!/usr/bin/env python3
"""
Discover IR event page URLs and press release RSS feeds for universe tickers.

Strategy:
  1. Batch-lookup company websites via yfinance (reliable, has website field)
  2. Derive IR subdomain from company website
  3. Probe common IR/RSS URL patterns with HEAD/GET requests
  4. Output working URLs to company_calendar_sources.json

Usage:
    python3 scripts/discover_ir_urls.py
    python3 scripts/discover_ir_urls.py --ticker ACAD
    python3 scripts/discover_ir_urls.py --skip-existing --max-tickers 50
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import requests

logger = logging.getLogger(__name__)

_TIMEOUT = 8
_USER_AGENT = "WakeRobinResearch/1.0 (ir-url-discovery)"
_HEADERS = {"User-Agent": _USER_AGENT}

# IR event page path suffixes to probe (ordered by likelihood)
_IR_SUFFIXES = [
    "/events-presentations",
    "/events-and-presentations",
    "/events",
    "/events-presentations/events",
    "/events-and-presentations/events",
    "/news-events/events",
    "/events-and-presentations/events-and-recent",
    "/events-and-presentations/default.aspx",
    "/news-and-events/events-calendar",
    "/event-calendar",
]

# IR subdomain prefixes (ordered by likelihood)
_IR_PREFIXES = ["investors", "investor", "ir"]

# RSS feed path suffix
_RSS_SUFFIX = "/rss/news-releases.xml"


def _head_ok(url: str, session: requests.Session) -> bool:
    """Quick HEAD check. Returns True if status < 400."""
    try:
        resp = session.head(url, timeout=_TIMEOUT, headers=_HEADERS, allow_redirects=True)
        return resp.status_code < 400
    except Exception:
        return False


def _get_is_rss(url: str, session: requests.Session) -> bool:
    """GET and check if response looks like RSS/XML."""
    try:
        resp = session.get(url, timeout=_TIMEOUT, headers=_HEADERS, allow_redirects=True)
        if resp.status_code >= 400:
            return False
        low = resp.text[:500].lower()
        return "<rss" in low or "<feed" in low or "<?xml" in low
    except Exception:
        return False


def _get_yfinance_info(ticker: str) -> Tuple[Optional[str], Optional[str]]:
    """Get company website and IR website from yfinance.

    Returns (website, ir_website).
    """
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info or {}
        website = info.get("website") or None
        ir_website = info.get("irWebsite") or None
        return website, ir_website
    except Exception:
        return None, None


def _derive_ir_domains(website: str, ticker: str) -> List[str]:
    """Derive candidate IR subdomains from company website.

    Given website='https://www.acadia.com', returns:
      ['investors.acadia.com', 'investor.acadia.com', 'ir.acadia.com']
    """
    domains: List[str] = []
    if website:
        parsed = urlparse(website)
        host = parsed.netloc or parsed.path.split("/")[0]
        host = host.replace("www.", "")
        if host:
            for prefix in _IR_PREFIXES:
                domains.append(f"{prefix}.{host}")
    return domains


def discover_for_ticker(
    ticker: str,
    session: requests.Session,
    existing_ir: str = "",
    existing_rss: str = "",
) -> Dict[str, str]:
    """Discover IR event page URL and RSS feed URL for a single ticker."""
    result: Dict[str, str] = {}

    if existing_ir and existing_rss:
        return {"ir_url": existing_ir, "pr_rss_url": existing_rss}

    # Step 1: Get company website + IR website from yfinance
    website, ir_website = _get_yfinance_info(ticker)

    # If yfinance gave us a direct IR website, derive domains from it too
    ir_domains = _derive_ir_domains(website or "", ticker)

    # Also add the IR website domain directly if available
    if ir_website:
        parsed = urlparse(ir_website)
        ir_host = parsed.netloc or ""
        if ir_host:
            ir_host = ir_host.replace("www.", "")
            # Insert at front — highest priority
            if ir_host not in ir_domains:
                ir_domains.insert(0, ir_host)

    if not ir_domains:
        return result

    # Step 3: Find IR event page
    if not existing_ir:
        for domain in ir_domains:
            for suffix in _IR_SUFFIXES:
                url = f"https://{domain}{suffix}"
                if _head_ok(url, session):
                    result["ir_url"] = url
                    break
            if "ir_url" in result:
                break

    # Step 4: Find RSS feed on the IR domain that worked (or all candidates)
    if not existing_rss:
        check_domains = []
        if result.get("ir_url"):
            found_host = urlparse(result["ir_url"]).netloc
            check_domains.append(found_host)
        elif existing_ir:
            found_host = urlparse(existing_ir).netloc
            check_domains.append(found_host)
        check_domains.extend(d for d in ir_domains if d not in check_domains)

        for domain in check_domains[:4]:
            rss_url = f"https://{domain}{_RSS_SUFFIX}"
            if _get_is_rss(rss_url, session):
                result["pr_rss_url"] = rss_url
                break

    if existing_ir:
        result["ir_url"] = existing_ir
    if existing_rss:
        result["pr_rss_url"] = existing_rss

    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Discover IR URLs for universe tickers")
    parser.add_argument("--data-dir", default="production_data")
    parser.add_argument("--output", default="production_data/company_calendar_sources.json")
    parser.add_argument("--max-tickers", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--ticker", type=str, default="")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    data_dir = Path(args.data_dir)
    universe = json.loads((data_dir / "universe.json").read_text())

    output_path = Path(args.output)
    existing_sources: Dict[str, Dict[str, str]] = {}
    if output_path.exists():
        data = json.loads(output_path.read_text())
        existing_sources = data.get("sources", {})

    tickers_to_process = []
    for row in universe:
        tk = row.get("ticker", "").strip().upper()
        if not tk or tk.startswith("_"):
            continue
        if args.ticker and tk != args.ticker.upper():
            continue
        if args.skip_existing and tk in existing_sources:
            has_both = existing_sources[tk].get("ir_url") and existing_sources[tk].get("pr_rss_url")
            if has_both:
                continue
        tickers_to_process.append(tk)

    if args.max_tickers > 0:
        tickers_to_process = tickers_to_process[:args.max_tickers]

    print(f"Processing {len(tickers_to_process)} tickers (existing sources: {len(existing_sources)})...")

    session = requests.Session()
    found_ir = 0
    found_rss = 0
    processed = 0
    no_website = 0

    for tk in tickers_to_process:
        processed += 1
        existing = existing_sources.get(tk, {})
        existing_ir = existing.get("ir_url", "")
        existing_rss = existing.get("pr_rss_url", "")

        try:
            urls = discover_for_ticker(tk, session, existing_ir, existing_rss)
        except Exception as e:
            logger.error("Error for %s: %s", tk, e)
            urls = {}

        if not urls:
            no_website += 1
            print(f"  [{processed}/{len(tickers_to_process)}] {tk}: MISS")
            continue

        new_ir = urls.get("ir_url") and not existing_ir
        new_rss = urls.get("pr_rss_url") and not existing_rss
        if new_ir:
            found_ir += 1
        if new_rss:
            found_rss += 1

        if urls.get("ir_url") or urls.get("pr_rss_url"):
            existing_sources[tk] = {**existing_sources.get(tk, {}), **urls}

        parts = []
        if urls.get("ir_url"):
            parts.append(f"ir={'NEW' if new_ir else 'exist'}")
        if urls.get("pr_rss_url"):
            parts.append(f"rss={'NEW' if new_rss else 'exist'}")
        if not parts:
            parts.append("MISS")
        print(f"  [{processed}/{len(tickers_to_process)}] {tk}: {', '.join(parts)}")

    # Write output
    output = {
        "_schema": "company_calendar_sources.v1",
        "_updated_at": "2026-02-28",
        "sources": dict(sorted(existing_sources.items())),
    }
    output_path.write_text(json.dumps(output, indent=2, sort_keys=False) + "\n")

    total_ir = sum(1 for v in existing_sources.values() if v.get("ir_url"))
    total_rss = sum(1 for v in existing_sources.values() if v.get("pr_rss_url"))

    print(f"\n{'='*60}")
    print(f"DISCOVERY SUMMARY")
    print(f"{'='*60}")
    print(f"Tickers processed:  {processed}")
    print(f"No website found:   {no_website}")
    print(f"New IR URLs found:  {found_ir}")
    print(f"New RSS URLs found: {found_rss}")
    print(f"Total in sources:   {len(existing_sources)} tickers")
    print(f"  with ir_url:      {total_ir}")
    print(f"  with pr_rss_url:  {total_rss}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
