#!/usr/bin/env python3
"""
build_pit_financials.py - Build point-in-time financial fact store from SEC EDGAR.

Downloads ALL historical XBRL facts (not just latest) for each ticker in
universe.json, preserving the 'filed' date so that backtests can query
financials as-known-on any historical date without look-ahead bias.

Output: production_data/pit_financials/{ticker}.json

Usage:
    python tools/build_pit_financials.py                  # full universe
    python tools/build_pit_financials.py --ticker ALNY    # single ticker
    python tools/build_pit_financials.py --force           # re-download all
    python tools/build_pit_financials.py --max-age-days 7  # re-download files older than 7 days
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PRODUCTION_DATA = PROJECT_ROOT / "production_data"
PIT_DIR = PRODUCTION_DATA / "pit_financials"
UNIVERSE_PATH = PRODUCTION_DATA / "universe.json"
SEC_CACHE_DIR = PROJECT_ROOT / "wake_robin_data_pipeline" / "cache" / "sec"

# ---------------------------------------------------------------------------
# SEC EDGAR config
# ---------------------------------------------------------------------------
USER_AGENT = os.environ.get(
    "SEC_USER_AGENT",
    "biotech-screener research@example.com",
)
RATE_LIMIT_DELAY = 0.22  # ~4.5 req/sec, safely under SEC 10 req/sec limit

# ---------------------------------------------------------------------------
# XBRL tag definitions: (normalized_name, primary_tag, fallback_tags, unit, is_instant)
#
# is_instant: True for balance-sheet items (point-in-time), False for
#             income-statement / cash-flow items (duration).
# ---------------------------------------------------------------------------
XBRL_TAGS = [
    # Balance sheet (instant)
    ("cash", "CashAndCashEquivalentsAtCarryingValue", ["CashAndCashEquivalents", "Cash"], "USD", True),
    ("assets", "Assets", [], "USD", True),
    ("liabilities", "Liabilities", [], "USD", True),
    ("stockholders_equity", "StockholdersEquity", [], "USD", True),
    (
        "shares_outstanding",
        "CommonStockSharesOutstanding",
        ["CommonStockSharesIssued", "WeightedAverageNumberOfShareOutstandingBasicAndDiluted"],
        "shares",
        True,
    ),
    ("long_term_debt", "LongTermDebt", ["LongTermDebtNoncurrent", "ConvertibleDebt"], "USD", True),
    (
        "short_term_investments",
        "MarketableSecuritiesCurrent",
        ["ShortTermInvestments", "AvailableForSaleSecuritiesCurrent", "MarketableSecurities"],
        "USD",
        True,
    ),
    # Income statement / cash flow (duration)
    (
        "revenue",
        "Revenues",
        [
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "SalesRevenueNet",
            "RevenueFromContractWithCustomerIncludingAssessedTax",
        ],
        "USD",
        False,
    ),
    ("operating_expenses", "OperatingExpenses", ["CostsAndExpenses"], "USD", False),
    ("research_and_development", "ResearchAndDevelopmentExpense", [], "USD", False),
    ("net_income", "NetIncomeLoss", ["ProfitLoss"], "USD", False),
    (
        "operating_cash_flow",
        "NetCashProvidedByUsedInOperatingActivities",
        ["NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
        "USD",
        False,
    ),
]

# IFRS fallback mappings: ifrs_tag -> same normalized name
IFRS_TAGS = {
    "cash": ["CashAndCashEquivalents", "Cash"],
    "assets": ["Assets"],
    "liabilities": ["Liabilities"],
    "stockholders_equity": ["Equity", "EquityAttributableToOwnersOfParent"],
    "revenue": ["Revenue", "RevenueFromContractsWithCustomers"],
    "operating_expenses": ["OtherExpenseByNature", "AdministrativeExpense"],
    "research_and_development": ["ResearchAndDevelopmentExpense"],
    "net_income": ["ProfitLoss", "ProfitLossAttributableToOwnersOfParent"],
    "operating_cash_flow": ["CashFlowsFromUsedInOperatingActivities"],
    "shares_outstanding": ["IssuedCapital"],
    "long_term_debt": ["NoncurrentLiabilities", "BorrowingsNoncurrent"],
    "short_term_investments": ["OtherCurrentFinancialAssets", "CurrentFinancialAssets"],
}

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CIK resolution
# ---------------------------------------------------------------------------
def load_cik_from_cache(ticker: str) -> Optional[str]:
    """Try to load CIK from the existing sec_collector cache."""
    cache_path = SEC_CACHE_DIR / f"{ticker}_cik_mapping.json"
    if cache_path.exists():
        with open(cache_path) as f:
            data = json.load(f)
            cik = data.get("cik")
            if cik:
                return cik
    return None


def load_cik_from_universe(ticker: str, universe: list[dict]) -> Optional[str]:
    """Try to load CIK from universe.json entry."""
    for entry in universe:
        if entry.get("ticker", "").upper() == ticker.upper():
            cik = entry.get("cik")
            if cik:
                return cik
    return None


_company_tickers_cache: Optional[dict] = None


def resolve_cik_from_sec(ticker: str, session: requests.Session) -> Optional[str]:
    """Resolve CIK via SEC company_tickers.json (cached in memory)."""
    global _company_tickers_cache
    if _company_tickers_cache is None:
        url = "https://www.sec.gov/files/company_tickers.json"
        resp = session.get(url, headers={"User-Agent": USER_AGENT}, timeout=15)
        resp.raise_for_status()
        _company_tickers_cache = resp.json()
        time.sleep(RATE_LIMIT_DELAY)

    ticker_upper = ticker.upper()
    for entry in _company_tickers_cache.values():
        if entry.get("ticker", "").upper() == ticker_upper:
            return str(entry["cik_str"]).zfill(10)
    return None


def resolve_cik(ticker: str, universe: list[dict], session: requests.Session) -> Optional[str]:
    """Resolve CIK using cache -> universe -> SEC API (in priority order)."""
    cik = load_cik_from_cache(ticker)
    if cik:
        return cik
    cik = load_cik_from_universe(ticker, universe)
    if cik:
        return cik
    cik = resolve_cik_from_sec(ticker, session)
    return cik


# ---------------------------------------------------------------------------
# XBRL fact extraction
# ---------------------------------------------------------------------------
def extract_facts_for_tag(
    facts_data: dict,
    tag_name: str,
    unit_type: str,
    namespace: str,
) -> list[dict]:
    """Extract all historical facts for a single XBRL tag.

    Returns list of dicts with keys: end, val, filed, form, accn, start (if duration).
    """
    ns_data = facts_data.get("facts", {}).get(namespace, {})
    if tag_name not in ns_data:
        return []

    metric_data = ns_data[tag_name]
    units_data = metric_data.get("units", {})

    # Find the right unit bucket
    values = None
    if unit_type in units_data:
        values = units_data[unit_type]
    elif unit_type == "shares":
        # Try common share unit names
        for candidate in ["shares", "pure"]:
            if candidate in units_data:
                values = units_data[candidate]
                break
    elif "USD" in units_data:
        values = units_data["USD"]

    if not values:
        # Take first available unit
        if units_data:
            values = list(units_data.values())[0]
        else:
            return []

    results = []
    for v in values:
        fact = {
            "end": v.get("end"),
            "val": v.get("val"),
            "filed": v.get("filed"),
            "form": v.get("form"),
            "accn": v.get("accn"),
        }
        # Duration items have a start date
        if "start" in v:
            fact["start"] = v["start"]
        results.append(fact)

    return results


def extract_all_facts(facts_data: dict) -> dict[str, list[dict]]:
    """Extract all PIT facts from the EDGAR companyfacts response.

    Returns a dict of normalized_name -> list of fact dicts.
    Tries us-gaap first, then ifrs-full for each metric.
    """
    result: dict[str, list[dict]] = {}

    for normalized_name, primary_tag, fallbacks, unit_type, _is_instant in XBRL_TAGS:
        # Try us-gaap namespace first
        facts = extract_facts_for_tag(facts_data, primary_tag, unit_type, "us-gaap")
        if not facts:
            for fb_tag in fallbacks:
                facts = extract_facts_for_tag(facts_data, fb_tag, unit_type, "us-gaap")
                if facts:
                    break

        # Try ifrs-full namespace if us-gaap yielded nothing
        if not facts and normalized_name in IFRS_TAGS:
            for ifrs_tag in IFRS_TAGS[normalized_name]:
                facts = extract_facts_for_tag(facts_data, ifrs_tag, unit_type, "ifrs-full")
                if facts:
                    break

        if facts:
            result[normalized_name] = facts

    return result


# ---------------------------------------------------------------------------
# Raw API response cache
# ---------------------------------------------------------------------------
def get_raw_cache_path(cik: str) -> Path:
    """Cache path for raw EDGAR companyfacts response."""
    cache_dir = PROJECT_ROOT / "wake_robin_data_pipeline" / "cache" / "sec" / "companyfacts"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"CIK{cik}.json"


def fetch_companyfacts(cik: str, session: requests.Session, force: bool = False) -> Optional[dict]:
    """Fetch companyfacts from EDGAR with raw response caching.

    Returns parsed JSON or None on failure.
    """
    cache_path = get_raw_cache_path(cik)

    if not force and cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    try:
        resp = session.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        # Cache the raw response
        with open(cache_path, "w") as f:
            json.dump(data, f)

        return data
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            logger.warning(f"CIK {cik} not found in EDGAR (404)")
        else:
            logger.warning(f"HTTP error for CIK {cik}: {e}")
        return None
    except Exception as e:
        logger.warning(f"Error fetching CIK {cik}: {e}")
        return None


# ---------------------------------------------------------------------------
# Per-ticker pipeline
# ---------------------------------------------------------------------------
def build_pit_record(
    ticker: str,
    cik: str,
    session: requests.Session,
    force: bool = False,
) -> Optional[dict]:
    """Build PIT financial record for a single ticker.

    Returns the record dict, or None on failure.
    """
    facts_data = fetch_companyfacts(cik, session, force=force)
    if facts_data is None:
        return None

    all_facts = extract_all_facts(facts_data)

    record = {
        "ticker": ticker,
        "cik": cik,
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "facts": all_facts,
    }
    return record


def save_pit_record(record: dict, output_dir: Path) -> Path:
    """Save PIT record to disk."""
    ticker = record["ticker"]
    out_path = output_dir / f"{ticker}.json"
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)
    return out_path


def is_fresh(path: Path, max_age_days: Optional[float], now: Optional[float] = None) -> bool:
    """Return True if a cached PIT record should be reused instead of re-fetched.

    With max_age_days=None any existing file counts as fresh (the historical
    behaviour). Otherwise the file must also be younger than max_age_days, so a
    scheduled refresh actually picks up newly filed 10-Qs/10-Ks. Facts carry
    their SEC 'filed' date, so re-fetching never introduces look-ahead.
    """
    if not path.exists():
        return False
    if max_age_days is None:
        return True
    age_s = (time.time() if now is None else now) - path.stat().st_mtime
    return age_s < max_age_days * 86400


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Build PIT financial fact store from SEC EDGAR")
    parser.add_argument("--ticker", type=str, default=None, help="Process single ticker (default: full universe)")
    parser.add_argument("--force", action="store_true", help="Re-download even if cached")
    parser.add_argument(
        "--max-age-days",
        type=float,
        default=None,
        help="Re-download tickers whose cached record is older than this many days (default: never)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Output directory (default: production_data/pit_financials/)"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    output_dir = Path(args.output_dir) if args.output_dir else PIT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load universe
    if not UNIVERSE_PATH.exists():
        logger.error(f"Universe file not found: {UNIVERSE_PATH}")
        sys.exit(1)

    with open(UNIVERSE_PATH) as f:
        universe = json.load(f)

    # Determine ticker list
    if args.ticker:
        tickers = [args.ticker.upper()]
    else:
        tickers = [e["ticker"] for e in universe if e.get("status") == "active"]
        if not tickers:
            tickers = [e["ticker"] for e in universe]

    total = len(tickers)
    logger.info(f"Processing {total} tickers -> {output_dir}")

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    success = 0
    skipped = 0
    failed = 0
    no_cik = 0

    for i, ticker in enumerate(tickers, 1):
        # Skip if already exists and fresh enough (unless --force)
        out_path = output_dir / f"{ticker}.json"
        refetch = args.force or out_path.exists()
        if not args.force and is_fresh(out_path, args.max_age_days):
            skipped += 1
            if i % 50 == 0 or i == total:
                print(f"[{i}/{total}] {ticker} ... skipped (cached)")
            continue

        # Resolve CIK
        cik = resolve_cik(ticker, universe, session)
        if not cik:
            logger.warning(f"[{i}/{total}] {ticker} ... no CIK found")
            no_cik += 1
            continue

        # Fetch and build
        # A stale record must bypass the raw companyfacts cache too, or the
        # rebuild would just re-read the same old EDGAR response.
        record = build_pit_record(ticker, cik, session, force=refetch)
        if record is None:
            logger.warning(f"[{i}/{total}] {ticker} ... EDGAR fetch failed")
            failed += 1
            time.sleep(RATE_LIMIT_DELAY)
            continue

        save_pit_record(record, output_dir)
        n_fields = len(record["facts"])
        n_total_facts = sum(len(v) for v in record["facts"].values())
        print(f"[{i}/{total}] {ticker} ... {n_fields} fields, {n_total_facts} facts")
        success += 1

        # Rate limit: ~4.5 req/sec
        time.sleep(RATE_LIMIT_DELAY)

    print(f"\nDone: {success} downloaded, {skipped} cached, {failed} failed, {no_cik} no CIK")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
