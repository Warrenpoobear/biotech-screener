#!/usr/bin/env python3
"""
collect_financial_data.py - Collect Financial Data from SEC EDGAR

Fetches 10-K/10-Q filings and extracts key financial metrics.

Usage:
    python collect_financial_data.py --universe production_data/universe.json
"""

import argparse
import json
import logging
import time
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_DURATION_NAMES = {"R&D", "NetIncome", "Revenue", "CFO", "OperatingExpenses", "InterestExpense", "COGS"}

import requests

# Maximum age in days for financial data to be considered valid.
# 540 days (18 months) to cover annual-only filers (e.g. 20-F foreign
# private issuers) whose latest data can be 15+ months old at collection.
MAX_DATA_AGE_DAYS = 540


def _fact_duration_days(entry: Dict) -> Optional[int]:
    """Inclusive period length. None when the fact has no start date."""
    start, end = entry.get("start"), entry.get("end")
    if not start or not end:
        return None
    try:
        return (datetime.fromisoformat(str(end)[:10]) - datetime.fromisoformat(str(start)[:10])).days
    except ValueError:
        return None


def _period_months_from_fact(entry: Dict) -> Optional[int]:
    """Map an XBRL duration to 3, 6, 9, or 12 months."""
    days = _fact_duration_days(entry)
    if days is None:
        return None
    if days < 150:
        return 3
    if days < 240:
        return 6
    if days < 330:
        return 9
    return 12


def _select_fresh_fact(values: List[Dict], prefer_longest: bool = False) -> Optional[Dict]:
    """Latest fresh period end. Duration facts prefer the longest period at that end.

    A 10-Q files a quarter and a year-to-date fact with the same ``end``.
    Taking whichever row the API listed first mixes those two scales.
    """
    fresh = [entry for entry in values if is_data_fresh(entry.get("end"))]
    if not fresh:
        return None
    latest_end = max((entry.get("end") or "") for entry in fresh)
    at_end = [entry for entry in fresh if entry.get("end") == latest_end]
    if prefer_longest:
        dated = [entry for entry in at_end if _fact_duration_days(entry) is not None]
        if dated:
            return max(dated, key=lambda entry: _fact_duration_days(entry) or 0)
    return at_end[0]


def _store_fact(financial_data: Dict, friendly_name: str, entry: Dict) -> None:
    financial_data[friendly_name] = entry.get("val")
    financial_data[f"{friendly_name}_date"] = entry.get("end")
    months = _period_months_from_fact(entry)
    if months and friendly_name in _DURATION_NAMES:
        financial_data[f"{friendly_name}_period_months"] = months


def is_data_fresh(date_str: str, max_age_days: int = MAX_DATA_AGE_DAYS) -> bool:
    """Check if data date is within acceptable age range."""
    if not date_str:
        return False
    try:
        data_date = datetime.fromisoformat(date_str)
        age_days = (datetime.now() - data_date).days
        return age_days <= max_age_days
    except (ValueError, TypeError):
        return False


_CIK_CACHE: Optional[Dict[str, str]] = None


def _load_cik_cache() -> Dict[str, str]:
    """Load and cache the SEC ticker→CIK mapping (single request)."""
    global _CIK_CACHE
    if _CIK_CACHE is not None:
        return _CIK_CACHE

    url = "https://www.sec.gov/files/company_tickers.json"
    headers = {"User-Agent": "WakeRobinCapital research@wakerobincapital.com"}

    try:
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code == 200:
            data = response.json()
            _CIK_CACHE = {
                entry["ticker"].upper(): str(entry.get("cik_str")).zfill(10)
                for entry in data.values()
                if entry.get("ticker")
            }
        else:
            _CIK_CACHE = {}
    except Exception:
        _CIK_CACHE = {}

    return _CIK_CACHE


def get_cik_from_ticker(ticker: str) -> Optional[str]:
    """Get CIK (Central Index Key) from ticker using cached SEC mapping."""
    cache = _load_cik_cache()
    return cache.get(ticker.upper())


def get_company_facts(cik: str, ticker: str) -> Optional[Dict]:
    """Get company facts from SEC EDGAR API"""

    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    headers = {"User-Agent": "WakeRobinCapital research@wakerobincapital.com", "Accept-Encoding": "gzip, deflate"}

    try:
        response = requests.get(url, headers=headers, timeout=30)

        if response.status_code == 200:
            data = response.json()
            all_facts = data.get("facts", {})
            facts = all_facts.get("us-gaap", {})
            ifrs_facts = all_facts.get("ifrs-full", {})

            # Key metrics to extract (single XBRL tag -> friendly name)
            metrics = {
                "Assets": "Assets",
                "AssetsCurrent": "CurrentAssets",
                "Liabilities": "Liabilities",
                "LiabilitiesCurrent": "CurrentLiabilities",
                "StockholdersEquity": "ShareholdersEquity",
                "CashAndCashEquivalentsAtCarryingValue": "Cash",
                "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents": "CashRestricted",
                "CashEquivalentsAtCarryingValue": "CashEquivalentsOnly",
                "MarketableSecuritiesCurrent": "MarketableSecurities",
                # MarketableSecurities (non-current tag) is a FALLBACK only — used when
                # MarketableSecuritiesCurrent is absent. Adding both would double-count
                # filers like ABOS that report under both tags.
                # Handled in post-processing below, not here as a separate metrics key.
                "ShortTermInvestments": "ShortTermInvestments",
                # Fallback tags for STI — some filers (e.g. EWTX) report under these
                # instead of ShortTermInvestments. AvailableForSaleSecuritiesCurrent and
                # AvailableForSaleSecuritiesDebtSecuritiesCurrent are already fetched
                # separately; DebtSecuritiesAvailableForSaleCurrent is the ASC 320 (2018+)
                # replacement used by some biotechs for their treasury note holdings.
                "DebtSecuritiesAvailableForSaleCurrent": "DebtSecuritiesAvailableForSaleCurrent",
                "AvailableForSaleSecuritiesCurrent": "AvailableForSaleSecurities",
                "AvailableForSaleSecuritiesDebtSecuritiesCurrent": "AvailableForSaleDebtCurrent",
                "CostOfRevenue": "COGS",
                "ResearchAndDevelopmentExpense": "R&D",
                "NetIncomeLoss": "NetIncome",
                "LongTermDebt": "LongTermDebt",
                "LongTermDebtCurrent": "LongTermDebtCurrent",
                "ConvertibleNotesPayable": "ConvertibleDebt",
            }

            # IFRS equivalents: ifrs-full tag -> same friendly name
            ifrs_metrics = {
                "Assets": "Assets",
                "CurrentAssets": "CurrentAssets",
                "Liabilities": "Liabilities",
                "CurrentLiabilities": "CurrentLiabilities",
                "Equity": "ShareholdersEquity",
                "CashAndCashEquivalents": "Cash",
                "NoncurrentFinancialAssets": "MarketableSecurities",
                "CostOfSales": "COGS",
                "ResearchAndDevelopmentExpense": "R&D",
                "ProfitLoss": "NetIncome",
                "NoncurrentPortionOfNoncurrentBorrowings": "LongTermDebt",
                "CurrentPortionOfNoncurrentBorrowings": "LongTermDebtCurrent",
            }

            # Metrics with fallback tags (try in order, use first found)
            metrics_with_fallback = {
                "MarketableSecurities": [
                    # Fallback: use broad MarketableSecurities tag ONLY when
                    # MarketableSecuritiesCurrent returned nothing. This fires
                    # AFTER metrics_with_fallback processing but MarketableSecuritiesCurrent
                    # from the base metrics dict takes priority (it's fetched first).
                    # The aggregation below uses max() to avoid double-counting.
                    "MarketableSecurities",
                ],
                "Revenue": [
                    "RevenueFromContractWithCustomerExcludingAssessedTax",  # ASC 606 (2018+)
                    "Revenues",  # Legacy
                    "SalesRevenueNet",  # Alternative
                    "SalesRevenueGoodsNet",  # Product-specific
                ],
                "CFO": [
                    "NetCashProvidedByUsedInOperatingActivities",  # Standard
                    "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",  # Alternative
                    "CashFlowsFromUsedInOperatingActivities",  # IFRS-style
                ],
                "OperatingExpenses": [
                    "OperatingExpenses",  # Standard
                    "CostsAndExpenses",  # Alternative (includes COGS)
                    "OperatingCostsAndExpenses",  # Alternative
                ],
                "InterestExpense": [
                    "InterestExpense",  # Standard
                    "InterestAndDebtExpense",  # Alternative
                    "InterestExpenseDebt",  # Specific to debt
                ],
            }

            # IFRS fallback tags for multi-tag metrics
            ifrs_metrics_with_fallback = {
                "Revenue": [
                    "Revenue",  # IFRS standard
                    "RevenueFromContractsWithCustomers",
                ],
                "CFO": [
                    "CashFlowsFromUsedInOperatingActivities",
                ],
                "OperatingExpenses": [
                    "AdministrativeExpense",
                ],
                "InterestExpense": [
                    "InterestExpenseOnBorrowings",
                    "FinanceCosts",
                ],
            }

            financial_data = {"ticker": ticker, "cik": cik}

            # Extract most recent values (simple metrics) with staleness filter
            for gaap_key, friendly_name in metrics.items():
                if gaap_key in facts:
                    units = facts[gaap_key].get("units", {})

                    if "USD" in units:
                        chosen = _select_fresh_fact(
                            units["USD"],
                            prefer_longest=friendly_name in _DURATION_NAMES,
                        )
                        if chosen is not None:
                            _store_fact(financial_data, friendly_name, chosen)

            # Extract metrics with fallback. Across tags, keep the latest end,
            # and for duration facts the longest period at that end.
            for friendly_name, tag_list in metrics_with_fallback.items():
                candidates = []
                for gaap_key in tag_list:
                    if gaap_key in facts:
                        units = facts[gaap_key].get("units", {})
                        if "USD" in units:
                            candidates.extend(units["USD"])
                chosen = _select_fresh_fact(candidates, prefer_longest=friendly_name in _DURATION_NAMES)
                if chosen is not None:
                    _store_fact(financial_data, friendly_name, chosen)

            # --- IFRS fallback for foreign filers (AZN, SNY, ARGX, BNTX, etc.) ---
            # If us-gaap yielded very few fields, try ifrs-full tags for missing metrics.
            if ifrs_facts:
                # Simple IFRS metrics — only fill gaps (don't overwrite US-GAAP)
                for ifrs_key, friendly_name in ifrs_metrics.items():
                    if friendly_name not in financial_data and ifrs_key in ifrs_facts:
                        units = ifrs_facts[ifrs_key].get("units", {})
                        for currency in (
                            "USD",
                            "EUR",
                            "GBP",
                            "DKK",
                            "CHF",
                            "AUD",
                            "CAD",
                            "SEK",
                            "NOK",
                            "JPY",
                            "KRW",
                            "CNY",
                            "ILS",
                        ):
                            if currency in units:
                                values = units[currency]
                                recent = sorted(values, key=lambda x: x.get("end", ""), reverse=True)
                                for entry in recent:
                                    entry_date = entry.get("end")
                                    if is_data_fresh(entry_date):
                                        financial_data[friendly_name] = entry.get("val")
                                        financial_data[f"{friendly_name}_date"] = entry_date
                                        if currency != "USD":
                                            financial_data[f"{friendly_name}_currency"] = currency
                                        break
                                if friendly_name in financial_data:
                                    break

                # IFRS multi-tag fallbacks — only fill gaps
                for friendly_name, tag_list in ifrs_metrics_with_fallback.items():
                    if friendly_name in financial_data:
                        continue
                    best_val, best_date = None, None
                    for ifrs_key in tag_list:
                        if ifrs_key in ifrs_facts:
                            units = ifrs_facts[ifrs_key].get("units", {})
                            for currency in (
                                "USD",
                                "EUR",
                                "GBP",
                                "DKK",
                                "CHF",
                                "AUD",
                                "CAD",
                                "SEK",
                                "NOK",
                                "JPY",
                                "KRW",
                                "CNY",
                                "ILS",
                            ):
                                if currency in units:
                                    values = units[currency]
                                    recent = sorted(values, key=lambda x: x.get("end", ""), reverse=True)
                                    for entry in recent:
                                        val, dt = entry.get("val"), entry.get("end")
                                        if is_data_fresh(dt):
                                            if best_date is None or (dt and dt > best_date):
                                                best_val, best_date = val, dt
                                            break
                                    if best_val is not None:
                                        break
                    if best_val is not None:
                        financial_data[friendly_name] = best_val
                        financial_data[f"{friendly_name}_date"] = best_date

            # Derive Liabilities = Assets - Equity when direct tag absent
            # Many large-cap filers (JAZZ, AMGN, GILD, ARWR, etc.) only report
            # LiabilitiesAndStockholdersEquity and ShareholdersEquity but not Liabilities.
            if financial_data.get("Liabilities") is None:
                assets_val = financial_data.get("Assets")
                equity_val = financial_data.get("ShareholdersEquity")
                if assets_val is not None and equity_val is not None:
                    derived = assets_val - equity_val
                    if derived >= 0:  # only write if non-negative (sanity check)
                        financial_data["Liabilities"] = derived
                        financial_data["Liabilities_date"] = financial_data.get("Assets_date")
                        financial_data["Liabilities_source"] = "derived:Assets-Equity"
            # Aggregate TotalDebt from components if not already present
            if (
                financial_data.get("LongTermDebt")
                or financial_data.get("LongTermDebtCurrent")
                or financial_data.get("ConvertibleDebt")
            ):
                total_debt = 0
                debt_components = []

                if financial_data.get("LongTermDebt"):
                    total_debt += financial_data["LongTermDebt"]
                    debt_components.append("LongTermDebt")
                if financial_data.get("LongTermDebtCurrent"):
                    total_debt += financial_data["LongTermDebtCurrent"]
                    debt_components.append("LongTermDebtCurrent")
                if financial_data.get("ConvertibleDebt"):
                    total_debt += financial_data["ConvertibleDebt"]
                    debt_components.append("ConvertibleDebt")

                if total_debt > 0:
                    financial_data["TotalDebt"] = total_debt
                    financial_data["TotalDebt_components"] = debt_components

            # Cash fallback: CashAndCashEquivalentsAtCarryingValue → CashRestricted → CashEquivalentsOnly
            cash = financial_data.get("Cash", 0) or 0
            if cash == 0:
                cash = financial_data.get("CashRestricted", 0) or 0
                if cash > 0:
                    financial_data["Cash"] = cash
                    financial_data["Cash_date"] = financial_data.get("CashRestricted_date")
                    financial_data["Cash_source"] = "CashCashEquivalentsRestricted"
            if cash == 0:
                cash = financial_data.get("CashEquivalentsOnly", 0) or 0
                if cash > 0:
                    financial_data["Cash"] = cash
                    financial_data["Cash_date"] = financial_data.get("CashEquivalentsOnly_date")
                    financial_data["Cash_source"] = "CashEquivalentsAtCarryingValue"

            # Aggregate CashAndSecurities (Cash + non-cash liquid securities)
            # DEDUP NOTE: ShortTermInvestments, MarketableSecurities,
            # AvailableForSaleSecurities, AvailableForSaleDebtCurrent, and
            # DebtSecuritiesAvailableForSaleCurrent all represent the SAME
            # treasury/bond portfolio for many filers — just different XBRL tags.
            # Adding them causes double/triple-counting (e.g. ABUS: STI+AvailDebt).
            # Fix: take MAX across all investment tags as one component.
            mkt_sec = financial_data.get("MarketableSecurities", 0) or 0
            st_inv = financial_data.get("ShortTermInvestments", 0) or 0
            avail = financial_data.get("AvailableForSaleSecurities", 0) or 0
            avail_debt = financial_data.get("AvailableForSaleDebtCurrent", 0) or 0
            debt_avail_current = financial_data.get("DebtSecuritiesAvailableForSaleCurrent", 0) or 0
            investment_component = max(st_inv, mkt_sec, avail, avail_debt, debt_avail_current)
            total_liquid = cash + investment_component
            if total_liquid > 0:
                financial_data["CashAndSecurities"] = total_liquid
                # Use most recent date from components
                dates = [
                    financial_data.get("Cash_date"),
                    financial_data.get("MarketableSecurities_date"),
                    financial_data.get("ShortTermInvestments_date"),
                    financial_data.get("AvailableForSaleDebtCurrent_date"),
                ]
                dates = [d for d in dates if d]
                if dates:
                    financial_data["CashAndSecurities_date"] = max(dates)

            financial_data["collected_at"] = date.today().isoformat()
            return financial_data

        return None

    except Exception:
        logger.exception("EDGAR company-facts fetch failed for %s", ticker)
        return None


def get_yfinance_fallback(ticker: str) -> Optional[Dict]:
    """Fallback: fetch financials from yfinance when EDGAR has no data.

    Covers tickers with no CIK (pre-IPO tickers, foreign filers without
    EDGAR filings, etc.).  Returns a dict matching the EDGAR output schema
    or None on failure.
    """
    try:
        import yfinance as yf
    except ImportError:
        return None

    try:
        t = yf.Ticker(ticker)
        bs = t.balance_sheet
        fs = t.financials  # income statement
        cf = t.cashflow

        if bs is None or bs.empty:
            return None

        data: Dict = {"ticker": ticker, "cik": "", "source": "yfinance"}

        def _latest(df, *labels):
            """Return most recent non-null value for first matching label."""
            for label in labels:
                if label in df.index:
                    row = df.loc[label].dropna()
                    if not row.empty:
                        return float(row.iloc[0]), (
                            str(row.index[0].date()) if hasattr(row.index[0], "date") else str(row.index[0])
                        )
            return None, None

        # Balance sheet
        val, dt = _latest(bs, "Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments")
        if val is not None:
            data["Cash"] = val
            data["Cash_date"] = dt

        val, dt = _latest(bs, "Other Short Term Investments", "Short Term Investments")
        if val is not None:
            data["ShortTermInvestments"] = val

        val, dt = _latest(bs, "Total Assets")
        if val is not None:
            data["Assets"] = val
            data["Assets_date"] = dt

        val, dt = _latest(bs, "Current Assets")
        if val is not None:
            data["CurrentAssets"] = val

        val, dt = _latest(bs, "Total Liabilities Net Minority Interest", "Total Liabilities")
        if val is not None:
            data["Liabilities"] = val

        val, dt = _latest(bs, "Current Liabilities")
        if val is not None:
            data["CurrentLiabilities"] = val

        val, dt = _latest(bs, "Stockholders Equity", "Total Equity Gross Minority Interest")
        if val is not None:
            data["ShareholdersEquity"] = val

        val, dt = _latest(bs, "Long Term Debt")
        if val is not None:
            data["LongTermDebt"] = val

        # Income statement
        if fs is not None and not fs.empty:
            val, dt = _latest(fs, "Net Income", "Net Income Common Stockholders")
            if val is not None:
                data["NetIncome"] = val
                data["NetIncome_date"] = dt

            val, dt = _latest(fs, "Total Revenue", "Operating Revenue")
            if val is not None:
                data["Revenue"] = val
                data["Revenue_date"] = dt

            val, dt = _latest(fs, "Research And Development")
            if val is not None:
                data["R&D"] = val

        # Cash flow
        if cf is not None and not cf.empty:
            val, dt = _latest(cf, "Operating Cash Flow", "Cash Flow From Continuing Operating Activities")
            if val is not None:
                data["CFO"] = val
                data["CFO_date"] = dt

        # Aggregate CashAndSecurities — max across all overlapping investment tags
        cash = data.get("Cash", 0) or 0
        sti = data.get("ShortTermInvestments", 0) or 0
        debt_avail = data.get("DebtSecuritiesAvailableForSaleCurrent", 0) or 0
        avail = data.get("AvailableForSaleSecurities", 0) or 0
        avail_debt = data.get("AvailableForSaleDebtCurrent", 0) or 0
        investment_component = max(sti, debt_avail, avail, avail_debt)
        total_liquid = cash + investment_component
        if total_liquid > 0:
            data["CashAndSecurities"] = total_liquid
            data["CashAndSecurities_date"] = data.get("Cash_date")

        if len(data) <= 3:  # only ticker, cik, source
            return None

        data["collected_at"] = date.today().isoformat()
        return data

    except Exception:
        return None


def collect_all_financial_data(universe_file: Path, output_file: Path, only_tickers=None):
    """Collect financial data for all tickers.

    If only_tickers is a non-empty iterable of tickers, only those are refreshed
    and the existing output_file is preserved for all other tickers (merge mode).
    """

    print("=" * 80)
    print("FINANCIAL DATA COLLECTION (SEC EDGAR)")
    print("=" * 80)
    print(f"Date: {date.today()}")
    print("\n⚠️  Note: SEC rate limits to 10 requests/second")
    print("         This will take 30-60 minutes for 300+ tickers")

    # Load universe
    with open(universe_file) as f:
        universe = json.load(f)

    tickers = [s["ticker"] for s in universe if s.get("ticker") and s["ticker"] != "_XBI_BENCHMARK_"]

    # Subset mode: filter to requested tickers only
    only_set = set(t.upper() for t in (only_tickers or []))
    if only_set:
        tickers = [t for t in tickers if t.upper() in only_set]
        missing = sorted(only_set - set(t.upper() for t in tickers))
        if missing:
            print(f"\n⚠️  Requested tickers not in universe: {missing}")
        print(f"\nSubset mode: refreshing {len(tickers)} ticker(s): {tickers}")
    # Build CIK lookup from universe.json (avoids SEC API calls for known CIKs)
    universe_ciks = {}
    for s in universe:
        t = s.get("ticker", "")
        c = s.get("cik", "")
        if t and c and c not in ("", "None", None):
            # Normalize to 10-digit zero-padded
            universe_ciks[t] = str(c).lstrip("0").zfill(10) if c else ""

    print(f"\nUniverse: {len(tickers)} tickers ({len(universe_ciks)} with CIK in universe.json)")
    print(f"Output: {output_file}")
    print(f"Estimated time: {len(tickers) * 0.2 / 60:.1f} minutes")

    # Pre-load SEC CIK cache (single request)
    print("Loading SEC ticker→CIK mapping...", flush=True)
    _load_cik_cache()
    print(f"  Loaded {len(_CIK_CACHE or {})} ticker→CIK mappings from SEC")

    # Collect
    all_data = []
    stats = {"total": len(tickers), "successful": 0, "no_cik": 0, "no_data": 0}

    print(f"\n{'='*80}")
    print("COLLECTING FINANCIAL DATA")
    print(f"{'='*80}\n")

    for i, ticker in enumerate(tickers, 1):
        print(f"[{i:3d}/{len(tickers)}] {ticker:6s}", end=" ", flush=True)

        # Get CIK: prefer universe.json, then SEC cache
        cik = universe_ciks.get(ticker) or get_cik_from_ticker(ticker)

        if not cik:
            # No EDGAR CIK — try yfinance fallback immediately
            data = get_yfinance_fallback(ticker)
            if data and len(data.keys()) > 3:
                all_data.append(data)
                stats["successful"] += 1
                cash = data.get("Cash", 0)
                cash_str = f"${cash/1e9:.1f}B" if cash and cash > 1e9 else f"${cash/1e6:.0f}M" if cash else "N/A"
                print(f"✅ (yfinance) Cash: {cash_str:>8s}")
            else:
                stats["no_cik"] += 1
                print("❌ No CIK, yfinance fallback failed")
            time.sleep(0.1)
            continue

        # Get financial data from EDGAR
        data = get_company_facts(cik, ticker)

        if data and len(data.keys()) > 3:
            all_data.append(data)
            stats["successful"] += 1

            cash = data.get("Cash", 0)
            revenue = data.get("Revenue", 0)

            cash_str = f"${cash/1e9:.1f}B" if cash and cash > 1e9 else f"${cash/1e6:.0f}M" if cash else "N/A"
            rev_str = (
                f"${revenue/1e9:.1f}B" if revenue and revenue > 1e9 else f"${revenue/1e6:.0f}M" if revenue else "N/A"
            )

            print(f"✅ Cash: {cash_str:>8s}, Rev: {rev_str:>8s}")
        else:
            # EDGAR returned nothing — try yfinance fallback
            data = get_yfinance_fallback(ticker)
            if data and len(data.keys()) > 3:
                all_data.append(data)
                stats["successful"] += 1
                cash = data.get("Cash", 0)
                cash_str = f"${cash/1e9:.1f}B" if cash and cash > 1e9 else f"${cash/1e6:.0f}M" if cash else "N/A"
                print(f"✅ (yfinance) Cash: {cash_str:>8s}")
            else:
                stats["no_data"] += 1
                print("⚠️  No filings (EDGAR + yfinance)")

        # SEC rate limit: 10 req/sec
        time.sleep(0.15)

        if i % 50 == 0:
            print(f"\n  Progress: {i}/{len(tickers)} ({i/len(tickers)*100:.1f}%)")
            print(f"  Success: {stats['successful']/i*100:.1f}%")
            print(f"  Time remaining: {(len(tickers)-i)*0.15/60:.1f} minutes\n")

    # Save. In subset mode, merge refreshed entries into existing output so
    # untouched tickers are preserved.
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if only_set and output_file.exists():
        try:
            with open(output_file) as f:
                existing = json.load(f)
            if isinstance(existing, list):
                refreshed_tickers = {d.get("ticker") for d in all_data if d.get("ticker")}
                kept = [e for e in existing if e.get("ticker") not in refreshed_tickers]
                all_data = kept + all_data
                print(f"\nMerged {len(kept)} preserved + {len(all_data) - len(kept)} refreshed entries")
        except (json.JSONDecodeError, OSError) as e:
            print(f"\n⚠️  Could not merge with existing {output_file}: {e}. Writing subset only.")
    with open(output_file, "w") as f:
        json.dump(all_data, f, indent=2)

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total: {stats['total']}")
    print(f"Successful: {stats['successful']}")
    print(f"No CIK: {stats['no_cik']}")
    print(f"No filings: {stats['no_data']}")
    print(f"Coverage: {stats['successful'] / stats['total'] * 100:.1f}%")
    print(f"✅ Saved to: {output_file}")

    # Data quality summary
    print(f"\n{'='*80}")
    print("DATA QUALITY SUMMARY")
    print(f"{'='*80}")

    # Count key field coverage
    has_cash = sum(1 for d in all_data if d.get("Cash"))
    has_revenue = sum(1 for d in all_data if d.get("Revenue"))
    has_rd = sum(1 for d in all_data if d.get("R&D"))
    has_assets = sum(1 for d in all_data if d.get("Assets"))
    has_cfo = sum(1 for d in all_data if d.get("CFO"))
    has_total_debt = sum(1 for d in all_data if d.get("TotalDebt"))
    has_long_term_debt = sum(1 for d in all_data if d.get("LongTermDebt"))

    collected = len(all_data)
    if collected > 0:
        print(f"Cash:           {has_cash:4d}/{collected} ({has_cash/collected*100:5.1f}%)")
        print(f"Revenue:        {has_revenue:4d}/{collected} ({has_revenue/collected*100:5.1f}%)")
        print(f"R&D:            {has_rd:4d}/{collected} ({has_rd/collected*100:5.1f}%)")
        print(f"Assets:         {has_assets:4d}/{collected} ({has_assets/collected*100:5.1f}%)")
        print(f"CFO:            {has_cfo:4d}/{collected} ({has_cfo/collected*100:5.1f}%)")
        print(f"LongTermDebt:   {has_long_term_debt:4d}/{collected} ({has_long_term_debt/collected*100:5.1f}%)")
        print(f"TotalDebt:      {has_total_debt:4d}/{collected} ({has_total_debt/collected*100:5.1f}%)")

        # Data freshness check
        fresh_count = sum(1 for d in all_data if d.get("Cash_date") and is_data_fresh(d["Cash_date"], 180))
        print(f"\nCash data <6mo: {fresh_count:4d}/{collected} ({fresh_count/collected*100:5.1f}%)")

    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Collect financial data from SEC EDGAR")
    parser.add_argument("--universe", type=Path, default=Path("production_data/universe.json"))
    parser.add_argument("--output", type=Path, default=Path("production_data/financial_data.json"))
    parser.add_argument(
        "--ticker",
        action="append",
        default=[],
        help="Refresh only this ticker (repeatable or comma-separated). Merges into existing output.",
    )
    args = parser.parse_args()

    # Flatten comma-separated values across --ticker occurrences
    only_tickers = []
    for t in args.ticker:
        only_tickers.extend(s.strip() for s in t.split(",") if s.strip())

    if not args.universe.exists():
        print(f"❌ Universe file not found: {args.universe}")
        return 1

    try:
        collect_all_financial_data(args.universe, args.output, only_tickers=only_tickers)
        return 0
    except KeyboardInterrupt:
        print("\n\n❌ Cancelled by user")
        return 1
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
