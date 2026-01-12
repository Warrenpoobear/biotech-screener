#!/usr/bin/env python3
"""
SEC EDGAR data fetcher - latest 10-Q/10-K filings.

Free, public data, no API key required.

Fetches financial data from SEC EDGAR API including:
- Cash and cash equivalents
- Total debt
- Revenue
- R&D expenses
- Filing dates

Note: SEC requires a User-Agent header with contact info.

Author: Wake Robin Capital Management
Version: 1.0.0
"""

import urllib.request
import urllib.parse
import json
import ssl
from decimal import Decimal
from datetime import datetime
from typing import Dict, List, Optional, Any


# Module metadata
__version__ = "1.0.0"

# SEC requires identifying User-Agent
USER_AGENT = "Wake Robin Capital (compliance@wakerobincapital.com)"


def _create_ssl_context() -> ssl.SSLContext:
    """Create SSL context for HTTPS requests."""
    context = ssl.create_default_context()
    return context


def fetch_company_facts(cik: str) -> Optional[Dict[str, Any]]:
    """
    Fetch company facts from SEC EDGAR API.

    The Company Facts API provides all XBRL data from company filings.

    Args:
        cik: 10-digit CIK (e.g., "0001318605" for Tesla)
             Can also pass without leading zeros.

    Returns:
        Dict containing:
        - cik: str
        - company_name: str
        - cash: float (latest cash position)
        - total_debt: float
        - revenue: float
        - rd_expense: float
        - filing_date: str
        - form_type: str (10-K or 10-Q)
        - fiscal_period: str
        - source: str

        Returns None if fetch fails.
    """
    try:
        # Pad CIK to 10 digits
        cik_padded = str(cik).zfill(10)

        # SEC EDGAR Company Facts API
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_padded}.json"

        headers = {
            "User-Agent": USER_AGENT,
            "Accept": "application/json"
        }

        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15, context=_create_ssl_context()) as response:
            data = json.loads(response.read().decode('utf-8'))

        company_name = data.get("entityName", "Unknown")
        facts = data.get("facts", {})

        # Try US-GAAP first, then IFRS
        gaap = facts.get("us-gaap", {})
        ifrs = facts.get("ifrs-full", {})

        # Extract latest cash position
        cash_data = _get_latest_fact(
            gaap.get("CashAndCashEquivalentsAtCarryingValue") or
            gaap.get("Cash") or
            ifrs.get("CashAndCashEquivalents"),
            "USD"
        )

        # Extract total debt
        debt_data = _get_latest_fact(
            gaap.get("LongTermDebt") or
            gaap.get("DebtCurrent") or
            gaap.get("TotalDebt"),
            "USD"
        )

        # Extract revenue
        revenue_data = _get_latest_fact(
            gaap.get("Revenues") or
            gaap.get("RevenueFromContractWithCustomerExcludingAssessedTax") or
            gaap.get("SalesRevenueNet"),
            "USD"
        )

        # Extract R&D expense
        rd_data = _get_latest_fact(
            gaap.get("ResearchAndDevelopmentExpense") or
            gaap.get("ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost"),
            "USD"
        )

        # Get filing info from cash data (most reliable)
        filing_date = None
        form_type = None
        fiscal_period = None

        if cash_data:
            filing_date = cash_data.get("filed")
            form_type = cash_data.get("form")
            fiscal_period = cash_data.get("fp")

        return {
            "cik": cik_padded,
            "company_name": company_name,
            "cash": cash_data.get("val") if cash_data else None,
            "total_debt": debt_data.get("val") if debt_data else None,
            "revenue": revenue_data.get("val") if revenue_data else None,
            "rd_expense": rd_data.get("val") if rd_data else None,
            "filing_date": filing_date,
            "form_type": form_type,
            "fiscal_period": fiscal_period,
            "collected_at": datetime.now().isoformat(),
            "source": "sec_edgar"
        }

    except urllib.error.HTTPError as e:
        if e.code == 404:
            print(f"  SEC EDGAR: No data found for CIK {cik}")
        else:
            print(f"  SEC EDGAR HTTP error for CIK {cik}: {e.code}")
        return None
    except urllib.error.URLError as e:
        print(f"  SEC EDGAR URL error for CIK {cik}: {e.reason}")
        return None
    except json.JSONDecodeError as e:
        print(f"  SEC EDGAR JSON error for CIK {cik}: {e}")
        return None
    except Exception as e:
        print(f"  SEC EDGAR fetch failed for CIK {cik}: {e}")
        return None


def _get_latest_fact(fact_data: Optional[Dict], unit: str = "USD") -> Optional[Dict]:
    """
    Extract the latest value from SEC fact data.

    Args:
        fact_data: XBRL fact data structure
        unit: Unit to filter by (e.g., "USD")

    Returns:
        Dict with val, filed, form, fp, or None
    """
    if not fact_data:
        return None

    units = fact_data.get("units", {})
    values = units.get(unit, [])

    if not values:
        return None

    # Sort by end date (most recent first)
    sorted_values = sorted(values, key=lambda x: x.get("end", ""), reverse=True)

    if sorted_values:
        return sorted_values[0]

    return None


def fetch_recent_filings(cik: str, form_types: Optional[List[str]] = None) -> Optional[List[Dict]]:
    """
    Fetch list of recent SEC filings for a company.

    Args:
        cik: Company CIK
        form_types: List of form types to filter (e.g., ["10-K", "10-Q", "8-K"])

    Returns:
        List of filing dicts with accession number, form type, date
    """
    try:
        cik_padded = str(cik).zfill(10)
        url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"

        headers = {
            "User-Agent": USER_AGENT,
            "Accept": "application/json"
        }

        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15, context=_create_ssl_context()) as response:
            data = json.loads(response.read().decode('utf-8'))

        filings = data.get("filings", {}).get("recent", {})

        forms = filings.get("form", [])
        dates = filings.get("filingDate", [])
        accessions = filings.get("accessionNumber", [])
        descriptions = filings.get("primaryDocument", [])

        results = []
        for i in range(len(forms)):
            form_type = forms[i] if i < len(forms) else None

            # Filter by form type if specified
            if form_types and form_type not in form_types:
                continue

            results.append({
                "form_type": form_type,
                "filing_date": dates[i] if i < len(dates) else None,
                "accession_number": accessions[i] if i < len(accessions) else None,
                "document": descriptions[i] if i < len(descriptions) else None,
            })

            # Limit to 20 filings
            if len(results) >= 20:
                break

        return results

    except Exception as e:
        print(f"  SEC EDGAR filings fetch failed for CIK {cik}: {e}")
        return None


def lookup_cik(ticker: str) -> Optional[str]:
    """
    Look up CIK for a ticker symbol.

    Uses SEC's company tickers JSON endpoint.

    Args:
        ticker: Stock ticker symbol

    Returns:
        CIK as string (10 digits with leading zeros), or None
    """
    try:
        url = "https://www.sec.gov/files/company_tickers.json"

        headers = {
            "User-Agent": USER_AGENT,
            "Accept": "application/json"
        }

        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15, context=_create_ssl_context()) as response:
            data = json.loads(response.read().decode('utf-8'))

        # Search for ticker
        ticker_upper = ticker.upper()
        for key, company in data.items():
            if company.get("ticker", "").upper() == ticker_upper:
                cik = company.get("cik_str")
                return str(cik).zfill(10) if cik else None

        return None

    except Exception as e:
        print(f"  SEC EDGAR CIK lookup failed for {ticker}: {e}")
        return None


def fetch_financials_by_ticker(ticker: str) -> Optional[Dict[str, Any]]:
    """
    Fetch financials for a ticker symbol.

    Convenience function that looks up CIK and fetches company facts.

    Args:
        ticker: Stock ticker symbol

    Returns:
        Company facts dict, or None
    """
    cik = lookup_cik(ticker)
    if not cik:
        print(f"  SEC EDGAR: Could not find CIK for {ticker}")
        return None

    result = fetch_company_facts(cik)
    if result:
        result["ticker"] = ticker

    return result


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("SEC EDGAR DATA FETCHER TEST")
    print("=" * 60)

    # Test with a known CIK (AbbVie)
    test_cik = "0001551152"  # AbbVie

    print(f"\nFetching company facts for CIK {test_cik}...")
    facts = fetch_company_facts(test_cik)
    if facts:
        print(f"  Company: {facts['company_name']}")
        print(f"  Cash: ${facts['cash']:,.0f}" if facts['cash'] else "  Cash: N/A")
        print(f"  Debt: ${facts['total_debt']:,.0f}" if facts['total_debt'] else "  Debt: N/A")
        print(f"  Revenue: ${facts['revenue']:,.0f}" if facts['revenue'] else "  Revenue: N/A")
        print(f"  R&D: ${facts['rd_expense']:,.0f}" if facts['rd_expense'] else "  R&D: N/A")
        print(f"  Filing: {facts['form_type']} on {facts['filing_date']}")
    else:
        print("  Failed to fetch company facts")

    # Test ticker lookup
    test_ticker = "NVAX"
    print(f"\nLooking up CIK for {test_ticker}...")
    cik = lookup_cik(test_ticker)
    if cik:
        print(f"  CIK: {cik}")

        # Fetch financials
        print(f"\nFetching financials for {test_ticker}...")
        financials = fetch_financials_by_ticker(test_ticker)
        if financials:
            print(f"  Company: {financials['company_name']}")
            print(f"  Cash: ${financials['cash']:,.0f}" if financials['cash'] else "  Cash: N/A")
        else:
            print("  Failed to fetch financials")
    else:
        print(f"  Could not find CIK for {test_ticker}")
