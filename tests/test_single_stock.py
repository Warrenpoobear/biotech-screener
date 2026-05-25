#!/usr/bin/env python3
"""
Single Stock Data Collection Test
Tests all data sources for a single ticker to diagnose collection issues.
"""

import json
import os
import sys
from datetime import datetime

import pytest

pytestmark = pytest.mark.network


@pytest.fixture
def ticker():
    """Opt-in ticker for this live-network diagnostic module."""
    value = os.environ.get("SINGLE_STOCK_TEST_TICKER")
    if not value:
        pytest.skip("Set SINGLE_STOCK_TEST_TICKER to run single-stock network diagnostics")
    return value.upper()


def test_yahoo_finance(ticker):
    """Test Yahoo Finance data collection."""
    print(f"\n{'='*60}")
    print(f"Testing Yahoo Finance for {ticker}")
    print(f"{'='*60}")

    try:
        import yfinance as yf

        stock = yf.Ticker(ticker)
        info = stock.info
        hist = stock.history(period="1y")

        # Basic info
        print("\n✅ Basic Info Retrieved:")
        print(f"   Name: {info.get('longName', 'N/A')}")
        print(f"   Sector: {info.get('sector', 'N/A')}")
        print(f"   Market Cap: ${info.get('marketCap', 0):,.0f}")
        print(f"   Price: ${info.get('currentPrice', info.get('regularMarketPrice', 0)):.2f}")

        # Financial data
        print("\n✅ Financial Data:")
        print(f"   Cash: ${info.get('totalCash', 0):,.0f}")
        print(f"   Debt: ${info.get('totalDebt', 0):,.0f}")
        print(f"   Revenue: ${info.get('totalRevenue', 0):,.0f}")

        # Price history
        print("\n✅ Price History:")
        print(f"   Days of data: {len(hist)}")
        print(f"   Date range: {hist.index[0].date()} to {hist.index[-1].date()}")

        # Calculate volatility
        if len(hist) > 20:
            returns = hist["Close"].pct_change().dropna()
            vol_daily = returns.std()
            vol_annual = vol_daily * (252**0.5)
            print(f"   Volatility (annual): {vol_annual:.2%}")

            # Calculate correlation with XBI (if we have it)
            try:
                xbi = yf.Ticker("XBI")
                xbi_hist = xbi.history(period="1y")

                # Align dates
                common_dates = hist.index.intersection(xbi_hist.index)
                if len(common_dates) > 20:
                    stock_returns = hist.loc[common_dates, "Close"].pct_change().dropna()
                    xbi_returns = xbi_hist.loc[common_dates, "Close"].pct_change().dropna()

                    # Align again after pct_change
                    common_dates2 = stock_returns.index.intersection(xbi_returns.index)
                    corr = stock_returns.loc[common_dates2].corr(xbi_returns.loc[common_dates2])
                    print(f"   Correlation with XBI: {corr:.2f}")
            except Exception:
                print("   Correlation with XBI: Could not calculate")

        return True

    except Exception as e:
        print(f"\n❌ Yahoo Finance Error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_clinicaltrials(ticker, company_name=None):
    """Test ClinicalTrials.gov data collection."""
    print(f"\n{'='*60}")
    print(f"Testing ClinicalTrials.gov for {ticker}")
    print(f"{'='*60}")

    try:
        import requests

        # Search by ticker and company name
        search_terms = [ticker]
        if company_name:
            search_terms.append(company_name)

        for term in search_terms:
            print(f"\n🔍 Searching for: {term}")

            url = "https://clinicaltrials.gov/api/v2/studies"
            params = {"query.term": term, "format": "json", "pageSize": 10}

            response = requests.get(url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()
                studies = data.get("studies", [])

                print(f"   ✅ Found {len(studies)} studies")

                if studies:
                    for i, study in enumerate(studies[:3], 1):
                        protocol = study.get("protocolSection", {})
                        id_module = protocol.get("identificationModule", {})
                        status_module = protocol.get("statusModule", {})

                        nct_id = id_module.get("nctId", "N/A")
                        title = id_module.get("briefTitle", "N/A")
                        status = status_module.get("overallStatus", "N/A")

                        print(f"\n   Study {i}:")
                        print(f"     NCT ID: {nct_id}")
                        print(f"     Title: {title[:60]}...")
                        print(f"     Status: {status}")

                return True
            else:
                print(f"   ❌ HTTP {response.status_code}: {response.reason}")
                return False

    except Exception as e:
        print(f"\n❌ ClinicalTrials.gov Error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_sec_edgar(ticker):
    """Test SEC EDGAR data access."""
    print(f"\n{'='*60}")
    print(f"Testing SEC EDGAR for {ticker}")
    print(f"{'='*60}")

    try:
        import requests

        # SEC requires User-Agent
        headers = {"User-Agent": "Mozilla/5.0 (compatible; DataCollector/1.0; +http://example.com)"}

        print(f"\n🔍 Searching SEC for ticker: {ticker}")

        url = "https://www.sec.gov/cgi-bin/browse-edgar"
        params = {
            "action": "getcompany",
            "company": ticker,
            "type": "10-Q",
            "dateb": "",
            "owner": "exclude",
            "count": 5,
        }

        response = requests.get(url, params=params, headers=headers, timeout=10)

        if response.status_code == 200:
            print("   ✅ SEC EDGAR accessible")

            # Check if we got any results
            if "No matching" in response.text:
                print(f"   ⚠️  No filings found for {ticker}")
                print("   💡 Try searching by company name instead")
            else:
                print(f"   ✅ Found filings for {ticker}")

            return True
        else:
            print(f"   ❌ HTTP {response.status_code}: {response.reason}")
            return False

    except Exception as e:
        print(f"\n❌ SEC EDGAR Error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_all_sources(ticker):
    """Test all data sources for a ticker."""
    print(f"\n{'#'*60}")
    print(f"# DATA COLLECTION TEST FOR {ticker}")
    print(f"{'#'*60}")
    print(f"\nStarting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    results = {"ticker": ticker, "timestamp": datetime.now().isoformat(), "tests": {}}

    # Get company name from Yahoo first
    company_name = None
    try:
        import yfinance as yf

        stock = yf.Ticker(ticker)
        company_name = stock.info.get("longName")
        print(f"\nCompany: {company_name}")
    except Exception:
        pass

    # Test each source
    results["tests"]["yahoo_finance"] = test_yahoo_finance(ticker)
    results["tests"]["clinicaltrials"] = test_clinicaltrials(ticker, company_name)
    results["tests"]["sec_edgar"] = test_sec_edgar(ticker)

    # Summary
    print(f"\n{'='*60}")
    print(f"TEST SUMMARY FOR {ticker}")
    print(f"{'='*60}")

    passed = sum(1 for v in results["tests"].values() if v)
    total = len(results["tests"])

    print(f"\nPassed: {passed}/{total} tests")

    for source, result in results["tests"].items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} - {source}")

    if passed == total:
        print(f"\n🎉 All tests passed! Data collection should work for {ticker}")
    elif passed > 0:
        print("\n⚠️  Some tests failed. Data collection will be incomplete.")
    else:
        print("\n❌ All tests failed. Check network, API keys, and dependencies.")

    print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*60}\n")

    return results


if __name__ == "__main__":
    # Check dependencies
    try:
        pass
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("\nInstall with:")
        print("  pip install yfinance requests")
        sys.exit(1)

    # Get ticker from command line or use default
    if len(sys.argv) > 1:
        ticker = sys.argv[1].upper()
    else:
        ticker = "GILD"
        print(f"No ticker specified, testing default: {ticker}")
        print("Usage: python test_single_stock.py TICKER")

    # Run tests
    results = test_all_sources(ticker)

    # Save results
    output_file = f"test_results_{ticker}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")
