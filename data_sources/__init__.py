"""
Data source integrations for biotech screener.

Available data sources:
- yahoo_finance: Real-time market data (price, volume, market cap)
- sec_edgar: Financial data from SEC filings (cash, debt, R&D)
- clinicaltrials_gov: Clinical trial status and results
- ctgov_client: ClinicalTrials.gov API client (legacy)

All data sources are stdlib-only and require no API keys.
"""

from data_sources.yahoo_finance import (
    fetch_quote,
    fetch_key_statistics,
    fetch_historical,
)
from data_sources.sec_edgar import (
    fetch_company_facts,
    fetch_recent_filings,
    lookup_cik,
    fetch_financials_by_ticker,
)
from data_sources.clinicaltrials_gov import (
    fetch_trial_status,
    search_trials_by_sponsor,
    search_trials_by_condition,
    fetch_multiple_trials,
    get_trial_results_status,
)

__all__ = [
    # Yahoo Finance
    "fetch_quote",
    "fetch_key_statistics",
    "fetch_historical",
    # SEC EDGAR
    "fetch_company_facts",
    "fetch_recent_filings",
    "lookup_cik",
    "fetch_financials_by_ticker",
    # ClinicalTrials.gov
    "fetch_trial_status",
    "search_trials_by_sponsor",
    "search_trials_by_condition",
    "fetch_multiple_trials",
    "get_trial_results_status",
]
