"""
Wake Robin Data Pipeline - Collectors Package

Modules for collecting biotech data from free, public sources:
- yahoo_collector: Market data from Yahoo Finance
- sec_collector: Financial data from SEC EDGAR
- trials_collector: Clinical trial data from ClinicalTrials.gov
"""

from . import yahoo_collector
from . import sec_collector
from . import trials_collector

__all__ = ['yahoo_collector', 'sec_collector', 'trials_collector']
