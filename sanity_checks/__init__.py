"""
Sanity Check Framework

Battle-tested business logic validation for biotech investment screening.

This framework implements comprehensive cross-validation rules that catch
logical contradictions investment professionals would immediately flag.

Tiers:
- Tier 7: Business Logic & Sanity Check Framework
  - 7.1: Cross-Validation Sanity Checks
  - 7.2: Benchmark Comparison Checks
  - 7.3: Time Series Coherence Checks
  - 7.4: Biotech Domain Expert Checks
  - 7.5: Insider/Market Microstructure Checks
  - 7.6: Portfolio Construction Reality Checks
  - 7.7: Regression Testing Against Known Cases
  - 7.8: Expert Override & Manual Review Triggers

- Tier 8: Output Presentation & Usability
  - 8.1: Executive Dashboard Validation

Author: Wake Robin Capital Management
Version: 1.0.0
"""

from sanity_checks.types import (
    FlagSeverity,
    SanityFlag,
    SanityCheckResult,
    ValidationReport,
    ReviewRequirement,
)
from sanity_checks.cross_validation import CrossValidationChecker
from sanity_checks.benchmark_checks import BenchmarkChecker
from sanity_checks.time_series_checks import TimeSeriesChecker
from sanity_checks.domain_expert_checks import DomainExpertChecker
from sanity_checks.market_microstructure_checks import MarketMicrostructureChecker
from sanity_checks.portfolio_construction_checks import PortfolioConstructionChecker
from sanity_checks.regression_tests import RegressionTestRunner
from sanity_checks.review_triggers import ReviewTriggerChecker
from sanity_checks.executive_dashboard import ExecutiveDashboardValidator
from sanity_checks.runner import SanityCheckRunner, run_all_sanity_checks

__all__ = [
    # Types
    "FlagSeverity",
    "SanityFlag",
    "SanityCheckResult",
    "ValidationReport",
    "ReviewRequirement",
    # Checkers
    "CrossValidationChecker",
    "BenchmarkChecker",
    "TimeSeriesChecker",
    "DomainExpertChecker",
    "MarketMicrostructureChecker",
    "PortfolioConstructionChecker",
    "RegressionTestRunner",
    "ReviewTriggerChecker",
    "ExecutiveDashboardValidator",
    # Runner
    "SanityCheckRunner",
    "run_all_sanity_checks",
]

__version__ = "1.0.0"
