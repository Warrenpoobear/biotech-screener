"""
Sanity Check Runner

Orchestrates all sanity checks and produces a comprehensive validation report.

This is the main entry point for the sanity check framework.

Usage:
    from sanity_checks.runner import SanityCheckRunner, run_all_sanity_checks

    # Quick run with defaults
    report = run_all_sanity_checks(securities, as_of_date="2026-01-15")

    # Detailed run with all data sources
    runner = SanityCheckRunner(config=custom_config)
    report = runner.run(
        securities=securities,
        as_of_date="2026-01-15",
        historical_snapshots=snapshots,
        trial_details=trials,
        ...
    )

Author: Wake Robin Capital Management
Version: 1.0.0
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from sanity_checks.types import (
    CheckCategory,
    FlagSeverity,
    RankingSnapshot,
    SanityCheckResult,
    SanityFlag,
    SecurityContext,
    ThresholdConfig,
    ValidationReport,
    DEFAULT_THRESHOLDS,
)
from sanity_checks.cross_validation import CrossValidationChecker
from sanity_checks.benchmark_checks import BenchmarkChecker, PeerGroup
from sanity_checks.time_series_checks import TimeSeriesChecker, CatalystEvent
from sanity_checks.domain_expert_checks import (
    DomainExpertChecker,
    TrialDetails,
    PartnershipInfo,
    CompetitiveLandscape,
)
from sanity_checks.market_microstructure_checks import (
    MarketMicrostructureChecker,
    OptionsFlow,
    InsiderTransaction,
    ConferencePresentation,
    AnalystRating,
)
from sanity_checks.portfolio_construction_checks import (
    PortfolioConstructionChecker,
    FundMandate,
)
from sanity_checks.regression_tests import RegressionTestRunner, GoldenTestCase
from sanity_checks.review_triggers import ReviewTriggerChecker, ICDocumentation
from sanity_checks.executive_dashboard import (
    ExecutiveDashboardValidator,
    OnePagerContent,
)

logger = logging.getLogger(__name__)


class SanityCheckRunner:
    """
    Main orchestrator for all sanity checks.

    Runs all configured checks and produces a comprehensive validation report.
    """

    def __init__(
        self,
        config: Optional[ThresholdConfig] = None,
        fund_mandate: Optional[FundMandate] = None,
        enabled_checks: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize the sanity check runner.

        Args:
            config: Threshold configuration for all checks
            fund_mandate: Fund mandate for portfolio construction checks
            enabled_checks: List of check categories to run (None = all)
        """
        self.config = config or DEFAULT_THRESHOLDS
        self.fund_mandate = fund_mandate or FundMandate(name="default")

        # Default to all checks enabled
        self.enabled_checks = enabled_checks or [
            "cross_validation",
            "benchmark",
            "time_series",
            "domain_expert",
            "market_microstructure",
            "portfolio_construction",
            "regression",
            "review_trigger",
            "dashboard",
        ]

        # Initialize checkers
        self._init_checkers()

    def _init_checkers(self) -> None:
        """Initialize all checker instances."""
        self.cross_validation_checker = CrossValidationChecker(self.config)
        self.benchmark_checker = BenchmarkChecker(self.config)
        self.time_series_checker = TimeSeriesChecker(self.config)
        self.domain_expert_checker = DomainExpertChecker(self.config)
        self.market_microstructure_checker = MarketMicrostructureChecker(self.config)
        self.portfolio_construction_checker = PortfolioConstructionChecker(
            mandate=self.fund_mandate,
            config=self.config,
        )
        self.regression_test_runner = RegressionTestRunner(config=self.config)
        self.review_trigger_checker = ReviewTriggerChecker(self.config)
        self.dashboard_validator = ExecutiveDashboardValidator(self.config)

    def run(
        self,
        securities: List[SecurityContext],
        as_of_date: Union[str, date],
        # Historical data
        historical_snapshots: Optional[List[RankingSnapshot]] = None,
        previous_snapshot: Optional[RankingSnapshot] = None,
        previous_ranks: Optional[Dict[str, int]] = None,
        # Domain data
        trial_details: Optional[List[TrialDetails]] = None,
        partnerships: Optional[List[PartnershipInfo]] = None,
        competitive_data: Optional[Dict[str, CompetitiveLandscape]] = None,
        # Market data
        options_data: Optional[List[OptionsFlow]] = None,
        insider_transactions: Optional[List[InsiderTransaction]] = None,
        upcoming_conferences: Optional[List[ConferencePresentation]] = None,
        analyst_ratings: Optional[List[AnalystRating]] = None,
        # Portfolio data
        existing_holdings: Optional[Dict[str, Decimal]] = None,
        # Catalyst data
        catalyst_events: Optional[List[CatalystEvent]] = None,
        # Regression data
        golden_positive_cases: Optional[List[GoldenTestCase]] = None,
        golden_negative_cases: Optional[List[GoldenTestCase]] = None,
        # Documentation
        ic_documentation: Optional[Dict[str, ICDocumentation]] = None,
        one_pagers: Optional[Dict[str, OnePagerContent]] = None,
        # Benchmark data
        peer_groups: Optional[List[PeerGroup]] = None,
    ) -> ValidationReport:
        """
        Run all enabled sanity checks.

        Args:
            securities: List of security contexts to validate
            as_of_date: Analysis date
            historical_snapshots: Historical ranking snapshots
            previous_snapshot: Previous period snapshot
            previous_ranks: Previous period ranks by ticker
            trial_details: Detailed trial information
            partnerships: Partnership information
            competitive_data: Competitive landscape by indication
            options_data: Options flow data
            insider_transactions: Insider trading transactions
            upcoming_conferences: Conference presentations
            analyst_ratings: Analyst ratings
            existing_holdings: Current portfolio holdings
            catalyst_events: Known catalyst events
            golden_positive_cases: Custom positive test cases
            golden_negative_cases: Custom negative test cases
            ic_documentation: IC documentation status
            one_pagers: One-pager content by ticker
            peer_groups: Peer group definitions

        Returns:
            ValidationReport with all check results
        """
        # Normalize as_of_date
        if isinstance(as_of_date, date):
            as_of_str = as_of_date.isoformat()
        else:
            as_of_str = as_of_date

        # Build current snapshot for some checks
        current_snapshot = RankingSnapshot(
            as_of_date=as_of_str,
            securities=securities,
        )

        # Initialize report
        report = ValidationReport(as_of_date=as_of_str)

        # Run each enabled check category
        if "cross_validation" in self.enabled_checks:
            logger.info("Running cross-validation checks...")
            result = self.cross_validation_checker.run_all_checks(securities)
            report.check_results.append(result)

        if "benchmark" in self.enabled_checks:
            logger.info("Running benchmark comparison checks...")
            result = self.benchmark_checker.run_all_checks(
                securities=securities,
                historical_snapshots=historical_snapshots,
                peer_groups=peer_groups,
            )
            report.check_results.append(result)

        if "time_series" in self.enabled_checks:
            logger.info("Running time series coherence checks...")
            result = self.time_series_checker.run_all_checks(
                current_snapshot=current_snapshot,
                previous_snapshot=previous_snapshot,
                historical_snapshots=historical_snapshots,
                catalyst_events=catalyst_events,
            )
            report.check_results.append(result)

        if "domain_expert" in self.enabled_checks:
            logger.info("Running domain expert checks...")
            result = self.domain_expert_checker.run_all_checks(
                securities=securities,
                trial_details=trial_details,
                partnerships=partnerships,
                competitive_data=competitive_data,
            )
            report.check_results.append(result)

        if "market_microstructure" in self.enabled_checks:
            logger.info("Running market microstructure checks...")
            result = self.market_microstructure_checker.run_all_checks(
                securities=securities,
                options_data=options_data,
                insider_transactions=insider_transactions,
                upcoming_conferences=upcoming_conferences,
                analyst_ratings=analyst_ratings,
            )
            report.check_results.append(result)

        if "portfolio_construction" in self.enabled_checks:
            logger.info("Running portfolio construction checks...")
            result = self.portfolio_construction_checker.run_all_checks(
                securities=securities,
                existing_holdings=existing_holdings,
            )
            report.check_results.append(result)

        if "regression" in self.enabled_checks:
            logger.info("Running regression tests...")
            # Update runner with custom cases if provided
            if golden_positive_cases:
                self.regression_test_runner.golden_positive = golden_positive_cases
            if golden_negative_cases:
                self.regression_test_runner.golden_negative = golden_negative_cases

            result = self.regression_test_runner.run_all_tests(
                current_snapshot=current_snapshot,
                universe_size=len(securities),
            )
            report.check_results.append(result)

        if "review_trigger" in self.enabled_checks:
            logger.info("Running review trigger checks...")
            result = self.review_trigger_checker.run_all_checks(
                securities=securities,
                previous_ranks=previous_ranks,
                documentation=ic_documentation,
            )
            report.check_results.append(result)

            # Extract review requirements from result metrics
            if "review_requirements" in result.metrics:
                from sanity_checks.types import ReviewRequirement, ReviewLevel
                for req_dict in result.metrics["review_requirements"]:
                    report.review_requirements.append(ReviewRequirement(
                        ticker=req_dict["ticker"],
                        rank=req_dict["rank"],
                        level=ReviewLevel(req_dict["level"]),
                        reasons=req_dict["reasons"],
                        requires_memo=req_dict["requires_memo"],
                        requires_sign_off=req_dict["requires_sign_off"],
                        blocking=req_dict["blocking"],
                    ))

        if "dashboard" in self.enabled_checks:
            logger.info("Running executive dashboard validation...")
            result = self.dashboard_validator.run_all_checks(
                securities=securities,
                one_pagers=one_pagers,
                historical_snapshots=historical_snapshots,
                existing_holdings=existing_holdings,
            )
            report.check_results.append(result)

        # Log summary
        self._log_summary(report)

        return report

    def _log_summary(self, report: ValidationReport) -> None:
        """Log a summary of the validation results."""
        logger.info("=" * 70)
        logger.info("SANITY CHECK SUMMARY")
        logger.info("=" * 70)
        logger.info(f"As of date: {report.as_of_date}")
        logger.info(f"Total flags: {report.total_flags}")
        logger.info(f"  - Critical: {len(report.critical_flags)}")
        logger.info(f"  - High: {len(report.high_flags)}")
        logger.info(f"Verdict: {report.verdict}")
        logger.info(f"IC Review Blocked: {report.ic_review_blocked}")
        logger.info("=" * 70)

        if report.critical_flags:
            logger.warning("CRITICAL FLAGS:")
            for flag in report.critical_flags:
                logger.warning(f"  [{flag.ticker or 'GLOBAL'}] {flag.message}")

    def save_report(
        self,
        report: ValidationReport,
        output_path: Union[str, Path],
    ) -> None:
        """
        Save validation report to JSON file.

        Args:
            report: Validation report to save
            output_path: Path to output file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2, default=str)

        logger.info(f"Report saved to: {output_path}")


def run_all_sanity_checks(
    securities: List[SecurityContext],
    as_of_date: Union[str, date],
    config: Optional[ThresholdConfig] = None,
    **kwargs: Any,
) -> ValidationReport:
    """
    Convenience function to run all sanity checks with defaults.

    Args:
        securities: List of security contexts to validate
        as_of_date: Analysis date
        config: Optional threshold configuration
        **kwargs: Additional data sources passed to runner.run()

    Returns:
        ValidationReport with all check results

    Example:
        from sanity_checks import run_all_sanity_checks

        report = run_all_sanity_checks(
            securities=securities,
            as_of_date="2026-01-15",
        )

        if report.ic_review_blocked:
            print("IC presentation BLOCKED - fix critical issues first")
        else:
            print(f"Ready for IC review. Verdict: {report.verdict}")
    """
    runner = SanityCheckRunner(config=config)
    return runner.run(securities=securities, as_of_date=as_of_date, **kwargs)


def generate_battle_tested_report(
    report: ValidationReport,
) -> Dict[str, Any]:
    """
    Generate the battle-tested status report.

    Checks pass criteria:
    - Zero critical flags
    - Historical regression tests pass at 100%
    - >90% of rank changes have documented explanations
    - Manual review override rate <10%
    - IC questions answered in automated output >80%

    Args:
        report: Validation report to analyze

    Returns:
        Battle-tested status report
    """
    # Extract metrics from each check
    regression_result = next(
        (r for r in report.check_results if r.check_name == "regression_tests"),
        None,
    )

    review_result = next(
        (r for r in report.check_results if r.check_name == "review_triggers"),
        None,
    )

    dashboard_result = next(
        (r for r in report.check_results if r.check_name == "executive_dashboard"),
        None,
    )

    # Calculate battle-tested criteria
    criteria = {
        "zero_critical_flags": len(report.critical_flags) == 0,
        "regression_tests_pass": (
            regression_result and
            regression_result.metrics.get("all_passed", False)
        ) if regression_result else None,
        "rank_changes_documented": None,  # Would need time series data
        "override_rate_low": None,  # Would need historical override data
        "ic_coverage_high": (
            dashboard_result and
            dashboard_result.metrics.get("one_pager_quality", {}).get("completeness_pct", 0) >= 0.8
        ) if dashboard_result else None,
    }

    # Determine overall status
    critical_criteria = ["zero_critical_flags", "regression_tests_pass"]
    critical_passed = all(
        criteria.get(c, False) for c in critical_criteria
        if criteria.get(c) is not None
    )

    return {
        "battle_tested": critical_passed,
        "criteria": criteria,
        "summary": {
            "critical_flags": len(report.critical_flags),
            "high_flags": len(report.high_flags),
            "total_flags": report.total_flags,
            "verdict": report.verdict,
        },
        "recommendation": (
            "READY for IC presentation"
            if critical_passed
            else "NOT READY - address critical issues"
        ),
    }


def print_validation_summary(report: ValidationReport) -> None:
    """
    Print a human-readable validation summary.

    Args:
        report: Validation report to summarize
    """
    print("\n" + "=" * 70)
    print("SANITY CHECK VALIDATION SUMMARY")
    print("=" * 70)
    print(f"As of: {report.as_of_date}")
    print(f"Verdict: {report.verdict}")
    print(f"IC Review Blocked: {'YES' if report.ic_review_blocked else 'NO'}")
    print("-" * 70)

    # Flag summary
    print(f"\nFlag Summary:")
    print(f"  Total: {report.total_flags}")
    print(f"  Critical: {len(report.critical_flags)}")
    print(f"  High: {len(report.high_flags)}")

    # Check results
    print(f"\nCheck Results:")
    for result in report.check_results:
        status = "PASS" if result.passed else "FAIL"
        print(f"  [{status}] {result.check_name}: {len(result.flags)} flags")

    # Critical flags detail
    if report.critical_flags:
        print(f"\nCRITICAL FLAGS (must fix before IC):")
        for flag in report.critical_flags:
            ticker = flag.ticker or "GLOBAL"
            print(f"  [{ticker}] {flag.check_name}")
            print(f"    {flag.message}")
            if flag.recommendation:
                print(f"    -> {flag.recommendation}")

    # Review requirements
    if report.review_requirements:
        print(f"\nReview Requirements:")
        blocking = [r for r in report.review_requirements if r.blocking]
        if blocking:
            print(f"  BLOCKING ({len(blocking)}):")
            for req in blocking:
                print(f"    {req.ticker}: {', '.join(req.reasons[:2])}")

    print("\n" + "=" * 70)
