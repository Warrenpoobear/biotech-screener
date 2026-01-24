"""
Integration Tests for Sanity Check Framework

Tests the comprehensive business logic and sanity check framework
that catches logical contradictions investment professionals would flag.

Tests cover:
- Query 7.1: Cross-Validation Sanity Checks
- Query 7.2: Benchmark Comparison Checks
- Query 7.3: Time Series Coherence Checks
- Query 7.4: Biotech Domain Expert Checks
- Query 7.5: Insider/Market Microstructure Checks
- Query 7.6: Portfolio Construction Reality Checks
- Query 7.7: Regression Testing Against Known Cases
- Query 7.8: Expert Override & Manual Review Triggers
- Query 8.1: Executive Dashboard Validation

Author: Wake Robin Capital Management
"""

from __future__ import annotations

import pytest
from decimal import Decimal
from datetime import date, timedelta

from sanity_checks.types import (
    CheckCategory,
    FlagSeverity,
    ReviewLevel,
    RankingSnapshot,
    SecurityContext,
    ThresholdConfig,
    GoldenTestCase,
)
from sanity_checks.cross_validation import CrossValidationChecker
from sanity_checks.benchmark_checks import BenchmarkChecker, PeerGroup
from sanity_checks.time_series_checks import TimeSeriesChecker, CatalystEvent
from sanity_checks.domain_expert_checks import DomainExpertChecker, TrialDetails
from sanity_checks.market_microstructure_checks import (
    MarketMicrostructureChecker,
    OptionsFlow,
    InsiderTransaction,
    AnalystRating,
)
from sanity_checks.portfolio_construction_checks import (
    PortfolioConstructionChecker,
    FundMandate,
)
from sanity_checks.regression_tests import RegressionTestRunner
from sanity_checks.review_triggers import ReviewTriggerChecker, ICDocumentation
from sanity_checks.executive_dashboard import ExecutiveDashboardValidator, OnePagerContent
from sanity_checks.runner import (
    SanityCheckRunner,
    run_all_sanity_checks,
    generate_battle_tested_report,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def as_of_date() -> date:
    """Standard test date."""
    return date(2026, 1, 15)


@pytest.fixture
def sample_securities() -> list[SecurityContext]:
    """Sample securities with various characteristics for testing."""
    return [
        # Top ranked healthy company
        SecurityContext(
            ticker="ACME",
            rank=1,
            composite_score=Decimal("85.50"),
            market_cap_mm=Decimal("2000"),
            cash_mm=Decimal("500"),
            runway_months=Decimal("24"),
            financial_score=Decimal("80"),
            clinical_score=Decimal("85"),
            catalyst_score=Decimal("0.75"),
            lead_phase="Phase 3",
            days_to_catalyst=90,
            pos_score=Decimal("0.55"),
            return_3m=Decimal("0.15"),
            short_interest_pct=Decimal("0.10"),
            elite_holder_count=5,
            specialist_holder_count=3,
            total_13f_holders=25,
            sector="oncology",
            indication="NSCLC",
            adv_dollars=Decimal("10000000"),
        ),
        # Problematic: Short runway but highly ranked
        SecurityContext(
            ticker="RISK",
            rank=5,
            composite_score=Decimal("75.00"),
            market_cap_mm=Decimal("500"),
            cash_mm=Decimal("50"),
            runway_months=Decimal("3"),  # Critical: short runway
            financial_score=Decimal("30"),
            clinical_score=Decimal("80"),
            lead_phase="Phase 2",
            days_to_catalyst=180,
            pos_score=Decimal("0.30"),
            return_3m=Decimal("-0.20"),
            elite_holder_count=0,
            specialist_holder_count=0,
            total_13f_holders=5,
            sector="rare_disease",
            indication="Duchenne MD",
            adv_dollars=Decimal("1000000"),
        ),
        # Problematic: Trading below cash
        SecurityContext(
            ticker="CASH",
            rank=10,
            composite_score=Decimal("70.00"),
            market_cap_mm=Decimal("100"),
            cash_mm=Decimal("300"),  # Cash > market cap
            runway_months=Decimal("36"),
            financial_score=Decimal("90"),
            clinical_score=Decimal("65"),
            lead_phase="Phase 2",
            pos_score=Decimal("0.28"),
            elite_holder_count=2,
            total_13f_holders=10,
            sector="CNS",
            indication="Alzheimer's",
            adv_dollars=Decimal("500000"),
        ),
        # Problematic: Phase 2 with high PoS
        SecurityContext(
            ticker="POSH",
            rank=15,
            composite_score=Decimal("68.00"),
            market_cap_mm=Decimal("800"),
            cash_mm=Decimal("200"),
            runway_months=Decimal("18"),
            clinical_score=Decimal("70"),
            lead_phase="Phase 2",
            pos_score=Decimal("0.75"),  # Too high for Phase 2
            elite_holder_count=3,
            total_13f_holders=15,
            sector="oncology",
            indication="Breast cancer",
            adv_dollars=Decimal("3000000"),
        ),
        # Micro-cap pre-revenue highly ranked
        SecurityContext(
            ticker="TINY",
            rank=8,
            composite_score=Decimal("72.00"),
            market_cap_mm=Decimal("150"),  # Micro-cap
            cash_mm=Decimal("80"),
            runway_months=Decimal("12"),
            financial_score=Decimal("50"),
            clinical_score=Decimal("75"),
            lead_phase="Phase 1",  # Pre-revenue
            pos_score=Decimal("0.10"),
            elite_holder_count=1,
            total_13f_holders=3,
            sector="oncology",
            indication="AML",
            adv_dollars=Decimal("200000"),  # Low liquidity
        ),
    ]


@pytest.fixture
def historical_snapshots(sample_securities) -> list[RankingSnapshot]:
    """Historical ranking snapshots for time series tests."""
    snapshots = []
    base_date = date(2026, 1, 15)

    for weeks_ago in range(4, 0, -1):
        snap_date = base_date - timedelta(weeks=weeks_ago)
        # Create modified securities for each snapshot
        securities = []
        for sec in sample_securities:
            modified = SecurityContext(
                ticker=sec.ticker,
                rank=sec.rank + (weeks_ago * 2) if sec.rank else None,  # Ranks were higher before
                composite_score=sec.composite_score - Decimal(str(weeks_ago)),
                market_cap_mm=sec.market_cap_mm,
                clinical_score=sec.clinical_score,
                financial_score=sec.financial_score,
                lead_phase=sec.lead_phase,
                sector=sec.sector,
                indication=sec.indication,
            )
            securities.append(modified)

        snapshots.append(RankingSnapshot(
            as_of_date=snap_date.isoformat(),
            securities=securities,
        ))

    return snapshots


# ============================================================================
# CROSS-VALIDATION TESTS (Query 7.1)
# ============================================================================

class TestCrossValidationChecker:
    """Tests for Query 7.1: Cross-Validation Sanity Checks."""

    def test_detects_imminent_dilution_risk(self, sample_securities):
        """Test detection of short runway in top-ranked candidate."""
        checker = CrossValidationChecker()
        result = checker.run_all_checks(sample_securities)

        # Should detect RISK ticker with 3-month runway
        dilution_flags = [
            f for f in result.flags
            if f.check_name == "imminent_dilution_risk"
        ]
        assert len(dilution_flags) >= 1
        assert dilution_flags[0].ticker == "RISK"
        assert dilution_flags[0].severity == FlagSeverity.CRITICAL

    def test_detects_trading_below_cash(self, sample_securities):
        """Test detection of trading below cash value."""
        checker = CrossValidationChecker()
        result = checker.run_all_checks(sample_securities)

        cash_flags = [
            f for f in result.flags
            if f.check_name == "trading_below_cash"
        ]
        assert len(cash_flags) >= 1
        assert cash_flags[0].ticker == "CASH"
        assert cash_flags[0].severity == FlagSeverity.CRITICAL

    def test_detects_pos_calibration_error(self, sample_securities):
        """Test detection of PoS calibration error."""
        checker = CrossValidationChecker()
        result = checker.run_all_checks(sample_securities)

        pos_flags = [
            f for f in result.flags
            if f.check_name == "pos_calibration_error"
        ]
        assert len(pos_flags) >= 1
        assert pos_flags[0].ticker == "POSH"

    def test_result_structure(self, sample_securities):
        """Test that result has expected structure."""
        checker = CrossValidationChecker()
        result = checker.run_all_checks(sample_securities)

        assert result.check_name == "cross_validation"
        assert result.category == CheckCategory.CROSS_VALIDATION
        assert isinstance(result.passed, bool)
        assert isinstance(result.flags, list)
        assert isinstance(result.metrics, dict)
        assert "total_securities_checked" in result.metrics


# ============================================================================
# BENCHMARK COMPARISON TESTS (Query 7.2)
# ============================================================================

class TestBenchmarkChecker:
    """Tests for Query 7.2: Benchmark Comparison Checks."""

    def test_detects_sector_concentration(self):
        """Test detection of sector concentration in top 10."""
        # Create 6 oncology companies in top 10
        securities = [
            SecurityContext(
                ticker=f"ONCO{i}",
                rank=i,
                composite_score=Decimal("80") - Decimal(str(i)),
                sector="oncology",
                market_cap_mm=Decimal("1000"),
            )
            for i in range(1, 7)
        ]
        # Add some other sectors
        securities.extend([
            SecurityContext(
                ticker=f"OTHER{i}",
                rank=i + 6,
                composite_score=Decimal("70") - Decimal(str(i)),
                sector="rare_disease",
                market_cap_mm=Decimal("1000"),
            )
            for i in range(1, 5)
        ])

        checker = BenchmarkChecker()
        result = checker.run_all_checks(securities)

        concentration_flags = [
            f for f in result.flags
            if "concentration" in f.check_name
        ]
        assert len(concentration_flags) >= 1

    def test_detects_extreme_outlier(self):
        """Test detection of extreme outlier score."""
        securities = [
            # Extreme outlier
            SecurityContext(
                ticker="OUTLIER",
                rank=1,
                composite_score=Decimal("95"),
                sector="oncology",
                market_cap_mm=Decimal("1000"),
            ),
            # Normal scores
            *[
                SecurityContext(
                    ticker=f"NORM{i}",
                    rank=i + 1,
                    composite_score=Decimal("30"),
                    sector="oncology",
                    market_cap_mm=Decimal("500"),
                )
                for i in range(5)
            ],
        ]

        checker = BenchmarkChecker()
        result = checker.run_all_checks(securities)

        outlier_flags = [
            f for f in result.flags
            if "outlier" in f.check_name
        ]
        assert len(outlier_flags) >= 1


# ============================================================================
# TIME SERIES COHERENCE TESTS (Query 7.3)
# ============================================================================

class TestTimeSeriesChecker:
    """Tests for Query 7.3: Time Series Coherence Checks."""

    def test_detects_large_rank_jump(self):
        """Test detection of large rank jump."""
        current = RankingSnapshot(
            as_of_date="2026-01-15",
            securities=[
                SecurityContext(ticker="JUMP", rank=2, composite_score=Decimal("80")),
                SecurityContext(ticker="STABLE", rank=5, composite_score=Decimal("75")),
            ],
        )
        previous = RankingSnapshot(
            as_of_date="2026-01-08",
            securities=[
                SecurityContext(ticker="JUMP", rank=50, composite_score=Decimal("50")),  # Big jump
                SecurityContext(ticker="STABLE", rank=6, composite_score=Decimal("74")),
            ],
        )

        checker = TimeSeriesChecker()
        result = checker.run_all_checks(
            current_snapshot=current,
            previous_snapshot=previous,
        )

        jump_flags = [
            f for f in result.flags
            if "rank_jump" in f.check_name
        ]
        assert len(jump_flags) >= 1
        assert jump_flags[0].ticker == "JUMP"

    def test_velocity_metrics(self):
        """Test that velocity metrics are calculated."""
        current = RankingSnapshot(
            as_of_date="2026-01-15",
            securities=[
                SecurityContext(ticker="A", rank=1, composite_score=Decimal("80")),
                SecurityContext(ticker="B", rank=2, composite_score=Decimal("75")),
            ],
        )
        previous = RankingSnapshot(
            as_of_date="2026-01-08",
            securities=[
                SecurityContext(ticker="A", rank=3, composite_score=Decimal("78")),
                SecurityContext(ticker="B", rank=2, composite_score=Decimal("75")),
            ],
        )

        checker = TimeSeriesChecker()
        result = checker.run_all_checks(
            current_snapshot=current,
            previous_snapshot=previous,
        )

        assert "rank_velocity" in result.metrics
        assert "avg_rank_change" in result.metrics["rank_velocity"]


# ============================================================================
# DOMAIN EXPERT TESTS (Query 7.4)
# ============================================================================

class TestDomainExpertChecker:
    """Tests for Query 7.4: Biotech Domain Expert Checks."""

    def test_detects_underpowered_trial(self):
        """Test detection of underpowered Phase 3 oncology trial."""
        trials = [
            TrialDetails(
                ticker="SMALL",
                nct_id="NCT12345678",
                phase="Phase 3",
                indication="Non-small cell lung cancer",
                primary_endpoint="Overall Survival",
                enrollment=50,  # Too small for oncology Phase 3
            ),
        ]

        securities = [
            SecurityContext(
                ticker="SMALL",
                rank=5,
                composite_score=Decimal("75"),
                lead_phase="Phase 3",
            ),
        ]

        checker = DomainExpertChecker()
        result = checker.run_all_checks(
            securities=securities,
            trial_details=trials,
        )

        underpowered_flags = [
            f for f in result.flags
            if "underpowered" in f.check_name
        ]
        assert len(underpowered_flags) >= 1

    def test_detects_early_stage_high_rank(self, sample_securities):
        """Test detection of pre-Phase 2 company ranked highly."""
        checker = DomainExpertChecker()
        result = checker.run_all_checks(sample_securities)

        early_stage_flags = [
            f for f in result.flags
            if "early_stage" in f.check_name
        ]
        # TINY is Phase 1 at rank 8
        assert len(early_stage_flags) >= 1


# ============================================================================
# MARKET MICROSTRUCTURE TESTS (Query 7.5)
# ============================================================================

class TestMarketMicrostructureChecker:
    """Tests for Query 7.5: Insider/Market Microstructure Checks."""

    def test_detects_csuite_selling_mismatch(self):
        """Test detection of C-suite selling in top-ranked name."""
        securities = [
            SecurityContext(
                ticker="SELL",
                rank=3,
                composite_score=Decimal("80"),
            ),
        ]

        insider_transactions = [
            InsiderTransaction(
                ticker="SELL",
                insider_name="John CEO",
                insider_title="CEO",
                transaction_type="sell",
                shares=100000,
                value_usd=Decimal("1500000"),
                is_c_suite=True,
                is_director=False,
            ),
            InsiderTransaction(
                ticker="SELL",
                insider_name="Jane CFO",
                insider_title="CFO",
                transaction_type="sell",
                shares=50000,
                value_usd=Decimal("750000"),
                is_c_suite=True,
                is_director=False,
            ),
        ]

        checker = MarketMicrostructureChecker()
        result = checker.run_all_checks(
            securities=securities,
            insider_transactions=insider_transactions,
        )

        csuite_flags = [
            f for f in result.flags
            if "csuite" in f.check_name
        ]
        assert len(csuite_flags) >= 1

    def test_detects_contrarian_position(self):
        """Test detection of contrarian position vs analyst consensus."""
        securities = [
            SecurityContext(
                ticker="CONTRA",
                rank=5,
                composite_score=Decimal("78"),
            ),
        ]

        analyst_ratings = [
            AnalystRating(ticker="CONTRA", rating="sell", target_price=None, prior_rating=None, rating_change_date=None),
            AnalystRating(ticker="CONTRA", rating="sell", target_price=None, prior_rating=None, rating_change_date=None),
            AnalystRating(ticker="CONTRA", rating="underperform", target_price=None, prior_rating=None, rating_change_date=None),
            AnalystRating(ticker="CONTRA", rating="hold", target_price=None, prior_rating=None, rating_change_date=None),
        ]

        checker = MarketMicrostructureChecker()
        result = checker.run_all_checks(
            securities=securities,
            analyst_ratings=analyst_ratings,
        )

        contrarian_flags = [
            f for f in result.flags
            if "contrarian" in f.check_name
        ]
        assert len(contrarian_flags) >= 1


# ============================================================================
# PORTFOLIO CONSTRUCTION TESTS (Query 7.6)
# ============================================================================

class TestPortfolioConstructionChecker:
    """Tests for Query 7.6: Portfolio Construction Reality Checks."""

    def test_detects_insufficient_liquidity(self, sample_securities):
        """Test detection of insufficient liquidity."""
        mandate = FundMandate(
            name="test_fund",
            aum_mm=Decimal("775"),
            min_market_cap_mm=Decimal("200"),
        )

        checker = PortfolioConstructionChecker(mandate=mandate)
        result = checker.run_all_checks(sample_securities)

        liquidity_flags = [
            f for f in result.flags
            if "liquidity" in f.check_name
        ]
        # TINY has low ADV
        assert len(liquidity_flags) >= 1

    def test_detects_mandate_violation(self, sample_securities):
        """Test detection of market cap below mandate minimum."""
        mandate = FundMandate(
            name="test_fund",
            aum_mm=Decimal("500"),
            min_market_cap_mm=Decimal("200"),  # TINY is 150
        )

        checker = PortfolioConstructionChecker(mandate=mandate)
        result = checker.run_all_checks(sample_securities)

        mandate_flags = [
            f for f in result.flags
            if "mandate" in f.check_name
        ]
        assert len(mandate_flags) >= 1


# ============================================================================
# REGRESSION TESTS (Query 7.7)
# ============================================================================

class TestRegressionTestRunner:
    """Tests for Query 7.7: Regression Testing Against Known Cases."""

    def test_edge_case_detection(self, sample_securities):
        """Test detection of edge cases (micro-cap, short runway, etc.)."""
        current = RankingSnapshot(
            as_of_date="2026-01-15",
            securities=sample_securities,
        )

        runner = RegressionTestRunner()
        result = runner.run_all_tests(
            current_snapshot=current,
            universe_size=len(sample_securities),
        )

        # Should detect edge cases in the sample data
        assert result.check_name == "regression_tests"
        assert "total_tests" in result.metrics

    def test_custom_golden_cases(self):
        """Test with custom golden test cases."""
        securities = [
            SecurityContext(ticker="WINNER", rank=5, composite_score=Decimal("80")),
            SecurityContext(ticker="LOSER", rank=95, composite_score=Decimal("20")),
        ]
        current = RankingSnapshot(
            as_of_date="2026-01-15",
            securities=securities,
        )

        golden_positive = [
            GoldenTestCase(
                ticker="WINNER",
                as_of_date="2026-01-15",
                expected_outcome="top_20",
                case_type="positive",
                description="Test winner",
                threshold_rank=20,
            ),
        ]
        golden_negative = [
            GoldenTestCase(
                ticker="LOSER",
                as_of_date="2026-01-15",
                expected_outcome="bottom_50",
                case_type="negative",
                description="Test loser",
            ),
        ]

        runner = RegressionTestRunner(
            golden_positive=golden_positive,
            golden_negative=golden_negative,
        )
        result = runner.run_all_tests(
            current_snapshot=current,
            universe_size=100,
        )

        # Both should pass
        assert result.metrics["by_type"]["positive"]["passed"] == 1
        assert result.metrics["by_type"]["negative"]["passed"] == 1


# ============================================================================
# REVIEW TRIGGER TESTS (Query 7.8)
# ============================================================================

class TestReviewTriggerChecker:
    """Tests for Query 7.8: Expert Override & Manual Review Triggers."""

    def test_detects_unknown_score_components(self):
        """Test detection of UNKNOWN score components."""
        securities = [
            SecurityContext(
                ticker="UNKNOWN",
                rank=5,
                composite_score=Decimal("70"),
                clinical_score=None,  # Unknown
                financial_score=Decimal("60"),
                catalyst_score=None,  # Unknown
            ),
        ]

        checker = ReviewTriggerChecker()
        result = checker.run_all_checks(securities)

        unknown_flags = [
            f for f in result.flags
            if "unknown" in f.check_name
        ]
        assert len(unknown_flags) >= 1

    def test_generates_review_requirements(self, sample_securities):
        """Test that review requirements are generated for top candidates."""
        checker = ReviewTriggerChecker()
        result = checker.run_all_checks(sample_securities)

        # Should have review requirements in metrics
        assert "review_requirements" in result.metrics

    def test_detects_large_rank_change(self, sample_securities):
        """Test detection of large week-over-week rank change."""
        previous_ranks = {
            "ACME": 20,  # Was 20, now 1 = 19 position change
            "RISK": 5,
        }

        checker = ReviewTriggerChecker()
        result = checker.run_all_checks(
            securities=sample_securities,
            previous_ranks=previous_ranks,
        )

        rank_change_flags = [
            f for f in result.flags
            if "rank_change" in f.check_name
        ]
        assert len(rank_change_flags) >= 1


# ============================================================================
# EXECUTIVE DASHBOARD TESTS (Query 8.1)
# ============================================================================

class TestExecutiveDashboardValidator:
    """Tests for Query 8.1: Executive Dashboard Validation."""

    def test_detects_missing_one_pagers(self, sample_securities):
        """Test detection of missing one-pagers for top 10."""
        # No one-pagers provided
        validator = ExecutiveDashboardValidator()
        result = validator.run_all_checks(
            securities=sample_securities,
            one_pagers=None,
        )

        missing_flags = [
            f for f in result.flags
            if "one_pager" in f.check_name
        ]
        assert len(missing_flags) >= 1
        assert any(f.severity == FlagSeverity.CRITICAL for f in missing_flags)

    def test_detects_incomplete_one_pager(self, sample_securities):
        """Test detection of incomplete one-pager content."""
        one_pagers = {
            "ACME": OnePagerContent(
                ticker="ACME",
                company_name="ACME Biotech",
                market_cap_mm=Decimal("2000"),
                sector="oncology",
                current_price=None,
                price_52w_high=None,
                price_52w_low=None,
                thesis=None,  # Missing
                catalyst_date="2026-04-15",
                catalyst_type="PDUFA",
                top_bulls=["Bull 1"],  # Not enough
                top_bears=["Bear 1"],  # Not enough
                composite_score=Decimal("85.50"),
                score_breakdown={},
            ),
        }

        validator = ExecutiveDashboardValidator()
        result = validator.run_all_checks(
            securities=sample_securities,
            one_pagers=one_pagers,
        )

        incomplete_flags = [
            f for f in result.flags
            if "incomplete_one_pager" in f.check_name
        ]
        assert len(incomplete_flags) >= 1


# ============================================================================
# FULL RUNNER TESTS
# ============================================================================

class TestSanityCheckRunner:
    """Tests for the main SanityCheckRunner."""

    def test_runs_all_checks(self, sample_securities, as_of_date):
        """Test that runner executes all enabled checks."""
        report = run_all_sanity_checks(
            securities=sample_securities,
            as_of_date=as_of_date,
        )

        assert report.as_of_date == as_of_date.isoformat()
        assert len(report.check_results) > 0
        assert report.verdict in ("BLOCKED", "INVESTIGATE", "REVIEW", "PROCEED")

    def test_detects_critical_issues(self, sample_securities, as_of_date):
        """Test that critical issues are detected."""
        report = run_all_sanity_checks(
            securities=sample_securities,
            as_of_date=as_of_date,
        )

        # Should have critical flags (imminent dilution, trading below cash)
        assert len(report.critical_flags) > 0
        assert report.ic_review_blocked

    def test_selective_check_categories(self, sample_securities, as_of_date):
        """Test running only selected check categories."""
        runner = SanityCheckRunner(
            enabled_checks=["cross_validation", "domain_expert"],
        )

        report = runner.run(
            securities=sample_securities,
            as_of_date=as_of_date,
        )

        # Should only have results for enabled categories
        categories = {r.category for r in report.check_results}
        assert CheckCategory.CROSS_VALIDATION in categories
        assert CheckCategory.DOMAIN_EXPERT in categories
        assert CheckCategory.BENCHMARK not in categories

    def test_battle_tested_report(self, sample_securities, as_of_date):
        """Test battle-tested report generation."""
        report = run_all_sanity_checks(
            securities=sample_securities,
            as_of_date=as_of_date,
        )

        battle_tested = generate_battle_tested_report(report)

        assert "battle_tested" in battle_tested
        assert "criteria" in battle_tested
        assert "recommendation" in battle_tested

    def test_report_serialization(self, sample_securities, as_of_date):
        """Test that report can be serialized to dict."""
        report = run_all_sanity_checks(
            securities=sample_securities,
            as_of_date=as_of_date,
        )

        report_dict = report.to_dict()

        assert "as_of_date" in report_dict
        assert "verdict" in report_dict
        assert "check_results" in report_dict
        assert isinstance(report_dict["check_results"], list)


# ============================================================================
# THRESHOLD CONFIGURATION TESTS
# ============================================================================

class TestThresholdConfiguration:
    """Tests for threshold configuration."""

    def test_custom_thresholds(self, sample_securities):
        """Test that custom thresholds are respected."""
        # Very strict thresholds
        config = ThresholdConfig(
            min_runway_months_for_top_rank=Decimal("12"),
            max_rank_jump_single_week=5,
        )

        checker = CrossValidationChecker(config=config)
        result = checker.run_all_checks(sample_securities)

        # With 12-month runway requirement, more securities should be flagged
        dilution_flags = [
            f for f in result.flags
            if "dilution" in f.check_name
        ]
        assert len(dilution_flags) >= 1

    def test_default_thresholds_immutable(self):
        """Test that default thresholds cannot be modified."""
        from sanity_checks.types import DEFAULT_THRESHOLDS

        # ThresholdConfig is frozen
        with pytest.raises(AttributeError):
            DEFAULT_THRESHOLDS.min_runway_months_for_top_rank = Decimal("99")


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestFullIntegration:
    """Full integration tests."""

    def test_complete_workflow(self, sample_securities, historical_snapshots, as_of_date):
        """Test complete sanity check workflow."""
        # Build current snapshot
        current_snapshot = RankingSnapshot(
            as_of_date=as_of_date.isoformat(),
            securities=sample_securities,
        )

        # Previous snapshot (from historical)
        previous_snapshot = historical_snapshots[-1] if historical_snapshots else None

        # Run full checks
        runner = SanityCheckRunner()
        report = runner.run(
            securities=sample_securities,
            as_of_date=as_of_date,
            historical_snapshots=historical_snapshots,
            previous_snapshot=previous_snapshot,
        )

        # Verify report structure
        assert report.as_of_date == as_of_date.isoformat()
        assert len(report.check_results) >= 7  # At least 7 check categories

        # Verify all categories ran
        categories = {r.check_name for r in report.check_results}
        expected = {
            "cross_validation",
            "benchmark_comparison",
            "time_series_coherence",
            "domain_expert",
            "market_microstructure",
            "portfolio_construction",
            "regression_tests",
        }
        assert expected.issubset(categories)

    def test_flags_are_actionable(self, sample_securities, as_of_date):
        """Test that generated flags have actionable information."""
        report = run_all_sanity_checks(
            securities=sample_securities,
            as_of_date=as_of_date,
        )

        for result in report.check_results:
            for flag in result.flags:
                # Every flag should have these
                assert flag.severity in FlagSeverity
                assert flag.category in CheckCategory
                assert flag.check_name
                assert flag.message
                # Critical and high flags should have recommendations
                if flag.severity in (FlagSeverity.CRITICAL, FlagSeverity.HIGH):
                    assert flag.recommendation


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
