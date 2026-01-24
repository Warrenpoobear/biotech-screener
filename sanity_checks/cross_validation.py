"""
Query 7.1: Cross-Validation Sanity Checks

Implements comprehensive cross-validation rules that catch logical contradictions:

1. Financial-Clinical Contradictions
2. Institutional Signal Contradictions
3. Momentum-Fundamental Contradictions
4. PoS-Stage Contradictions

These are the "smell tests" that catch errors investment professionals
would immediately flag.

Author: Wake Robin Capital Management
Version: 1.0.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from sanity_checks.types import (
    CheckCategory,
    ContradictionType,
    FlagSeverity,
    SanityCheckResult,
    SanityFlag,
    SecurityContext,
    ThresholdConfig,
    DEFAULT_THRESHOLDS,
)

logger = logging.getLogger(__name__)


class CrossValidationChecker:
    """
    Cross-validation sanity checker.

    Detects logical contradictions between different data sources
    and scoring components.
    """

    def __init__(self, config: Optional[ThresholdConfig] = None) -> None:
        self.config = config or DEFAULT_THRESHOLDS

    def run_all_checks(
        self,
        securities: List[SecurityContext],
    ) -> SanityCheckResult:
        """
        Run all cross-validation checks on the securities.

        Args:
            securities: List of security contexts with all relevant data

        Returns:
            SanityCheckResult with all flags
        """
        flags: List[SanityFlag] = []

        for sec in securities:
            # Only check ranked securities
            if sec.rank is None:
                continue

            # 1. Financial-Clinical Contradictions
            flags.extend(self._check_financial_clinical(sec))

            # 2. Institutional Signal Contradictions
            flags.extend(self._check_institutional_signal(sec))

            # 3. Momentum-Fundamental Contradictions
            flags.extend(self._check_momentum_fundamental(sec))

            # 4. PoS-Stage Contradictions
            flags.extend(self._check_pos_stage(sec))

        # Calculate metrics
        metrics = self._calculate_metrics(flags, securities)

        # Determine pass/fail - fail if any CRITICAL flags
        passed = not any(f.severity == FlagSeverity.CRITICAL for f in flags)

        return SanityCheckResult(
            check_name="cross_validation",
            category=CheckCategory.CROSS_VALIDATION,
            passed=passed,
            flags=flags,
            metrics=metrics,
        )

    def _check_financial_clinical(
        self,
        sec: SecurityContext,
    ) -> List[SanityFlag]:
        """
        Check for Financial-Clinical contradictions.

        Flags:
        - Pre-revenue company ranked #1 BUT no Phase 3 catalyst within 12 months
        - Runway <6 months BUT scored as top candidate
        - Cash >$500M but market cap <$200M (trading below cash)
        """
        flags: List[SanityFlag] = []

        # Check 1: Pre-revenue #1 without near-term Phase 3
        if sec.rank == 1 and sec.is_pre_revenue:
            has_phase3_catalyst = (
                sec.lead_phase == "Phase 3" and
                sec.days_to_catalyst is not None and
                sec.days_to_catalyst <= self.config.phase3_catalyst_window_months * 30
            )
            if not has_phase3_catalyst:
                flags.append(SanityFlag(
                    severity=FlagSeverity.HIGH,
                    category=CheckCategory.CROSS_VALIDATION,
                    ticker=sec.ticker,
                    check_name="revenue_timeline_mismatch",
                    message=f"Pre-revenue company ranked #1 without Phase 3 catalyst within 12 months",
                    details={
                        "rank": sec.rank,
                        "lead_phase": sec.lead_phase,
                        "days_to_catalyst": sec.days_to_catalyst,
                    },
                    recommendation="Verify catalyst timeline and revenue path",
                ))

        # Check 2: Short runway but top ranked
        if sec.rank is not None and sec.rank <= 10:
            if sec.runway_months is not None and sec.runway_months < self.config.min_runway_months_for_top_rank:
                flags.append(SanityFlag(
                    severity=FlagSeverity.CRITICAL,
                    category=CheckCategory.CROSS_VALIDATION,
                    ticker=sec.ticker,
                    check_name="imminent_dilution_risk",
                    message=f"Runway {sec.runway_months:.1f} months BUT scored as top 10 candidate",
                    details={
                        "rank": sec.rank,
                        "runway_months": float(sec.runway_months),
                        "threshold": float(self.config.min_runway_months_for_top_rank),
                    },
                    recommendation="Imminent dilution risk not reflected in scoring",
                ))

        # Check 3: Trading below cash
        if (sec.cash_mm is not None and
            sec.market_cap_mm is not None and
            sec.market_cap_mm > 0):
            cash_to_mcap = sec.cash_mm / sec.market_cap_mm
            if cash_to_mcap > self.config.cash_to_market_cap_anomaly_ratio:
                flags.append(SanityFlag(
                    severity=FlagSeverity.CRITICAL,
                    category=CheckCategory.CROSS_VALIDATION,
                    ticker=sec.ticker,
                    check_name="trading_below_cash",
                    message=f"Cash ${sec.cash_mm:.0f}M but market cap ${sec.market_cap_mm:.0f}M - trading below cash",
                    details={
                        "cash_mm": float(sec.cash_mm),
                        "market_cap_mm": float(sec.market_cap_mm),
                        "ratio": float(cash_to_mcap),
                    },
                    recommendation="Potential data error or special situation - investigate",
                ))

        return flags

    def _check_institutional_signal(
        self,
        sec: SecurityContext,
    ) -> List[SanityFlag]:
        """
        Check for Institutional Signal contradictions.

        Flags:
        - Zero specialist ownership BUT ranked top 10
        - Elite manager exit BUT high catalyst score
        - 5+ elite managers hold BUT ranked below #50
        """
        flags: List[SanityFlag] = []

        # Check 1: No specialist ownership in top 10
        if sec.rank is not None and sec.rank <= 10:
            if sec.specialist_holder_count is not None and sec.specialist_holder_count == 0:
                flags.append(SanityFlag(
                    severity=FlagSeverity.HIGH,
                    category=CheckCategory.CROSS_VALIDATION,
                    ticker=sec.ticker,
                    check_name="no_smart_money_validation",
                    message=f"13F shows ZERO specialist ownership BUT ranked #{sec.rank}",
                    details={
                        "rank": sec.rank,
                        "specialist_holders": sec.specialist_holder_count,
                        "total_13f_holders": sec.total_13f_holders,
                    },
                    recommendation="No smart money validation - verify thesis",
                ))

        # Check 2: Elite manager exit with high catalyst score
        # (Baker Bros or RA Capital exit but still high catalyst score)
        if (sec.catalyst_score is not None and
            sec.catalyst_score >= Decimal("0.9")):
            # Check if key funds have exited
            if (sec.baker_bros_holds is False or sec.ra_capital_holds is False):
                # Check net position change
                if (sec.net_position_change_pct is not None and
                    sec.net_position_change_pct < Decimal("-0.30")):
                    flags.append(SanityFlag(
                        severity=FlagSeverity.HIGH,
                        category=CheckCategory.CROSS_VALIDATION,
                        ticker=sec.ticker,
                        check_name="negative_institutional_signal_ignored",
                        message=f"Elite manager exit BUT catalyst score = {sec.catalyst_score:.2f}",
                        details={
                            "catalyst_score": float(sec.catalyst_score),
                            "net_position_change_pct": float(sec.net_position_change_pct),
                            "baker_bros_holds": sec.baker_bros_holds,
                            "ra_capital_holds": sec.ra_capital_holds,
                        },
                        recommendation="Negative institutional signal may be ignored",
                    ))

        # Check 3: Strong conviction but low rank
        if sec.rank is not None and sec.rank > 50:
            if (sec.elite_holder_count is not None and
                sec.elite_holder_count >= self.config.min_elite_holders_for_conviction):
                flags.append(SanityFlag(
                    severity=FlagSeverity.MEDIUM,
                    category=CheckCategory.CROSS_VALIDATION,
                    ticker=sec.ticker,
                    check_name="conviction_signal_underweighted",
                    message=f"{sec.elite_holder_count} elite managers hold BUT ranked #{sec.rank}",
                    details={
                        "rank": sec.rank,
                        "elite_holders": sec.elite_holder_count,
                        "threshold": self.config.min_elite_holders_for_conviction,
                    },
                    recommendation="Strong conviction signal may be underweighted",
                ))

        return flags

    def _check_momentum_fundamental(
        self,
        sec: SecurityContext,
    ) -> List[SanityFlag]:
        """
        Check for Momentum-Fundamental contradictions.

        Flags:
        - 3-month return = -40% BUT financial health = A+
        - 6-month return = +200% BUT no clinical catalyst
        - Short interest >40% AND momentum score >0.8
        """
        flags: List[SanityFlag] = []

        # Check 1: Severe drawdown but high financial score
        if (sec.return_3m is not None and
            sec.return_3m <= self.config.severe_drawdown_pct):
            if (sec.financial_score is not None and
                sec.financial_score >= Decimal("80")):
                flags.append(SanityFlag(
                    severity=FlagSeverity.HIGH,
                    category=CheckCategory.CROSS_VALIDATION,
                    ticker=sec.ticker,
                    check_name="market_pricing_risk_not_captured",
                    message=f"3-month return = {sec.return_3m:.0%} BUT financial score = {sec.financial_score:.0f}",
                    details={
                        "return_3m": float(sec.return_3m),
                        "financial_score": float(sec.financial_score),
                    },
                    recommendation="Market pricing in risk not captured by model",
                ))

        # Check 2: Exceptional gain without catalyst
        if (sec.return_6m is not None and
            sec.return_6m >= self.config.exceptional_gain_pct):
            if sec.days_to_catalyst is None or sec.catalyst_type is None:
                flags.append(SanityFlag(
                    severity=FlagSeverity.MEDIUM,
                    category=CheckCategory.CROSS_VALIDATION,
                    ticker=sec.ticker,
                    check_name="momentum_without_driver",
                    message=f"6-month return = {sec.return_6m:.0%} BUT no clinical catalyst reported",
                    details={
                        "return_6m": float(sec.return_6m),
                        "days_to_catalyst": sec.days_to_catalyst,
                        "catalyst_type": sec.catalyst_type,
                    },
                    recommendation="Momentum without fundamental driver - investigate cause",
                ))

        # Check 3: High short interest with high momentum
        if (sec.short_interest_pct is not None and
            sec.short_interest_pct >= self.config.short_interest_warning_pct):
            # Check for high momentum score (using return as proxy)
            if (sec.return_3m is not None and
                sec.return_3m >= Decimal("0.30")):
                flags.append(SanityFlag(
                    severity=FlagSeverity.HIGH,
                    category=CheckCategory.CROSS_VALIDATION,
                    ticker=sec.ticker,
                    check_name="crowded_short_position",
                    message=f"Short interest {sec.short_interest_pct:.0%} with positive momentum {sec.return_3m:.0%}",
                    details={
                        "short_interest_pct": float(sec.short_interest_pct),
                        "return_3m": float(sec.return_3m),
                    },
                    recommendation="Crowded short position creates technical risk",
                ))

        return flags

    def _check_pos_stage(
        self,
        sec: SecurityContext,
    ) -> List[SanityFlag]:
        """
        Check for PoS-Stage contradictions.

        Flags:
        - Phase 2 asset with PoS = 75% (should be ~30%)
        - Phase 3 with PoS <20% still ranked top 20
        - Approved drug with PoS <95%
        """
        flags: List[SanityFlag] = []

        if sec.pos_score is None or sec.lead_phase is None:
            return flags

        # Check 1: Phase 2 with unrealistically high PoS
        if sec.lead_phase == "Phase 2":
            if sec.pos_score > self.config.phase2_pos_max:
                flags.append(SanityFlag(
                    severity=FlagSeverity.CRITICAL,
                    category=CheckCategory.CROSS_VALIDATION,
                    ticker=sec.ticker,
                    check_name="pos_calibration_error",
                    message=f"Phase 2 asset with PoS = {sec.pos_score:.0%} (should be ~30%)",
                    details={
                        "lead_phase": sec.lead_phase,
                        "pos_score": float(sec.pos_score),
                        "expected_max": float(self.config.phase2_pos_max),
                    },
                    recommendation="PoS calibration error or exceptional modifiers - verify",
                ))

        # Check 2: Phase 3 with very low PoS still ranked highly
        if sec.lead_phase == "Phase 3":
            if sec.pos_score < self.config.phase3_pos_min:
                if sec.rank is not None and sec.rank <= 20:
                    flags.append(SanityFlag(
                        severity=FlagSeverity.HIGH,
                        category=CheckCategory.CROSS_VALIDATION,
                        ticker=sec.ticker,
                        check_name="binary_risk_underweighted",
                        message=f"Phase 3 with PoS = {sec.pos_score:.0%} still ranked #{sec.rank}",
                        details={
                            "lead_phase": sec.lead_phase,
                            "pos_score": float(sec.pos_score),
                            "rank": sec.rank,
                            "threshold": float(self.config.phase3_pos_min),
                        },
                        recommendation="Binary risk may be underweighted",
                    ))

        # Check 3: Approved with low PoS
        if sec.lead_phase in ("Approved", "Phase 4"):
            if sec.pos_score < self.config.approved_pos_min:
                flags.append(SanityFlag(
                    severity=FlagSeverity.CRITICAL,
                    category=CheckCategory.CROSS_VALIDATION,
                    ticker=sec.ticker,
                    check_name="stage_classification_error",
                    message=f"Approved drug with PoS = {sec.pos_score:.0%} (should be >95%)",
                    details={
                        "lead_phase": sec.lead_phase,
                        "pos_score": float(sec.pos_score),
                        "expected_min": float(self.config.approved_pos_min),
                    },
                    recommendation="Stage classification error - verify status",
                ))

        return flags

    def _calculate_metrics(
        self,
        flags: List[SanityFlag],
        securities: List[SecurityContext],
    ) -> Dict[str, Any]:
        """Calculate summary metrics for the check results."""
        by_type = {}
        for contradiction_type in ContradictionType:
            by_type[contradiction_type.value] = sum(
                1 for f in flags
                if contradiction_type.value in f.check_name
            )

        # Count tickers with issues
        tickers_with_issues = len(set(f.ticker for f in flags if f.ticker))

        return {
            "total_securities_checked": len(securities),
            "total_flags": len(flags),
            "tickers_with_issues": tickers_with_issues,
            "by_severity": {
                "critical": sum(1 for f in flags if f.severity == FlagSeverity.CRITICAL),
                "high": sum(1 for f in flags if f.severity == FlagSeverity.HIGH),
                "medium": sum(1 for f in flags if f.severity == FlagSeverity.MEDIUM),
                "low": sum(1 for f in flags if f.severity == FlagSeverity.LOW),
            },
            "by_check": {
                f.check_name: sum(1 for f2 in flags if f2.check_name == f.check_name)
                for f in flags
            },
        }


def check_financial_clinical_contradictions(
    securities: List[SecurityContext],
    config: Optional[ThresholdConfig] = None,
) -> List[SanityFlag]:
    """
    Convenience function to run only financial-clinical checks.

    Args:
        securities: Securities to check
        config: Optional threshold configuration

    Returns:
        List of flags
    """
    checker = CrossValidationChecker(config)
    flags = []
    for sec in securities:
        if sec.rank is not None:
            flags.extend(checker._check_financial_clinical(sec))
    return flags


def check_institutional_contradictions(
    securities: List[SecurityContext],
    config: Optional[ThresholdConfig] = None,
) -> List[SanityFlag]:
    """
    Convenience function to run only institutional signal checks.

    Args:
        securities: Securities to check
        config: Optional threshold configuration

    Returns:
        List of flags
    """
    checker = CrossValidationChecker(config)
    flags = []
    for sec in securities:
        if sec.rank is not None:
            flags.extend(checker._check_institutional_signal(sec))
    return flags
