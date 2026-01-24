"""
Query 7.8: Expert Override & Manual Review Triggers

Implements automated triggers that force human review before IC presentation:

1. Auto-Flag for Manual Review Triggers
2. Mandatory Analyst Sign-Off Requirements
3. Audit Trail Documentation
4. Investment Committee Readiness

These checks ensure appropriate human oversight for high-stakes decisions.

Author: Wake Robin Capital Management
Version: 1.0.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Set

from sanity_checks.types import (
    CheckCategory,
    FlagSeverity,
    ReviewLevel,
    ReviewRequirement,
    SanityCheckResult,
    SanityFlag,
    SecurityContext,
    ThresholdConfig,
    DEFAULT_THRESHOLDS,
)

logger = logging.getLogger(__name__)


@dataclass
class OverrideRecord:
    """Record of a manual override decision."""
    ticker: str
    override_type: str
    original_rank: Optional[int]
    adjusted_rank: Optional[int]
    justification: str
    reviewer_id: str
    review_level: ReviewLevel
    approved: bool


@dataclass
class ICReadinessItem:
    """Investment committee readiness checklist item."""
    item: str
    required: bool
    present: bool
    ticker: Optional[str] = None


@dataclass
class ICDocumentation:
    """IC documentation requirements for a candidate."""
    ticker: str
    rank: int
    has_thesis: bool
    has_risks: bool
    has_catalyst_timeline: bool
    has_competitive_analysis: bool
    has_valuation: bool
    complete: bool


class ReviewTriggerChecker:
    """
    Review trigger checker.

    Determines when human review is required and tracks
    oversight requirements.
    """

    # Review level thresholds
    DIRECTOR_THRESHOLD = 10
    SENIOR_ANALYST_THRESHOLD = 20
    JUNIOR_ANALYST_THRESHOLD = 50

    def __init__(self, config: Optional[ThresholdConfig] = None) -> None:
        self.config = config or DEFAULT_THRESHOLDS

    def run_all_checks(
        self,
        securities: List[SecurityContext],
        previous_ranks: Optional[Dict[str, int]] = None,
        documentation: Optional[Dict[str, ICDocumentation]] = None,
    ) -> SanityCheckResult:
        """
        Run all review trigger checks.

        Args:
            securities: List of security contexts
            previous_ranks: Optional previous week's ranks
            documentation: Optional IC documentation status

        Returns:
            SanityCheckResult with review requirements
        """
        flags: List[SanityFlag] = []
        review_requirements: List[ReviewRequirement] = []

        # 1. Determine Review Requirements for Each Security
        for sec in securities:
            if sec.rank is None:
                continue

            # Get review requirements based on rank
            requirements = self._get_review_requirements(
                sec, previous_ranks, documentation
            )
            if requirements:
                review_requirements.append(requirements)

        # 2. Check for Auto-Flag Conditions
        auto_flags = self._check_auto_flag_conditions(securities, previous_ranks)
        flags.extend(auto_flags)

        # 3. IC Readiness Validation
        if documentation:
            readiness_flags = self._check_ic_readiness(securities, documentation)
            flags.extend(readiness_flags)

        # Calculate metrics
        metrics = self._calculate_metrics(flags, review_requirements)

        # Determine if IC presentation is blocked
        ic_blocked = any(
            r.blocking for r in review_requirements
        ) or any(
            f.severity == FlagSeverity.CRITICAL for f in flags
        )

        passed = not ic_blocked

        result = SanityCheckResult(
            check_name="review_triggers",
            category=CheckCategory.REVIEW_TRIGGER,
            passed=passed,
            flags=flags,
            metrics=metrics,
        )

        # Attach review requirements to metrics
        result.metrics["review_requirements"] = [r.to_dict() for r in review_requirements]

        return result

    def _get_review_requirements(
        self,
        sec: SecurityContext,
        previous_ranks: Optional[Dict[str, int]],
        documentation: Optional[Dict[str, ICDocumentation]],
    ) -> Optional[ReviewRequirement]:
        """Determine review requirements for a security."""
        if sec.rank is None:
            return None

        reasons: List[str] = []
        blocking = False
        requires_memo = False
        requires_sign_off = False

        # Determine base review level
        if sec.rank <= self.DIRECTOR_THRESHOLD:
            level = ReviewLevel.DIRECTOR
            requires_memo = True
            requires_sign_off = True
        elif sec.rank <= self.SENIOR_ANALYST_THRESHOLD:
            level = ReviewLevel.SENIOR_ANALYST
            requires_sign_off = True
        elif sec.rank <= self.JUNIOR_ANALYST_THRESHOLD:
            level = ReviewLevel.JUNIOR_ANALYST
        else:
            level = ReviewLevel.AUTOMATED

        # Add specific review triggers
        if sec.rank <= 30:
            # Check for UNKNOWN score components
            if sec.clinical_score is None:
                reasons.append("Clinical score is UNKNOWN")
                if sec.rank <= 10:
                    blocking = True

            if sec.financial_score is None:
                reasons.append("Financial score is UNKNOWN")
                if sec.rank <= 10:
                    blocking = True

            if sec.catalyst_score is None:
                reasons.append("Catalyst score is UNKNOWN")

        # Check rank change
        if previous_ranks and sec.ticker in previous_ranks:
            prev_rank = previous_ranks[sec.ticker]
            rank_change = abs(sec.rank - prev_rank)
            if rank_change > self.config.max_rank_jump_single_week:
                reasons.append(f"Rank changed {rank_change} positions week-over-week")
                if sec.rank <= 10:
                    requires_memo = True

        # Check PoS deviation
        if sec.pos_score is not None and sec.lead_phase:
            base_rates = {
                "Phase 1": Decimal("0.10"),
                "Phase 2": Decimal("0.30"),
                "Phase 3": Decimal("0.60"),
            }
            if sec.lead_phase in base_rates:
                base = base_rates[sec.lead_phase]
                deviation = abs(sec.pos_score - base)
                if deviation > Decimal("0.20"):
                    reasons.append(f"PoS deviates {deviation:.0%} from base rate")

        # Check for zero institutional holders
        if sec.total_13f_holders == 0 and sec.rank <= 20:
            reasons.append("Zero 13F holders")

        # Check catalyst timing
        if sec.days_to_catalyst is not None:
            if sec.days_to_catalyst > 365 and sec.rank <= 10:
                reasons.append("Catalyst >12 months out for top 10 candidate")

        # Check documentation completeness
        if documentation and sec.ticker in documentation:
            doc = documentation[sec.ticker]
            if sec.rank <= 10 and not doc.complete:
                blocking = True
                reasons.append("IC documentation incomplete")

        if not reasons:
            return None

        return ReviewRequirement(
            ticker=sec.ticker,
            rank=sec.rank,
            level=level,
            reasons=reasons,
            requires_memo=requires_memo,
            requires_sign_off=requires_sign_off,
            blocking=blocking,
        )

    def _check_auto_flag_conditions(
        self,
        securities: List[SecurityContext],
        previous_ranks: Optional[Dict[str, int]],
    ) -> List[SanityFlag]:
        """Check conditions that auto-flag for manual review."""
        flags: List[SanityFlag] = []

        for sec in securities:
            if sec.rank is None:
                continue

            # UNKNOWN components in top 30
            if sec.rank <= 30:
                unknown_components = []
                if sec.clinical_score is None:
                    unknown_components.append("clinical")
                if sec.financial_score is None:
                    unknown_components.append("financial")
                if sec.catalyst_score is None:
                    unknown_components.append("catalyst")

                if unknown_components:
                    flags.append(SanityFlag(
                        severity=FlagSeverity.HIGH if sec.rank <= 10 else FlagSeverity.MEDIUM,
                        category=CheckCategory.REVIEW_TRIGGER,
                        ticker=sec.ticker,
                        check_name="unknown_score_components",
                        message=f"Top {sec.rank} has UNKNOWN components: {', '.join(unknown_components)}",
                        details={
                            "rank": sec.rank,
                            "unknown_components": unknown_components,
                        },
                        recommendation="Manual review required - score data incomplete",
                    ))

            # Large rank changes
            if previous_ranks and sec.ticker in previous_ranks:
                prev_rank = previous_ranks[sec.ticker]
                rank_change = abs(sec.rank - prev_rank)

                if rank_change > self.config.max_rank_jump_single_week and sec.rank <= 50:
                    flags.append(SanityFlag(
                        severity=FlagSeverity.MEDIUM,
                        category=CheckCategory.REVIEW_TRIGGER,
                        ticker=sec.ticker,
                        check_name="large_rank_change",
                        message=f"Rank changed {rank_change} positions ({prev_rank} -> {sec.rank})",
                        details={
                            "previous_rank": prev_rank,
                            "current_rank": sec.rank,
                            "change": rank_change,
                        },
                        recommendation="Document rank change explanation",
                    ))

            # Financial coverage check
            if sec.rank <= 20:
                coverage_issues = []
                if sec.cash_mm is None:
                    coverage_issues.append("cash")
                if sec.runway_months is None:
                    coverage_issues.append("runway")
                if sec.market_cap_mm is None:
                    coverage_issues.append("market_cap")

                coverage_pct = 1 - (len(coverage_issues) / 3)
                if coverage_pct < 0.5:
                    flags.append(SanityFlag(
                        severity=FlagSeverity.HIGH,
                        category=CheckCategory.REVIEW_TRIGGER,
                        ticker=sec.ticker,
                        check_name="low_financial_coverage",
                        message=f"Financial coverage {coverage_pct:.0%} for top {sec.rank} candidate",
                        details={
                            "rank": sec.rank,
                            "missing": coverage_issues,
                            "coverage_pct": coverage_pct,
                        },
                        recommendation="Insufficient financial data for top candidate",
                    ))

        return flags

    def _check_ic_readiness(
        self,
        securities: List[SecurityContext],
        documentation: Dict[str, ICDocumentation],
    ) -> List[SanityFlag]:
        """Check IC readiness for top candidates."""
        flags: List[SanityFlag] = []

        for sec in securities:
            if sec.rank is None or sec.rank > 10:
                continue

            doc = documentation.get(sec.ticker)

            if doc is None:
                flags.append(SanityFlag(
                    severity=FlagSeverity.CRITICAL,
                    category=CheckCategory.REVIEW_TRIGGER,
                    ticker=sec.ticker,
                    check_name="missing_ic_documentation",
                    message=f"Top {sec.rank} candidate missing all IC documentation",
                    details={"rank": sec.rank},
                    recommendation="BLOCK IC presentation - documentation required",
                ))
                continue

            missing = []
            if not doc.has_thesis:
                missing.append("investment thesis")
            if not doc.has_risks:
                missing.append("key risks")
            if not doc.has_catalyst_timeline:
                missing.append("catalyst timeline")
            if not doc.has_competitive_analysis:
                missing.append("competitive positioning")
            if not doc.has_valuation:
                missing.append("valuation framework")

            if missing:
                flags.append(SanityFlag(
                    severity=FlagSeverity.CRITICAL if len(missing) >= 2 else FlagSeverity.HIGH,
                    category=CheckCategory.REVIEW_TRIGGER,
                    ticker=sec.ticker,
                    check_name="incomplete_ic_documentation",
                    message=f"Top {sec.rank} missing: {', '.join(missing)}",
                    details={
                        "rank": sec.rank,
                        "missing": missing,
                        "has_thesis": doc.has_thesis,
                        "has_risks": doc.has_risks,
                        "has_catalyst_timeline": doc.has_catalyst_timeline,
                        "has_competitive_analysis": doc.has_competitive_analysis,
                        "has_valuation": doc.has_valuation,
                    },
                    recommendation="Complete IC documentation before presentation",
                ))

        return flags

    def _calculate_metrics(
        self,
        flags: List[SanityFlag],
        requirements: List[ReviewRequirement],
    ) -> Dict[str, Any]:
        """Calculate review trigger metrics."""
        by_level = {
            ReviewLevel.DIRECTOR.value: sum(1 for r in requirements if r.level == ReviewLevel.DIRECTOR),
            ReviewLevel.SENIOR_ANALYST.value: sum(1 for r in requirements if r.level == ReviewLevel.SENIOR_ANALYST),
            ReviewLevel.JUNIOR_ANALYST.value: sum(1 for r in requirements if r.level == ReviewLevel.JUNIOR_ANALYST),
        }

        return {
            "total_flags": len(flags),
            "total_review_required": len(requirements),
            "blocking_count": sum(1 for r in requirements if r.blocking),
            "by_review_level": by_level,
            "by_severity": {
                "critical": sum(1 for f in flags if f.severity == FlagSeverity.CRITICAL),
                "high": sum(1 for f in flags if f.severity == FlagSeverity.HIGH),
                "medium": sum(1 for f in flags if f.severity == FlagSeverity.MEDIUM),
                "low": sum(1 for f in flags if f.severity == FlagSeverity.LOW),
            },
        }


def track_override(
    ticker: str,
    override_type: str,
    original_rank: Optional[int],
    adjusted_rank: Optional[int],
    justification: str,
    reviewer_id: str,
    review_level: ReviewLevel,
    approved: bool,
) -> OverrideRecord:
    """
    Create an override record for audit trail.

    Args:
        ticker: Affected ticker
        override_type: Type of override (e.g., "rank_adjustment", "exclusion")
        original_rank: Original model rank
        adjusted_rank: Adjusted rank after override
        justification: Documented justification
        reviewer_id: ID of reviewer
        review_level: Level of review
        approved: Whether override was approved

    Returns:
        OverrideRecord for audit trail
    """
    return OverrideRecord(
        ticker=ticker,
        override_type=override_type,
        original_rank=original_rank,
        adjusted_rank=adjusted_rank,
        justification=justification,
        reviewer_id=reviewer_id,
        review_level=review_level,
        approved=approved,
    )


def generate_review_summary(
    requirements: List[ReviewRequirement],
) -> Dict[str, Any]:
    """
    Generate summary of review requirements.

    Args:
        requirements: List of review requirements

    Returns:
        Summary report dict
    """
    return {
        "total_requiring_review": len(requirements),
        "blocking": [r.to_dict() for r in requirements if r.blocking],
        "director_level": [r.to_dict() for r in requirements if r.level == ReviewLevel.DIRECTOR],
        "senior_analyst": [r.to_dict() for r in requirements if r.level == ReviewLevel.SENIOR_ANALYST],
        "junior_analyst": [r.to_dict() for r in requirements if r.level == ReviewLevel.JUNIOR_ANALYST],
        "requires_memo": [r.ticker for r in requirements if r.requires_memo],
        "requires_sign_off": [r.ticker for r in requirements if r.requires_sign_off],
    }
