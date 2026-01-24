"""
Sanity Check Type Definitions

Core data structures for the sanity check framework.

Design Philosophy:
- IMMUTABLE: Frozen dataclasses for results
- DECIMAL-ONLY: All numeric comparisons use Decimal
- SERIALIZABLE: All types can be converted to JSON
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union


class FlagSeverity(str, Enum):
    """
    Severity levels for sanity check flags.

    Matches investment operations escalation levels:
    - CRITICAL: Likely data error - block IC review
    - HIGH: Logical inconsistency - requires explanation
    - MEDIUM: Unusual pattern - monitor closely
    - LOW: Notable but reasonable
    """
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

    def __lt__(self, other: "FlagSeverity") -> bool:
        order = {
            FlagSeverity.LOW: 0,
            FlagSeverity.MEDIUM: 1,
            FlagSeverity.HIGH: 2,
            FlagSeverity.CRITICAL: 3,
        }
        return order[self] < order[other]


class CheckCategory(str, Enum):
    """Categories of sanity checks."""
    CROSS_VALIDATION = "cross_validation"
    BENCHMARK = "benchmark"
    TIME_SERIES = "time_series"
    DOMAIN_EXPERT = "domain_expert"
    MARKET_MICROSTRUCTURE = "market_microstructure"
    PORTFOLIO_CONSTRUCTION = "portfolio_construction"
    REGRESSION = "regression"
    REVIEW_TRIGGER = "review_trigger"
    DASHBOARD = "dashboard"


class ContradictionType(str, Enum):
    """Types of cross-validation contradictions."""
    FINANCIAL_CLINICAL = "financial_clinical"
    INSTITUTIONAL_SIGNAL = "institutional_signal"
    MOMENTUM_FUNDAMENTAL = "momentum_fundamental"
    POS_STAGE = "pos_stage"
    DATA_CONSISTENCY = "data_consistency"


class ReviewLevel(str, Enum):
    """Review requirement levels."""
    DIRECTOR = "director"
    SENIOR_ANALYST = "senior_analyst"
    JUNIOR_ANALYST = "junior_analyst"
    AUTOMATED = "automated"


@dataclass(frozen=True)
class SanityFlag:
    """
    A single sanity check flag.

    Immutable record of a detected issue.
    """
    severity: FlagSeverity
    category: CheckCategory
    ticker: Optional[str]
    check_name: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    recommendation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dict."""
        d = {
            "severity": self.severity.value,
            "category": self.category.value,
            "ticker": self.ticker,
            "check_name": self.check_name,
            "message": self.message,
        }
        if self.details:
            d["details"] = self.details
        if self.recommendation:
            d["recommendation"] = self.recommendation
        return d


@dataclass
class SanityCheckResult:
    """
    Result of a single sanity check.

    Contains flags generated and pass/fail status.
    """
    check_name: str
    category: CheckCategory
    passed: bool
    flags: List[SanityFlag] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def critical_count(self) -> int:
        return sum(1 for f in self.flags if f.severity == FlagSeverity.CRITICAL)

    @property
    def high_count(self) -> int:
        return sum(1 for f in self.flags if f.severity == FlagSeverity.HIGH)

    @property
    def max_severity(self) -> Optional[FlagSeverity]:
        if not self.flags:
            return None
        return max(f.severity for f in self.flags)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "check_name": self.check_name,
            "category": self.category.value,
            "passed": self.passed,
            "flags": [f.to_dict() for f in self.flags],
            "metrics": self.metrics,
            "critical_count": self.critical_count,
            "high_count": self.high_count,
        }


@dataclass
class ReviewRequirement:
    """
    Manual review requirement for a candidate.
    """
    ticker: str
    rank: int
    level: ReviewLevel
    reasons: List[str]
    requires_memo: bool = False
    requires_sign_off: bool = False
    blocking: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "rank": self.rank,
            "level": self.level.value,
            "reasons": self.reasons,
            "requires_memo": self.requires_memo,
            "requires_sign_off": self.requires_sign_off,
            "blocking": self.blocking,
        }


@dataclass
class ValidationReport:
    """
    Complete validation report from all sanity checks.
    """
    as_of_date: str
    check_results: List[SanityCheckResult] = field(default_factory=list)
    review_requirements: List[ReviewRequirement] = field(default_factory=list)

    @property
    def total_flags(self) -> int:
        return sum(len(r.flags) for r in self.check_results)

    @property
    def critical_flags(self) -> List[SanityFlag]:
        flags = []
        for result in self.check_results:
            flags.extend(f for f in result.flags if f.severity == FlagSeverity.CRITICAL)
        return flags

    @property
    def high_flags(self) -> List[SanityFlag]:
        flags = []
        for result in self.check_results:
            flags.extend(f for f in result.flags if f.severity == FlagSeverity.HIGH)
        return flags

    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self.check_results)

    @property
    def ic_review_blocked(self) -> bool:
        """IC review should be blocked if any CRITICAL flags exist."""
        return len(self.critical_flags) > 0

    @property
    def verdict(self) -> str:
        if self.ic_review_blocked:
            return "BLOCKED"
        elif self.high_flags:
            return "INVESTIGATE"
        elif not self.all_passed:
            return "REVIEW"
        else:
            return "PROCEED"

    def get_flags_by_ticker(self, ticker: str) -> List[SanityFlag]:
        """Get all flags for a specific ticker."""
        flags = []
        for result in self.check_results:
            flags.extend(f for f in result.flags if f.ticker == ticker)
        return flags

    def get_flags_by_category(self, category: CheckCategory) -> List[SanityFlag]:
        """Get all flags for a specific category."""
        flags = []
        for result in self.check_results:
            if result.category == category:
                flags.extend(result.flags)
        return flags

    def to_dict(self) -> Dict[str, Any]:
        return {
            "as_of_date": self.as_of_date,
            "verdict": self.verdict,
            "ic_review_blocked": self.ic_review_blocked,
            "total_flags": self.total_flags,
            "critical_count": len(self.critical_flags),
            "high_count": len(self.high_flags),
            "all_passed": self.all_passed,
            "check_results": [r.to_dict() for r in self.check_results],
            "review_requirements": [r.to_dict() for r in self.review_requirements],
        }


# Type aliases for common patterns
TickerData = Dict[str, Any]
RankedSecurities = List[Dict[str, Any]]
TrialRecords = List[Dict[str, Any]]
HoldingsData = Dict[str, Any]


@dataclass(frozen=True)
class ThresholdConfig:
    """
    Configuration thresholds for sanity checks.

    Immutable to prevent modification during runs.
    """
    # Financial-Clinical thresholds
    min_runway_months_for_top_rank: Decimal = Decimal("6")
    cash_to_market_cap_anomaly_ratio: Decimal = Decimal("2.5")
    phase3_catalyst_window_months: int = 12

    # Institutional thresholds
    min_elite_holders_for_conviction: int = 5
    elite_exit_lookback_quarters: int = 2
    min_specialist_ownership_for_top10: int = 1

    # Momentum thresholds
    severe_drawdown_pct: Decimal = Decimal("-0.40")
    exceptional_gain_pct: Decimal = Decimal("2.00")
    short_interest_warning_pct: Decimal = Decimal("0.40")

    # PoS thresholds
    phase2_pos_max: Decimal = Decimal("0.45")
    phase3_pos_min: Decimal = Decimal("0.20")
    approved_pos_min: Decimal = Decimal("0.95")

    # Portfolio construction
    max_same_sector_in_top10: int = 5
    max_same_catalyst_window_days: int = 14
    min_adv_for_position: Decimal = Decimal("500000")
    max_position_impact_pct: Decimal = Decimal("0.02")

    # Time series
    max_rank_jump_single_week: int = 15
    min_rank_correlation_weekly: Decimal = Decimal("0.30")
    max_top_quintile_churn: Decimal = Decimal("0.50")

    # Review triggers
    top10_review_level: ReviewLevel = ReviewLevel.DIRECTOR
    top20_review_level: ReviewLevel = ReviewLevel.SENIOR_ANALYST
    top50_review_level: ReviewLevel = ReviewLevel.JUNIOR_ANALYST


# Default configuration
DEFAULT_THRESHOLDS = ThresholdConfig()


@dataclass
class SecurityContext:
    """
    Complete context for a security being checked.

    Aggregates data from all modules for cross-validation.
    """
    ticker: str
    rank: Optional[int] = None
    composite_score: Optional[Decimal] = None

    # Financial metrics
    market_cap_mm: Optional[Decimal] = None
    cash_mm: Optional[Decimal] = None
    runway_months: Optional[Decimal] = None
    financial_score: Optional[Decimal] = None
    burn_rate_mm: Optional[Decimal] = None

    # Clinical metrics
    clinical_score: Optional[Decimal] = None
    lead_phase: Optional[str] = None
    trial_count: Optional[int] = None
    pos_score: Optional[Decimal] = None

    # Catalyst metrics
    catalyst_score: Optional[Decimal] = None
    days_to_catalyst: Optional[int] = None
    catalyst_type: Optional[str] = None
    next_catalyst_date: Optional[str] = None

    # Momentum/Market metrics
    return_3m: Optional[Decimal] = None
    return_6m: Optional[Decimal] = None
    return_12m: Optional[Decimal] = None
    short_interest_pct: Optional[Decimal] = None
    adv_dollars: Optional[Decimal] = None
    volatility: Optional[Decimal] = None

    # Institutional metrics
    elite_holder_count: Optional[int] = None
    specialist_holder_count: Optional[int] = None
    total_13f_holders: Optional[int] = None
    net_position_change_pct: Optional[Decimal] = None
    baker_bros_holds: Optional[bool] = None
    ra_capital_holds: Optional[bool] = None

    # Stage/Sector
    stage_bucket: Optional[str] = None
    sector: Optional[str] = None
    indication: Optional[str] = None

    # Flags
    flags: List[str] = field(default_factory=list)

    @property
    def is_pre_revenue(self) -> bool:
        """Check if company is pre-revenue (typical for early biotech)."""
        return self.lead_phase in ("Phase 1", "Phase 2", "Preclinical")

    @property
    def has_near_term_catalyst(self) -> bool:
        """Check if catalyst is within 6 months."""
        if self.days_to_catalyst is None:
            return False
        return self.days_to_catalyst <= 180

    @property
    def is_micro_cap(self) -> bool:
        """Check if market cap < $200M."""
        if self.market_cap_mm is None:
            return False
        return self.market_cap_mm < Decimal("200")


@dataclass
class RankingSnapshot:
    """
    A point-in-time snapshot of rankings.

    Used for time series analysis.
    """
    as_of_date: str
    securities: List[SecurityContext] = field(default_factory=list)

    def get_by_ticker(self, ticker: str) -> Optional[SecurityContext]:
        for sec in self.securities:
            if sec.ticker == ticker:
                return sec
        return None

    def get_top_n(self, n: int) -> List[SecurityContext]:
        sorted_secs = sorted(
            [s for s in self.securities if s.rank is not None],
            key=lambda x: x.rank
        )
        return sorted_secs[:n]

    def get_by_rank(self, rank: int) -> Optional[SecurityContext]:
        for sec in self.securities:
            if sec.rank == rank:
                return sec
        return None


@dataclass
class GoldenTestCase:
    """
    A historical test case for regression validation.
    """
    ticker: str
    as_of_date: str
    expected_outcome: str  # "top_20", "bottom_50", etc.
    case_type: str  # "positive", "negative", "edge"
    description: str
    threshold_rank: Optional[int] = None
    actual_outcome: Optional[str] = None
    passed: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "as_of_date": self.as_of_date,
            "expected_outcome": self.expected_outcome,
            "case_type": self.case_type,
            "description": self.description,
            "threshold_rank": self.threshold_rank,
            "actual_outcome": self.actual_outcome,
            "passed": self.passed,
        }
