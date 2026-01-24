#!/usr/bin/env python3
"""
staleness_gates.py - Phase-Dependent Data Staleness Gates

Enforces data freshness requirements that vary by clinical trial phase
and data type. More critical data (Phase 3, financial) requires fresher inputs.

P1 Enhancement: Prevents stale data from contaminating production scores.

Design Philosophy:
- DETERMINISTIC: All thresholds are explicit constants
- FAIL-LOUD: Hard gates raise exceptions, soft gates return severity
- PIT-SAFE: Uses as_of_date for all comparisons

Author: Wake Robin Capital Management
Version: 1.0.0
"""

from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

__version__ = "1.0.0"


class StalenessAction(str, Enum):
    """Action to take when staleness is detected."""
    PASS = "PASS"           # Data is fresh
    WARN = "WARN"           # Data is aging, log warning
    SOFT_GATE = "SOFT_GATE" # Apply penalty but continue
    HARD_GATE = "HARD_GATE" # Fail pipeline / exclude ticker


class DataType(str, Enum):
    """Types of data with different staleness requirements."""
    FINANCIAL = "financial"
    TRIAL = "trial"
    MARKET = "market"
    SHORT_INTEREST = "short_interest"
    HOLDINGS_13F = "holdings_13f"


@dataclass(frozen=True)
class StalenessThreshold:
    """Staleness thresholds for a specific data type/phase combination."""
    data_type: DataType
    phase: Optional[str]  # None = applies to all phases
    warn_days: int        # Days before warning
    soft_gate_days: int   # Days before penalty applied
    hard_gate_days: int   # Days before exclusion/failure
    penalty_multiplier: Decimal  # Score multiplier when soft-gated (e.g., 0.5)


# Default staleness thresholds by data type and phase
DEFAULT_STALENESS_THRESHOLDS: List[StalenessThreshold] = [
    # Financial data - stricter for all phases
    StalenessThreshold(DataType.FINANCIAL, None, warn_days=60, soft_gate_days=90, hard_gate_days=120, penalty_multiplier=Decimal("0.5")),

    # Trial data - varies by phase
    StalenessThreshold(DataType.TRIAL, "phase_3", warn_days=90, soft_gate_days=120, hard_gate_days=180, penalty_multiplier=Decimal("0.6")),
    StalenessThreshold(DataType.TRIAL, "phase_2", warn_days=180, soft_gate_days=270, hard_gate_days=365, penalty_multiplier=Decimal("0.7")),
    StalenessThreshold(DataType.TRIAL, "phase_1", warn_days=270, soft_gate_days=365, hard_gate_days=545, penalty_multiplier=Decimal("0.8")),
    StalenessThreshold(DataType.TRIAL, None, warn_days=180, soft_gate_days=270, hard_gate_days=365, penalty_multiplier=Decimal("0.7")),

    # Market data - needs to be very fresh
    StalenessThreshold(DataType.MARKET, None, warn_days=3, soft_gate_days=5, hard_gate_days=10, penalty_multiplier=Decimal("0.3")),

    # Short interest - FINRA data is already 2-week delayed
    StalenessThreshold(DataType.SHORT_INTEREST, None, warn_days=20, soft_gate_days=30, hard_gate_days=45, penalty_multiplier=Decimal("0.5")),

    # 13F holdings - 45-day SEC lag built in, so threshold is relative to filing date
    StalenessThreshold(DataType.HOLDINGS_13F, None, warn_days=60, soft_gate_days=90, hard_gate_days=135, penalty_multiplier=Decimal("0.4")),
]


@dataclass
class StalenessCheckResult:
    """Result of a staleness check."""
    data_type: DataType
    phase: Optional[str]
    data_date: Optional[date]
    as_of_date: date
    age_days: int
    action: StalenessAction
    penalty_multiplier: Decimal
    threshold_used: StalenessThreshold
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "data_type": self.data_type.value,
            "phase": self.phase,
            "data_date": self.data_date.isoformat() if self.data_date else None,
            "as_of_date": self.as_of_date.isoformat(),
            "age_days": self.age_days,
            "action": self.action.value,
            "penalty_multiplier": str(self.penalty_multiplier),
            "message": self.message,
        }


class StalenessGateEngine:
    """
    Engine for enforcing data staleness gates.

    Provides phase-aware staleness checking with configurable thresholds.

    Usage:
        engine = StalenessGateEngine()
        result = engine.check_staleness(
            data_type=DataType.TRIAL,
            data_date=date(2025, 10, 1),
            as_of_date=date(2026, 1, 15),
            phase="phase_3"
        )
        if result.action == StalenessAction.HARD_GATE:
            raise ValueError(f"Data too stale: {result.message}")
    """

    VERSION = "1.0.0"

    def __init__(
        self,
        thresholds: Optional[List[StalenessThreshold]] = None,
        strict_mode: bool = False,
    ):
        """
        Initialize staleness gate engine.

        Args:
            thresholds: Custom thresholds (uses defaults if None)
            strict_mode: If True, soft gates become hard gates
        """
        self.thresholds = thresholds or DEFAULT_STALENESS_THRESHOLDS
        self.strict_mode = strict_mode
        self._threshold_cache: Dict[Tuple[DataType, Optional[str]], StalenessThreshold] = {}
        self._build_cache()

    def _build_cache(self) -> None:
        """Build lookup cache for thresholds."""
        # Phase-specific thresholds take priority
        for t in self.thresholds:
            key = (t.data_type, t.phase)
            self._threshold_cache[key] = t

    def _get_threshold(
        self,
        data_type: DataType,
        phase: Optional[str],
    ) -> StalenessThreshold:
        """Get threshold for data type/phase combination."""
        # Try phase-specific first
        key = (data_type, phase)
        if key in self._threshold_cache:
            return self._threshold_cache[key]

        # Fall back to generic (phase=None)
        generic_key = (data_type, None)
        if generic_key in self._threshold_cache:
            return self._threshold_cache[generic_key]

        # Ultimate fallback - very lenient
        return StalenessThreshold(
            data_type=data_type,
            phase=None,
            warn_days=365,
            soft_gate_days=545,
            hard_gate_days=730,
            penalty_multiplier=Decimal("0.5"),
        )

    def check_staleness(
        self,
        data_type: DataType,
        data_date: Optional[date],
        as_of_date: date,
        phase: Optional[str] = None,
    ) -> StalenessCheckResult:
        """
        Check staleness of data against thresholds.

        Args:
            data_type: Type of data being checked
            data_date: Date of the data (None = unknown)
            as_of_date: Analysis date for comparison
            phase: Clinical phase (for trial data)

        Returns:
            StalenessCheckResult with action and penalty
        """
        threshold = self._get_threshold(data_type, phase)

        # Handle unknown data date
        if data_date is None:
            return StalenessCheckResult(
                data_type=data_type,
                phase=phase,
                data_date=None,
                as_of_date=as_of_date,
                age_days=-1,
                action=StalenessAction.SOFT_GATE,
                penalty_multiplier=threshold.penalty_multiplier,
                threshold_used=threshold,
                message=f"Unknown data date for {data_type.value}. Applying soft gate penalty.",
            )

        age_days = (as_of_date - data_date).days

        # Future data is invalid (lookahead bias)
        if age_days < 0:
            return StalenessCheckResult(
                data_type=data_type,
                phase=phase,
                data_date=data_date,
                as_of_date=as_of_date,
                age_days=age_days,
                action=StalenessAction.HARD_GATE,
                penalty_multiplier=Decimal("0"),
                threshold_used=threshold,
                message=f"LOOKAHEAD BIAS: {data_type.value} data from {data_date} is AFTER as_of_date {as_of_date}",
            )

        # Determine action based on age
        if age_days > threshold.hard_gate_days:
            action = StalenessAction.HARD_GATE
            penalty = Decimal("0")
            message = f"{data_type.value} data is {age_days} days old (limit: {threshold.hard_gate_days}). HARD GATE triggered."
        elif age_days > threshold.soft_gate_days:
            action = StalenessAction.HARD_GATE if self.strict_mode else StalenessAction.SOFT_GATE
            penalty = threshold.penalty_multiplier
            message = f"{data_type.value} data is {age_days} days old (soft gate: {threshold.soft_gate_days}). Applying {penalty}x penalty."
        elif age_days > threshold.warn_days:
            action = StalenessAction.WARN
            penalty = Decimal("1.0")
            message = f"{data_type.value} data is {age_days} days old (warn: {threshold.warn_days}). Consider refreshing."
        else:
            action = StalenessAction.PASS
            penalty = Decimal("1.0")
            message = f"{data_type.value} data is fresh ({age_days} days old)."

        return StalenessCheckResult(
            data_type=data_type,
            phase=phase,
            data_date=data_date,
            as_of_date=as_of_date,
            age_days=age_days,
            action=action,
            penalty_multiplier=penalty,
            threshold_used=threshold,
            message=message,
        )

    def check_pipeline_staleness(
        self,
        as_of_date: date,
        financial_date: Optional[date] = None,
        trial_date: Optional[date] = None,
        market_date: Optional[date] = None,
        holdings_date: Optional[date] = None,
        phase_map: Optional[Dict[str, str]] = None,
    ) -> Dict[str, StalenessCheckResult]:
        """
        Check staleness for all pipeline data sources.

        Args:
            as_of_date: Analysis date
            financial_date: Date of financial data
            trial_date: Date of trial records
            market_date: Date of market data
            holdings_date: Date of 13F holdings
            phase_map: Dict of ticker -> phase (for phase-specific trial checks)

        Returns:
            Dict of data_type -> StalenessCheckResult
        """
        results = {}

        if financial_date is not None:
            results["financial"] = self.check_staleness(
                DataType.FINANCIAL, financial_date, as_of_date
            )

        if trial_date is not None:
            # For trial data, check with most restrictive phase or default
            trial_phase = "phase_3" if phase_map else None
            results["trial"] = self.check_staleness(
                DataType.TRIAL, trial_date, as_of_date, phase=trial_phase
            )

        if market_date is not None:
            results["market"] = self.check_staleness(
                DataType.MARKET, market_date, as_of_date
            )

        if holdings_date is not None:
            results["holdings_13f"] = self.check_staleness(
                DataType.HOLDINGS_13F, holdings_date, as_of_date
            )

        return results

    def get_pipeline_health(
        self,
        staleness_results: Dict[str, StalenessCheckResult],
    ) -> Tuple[str, List[str], List[str]]:
        """
        Summarize pipeline health from staleness results.

        Returns:
            (status, errors, warnings) where status is 'OK', 'DEGRADED', or 'FAIL'
        """
        errors = []
        warnings = []

        for name, result in staleness_results.items():
            if result.action == StalenessAction.HARD_GATE:
                errors.append(result.message)
            elif result.action == StalenessAction.SOFT_GATE:
                warnings.append(result.message)
            elif result.action == StalenessAction.WARN:
                warnings.append(result.message)

        if errors:
            return ("FAIL", errors, warnings)
        elif warnings:
            return ("DEGRADED", errors, warnings)
        else:
            return ("OK", errors, warnings)


# SEC 13F filing lag constant
SEC_13F_FILING_LAG_DAYS = 45


def compute_13f_effective_date(
    held_as_of_date: date,
    filing_date: Optional[date] = None,
) -> date:
    """
    Compute the effective availability date for 13F holdings.

    SEC 13F filings are due 45 days after quarter end.
    Holdings as of Q1 (Mar 31) are filed by May 15.

    The effective date is when the data becomes publicly available,
    NOT the holdings date.

    Args:
        held_as_of_date: Quarter-end date when holdings were held
        filing_date: Actual filing date (if known)

    Returns:
        Date when holdings data is PIT-safe to use
    """
    if filing_date:
        # If we know the actual filing date, use it
        return filing_date

    # Otherwise, assume standard 45-day lag
    return held_as_of_date + timedelta(days=SEC_13F_FILING_LAG_DAYS)


def validate_13f_pit_safety(
    held_as_of_date: date,
    as_of_date: date,
    filing_date: Optional[date] = None,
) -> Tuple[bool, str]:
    """
    Validate that 13F holdings are PIT-safe to use.

    Args:
        held_as_of_date: Quarter-end date of holdings
        as_of_date: Analysis date
        filing_date: Actual filing date (if known)

    Returns:
        (is_safe, reason)
    """
    effective_date = compute_13f_effective_date(held_as_of_date, filing_date)

    if as_of_date < effective_date:
        days_early = (effective_date - as_of_date).days
        return (
            False,
            f"13F data for {held_as_of_date} not available until {effective_date} "
            f"({days_early} days after as_of_date {as_of_date}). "
            "Using this data would create lookahead bias."
        )

    return (True, f"13F data is PIT-safe (effective {effective_date} <= as_of_date {as_of_date})")


if __name__ == "__main__":
    # Demonstration
    engine = StalenessGateEngine()

    print("=" * 70)
    print("STALENESS GATE ENGINE - DEMONSTRATION")
    print("=" * 70)

    as_of = date(2026, 1, 15)

    # Test various scenarios
    tests = [
        (DataType.FINANCIAL, date(2025, 11, 1), None, "Financial ~2.5 months old"),
        (DataType.FINANCIAL, date(2025, 8, 1), None, "Financial ~5.5 months old"),
        (DataType.TRIAL, date(2025, 10, 1), "phase_3", "Phase 3 trial ~3.5 months"),
        (DataType.TRIAL, date(2025, 10, 1), "phase_1", "Phase 1 trial ~3.5 months"),
        (DataType.MARKET, date(2026, 1, 10), None, "Market 5 days old"),
        (DataType.MARKET, date(2026, 1, 1), None, "Market 14 days old"),
    ]

    for data_type, data_date, phase, desc in tests:
        result = engine.check_staleness(data_type, data_date, as_of, phase)
        print(f"\n{desc}:")
        print(f"  Action: {result.action.value}")
        print(f"  Penalty: {result.penalty_multiplier}x")
        print(f"  Message: {result.message}")

    # Test 13F PIT safety
    print("\n" + "=" * 70)
    print("13F PIT SAFETY CHECK")
    print("=" * 70)

    q3_holdings = date(2025, 9, 30)  # Q3 holdings
    is_safe, reason = validate_13f_pit_safety(q3_holdings, as_of)
    print(f"\nQ3 2025 holdings (held {q3_holdings}) as of {as_of}:")
    print(f"  Safe: {is_safe}")
    print(f"  Reason: {reason}")
