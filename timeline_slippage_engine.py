#!/usr/bin/env python3
"""
timeline_slippage_engine.py

Timeline Slippage Engine for Biotech Screener

Detects execution risk from trial timeline changes:
- Pushouts (delays) indicate management/execution issues
- Pull-ins (accelerations) indicate positive momentum
- Repeated slippage patterns are penalized more heavily

Design Philosophy:
- Deterministic: No datetime.now(), timestamps from as_of_date
- Fail-closed: Missing data = no adjustment (neutral score)
- Auditable: Every calculation tracked with confidence
- Stdlib-only: No external dependencies

Author: Wake Robin Capital Management
Version: 1.0.0
"""

from decimal import Decimal, ROUND_HALF_UP
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

__version__ = "1.0.0"


class SlippageDirection(str, Enum):
    """Direction of timeline slippage."""
    PUSHOUT = "pushout"      # Delayed vs prior estimate
    PULLIN = "pullin"        # Accelerated vs prior estimate
    STABLE = "stable"        # No significant change
    UNKNOWN = "unknown"      # Insufficient data


class SlippageSeverity(str, Enum):
    """Severity of slippage impact."""
    SEVERE = "severe"        # >180 days pushout
    MODERATE = "moderate"    # 60-180 days pushout
    MINOR = "minor"          # 30-60 days pushout
    NONE = "none"            # <30 days or stable
    ACCELERATED = "accelerated"  # Pull-in (positive)


@dataclass
class TrialTimelineSnapshot:
    """Point-in-time trial timeline data."""
    nct_id: str
    snapshot_date: date
    primary_completion_date: Optional[date]
    completion_date: Optional[date]
    status: str
    phase: str


@dataclass
class SlippageResult:
    """Result of slippage analysis for a single trial."""
    nct_id: str
    ticker: str
    direction: SlippageDirection
    severity: SlippageSeverity
    days_slipped: int
    slippage_score: Decimal  # 0-100, 50=neutral, <50=bad, >50=good
    confidence: str  # "high", "medium", "low"
    prior_date: Optional[str]
    current_date: Optional[str]
    flags: List[str] = field(default_factory=list)


@dataclass
class TickerSlippageScore:
    """Aggregated slippage score for a ticker."""
    ticker: str
    execution_risk_score: Decimal  # 0-100, higher=better execution
    avg_slippage_days: Decimal
    pushout_count: int
    pullin_count: int
    stable_count: int
    repeat_offender: bool  # Multiple significant pushouts
    confidence: str
    trial_results: List[SlippageResult] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)


class TimelineSlippageEngine:
    """
    Engine for detecting and scoring trial timeline execution risk.

    Compares current trial completion dates against prior snapshots to detect
    slippage patterns that indicate management/execution quality.

    Usage:
        engine = TimelineSlippageEngine()
        result = engine.calculate_slippage_score(
            ticker="ACME",
            current_trials=[...],
            prior_trials=[...],
            as_of_date=date(2026, 1, 15)
        )
    """

    VERSION = "1.0.0"

    # Slippage thresholds (days)
    SEVERE_THRESHOLD = 180
    MODERATE_THRESHOLD = 60
    MINOR_THRESHOLD = 30

    # Score adjustments
    SEVERE_PENALTY = Decimal("-20")
    MODERATE_PENALTY = Decimal("-10")
    MINOR_PENALTY = Decimal("-5")
    PULLIN_BONUS = Decimal("5")
    REPEAT_OFFENDER_PENALTY = Decimal("-15")

    # Confidence thresholds
    MIN_TRIALS_HIGH_CONFIDENCE = 3
    MIN_TRIALS_MEDIUM_CONFIDENCE = 2

    def __init__(self):
        """Initialize the engine."""
        self.audit_trail: List[Dict[str, Any]] = []

    def _parse_date(self, date_val: Union[str, date, None]) -> Optional[date]:
        """Parse date from string or date object."""
        if date_val is None:
            return None
        if isinstance(date_val, date):
            return date_val
        if isinstance(date_val, str):
            try:
                return date.fromisoformat(date_val[:10])
            except (ValueError, TypeError):
                return None
        return None

    def _calculate_days_slipped(
        self,
        prior_date: Optional[date],
        current_date: Optional[date],
    ) -> Tuple[int, SlippageDirection]:
        """
        Calculate days of slippage between dates.

        Returns:
            (days_slipped, direction)
            Positive days = pushout (bad)
            Negative days = pullin (good)
        """
        if prior_date is None or current_date is None:
            return (0, SlippageDirection.UNKNOWN)

        days_diff = (current_date - prior_date).days

        if days_diff > self.MINOR_THRESHOLD:
            return (days_diff, SlippageDirection.PUSHOUT)
        elif days_diff < -self.MINOR_THRESHOLD:
            return (abs(days_diff), SlippageDirection.PULLIN)
        else:
            return (abs(days_diff), SlippageDirection.STABLE)

    def _determine_severity(
        self,
        days_slipped: int,
        direction: SlippageDirection,
    ) -> SlippageSeverity:
        """Determine severity based on slippage magnitude."""
        if direction == SlippageDirection.PULLIN:
            return SlippageSeverity.ACCELERATED

        if direction == SlippageDirection.STABLE:
            return SlippageSeverity.NONE

        if direction == SlippageDirection.UNKNOWN:
            return SlippageSeverity.NONE

        # Pushout severities
        if days_slipped >= self.SEVERE_THRESHOLD:
            return SlippageSeverity.SEVERE
        elif days_slipped >= self.MODERATE_THRESHOLD:
            return SlippageSeverity.MODERATE
        elif days_slipped >= self.MINOR_THRESHOLD:
            return SlippageSeverity.MINOR
        else:
            return SlippageSeverity.NONE

    def _score_single_trial(
        self,
        current_trial: Dict[str, Any],
        prior_trial: Optional[Dict[str, Any]],
        ticker: str,
    ) -> SlippageResult:
        """Score slippage for a single trial."""
        nct_id = current_trial.get("nct_id", "UNKNOWN")
        flags = []

        # Get current completion date (prefer primary_completion_date)
        current_date = self._parse_date(
            current_trial.get("primary_completion_date")
            or current_trial.get("completion_date")
        )

        # Get prior completion date
        prior_date = None
        if prior_trial:
            prior_date = self._parse_date(
                prior_trial.get("primary_completion_date")
                or prior_trial.get("completion_date")
            )

        # Calculate slippage
        days_slipped, direction = self._calculate_days_slipped(prior_date, current_date)
        severity = self._determine_severity(days_slipped, direction)

        # Determine confidence
        if prior_date and current_date:
            confidence = "high"
        elif current_date:
            confidence = "medium"
            flags.append("MISSING_PRIOR_DATE")
        else:
            confidence = "low"
            flags.append("MISSING_DATES")

        # Calculate score (50 = neutral)
        score = Decimal("50")

        if direction == SlippageDirection.PUSHOUT:
            if severity == SlippageSeverity.SEVERE:
                score += self.SEVERE_PENALTY
                flags.append("SEVERE_PUSHOUT")
            elif severity == SlippageSeverity.MODERATE:
                score += self.MODERATE_PENALTY
                flags.append("MODERATE_PUSHOUT")
            elif severity == SlippageSeverity.MINOR:
                score += self.MINOR_PENALTY
                flags.append("MINOR_PUSHOUT")

        elif direction == SlippageDirection.PULLIN:
            score += self.PULLIN_BONUS
            flags.append("TIMELINE_ACCELERATED")

        # Clamp to 0-100
        score = max(Decimal("0"), min(Decimal("100"), score))

        return SlippageResult(
            nct_id=nct_id,
            ticker=ticker,
            direction=direction,
            severity=severity,
            days_slipped=days_slipped,
            slippage_score=score.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
            confidence=confidence,
            prior_date=prior_date.isoformat() if prior_date else None,
            current_date=current_date.isoformat() if current_date else None,
            flags=flags,
        )

    def calculate_slippage_score(
        self,
        ticker: str,
        current_trials: List[Dict[str, Any]],
        prior_trials: Optional[List[Dict[str, Any]]] = None,
        as_of_date: Union[str, date] = None,
    ) -> TickerSlippageScore:
        """
        Calculate aggregated slippage score for a ticker.

        Args:
            ticker: Stock ticker
            current_trials: Current trial records
            prior_trials: Prior snapshot of trial records (optional)
            as_of_date: Analysis date

        Returns:
            TickerSlippageScore with execution risk assessment
        """
        if as_of_date is None:
            raise ValueError("as_of_date is required for PIT discipline")

        if isinstance(as_of_date, str):
            as_of_date = date.fromisoformat(as_of_date)

        # Build prior trial lookup
        prior_by_nct = {}
        if prior_trials:
            for trial in prior_trials:
                nct_id = trial.get("nct_id")
                if nct_id:
                    prior_by_nct[nct_id] = trial

        # Score each trial
        trial_results = []
        for trial in current_trials:
            nct_id = trial.get("nct_id")
            prior_trial = prior_by_nct.get(nct_id)
            result = self._score_single_trial(trial, prior_trial, ticker)
            trial_results.append(result)

        # Aggregate results
        if not trial_results:
            return TickerSlippageScore(
                ticker=ticker,
                execution_risk_score=Decimal("50"),  # Neutral
                avg_slippage_days=Decimal("0"),
                pushout_count=0,
                pullin_count=0,
                stable_count=0,
                repeat_offender=False,
                confidence="low",
                trial_results=[],
                flags=["NO_TRIALS"],
            )

        # Count directions
        pushout_count = sum(1 for r in trial_results if r.direction == SlippageDirection.PUSHOUT)
        pullin_count = sum(1 for r in trial_results if r.direction == SlippageDirection.PULLIN)
        stable_count = sum(1 for r in trial_results if r.direction == SlippageDirection.STABLE)

        # Calculate average slippage
        total_slippage = sum(
            r.days_slipped for r in trial_results
            if r.direction == SlippageDirection.PUSHOUT
        )
        avg_slippage = Decimal(str(total_slippage / len(trial_results) if trial_results else 0))

        # Check for repeat offender pattern
        severe_moderate_count = sum(
            1 for r in trial_results
            if r.severity in (SlippageSeverity.SEVERE, SlippageSeverity.MODERATE)
        )
        repeat_offender = severe_moderate_count >= 2

        # Calculate aggregate score
        scores = [r.slippage_score for r in trial_results]
        avg_score = sum(scores) / len(scores) if scores else Decimal("50")

        # Apply repeat offender penalty
        flags = []
        if repeat_offender:
            avg_score += self.REPEAT_OFFENDER_PENALTY
            flags.append("REPEAT_SLIPPAGE_OFFENDER")

        # Clamp to 0-100
        execution_risk_score = max(Decimal("0"), min(Decimal("100"), avg_score))

        # Determine confidence
        if len(trial_results) >= self.MIN_TRIALS_HIGH_CONFIDENCE:
            confidence = "high"
        elif len(trial_results) >= self.MIN_TRIALS_MEDIUM_CONFIDENCE:
            confidence = "medium"
        else:
            confidence = "low"

        result = TickerSlippageScore(
            ticker=ticker,
            execution_risk_score=execution_risk_score.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
            avg_slippage_days=avg_slippage.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP),
            pushout_count=pushout_count,
            pullin_count=pullin_count,
            stable_count=stable_count,
            repeat_offender=repeat_offender,
            confidence=confidence,
            trial_results=trial_results,
            flags=flags,
        )

        # Add to audit trail
        self.audit_trail.append({
            "ticker": ticker,
            "as_of_date": as_of_date.isoformat(),
            "execution_risk_score": str(result.execution_risk_score),
            "pushout_count": pushout_count,
            "pullin_count": pullin_count,
            "repeat_offender": repeat_offender,
            "confidence": confidence,
            "trial_count": len(trial_results),
        })

        return result

    def score_universe(
        self,
        universe: List[Dict[str, Any]],
        current_trials_by_ticker: Dict[str, List[Dict[str, Any]]],
        prior_trials_by_ticker: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        as_of_date: Union[str, date] = None,
    ) -> Dict[str, Any]:
        """
        Score slippage risk for entire universe.

        Args:
            universe: List of company dicts with 'ticker' field
            current_trials_by_ticker: {ticker: [trial_records]}
            prior_trials_by_ticker: {ticker: [prior_trial_records]}
            as_of_date: Analysis date

        Returns:
            Dict with scores, diagnostics, and provenance
        """
        if as_of_date is None:
            raise ValueError("as_of_date is required")

        if isinstance(as_of_date, str):
            as_of_date = date.fromisoformat(as_of_date)

        prior_trials_by_ticker = prior_trials_by_ticker or {}

        scores = []
        for company in universe:
            ticker = company.get("ticker")
            if not ticker:
                continue

            ticker_upper = ticker.upper()
            current_trials = current_trials_by_ticker.get(ticker_upper, [])
            prior_trials = prior_trials_by_ticker.get(ticker_upper, [])

            result = self.calculate_slippage_score(
                ticker=ticker,
                current_trials=current_trials,
                prior_trials=prior_trials,
                as_of_date=as_of_date,
            )

            scores.append({
                "ticker": ticker,
                "execution_risk_score": str(result.execution_risk_score),
                "avg_slippage_days": str(result.avg_slippage_days),
                "pushout_count": result.pushout_count,
                "pullin_count": result.pullin_count,
                "stable_count": result.stable_count,
                "repeat_offender": result.repeat_offender,
                "confidence": result.confidence,
                "flags": result.flags,
            })

        # Diagnostics
        total_scored = len(scores)
        repeat_offenders = sum(1 for s in scores if s["repeat_offender"])
        high_conf = sum(1 for s in scores if s["confidence"] == "high")

        return {
            "as_of_date": as_of_date.isoformat(),
            "scores": scores,
            "diagnostic_counts": {
                "total_scored": total_scored,
                "repeat_offenders": repeat_offenders,
                "high_confidence_pct": round(high_conf / total_scored * 100, 1) if total_scored else 0,
            },
            "provenance": {
                "engine": "TimelineSlippageEngine",
                "version": self.VERSION,
            },
        }

    def get_diagnostic_counts(self) -> Dict[str, Any]:
        """Get diagnostic summary."""
        if not self.audit_trail:
            return {"total": 0}

        total = len(self.audit_trail)
        pushouts = sum(1 for a in self.audit_trail if a.get("pushout_count", 0) > 0)
        pullins = sum(1 for a in self.audit_trail if a.get("pullin_count", 0) > 0)
        repeat_offenders = sum(1 for a in self.audit_trail if a.get("repeat_offender", False))

        return {
            "total": total,
            "with_pushouts": pushouts,
            "with_pullins": pullins,
            "repeat_offenders": repeat_offenders,
            "pushout_pct": round(pushouts / total * 100, 1) if total else 0,
        }


# =============================================================================
# SELF-CHECKS
# =============================================================================

def _run_self_checks() -> List[str]:
    """Run self-checks to verify engine correctness."""
    errors = []

    engine = TimelineSlippageEngine()

    # CHECK 1: Severe pushout detection
    current = [{"nct_id": "NCT001", "primary_completion_date": "2027-01-01"}]
    prior = [{"nct_id": "NCT001", "primary_completion_date": "2026-01-01"}]

    result = engine.calculate_slippage_score(
        ticker="TEST",
        current_trials=current,
        prior_trials=prior,
        as_of_date=date(2026, 1, 15),
    )

    if result.trial_results[0].direction != SlippageDirection.PUSHOUT:
        errors.append(f"CHECK1 FAIL: Expected PUSHOUT, got {result.trial_results[0].direction}")

    if result.trial_results[0].severity != SlippageSeverity.SEVERE:
        errors.append(f"CHECK1 FAIL: Expected SEVERE, got {result.trial_results[0].severity}")

    # CHECK 2: Pullin detection
    engine2 = TimelineSlippageEngine()
    current2 = [{"nct_id": "NCT002", "primary_completion_date": "2026-03-01"}]
    prior2 = [{"nct_id": "NCT002", "primary_completion_date": "2026-09-01"}]

    result2 = engine2.calculate_slippage_score(
        ticker="TEST2",
        current_trials=current2,
        prior_trials=prior2,
        as_of_date=date(2026, 1, 15),
    )

    if result2.trial_results[0].direction != SlippageDirection.PULLIN:
        errors.append(f"CHECK2 FAIL: Expected PULLIN, got {result2.trial_results[0].direction}")

    # CHECK 3: Repeat offender detection
    engine3 = TimelineSlippageEngine()
    current3 = [
        {"nct_id": "NCT001", "primary_completion_date": "2027-01-01"},
        {"nct_id": "NCT002", "primary_completion_date": "2027-06-01"},
    ]
    prior3 = [
        {"nct_id": "NCT001", "primary_completion_date": "2026-01-01"},
        {"nct_id": "NCT002", "primary_completion_date": "2026-06-01"},
    ]

    result3 = engine3.calculate_slippage_score(
        ticker="TEST3",
        current_trials=current3,
        prior_trials=prior3,
        as_of_date=date(2026, 1, 15),
    )

    if not result3.repeat_offender:
        errors.append("CHECK3 FAIL: Should be flagged as repeat offender")

    # CHECK 4: Score bounds
    if result.execution_risk_score < Decimal("0") or result.execution_risk_score > Decimal("100"):
        errors.append(f"CHECK4 FAIL: Score out of bounds: {result.execution_risk_score}")

    return errors


if __name__ == "__main__":
    errors = _run_self_checks()
    if errors:
        print("SELF-CHECK FAILURES:")
        for e in errors:
            print(f"  {e}")
        exit(1)
    else:
        print("All self-checks passed!")
        exit(0)
