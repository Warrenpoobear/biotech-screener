#!/usr/bin/env python3
"""
Phase Transition Momentum Engine

Measures the velocity and direction of a company's clinical development
progression through trial phases. Companies rapidly advancing through
phases have positive momentum; those stalled or with setbacks have negative.

Key signals:
- Phase Advancement: Recent transition to higher phase
- Phase Velocity: Speed of progression through phases
- Trial Activity: New trials in higher phases
- Stall Detection: Extended time in current phase
- Setback Detection: Terminated/withdrawn trials

PIT Safety:
- All calculations anchored to as_of_date
- Only uses data available at evaluation time
"""

from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class PhaseMomentum(Enum):
    """Classification of phase transition momentum."""
    STRONG_POSITIVE = "strong_positive"    # Recent phase advancement + active pipeline
    POSITIVE = "positive"                   # Active progression, new trials
    NEUTRAL = "neutral"                     # Stable, no recent changes
    NEGATIVE = "negative"                   # Stalled trials, slow progress
    STRONG_NEGATIVE = "strong_negative"    # Setbacks, terminated trials
    UNKNOWN = "unknown"


class PhaseLevel(Enum):
    """Clinical trial phases with numeric values for comparison."""
    PRECLINICAL = 0
    PHASE_1 = 1
    PHASE_1_2 = 2
    PHASE_2 = 3
    PHASE_2_3 = 4
    PHASE_3 = 5
    PHASE_4 = 6
    APPROVED = 7


@dataclass
class PhaseTransitionResult:
    """Result of phase transition momentum analysis."""
    ticker: str
    momentum: PhaseMomentum
    current_lead_phase: str
    phase_level: int
    phase_velocity_score: Decimal  # 0-100, higher = faster progression
    recent_advancements: int       # Count of phase advancements in lookback
    recent_initiations: int        # New trials started in lookback
    stalled_trials: int            # Trials without progress
    terminated_trials: int         # Recent terminated/withdrawn
    score_modifier: Decimal        # [-2.0, +2.0]
    confidence: Decimal
    flags: List[str]


class PhaseTransitionEngine:
    """
    Engine for analyzing clinical phase transition momentum.

    Scoring philosophy:
    - Strong positive (recent advancement + active): +1.5 to +2.0
    - Positive (active progression): +0.5 to +1.0
    - Neutral (stable pipeline): 0
    - Negative (stalled, slow): -0.5 to -1.0
    - Strong negative (setbacks): -1.5 to -2.0
    """

    VERSION = "1.0.0"

    # Phase mapping
    PHASE_MAP = {
        "preclinical": PhaseLevel.PRECLINICAL,
        "phase 1": PhaseLevel.PHASE_1,
        "phase1": PhaseLevel.PHASE_1,
        "phase_1": PhaseLevel.PHASE_1,
        "phase 1/2": PhaseLevel.PHASE_1_2,
        "phase1/2": PhaseLevel.PHASE_1_2,
        "phase_1_2": PhaseLevel.PHASE_1_2,
        "phase 2": PhaseLevel.PHASE_2,
        "phase2": PhaseLevel.PHASE_2,
        "phase_2": PhaseLevel.PHASE_2,
        "phase 2/3": PhaseLevel.PHASE_2_3,
        "phase2/3": PhaseLevel.PHASE_2_3,
        "phase_2_3": PhaseLevel.PHASE_2_3,
        "phase 3": PhaseLevel.PHASE_3,
        "phase3": PhaseLevel.PHASE_3,
        "phase_3": PhaseLevel.PHASE_3,
        "phase 4": PhaseLevel.PHASE_4,
        "phase4": PhaseLevel.PHASE_4,
        "phase_4": PhaseLevel.PHASE_4,
        "approved": PhaseLevel.APPROVED,
    }

    # Thresholds
    LOOKBACK_DAYS = 180  # 6 months for recent activity
    ADVANCEMENT_LOOKBACK_DAYS = 365  # 1 year for phase advancement
    STALL_THRESHOLD_DAYS = 365  # Trial is stalled if no update in 1 year

    # Score modifiers
    MODIFIER_STRONG_POSITIVE = Decimal("1.5")
    MODIFIER_POSITIVE = Decimal("0.75")
    MODIFIER_NEUTRAL = Decimal("0")
    MODIFIER_NEGATIVE = Decimal("-0.75")
    MODIFIER_STRONG_NEGATIVE = Decimal("-1.5")

    def __init__(self):
        """Initialize the phase transition engine."""
        self.audit_trail: List[Dict[str, Any]] = []

    def compute_momentum(
        self,
        ticker: str,
        trial_records: List[Dict[str, Any]],
        historical_phases: Optional[Dict[str, List[Dict]]] = None,
        as_of_date: date = None,
    ) -> PhaseTransitionResult:
        """
        Compute phase transition momentum for a ticker.

        Args:
            ticker: Stock ticker
            trial_records: List of trial records for this ticker
            historical_phases: Optional historical phase data
            as_of_date: Point-in-time date

        Returns:
            PhaseTransitionResult with momentum classification
        """
        ticker = ticker.upper()
        flags = []

        if as_of_date is None:
            as_of_date = date.today()

        # Filter to PIT-safe trials
        pit_trials = self._filter_pit_safe(trial_records, as_of_date)

        if not pit_trials:
            return PhaseTransitionResult(
                ticker=ticker,
                momentum=PhaseMomentum.UNKNOWN,
                current_lead_phase="unknown",
                phase_level=0,
                phase_velocity_score=Decimal("0"),
                recent_advancements=0,
                recent_initiations=0,
                stalled_trials=0,
                terminated_trials=0,
                score_modifier=Decimal("0"),
                confidence=Decimal("0"),
                flags=["no_trials"],
            )

        # Find current lead phase
        lead_phase, phase_level = self._find_lead_phase(pit_trials)

        # Calculate phase velocity
        velocity_score = self._calculate_velocity_score(
            pit_trials, historical_phases, as_of_date
        )

        # Count recent activity
        recent_initiations = self._count_recent_initiations(
            pit_trials, as_of_date
        )

        # Detect recent phase advancements
        recent_advancements = self._detect_advancements(
            pit_trials, historical_phases, as_of_date
        )

        # Count stalled trials
        stalled_trials = self._count_stalled_trials(pit_trials, as_of_date)

        # Count terminated/withdrawn trials
        terminated_trials = self._count_terminated_trials(pit_trials, as_of_date)

        # Classify momentum (pass total trials to avoid small-denominator overreaction)
        momentum = self._classify_momentum(
            recent_advancements,
            recent_initiations,
            stalled_trials,
            terminated_trials,
            velocity_score,
            phase_level,
            total_trials=len(pit_trials),
        )

        # Add flags
        if recent_advancements > 0:
            flags.append("recent_phase_advancement")
        if recent_initiations >= 3:
            flags.append("active_pipeline")
        if stalled_trials > 2:
            flags.append("multiple_stalled_trials")
        if terminated_trials > 0:
            flags.append("recent_terminations")
        if phase_level >= PhaseLevel.PHASE_3.value:
            flags.append("late_stage_company")
        if velocity_score >= Decimal("70"):
            flags.append("high_velocity")
        elif velocity_score <= Decimal("30"):
            flags.append("low_velocity")

        # Calculate confidence
        confidence = self._calculate_confidence(
            len(pit_trials), recent_initiations, phase_level
        )

        # Calculate score modifier
        score_modifier = self._calculate_score_modifier(
            momentum, confidence, velocity_score
        )

        result = PhaseTransitionResult(
            ticker=ticker,
            momentum=momentum,
            current_lead_phase=lead_phase,
            phase_level=phase_level,
            phase_velocity_score=velocity_score,
            recent_advancements=recent_advancements,
            recent_initiations=recent_initiations,
            stalled_trials=stalled_trials,
            terminated_trials=terminated_trials,
            score_modifier=score_modifier,
            confidence=confidence,
            flags=flags,
        )

        self._add_audit(ticker, as_of_date, result)
        return result

    def _filter_pit_safe(
        self,
        trials: List[Dict[str, Any]],
        as_of_date: date,
    ) -> List[Dict[str, Any]]:
        """Filter trials to only those available at as_of_date."""
        pit_safe = []
        for trial in trials:
            # Use last_update_posted as the availability date
            update_date = self._parse_date(trial.get("last_update_posted"))
            if update_date and update_date <= as_of_date:
                pit_safe.append(trial)
        return pit_safe

    def _find_lead_phase(
        self,
        trials: List[Dict[str, Any]],
    ) -> Tuple[str, int]:
        """Find the most advanced phase among active trials."""
        lead_phase = "preclinical"
        max_level = 0

        for trial in trials:
            status = trial.get("status", "").upper()
            # Only count active trials
            if status in ("WITHDRAWN", "TERMINATED", "SUSPENDED"):
                continue

            phase_str = self._normalize_phase(trial.get("phase", ""))
            phase_level = self.PHASE_MAP.get(phase_str, PhaseLevel.PRECLINICAL)

            if phase_level.value > max_level:
                max_level = phase_level.value
                lead_phase = phase_str

        return lead_phase, max_level

    def _normalize_phase(self, phase_str: Optional[str]) -> str:
        """Normalize phase string to standard format."""
        if not phase_str:
            return "preclinical"

        phase = phase_str.lower().strip()

        # Handle various formats
        if "approved" in phase or phase == "phase4" or phase == "phase 4":
            return "approved"
        if "3" in phase:
            if "2" in phase:
                return "phase 2/3"
            return "phase 3"
        if "2" in phase:
            if "1" in phase:
                return "phase 1/2"
            return "phase 2"
        if "1" in phase:
            return "phase 1"

        return "preclinical"

    def _calculate_velocity_score(
        self,
        trials: List[Dict[str, Any]],
        historical_phases: Optional[Dict[str, List[Dict]]],
        as_of_date: date,
    ) -> Decimal:
        """
        Calculate phase velocity score (0-100).

        Based on:
        - Diversity of phases (multiple active phases = higher velocity)
        - Recent trial initiations
        - Time between phase transitions
        """
        if not trials:
            return Decimal("50")

        score = Decimal("50")  # Base score

        # Count active trials by phase
        phase_counts = {}
        recent_starts = 0
        lookback = as_of_date - timedelta(days=self.LOOKBACK_DAYS)

        for trial in trials:
            status = trial.get("status", "").upper()
            if status in ("WITHDRAWN", "TERMINATED", "SUSPENDED", "COMPLETED"):
                continue

            phase = self._normalize_phase(trial.get("phase", ""))
            phase_counts[phase] = phase_counts.get(phase, 0) + 1

            # Check for recent starts
            start_date = self._parse_date(trial.get("start_date"))
            if start_date and start_date >= lookback:
                recent_starts += 1

        # Bonus for phase diversity (multi-phase pipeline)
        num_phases = len(phase_counts)
        if num_phases >= 3:
            score += Decimal("15")
        elif num_phases >= 2:
            score += Decimal("8")

        # Bonus for late-stage presence
        if "phase 3" in phase_counts or "phase 2/3" in phase_counts:
            score += Decimal("10")
        if "approved" in phase_counts:
            score += Decimal("5")

        # Bonus for recent activity
        if recent_starts >= 5:
            score += Decimal("15")
        elif recent_starts >= 3:
            score += Decimal("10")
        elif recent_starts >= 1:
            score += Decimal("5")

        # Cap at 0-100
        score = max(Decimal("0"), min(Decimal("100"), score))

        return score.quantize(Decimal("0.01"))

    def _count_recent_initiations(
        self,
        trials: List[Dict[str, Any]],
        as_of_date: date,
    ) -> int:
        """Count trials initiated in the lookback period."""
        lookback = as_of_date - timedelta(days=self.LOOKBACK_DAYS)
        count = 0

        for trial in trials:
            start_date = self._parse_date(trial.get("start_date"))
            if not start_date:
                # Fall back to last_update_posted for new trials
                update_date = self._parse_date(trial.get("last_update_posted"))
                if update_date and update_date >= lookback:
                    status = trial.get("status", "").upper()
                    if status in ("RECRUITING", "NOT_YET_RECRUITING", "ACTIVE_NOT_RECRUITING"):
                        count += 1
            elif start_date >= lookback:
                count += 1

        return count

    def _detect_advancements(
        self,
        trials: List[Dict[str, Any]],
        historical_phases: Optional[Dict[str, List[Dict]]],
        as_of_date: date,
    ) -> int:
        """
        Detect phase advancements in the lookback period.

        Heuristic: Look for trials in higher phases that started recently.
        """
        lookback = as_of_date - timedelta(days=self.ADVANCEMENT_LOOKBACK_DAYS)
        advancements = 0

        # Group trials by phase level
        phase_trials = {}
        for trial in trials:
            phase = self._normalize_phase(trial.get("phase", ""))
            level = self.PHASE_MAP.get(phase, PhaseLevel.PRECLINICAL).value
            if level not in phase_trials:
                phase_trials[level] = []
            phase_trials[level].append(trial)

        # Look for new trials in higher phases
        for level in sorted(phase_trials.keys(), reverse=True):
            if level < 3:  # Phase 2 or higher
                continue

            for trial in phase_trials[level]:
                start_date = self._parse_date(trial.get("start_date"))
                update_date = self._parse_date(trial.get("last_update_posted"))

                # Check if this is a recent high-phase trial
                check_date = start_date or update_date
                if check_date and check_date >= lookback:
                    status = trial.get("status", "").upper()
                    if status not in ("WITHDRAWN", "TERMINATED"):
                        advancements += 1

        return min(advancements, 5)  # Cap at 5

    def _count_stalled_trials(
        self,
        trials: List[Dict[str, Any]],
        as_of_date: date,
    ) -> int:
        """Count trials that appear stalled (no updates in threshold period)."""
        stall_cutoff = as_of_date - timedelta(days=self.STALL_THRESHOLD_DAYS)
        stalled = 0

        for trial in trials:
            status = trial.get("status", "").upper()

            # Only count active trials as potentially stalled
            if status not in ("RECRUITING", "ACTIVE_NOT_RECRUITING", "ENROLLING_BY_INVITATION"):
                continue

            update_date = self._parse_date(trial.get("last_update_posted"))
            if update_date and update_date < stall_cutoff:
                stalled += 1

        return stalled

    def _count_terminated_trials(
        self,
        trials: List[Dict[str, Any]],
        as_of_date: date,
    ) -> int:
        """Count recently terminated or withdrawn trials."""
        lookback = as_of_date - timedelta(days=self.LOOKBACK_DAYS)
        terminated = 0

        for trial in trials:
            status = trial.get("status", "").upper()
            if status not in ("TERMINATED", "WITHDRAWN"):
                continue

            update_date = self._parse_date(trial.get("last_update_posted"))
            if update_date and update_date >= lookback:
                terminated += 1

        return terminated

    def _classify_momentum(
        self,
        advancements: int,
        initiations: int,
        stalled: int,
        terminated: int,
        velocity: Decimal,
        phase_level: int,
        total_trials: int = 0,
    ) -> PhaseMomentum:
        """Classify overall momentum based on signals.

        Governance notes:
        - Require 2 independent negatives for strong_negative
        - Minimum trial count (3) before applying harsh negatives
        - Avoid small-denominator overreaction
        """
        # Strong positive: recent advancement + high activity
        if advancements >= 1 and initiations >= 2 and terminated == 0:
            return PhaseMomentum.STRONG_POSITIVE

        # Strong negative: require BOTH terminations AND stalls (2 independent negatives)
        # Also require minimum trial count to avoid overreaction
        if total_trials >= 3 and terminated >= 2 and stalled >= 2:
            return PhaseMomentum.STRONG_NEGATIVE

        # Positive: active pipeline with good velocity
        if initiations >= 2 and velocity >= Decimal("60") and terminated == 0:
            return PhaseMomentum.POSITIVE

        # Negative: multiple stalls with low activity (require minimum trials)
        if total_trials >= 3 and stalled >= 3 and initiations == 0 and velocity < Decimal("40"):
            return PhaseMomentum.NEGATIVE

        # Negative: terminations without offsetting activity (require minimum trials)
        if total_trials >= 3 and terminated >= 2 and initiations < 2:
            return PhaseMomentum.NEGATIVE

        # Neutral: stable pipeline (default)
        return PhaseMomentum.NEUTRAL

    def _calculate_confidence(
        self,
        trial_count: int,
        recent_activity: int,
        phase_level: int,
    ) -> Decimal:
        """Calculate confidence in momentum assessment."""
        if trial_count == 0:
            return Decimal("0")

        # More trials = higher confidence
        if trial_count >= 10:
            base = Decimal("0.8")
        elif trial_count >= 5:
            base = Decimal("0.7")
        elif trial_count >= 2:
            base = Decimal("0.6")
        else:
            base = Decimal("0.5")

        # Recent activity increases confidence
        if recent_activity >= 3:
            base += Decimal("0.1")

        # Late stage increases confidence (more data)
        if phase_level >= PhaseLevel.PHASE_3.value:
            base += Decimal("0.1")

        return min(Decimal("1.0"), base).quantize(Decimal("0.01"))

    def _calculate_score_modifier(
        self,
        momentum: PhaseMomentum,
        confidence: Decimal,
        velocity: Decimal,
    ) -> Decimal:
        """Calculate score modifier based on momentum classification."""
        if momentum == PhaseMomentum.UNKNOWN:
            return Decimal("0")

        # Base modifier by momentum
        modifier_map = {
            PhaseMomentum.STRONG_POSITIVE: self.MODIFIER_STRONG_POSITIVE,
            PhaseMomentum.POSITIVE: self.MODIFIER_POSITIVE,
            PhaseMomentum.NEUTRAL: self.MODIFIER_NEUTRAL,
            PhaseMomentum.NEGATIVE: self.MODIFIER_NEGATIVE,
            PhaseMomentum.STRONG_NEGATIVE: self.MODIFIER_STRONG_NEGATIVE,
        }

        base = modifier_map.get(momentum, Decimal("0"))

        # Adjust by velocity for positive/negative
        if momentum in (PhaseMomentum.STRONG_POSITIVE, PhaseMomentum.POSITIVE):
            # High velocity boosts positive
            if velocity >= Decimal("80"):
                base += Decimal("0.25")
        elif momentum in (PhaseMomentum.STRONG_NEGATIVE, PhaseMomentum.NEGATIVE):
            # Low velocity amplifies negative
            if velocity <= Decimal("30"):
                base -= Decimal("0.25")

        # Scale by confidence
        modifier = base * confidence

        # Clamp to bounds
        modifier = max(Decimal("-2.0"), min(Decimal("2.0"), modifier))

        return modifier.quantize(Decimal("0.01"))

    def _parse_date(self, date_str: Any) -> Optional[date]:
        """Parse date string to date object."""
        if isinstance(date_str, date):
            return date_str
        if not date_str:
            return None
        try:
            # Handle various formats
            date_str = str(date_str)[:10]
            return date.fromisoformat(date_str)
        except ValueError:
            return None

    def score_universe(
        self,
        universe: List[Dict[str, Any]],
        trials_by_ticker: Dict[str, List[Dict]],
        historical_phases: Optional[Dict[str, List[Dict]]] = None,
        as_of_date: date = None,
    ) -> Dict[str, Any]:
        """
        Score phase momentum for entire universe.

        Args:
            universe: List of {ticker: str}
            trials_by_ticker: Trial records keyed by ticker
            historical_phases: Optional historical phase data
            as_of_date: Point-in-time date

        Returns:
            Dictionary with scores and diagnostics
        """
        if as_of_date is None:
            as_of_date = date.today()

        scores_by_ticker = {}
        momentum_distribution = {m.value: 0 for m in PhaseMomentum}

        for record in universe:
            ticker = record.get("ticker", "").upper()
            if not ticker:
                continue

            trials = trials_by_ticker.get(ticker, [])
            hist = historical_phases.get(ticker) if historical_phases else None

            result = self.compute_momentum(
                ticker, trials, hist, as_of_date
            )

            scores_by_ticker[ticker] = {
                "ticker": ticker,
                "momentum": result.momentum.value,
                "current_lead_phase": result.current_lead_phase,
                "phase_level": result.phase_level,
                "velocity_score": str(result.phase_velocity_score),
                "recent_advancements": result.recent_advancements,
                "recent_initiations": result.recent_initiations,
                "stalled_trials": result.stalled_trials,
                "terminated_trials": result.terminated_trials,
                "score_modifier": str(result.score_modifier),
                "confidence": str(result.confidence),
                "flags": result.flags,
            }

            momentum_distribution[result.momentum.value] += 1

        return {
            "scores_by_ticker": scores_by_ticker,
            "diagnostic_counts": {
                "total_scored": len(scores_by_ticker),
                "momentum_distribution": momentum_distribution,
            },
            "provenance": {
                "engine": "PhaseTransitionEngine",
                "version": self.VERSION,
                "as_of_date": as_of_date.isoformat(),
                "lookback_days": self.LOOKBACK_DAYS,
            },
        }

    def _add_audit(
        self,
        ticker: str,
        as_of_date: date,
        result: PhaseTransitionResult,
    ) -> None:
        """Add entry to audit trail."""
        self.audit_trail.append({
            "ticker": ticker,
            "as_of_date": as_of_date.isoformat(),
            "momentum": result.momentum.value,
            "score_modifier": str(result.score_modifier),
            "lead_phase": result.current_lead_phase,
        })

    def get_audit_trail(self) -> List[Dict[str, Any]]:
        """Return audit trail."""
        return self.audit_trail.copy()

    def clear_audit_trail(self) -> None:
        """Clear audit trail."""
        self.audit_trail = []
