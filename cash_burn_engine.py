#!/usr/bin/env python3
"""
Cash Burn Trajectory Engine

Measures burn rate direction (accelerating vs decelerating) as a signal
for financial discipline vs uncontrolled spending.

Key design decisions:
- Uses operating cash flow (most consistent across companies)
- Context-aware: doesn't penalize Phase 3 ramp
- Runway interaction: short runway + accelerating = clearly bad
- Bounded modifier to prevent leaderboard rewrites

PIT Safety:
- All calculations anchored to filing dates
- Quarterly data must have period_end <= as_of_date
"""

from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class BurnTrajectory(Enum):
    """Classification of burn rate direction."""
    DECELERATING = "decelerating"      # Burn rate decreasing (discipline)
    STABLE = "stable"                   # Burn rate roughly flat
    ACCELERATING = "accelerating"       # Burn rate increasing
    ACCELERATING_JUSTIFIED = "accelerating_justified"  # Phase 3 ramp
    UNKNOWN = "unknown"


class BurnRiskLevel(Enum):
    """Overall burn risk assessment."""
    LOW = "low"           # Decelerating or stable with good runway
    MODERATE = "moderate" # Accelerating but justified, or stable with short runway
    HIGH = "high"         # Accelerating + short runway
    CRITICAL = "critical" # Accelerating + very short runway (<6mo)
    UNKNOWN = "unknown"


@dataclass
class BurnTrajectoryResult:
    """Result of burn trajectory analysis."""
    ticker: str
    trajectory: BurnTrajectory
    risk_level: BurnRiskLevel
    burn_change_pct: Optional[Decimal]  # % change in burn rate
    current_quarterly_burn: Optional[Decimal]  # $MM
    prior_quarterly_burn: Optional[Decimal]  # $MM
    runway_months: Optional[Decimal]
    has_late_stage_catalyst: bool
    score_modifier: Decimal  # [-2.0, +2.0]
    confidence: Decimal
    flags: List[str]


class CashBurnEngine:
    """
    Engine for analyzing cash burn trajectory.

    Scoring philosophy:
    - Decelerating burn with good runway → positive signal (+1.0 to +2.0)
    - Stable burn → neutral (0)
    - Accelerating burn with Phase 3 justification → small penalty (-0.5)
    - Accelerating burn without justification → moderate penalty (-1.0)
    - Accelerating burn + short runway → large penalty (-1.5 to -2.0)
    """

    VERSION = "1.0.0"

    # Thresholds for trajectory classification
    DECEL_THRESHOLD = Decimal("-0.15")   # >15% reduction = decelerating
    ACCEL_THRESHOLD = Decimal("0.15")    # >15% increase = accelerating
    # Between -15% and +15% = stable

    # Runway thresholds (months)
    RUNWAY_SHORT = Decimal("12")
    RUNWAY_VERY_SHORT = Decimal("6")
    RUNWAY_COMFORTABLE = Decimal("24")

    # Score modifiers (bounded)
    MODIFIER_DECEL_GOOD_RUNWAY = Decimal("1.5")
    MODIFIER_DECEL_SHORT_RUNWAY = Decimal("0.5")
    MODIFIER_STABLE = Decimal("0")
    MODIFIER_ACCEL_JUSTIFIED = Decimal("-0.5")
    MODIFIER_ACCEL_UNJUSTIFIED = Decimal("-1.0")
    MODIFIER_ACCEL_SHORT_RUNWAY = Decimal("-1.5")
    MODIFIER_ACCEL_CRITICAL = Decimal("-2.0")

    def __init__(self):
        """Initialize the cash burn engine."""
        self.audit_trail: List[Dict[str, Any]] = []

    def compute_trajectory(
        self,
        ticker: str,
        financial_data: Dict[str, Any],
        clinical_data: Optional[Dict[str, Any]],
        as_of_date: date,
    ) -> BurnTrajectoryResult:
        """
        Compute cash burn trajectory for a ticker.

        Args:
            ticker: Stock ticker
            financial_data: Dict with quarterly financials
                - quarterly_burn: List of {period_end, operating_cash_flow} or
                - burn_rate_current, burn_rate_prior (simplified)
                - runway_months
                - cash_position
            clinical_data: Dict with trial info for Phase 3 detection
            as_of_date: Point-in-time date

        Returns:
            BurnTrajectoryResult with trajectory and score modifier
        """
        ticker = ticker.upper()
        flags = []

        # Extract burn rates
        burn_current, burn_prior, confidence = self._extract_burn_rates(
            financial_data, as_of_date
        )

        # Extract runway
        runway_months = self._to_decimal(financial_data.get("runway_months"))

        # Check for late-stage catalyst (Phase 3 justification)
        has_late_stage = self._has_late_stage_catalyst(clinical_data)

        # Compute trajectory
        if burn_current is None or burn_prior is None:
            trajectory = BurnTrajectory.UNKNOWN
            burn_change_pct = None
            flags.append("insufficient_burn_data")
        else:
            burn_change_pct = self._compute_change_pct(burn_current, burn_prior)
            trajectory = self._classify_trajectory(
                burn_change_pct, has_late_stage
            )

            if trajectory == BurnTrajectory.DECELERATING:
                flags.append("burn_decelerating")
            elif trajectory == BurnTrajectory.ACCELERATING:
                flags.append("burn_accelerating")
            elif trajectory == BurnTrajectory.ACCELERATING_JUSTIFIED:
                flags.append("burn_accelerating_phase3_justified")

        # Compute risk level
        risk_level = self._compute_risk_level(
            trajectory, runway_months, burn_change_pct
        )

        if risk_level == BurnRiskLevel.CRITICAL:
            flags.append("burn_risk_critical")
        elif risk_level == BurnRiskLevel.HIGH:
            flags.append("burn_risk_high")

        # Compute score modifier
        score_modifier = self._compute_score_modifier(
            trajectory, risk_level, runway_months, confidence
        )

        result = BurnTrajectoryResult(
            ticker=ticker,
            trajectory=trajectory,
            risk_level=risk_level,
            burn_change_pct=burn_change_pct,
            current_quarterly_burn=burn_current,
            prior_quarterly_burn=burn_prior,
            runway_months=runway_months,
            has_late_stage_catalyst=has_late_stage,
            score_modifier=score_modifier,
            confidence=confidence,
            flags=flags,
        )

        self._add_audit(ticker, as_of_date, result)
        return result

    def _extract_burn_rates(
        self,
        financial_data: Dict[str, Any],
        as_of_date: date,
    ) -> Tuple[Optional[Decimal], Optional[Decimal], Decimal]:
        """
        Extract current and prior burn rates from financial data.

        Returns:
            (current_burn, prior_burn, confidence)
        """
        confidence = Decimal("0.5")

        # Method 1: Pre-computed burn rates (simplified)
        if "burn_rate_current" in financial_data:
            current = self._to_decimal(financial_data.get("burn_rate_current"))
            prior = self._to_decimal(financial_data.get("burn_rate_prior"))
            if current is not None and prior is not None:
                confidence = Decimal("0.7")
            return current, prior, confidence

        # Method 2: Quarterly operating cash flow series
        quarterly_data = financial_data.get("quarterly_burn", [])
        if not quarterly_data:
            # Fallback: estimate from runway and cash
            cash = self._to_decimal(financial_data.get("cash_position"))
            runway = self._to_decimal(financial_data.get("runway_months"))
            if cash is not None and runway is not None and runway > 0:
                estimated_burn = cash / runway * Decimal("3")  # Quarterly
                return estimated_burn, None, Decimal("0.3")
            return None, None, Decimal("0")

        # Filter to PIT-safe data (period_end <= as_of_date)
        valid_quarters = [
            q for q in quarterly_data
            if self._parse_date(q.get("period_end")) is not None
            and self._parse_date(q.get("period_end")) <= as_of_date
        ]

        if len(valid_quarters) < 2:
            return None, None, Decimal("0.3")

        # Sort by period_end descending
        valid_quarters.sort(
            key=lambda q: self._parse_date(q.get("period_end")),
            reverse=True
        )

        # Get most recent two quarters
        current_q = valid_quarters[0]
        prior_q = valid_quarters[1]

        current_burn = self._to_decimal(current_q.get("operating_cash_flow"))
        prior_burn = self._to_decimal(prior_q.get("operating_cash_flow"))

        # Operating cash flow is typically negative for biotechs
        # Convert to positive burn rate
        if current_burn is not None:
            current_burn = abs(current_burn)
        if prior_burn is not None:
            prior_burn = abs(prior_burn)

        confidence = Decimal("0.8") if len(valid_quarters) >= 4 else Decimal("0.6")

        return current_burn, prior_burn, confidence

    def _compute_change_pct(
        self,
        current: Decimal,
        prior: Decimal,
    ) -> Optional[Decimal]:
        """Compute percentage change in burn rate."""
        if prior == 0:
            return None
        return ((current - prior) / prior).quantize(Decimal("0.0001"))

    def _classify_trajectory(
        self,
        change_pct: Optional[Decimal],
        has_late_stage: bool,
    ) -> BurnTrajectory:
        """Classify burn trajectory based on change percentage."""
        if change_pct is None:
            return BurnTrajectory.UNKNOWN

        if change_pct <= self.DECEL_THRESHOLD:
            return BurnTrajectory.DECELERATING
        elif change_pct >= self.ACCEL_THRESHOLD:
            if has_late_stage:
                return BurnTrajectory.ACCELERATING_JUSTIFIED
            return BurnTrajectory.ACCELERATING
        else:
            return BurnTrajectory.STABLE

    def _compute_risk_level(
        self,
        trajectory: BurnTrajectory,
        runway_months: Optional[Decimal],
        burn_change_pct: Optional[Decimal],
    ) -> BurnRiskLevel:
        """Compute overall burn risk level."""
        if trajectory == BurnTrajectory.UNKNOWN:
            return BurnRiskLevel.UNKNOWN

        # Decelerating is generally low risk
        if trajectory == BurnTrajectory.DECELERATING:
            return BurnRiskLevel.LOW

        # Stable depends on runway
        if trajectory == BurnTrajectory.STABLE:
            if runway_months is None:
                return BurnRiskLevel.MODERATE
            if runway_months >= self.RUNWAY_COMFORTABLE:
                return BurnRiskLevel.LOW
            if runway_months >= self.RUNWAY_SHORT:
                return BurnRiskLevel.MODERATE
            return BurnRiskLevel.HIGH

        # Accelerating (justified or not)
        if runway_months is None:
            return BurnRiskLevel.MODERATE

        if runway_months < self.RUNWAY_VERY_SHORT:
            return BurnRiskLevel.CRITICAL
        if runway_months < self.RUNWAY_SHORT:
            return BurnRiskLevel.HIGH

        # Accelerating but comfortable runway
        if trajectory == BurnTrajectory.ACCELERATING_JUSTIFIED:
            return BurnRiskLevel.LOW
        return BurnRiskLevel.MODERATE

    def _compute_score_modifier(
        self,
        trajectory: BurnTrajectory,
        risk_level: BurnRiskLevel,
        runway_months: Optional[Decimal],
        confidence: Decimal,
    ) -> Decimal:
        """
        Compute score modifier based on trajectory and risk.

        Bounded to [-2.0, +2.0], scaled by confidence.
        """
        if trajectory == BurnTrajectory.UNKNOWN:
            return Decimal("0")

        # Base modifier by trajectory
        if trajectory == BurnTrajectory.DECELERATING:
            if runway_months and runway_months >= self.RUNWAY_COMFORTABLE:
                base = self.MODIFIER_DECEL_GOOD_RUNWAY
            else:
                base = self.MODIFIER_DECEL_SHORT_RUNWAY
        elif trajectory == BurnTrajectory.STABLE:
            base = self.MODIFIER_STABLE
        elif trajectory == BurnTrajectory.ACCELERATING_JUSTIFIED:
            base = self.MODIFIER_ACCEL_JUSTIFIED
        else:  # ACCELERATING
            if risk_level == BurnRiskLevel.CRITICAL:
                base = self.MODIFIER_ACCEL_CRITICAL
            elif risk_level == BurnRiskLevel.HIGH:
                base = self.MODIFIER_ACCEL_SHORT_RUNWAY
            else:
                base = self.MODIFIER_ACCEL_UNJUSTIFIED

        # Scale by confidence
        modifier = base * confidence

        # Clamp to bounds
        modifier = max(Decimal("-2.0"), min(Decimal("2.0"), modifier))

        return modifier.quantize(Decimal("0.01"))

    def _has_late_stage_catalyst(
        self,
        clinical_data: Optional[Dict[str, Any]],
    ) -> bool:
        """Check if company has late-stage (Phase 3+) programs."""
        if not clinical_data:
            return False

        lead_phase = clinical_data.get("lead_phase", "")
        if isinstance(lead_phase, str):
            lead_phase_lower = lead_phase.lower()
            if "phase 3" in lead_phase_lower or "phase_3" in lead_phase_lower:
                return True
            if "phase 4" in lead_phase_lower or "phase_4" in lead_phase_lower:
                return True
            if "approved" in lead_phase_lower:
                return True

        # Check trial counts
        phase_3_trials = clinical_data.get("phase_3_trials", 0)
        if phase_3_trials and int(phase_3_trials) > 0:
            return True

        return False

    def _to_decimal(self, value: Any) -> Optional[Decimal]:
        """Convert value to Decimal safely."""
        if value is None:
            return None
        try:
            return Decimal(str(value))
        except (ValueError, TypeError):
            return None

    def _parse_date(self, date_str: Any) -> Optional[date]:
        """Parse date string to date object."""
        if isinstance(date_str, date):
            return date_str
        if not date_str:
            return None
        try:
            return date.fromisoformat(str(date_str))
        except ValueError:
            return None

    def score_universe(
        self,
        universe: List[Dict[str, Any]],
        financial_by_ticker: Dict[str, Dict],
        clinical_by_ticker: Dict[str, Dict],
        as_of_date: date,
    ) -> Dict[str, Any]:
        """
        Score burn trajectory for entire universe.

        Args:
            universe: List of {ticker: str}
            financial_by_ticker: Financial data keyed by ticker
            clinical_by_ticker: Clinical data keyed by ticker
            as_of_date: Point-in-time date

        Returns:
            Dictionary with scores and diagnostics
        """
        scores_by_ticker = {}
        trajectory_distribution = {t.value: 0 for t in BurnTrajectory}
        risk_distribution = {r.value: 0 for r in BurnRiskLevel}

        for record in universe:
            ticker = record.get("ticker", "").upper()
            if not ticker:
                continue

            financial = financial_by_ticker.get(ticker, {})
            clinical = clinical_by_ticker.get(ticker, {})

            result = self.compute_trajectory(
                ticker, financial, clinical, as_of_date
            )

            scores_by_ticker[ticker] = {
                "ticker": ticker,
                "trajectory": result.trajectory.value,
                "risk_level": result.risk_level.value,
                "burn_change_pct": str(result.burn_change_pct) if result.burn_change_pct else None,
                "score_modifier": str(result.score_modifier),
                "confidence": str(result.confidence),
                "flags": result.flags,
                "has_late_stage_catalyst": result.has_late_stage_catalyst,
            }

            trajectory_distribution[result.trajectory.value] += 1
            risk_distribution[result.risk_level.value] += 1

        return {
            "scores_by_ticker": scores_by_ticker,
            "diagnostic_counts": {
                "total_scored": len(scores_by_ticker),
                "trajectory_distribution": trajectory_distribution,
                "risk_distribution": risk_distribution,
            },
            "provenance": {
                "engine": "CashBurnEngine",
                "version": self.VERSION,
                "as_of_date": as_of_date.isoformat(),
            },
        }

    def _add_audit(
        self,
        ticker: str,
        as_of_date: date,
        result: BurnTrajectoryResult,
    ) -> None:
        """Add entry to audit trail."""
        self.audit_trail.append({
            "ticker": ticker,
            "as_of_date": as_of_date.isoformat(),
            "trajectory": result.trajectory.value,
            "risk_level": result.risk_level.value,
            "score_modifier": str(result.score_modifier),
        })

    def get_audit_trail(self) -> List[Dict[str, Any]]:
        """Return audit trail."""
        return self.audit_trail.copy()

    def clear_audit_trail(self) -> None:
        """Clear audit trail."""
        self.audit_trail = []
